#================================================
# controller.py
#================================================
"""
RAG 처리 단계를 메서드로 제공한다.

메서드 하나 = 파이프라인의 한 단계. 여러 단계를 묶어서 자체적으로
오케스트레이션하지 않는다 — 순서·재시도·단계 간 데이터 전달은 부르는 쪽 책임이고,
여기서는 요청받은 단계 하나만 실행해서 결과를 돌려준다.

실패는 예외로 올린다. 삼켜서 상태값으로 돌려주지 않는다.
"""

import logging

from .models.chunk_model import ChunkedDocument
from .models.search_model import DEFAULT_MERGE_RATIO, RetrievedContext
from .service.chunker_service import chunk
from .service.db_service import DbService
from .service.embedded_service import EmbeddedService
from .service.parser_service import parse
from .service.reranker_service import RerankerService

logger = logging.getLogger(__name__)


class RagController:

    def __init__(
        self,
        embedding_model_path: str,
        reranker_model_path: str,
        *,
        device: str | None = None,
        use_fp16: bool = True,
        passage_max_length: int = 8192,
        query_max_length: int = 8192,
        reranker_max_length: int = 512,
        unpack_dir: str = "unpacked",
        db_config: dict | None = None,
    ):
        """설정은 전부 인자로 받는다.

        환경변수/.env를 읽지 않는다. 설정을 어디서 가져올지는 이 모듈을 쓰는
        애플리케이션이 정할 일이고, 라이브러리가 남의 os.environ을 건드리거나
        import 시점 값에 기본값을 묶어두면 쓰는 쪽이 예측할 수 없다.

        device=None 이면 라이브러리가 자동 감지한다(GPU 있으면 GPU).
        """
        self.embedding_model_path = embedding_model_path
        self.reranker_model_path = reranker_model_path
        self.device = device
        self.use_fp16 = use_fp16
        self.passage_max_length = passage_max_length
        self.query_max_length = query_max_length
        self.reranker_max_length = reranker_max_length
        self.unpack_dir = unpack_dir
        self.db_config = db_config

        self._embedder = EmbeddedService(
            embedding_model_path,
            device=device,
            use_fp16=use_fp16,
            passage_max_length=passage_max_length,
            query_max_length=query_max_length,
        )
        self._reranker = RerankerService(
            reranker_model_path,
            max_length=reranker_max_length,
            device=device,
            use_fp16=use_fp16,
        )
        # sparse 차원은 하드코딩하지 않고 모델에게 묻는다(= 토크나이저 vocab 크기).
        # sql/schema.sql 의 SPARSEVEC(N) 과 이 값이 어긋나면 저장에서 실패한다.
        self._db = DbService(
            config=db_config,
            sparse_dim=self._embedder.sparse_dimension,
        )

    # ── 문서 등록 ────────────────────────────────────────────────────────

    def parse_document(self, file_path: str):
        """hwpx 문서를 구조화된 DocumentModel로 만든다."""
        logger.info("문서 파싱: %s", file_path)
        return parse(file_path, unpack_dir=self.unpack_dir)

    def chunk_parent_child(self, parsed) -> ChunkedDocument:
        """DocumentModel을 목차 기준 parent/child 청크로 나눈다."""
        document = chunk(parsed)
        logger.info("청킹 완료: parent %d, child %d",
                    len(document.parents), len(document.children()))
        return document

    def embed_bge_m3(self, document: ChunkedDocument) -> ChunkedDocument:
        """각 child에 dense 벡터와 sparse 가중치를 채워 넣는다. 같은 객체를 돌려준다.

        embedded 라이브러리가 dense/sparse 를 한 번에 주는 메서드를 열어두지 않아
        forward 가 두 번 돈다. 합치려면 그쪽에 encode_all() 을 추가해야 한다.
        """
        children = document.children()
        texts = [c.content for c in children]

        vectors = self._embedder.encode_documents(texts)
        weights = self._embedder.encode_sparse(texts)
        for child, vector, weight in zip(children, vectors, weights):
            child.vector = vector
            child.sparse = weight

        logger.info("임베딩 완료: %d개 (dense+sparse)", len(children))
        return document

    def save_to_vector_db(self, document: ChunkedDocument) -> int:
        """저장하고 저장한 child 수를 돌려준다."""
        self._db.save_document(document)
        logger.info("DB 저장 완료: %d개", len(document.children()))
        return len(document.children())

    # ── 질의 검색 ────────────────────────────────────────────────────────

    def embed_query(self, query: str):
        """질의를 (dense 벡터, sparse 가중치) 로 만든다.

        검색이 둘 다 쓰므로 함께 돌려준다. 질의는 한 문장이라 두 번 호출해도
        비용이 거의 없다(문서 임베딩과 달리).
        """
        logger.info("질의 임베딩: %s", query)
        vector = self._embedder.encode_queries([query])[0]
        weights = self._embedder.encode_sparse([query])[0]
        return vector, weights

    def hybrid_search(self, query_vector, query_weights=None, top_k: int = 5) -> list:
        """dense + sparse 를 RRF 로 합쳐 검색한다.

        query_weights 가 없으면 dense 단독으로 떨어진다.
        """
        logger.info("검색: top_k=%d (%s)", top_k, "hybrid" if query_weights else "dense")
        return self._db.search_hybrid(query_vector, query_weights, top_k)

    def build_contexts(self, hits: list, merge_ratio: float = DEFAULT_MERGE_RATIO,
                       limit: int | None = None) -> list[RetrievedContext]:
        """검색된 조각을 LLM 입력 단위로 묶는다.

        조각이 merge_ratio 이상 걸린 섹션만 본문으로 승격한다. 조금 걸린 섹션은
        조각 그대로 둔다 — 같은 본문이 여러 번 실려가는 걸 막고(실측: 본문의 56%가
        중복), 2/16 만 걸린 섹션에 수천 자를 붙이는 낭비도 없다.

        limit 은 기본으로 걸지 않는다. 약한 신호(유사도)로 미리 자른 뒤 강한 신호
        (리랭커)에게 남은 것만 주는 건 순서가 거꾸로다. 실제로 limit=10 을 걸었다가
        정답이 잘려 recall 이 93%->83% 로 떨어졌다. 리랭크 비용은 후보 수에 비례하지만
        수십 개는 GPU 에서 무료에 가깝다.
        """
        contexts = RetrievedContext.from_rows(hits, merge_ratio=merge_ratio)
        merged = sum(1 for c in contexts if c.merged)
        logger.info("맥락 조립: 조각 %d개 -> %d개 (승격 %d, 낱개 %d)",
                    len(hits), len(contexts), merged, len(contexts) - merged)
        return contexts[:limit] if limit else contexts

    def rerank(self, query: str, contexts: list, top_k: int = 3) -> list:
        logger.info("리랭크: %d개 -> top_k=%d", len(contexts), top_k)
        return self._reranker.rerank(query, contexts, top_k)
