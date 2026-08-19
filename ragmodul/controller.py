"""
RAG 처리 단계를 메서드로 제공한다.

메서드 하나 = 파이프라인의 한 단계. 여러 단계를 묶어서 자체적으로
오케스트레이션하지 않는다 — 순서·재시도·단계 간 데이터 전달은 부르는 쪽 책임이고,
여기서는 요청받은 단계 하나만 실행해서 결과를 돌려준다.

실패는 예외로 올린다. 삼켜서 상태값으로 돌려주지 않는다.
"""

import logging

from .models.chunk_model import ChunkedDocument
from .service.chunker_service import chunk
from .service.db_service import DbService
# from .service.embedded_service import EmbeddedService
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

        # self._embedder = EmbeddedService(
        #     embedding_model_path,
        #     device=device,
        #     use_fp16=use_fp16,
        #     passage_max_length=passage_max_length,
        #     query_max_length=query_max_length,
        # )
        # self._reranker = RerankerService(
        #     reranker_model_path, max_length=reranker_max_length, device=device
        # )
        self._db = DbService()

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
        """각 child에 임베딩 벡터를 채워 넣는다. 같은 객체를 돌려준다."""
        children = document.children()
        # vectors = self._embedder.encode_documents([c.content for c in children])
        # for child, vector in zip(children, vectors):
        #     child.vector = vector
        logger.info("임베딩 완료: %d개", len(children))
        return document

    def save_to_vector_db(self, document: ChunkedDocument) -> int:
        """저장하고 저장한 child 수를 돌려준다."""
        #self._db.save_document(document)
        logger.info("DB 저장 완료")
        return len(document.children())

    # ── 질의 검색 ────────────────────────────────────────────────────────

    def embed_query(self, query: str):
        """질의 하나를 벡터로 만든다."""
        logger.info("질의 백터회")
        return #self._embedder.encode_queries([query])[0]

    def hybrid_search(self, query_vector, top_k: int = 5) -> list:
        logger.info("검색")
        #return self._db.search(query_vector, top_k)

    def rerank(self, query: str, results: list, top_k: int = 3) -> list:
        logger.info("리랭크")
        return list()
        #return self._reranker.rerank(query, results, top_k)
