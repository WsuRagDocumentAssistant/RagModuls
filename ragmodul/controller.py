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
from .models.vocab_model import VocabPair
from .service.chunker_service import chunk
from .service.db_service import DbService
from .service.embedded_service import EmbeddedService
from .service.parser_service import parse
from .service.reranker_service import RerankerService
from .util import expand_query, filter_vocab_pairs

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
        image_dir: str | None = None,
        db_config: dict | None = None,
        llm_api_config=None,
        local_llm_config=None,
        llm_default: str | None = None,
    ):
        """설정은 전부 인자로 받는다.

        환경변수/.env를 읽지 않는다. 설정을 어디서 가져올지는 이 모듈을 쓰는
        애플리케이션이 정할 일이고, 라이브러리가 남의 os.environ을 건드리거나
        import 시점 값에 기본값을 묶어두면 쓰는 쪽이 예측할 수 없다.

        device=None 이면 라이브러리가 자동 감지한다(GPU 있으면 GPU).

        llm_api_config / local_llm_config 는 ai-rag-comm 의 load_config() 결과를
        넘긴다(cfg.llm_api, cfg.local_llm). 둘 다 없으면 self.llm 은 None 이고
        검색까지만 된다.
        """
        self.embedding_model_path = embedding_model_path
        self.reranker_model_path = reranker_model_path
        self.device = device
        self.use_fp16 = use_fp16
        self.passage_max_length = passage_max_length
        self.query_max_length = query_max_length
        self.reranker_max_length = reranker_max_length
        self.unpack_dir = unpack_dir
        self.image_dir = image_dir
        self.db_config = db_config
        self.llm_api_config = llm_api_config
        self.local_llm_config = local_llm_config
        self.llm_default = llm_default

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
        # LLM 설정을 안 준 사람은 LLM 스택을 안 깐 사람이다(ragmodul[llm] 은 선택
        # 의존성). import 를 여기 두는 건 그래서다 — 모듈 맨 위에 두면 검색만 쓰는
        # 사람이 ragmodul 을 import 조차 못 한다. 객체 자체는 만들어 둔다. 설정을
        # _Provider 넷으로 정리할 뿐이고 채널은 처음 쓸 때 각자 만든다.
        self.llm = None
        if llm_api_config is not None or local_llm_config is not None:
            from .service.llm_service import LlmService

            self.llm = LlmService(llm_api_config, local_llm_config, default=llm_default)

    #------------------------------------------------┌> 문서 등록

    def parse_document(self, file_path: str):
        """hwpx 문서를 구조화된 DocumentModel로 만든다.

        image_dir 가 있으면 문서 이미지도 그 폴더로 빼낸다.
        """
        logger.info("문서 파싱: %s", file_path)
        return parse(file_path, unpack_dir=self.unpack_dir, image_dir=self.image_dir)

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

    #------------------------------------------------┌> 질의 검색

    def embed_query(self, query: str, vocab: dict[str, list[str]] | None = None):
        """질의를 (dense 벡터, sparse 가중치) 로 만든다.

        검색이 둘 다 쓰므로 함께 돌려준다. 질의는 한 문장이라 두 번 호출해도
        비용이 거의 없다(문서 임베딩과 달리).

        vocab 을 주면(load_vocab() 결과) 축약어의 짝을 질의에 덧붙여서 임베딩한다.
        dense·sparse 양쪽에 같은 확장 질의를 쓴다 — 실측에서 둘 다 좋아졌고 나빠진
        사례가 없었다. 자세한 근거는 util.expand_query 에 적어뒀다.

        vocab=None 이면 확장하지 않는다. 기존 호출부는 그대로 동작한다.
        """
        text = expand_query(query, vocab)
        if text != query:
            logger.info("질의 임베딩(확장): %s", text)
        else:
            logger.info("질의 임베딩: %s", query)
        vector = self._embedder.encode_queries([text])[0]
        weights = self._embedder.encode_sparse([text])[0]
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

    #------------------------------------------------┌> 답변 생성 (선택 의존성)

    def answer(self, query: str, contexts: list, provider: str | None = None) -> str:
        """검색된 맥락으로 답변을 만든다. rerank() 다음 단계다."""
        return self._require_llm().answer(query, contexts, provider=provider)

    async def aanswer(self, query: str, contexts: list, provider: str | None = None) -> str:
        """answer() 의 async 판. 이미 이벤트 루프 안이면 이쪽을 await 한다."""
        return await self._require_llm().aanswer(query, contexts, provider=provider)

    def refine(self, query: str, contexts: list, draft: str,
               provider: str | None = None) -> str:
        """다른 모델이 만든 답변 초안을 다듬는다. LLM 한 번.

        local_llm 이 초안을 만들고 사용자가 고른 모델이 다듬는 단계다. 고른 모델이
        여럿이면 각자 같은 초안을 받아 따로 다듬는다 — 서로 이어 붙이지 않는다.
        순차로 넘기면 앞 모델의 판단이 뒤로 갈수록 굳어진다.

        Context 를 함께 넘긴다. 사실 검증과 수치 서식 판단에 필요하다.
        """
        return self._require_llm().refine(query, contexts, draft, provider=provider)

    async def arefine(self, query: str, contexts: list, draft: str,
                      provider: str | None = None) -> str:
        """refine() 의 async 판."""
        return await self._require_llm().arefine(query, contexts, draft,
                                                 provider=provider)

    def refine_all(self, query: str, contexts: list, draft: str,
                   providers: list[str], skip: str | None = None,
                   parallel: bool = True) -> dict[str, str]:
        """고른 모델들이 같은 초안을 각자 다듬는다. {provider: 다듬은 답변}.

        사용자가 한 질의에 모델을 여러 개 골랐을 때 쓰는 단계다. 목록 길이만큼
        호출한다 — 하나면 1번, 셋이면 3번.

        기본은 동시 호출이라 걸리는 시간이 합이 아니라 가장 느린 하나다. 분당 토큰
        한도에 걸리면 parallel=False 로 순차로 돌린다.

        하나가 죽어도 나머지는 돌려준다. 실패한 provider 는 결과에 없으니 providers 와
        대조하면 무엇이 빠졌는지 알 수 있다.

        skip 에 초안을 만든 provider 를 주면 건너뛴다.
        """
        return self._require_llm().refine_all(query, contexts, draft, providers,
                                              skip, parallel)

    async def arefine_all(self, query: str, contexts: list, draft: str,
                          providers: list[str], skip: str | None = None,
                          parallel: bool = True) -> dict[str, str]:
        """refine_all() 의 async 판."""
        return await self._require_llm().arefine_all(query, contexts, draft,
                                                     providers, skip, parallel)

    def merge(self, question: str, answers: list[str],
              provider: str | None = None) -> str:
        """여러 모델의 답변을 하나로 합친다. LLM 한 번.

        사용자가 provider 를 둘 이상 골랐을 때 쓴다. 개수 제한은 없다 — 셋이면 셋을
        한 번에 넘긴다. 병합에 쓸 모델은 답변을 낸 것과 달라도 된다.

        답변이 하나뿐이면 그대로 돌려준다(호출하지 않는다).

        병합 모델은 Context 를 못 본다(질문과 답변들이 전부다). 그래서 수치가
        원데이터인지 계산값인지 다시 판단하지 않고, 입력이 매겨둔 구분을 그대로
        옮기도록 프롬프트가 지시한다.
        """
        return self._require_llm().merge(question, answers, provider=provider)

    async def amerge(self, question: str, answers: list[str],
                     provider: str | None = None) -> str:
        """merge() 의 async 판."""
        return await self._require_llm().amerge(question, answers, provider=provider)

    #------------------------------------------------┌> 축약어 사전

    def extract_vocab(self, text: str, provider: str | None = None) -> list[VocabPair]:
        """글에서 축약어/확장어 짝을 뽑는다. LLM 한 번.

        문서를 통째로 넘긴다. 부모 단위로 쪼개 돌리면 호출이 32 번인데 결과는 17 개고,
        통째로 한 번(12 개) 뒤에 recheck_vocab 을 한 번 더 돌리면 18 개다 — 호출 2 번이
        더 많이 찾는다. 쪼개면 문서 앞뒤에 흩어진 '축약어 ... 풀어쓴 말' 을 못 잇는다.

        로컬 모델은 컨텍스트가 8192 토큰이라 문서 전체가 안 들어간다. 클라우드
        provider 를 쓰거나, 로컬로 하려면 글을 잘라서 여러 번 불러야 한다.
        """
        return self._require_llm().extract_vocab(text, provider=provider)

    def recheck_vocab(self, text: str, found: list[VocabPair],
                      provider: str | None = None) -> list[VocabPair]:
        """이미 뽑은 목록을 주고 빠뜨린 것만 더 찾는다. LLM 한 번.

        한 번에 훑으면 뒷부분을 놓친다(실측: 11 개 -> 재검토로 7 개 추가).
        돌려주는 건 새로 찾은 것만이므로, found 와 합쳐서 쓴다.
        """
        return self._require_llm().recheck_vocab(text, found, provider=provider)

    def filter_vocab(self, pairs: list[VocabPair], skip: set[str] | None = None):
        """못 쓸 짝을 걸러낸다. (통과, [(버린 짝, 이유)]) — 모델도 DB도 안 쓴다.

        완벽한 선별이 아니라 사람이 검수할 양을 줄이는 것이 목적이다. 통과한 것에도
        고칠 것이 남으므로 save_vocab 전에 눈으로 본다.
        """
        return filter_vocab_pairs(pairs, skip)

    def save_vocab(self, pairs: list[VocabPair]) -> int:
        """검수한 짝을 DB 에 넣고 새로 들어간 확장어 수를 돌려준다.

        같은 것을 또 넣어도 늘지 않는다 — 재실행해도 사전이 부풀지 않는다.
        """
        return self._db.save_vocab(pairs)

    def load_vocab(self) -> dict[str, list[str]]:
        """{축약어: [확장어, ...]}. 질의 확장이 이 형태로 쓴다."""
        return self._db.load_vocab()

    #------------------------------------------------┌> 내부

    def _require_llm(self):
        if self.llm is None:
            raise ValueError(
                "LLM 설정이 없습니다. RagController(..., llm_api_config=cfg.llm_api, "
                "local_llm_config=cfg.local_llm) 로 넘기세요 "
                "(cfg = ai_rag_comm.load_config())."
            )
        return self.llm

    def close(self) -> None:
        """만들어둔 연결을 정리한다. 프로세스 종료 시 한 번."""
        if self.llm is not None:
            self.llm.close()
