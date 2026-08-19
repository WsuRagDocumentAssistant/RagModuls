"""
임베딩 단계 — zlfm78/TestEmbeddingModelRepository 패키지의 BGEM3Model을 얇게 감싼다.
사전 설치 필요: pip install git+https://github.com/zlfm78/TestEmbeddingModelRepository.git

device를 넘기지 않으면 가중치가 CPU에 남고 연산할 때만 GPU로 복사된다(실측 확인).
매 배치마다 복사가 일어나므로 GPU가 있으면 'cuda'를 명시해 올려두는 편이 낫다.
"""

import logging

# from embedded import BGEM3Model

logger = logging.getLogger(__name__)


class EmbeddedService:

    def __init__(
        self,
        model_path: str,
        *,
        device: str | None = None,
        use_fp16: bool = True,
        passage_max_length: int = 8192,
        query_max_length: int = 8192,
    ):
        self.model_path = model_path
        self.device = device
        self.use_fp16 = use_fp16
        self.passage_max_length = passage_max_length
        self.query_max_length = query_max_length

        logger.info("임베딩 초기화 진행 (device=%s)", device or "자동")
        # self._model = BGEM3Model(
        #     model_path,
        #     device=device,
        #     use_fp16=use_fp16,
        #     passage_max_length=passage_max_length,
        #     query_max_length=query_max_length,
        # )
        logger.info("임베딩 초기화 완료")

    # def encode_documents(self, texts: list[str]):
    #     return self._model.encode_documents(texts)

    # def encode_queries(self, texts: list[str]):
    #     return self._model.encode_queries(texts)

    # def encode_sparse(self, texts: list[str]):
    #     return self._model.encode_sparse(texts)

    # def unload(self) -> None:
    #     self._model.unload()
