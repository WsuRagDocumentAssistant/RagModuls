#================================================
# embedded_service.py
#================================================
"""
임베딩 단계 — zlfm78/TestEmbeddingModelRepository 패키지의 BGEM3Model을 얇게 감싼다.
사전 설치 필요: pip install git+https://github.com/zlfm78/TestEmbeddingModelRepository.git

device를 넘기지 않으면 가중치가 CPU에 남고 연산할 때만 GPU로 복사된다(실측 확인).
매 배치마다 복사가 일어나므로 GPU가 있으면 'cuda'를 명시해 올려두는 편이 낫다.
"""

import logging

from embedded import BGEM3Model

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
        self._logged_device = False

        logger.info("임베딩 초기화 진행 (device=%s)", device or "자동")
        self._model = BGEM3Model(
            model_path,
            device=device,
            use_fp16=use_fp16,
            passage_max_length=passage_max_length,
            query_max_length=query_max_length,
        )
        logger.info("임베딩 초기화 완료")

    @property
    def dimension(self) -> int:
        """dense 벡터 차원. DB 컬럼 vector(N) 의 N 과 같아야 한다."""
        return self._model.dimension

    @property
    def sparse_dimension(self) -> int:
        """sparse 벡터 차원(= 토크나이저 vocab 크기). sparsevec(N) 의 N 과 같아야 한다."""
        return self._model.sparse_dimension

    def encode_documents(self, texts: list[str]):
        vectors = self._model.encode_documents(texts)
        self._log_device(len(texts))
        return vectors

    def encode_queries(self, texts: list[str]):
        vectors = self._model.encode_queries(texts)
        self._log_device(len(texts))
        return vectors

    def encode_sparse(self, texts: list[str]):
        return self._model.encode_sparse(texts)

    def unload(self) -> None:
        self._model.unload()

    #------------------------------------------------┌> 진단

    def _log_device(self, count: int) -> None:
        """실제로 어느 장치에서 돌았는지 남긴다.

        생성 시점의 device 인자만 찍으면 안 된다. FlagEmbedding 은 encode() 안에서
        model.to(device) 를 호출하므로, 로드 직후에는 device='cuda' 를 줬어도
        가중치가 아직 CPU 에 있다(실측 확인). 그래서 임베딩이 끝난 뒤에 본다.
        """
        if self._logged_device:
            return
        self._logged_device = True
        try:
            import torch

            param = next(self._model._model.model.parameters())
            used = str(param.device)
            note = ""
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024 ** 3
                note = f", VRAM {allocated:.2f}GB"
            elif used.startswith("cpu"):
                note = ", CUDA 사용 불가(torch가 CPU 빌드이거나 GPU 없음)"
            logger.info("임베딩 실행 장치: %s (요청=%s, dtype=%s%s)",
                        used, self.device or "자동", param.dtype, note)
        except Exception as e:                      # 진단용이므로 실패해도 넘어간다
            logger.debug("장치 확인 실패: %s", e)
