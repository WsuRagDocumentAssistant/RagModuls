#================================================
# reranker_service.py
#================================================
"""
리랭킹 단계 — bge-reranker-v2-m3 로 검색 결과를 재정렬한다.

sentence-transformers 의 CrossEncoder 를 쓴다.
  - 예전에 401 로그가 뜬 건 "models/bge-reranker-v2-m3" 같은 상대경로를 Hub 저장소
    ID 로도 해석해 huggingface.co 에 조회를 걸었기 때문이다. local_files_only 로 막는다.
  - FlagEmbedding 의 FlagReranker 는 쓰지 않는다. 내부에서 tokenizer.prepare_for_model()
    을 부르는데 transformers 5 에서 그 API 가 사라져 AttributeError 로 죽는다.

주의(Embedde에서 겪은 문제): 입력이 max_length 에서 잘린다. parent 전체 본문을
넣었더니 앞부분만 보고 판정해서, 매칭된 내용이 뒤에 있으면 죄다 "무관"이 됐다.
그래서 실제 매칭된 짧은 조각(child_content)이 있으면 그걸 쓴다.
"""

import logging
import os

logger = logging.getLogger(__name__)


class RerankerService:

    def __init__(
        self,
        model_path: str,
        *,
        max_length: int = 512,
        device: str | None = None,
        use_fp16: bool = True,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import CrossEncoder

        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"리랭커 모델 폴더를 찾을 수 없습니다: {model_path!r}")

        self.batch_size = batch_size

        self._model = CrossEncoder(
            model_path,
            max_length=max_length,
            device=device,              # None 이면 자동 감지
            local_files_only=True,      # 로컬 폴더만 본다 (401 방지)
        )

        # fp16 은 GPU 에서만. CPU 에서 half 는 커널이 없어 느려지거나 죽는다.
        # 모델을 올린 뒤에 실제 장치를 보고 정한다.
        self.device = str(self._model.model.device)
        self.use_fp16 = use_fp16 and self.device.startswith("cuda")
        if self.use_fp16:
            self._model.model.half()

        logger.info("리랭커 로드 완료: %s (device=%s, fp16=%s)", model_path, self.device, self.use_fp16)

    def rerank(self, query: str, contexts: list, top_k: int) -> list:
        """contexts 를 재정렬해 상위 top_k 를 돌려준다.

        contexts 는 rerank_text(읽기)와 rerank_score(쓰기)를 가진 객체여야 한다
        (RetrievedContext). 그 두 이름만 쓰고 타입은 보지 않는다 — 리랭커가 우리
        데이터 모델을 알 필요는 없다.

        기존 점수(similarity/score)는 건드리지 않는다. 예전에 그걸 덮어써서 화면에
        코사인 유사도인 것처럼 표시된 적이 있다.
        점수는 0~1 — 라벨이 1개인 리랭커라 CrossEncoder 가 sigmoid 를 기본으로 씌운다.
        """
        if not contexts:
            return contexts

        # rerank_text 는 '짧은' 텍스트다(최고점 조각). 승격된 섹션 본문을 넣으면
        # max_length 에서 잘려 앞부분만 보고 판정한다.
        pairs = [[query, c.rerank_text] for c in contexts]
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        for context, score in zip(contexts, scores):
            context.rerank_score = float(score)

        ordered = sorted(contexts, key=lambda c: c.rerank_score, reverse=True)
        return ordered[:top_k]
