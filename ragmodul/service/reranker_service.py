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

# 최종 자리에 담을 최소 점수. 이보다 낮으면 버린다.
#
# 왜 필요한가: 검색은 늘 top_k 개를 돌려준다. 문서와 아무 상관 없는 질의("오늘 날씨")
# 가 와도 조각 40개가 잡히고, 그게 그대로 LLM 에 실려 나간다(실측 1,944자). 근거가
# 되지도 못하면서 토큰만 쓰고, 모델이 억지로 엮으면 엉뚱한 답이 된다.
#
# 0.01 인 근거(실측, 같은 문서 대상):
#   관련 있는 질의 셋의 최고점   0.341  0.685  0.986
#   관련 없는 질의 셋의 최고점   0.000046  0.000445  0.000193
# 무관한 쪽 최고값이 0.000445 다. 0.01 은 그 22배 위라 여유가 있다. 0.001 로 잡으면
# 2.2배뿐이라 문서가 바뀌면 무관한 맥락이 새어 들어온다. 반대로 0.1 은 관련 있는
# 질의에서 13개가 4개로 깎여 과하다.
#
# 점수는 sigmoid 라 0~1 이다. 모델을 바꾸면 분포가 달라지므로 다시 재야 한다.
DEFAULT_MIN_SCORE = 0.01


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

        # device=None 을 sentence-transformers 에 그대로 넘기면 GPU 가 있어도 CPU 로
        # 떨어진다("No device provided, using cpu"). 자동 감지를 해주지 않는다.
        # 임베더 쪽은 알아서 GPU 를 잡으므로, 안 맞춰주면 한쪽만 CPU 로 도는 상태가
        # 조용히 만들어진다(실측: RagSystem 워커에서 리랭커만 device=cpu).
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = CrossEncoder(
            model_path,
            max_length=max_length,
            device=device,
            local_files_only=True,      # 로컬 폴더만 본다 (401 방지)
        )

        # fp16 은 GPU 에서만. CPU 에서 half 는 커널이 없어 느려지거나 죽는다.
        # 모델을 올린 뒤에 실제 장치를 보고 정한다.
        self.device = str(self._model.model.device)
        self.use_fp16 = use_fp16 and self.device.startswith("cuda")
        if self.use_fp16:
            self._model.model.half()

        logger.info("리랭커 로드 완료: %s (device=%s, fp16=%s)", model_path, self.device, self.use_fp16)

    def rerank_texts(self, query: str, texts: list[str], top_k: int | None = None,
                     min_score: float | None = DEFAULT_MIN_SCORE,
                     ) -> list[tuple[int, float]]:
        """텍스트 목록에 점수를 매겨 (원래 인덱스, 점수) 를 점수순으로 돌려준다.

        문자열만 받는다 — 리랭커가 우리 데이터 모델을 알 필요가 없다. 검색 조각을
        부모로 묶는 일은 RagController.rerank() 가 이 결과로 한다. DB 에서 dict 로
        오는 것(이미지 설명 등)도 껍데기 클래스 없이 리랭킹할 수 있다.

        점수는 0~1 — 라벨이 1개인 리랭커라 CrossEncoder 가 sigmoid 를 기본으로 씌운다.

        인덱스를 주는 이유: 부르는 쪽은 그 텍스트가 어느 행의 것인지 알아야 하는데,
        텍스트만 돌려주면 되짚을 수 없다(같은 설명이 두 행에 있을 수 있다).

            for index, score in rerank_texts(query, [r["ai_summary"] for r in rows]):
                picked.append(rows[index])

        top_k=None 이면 전부 돌려준다. min_score=None 이면 문턱을 안 건다.

        min_score 아래는 버린다. 전부 걸리면 빈 목록이다 — 문서와 무관한 질의라는
        뜻이고, 그때는 맥락 없이 답하는 게 맞다. 억지로 몇 개 남겨두면 모델이
        상관없는 글로 답을 엮는다.

        min_score 기본값 0.01 은 문서 맥락으로 잰 값이다(무관한 질의 최고점
        0.000445 의 22배). 다른 종류의 글은 분포가 다를 수 있으니 그때는 직접 재서
        넘긴다 — 짧은 글일수록 점수가 낮게 깔린다(제목+출처 60자에서 0.5088).
        """
        if not texts:
            return []

        pairs = [[query, text] for text in texts]
        scores = self._model.predict(pairs, batch_size=self.batch_size,
                                     show_progress_bar=False)

        ranked = sorted(enumerate(float(s) for s in scores),
                        key=lambda pair: pair[1], reverse=True)

        if min_score is not None:
            kept = [pair for pair in ranked if pair[1] >= min_score]
            if len(kept) != len(ranked):
                logger.info("점수 미달 버림: %d개 중 %d개 남음 (기준 %.3f, 최고 %.6f)",
                            len(ranked), len(kept), min_score, ranked[0][1])
            ranked = kept

        return ranked[:top_k] if top_k is not None else ranked
