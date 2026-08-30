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

# 최종 자리에 담을 한 부모당 조각 수. 한 부모가 자리를 독점하는 걸 막는다.
#
# 왜 필요한가(실측): child 22개인 부모에서 3개가 걸렸는데, 승격 기준이 hit 비율 0.5
# 라서 섹션으로 묶이지 않고 조각 셋이 최종 5칸 중 3칸을 먹었다. 큰 부모는 12개가
# 걸려야 승격되는데 후보를 40개 뽑는 상황에서 그런 일은 거의 없다.
#
# 그 조각들은 앞부분을 공유한다 — 표를 쪼갤 때 머리글을 조각마다 붙이기 때문이다.
# 그래서 LLM 이 거의 같은 글을 세 번 보고 출처는 두 곳으로 줄어든다.
#
# 2 인 이유: 한 섹션 안에 답이 두 군데 흩어진 경우를 살리면서 독점만 막는다.
DEFAULT_MAX_PER_PARENT = 2

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


def _limit_per_parent(ordered: list, top_k: int, max_per_parent: int) -> list:
    """점수순 목록에서 부모당 개수를 제한해 top_k 를 고른다.

    제한 때문에 자리를 못 채우면 제한을 풀고 남은 것으로 메운다 — 안 그러면 답이 한
    섹션에만 있는 질의에서 맥락이 한두 개로 줄어 지금보다 나빠진다.
    """
    picked: list = []
    per_parent: dict = {}
    for context in ordered:
        parent = getattr(context, "parent_id", None)
        if per_parent.get(parent, 0) >= max_per_parent:
            continue
        picked.append(context)
        per_parent[parent] = per_parent.get(parent, 0) + 1
        if len(picked) == top_k:
            return picked

    taken = {id(c) for c in picked}
    picked.extend(c for c in ordered if id(c) not in taken)
    return picked[:top_k]


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

    def rerank(self, query: str, contexts: list, top_k: int,
               max_per_parent: int | None = DEFAULT_MAX_PER_PARENT,
               min_score: float | None = DEFAULT_MIN_SCORE) -> list:
        """contexts 를 재정렬해 상위 top_k 를 돌려준다.

        contexts 는 rerank_text(읽기)와 rerank_score(쓰기)를 가진 객체여야 한다
        (RetrievedContext). 그 두 이름만 쓰고 타입은 보지 않는다 — 리랭커가 우리
        데이터 모델을 알 필요는 없다.

        기존 점수(similarity/score)는 건드리지 않는다. 예전에 그걸 덮어써서 화면에
        코사인 유사도인 것처럼 표시된 적이 있다.
        점수는 0~1 — 라벨이 1개인 리랭커라 CrossEncoder 가 sigmoid 를 기본으로 씌운다.

        max_per_parent 는 최종 자리에 한 부모의 조각을 몇 개까지 담을지다. None 이면
        제한하지 않는다.

        min_score 아래는 버린다. 전부 걸리면 빈 목록을 돌려준다 — 문서와 무관한
        질의라는 뜻이고, 그때는 맥락 없이 답하는 게 맞다. 억지로 몇 개 남겨두면
        모델이 상관없는 글로 답을 엮는다. None 이면 거르지 않는다(점수만 매긴다).
        점수는 걸러진 것에도 매겨져 있으므로 부르는 쪽에서 확인할 수 있다.
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

        if min_score is not None:
            kept = [c for c in ordered if c.rerank_score >= min_score]
            if len(kept) != len(ordered):
                top = ordered[0].rerank_score if ordered else 0.0
                logger.info("점수 미달 버림: %d개 중 %d개 남음 (기준 %.3f, 최고 %.6f)",
                            len(ordered), len(kept), min_score, top)
            ordered = kept
            if not ordered:
                return []

        if max_per_parent is None:
            return ordered[:top_k]
        return _limit_per_parent(ordered, top_k, max_per_parent)
