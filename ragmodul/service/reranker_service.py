"""
리랭킹 단계 — BAAI/bge-reranker-v2-m3 Cross-Encoder로 검색 결과를 재정렬한다.

주의(Embedde에서 겪은 문제): CrossEncoder는 max_length=512로 입력을 자른다.
parent 전체 본문처럼 긴 텍스트를 넣으면 앞부분만 보고 판정해서, 실제 매칭된
내용이 뒤에 있으면 죄다 "무관"으로 판정되는 문제가 있었다. 그래서 결과에
child_content(실제 매칭된 짧은 조각)가 있으면 그걸 우선 쓰고, 없으면 content로
폴백한다. DbService의 실제 반환 모양이 정해지면 이 fallback을 다시 확인해야 한다.
"""

import logging

logger = logging.getLogger(__name__)


class RerankerService:

    def __init__(self, model_path: str, *, max_length: int = 512, device: str | None = None) -> None:
        import os
        from sentence_transformers import CrossEncoder

        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"리랭커 모델 폴더를 찾을 수 없습니다: {model_path!r}\n"
                "이미 받아둔 로컬 폴더 경로를 지정하세요."
            )

        self.model_path = model_path
        self.max_length = max_length
        self.device = device

        logger.info("리랭커 로드 시작: %s (device=%s)", model_path, device or "자동")
        self._model = CrossEncoder(model_path, max_length=max_length, device=device)
        logger.info("리랭커 로드 완료")

    def rerank(self, query: str, results: list, top_k: int) -> list:
        if not results:
            return results

        pairs = [[query, r.get("child_content") or r.get("content", "")] for r in results]
        scores = self._model.predict(pairs)

        scored = []
        for r, s in zip(results, scores):
            item = dict(r)
            item["rerank_score"] = float(s)
            scored.append(item)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]
