#================================================
# search_model.py
#================================================
"""
검색 결과의 데이터 모델.

청킹 결과(chunk_model)와 다른 물건이라 파일을 나눈다. 청킹 쪽은 'child::1' 같은
문자열 id 와 임베딩 벡터를 들고 다니고, 여기는 DB 의 정수 id 와 유사도 점수를 든다.

핵심 결정: 결과를 한 타입으로 통일한다.
검색은 조각으로 하지만 LLM 에 넣는 건 섹션 본문이다. 조각을 그대로 돌려주면 같은
섹션에서 여러 조각이 걸릴 때 본문이 여러 번 실려간다(실측: 조각 10개가 섹션 2곳,
본문의 56%가 중복).

그래서 조각이 충분히 많이 걸린 섹션만 본문으로 승격하고(merged=True), 조금 걸린
섹션은 조각 그대로 둔다(merged=False). LlamaIndex AutoMergingRetriever 와 같은
방식이다. 두 경우를 별 타입으로 두면 쓰는 쪽이 매번 isinstance 를 해야 하므로
한 타입에 merged 플래그로 구분한다.

승격 기준을 '걸린 수'가 아니라 '비율'로 잡는 이유: 큰 섹션은 조각이 많아 걸릴 기회도
많다. 실측에서 2회 이상 걸린 섹션의 평균 조각 수는 17.8, 1회만 걸린 섹션은 11.1 이었다.
비율로 보면 그 편향이 사라진다 — 조각 14개 중 14개가 걸린 섹션(우연의 9.3배)과
39개 중 6개가 걸린 섹션(1.4배, 우연 수준)이 제대로 갈린다.
"""

from dataclasses import dataclass, field
from typing import Optional

DEFAULT_MERGE_RATIO = 0.5       # LlamaIndex 기본값. 이 문서 기준값은 아직 측정 안 함.


@dataclass
class RetrievedChild:
    """검색에 걸린 조각 하나. '섹션의 어디가 맞았나'를 가리킨다."""
    content: str
    similarity: float
    child_id: Optional[int] = None


@dataclass
class RetrievedContext:
    """검색 결과 한 덩어리. LLM 에 넘길 단위.

    merged=True   content 가 섹션 본문 전체. 조각이 비율 이상 걸려서 승격됐다.
    merged=False  content 가 걸린 조각 그 자체. 섹션 본문은 붙이지 않는다.
    """
    parent_id: int
    content: str                    # LLM 에 실제로 넣을 텍스트
    merged: bool
    score: float                    # 걸린 조각 점수의 평균
    children: list[RetrievedChild] = field(default_factory=list)
    heading: Optional[str] = None
    breadcrumb: str = ""
    document_id: Optional[int] = None
    document_title: Optional[str] = None
    rerank_score: Optional[float] = None

    @property
    def hit_count(self) -> int:
        return len(self.children)

    @property
    def rerank_text(self) -> str:
        """리랭커에 줄 텍스트. 최고점 조각을 쓴다.

        content 를 쓰면 안 된다 — 승격된 섹션은 수천 자라 max_length 에서 잘려
        앞부분만 보고 판정하고, 답이 뒤에 있으면 무관으로 떨어진다(Embedde 에서 겪음).
        """
        return self.children[0].content

    @classmethod
    def from_rows(
        cls,
        rows: list[dict],
        merge_ratio: float = DEFAULT_MERGE_RATIO,
    ) -> list["RetrievedContext"]:
        """DbService.search_hybrid() 결과를 LLM 입력 단위로 묶는다.

        Args:
            rows: parent_id / child_content / similarity / content /
                  parent_child_count 를 가진 dict 목록.
                  parent_child_count 가 비율의 분모다 — 검색 결과에는 '걸린' 조각만
                  있어서 쿼리가 전체 수를 함께 실어 준다. 없으면 걸린 수로 대체하는데,
                  그러면 비율이 항상 1.0 이라 전부 승격된다.
            merge_ratio: 이 비율을 넘으면 섹션 본문으로 승격한다.

        Returns:
            점수 내림차순. 최종 순서는 리랭커가 정하므로 여기 순서는 후보 선별용이다.
        """
        grouped: dict[int, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row["parent_id"], []).append(row)

        out: list[RetrievedContext] = []
        for parent_id, hits in grouped.items():
            # 점수순으로 세운다. 대표 조각(rerank_text)이 최고점이어야 한다.
            hits.sort(key=lambda r: r["similarity"], reverse=True)
            children = [
                RetrievedChild(
                    content=r.get("child_content") or "",
                    similarity=r["similarity"],
                    child_id=r.get("child_id"),
                )
                for r in hits
            ]
            head = hits[0]
            total = head.get("parent_child_count") or len(children)
            avg = sum(c.similarity for c in children) / len(children)

            if len(children) / total > merge_ratio:
                # 승격: 섹션 본문 하나로 합친다. 조각 목록은 '어디가 맞았나'로 남긴다.
                out.append(cls(
                    parent_id=parent_id,
                    content=head.get("content") or "",
                    merged=True,
                    score=avg,
                    children=children,
                    heading=head.get("heading"),
                    breadcrumb=head.get("breadcrumb") or "",
                    document_id=head.get("document_id"),
                    document_title=head.get("document_title"),
                ))
            else:
                # 미달: 조각을 낱개로 남긴다. 섹션 본문은 붙이지 않는다 —
                # 39개 중 2개만 걸린 섹션에 수천 자를 붙이는 건 과하다.
                for child in children:
                    out.append(cls(
                        parent_id=parent_id,
                        content=child.content,
                        merged=False,
                        score=child.similarity,
                        children=[child],
                        heading=head.get("heading"),
                        breadcrumb=head.get("breadcrumb") or "",
                        document_id=head.get("document_id"),
                        document_title=head.get("document_title"),
                    ))

        out.sort(key=lambda c: c.score, reverse=True)
        return out
