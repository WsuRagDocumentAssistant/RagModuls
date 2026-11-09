#================================================
# search_model.py
#================================================
"""
검색 결과의 데이터 모델.

청킹 결과(chunk_model)와 다른 물건이라 파일을 나눈다. 청킹 쪽은 'child::1' 같은
문자열 id 와 임베딩 벡터를 들고 다니고, 여기는 DB 의 정수 id 와 유사도 점수를 든다.

여기는 그릇만 둔다. 어느 부모를 고를지는 RagController.rerank() 가 정한다 —
조각으로 찾고, 조각으로 리랭크하고, 상위 조각의 부모 본문을 넘긴다(small-to-big).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrievedChild:
    """검색에 걸린 조각 하나. '부모 본문의 어디가 맞았나'를 가리킨다."""
    content: str
    similarity: float
    child_id: Optional[int] = None
    rerank_score: Optional[float] = None


@dataclass
class RetrievedContext:
    """LLM 에 넘길 단위. 부모 하나.

    content 는 부모 본문 전체다. children 은 그 부모에서 리랭크 상위에 든 조각들이고
    리랭크 점수순이다 — RagSystem 출처 화면이 children[0] 을 '맞은 부분'으로,
    children[:N] 을 인용문으로 보여준다.
    """
    parent_id: int
    content: str                    # LLM 에 실제로 넣을 텍스트 (부모 본문)
    score: float                    # 걸린 조각 중 최고 유사도
    children: list[RetrievedChild] = field(default_factory=list)
    heading: Optional[str] = None
    breadcrumb: str = ""
    document_id: Optional[int] = None
    document_title: Optional[str] = None
    rerank_score: Optional[float] = None   # 최고 조각의 리랭크 점수

    @property
    def hit_count(self) -> int:
        return len(self.children)

    @property
    def rerank_text(self) -> str:
        """리랭크 점수가 가장 높았던 조각."""
        return self.children[0].content

    @classmethod
    def from_parent(cls, row: dict,
                    children: list[RetrievedChild]) -> "RetrievedContext":
        """검색 행 하나(부모 정보)와 그 부모의 조각들로 맥락을 만든다.

        row 는 search_hybrid() 결과 행이다. 같은 부모의 행은 부모 칼럼이 같으므로
        아무 행이나 줘도 된다. children 은 리랭크 점수순이어야 한다.
        """
        return cls(
            parent_id=row["parent_id"],
            content=row.get("content") or "",
            score=max(c.similarity for c in children),
            children=children,
            heading=row.get("heading"),
            breadcrumb=row.get("breadcrumb") or "",
            document_id=row.get("document_id"),
            document_title=row.get("document_title"),
            rerank_score=children[0].rerank_score,
        )
