"""
청킹 결과의 데이터 모델.

parent = 맥락용(LLM에 통째로 전달), child = 검색용(임베딩 대상).
child를 parent 안에 중첩해 담으므로 parent_id 같은 연결 키가 필요 없다.
(DB에 넣을 때는 관계형이라 parent_id가 필요하지만, 그건 저장 시점에 만든다.)

벡터는 ChildChunk 안에 둔다. 밖에 리스트로 따로 두면 children[i]와
vectors[i]가 같은 순서라는 암묵적 약속에 기대게 되는데, 중간에 걸러내거나
정렬하면 조용히 어긋나도 에러가 안 난다. 안에 두면 어긋날 수가 없다.

file을 함께 들고 다니는 이유: 큐로 한 작업씩 오가는 구조라 각 단계가 다음
단계에 필요한 걸 전부 넘겨야 하는데, 파일 정보를 빠뜨리면 저장 단계에서
출처를 알 수 없다.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ChildChunk:
    """검색 단위. content가 임베딩 대상이고 vector에 그 결과를 담는다."""
    id: str
    content: str
    vector: Optional[Any] = None      # numpy.ndarray, 임베딩 전에는 None

    @property
    def is_embedded(self) -> bool:
        return self.vector is not None


@dataclass
class ParentChunk:
    """맥락 단위. 검색된 child의 부모로서 LLM에 통째로 전달된다."""
    id: str
    content: str
    heading: Optional[str] = None
    breadcrumb: str = ""
    children: list[ChildChunk] = field(default_factory=list)


@dataclass
class ChunkedDocument:
    file: Any                          # hwpx FileInfo (파서가 준 것 그대로)
    parents: list[ParentChunk] = field(default_factory=list)

    def children(self) -> list[ChildChunk]:
        """임베딩·저장처럼 평탄한 목록이 필요할 때 쓴다."""
        return [child for parent in self.parents for child in parent.children]
