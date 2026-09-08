#================================================
# image_model.py
#================================================
"""
이미지 설명의 데이터 모델.

세 값을 이어붙여 임베딩한다. 그래서 이건 사람이 읽을 글이 아니라 '검색에 걸리는 말'
이다 — 사용자가 "2026년 취업률 그래프 보여줘" 라고 물었을 때 겹칠 낱말이 들어 있어야
한다. 뭉뚱그리면("취업률이 상승세") 연도로 묻는 질문에 안 걸린다.
"""

from pydantic import BaseModel, Field


class ImageQuery(BaseModel):
    """is_image_query 의 출력. 사용자가 그림을 찾고 있는지."""
    wants_image: bool = Field(
        description="그림·사진·도표·그래프·차트·이미지·표 를 직접 찾는 말이면 true. "
                    "내용을 묻는 말은 그림이 도움이 될 것 같아도 false")


class ImageDescription(BaseModel):
    """describe_image 의 출력. 설명할 것이 없는 그림이면 세 값이 다 비어도 된다."""
    ai_summary: str = Field(
        default="",
        description="그림이 무엇을 보여주는지 한 문장. 그림 종류(막대그래프/조직도/"
                    "절차도/표)를 포함한다. 화면에도 보이는 값이다")
    key_facts: list[str] = Field(
        default_factory=list,
        description='그림 안의 수치와 항목을 그대로. 예: "2026년 전체 취업률 68.7%", '
                    '"간호학과 92%". 검색의 주력이다')
    key_phrases: list[str] = Field(
        default_factory=list,
        description='그림에 인쇄된 말 — 제목, 범례, 축 이름, 단계 이름. 낱말 단위로. '
                    '예: "학생역량 진단", "역량 강화", "환류"')
