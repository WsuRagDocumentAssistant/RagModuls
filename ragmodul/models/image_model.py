#================================================
# image_model.py
#================================================
"""
이미지 관련 데이터 모델. 성격이 다른 둘이 들어 있다.

  DocumentImage   파서가 문서에서 뽑아낸 그림 하나. 순수 dataclass.
  ImageQuery      LLM 출력 검증 스키마 (pydantic)
  ImageDescription  〃

아래 둘이 pydantic 인 이유는 LLM 출력의 response_format 으로 쓰이기 때문이다.
DocumentImage 는 우리가 만드는 값이라 검증할 것이 없어 dataclass 다.

설명 세 값(ai_summary/key_facts/key_phrases)은 이어붙여 임베딩한다. 그래서 그건
사람이 읽을 글이 아니라 '검색에 걸리는 말' 이다 — 사용자가 "2026년 취업률 그래프
보여줘" 라고 물었을 때 겹칠 낱말이 들어 있어야 한다. 뭉뚱그리면("취업률이 상승세")
연도로 묻는 질문에 안 걸린다.
"""

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


@dataclass
class DocumentImage:
    """문서에서 뽑아낸 그림 하나. parse(image_dir=...) 가 만든다.

    폴더를 훑는 대신 이걸 쓰면 이번 파싱에서 나온 그림만 등록된다 — 지난번 잔재나
    확장자만 다른 옛 파일이 섞이지 않는다.
    """
    ref: str                                # 문서 안에서의 이름. 예: 'image10'
    path: str                               # 실제로 저장된 경로
    # 문서 전체에서 몇 번째 블록인가. 목록 정렬용.
    # 본문 그림 블록(block.figure.image)만 담으므로 늘 진짜 위치다. 표 셀 안·묶음
    # 개체 안의 그림은 목록에 넣지 않는다(parser_service._extract_images 참고).
    order: int
    section: int = 0
    media_type: str | None = None
    # 그림이 있던 자리의 제목 경로. ['Ⅱ. 교육혁신', '2-1 전공 개편', ...]
    # major_title / mid_title / minor_title 을 여기서 앞 세 개로 채우면 된다.
    heading_path: list[str] = field(default_factory=list)
    # 문서에 캡션이 달려 있으면 그 글. 대개 None 이다 — 한글은 그림마다 캡션 자리를
    # 만들어두지만 사용자가 채우지 않으면 빈 요소로 남는다(실측: 그림 39개 중
    # <hp:caption> 6개, 그 6개도 전부 텍스트 없음).
    caption: str | None = None


class SvgImage(BaseModel):
    """vectorize_image 의 출력. 클라이언트가 그대로 화면에 넣는다."""
    svg: str = Field(
        description="<svg> 로 시작해 </svg> 로 끝나는 마크업. viewBox 를 반드시 "
                    "포함한다. 설명이나 코드펜스를 붙이지 않는다")


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
