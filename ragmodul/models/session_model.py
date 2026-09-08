#================================================
# session_model.py
#================================================
"""
세션 요약의 데이터 모델.

pydantic 인 이유는 vocab_model 과 같다 — 이게 LLM 출력의 검증 스키마다.
요약과 주제 두 값을 한 번의 호출로 받아야 하는데, 평문으로 받아 "요약:" / "주제:"
를 잘라 쓰면 모델이 형식을 흘릴 때마다 깨진다. response_format 으로 강제한다.
"""

from pydantic import BaseModel, Field


class SessionSummary(BaseModel):
    """summarize_session 의 출력."""
    summary: str = Field(
        description="창 밖으로 밀려난 대화의 압축. 세 문장 이내. 수치는 담지 않는다")
    topic: str = Field(
        description='이 대화가 무엇에 관한 것인지 한 줄. 예: "2026년 우송대 학과별 취업률"')
