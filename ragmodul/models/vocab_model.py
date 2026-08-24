#================================================
# vocab_model.py
#================================================
"""
LLM 추출 결과의 데이터 모델.

pydantic 모델인 이유: 이게 LLM 출력의 검증 스키마이기도 하다. ai-rag-comm 에
구조화 출력이 없어서 평문으로 받은 JSON 을 이 타입으로 검증하고, 같은 모델의
JSON Schema 를 프롬프트에 붙여 형식을 알려준다. dataclass 로 하면 그 두 가지가 안 된다.

VocabPair 는 sql/schema.sql 의 두 테이블에 그대로 대응한다.
    {"term": "MD", "expansion": "마이크로디그리"}
      │              │
      └ vocab_short  └ vocab_expansion (short_id = 위 행의 id)
"""

from pydantic import BaseModel, Field


class VocabPair(BaseModel):
    """축약어 하나와 풀어쓴 말 하나. 확장어가 여럿이면 항목이 여러 개다.

    납작한 짝으로 두는 이유: vocab_expansion 행과 1:1 이고, 확장어 목록으로 묶으면
    IR 의 공백 변형이나 PAMS 의 Asia/Asian 변형을 넣을 때 어느 행이 어느 것인지
    다시 풀어야 한다.
    """
    term: str = Field(description="축약어. 예: MD")
    expansion: str = Field(description="풀어쓴 말. 예: 마이크로디그리")


class VocabPairs(BaseModel):
    """extract_vocab 의 출력. 없으면 빈 목록이다."""
    pairs: list[VocabPair]


class QueryTerms(BaseModel):
    """extract_query_terms 의 출력. 질의에 나온 축약어만 담는다."""
    terms: list[str] = Field(description="질의에 나온 축약어. 없으면 빈 목록")
