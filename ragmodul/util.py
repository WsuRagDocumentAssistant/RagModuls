#================================================
# util.py
#================================================
"""
순수 계산 도우미. 모델도 DB도 네트워크도 쓰지 않는다.
"""

import re

from .models.vocab_model import VocabPair

# 확장어에 콜론이 있으면 풀어쓴 말이 아니라 정의 문장이다.
_COLON = re.compile(r"[:：]")

# 풀어쓴 말은 이름이라 짧고 설명은 길다. 실측 —
#   정상 최대 49자 'Common European Framework of Reference of Language'
#   설명      62자 'Alpha(2개월)-Beta(4개월, 학기)-Gold(1년)의 ... 환류체계'
DEFAULT_MAX_EXPANSION = 60


def filter_vocab_pairs(pairs: list[VocabPair], skip: set[str] | None = None,
                       max_expansion: int = DEFAULT_MAX_EXPANSION,
                       ) -> tuple[list[VocabPair], list[tuple[VocabPair, str]]]:
    """LLM 이 뽑은 축약어 짝에서 못 쓸 것을 걸러낸다. (통과, [(버린 짝, 이유)])

    목적은 완벽한 선별이 아니라 사람이 검수할 양을 줄이는 것이다. 실측(문서 하나,
    로컬 모델 temperature=0, 재현 확인) — 45개 중 20개를 버렸다.

        term == expansion            6   '단기 -> 단기', '중장기 -> 중장기'
        콜론                          5   'SMART -> Attributable: 달성가능성'
        term 한 글자                   4   'A -> 평가 결과 반영 개선안 마련'  (표의 열 이름)
        skip 목록                     3   AI / DS / XR
        expansion 이 term 을 괄호로 품음  1   '7-Core -> 핵심역량(7-Core)'
        길이 초과                      1   'Agile -> Alpha(2개월)-Beta(...'

    통과한 25개 중 8~9개는 여전히 못 쓴다. 규칙으로 잡히지 않는 것들이다.
        SMART -> Specific / Measurable / ...   낱글자. 콜론이 없어 안 걸린다
        선이수프로그램 -> Pre-College            방향이 뒤집혔다(버릴 게 아니라 고칠 것)
        집중학기제 -> 현장실습 및 조기취업을 지원하는 제도   짧은 설명문
    규칙을 더 붙이면 이 문서에 맞춰지고 다른 문서에서 정상 항목을 버린다 — 실제로
    'expansion 이 term 을 포함하면 버림' 규칙이 '우송대 -> 우송대학교' 를 오탐한 적이
    있어서, 지금은 괄호로 감싼 경우만 잡는다.

    Args:
        skip: 제외할 축약어. AI/DS/XR 처럼 문서 고유가 아닌 일반어는 기계로 구분할
              수 없어서 목록으로 받는다.
    """
    skip = {s.casefold() for s in (skip or set())}
    kept: list[VocabPair] = []
    dropped: list[tuple[VocabPair, str]] = []
    seen: set[tuple[str, str]] = set()

    for pair in pairs:
        term, expansion = pair.term.strip(), pair.expansion.strip()
        reason = _reject(term, expansion, skip, max_expansion, seen)
        if reason:
            dropped.append((pair, reason))
        else:
            seen.add((term, expansion))
            kept.append(VocabPair(term=term, expansion=expansion))
    return kept, dropped


def _reject(term: str, expansion: str, skip: set[str],
            max_expansion: int, seen: set) -> str | None:
    """버릴 이유. 쓸 만하면 None."""
    if not term or not expansion:
        return "빈 값"
    if (term, expansion) in seen:
        return "중복"
    if _norm(term) == _norm(expansion):
        return "자기 자신"
    if len(term) <= 1:
        # 표의 열 이름(목푯값(A), 달성도(B/A))과 PDCA 순환을 축약어로 읽은 경우다.
        # 프롬프트로 배제해도 모델이 안 지킨다(로컬 12B 실측).
        return "한 글자"
    if term.casefold() in skip:
        return "일반어(목록)"
    if _COLON.search(expansion):
        return "콜론(정의문)"
    if re.search(rf"[(（]\s*{re.escape(term)}\s*[)）]", expansion):
        # '핵심역량(7-Core)' 를 그대로 확장어로 잡은 것. 원문 표기라 term 이 안에 있다.
        return "expansion 이 term 을 괄호로 품음"
    if len(expansion) > max_expansion:
        return f"{len(expansion)}자(설명문)"
    return None


def _norm(text: str) -> str:
    """공백을 없앤다. 표기 변형을 흡수해야 비교가 된다."""
    return re.sub(r"\s+", "", text or "")
