#================================================
# util.py
#================================================
"""
순수 계산 도우미. 모델도 DB도 네트워크도 쓰지 않는다.
"""

import logging
import re

from .models.vocab_model import VocabPair

logger = logging.getLogger(__name__)

# 확장어에 콜론이 있으면 풀어쓴 말이 아니라 정의 문장이다.
_COLON = re.compile(r"[:：]")

_SPACE = re.compile(r"\s+")

# 영문·숫자로 시작하는 표기. 이런 건 단어 경계로 찾는다.
_ASCII_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+\-]*$")

# 확장어 길이 상한. 문단을 통째로 확장어에 넣는 사고만 막는 역할이다.
#
# 한때 60자였다. 표본 하나로 그은 선이었다 —
#   정상 최대 50자 'Common European Framework of Reference of Language'
#   설명      62자 'Alpha(2개월)-Beta(4개월, 학기)-Gold(1년)의 ... 환류체계'
# 둘의 간격이 12자뿐이라 조금만 긴 정식 명칭도 잘렸다. 영문 기관명은 이보다 긴 게
# 흔하다. 설명문은 콜론 규칙과 프롬프트가 이미 잡는다(gpt 17개 중 설명문 0개).
DEFAULT_MAX_EXPANSION = 80


def filter_vocab_pairs(pairs: list[VocabPair], skip: set[str] | None = None,
                       max_expansion: int = DEFAULT_MAX_EXPANSION,
                       ) -> tuple[list[VocabPair], list[tuple[VocabPair, str]]]:
    """LLM 이 뽑은 축약어 짝에서 못 쓸 것을 걸러낸다. (통과, [(버린 짝, 이유)])

    지금은 gpt 로 뽑으면 한 건도 안 버린다(11개 중 0개, 17개 중 0개). 남겨두는 이유는
    둘이다 — 중복 제거(recheck_vocab 을 켜면 1차·2차 결과를 이어 붙인다. 2차가 '이미
    있는 건 내지 마라'를 늘 지키지는 않는다)와, 모델·프롬프트를 바꿀 때의 안전망이다.
    순수 계산이라 비용이 없고, dropped 목록이 뭔가 이상해졌다는 신호가 된다.

    선별이 실제로 필요했던 때의 실측(문서 하나, 로컬 모델 temperature=0, 재현 확인) —
    45개 중 20개를 버렸다.

        term == expansion            6   '단기 -> 단기', '중장기 -> 중장기'
        콜론                          5   'SMART -> Attributable: 달성가능성'
        term 한 글자                   4   'A -> 평가 결과 반영 개선안 마련'  (표의 열 이름)
        skip 목록                     3   AI / DS / XR  (지금은 목록을 비워둔다)
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
    return _SPACE.sub("", text or "")


#------------------------------------------------┌> 글 묶기

# 로컬 모델(gemma-4-12B-it) 한 요청에 들어가는 본문 크기. 실측으로 계산했다 —
#   컨텍스트 8192 토큰(입력+출력). 엔드포인트가 그대로 알려준다:
#     "This model's maximum context length is 8192 tokens. However, you requested
#      8 output tokens and your prompt contains at least 8185 input tokens"
#   한국어 12,000자 = 8,185 토큰 -> 글자당 0.68 토큰
#   8192 - 2048(출력 자리) - 950(vocab 시스템 프롬프트 1,395자) = 5,194 토큰 ≈ 7,600자
# 7,600 을 그대로 쓰면 프롬프트를 조금 손볼 때마다 넘치므로 여유를 둔다.
DEFAULT_PACK_CHARS = 6500


def pack_texts(texts: list[str], max_chars: int = DEFAULT_PACK_CHARS) -> list[str]:
    """글 조각을 순서대로 이어 붙여 max_chars 이하 묶음들로 만든다.

    컨텍스트가 좁은 모델에 문서를 나눠 보낼 때 쓴다. 부모(헤딩) 단위 조각을 넘기면
    조각 경계에서만 끊으므로 헤딩이 반토막 나지 않는다.

    자르거나 버리지 않는다. 넘칠 조각은 다음 묶음의 첫 조각이 된다 — 모든 조각이
    정확히 한 번 들어가고, 묶음들의 글자 합은 입력 합과 같다.

    조각 하나가 혼자 max_chars 보다 크면 그 조각만 담은 묶음이 되고 그 묶음은 상한을
    넘는다. 부르는 쪽이 잘라야 한다(실측 문서는 최대 부모가 5,987자라 해당 없음).

    묶는 이유는 컨텍스트를 채우는 것보다 경계를 줄이는 데 있다. 조각마다 한 번씩
    부르면 조각 사이 모든 지점이 요청 경계가 되고, 앞에서 정의한 축약어와 뒤에서 쓴
    자리가 갈라진다(실측: 부모별 32회가 통째 1회보다 적게 찾았다).
    """
    groups: list[str] = []
    current: list[str] = []
    size = 0
    for text in texts:
        if not text:
            continue
        if current and size + len(text) > max_chars:
            groups.append("\n\n".join(current))
            current, size = [], 0
        current.append(text)
        size += len(text)
    if current:
        groups.append("\n\n".join(current))
    return groups


#------------------------------------------------┌> 질의 확장

def expand_query(query: str, vocab: dict[str, list[str]] | None) -> str:
    """질의에 걸리는 표기의 짝을 뒤에 덧붙인다. 원문은 그대로 둔다.

    vocab 은 load_vocab() 이 주는 {축약어: [확장어, ...]} 다.

    왜 문서가 아니라 질의인가
      색인 쪽에 확장어를 박으면 그 말이 실제보다 흔해 보여 IDF 가 떨어지고 sparse 가
      변별력을 잃는다 — 도우려는 짓이 망친다. 사전을 고칠 때마다 재색인해야 하는
      문제도 있다. Elasticsearch 도 query-time 을 권한다.

    왜 치환이 아니라 덧붙이기인가
      치환하면 원래 표기로 물어본 사람을 놓친다. 덧붙이면 잃는 게 없다.

    dense·sparse 양쪽에 같은 확장 질의를 쓴다. 실측(문서 하나, 사례 8건) —
        dense  개선 5 / 악화 0 / 변화없음 3
        sparse 개선 3 / 악화 0 / 변화없음 5
    둘은 서로 보완한다. '대학성과통합관리 시스템' 은 sparse 가 못 잡고 dense 가 잡고,
    'MD' 는 sparse 가 더 많이 잡는다. 어느 쪽도 나빠진 사례가 없어 함께 쓴다.

    다만 그 8건은 'A 표기로 묻고 B 표기만 있는 청크를 목표로' 만든 사례다. 사전
    단어가 스치듯 들어간 평범한 질의가 나빠지는지는 재지 못했다(평가셋 30문항 중
    사전을 건드리는 질문이 0개였다).
    """
    if not query or not vocab:
        return query

    index = _expansion_index(vocab)
    norm_query = _norm(query)
    additions: list[str] = []

    for key, values in index.items():
        if not _in_query(key, query, norm_query):
            continue
        for value in values:
            # 이미 질의에 있는 말은 덧붙이지 않는다
            if _norm(value) not in norm_query and value not in additions:
                additions.append(value)

    if not additions:
        return query
    logger.debug("질의 확장: +%s", additions)
    return f"{query} {' '.join(additions)}"


def _expansion_index(vocab: dict[str, list[str]]) -> dict[str, list[str]]:
    """{축약어: [확장어]} 를 양방향 조회표로 바꾼다.

    양방향이어야 질의 한쪽만 건드려서 문서 양쪽을 덮는다. 문서에는 축약어만 적힌
    청크도 있고 풀어쓴 말만 적힌 청크도 있는데 문서를 못 고치기 때문이다.
        'IR' 이 오면            -> '대학성과통합관리 시스템' 을 덧붙인다
        '대학성과통합관리' 가 오면 -> 'IR' 을 덧붙인다

    같은 축약어에 달린 다른 표기 변형끼리도 서로 덧붙인다(PAMS 의 Asia/Asian).

    사전이 십여 개라 질의마다 만들어도 비용이 없다. 미리 만들어 들고 다니면 사전을
    고쳤을 때 낡은 색인을 쓰게 된다.
    """
    index: dict[str, list[str]] = {}

    def add(key: str, values: list[str]) -> None:
        if not key:
            return
        bucket = index.setdefault(key, [])
        for value in values:
            if value and value not in bucket:
                bucket.append(value)

    for short, expansions in vocab.items():
        short = (short or "").strip()
        expansions = [e.strip() for e in expansions if e and e.strip()]
        if not short or not expansions:
            continue
        add(short, expansions)
        for expansion in expansions:
            add(expansion, [short] + [e for e in expansions if e != expansion])
    return index


def _in_query(key: str, query: str, norm_query: str) -> bool:
    """이 표기가 질의에 나오나.

    영문·숫자 축약어는 단어 경계로 본다. 부분문자열로 찾으면 'IR' 이 'IRB' 나
    'FIRST' 에 걸려 엉뚱한 확장어가 붙는다. 경계를 [A-Za-z0-9] 아님으로 잡으므로
    'IR성과관리팀' 처럼 한글이 바로 붙은 경우는 정상적으로 걸린다.

    한글이 섞인 표기는 공백만 무시하고 부분문자열로 본다. 긴 구절이라 오탐 위험이
    낮고, 조사가 붙어도('시스템은') 앞부분이 걸려야 한다.
    """
    if _ASCII_KEY.match(key):
        pattern = rf"(?<![A-Za-z0-9]){re.escape(key)}(?![A-Za-z0-9])"
        return re.search(pattern, query, re.IGNORECASE) is not None
    return _norm(key) in norm_query
