#================================================
# llm_service.py
#================================================
"""
LLM 호출 — openai SDK 를 직접 쓴다.

기능 두 개.
  extract_vocab(text)         문서에서 축약어-확장어 짝을 뽑는다 (사전을 만들 때)
  extract_query_terms(query)  사용자 질의에 나온 축약어를 뽑는다 (검색할 때)

프롬프트는 prompt/prompt.py, 출력 타입은 models/vocab_model.py 에 있다.
이 파일은 '날리고 받아 검증하는' 일만 한다.

ai-rag-comm 을 거치지 않는 이유
  그 모듈은 RestChannel -> BaseLLMApiInterface.chat -> OpenAIService 세 층이
  (prompt, model, max_tokens) 만 넘긴다. 그래서 아래 둘을 전달할 방법이 없다.
    base_url         로컬 OpenAI 호환 엔드포인트를 부를 수 없다.
    response_format  구조화 출력이 없어 평문에서 JSON 을 떼어내야 한다.
  세 층을 다 고치면 쓸 수 있지만, 그때까지는 SDK 를 직접 부른다.
  (temperature 도 못 넘기지만 그건 상관없다 — gpt-5.5 자체가 거부한다.)

동기 클라이언트를 쓴다. 이 패키지의 다른 서비스가 다 동기라 asyncio 를 끌어들일
이유가 없다.
"""

import json
import logging
import re

from pydantic import BaseModel, ValidationError

from ..models.vocab_model import QueryTerms, VocabPair, VocabPairs
from ..prompt import get_prompt

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.5"

# RAG 답변은 길다. 맥락 5개(5,000~6,000자)를 근거로 서술하면 답변만 수천 토큰이 되고,
# 추론 토큰을 먼저 쓰는 모델(gemini-2.5)은 한도가 모자라면 content 가 빈 채로 온다
# (실측: max_completion_tokens=16 에 응답이 None).
DEFAULT_MAX_TOKENS = 8192

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class LlmService:

    def __init__(self, api_key: str, *, model: str = DEFAULT_MODEL,
                 base_url: str | None = None, headers: dict | None = None,
                 temperature: float | None = None, structured: bool = True,
                 max_tokens: int = DEFAULT_MAX_TOKENS, retries: int = 1,
                 timeout: float = 60.0) -> None:
        """설정은 전부 인자로 받는다. 환경변수나 config.json 을 읽지 않는다.

        base_url:    OpenAI 호환 로컬 엔드포인트. '/v1' 까지 준다
                     (예 'http://117.16.166.22/v1'). SDK 가 '/chat/completions' 를 붙인다.
        headers:     엔드포인트가 요구하는 헤더. 예 {'x-user-id': 'npark-01'}
        temperature: 기본은 None — 아예 안 보낸다. gpt-5.5 는 temperature 를 거부한다
                     ("does not support 0.0 with this model. Only the default (1)").
                     추출은 0 으로 고정하고 싶지만 이 모델로는 불가능하고, 그래서
                     같은 문서에서도 결과가 흔들린다(실측 고유 축약어 17개 / 25개).
                     받는 모델을 쓸 때만 값을 준다.
        structured:  True 면 response_format 으로 스키마를 강제한다(API 가 보장).
                     로컬 서버가 그걸 못 받으면 False — 프롬프트에 스키마를 붙이고
                     평문에서 JSON 을 떼어낸다.
        retries:     파싱이 실패했을 때 다시 물어볼 횟수.
        """
        from openai import OpenAI

        if not api_key:
            raise ValueError(
                "API 키가 필요합니다. 애플리케이션에서 읽어 넘기세요 "
                "(이 라이브러리는 .env 나 config.json 을 직접 읽지 않습니다). "
                "키를 안 받는 로컬 엔드포인트라도 SDK 가 빈 값을 거부하므로 아무 값이나 넣으세요."
            )
        self.model = model
        self.temperature = temperature
        self.structured = structured
        self.max_tokens = max_tokens
        self.retries = retries

        self._client = OpenAI(api_key=api_key, base_url=base_url,
                              default_headers=headers, timeout=timeout)
        logger.info("LLM 준비: %s%s (temperature=%s, structured=%s)",
                    model, f" @ {base_url}" if base_url else "", temperature, structured)

    #------------------------------------------------┌> 기능

    def extract_vocab(self, text: str) -> list[VocabPair]:
        """문서 텍스트 하나에서 축약어 짝을 뽑는다.

        호출 하나 = 텍스트 하나다. 여러 개를 돌리는 건 부르는 쪽이 한다 — 이 모듈이
        목록을 삼키면 어디서 실패했는지, 중간에 멈출지를 쓰는 쪽이 통제할 수 없다.

            pairs = [p for parent in document.parents
                       for p in llm.extract_vocab(parent.content)]
        """
        result = self.send(get_prompt("vocab_user", text=text), VocabPairs,
                           system=get_prompt("vocab_system"))
        if result is None:
            logger.warning("축약어 추출 실패: %s...", text[:40])
            return []
        return list(result.pairs)



    def extract_query_terms(self, query: str) -> list[str]:
        """사용자 질의에 나온 축약어를 뽑는다. 이걸 vocab_short 에서 찾아 확장어를 붙인다."""
        result = self.send(get_prompt("query_terms_user", query=query), QueryTerms,
                           system=get_prompt("query_terms_system"))
        if result is None:
            logger.warning("질의 축약어 추출 실패: %s", query[:40])
            return []
        return [t.strip() for t in result.terms if t and t.strip()]



    def answer(self, query: str, contexts: list) -> str:
        """검색된 맥락으로 질문에 답한다.

        contexts 는 rerank() 를 지난 RetrievedContext 목록이다. 출처(breadcrumb)를
        번호와 함께 붙여 넘긴다 — 어느 맥락을 근거로 답했는지 확인할 수 있어야 하고,
        한 섹션 안에 비슷한 항목이 여러 개 있을 때(세부과제 2-1 과 2-2 처럼) 구분에도
        쓰인다.

        구조화 출력을 쓰지 않는다. 서식(수치에 이탤릭·밑줄)이 붙은 평문이 결과물이라
        JSON 스키마로 감싸면 서식과 싸운다.
        """
        block = _format_contexts(contexts)
        text = self.ask(
            get_prompt("answer_user", context=block, query=query),
            system=get_prompt("answer_system"),
        )
        logger.info("답변 생성: 맥락 %d개(%d자) -> %d자", len(contexts), len(block), len(text))
        return text



    def merge(self, question: str, answer_a: str, answer_b: str) -> str:
        """LLM 두 개가 낸 답변을 하나로 합친다.

        사용자가 provider 를 둘 고르면 각 LlmService 에 따로 answer() 를 부르고
        그 결과를 여기에 넘긴다. 어느 LLM 으로 병합할지는 부르는 쪽이 정한다 —
        이 메서드를 가진 서비스가 병합을 수행한다.

            a = services["gpt"].answer(q, ctxs)
            b = services["local_llm"].answer(q, ctxs)
            final = services["gpt"].merge(q, a, b)

        두 개까지만 받는다. 프롬프트가 '두 AI' 를 전제로 쓰여 있어서, 셋을 넣으려면
        프롬프트부터 바꿔야 한다.
        """
        if not answer_a or not answer_b:
            # 한쪽이 비면 합칠 게 없다. 있는 쪽을 그대로 준다.
            logger.warning("병합할 답변이 하나뿐이다. 그대로 돌려준다.")
            return answer_a or answer_b or ""

        text = self.ask(
            get_prompt("merge_user", question=question, answer_a=answer_a, answer_b=answer_b),
            system=get_prompt("merge_system"),
        )
        logger.info("병합: %d자 + %d자 -> %d자", len(answer_a), len(answer_b), len(text))
        return text



    def ask(self, prompt: str, system: str | None = None) -> str:
        """평문 답변. 구조화 출력을 쓰지 않는 호출에 쓴다."""
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=_messages(prompt, system),
            max_completion_tokens=self.max_tokens,
            **self._temperature_arg(),
        )
        return completion.choices[0].message.content or ""

    #------------------------------------------------┌> 전송

    def send(self, prompt: str, schema: type[BaseModel],
             system: str | None = None) -> BaseModel | None:
        """완성된 프롬프트를 날려 schema 로 돌려준다. 끝까지 실패하면 None.

        예외로 올리지 않는 이유: 청크 수백 개를 돌리는 중에 하나가 어긋났다고
        전체가 멈출 이유가 없다.
        """
        for attempt in range(self.retries + 1):
            result = (self._parse(prompt, schema, system) if self.structured
                      else self._parse_text(prompt, schema, system))
            if result is not None:
                return result
            logger.warning("응답 파싱 실패 (%d/%d)", attempt + 1, self.retries + 1)
        return None

    def _parse(self, prompt: str, schema: type[BaseModel],
               system: str | None) -> BaseModel | None:
        """response_format 으로 스키마를 강제한다. 형식이 어긋날 수가 없다."""
        completion = self._client.chat.completions.parse(
            model=self.model,
            messages=_messages(prompt, system),
            response_format=schema,
            max_completion_tokens=self.max_tokens,
            **self._temperature_arg(),
        )
        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            logger.warning("모델이 거부: %s", message.refusal)
            return None
        return message.parsed

    def _parse_text(self, prompt: str, schema: type[BaseModel],
                    system: str | None) -> BaseModel | None:
        """구조화 출력을 못 쓸 때. 스키마를 지시에 붙이고 평문에서 떼어낸다."""
        schema_note = (
            "\n\n아래 JSON 스키마에 정확히 맞는 JSON 만 출력한다. 설명이나 코드펜스 없이 JSON 만.\n"
            f"{json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)}"
        )
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=_messages(prompt, (system or "") + schema_note),
            max_completion_tokens=self.max_tokens,
            **self._temperature_arg(),
        )
        return _extract(schema, completion.choices[0].message.content or "")

    def _temperature_arg(self) -> dict:
        """None 이면 아예 안 보낸다 — 일부 모델은 temperature 자체를 거부한다."""
        return {} if self.temperature is None else {"temperature": self.temperature}


#------------------------------------------------┌> 내부

def _messages(prompt: str, system: str | None) -> list[dict]:
    """지시는 system, 데이터는 user 로 나눠 담는다."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _format_contexts(contexts: list) -> str:
    """검색 결과를 번호와 출처가 붙은 블록으로 만든다.

    출처(breadcrumb)를 붙이는 이유가 두 개다. 어느 맥락을 근거로 답했는지 확인할 수
    있어야 하고, 한 섹션 안에 비슷한 항목이 여러 개일 때(세부과제 2-1 과 2-2 가 같은
    3,011자 섹션 안에 있다) 모델이 구분할 단서가 된다.

    RetrievedContext 를 받지만 타입을 보지 않는다 — content/breadcrumb 만 쓴다.
    문자열 목록을 넘겨도 돈다.
    """
    blocks = []
    for i, context in enumerate(contexts, 1):
        if isinstance(context, str):
            blocks.append(f"[{i}]\n{context}")
            continue
        source = getattr(context, "breadcrumb", "") or getattr(context, "heading", "") or ""
        head = f"[{i}] 출처: {source}" if source else f"[{i}]"
        blocks.append(f"{head}\n{getattr(context, 'content', '')}")
    return "\n\n".join(blocks)


def _extract(schema: type[BaseModel], text: str) -> BaseModel | None:
    """평문에서 JSON 을 떼어내 검증한다.

    모델이 코드펜스나 앞뒤 설명을 붙이는 일이 흔해서 그대로 파싱하면 실패한다.
    """
    if not text:
        return None

    candidates = [text]
    fence = _FENCE.search(text)
    if fence:
        candidates.insert(0, fence.group(1))
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            return schema.model_validate_json(candidate)
        except ValidationError:
            continue
    return None
