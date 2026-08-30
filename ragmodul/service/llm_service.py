#================================================
# llm_service.py
#================================================
"""
LLM 호출 — ai-rag-comm 의 채널로 나간다.

하나의 LlmService 가 provider 넷을 들고 관리한다. 호출할 때 고른다.

    from ai_rag_comm import load_config
    cfg = load_config()                       # config.json + .env
    llm = LlmService(cfg.llm_api, cfg.local_llm, default="local_llm")

    llm.answer(q, ctxs)                       # 기본 provider
    llm.answer(q, ctxs, provider="gpt")
    llm.merge(q, a, b, provider="claude")
    await llm.aanswer(q, ctxs, provider="gemini")

설정을 직접 읽지 않는다. ai-rag-comm 의 load_config() 결과를 받는다 — 모델명·엔드포인트·
키·타임아웃이 이미 거기 다 있어서, 우리가 표를 또 만들면 두 곳이 어긋난다.

provider 이름
    gpt / claude / gemini    llm_api_config 에서. RestChannel 로 나간다.
    local_llm                local_llm_config 에서. LocalLLMChannel 로 나간다.

채널은 처음 쓸 때 만들어 붙들고 있다. 요청마다 만들면 연결이 쌓인다 — LocalLLMChannel 은
같은 인자면 클라이언트를 재사용하지만(그쪽 캐시), 채널 객체를 매번 만드는 비용은 남는다.

기능마다 async 본체와 동기 껍데기가 짝으로 있다. 채널이 async 라서 그렇다. 동기 쪽에서
이미 이벤트 루프가 돌고 있으면 a- 접두사 쪽을 await 하면 된다.

프롬프트는 prompt/prompt.py(지시/데이터 두 벌), 출력 타입은 models/vocab_model.py.
이 파일은 '날리고 받아 검증하는' 일만 한다.
"""

import asyncio
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from ..models.vocab_model import QueryTerms, VocabPair, VocabPairs
from ..prompt import get_prompt

logger = logging.getLogger(__name__)

CLOUD_PROVIDERS = ("gpt", "claude", "gemini")
LOCAL_PROVIDER = "local_llm"

# RAG 답변은 길다. 맥락 5개(5,000~6,000자)를 근거로 서술하면 답변만 수천 토큰이 되고,
# 추론 토큰을 먼저 쓰는 모델(gemini)은 한도가 모자라면 content 가 빈 채로 온다
# (실측: max_tokens=16 에 응답이 None).
CLOUD_MAX_TOKENS = 8192

# 로컬 모델은 입력+출력 합쳐 8192 토큰이 상한이다. 출력에 다 주면 입력 자리가 0 이 되어
# 400 이 난다("you requested 8192 output tokens ... upper bound for 0 input tokens").
LOCAL_MAX_TOKENS = 2048

# 로컬에 실을 맥락 상한(글자). 실측으로 계산했다 —
#   컨텍스트 8192 토큰, 한국어 글자당 0.68 토큰(12,000자 = 8,185 토큰)
#   8192 - 2048(출력) - 1,400(답변 시스템 프롬프트) ≈ 4,700 토큰 ≈ 6,900자
# 여유를 두고 6,500 으로 잡는다.
#
# 없으면 실제로 터진다: 문서와 무관한 질의가 오면 조각이 여러 부모에 흩어지고 섹션이
# 전부 승격돼(child 하나뿐인 부모는 hit 1개로 비율 1.0) 맥락이 17,286자가 됐다.
# 게이트웨이가 413 Payload Too Large 로 잘랐다.
LOCAL_CONTEXT_CHARS = 6500

# temperature 를 보내면 400 이 나는 provider. claude 는 ai-rag-comm 이 경고 후 무시하고,
# gemini/local 은 정상으로 받는다. gpt 만 모델이 거부한다 —
# "does not support 0.0 with this model. Only the default (1) value is supported".
# 모델을 바꾸면(gpt-4o 등) 달라지므로 생성 시 덮어쓸 수 있다.
NO_TEMPERATURE = ("gpt",)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass
class _Provider:
    """provider 하나의 설정과 채널. 채널은 처음 쓸 때 만든다."""
    name: str
    model: str
    max_tokens: int
    temperature: float | None = None
    api_key: str = ""
    base_url: str | None = None
    headers: dict | None = None
    timeout: float | None = None
    # 맥락에 실을 수 있는 글자 수. None 이면 제한하지 않는다(클라우드는 넉넉하다).
    context_chars: int | None = None
    llm_api_config: Any = field(default=None, repr=False)
    # 웹서치 여부로 채널이 갈린다. enable_web_search 가 RestChannel 의 생성자 인자라
    # 호출마다 켜고 끌 수 없어서, 켠 것과 끈 것을 따로 들고 있는다.
    # 채널은 껍데기고 무거운 HTTP 클라이언트는 ai-rag-comm 의 모듈 캐시가 관리한다.
    channels: dict = field(default_factory=dict, repr=False)


class LlmService:

    def __init__(self, llm_api_config=None, local_llm_config=None, *,
                 default: str | None = None,
                 temperature: float | None = 0.0,
                 no_temperature: tuple[str, ...] = NO_TEMPERATURE,
                 structured: bool = True) -> None:
        """ai-rag-comm 의 load_config() 결과를 받는다.

        Args:
            llm_api_config:   cfg.llm_api. 없으면 클라우드 provider 를 안 만든다.
            local_llm_config: cfg.local_llm. 없으면 local_llm 을 안 만든다.
            default:          provider 를 안 넘겼을 때 쓸 이름.
            temperature:      기본 0 — 같은 입력에 같은 결과가 나오게. no_temperature
                              에 든 provider 에는 안 보낸다.
            no_temperature:   temperature 를 거부하는 provider. 모델을 바꾸면 조정한다.
            structured:       response_format 으로 스키마를 강제한다. 넷 다 지원하는
                              것을 확인했다(로컬 포함, strict=True 까지).
        """
        self.structured = structured
        self._closed = False
        self._providers: dict[str, _Provider] = {}

        if llm_api_config is not None:
            keys = {
                "gpt": getattr(llm_api_config, "openai_api_key", ""),
                "claude": getattr(llm_api_config, "anthropic_api_key", ""),
                "gemini": getattr(llm_api_config, "gemini_api_key", ""),
            }
            models = getattr(llm_api_config, "default_models", {}) or {}
            for name in CLOUD_PROVIDERS:
                if not keys[name] or name not in models:
                    continue        # 키나 모델이 없으면 그 provider 는 만들지 않는다
                self._providers[name] = _Provider(
                    name=name, model=models[name], max_tokens=CLOUD_MAX_TOKENS,
                    temperature=None if name in no_temperature else temperature,
                    api_key=keys[name],
                    timeout=getattr(llm_api_config, "timeout", None),
                    llm_api_config=llm_api_config,
                )

        if local_llm_config is not None:
            self._providers[LOCAL_PROVIDER] = _Provider(
                name=LOCAL_PROVIDER,
                model=local_llm_config.model,
                max_tokens=LOCAL_MAX_TOKENS,
                temperature=None if LOCAL_PROVIDER in no_temperature else temperature,
                base_url=local_llm_config.base_url,
                headers=getattr(local_llm_config, "headers", None),
                timeout=getattr(local_llm_config, "timeout", None),
                context_chars=LOCAL_CONTEXT_CHARS,
            )

        if not self._providers:
            raise ValueError(
                "쓸 수 있는 provider 가 없습니다. llm_api_config 에 키와 default_models 가 "
                "있는지, local_llm_config 가 있는지 확인하세요."
            )
        self.default = default or next(iter(self._providers))
        if self.default not in self._providers:
            raise ValueError(
                f"default {self.default!r} 를 만들지 못했습니다 "
                f"(가능: {', '.join(self._providers)}). 키나 모델 설정을 확인하세요."
            )
        logger.info("LLM 준비: %s (기본 %s)", ", ".join(self._providers), self.default)

    #------------------------------------------------┌> 답변 (async 본체)

    async def aanswer(self, query: str, contexts: list, provider: str | None = None,
                      web_search: bool = True, external: list | None = None) -> str:
        """검색된 맥락으로 질문에 답한다.

        contexts 는 rerank() 를 지난 RetrievedContext 목록이다. 출처(breadcrumb)를
        번호와 함께 붙여 넘긴다 — 어느 맥락을 근거로 답했는지 확인할 수 있어야 하고,
        한 섹션 안에 비슷한 항목이 여러 개 있을 때(세부과제 2-1 과 2-2 처럼) 구분에도
        쓰인다.

        구조화 출력을 쓰지 않는다. 결과물이 서식(수치에 이탤릭·밑줄)이 붙은 평문이라
        JSON 스키마로 감싸면 서식과 싸운다. 그래서 웹서치를 켤 수 있다.

        web_search 기본이 켜짐이다. 맥락에 없는 것을 물으면 모델이 웹에서 찾아
        보완한다. 로컬은 지원하지 않아 무시된다(초안 모델이 로컬이면 자동으로 꺼진다).

        external 은 유사도로 찾은 외부 API 목록이다. 맥락과 섞지 않고 별도 절로
        내려보낸다 — 제목뿐이라 근거가 될 수 없는데 Context 에 끼면 모델이 사실처럼
        인용한다. 맥락 예산(context_chars)에는 넣지 않는다. top_k=1 이라 한 줄이다.
        """
        prov = self._provider(provider)
        block, used = _format_contexts(contexts, prov.context_chars)
        system, user = get_prompt("answer", context=block, query=query,
                                  external=_format_external(external))
        text = await self.aask(user, provider, system=system, web_search=web_search)
        logger.info("[%s] 답변: 맥락 %d개(%d자) -> %d자",
                    provider or self.default, used, len(block), len(text))
        return text

    async def arefine(self, query: str, contexts: list, draft: str,
                      provider: str | None = None, web_search: bool = True,
                      external: list | None = None) -> str:
        """다른 모델이 만든 답변 초안을 Context 와 견주어 고친다.

        local_llm 이 초안을 만들고 사용자가 고른 모델이 다듬는 흐름에 쓴다.

        Context 를 함께 넘긴다 — 초안만 주면 사실이 맞는지 볼 수 없고, 수치가
        원데이터인지 계산값인지도 다시 매길 수 없다. 대신 클라우드로 나가는 입력이
        직접 답할 때와 비슷해진다(맥락 + 초안). 토큰을 아끼려면 초안만 넘기는 별도
        경로가 필요한데, 그때는 초안의 오류를 그대로 물려받는다.

        초안이 비어 있으면 호출하지 않는다 — 다듬을 게 없다.

        web_search 기본이 켜짐이다. 초안이 로컬 모델이라 바깥을 못 보므로, 다듬는
        쪽에서 웹을 뒤져 보완한다.
        """
        if not draft or not draft.strip():
            logger.warning("초안이 비어 있다. 다듬기를 건너뛴다.")
            return ""

        prov = self._provider(provider)
        block, used = _format_contexts(contexts, prov.context_chars)
        system, user = get_prompt("refine", context=block, query=query, draft=draft,
                                  external=_format_external(external))
        text = await self.aask(user, provider, system=system, web_search=web_search)
        logger.info("[%s] 다듬기: 초안 %d자 + 맥락 %d개(%d자) -> %d자",
                    provider or self.default, len(draft), used, len(block), len(text))
        return text

    async def arefine_all(self, query: str, contexts: list, draft: str,
                          providers: list[str], parallel: bool = True,
                          web_search: bool = True,
                          external: list | None = None) -> dict[str, str]:
        """고른 모델들이 같은 초안을 각자 다듬는다. {provider: 다듬은 답변}.

        하나가 죽어도 나머지는 돌려준다 — 한도(429)나 키 없음으로 한쪽만 실패하는 게
        흔하다. 실패한 provider 는 결과에 없으므로, 부르는 쪽이 providers 와 대조하면
        무엇이 빠졌는지 알 수 있다.

        같은 provider 가 두 번 들어오면 한 번만 부른다.

        초안을 만든 provider 를 여기 넣지 않는 건 부르는 쪽 책임이다. 자기 초안을
        자기가 다듬으면 호출만 하나 늘고 결과는 거의 같다.

        parallel=True 면 동시에 던진다. LLM 호출은 전부 네트워크 대기라 GIL 이 걸림돌이
        아니다 — 기다리는 동안 GIL 을 놓으므로 단일 스레드에서도 실제로 겹친다. 걸리는
        시간이 합이 아니라 가장 느린 하나가 된다.

        parallel=False 는 분당 토큰 한도에 걸릴 때 쓴다. 큰 맥락이 붙은 요청 여러 개가
        같은 순간에 나가면 429 가 나는데(실측), 순차면 요청이 끝나야 다음이 나가 자연히
        벌어진다.
        """
        # 중복을 지우면서 순서는 유지한다 — 결과 dict 의 순서가 요청 순서와 맞는다
        targets = list(dict.fromkeys(p for p in providers if p))
        if not targets:
            return {}

        if parallel:
            # return_exceptions 를 안 켜면 하나가 터질 때 나머지가 취소되고 예외만 올라온다
            results = await asyncio.gather(
                *(self.arefine(query, contexts, draft, name, web_search, external)
                  for name in targets),
                return_exceptions=True,
            )
        else:
            results = []
            for name in targets:
                try:
                    results.append(
                        await self.arefine(query, contexts, draft, name, web_search,
                                           external))
                except Exception as e:
                    results.append(e)

        refined: dict[str, str] = {}
        for name, result in zip(targets, results):
            if isinstance(result, BaseException):
                logger.warning("[%s] 다듬기 실패: %s - %s",
                               name, type(result).__name__, result)
            else:
                refined[name] = result
        return refined

    async def amerge(self, question: str, answers: list[str],
                     provider: str | None = None) -> str:
        """여러 LLM 이 낸 답변을 하나로 합친다. 개수 제한은 없다.

        어느 provider 로 병합할지는 부르는 쪽이 정한다 — 답변을 낸 것과 달라도 된다.
        판단 작업이라 더 센 모델을 쓰고 싶을 수 있다.

        셋 이상도 한 번에 넘긴다. 둘씩 접어 올리면(merge(A,B) 뒤에 merge(AB,C))
        호출이 늘 뿐 아니라, C 가 '이미 합쳐진 것' 과 1:1 로 겨루게 되어 A·B 의 근거가
        묽어진다. 프롬프트의 '더 근거가 명확한 쪽을 따르라' 는 전부 나란히 놓고 봐야
        성립한다.
        """
        answers = [a for a in answers if a and a.strip()]
        if len(answers) < 2:
            logger.warning("병합할 답변이 %d개다. 그대로 돌려준다.", len(answers))
            return answers[0] if answers else ""

        blocks = "\n\n".join(f"<답변{i}>\n{a}\n</답변{i}>"
                             for i, a in enumerate(answers, 1))
        system, user = get_prompt("merge", question=question, answers=blocks)
        # 병합은 웹서치를 안 쓴다. 이미 만들어진 답변들을 합치는 일이라 바깥을 볼
        # 이유가 없고, 켜면 없던 내용이 새로 섞여 들어온다.
        text = await self.aask(user, provider, system=system)
        logger.info("[%s] 병합: %s -> %d자", provider or self.default,
                    " + ".join(f"{len(a):,}자" for a in answers), len(text))
        return text

    #------------------------------------------------┌> 축약어 사전 (async 본체)

    async def aextract_vocab(self, text: str, provider: str | None = None) -> list[VocabPair]:
        """텍스트에서 축약어 짝을 뽑는다.

        문서 전체를 한 덩어리로 넣는다. 부모별로 나눠 32번 부르면 표기 변형
        (7-Core / 7-CORE)을 모델이 정리하지 못한다 — 각 호출이 자기 텍스트만 보기
        때문이다. 놓치는 건 recheck_vocab 으로 메운다.

        로컬 모델은 컨텍스트가 8192 토큰이라 문서 전체(약 45k)를 못 받는다. 클라우드로.

        웹서치는 쓰지 않는다. 문서에서 뽑는 작업이라 바깥을 볼 이유가 없고, 웹서치를
        켜면 ai-rag-comm 이 response_format 을 경고만 남기고 버려서 구조화 출력이
        깨진다(gemini 는 tools 와 response_schema 를 같이 못 쓴다).
        """
        system, user = get_prompt("vocab", text=text)
        result = await self.asend(user, VocabPairs, provider, system=system)
        if result is None:
            logger.warning("축약어 추출 실패: %s...", text[:40])
            return []
        return list(result.pairs)

    async def aextract_vocab_all(self, texts: list[str], provider: str | None = None,
                                 parallel: bool = True, max_concurrent: int = 4,
                                 ) -> list[VocabPair]:
        """여러 조각에서 각각 뽑아 합친다. 같은 짝은 한 번만 남긴다.

        컨텍스트가 좁은 모델(로컬)에 문서를 나눠 보낼 때 쓴다. 조각은 util.pack_texts
        로 만든다 — 부모 경계에서만 끊어 헤딩이 반토막 나지 않는다.

        조각 하나가 실패해도 나머지는 살린다. 로그에만 남기고 넘어간다.

        max_concurrent 는 동시에 나가는 요청 수다. 로컬 엔드포인트가 여러 대에 로드
        밸런싱되어 있어 그 대수만큼은 겹쳐도 된다. 상한을 안 두고 18개를 한꺼번에
        던지면 공용 게이트웨이가 버티지 못한다.

        정확도는 문서를 통째로 한 번 보내는 것보다 낮다. 조각 경계에서 앞의 정의와
        뒤의 사용이 갈라지기 때문이다(실측: 통째 1회 17짝 / 부모별 32회 17짝). 묶으면
        경계가 절반으로 줄어 그 손해가 작아진다.
        """
        if not texts:
            return []

        semaphore = asyncio.Semaphore(max_concurrent if parallel else 1)

        async def one(index: int, text: str):
            async with semaphore:
                pairs = await self.aextract_vocab(text, provider)
                logger.info("사전 추출 %d/%d: %d자 -> %d짝",
                            index, len(texts), len(text), len(pairs))
                return pairs

        results = await asyncio.gather(
            *(one(i, t) for i, t in enumerate(texts, 1)), return_exceptions=True)

        merged: list[VocabPair] = []
        seen: set[tuple[str, str]] = set()
        for index, result in enumerate(results, 1):
            if isinstance(result, BaseException):
                logger.warning("사전 추출 %d/%d 실패: %s - %s",
                               index, len(texts), type(result).__name__, result)
                continue
            for pair in result:
                key = (pair.term, pair.expansion)
                if key not in seen:
                    seen.add(key)
                    merged.append(pair)
        logger.info("사전 추출 합계: 조각 %d개 -> %d짝", len(texts), len(merged))
        return merged

    async def arecheck_vocab(self, text: str, found: list[VocabPair],
                             provider: str | None = None,
                             ) -> list[VocabPair]:
        """이미 뽑은 목록을 보여주고 빠뜨린 축약어를 다시 훑게 한다.

        한 번에 다 못 뽑는다. 실측 — 1차 11개, 재검토로 7개 추가(JA, K-MOOC, IPA,
        CEFR 등)해서 18개. 부모별 32회(14개)보다 많고 호출은 2회다.

        새로 찾은 것만 돌려준다. 합치는 건 부르는 쪽이 한다.
        """
        listing = "\n".join(f"- {p.term} -> {p.expansion}" for p in found) or "(없음)"
        system, user = get_prompt("vocab_recheck", found=listing, text=text)
        result = await self.asend(user, VocabPairs, provider, system=system)
        if result is None:
            logger.warning("축약어 재검토 실패")
            return []
        # 모델이 이미 있는 것을 다시 낼 때가 있어 여기서 한 번 더 거른다.
        known = {(p.term, p.expansion) for p in found}
        fresh = [p for p in result.pairs if (p.term, p.expansion) not in known]
        logger.info("재검토: %d개 중 새로 %d개", len(result.pairs), len(fresh))
        return fresh

    async def aextract_query_terms(self, query: str, provider: str | None = None) -> list[str]:
        """사용자 질의에 나온 축약어를 뽑는다. 이걸 vocab_short 에서 찾아 확장어를 붙인다."""
        system, user = get_prompt("query_terms", query=query)
        result = await self.asend(user, QueryTerms, provider, system=system)
        if result is None:
            logger.warning("질의 축약어 추출 실패: %s", query[:40])
            return []
        return [t.strip() for t in result.terms if t and t.strip()]

    #------------------------------------------------┌> 전송 (async 본체)

    async def asend(self, prompt: str, schema: type[BaseModel],
                    provider: str | None = None, system: str | None = None,
                    retries: int = 1) -> BaseModel | None:
        """스키마를 강제해 받고 검증한다. 끝까지 실패하면 None.

        structured 면 response_format 으로 API 가 형식을 보장한다. 아니면 지시 뒤에
        JSON Schema 를 붙이고 평문에서 떼어낸다 — 데이터가 길 때(문서 전체 추출은
        9만 자) 형식 지시가 앞에 있으면 묻히므로 뒤에 붙인다.

        웹서치는 여기서 절대 켜지 않는다. 켜면 ai-rag-comm 이 response_format 을
        경고만 남기고 버려서 형식 보장이 사라진다(gemini 는 tools 와 response_schema 를
        같은 요청에 못 쓴다). 구조화 출력과 웹서치는 함께 못 간다.

        예외로 올리지 않는 이유: 청크 수백 개를 돌리는 중에 하나가 어긋났다고
        전체가 멈출 이유가 없다.
        """
        fmt = _bare_schema(schema) if self.structured else None
        if not self.structured:
            import json
            system = (system or "") + (
                "\n\n아래 JSON 스키마에 정확히 맞는 JSON 만 출력한다. 설명이나 코드펜스 없이 JSON 만.\n"
                f"{json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)}"
            )

        name = provider or self.default
        for attempt in range(retries + 1):
            text = await self.aask(prompt, provider, system=system, response_format=fmt)
            parsed = _extract(schema, text)
            if parsed is not None:
                return parsed
            logger.warning("[%s] JSON 파싱 실패 (%d/%d)", name, attempt + 1, retries + 1)
        return None

    async def aask(self, prompt: str, provider: str | None = None,
                   system: str | None = None,
                   response_format: dict | None = None,
                   web_search: bool = False) -> str:
        """채널로 보내고 평문을 받는다.

        response_format 에는 '알맹이' JSON Schema 만 넣는다. provider 별 봉투는
        ai-rag-comm 이 씌운다 — 우리가 미리 씌우면 이중으로 감싸져 400 이 난다.

        web_search 는 기본이 꺼짐이다. 켜면 클라우드 provider 가 웹을 뒤져 답한다
        (로컬은 지원하지 않아 무시된다). response_format 과 함께 쓰면 형식 보장이
        사라지므로 둘을 같이 주지 않는다.
        """
        if web_search and response_format is not None:
            logger.warning("웹서치와 구조화 출력은 함께 못 씁니다. 웹서치를 끕니다.")
            web_search = False
        prov = self._provider(provider)
        payload: dict[str, Any] = {
            "prompt": prompt,
            "model": prov.model,
            "max_tokens": prov.max_tokens,
        }
        if system:
            payload["system"] = system
        if prov.temperature is not None:
            payload["temperature"] = prov.temperature
        if response_format is not None:
            payload["response_format"] = response_format
        return await self._channel(prov, web_search).call(payload) or ""


    #------------------------------------------------┌> 동기 껍데기

    def answer(self, query: str, contexts: list, provider: str | None = None,
               web_search: bool = True, external: list | None = None) -> str:
        return _run(self.aanswer(query, contexts, provider, web_search, external))

    def refine(self, query: str, contexts: list, draft: str,
               provider: str | None = None, web_search: bool = True,
               external: list | None = None) -> str:
        return _run(self.arefine(query, contexts, draft, provider, web_search,
                                 external))

    def refine_all(self, query: str, contexts: list, draft: str,
                   providers: list[str], parallel: bool = True,
                   web_search: bool = True,
                   external: list | None = None) -> dict[str, str]:
        return _run(self.arefine_all(query, contexts, draft, providers, parallel,
                                     web_search, external))

    def merge(self, question: str, answers: list[str],
              provider: str | None = None) -> str:
        return _run(self.amerge(question, answers, provider))

    def extract_vocab(self, text: str, provider: str | None = None) -> list[VocabPair]:
        return _run(self.aextract_vocab(text, provider))

    def extract_vocab_all(self, texts: list[str], provider: str | None = None,
                          parallel: bool = True, max_concurrent: int = 4,
                          ) -> list[VocabPair]:
        return _run(self.aextract_vocab_all(texts, provider, parallel, max_concurrent))

    def recheck_vocab(self, text: str, found: list[VocabPair],
                      provider: str | None = None) -> list[VocabPair]:
        return _run(self.arecheck_vocab(text, found, provider))

    def extract_query_terms(self, query: str, provider: str | None = None) -> list[str]:
        return _run(self.aextract_query_terms(query, provider))

    def ask(self, prompt: str, provider: str | None = None,
            system: str | None = None, response_format: dict | None = None,
            web_search: bool = False) -> str:
        return _run(self.aask(prompt, provider, system, response_format, web_search))

    #------------------------------------------------┌> 관리

    def providers(self) -> list[str]:
        """쓸 수 있는 provider 이름."""
        return list(self._providers)

    async def aclose(self) -> None:
        """만들어둔 채널의 연결을 정리한다. 프로세스 종료 시 한 번.

        되돌릴 수 없다. ai-rag-comm 의 클라이언트 캐시가 모듈 전역이라, 닫고 나면
        채널을 새로 만들어도 닫힌 클라이언트를 돌려받는다. 다시 쓰려면 LlmService
        자체를 새로 만들어야 한다.

        LocalLLMChannel 만 aclose() 를 가진다. RestChannel 은 그 메서드를 밖으로
        내보내지 않는다 — 미구현이 아니라 전달 누락이다. 감싸고 있는
        OpenAIService/ClaudeService/GeminiService 에는 셋 다 aclose() 가 있다.
        남의 패키지 비공개 속성(channel._client)을 건드리지 않기로 하고, 클라우드
        연결은 프로세스가 끝날 때 OS 가 정리하게 둔다.
        """
        self._closed = True
        for prov in self._providers.values():
            for channel in prov.channels.values():      # 웹서치 켠 것/끈 것 둘 다
                closer = getattr(channel, "aclose", None)
                if closer is not None:
                    await closer()
            prov.channels.clear()

    def close(self) -> None:
        """동기 쪽에서 부르는 정리. 채널을 닫고 이 스레드의 루프까지 닫는다."""
        _run(self.aclose())
        _close_loop()

    #------------------------------------------------┌> 내부

    def _provider(self, provider: str | None) -> _Provider:
        name = provider or self.default
        prov = self._providers.get(name)
        if prov is None:
            raise KeyError(
                f"모르는 provider: {name!r} (있는 것: {', '.join(self._providers)})")
        return prov

    def _channel(self, prov: _Provider, web_search: bool = False):
        """채널을 처음 쓸 때 만들어 붙들고 있는다. 웹서치 여부로 따로 만든다.

        생성자에서 다 만들지 않는 이유: 넷을 등록해도 실제로 쓰는 건 한둘일 때가 많고,
        채널을 만들면 HTTP 클라이언트와 연결 풀이 생긴다.

        웹서치는 RestChannel 의 생성자 인자라 호출마다 못 바꾼다. 그래서 켠 채널과 끈
        채널을 둘 다 들고 필요한 쪽을 준다. 하나만 붙들면 먼저 만들어진 쪽이 계속
        쓰여서, 구조화 출력이 필요한 호출까지 웹서치 채널로 나간다(그러면 모듈이
        response_format 을 경고만 남기고 버린다).

        로컬은 웹서치를 지원하지 않으므로 플래그와 무관하게 하나만 쓴다.
        """
        # close() 뒤에는 채널을 다시 만들어도 못 쓴다. ai-rag-comm 의 _client_cache 가
        # 모듈 전역이라 닫힌 클라이언트를 그대로 돌려주고, 그때 나는 에러가
        # APIConnectionError("Connection error") 라서 망 문제로 보인다(실측).
        # 여기서 미리 막아 원인을 알려준다.
        if self._closed:
            raise RuntimeError(
                "이미 close() 한 LlmService 입니다. close() 는 종료용이라 되돌릴 수 "
                "없습니다 — 다시 쓰려면 LlmService 를 새로 만드세요."
            )
        if prov.base_url:
            key = False                     # 로컬은 웹서치가 없어 하나로 충분하다
            if key not in prov.channels:
                from ai_rag_comm import LocalLLMChannel

                prov.channels[key] = LocalLLMChannel(
                    prov.base_url, prov.model, prov.headers, prov.timeout)
                logger.info("[%s] LocalLLMChannel @ %s", prov.name, prov.base_url)
            return prov.channels[key]

        key = bool(web_search)
        if key not in prov.channels:
            from ai_rag_comm import AIProvider, RestChannel

            prov.channels[key] = RestChannel(
                prov.llm_api_config, AIProvider(prov.name), prov.model,
                enable_web_search=key)
            logger.info("[%s] RestChannel / %s (웹서치 %s)",
                        prov.name, prov.model, "켬" if key else "끔")
        return prov.channels[key]


#------------------------------------------------┌> 모듈 내부

# 루프를 스레드마다 하나씩 둔다.
#   - asyncio.run 을 호출마다 쓰면 루프가 매번 닫히고, 채널의 HTTP 클라이언트가 그
#     루프에 묶여 있어 연결이 정리되지 못한다(실측: 호출 6회에 unclosed socket 6개).
#   - 그래서 루프를 재사용하는데, 전역 하나로 두면 스레드 두 개가 같은 루프를 밀어넣어
#     "This event loop is already running" 이 난다(실측: 스레드 3개 중 2개 실패).
#     FastAPI 가 동기 엔드포인트를 스레드풀에 던지는 경우가 그렇다.
_local = threading.local()


def _run(coro):
    """코루틴을 동기로 돌린다. 이미 루프 안이면 무엇을 해야 하는지 알려주고 멈춘다."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        coro.close()    # 안 닫으면 "coroutine was never awaited" 경고가 따라온다
        raise RuntimeError(
            "이미 async 안에서 동기 메서드를 불렀습니다. a- 접두사 쪽을 await 하세요 "
            "(answer -> aanswer, extract_vocab -> aextract_vocab)."
        )

    loop = getattr(_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = _local.loop = asyncio.new_event_loop()
    return loop.run_until_complete(coro)


def _close_loop() -> None:
    """이 스레드의 루프를 닫는다. 채널을 다 닫은 뒤에만 부른다.

    안 닫으면 종료 때 "unclosed event loop" 가 뜬다(실측).

    shutdown_asyncgens 는 asyncio.run 이 하는 것과 같다 — HTTP 스트리밍처럼 아직
    안 끝난 async 제너레이터가 남아 있으면 루프가 닫힌 뒤에 정리되려다 터진다.
    """
    loop = getattr(_local, "loop", None)
    if loop is None or loop.is_closed():
        return
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()
    _local.loop = None


def _bare_schema(schema: type[BaseModel]) -> dict:
    """pydantic 모델을 '알맹이' JSON Schema 로 바꾼다.

    provider 별 봉투는 ai-rag-comm 이 씌운다. 우리가 OpenAI 봉투를 미리 씌우면 이중으로
    감싸져 안쪽 스키마의 type 이 'json_schema' 가 되고 400 이 난다 — 에러 메시지가
    provider 마다 달라서(Grammar error / Invalid schema for response_format /
    unrecognized type at top-level) 원인이 같다는 걸 알아채기 어렵다.

    strict 모드(ai-rag-comm 기본값)는 중첩된 모든 object 에 additionalProperties: false
    와 '모든 속성이 required' 를 요구한다. pydantic 이 $defs 에 만드는 중첩 객체까지 훑는다.
    """
    json_schema = schema.model_json_schema()
    _strictify(json_schema)
    return json_schema


def _strictify(node) -> None:
    """스키마 트리를 돌며 object 마다 strict 요구사항을 채운다. 제자리에서 고친다."""
    if isinstance(node, list):
        for item in node:
            _strictify(item)
        return
    if not isinstance(node, dict):
        return

    if node.get("type") == "object":
        node["additionalProperties"] = False
        node["required"] = list(node.get("properties", {}))

    for value in node.values():
        _strictify(value)


def _format_contexts(contexts: list, max_chars: int | None = None) -> tuple[str, int]:
    """검색 결과를 번호와 출처가 붙은 블록으로 만든다. (블록 문자열, 담은 개수).

    담은 개수를 같이 돌려주는 이유: 예산에 걸려 잘리면 받은 개수와 보낸 개수가
    달라진다. 로그에 받은 개수만 찍으면 5개를 보낸 것처럼 보인다.

    출처(breadcrumb)를 붙이는 이유가 두 개다. 어느 맥락을 근거로 답했는지 확인할 수
    있어야 하고, 한 섹션 안에 비슷한 항목이 여러 개일 때(세부과제 2-1 과 2-2 가 같은
    3,011자 섹션 안에 있다) 모델이 구분할 단서가 된다.

    RetrievedContext 를 받지만 타입을 보지 않는다 — content/breadcrumb 만 쓴다.
    문자열 목록을 넘겨도 돈다.

    max_chars 를 주면 위(리랭크 상위)에서부터 담다가 넘칠 맥락에서 멈춘다. 개수로
    자르지 않는 이유는 부모 크기 편차가 크기 때문이다 — 실측으로 25자에서 5,987자다.
    3개로 제한해도 최악이면 17,961자라 컨텍스트가 좁은 모델을 넘긴다.

    맨 위 하나는 예산을 넘겨도 담는다. 맥락 없이 답하면 모델이 아는 대로 지어내는데,
    차라리 요청이 400/413 으로 실패해서 원인이 보이는 편이 낫다.
    """
    blocks: list[str] = []
    total = 0
    for i, context in enumerate(contexts, 1):
        if isinstance(context, str):
            block = f"[{i}]\n{context}"
        else:
            source = (getattr(context, "breadcrumb", "")
                      or getattr(context, "heading", "") or "")
            head = f"[{i}] 출처: {source}" if source else f"[{i}]"
            block = f"{head}\n{getattr(context, 'content', '')}"

        if max_chars is not None and blocks and total + len(block) > max_chars:
            logger.info("맥락 자름: %d개 중 %d개만 (%d자, 상한 %d자)",
                        len(contexts), len(blocks), total, max_chars)
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks), len(blocks)


def _format_external(refs: list | None) -> str:
    """외부 데이터 목록을 프롬프트 끝에 붙일 절로 만든다. 없으면 빈 문자열.

    빈 문자열을 돌려주는 게 중요하다. 제목만 있고 내용이 없는 '## 참고 가능한 외부
    데이터' 가 남으면 모델이 그 빈칸을 설명하려 든다.

    dict 목록을 받는다(search_api_data_vector 의 결과 그대로). 쓰는 키는 title 과
    source 뿐이다 — url 은 넣지 않는다. 그 API 를 실제로 부르는 건 우리 쪽 일이고,
    모델이 링크를 그대로 답변에 옮기면 사용자가 인증 없이 눌러 실패한다.
    key 와 data 는 애초에 프로시저가 돌려주지 않는다(API 키가 새면 안 된다).
    """
    if not refs:
        return ""

    lines = []
    for ref in refs:
        if isinstance(ref, str):
            lines.append(f"- {ref}")
            continue
        title = (ref.get("title") or "").strip()
        if not title:
            continue
        source = (ref.get("source") or "").strip()
        lines.append(f"- {title} ({source})" if source else f"- {title}")

    if not lines:
        return ""
    return "\n\n## 참고 가능한 외부 데이터\n" + "\n".join(lines)


def _extract(schema: type[BaseModel], text: str) -> BaseModel | None:
    """평문에서 JSON 을 떼어내 검증한다.

    response_format 을 쓰면 형식이 보장되지만, 못 쓰는 엔드포인트에서는 모델이 코드펜스나
    앞뒤 설명을 붙인다. 그대로 파싱하면 실패하므로 떼어낸다.
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
