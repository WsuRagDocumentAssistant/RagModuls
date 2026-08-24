# ServerCommunication (`ai-rag-comm`) 수정 요청

`ragmodul` 에서 이 모듈로 LLM 을 부르려다 막힌 지점들. 전부 실제로 돌려서 확인했다.

현재 `ragmodul/service/llm_service.py` 는 이 모듈을 **거치지 않고** `openai` SDK 를
직접 부른다. 아래 1~3 이 해결되면 갈아탈 수 있다.

- 확인 환경: `ai-rag-comm 0.1.0` (commit `bca01b5`), Python 3.11, Windows
- 설치: `pip install git+https://github.com/WsuRagDocumentAssistant/ServerCommunication.git`

---

## 1. `base_url` / `default_headers` 를 넘길 수 없다 — 로컬 LLM 을 못 부른다

**막히는 곳** `services/llm_api/openai_service.py`

```python
def __init__(self, api_key: str, default_model: str):
    from openai import AsyncOpenAI
    self._client = AsyncOpenAI(api_key=api_key)     # base_url, default_headers 없음
```

**왜 필요한가**

사내 로컬 LLM 이 OpenAI 호환 HTTP 엔드포인트다.

```
POST http://117.16.166.22/v1/chat/completions
headers: {"x-user-id": "npark-01"}
model:   gemma-4-12B-it
```

`AsyncOpenAI` 는 `base_url` 과 `default_headers` 를 이미 지원하는데 감싸는 쪽에서
안 넘긴다. 그래서 이 모듈로는 `api.openai.com` 밖에 못 부른다.

`SocketChannel` 은 대안이 아니다. TCP 소켓에 JSON 한 줄을 보내고 `<|END|>` 까지
읽는 프로토콜이라 HTTP 엔드포인트와 맞지 않는다.

**수정**

```python
def __init__(self, api_key: str, default_model: str,
             base_url: str | None = None,
             default_headers: dict | None = None):
    from openai import AsyncOpenAI
    self._client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,                  # None 이면 OpenAI 기본값
        default_headers=default_headers,
    )
    self._default_model = default_model
```

`chat()` / `stream_chat()` 은 안 건드려도 된다. `AsyncOpenAI` 가 알아서 붙인다.

---

## 2. `RestChannel` 이 추가 인자를 전달하지 않는다

**막히는 곳** `services/channels/rest_channel.py`

```python
client = entry["client_cls"](api_key=resolved_api_key, default_model=resolved_model)
```

1번을 고쳐도 채널을 거치면 전달이 안 된다. `PROVIDER_REGISTRY` 항목에 엔드포인트
정보를 담고 이 줄이 넘기게 해야 한다.

**수정 예**

```python
PROVIDER_REGISTRY = {
    AIProvider.GPT: {
        "client_cls": OpenAIService, "api_key_field": "openai_api_key", "model_key": "gpt",
    },
}

def _build_client(config, provider, model, api_key, base_url=None, headers=None):
    ...
    client = entry["client_cls"](
        api_key=resolved_api_key, default_model=resolved_model,
        base_url=base_url, default_headers=headers,
    )
```

`RestChannel.__init__` 도 그 둘을 받아 넘기면 된다.

---

## 3. `temperature` 를 넘길 수 없다 — 세 층에서 다 떨어진다

**1층** `services/channels/rest_channel.py` — payload 에서 세 키만 꺼낸다

```python
prompt = payload["prompt"]
model = payload.get("model")
max_tokens = payload.get("max_tokens", 1024)
```

payload 에 `temperature` 를 넣어도 읽지 않는다.

**2층** `interface/base_llm_api_interface.py` — 시그니처에 자리가 없다

```python
async def chat(self, prompt: str, model: Optional[str], max_tokens: int) -> ChatResponse: ...
```

**3층** `services/llm_api/openai_service.py` — API 에 보낼 때도 안 넣는다

```python
await self._client.chat.completions.create(
    model=_model, max_completion_tokens=max_tokens,
    messages=[...],                                  # temperature 없음
)
```

**왜 필요한가**

축약어 추출처럼 같은 입력에 같은 결과가 나와야 하는 작업이 있다. 고정하지 못하면
실행마다 결과가 달라져 "내가 고친 것" 과 "모델이 다르게 뽑은 것" 을 구분할 수 없다.

실측 — 같은 문서, 같은 프롬프트로 두 번 추출:

| 모델 | `temperature=0` | 결과 |
|---|---|---|
| `gpt-5.5` | **거부** (`does not support 0.0 with this model`) | 고유 축약어 17개 / 25개로 갈림 |
| `gemma-4-12B-it` (로컬) | 받음 | 45개 / 45개, **완전 일치** |
| `claude-sonnet-4-5` | 받음 | — |
| `gemini-2.5-flash` | 받음 | — |

`gpt-5.5` 하나만 거부한다. 나머지 셋에서는 값어치가 있다.

**수정**

```python
# base_llm_api_interface.py
async def chat(self, prompt, model, max_tokens,
               temperature: float | None = None) -> ChatResponse: ...

# openai_service.py — None 이면 아예 안 보낸다. 일부 모델이 인자 자체를 거부한다.
async def chat(self, prompt, model, max_tokens, temperature=None):
    kwargs = {} if temperature is None else {"temperature": temperature}
    response = await self._client.chat.completions.create(
        model=_model, max_completion_tokens=max_tokens, messages=[...], **kwargs)

# rest_channel.py
temperature = payload.get("temperature")
response = await self._client.chat(prompt, model, max_tokens, temperature)
```

---

## 4. `load_config()` 가 프로젝트 루트를 못 찾는다

**막히는 곳** `helpers/config_helper.py`

```python
_ROOT = Path(__file__).resolve().parents[3]
```

저장소에서 직접 실행할 때는 맞지만, `pip install` 로 설치하면 기준점이 다른 트리로
옮겨간다.

```
설치 후 경로: <venv>/Lib/site-packages/ai_rag_comm/helpers/config_helper.py
  parents[2] = site-packages
  parents[3] = Lib              ← 여기서 config.json 을 찾는다
```

실측:

```
config.json 위치        : C:\Users\user\Desktop\RAGModul\config.json      (있음)
load_config() 가 보는 곳 : C:\Users\user\Desktop\RAGModul\.venv\Lib\      (없음)
결과: FileNotFoundError - ...\.venv\Lib\config.json
```

`RagSystem` 에 `config.json` 을 만들어도 같은 이유로 못 읽는다. 숫자를 바꿔서는
해결되지 않는다 — 설치된 패키지 경로에 프로젝트 루트 정보가 없다.

그리고 `os.environ["DB_USER"]` / `["DB_PASSWORD"]` 를 무조건 읽어서, LLM 만 쓰려는
경우에도 없으면 `KeyError` 로 죽는다.

**수정**

```python
def load_config(root: Path | str | None = None) -> Config:
    root = Path(root or os.environ.get("APP_ROOT") or Path.cwd())
    load_dotenv(root / ".env")
    with open(root / "config.json", encoding="utf-8-sig") as f:
        raw = json.load(f)
    ...
    database=DatabaseConfig(
        ...
        user=os.environ.get("DB_USER", ""),          # 없어도 죽지 않게
        password=os.environ.get("DB_PASSWORD", ""),
    ),
```

기준을 실행하는 쪽이 정하게 하면 된다. `Path.cwd()` 기본값이면 `RagSystem` 에서
`python main.py` 로 띄울 때 바로 맞는다.

---

## 5. 구조화 출력(`response_format`)을 쓸 수 없다

`chat()` 이 `response_format` 을 안 받아서, 스키마를 강제할 수 없다. 지금은 프롬프트로
JSON 을 요구하고 평문에서 떼어내 검증한다 — 모델이 코드펜스나 앞뒤 설명을 붙이면
파싱이 실패한다.

`response_format` 을 쓰면 API 가 스키마를 보장한다. 넷 다(gpt / 로컬 / claude / gemini)
지원하는 것을 확인했다.

3번의 `temperature` 와 같은 방식으로 선택 인자를 하나 더 받으면 된다.

---

## 6. 버전을 정확히 고정한다 — 다른 패키지와 부딪힌다

`METADATA`:

```
Requires-Dist: pydantic==2.7.1
Requires-Dist: python-dotenv==1.0.1
Requires-Dist: openai==2.48.0
Requires-Dist: asyncpg==0.29.0
```

`==` 로 못 박혀 있어서 깨끗한 환경에 설치하면 pip 이 그 버전으로 **내려받는다.**
`ragmodul` 은 `sentence-transformers`, `transformers`, `pgvector` 등을 함께 쓰는데
`pydantic` 이 2.7.1 로 내려가면 부딪힐 수 있다.

참고로 이 저장소 개발 환경에는 `pydantic 2.13.4`, `openai 3.3.1` 이 깔려 있는데
**핀을 만족하지 않는데도 코드가 정상 동작한다** — 버전에 걸리는 API 를 안 쓴다는 뜻이다.

**수정** `>=` 로 푸는 것. 라이브러리는 하한만 정하고 상한은 애플리케이션이 정한다.

```
pydantic>=2.7
openai>=2.48
python-dotenv>=1.0
```

---

## 7. `asyncpg` 가 필수 의존성이다

LLM 호출만 쓰려는 경우에도 DB 드라이버가 깔린다. `[db]` extra 로 빼면 좋겠다.

```toml
[project.optional-dependencies]
db = ["asyncpg>=0.29"]
```

---

## 8. `local_llm` 설정이 소켓 전제다 — 지금 상황과 안 맞는다

모듈 docstring 과 `Transport` 주석이 "로컬 LLM 은 SOCKET 으로만 통신한다" 로 되어 있고,
`config.json` 의 `local_llm` 도 `host`/`port` 다.

```json
"local_llm": { "host": "10.101.96.71", "port": 8001, "timeout": 30.0 }
```

그런데 실제 로컬 LLM 은 HTTP(`http://117.16.166.22/v1/chat/completions`)로 바뀌었다.
`SocketChannel` / `Transport.SOCKET` / `LocalLLMConfig` 가 다 소켓 전제라 정리 대상이다.

1번(`base_url`)이 되면 로컬도 `RestChannel` 로 통일할 수 있다 — OpenAI 호환이라
새 채널 클래스가 필요 없다.

---

## 우선순위

| | 항목 | 없으면 |
|---|---|---|
| **높음** | 1. `base_url` / `default_headers` | 로컬 LLM 을 아예 못 부름 |
| **높음** | 2. `RestChannel` 전달 | 1번을 고쳐도 채널로는 못 씀 |
| 중간 | 4. `load_config()` 경로 | `config.json` 을 만들어도 안 읽힘 |
| 중간 | 5. `response_format` | 평문 파싱에 의존 |
| 중간 | 6. 버전 핀 | 깨끗한 설치에서 충돌 가능 |
| 낮음 | 3. `temperature` | `gpt-5.5` 는 어차피 거부. 다른 모델에서만 의미 |
| 낮음 | 7. `asyncpg` extra | 용량 문제 |
| 낮음 | 8. 소켓 경로 정리 | 지금은 안 쓰는 코드 |
