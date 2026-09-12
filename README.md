# ragmodul

hwpx 문서를 파싱·청킹·임베딩하고, 질의로 검색해 LLM에 넘길 맥락을 만들고,
그 맥락으로 답변을 만드는 파이썬 패키지.

메서드 하나가 파이프라인 한 단계다. 이 모듈은 여러 단계를 스스로 엮지 않는다 —
순서·재시도·단계 간 데이터 전달은 부르는 쪽(RagSystem)의 책임이고, 실패는 예외로
올라온다.

---

## 설치

```bash
pip install git+https://github.com/WsuRagDocumentAssistant/RagModuls.git
```

LLM 기능까지 쓰려면 `[llm]` extra 를 붙인다. 검색만 쓸 거면 안 깔아도 된다.

```bash
pip install "ragmodul[llm] @ git+https://github.com/WsuRagDocumentAssistant/RagModuls.git"
```

모델은 받아오지 않는다. 로컬에 미리 내려둔 폴더 경로를 넘겨야 한다.

| 모델 | 용도 |
|---|---|
| `BAAI/bge-m3` | 임베딩 (dense 1024차원 + sparse) |
| `BAAI/bge-reranker-v2-m3` | 리랭킹 |

DB 는 이 저장소가 관리하지 않는다. 운영 DB 와 저장 프로시저는 RagSystem 쪽
(`db_call`)이 담당하고, 여기 있는 `sql/schema.sql` 과 `DbService` 는 이 저장소를
혼자 돌려볼 때 쓰는 테스트용이다.

---

## 사용법

설정은 전부 인자로 받는다. 환경변수나 `.env` 를 읽지 않는다 — 설정을 어디서
가져올지는 이 모듈을 쓰는 애플리케이션이 정할 일이다.

```python
from ai_rag_comm import load_config
from ragmodul import RagController

cfg = load_config()                       # config.json + .env (cwd 기준)
rag = RagController(
    embedding_model_path="models/bge-m3",
    reranker_model_path="models/bge-reranker-v2-m3",
    device="cuda",                        # None 이면 자동 감지
    image_dir="images",                   # 주면 문서 이미지를 빼낸다. None 이면 안 함
    use_db=False,                         # 운영은 DB 를 RagSystem 이 직접 부른다
    llm_api_config=cfg.llm_api,           # gpt / claude / gemini
    local_llm_config=cfg.local_llm,       # local_llm
    llm_default="local_llm",
)
```

생성자가 임베딩 모델과 리랭커를 올리므로 몇십 초 걸린다. 프로세스당 한 번만 만든다.

### 문서 등록

```python
parsed   = rag.parse_document("보고서.hwpx")
document = rag.chunk_parent_child(parsed)
rag.embed_bge_m3(document)                # dense + sparse. forward 한 번
payload  = document_to_payload(document)  # ragmodul.util — 저장 프로시저에 넘길 dict
```

파싱·청킹만 필요하면 `RagController` 없이 `ragmodul.parse()` / `ragmodul.chunk()` 를
직접 부른다. 모델도 DB 도 안 올린다.

`parse()` 는 끝나면서 압축을 푼 자리(`unpack_dir/<문서명>/`)를 지운다. 파서가 푼
폴더를 알고 있으니 이 문서 것만 정확히 지우고, 부르는 쪽이 `unpack_dir` 을 통째로
비울 필요가 없다 — 통째로 비우면 같은 순간에 다른 워커가 파싱 중인 산출물까지
지운다. 파싱이 실패하면 남긴다. 풀린 파일을 보려면 `cleanup=False`.

### 질의 검색

```python
query = "교원 확보율은 얼마인가?"
qvec, qweights = rag.embed_query(query, vocab)     # vocab 은 load_vocab() 결과. None 이면 확장 안 함
hits     = db_call("search_documents_hybrid", ...)  # 조각. 넉넉히(40개) 뽑는다
contexts = rag.build_contexts(hits)                 # LLM 입력 단위로 묶기
contexts = rag.rerank(query, contexts, top_k=5)     # 최종 순서
```

`hits` 는 `parent_id / child_content / content / breadcrumb / similarity /
parent_child_count` 를 가진 dict 목록이다. `parent_child_count` 가 승격 비율의
분모다 — 없으면 걸린 조각 수로 대체되어 전부 승격된다.

`build_contexts` 에 `limit` 을 걸지 말 것. 약한 신호(유사도)로 미리 자른 뒤 강한
신호(리랭커)에게 남은 것만 주면 정답이 잘린다 — 실측으로 recall 이 93%에서 83%로
떨어졌다. 후보를 다 넘기고 리랭커가 `top_k` 로 줄이게 한다.

`rerank` 는 부모당 최대 2개(`max_per_parent`)만 담고, 점수 0.01 미만
(`min_score`)은 버린다. 문서와 무관한 질의면 빈 목록이 나오므로 부르는 쪽이 맥락
없이 답하는 경우를 처리해야 한다.

### 답변 생성

로컬 모델이 초안을 만들고, 사용자가 고른 클라우드 모델이 각자 다듬는다. 병합은
요청할 때만 한다.

```python
draft   = rag.answer(query, contexts, provider="local_llm",
                     history=recent, summary=summary)
answers = rag.refine_all(query, contexts, draft, ["gpt", "claude"],
                         external=refs, history=recent, summary=summary)
# {"gpt": "...", "claude": "..."}  — 실패한 provider 는 빠진다
merged  = rag.merge(query, list(answers.values()), provider="claude")
```

- `answer()` / `refine()` 은 맥락마다 번호와 출처(breadcrumb)를 붙여 넘긴다. 어느
  맥락을 근거로 답했는지 확인할 수 있어야 하고, 한 섹션 안에 비슷한 항목이 여러 개일
  때(세부과제 2-1 과 2-2 가 같은 3,011자 섹션) 구분에도 쓰인다.
- `refine_all` 은 같은 초안을 각 모델에 따로 준다. 순차로 넘기면 앞 모델의 판단이
  뒤로 갈수록 굳어진다. 기본은 동시 호출이고 429 가 나면 `parallel=False`.
- `web_search` 기본이 켜짐이다. 로컬은 지원하지 않아 무시된다.
- `external` 은 유사도로 찾은 외부 API 목록이다. 맥락과 섞지 않고 별도 절로 내려간다
  — 제목뿐이라 근거가 못 되는데 Context 에 끼면 모델이 사실처럼 인용한다.
- `history` 는 `{user_query, ai_response}` 목록이다. 글자 상한(30,000자)을 넘으면
  오래된 차례부터 버린다. 로컬은 맥락 예산(45,000자)에서 이력만큼 빼서 둘의 합이
  상한을 넘지 않게 한다.

### 세션 요약

이력에서 밀려난 차례를 요약에 눌러 담는다. 누적이라 '기존 요약 + 새로 밀려난
차례' 만 넣는다.

```python
summary, topic = rag.summarize_session(summary, dropped_turns)
```

밀려난 차례가 없으면 LLM 을 부르지 않고 기존 요약을 그대로 돌려준다. 주제가 빈
문자열이면 쓰던 주제를 유지한다.

### 이미지

`image_dir` 을 주면 **본문 그림 블록(`block.figure.image`)** 의 이미지만
`image_dir/<문서명>/` 으로 빼내고 `rag.document_images(parsed)` 로 목록을 받는다.
문서 순서·제목 경로·캡션이 들어 있어서 폴더를 훑지 않아도 된다(폴더 훑기는 지난번
잔재까지 등록한다).

hwpx 는 `hp:pic` 이 최상위 문단 바로 아래 있을 때만 그림 블록으로 올린다. 표 셀
안·묶음 개체 안·도형 안의 그림과 배경 채우기 이미지는 블록이 안 되고, 여기서도
빼내지 않는다. 실측으로 그 비중이 크다 — 2주기 보고서는 manifest 81개 중 그림
블록 13개, 표 셀 안 60개(중앙값 750KB, 아이콘이 아니다). 표 셀 안 그림은
`block.table.cells[].images` 로 위치까지 잡히므로 필요해지면 그 경로를 더하면 된다.

이미지는 본문 색인에 들어가지 않는다. 대신 색인 때 그림마다 설명을 만들어 따로
임베딩한다.

```python
descs = rag.describe_images_all([(data, "image/png"), ...], provider="gemini")
for desc in descs:
    if not (desc.ai_summary or desc.key_facts or desc.key_phrases):
        continue                          # 로고·장식은 건너뛴다
    text = " ".join([desc.ai_summary, *desc.key_facts, *desc.key_phrases])
    vector = rag.embed_texts([text])[0]
```

질의 때는 `is_image_query(query)` 로 그림을 찾는 말인지 먼저 가른다. "그림·사진·
도표·그래프·차트·이미지·표 를 직접 찾는 말" 일 때만 True 다. 내용을 묻는 말은
그림이 도움이 될 것 같아도 False 다. 검색과 `asyncio.gather` 로 함께 던지면 지연이
없다.

`vectorize_image` 는 그림을 SVG 로 다시 그린다. 사용자가 버튼을 눌러 기다리는
결과라 실패하면 예외를 올린다(잘린 SVG 도 예외).

provider 마다 받는 형식이 다르다(실측). gpt·claude 는 bmp 를 400 으로 거절하고
gemini 만 읽는다. hwpx 문서 그림은 절반쯤이 bmp 라(243장 중 117장) 이 용도의
provider 는 사실상 gemini 다.

### 축약어 사전

```python
pairs = rag.extract_vocab(whole_text, provider="gpt")          # 통째로 한 번
more  = rag.recheck_vocab(whole_text, pairs, provider="gpt")   # 빠뜨린 것 (선택)
kept, dropped = rag.filter_vocab(pairs + more)
vocab = rag.load_vocab()                                        # {축약어: [확장어]}
```

문서를 통째로 넘긴다. 부모 단위로 32번 나눠 부르면 17개인데, 통째 1번 + 재검토
1번이면 18개다 — 쪼개면 문서 앞뒤에 흩어진 '축약어 … 풀어쓴 말' 을 못 잇는다.

로컬처럼 한 요청이 길어지면 안 되는 쪽은 `pack_texts` 로 부모 경계에서만 끊어
묶고 `extract_vocab_all` 로 돌린다. 조각 하나가 실패해도 나머지는 살린다.

사전은 질의 쪽에 붙인다(`embed_query(query, vocab)`). 색인 쪽에 확장어를 박으면
그 말이 실제보다 흔해 보여 sparse 가 변별력을 잃는다. 치환이 아니라 덧붙이기다 —
원래 표기로 물어본 사람을 놓치지 않는다. 실측(8건) dense 개선 5 / 악화 0, sparse
개선 3 / 악화 0.

---

## LLM 호출

[ai-rag-comm](https://github.com/WsuRagDocumentAssistant/ServerCommunication) 의
채널로 나간다. `LlmService` 하나가 provider 넷(gpt / claude / gemini / local_llm)을
들고 있고 호출할 때 고른다. 설정은 `load_config()` 결과를 그대로 받는다 — 모델명·
엔드포인트·키·타임아웃이 이미 거기 다 있어서 여기서 표를 또 만들면 두 곳이 어긋난다.

기능마다 async 본체(`aanswer`)와 동기 껍데기(`answer`)가 짝으로 있다. 이미 이벤트
루프 안이면 a- 접두사 쪽을 await 한다. 동기 쪽은 스레드마다 루프를 하나씩 둔다 —
FastAPI 가 동기 엔드포인트를 스레드풀에 던져도 겹치지 않는다.

**호출 하나 = 텍스트 하나.** 목록을 돌리는 건 `_all` 메서드가 하고, 하나가 죽어도
나머지는 돌려준다.

### 프롬프트

`ragmodul/prompt/prompt.py` 에 키워드로 등록해 두고 꺼내 쓴다. 기능마다 두 개다 —
지시는 system, 데이터는 user.

```python
from ragmodul.prompt import get_prompt
system, user = get_prompt("answer", context=..., query=..., external="", history="", summary="")
```

지시를 system 으로 분리하면 모델이 그걸 '따를 규칙' 으로 다루고 데이터와 섞이지
않는다. 데이터가 모자라면 예외다.

### 알아둘 제약

- **출력 상한은 65,536 토큰.** 넷 다 받는 최대값이다. 8,192 로는 실제로 잘렸다 —
  추론 토큰을 먼저 쓰는 모델(gemini)이 SVG 변환에서 계획하는 글만 남기고 끊겼다.
- **gpt 는 `temperature` 를 거부한다.** 그래서 gpt 에는 안 보내고 나머지는 0 을
  보낸다. 모델을 바꾸면 `no_temperature` 로 조정한다.
- **구조화 출력(`response_format`)과 웹서치는 같이 못 쓴다.** 모델 제약이 아니라
  ai-rag-comm 이 gpt 웹서치 경로(Responses API)에서 `response_format` 을 버리기
  때문이다. 구조화 출력을 쓰는 작업(사전 추출·세션 요약·이미지 설명·질의 판정)은
  전부 바깥을 볼 이유가 없어서 일괄로 웹서치를 끈다.
- **`answer()` 는 구조화 출력을 쓰지 않는다.** 같은 이유다. 서식 때문은 아니다 —
  재보니 JSON 필드 하나로 받아도 마크다운 표와 서식이 유지됐다. ai-rag-comm 이
  text.format 변환을 구현하면 sessionId·sources 를 필드로 받을 수 있다.
- **로컬 맥락 상한 45,000자.** 리랭커가 고른 top_k 가 어떤 조합이어도 안 잘리는
  값이다(부모 5개 최악 27,296자). 더 올리지 않는 이유는 프리필 시간이다 — 글자당
  약 0.1ms 라 80,000자면 8초가 사용자 대기에 더해진다.

---

## 데이터 구조

### 청킹: parent / child

```
ParentChunk           목차 단위 덩어리. LLM 맥락용. 임베딩하지 않는다.
└─ ChildChunk         500자 단위 조각. 검색용. 이것만 임베딩한다.
```

parent를 임베딩하지 않는 이유: 같은 내용을 두 번 색인하면 검색에서 서로 경쟁하고,
큰 parent는 임베딩 길이 상한에 걸려 뒷부분이 잘린다.

child에는 목차 경로(breadcrumb)가 앞에 붙는다. 조각만 봐도 어느 섹션인지 알아야
검색 결과가 쓸모있다.

표는 만나면 끊고, 500자를 넘으면 잘라서 **각 조각에 머리글을 다시 붙인다.**
안 붙이면 머리글 없는 행만 남아 읽을 수 없다.

### 검색: RetrievedContext

검색은 조각 단위지만 LLM에 넣는 건 섹션 본문이다. 조각을 그대로 돌려주면 같은
섹션에서 여러 조각이 걸릴 때 본문이 여러 번 실려간다(실측: 조각 10개가 섹션 2곳,
본문의 56%가 중복).

그래서 **비율로 판단한다**:

```
걸린 조각 수 / 그 부모의 전체 조각 수 > 0.5   →  섹션 본문으로 승격 (merged=True)
                                      ≤ 0.5   →  조각 그대로 (merged=False)
```

16개 중 2개만 걸린 섹션에 5000자를 붙이는 건 과하므로 조각만 준다.
[LlamaIndex AutoMergingRetriever](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/retrievers/auto_merging_retriever.py)와 같은 방식이다.

`score`는 **걸린 조각 중 최고점**이다. 평균이 아니다 — 평균으로 했다가 recall이
93%에서 83%로 떨어졌다. 승격되는 섹션은 조각이 많이 걸린 섹션인데, 많이 걸릴수록
평균에 딸려오는 낮은 형제도 많아져 점수가 내려간다. 관련성이 높을수록 벌을 받는
구조였다.

`rerank_text`는 최고점 조각이다. 승격된 섹션 본문(수천 자)을 리랭커에 넣으면
512 토큰에서 잘려 앞부분만 보고 판정한다.

---

## 하이브리드 검색

`child_chunk`에 벡터가 둘 있다.

| 컬럼 | 담긴 것 | 잡는 것 |
|---|---|---|
| `embedding VECTOR(1024)` | dense | 뜻이 가까우면 |
| `lexical SPARSEVEC(250002)` | sparse (BGE-M3 lexical weight) | 같은 단어가 실제로 나왔으면 |

두 검색 결과를 **RRF**로 합친다. 점수를 더하지 않고 순위로 합치는 이유: dense는
코사인이라 0~1인데 sparse는 가중치 내적이라 상한이 없다. 그냥 더하면 sparse가
결과를 지배한다.

```
점수 = 1/(60 + dense순위) + 1/(60 + sparse순위)
```

한쪽에서만 걸린 조각은 그 항이 0이 된다. 역수를 쓰면 "없음"에 넣을 값이 0으로
자연히 정해지고, 사람이 벌점을 정할 필요가 없다.

sparse 차원은 하드코딩하지 않고 모델에게 묻는다(`rag.sparse_dimension`
= 토크나이저 vocab 크기). DB 의 `SPARSEVEC(N)` 과 어긋나면 저장에서 실패한다.

---

## 측정 결과

문서 1개(우송대 성과평가보고서), 질문 30문항. 정답 섹션이 상위 5개 안에 있으면
맞춘 것으로 본다.

| 설정 | Recall@5 | MRR |
|---|---|---|
| dense 단독, 조각 5개 | 93% | 0.892 |
| + sparse (RRF) | 93% | 0.883 |
| + 리랭크 | 90% | **0.900** |
| + 맥락 조립 (현재 기본) | 90% | **0.900** |

단계별 차이:

- **sparse 추가**: 측정 가능한 이득 없음. dense가 이미 약어·한영 교차 질의를 잡는다
- **리랭크 추가**: 수치조회 1문항을 잃고 순위 품질(MRR)을 얻는다
- **맥락 조립**: recall·MRR 그대로, **LLM 입력 14,841자 → 3,401자 (77% 절감)**

**해석에 주의.** 문서 1개, 문항 30개다. 1문항이 recall 3%p, MRR 0.03을 움직이므로
위의 3%p 차이는 잡음과 구분되지 않는다. 확실한 건 맥락 조립이 recall을 깎지 않고
문맥을 크게 줄인다는 것뿐이다.

---

## 알려진 한계

**복잡한 표를 일부러 비운다.** hwpx 파서가 구조를 신뢰할 수 없는 표(이 문서에서
14개)를 `제외:OCR`로 표시하고 내용을 비운다. `parse()` 가 필터 전 원본 셀로
되살리므로(`recover_excluded=True`) 수치는 색인에 들어가지만, 표 구조는 믿을 수 없다.

**승격 비율 `0.5`는 검증되지 않았다.** LlamaIndex 기본값이다. 문서 1개로 튜닝하면
그 문서에 맞추는 것밖에 안 된다. 비율의 분모는 부모당 조각 수이고 그건 청킹 설정
(`MAX_PARENT_CHARS`, 표 분할 규칙)이 정하므로, 지금 맞춘 값은 문서와 설정 양쪽에
묶인다. `merge_ratio` 인자로 받으므로 문서가 늘면 코드 없이 바꿀 수 있다.

**이미지 설명은 실패와 '설명할 것 없음' 이 구분되지 않는다.** `describe_images_all`
은 둘 다 빈 `ImageDescription` 을 돌려준다. 둘 다 임베딩을 건너뛰면 되는 경우라
같게 뒀다. 구분이 필요하면 로그를 본다.

**`summarize_session` 의 주제는 압축한 구간 기준이다.** 이 메서드가 보는 건
밀려난 차례들뿐이라 화제가 막 바뀐 직후에는 이전 주제가 남는다.

**청커 테스트 2개가 실패한다.** 500자 한도 초과 조각 39개(기대 2개 이하), 표 파편
117개(기대 0개). 표 행이 실제로 잘리는 건지 단정이 잘못된 건지 미확인.

**로컬 DB 는 테스트용이다.** `DbService` 와 `sql/schema.sql` 은 이 저장소를 혼자
돌려볼 때만 쓴다. 운영 DB 는 RagSystem 의 저장 프로시저가 담당하고, 이 저장소의
SQL 은 그쪽 스키마와 같다는 보장이 없다.

---

## 테스트

```bash
pytest tests -m "not model"    # 모델 없이 도는 것만 (빠름)
pytest tests -m "model"        # 임베딩·리랭커 로드 (느림, GPU)
pytest tests/test_search_model.py   # DB·모델 없이 검색 조립 로직만
```

`test_search_model.py`는 입력이 dict 목록뿐이라 단독으로 0.2초에 돈다.

`main.py` 는 눈으로 결과를 확인하는 수동 실행용이다. `python main.py parse chunk`
는 모델 없이 돌고, `python main.py query answer` 는 색인이 있어야 한다.

---

## 구조

```
ragmodul/
├─ controller.py              단계별 메서드. 서비스 생명주기를 스스로 관리한다
├─ util.py                    순수 계산 — 사전 선별, 질의 확장, 글 묶기, 저장 페이로드 변환
├─ models/
│  ├─ chunk_model.py          ChunkedDocument / ParentChunk / ChildChunk
│  ├─ search_model.py         RetrievedContext / RetrievedChild (승격 로직)
│  ├─ image_model.py          DocumentImage / ImageDescription / ImageQuery / SvgImage
│  ├─ session_model.py        SessionSummary
│  └─ vocab_model.py          VocabPair / VocabPairs / QueryTerms
├─ prompt/
│  └─ prompt.py               프롬프트 템플릿 + get_prompt(키, 데이터)
└─ service/
   ├─ parser_service.py       hwpx 파싱 + 제외 표 복구 + 이미지 추출 (parse 함수)
   ├─ chunker_service.py      목차 기준 parent/child 분할 (chunk 함수)
   ├─ embedded_service.py     BGE-M3 dense + sparse (forward 1회)
   ├─ reranker_service.py     CrossEncoder 리랭킹, 부모당 제한, 최소 점수
   ├─ llm_service.py          ai-rag-comm 채널. 답변·다듬기·병합·요약·사전·이미지
   ├─ db_service.py           테스트용 직접 SQL. 저장 / RRF 하이브리드 검색
   ├─ db_manager_service.py   저장 프로시저 어댑터 (db_backend="manager")
   └─ ocr_service.py          미구현
```

`vocab_model.py`, `image_model.py`(뒤 셋), `session_model.py` 는 dataclass 가 아니라
pydantic 이다. LLM 출력의 검증 스키마이면서 `response_format` 에 붙일 JSON Schema 의
출처라 dataclass 로는 둘 다 안 된다.

상태도 IO도 없는 작은 작업(`parse`, `chunk`, `util`)은 클래스로 감싸지 않고 함수로
둔다.
