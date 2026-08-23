# ragmodul

hwpx 문서를 파싱·청킹·임베딩해서 PostgreSQL(pgvector)에 넣고, 질의로 검색해
LLM에 넘길 맥락을 만드는 파이썬 패키지.

메서드 하나가 파이프라인 한 단계다. 이 모듈은 여러 단계를 스스로 엮지 않는다 —
순서·재시도·단계 간 데이터 전달은 부르는 쪽의 책임이고, 실패는 예외로 올라온다.

---

## 설치

```bash
pip install git+https://github.com/WsuRagDocumentAssistant/RagModuls.git
```

모델은 받아오지 않는다. 로컬에 미리 내려둔 폴더 경로를 넘겨야 한다.

| 모델 | 용도 |
|---|---|
| `BAAI/bge-m3` | 임베딩 (dense 1024차원 + sparse) |
| `BAAI/bge-reranker-v2-m3` | 리랭킹 |

DB는 PostgreSQL 17 + pgvector 0.8.x. 테이블은 [`sql/schema.sql`](sql/schema.sql)로 만든다.

---

## 사용법

```python
from ragmodul import RagController

rag = RagController(
    embedding_model_path="models/bge-m3",
    reranker_model_path="models/bge-reranker-v2-m3",
    device="cuda",                    # None 이면 자동 감지
    image_dir="images",               # 주면 문서 이미지를 빼낸다. None 이면 안 함
)

# 문서 등록
parsed   = rag.parse_document("보고서.hwpx")
document = rag.chunk_parent_child(parsed)
rag.embed_bge_m3(document)            # dense + sparse 를 채운다
rag.save_to_vector_db(document)

# 질의 검색
query = "교원 확보율은 얼마인가?"
qvec, qweights = rag.embed_query(query)
hits     = rag.hybrid_search(qvec, qweights, top_k=40)   # 조각. 넉넉히 뽑는다
contexts = rag.build_contexts(hits)                      # LLM 입력 단위로 묶기
contexts = rag.rerank(query, contexts, top_k=5)          # 최종 순서

for c in contexts:
    print(c.breadcrumb, c.rerank_score)
    print(c.content)                  # 이걸 LLM 에 넣는다
```

설정은 전부 인자로 받는다. 환경변수나 `.env`를 읽지 않는다 — 설정을 어디서
가져올지는 이 모듈을 쓰는 애플리케이션이 정할 일이다.

`build_contexts`에 `limit`을 걸지 말 것. 약한 신호(유사도)로 미리 자른 뒤 강한
신호(리랭커)에게 남은 것만 주면 정답이 잘린다 — 실측으로 recall이 93%에서 83%로
떨어졌다. 후보를 다 넘기고 리랭커가 `top_k`로 줄이게 한다.

### 이미지

`image_dir`을 주면 문서에 들어있던 이미지를 `image_dir/<문서명>/`으로 빼낸다.
(이 문서는 81개, 115MB.) 안 주면 복사하지 않는다 — 이미지는 `unpack_dir` 안에도
풀려 있으므로 오래 둬야 하는 쪽만 켜면 된다.

문서마다 하위 폴더를 만든다. 이미지 `ref`가 문서 안에서만 유일해서(`image1`,
`image2`...) 문서 두 개를 처리하면 `image1.jpg`가 서로 덮어쓴다.

**이미지는 검색에 들어가지 않는다.** 파서는 `role='그림'` 블록 39개에 위치 정보
(`figure.image`, `heading_path_text`)까지 붙여주지만, 청킹이 `block.text`만 보고
그림 블록은 `text=None`이라 색인에서 빠진다. 넣으려면 캡션이나 OCR 텍스트가 필요하다
(`OcrService`가 그 자리다).

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

sparse 차원은 하드코딩하지 않고 모델에게 묻는다(`EmbeddedService.sparse_dimension`
= 토크나이저 vocab 크기). `sql/schema.sql`의 `SPARSEVEC(N)`과 어긋나면 저장에서
실패한다.

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
14개)를 `제외:OCR`로 표시하고 내용을 비운다. 그 안의 수치는 색인에 없으므로
어떤 검색으로도 찾을 수 없다. 30문항 중 최소 2문항(예산 집행률, 유학생 유치 수)이
여기 걸린다. 검색 실패로 오해하지 말 것.

**승격 비율 `0.5`는 검증되지 않았다.** LlamaIndex 기본값이다. 문서 1개로 튜닝하면
그 문서에 맞추는 것밖에 안 된다. 비율의 분모는 부모당 조각 수이고 그건 청킹 설정
(`MAX_PARENT_CHARS`, 표 분할 규칙)이 정하므로, 지금 맞춘 값은 문서와 설정 양쪽에
묶인다. `merge_ratio` 인자로 받으므로 문서가 늘면 코드 없이 바꿀 수 있다.

**임베딩 forward가 두 번 돈다.** `embedded` 라이브러리가 dense/sparse를 한 번에
주는 메서드를 열어두지 않아 `encode_documents`와 `encode_sparse`를 따로 부른다.
합치려면 그쪽에 `encode_all()`을 추가해야 한다.

**테스트가 운영 DB에 쓴다.** `DbService()`가 기본 접속 정보로 바로 연결하므로,
저장을 호출하는 테스트는 실제 색인을 덮어쓴다. 실제로 낡은 테스트 하나가 색인을
벡터 없는 상태로 날린 적이 있다. 테스트 전용 DB를 분리해야 한다.

**청커 테스트 2개가 실패한다.** 500자 한도 초과 조각 39개(기대 2개 이하), 표 파편
117개(기대 0개). 표 행이 실제로 잘리는 건지 단정이 잘못된 건지 미확인.

---

## 테스트

```bash
pytest tests -m "not model"    # 모델 없이 도는 것만 (빠름)
pytest tests -m "model"        # 임베딩·리랭커 로드 (느림, GPU)
pytest tests/test_search_model.py   # DB·모델 없이 검색 조립 로직만
```

`test_search_model.py`는 입력이 dict 목록뿐이라 단독으로 0.2초에 돈다.

---

## 구조

```
ragmodul/
├─ controller.py              단계별 메서드. 서비스 생명주기를 스스로 관리한다
├─ models/
│  ├─ chunk_model.py          ChunkedDocument / ParentChunk / ChildChunk
│  └─ search_model.py         RetrievedContext / RetrievedChild
└─ service/
   ├─ parser_service.py       hwpx 파싱 + 이미지 추출 (parse 함수)
   ├─ chunker_service.py      목차 기준 parent/child 분할 (chunk 함수)
   ├─ embedded_service.py     BGE-M3 dense + sparse
   ├─ reranker_service.py     CrossEncoder 리랭킹
   ├─ db_service.py           저장 / RRF 하이브리드 검색
   └─ ocr_service.py          미구현
```

상태도 IO도 없는 작은 작업(`parse`, `chunk`)은 클래스로 감싸지 않고 함수로 둔다.
