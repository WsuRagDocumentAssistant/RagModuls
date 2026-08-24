-- ============================================================================
-- RagModul 저장 스키마
--
-- 대상: PostgreSQL 17 + pgvector 0.8.x  (검증 환경: PG 17.10 / pgvector 0.8.3)
-- 실행: pgAdmin 쿼리 도구에서 이 파일을 그대로 실행한다.
--
-- 차원 값의 근거 (임의 선택이 아니다)
--   vector(1024)        BGE-M3 의 hidden_size. config.json 에서 확인.
--   sparsevec(250002)   같은 모델의 vocab_size. 토크나이저에서 확인.
--
-- 설계 요지
--   - 벡터는 child_chunk 에만 둔다. parent 는 임베딩하지 않고 LLM 맥락으로만 쓴다.
--   - 청커가 만드는 'child::1' 같은 식별자와 문서 내 순번은 저장하지 않는다.
--     부모 연결·역추적·재색인은 bigserial PK 와 document_id 로 되고,
--     문서 내 순서는 삽입 순서대로 증가하는 id 로 정렬해서 얻는다.
--   - child 에 document_id 를 중복해서 둔다. 정규화상 parent 를 거치면 알 수 있지만,
--     문서 단위 삭제와 '이 문서 안에서만 검색' 필터가 벡터 인덱스와 같은 테이블에서
--     끝나야 빠르다.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;


-- ── 문서 ─────────────────────────────────────────────────────────────────────
-- 파서(hwpx)가 내놓는 FileInfo 를 그대로 받는다.
CREATE TABLE IF NOT EXISTS document (
    id                BIGSERIAL PRIMARY KEY,

    -- 같은 파일을 두 번 색인하지 않도록 경로를 유일 키로 둔다.
    source_path       TEXT        NOT NULL UNIQUE,

    filename          TEXT        NOT NULL,
    title             TEXT,
    creator           TEXT,
    last_saved_by     TEXT,

    -- 문서 자체의 시각. 재색인 여부는 doc_modified_at 비교로 판단한다.
    doc_created_at    TIMESTAMPTZ,
    doc_modified_at   TIMESTAMPTZ,

    language          TEXT,
    application       TEXT,
    app_version       TEXT,
    section_count     INTEGER     NOT NULL DEFAULT 0,
    table_count       INTEGER     NOT NULL DEFAULT 0,

    indexed_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE  document IS 'hwpx 문서 1건. 재색인 시 이 행을 지우면 청크가 함께 정리된다.';
COMMENT ON COLUMN document.doc_modified_at IS '문서 수정 시각. 이 값이 달라졌을 때만 다시 색인한다.';


-- ── parent 청크 (맥락용) ──────────────────────────────────────────────────────
-- 목차 단위로 묶인 덩어리. 검색 결과에 붙여 LLM 에 통째로 넘긴다.
-- 벡터 컬럼이 없다 — 같은 내용을 두 번 색인하면 검색에서 서로 경쟁하고,
-- 큰 parent 는 임베딩 길이 상한에 걸려 뒷부분이 잘린다.
CREATE TABLE IF NOT EXISTS parent_chunk (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES document(id) ON DELETE CASCADE,

    heading      TEXT,                      -- 직속 제목  예: '○ 비전 및 중장기 발전계획'
    breadcrumb   TEXT NOT NULL DEFAULT '',  -- 제목 경로  예: '3 대학의 교육혁신 성과 > 3.1 유연한 학사 운영'
    content      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_parent_document ON parent_chunk (document_id);

COMMENT ON TABLE parent_chunk IS '맥락 단위. 임베딩하지 않고 LLM 입력으로만 쓴다.';


-- ── child 청크 (검색용) ───────────────────────────────────────────────────────
-- 임베딩 대상. content 앞에 breadcrumb 이 붙어 있어 조각만 봐도 어느 섹션인지 안다.
CREATE TABLE IF NOT EXISTS child_chunk (
    id           BIGSERIAL PRIMARY KEY,
    parent_id    BIGINT NOT NULL REFERENCES parent_chunk(id) ON DELETE CASCADE,
    document_id  BIGINT NOT NULL REFERENCES document(id)     ON DELETE CASCADE,

    content      TEXT NOT NULL,             -- 이 문자열이 임베딩된다

    -- dense. 정규화된 벡터라 내적이 곧 코사인 유사도다.
    -- 임베딩 전에 저장할 수도 있으므로 NULL 을 허용한다.
    embedding    VECTOR(1024),

    -- sparse(lexical). BGE-M3 의 {토큰id: 가중치} 를 그대로 담는다.
    -- pgvector 는 비영 원소를 16,000 개까지 허용한다. 청크당 수백 개 수준이라 여유가 있다.
    -- 주의: pgvector 의 sparsevec 은 1부터 시작하는 색인을 쓴다. 파이썬 쪽
    -- SparseVector 가 넣고 꺼낼 때 자동으로 ±1 을 해주므로 코드에서 신경 쓸 일은 없다.
    lexical      SPARSEVEC(250002)
);

CREATE INDEX IF NOT EXISTS idx_child_parent   ON child_chunk (parent_id);
CREATE INDEX IF NOT EXISTS idx_child_document ON child_chunk (document_id);

-- dense 근사 최근접 검색. 벡터가 정규화되어 있으므로 cosine 으로 잡는다.
CREATE INDEX IF NOT EXISTS idx_child_embedding
    ON child_chunk USING hnsw (embedding vector_cosine_ops);

-- sparse 검색. lexical weight 는 내적으로 비교한다(<#> 는 음수 내적을 돌려주므로
-- 점수로 쓸 때는 부호를 뒤집는다).
CREATE INDEX IF NOT EXISTS idx_child_lexical
    ON child_chunk USING hnsw (lexical sparsevec_ip_ops);

COMMENT ON TABLE  child_chunk IS '검색 단위. 여기만 벡터를 갖는다.';
COMMENT ON COLUMN child_chunk.document_id IS '문서 필터·삭제를 벡터 인덱스와 같은 테이블에서 끝내기 위한 의도적 중복.';
COMMENT ON COLUMN child_chunk.lexical IS 'BGE-M3 lexical weight. 하이브리드 검색용.';


-- ── 단어 사전 ────────────────────────────────────────────────────────────────
-- 약어가 질의에 들어오면 확장어를 덧붙일 때 쓴다. sparse(lexical) 검색 전용이다.
--
-- 방향이 있다: 약어 -> 확장어 한 방향이고 반대로는 하지 않는다.
--   실측(질의 8방향) — 긴 구절을 덧붙이면 효과가 크고(MD->마이크로디그리 0->7건),
--   짧은 약어를 덧붙이면 거의 안 바뀐다(5건 중 1건, 그것도 1->2). 짧은 약어는
--   sparse 가중치가 약해 이미 긴 토큰이 있는 질의의 순위를 못 바꾼다.
--
-- 문서 색인에는 넣지 않는다. 색인 쪽에 확장어를 박으면 그 말이 실제보다 흔해 보여
-- IDF 가 떨어지고 변별력을 잃는다 - sparse 를 도우려는 짓이 sparse 를 망친다.
-- 질의 쪽이라 사전을 고쳐도 재색인이 필요 없다.

-- 축약어
CREATE TABLE IF NOT EXISTS vocab_short (
    id    BIGSERIAL PRIMARY KEY,
    term  TEXT NOT NULL UNIQUE
);

-- 확장어. 축약어를 가리키는 FK 가 그대로 기본키의 일부다 — 대리키(id)가 필요 없다.
-- term 이 키에 같이 들어가는 이유: 축약어 하나에 확장어가 여러 개인 경우가 있다.
--   IR   -> '대학성과통합관리 시스템', '대학성과통합관리시스템'  (공백 변형)
--   PAMS -> 'Partnership of Asia ...', 'Partnership of Asian ...' (오타 변형)
-- 복합키라 같은 짝이 두 번 들어가지도 않는다(seed 재실행 안전).
CREATE TABLE IF NOT EXISTS vocab_expansion (
    short_id  BIGINT NOT NULL REFERENCES vocab_short(id) ON DELETE CASCADE,
    term      TEXT   NOT NULL,
    PRIMARY KEY (short_id, term)
);

-- short_id 로 찾는 조회는 복합키 인덱스의 앞 컬럼을 타므로 별도 인덱스가 필요 없다.

COMMENT ON TABLE vocab_short     IS '축약어. 질의에서 이걸 찾는다.';
COMMENT ON TABLE vocab_expansion IS '확장어. 축약어가 걸리면 이걸 질의에 덧붙인다.';

-- 조회는 전체를 한 번 읽어 메모리에서 찾는다. 질의 문자열 '안에' 축약어가 있는지
-- 훑어야 해서 WHERE term = ? 로는 안 되고, 십수 개 수준이라 통째로 읽는 게 싸다.
--   SELECT s.term AS short, e.term AS expansion
--   FROM vocab_short s JOIN vocab_expansion e ON e.short_id = s.id;
-- 공백 변형('대학성과통합관리 시스템' vs '...시스템')은 맞출 때 Python 이 흡수한다.


-- ── 확인 ─────────────────────────────────────────────────────────────────────
SELECT
    (SELECT extversion FROM pg_extension WHERE extname = 'vector')  AS pgvector,
    (SELECT count(*) FROM information_schema.tables
      WHERE table_name IN ('document', 'parent_chunk', 'child_chunk',
                           'vocab_short', 'vocab_expansion'))                   AS 테이블,
    (SELECT count(*) FROM pg_indexes
      WHERE tablename IN ('parent_chunk', 'child_chunk')
        AND indexdef ILIKE '%hnsw%')                                AS hnsw_인덱스;
