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


-- ── 확인 ─────────────────────────────────────────────────────────────────────
SELECT
    (SELECT extversion FROM pg_extension WHERE extname = 'vector')  AS pgvector,
    (SELECT count(*) FROM information_schema.tables
      WHERE table_name IN ('document', 'parent_chunk', 'child_chunk')) AS 테이블,
    (SELECT count(*) FROM pg_indexes
      WHERE tablename IN ('parent_chunk', 'child_chunk')
        AND indexdef ILIKE '%hnsw%')                                AS hnsw_인덱스;
