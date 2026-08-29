"""
DB 저장/검색 단계 — PostgreSQL + pgvector.

테이블은 sql/schema.sql 로 미리 만들어 둔다.

메모리 구조는 child 를 parent 안에 중첩해 담지만 관계형은 그럴 수 없으므로,
저장할 때 parent 를 먼저 넣어 PK 를 받고 그 값을 child 행에 내려 붙인다.

재색인은 source_path 로 기존 문서를 지우고 다시 넣는다. document 행이 사라지면
FK 의 ON DELETE CASCADE 로 parent/child 가 함께 정리되므로 따로 지울 필요가 없다.
"""

import logging

import psycopg
from pgvector import SparseVector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

# 접속에 항상 붙이는 고정 옵션. 배포마다 달라지는 값(host/port/dbname/user/password)은
# 여기 두지 않는다 — 애플리케이션이 config 로 넘긴다(main.py 의 build_db_config 가
# .env 를 읽는다).
#
# 기본값을 두면 .env 에서 항목이 빠졌을 때 조용히 그 값으로 붙는다. DB_HOST 를 깜빡하면
# 에러 대신 localhost 에 연결돼서 "테이블이 없다" 같은 엉뚱한 에러를 보게 된다.
DB_OPTIONS = {
    "client_encoding": "utf-8",
}

# sparse 차원은 임베딩 모델의 vocab 크기다. 모델이 EmbeddedService.sparse_dimension
# 으로 알려주므로 컨트롤러가 그 값을 넘긴다. 아래는 모델 없이 DbService 만 쓸 때의
# 기본값이다(BGE-M3 기준). sql/schema.sql 의 SPARSEVEC 차원과 같아야 한다.
DEFAULT_SPARSE_DIM = 250002


class DbService:

    def __init__(self, config: dict | None = None, sparse_dim: int = DEFAULT_SPARSE_DIM) -> None:
        # 고정 옵션 위에 애플리케이션이 준 접속 정보를 얹는다.
        self.config = {**DB_OPTIONS, **(config or {})}
        self.sparse_dim = sparse_dim
        self._conn: psycopg.Connection | None = None
        self.load()

    def _to_sparsevec(self, weights) -> SparseVector | None:
        """{토큰id: 가중치} -> pgvector SparseVector.

        pgvector 의 sparsevec 은 1부터 세는 색인을 쓰지만, SparseVector 가 넣고 꺼낼 때
        ±1 을 자동으로 처리하므로 여기서는 신경 쓰지 않는다.
        """
        if not weights:
            return None
        return SparseVector({int(k): float(v) for k, v in weights.items()}, self.sparse_dim)

    def load(self) -> None:
        logger.info("PostgreSQL 연결: %s:%s/%s",
                    self.config.get("host"), self.config.get("port"), self.config.get("dbname"))
        self._conn = psycopg.connect(**self.config, row_factory=dict_row)
        register_vector(self._conn)          # vector / sparsevec 타입 어댑터 등록
        logger.info("PostgreSQL 연결 완료")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── 저장 ────────────────────────────────────────────────────────────

    def save_document(self, document) -> int:
        """ChunkedDocument 를 통째로 저장하고 document.id 를 돌려준다.

        같은 source_path 의 문서가 이미 있으면 지우고 다시 넣는다(재색인).
        """
        file = document.file
        source_path = getattr(file, "filename", None) or getattr(file, "title", "")

        with self._conn.cursor() as cur:
            # 재색인: 기존 문서를 지우면 CASCADE 로 청크가 함께 사라진다
            cur.execute("DELETE FROM document WHERE source_path = %s;", (source_path,))

            cur.execute(
                """
                INSERT INTO document (
                    source_path, filename, title, creator, last_saved_by,
                    doc_created_at, doc_modified_at, language, application,
                    app_version, section_count, table_count
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id;
                """,
                (
                    source_path,
                    getattr(file, "filename", ""),
                    getattr(file, "title", None),
                    getattr(file, "creator", None),
                    getattr(file, "last_saved_by", None),
                    getattr(file, "created_at", None),
                    getattr(file, "modified_at", None),
                    getattr(file, "language", None),
                    getattr(file, "application", None),
                    getattr(file, "app_version", None),
                    getattr(file, "section_count", 0) or 0,
                    getattr(file, "table_count", 0) or 0,
                ),
            )
            document_id = cur.fetchone()["id"]

            child_rows = []
            for parent in document.parents:
                cur.execute(
                    """
                    INSERT INTO parent_chunk (document_id, heading, breadcrumb, content)
                    VALUES (%s,%s,%s,%s)
                    RETURNING id;
                    """,
                    (document_id, parent.heading, parent.breadcrumb or "", parent.content),
                )
                parent_id = cur.fetchone()["id"]

                for child in parent.children:
                    child_rows.append((
                        parent_id,
                        document_id,
                        child.content,
                        child.vector,                                   # None 이면 NULL
                        self._to_sparsevec(getattr(child, "sparse", None)),  # 아직 안 만들면 NULL
                    ))

            if child_rows:
                cur.executemany(
                    """
                    INSERT INTO child_chunk (parent_id, document_id, content, embedding, lexical)
                    VALUES (%s,%s,%s,%s,%s);
                    """,
                    child_rows,
                )

        self._conn.commit()
        logger.info("저장 완료: document=%d, parent=%d, child=%d",
                    document_id, len(document.parents), len(child_rows))
        return document_id

    # ── 검색 ────────────────────────────────────────────────────────────

    def search(self, query_vector, top_k: int = 5, document_id: int | None = None) -> list[dict]:
        """dense 검색. child 로 찾고 parent 를 붙여 돌려준다.

        반환 항목
          child_content  실제로 매칭된 조각. 리랭커가 이걸 본다(짧아서 안 잘린다).
          content        parent 본문. LLM 에 맥락으로 넘긴다.
        """
        where = ["c.embedding IS NOT NULL"]
        params: list = [query_vector]
        if document_id is not None:
            where.append("c.document_id = %s")
            params.append(document_id)
        params += [query_vector, top_k]

        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    c.id            AS child_id,
                    c.content       AS child_content,
                    p.id            AS parent_id,
                    p.content       AS content,
                    p.heading       AS heading,
                    p.breadcrumb    AS breadcrumb,
                    d.id            AS document_id,
                    d.title         AS document_title,
                    1 - (c.embedding <=> %s::vector) AS similarity
                FROM child_chunk c
                JOIN parent_chunk p ON p.id = c.parent_id
                JOIN document     d ON d.id = c.document_id
                WHERE {' AND '.join(where)}
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s;
                """,
                params,
            )
            return cur.fetchall()

    def search_lexical(self, query_weights, top_k: int = 5) -> list[dict]:
        """sparse 검색. <#> 는 음수 내적을 돌려주므로 부호를 뒤집어 점수로 쓴다."""
        qvec = self._to_sparsevec(query_weights)
        if qvec is None:
            return []

        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.id            AS child_id,
                    c.content       AS child_content,
                    p.id            AS parent_id,
                    p.content       AS content,
                    p.heading       AS heading,
                    p.breadcrumb    AS breadcrumb,
                    d.id            AS document_id,
                    d.title         AS document_title,
                    -(c.lexical <#> %s::sparsevec) AS similarity
                FROM child_chunk c
                JOIN parent_chunk p ON p.id = c.parent_id
                JOIN document     d ON d.id = c.document_id
                WHERE c.lexical IS NOT NULL
                ORDER BY c.lexical <#> %s::sparsevec
                LIMIT %s;
                """,
                (qvec, qvec, top_k),
            )
            return cur.fetchall()

    def search_hybrid(self, query_vector, query_weights, top_k: int = 5,
                      document_id: int | None = None, k: int = 60) -> list[dict]:
        """dense + sparse 를 RRF 로 합쳐 검색한다.

        두 점수를 더하지 않고 순위로 합치는 이유: dense 는 코사인이라 0~1 인데
        sparse 는 가중치 내적이라 상한이 없다. 그냥 더하면 sparse 가 결과를 지배한다.
        RRF 는 점수를 버리고 1/(k+순위) 만 쓰므로 척도가 달라도 상관없다.
        k=60 은 RRF 원 논문의 기본값 — 클수록 상위권 순위 차이를 덜 벌린다.

        후보는 양쪽에서 top_k 의 4배씩 뽑는다. 한쪽에서만 잡힌 문서도 합산 대상이
        되려면 최종 개수보다 넉넉히 봐야 한다.
        """
        qvec = self._to_sparsevec(query_weights)
        if qvec is None:                       # sparse 가 없으면 dense 단독으로 떨어진다
            return self.search(query_vector, top_k, document_id)

        pool = top_k * 4
        doc_filter = "AND document_id = %s" if document_id is not None else ""
        doc_param = [document_id] if document_id is not None else []

        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                -- 순위는 LIMIT 안쪽에서 뽑은 뒤에 매긴다. 바깥에서 매기면 윈도우
                -- 함수가 테이블 전체를 정렬해서 HNSW 로 상위 N 개만 보는 이점이 사라진다.
                WITH dense AS (
                    SELECT id, similarity, ROW_NUMBER() OVER (ORDER BY similarity DESC) AS rank
                    FROM (
                        SELECT id, 1 - (embedding <=> %s::vector) AS similarity
                        FROM child_chunk
                        WHERE embedding IS NOT NULL {doc_filter}
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    ) t
                ),
                sparse AS (
                    SELECT id, similarity, ROW_NUMBER() OVER (ORDER BY similarity DESC) AS rank
                    FROM (
                        SELECT id, -(lexical <#> %s::sparsevec) AS similarity
                        FROM child_chunk
                        WHERE lexical IS NOT NULL {doc_filter}
                        ORDER BY lexical <#> %s::sparsevec
                        LIMIT %s
                    ) t
                )
                SELECT
                    c.id            AS child_id,
                    c.content       AS child_content,
                    p.id            AS parent_id,
                    p.content       AS content,
                    p.heading       AS heading,
                    p.breadcrumb    AS breadcrumb,
                    d.id            AS document_id,
                    d.title         AS document_title,
                    dense.similarity  AS dense_score,
                    sparse.similarity AS sparse_score,
                    COALESCE(1.0 / (%s + dense.rank),  0)
                  + COALESCE(1.0 / (%s + sparse.rank), 0) AS similarity
                -- USING (id) 를 쓰면 합쳐진 id 가 뒤이어 조인하는 child_chunk.id 와
                -- 이름이 겹쳐 "칼럼 참조 id 가 모호합니다" 가 난다. ON 으로 붙이고
                -- COALESCE 로 어느 쪽에서 왔든 id 를 하나 고른다(FULL OUTER 라
                -- 한쪽만 잡힌 행은 반대쪽이 NULL 이다).
                FROM dense
                FULL OUTER JOIN sparse ON sparse.id = dense.id
                JOIN child_chunk  c ON c.id = COALESCE(dense.id, sparse.id)
                JOIN parent_chunk p ON p.id = c.parent_id
                JOIN document     d ON d.id = c.document_id
                ORDER BY similarity DESC
                LIMIT %s;
                """,
                [query_vector, *doc_param, query_vector, pool,
                 qvec, *doc_param, qvec, pool,
                 k, k, top_k],
            )
            return cur.fetchall()

    # ── 축약어 사전 ──────────────────────────────────────────────────────

    def save_vocab(self, pairs) -> int:
        """축약어/확장어 짝을 넣고 새로 들어간 확장어 수를 돌려준다.

        pairs 는 납작한 VocabPair 목록이라 같은 축약어가 여러 번 올 수 있다
        (실측: 축약어 16개에 확장어 17개 — PAMS 가 둘이다). 먼저 축약어로 묶어서
        vocab_short 를 축약어당 한 번만 건드린다.

        이미 있는 축약어는 그 id 를 재사용한다. 새 행을 만들면 확장어가 두 id 로
        흩어져 조회에서 절반만 나온다.
        """
        by_term: dict[str, list[str]] = {}
        for pair in pairs:
            expansions = by_term.setdefault(pair.term, [])
            if pair.expansion not in expansions:        # 호출 안 중복 제거
                expansions.append(pair.expansion)

        added = 0
        with self._conn.cursor() as cur:
            for term, expansions in by_term.items():
                # DO NOTHING 이면 RETURNING 이 아무것도 안 주므로 id 는 따로 읽는다.
                # 사전이 십여 개라 왕복이 늘어도 상관없고, 이쪽이 읽기 쉽다.
                cur.execute(
                    "INSERT INTO vocab_short (term) VALUES (%s) ON CONFLICT (term) DO NOTHING;",
                    (term,),
                )
                cur.execute("SELECT id FROM vocab_short WHERE term = %s;", (term,))
                short_id = cur.fetchone()["id"]

                for expansion in expansions:
                    cur.execute(
                        """
                        INSERT INTO vocab_expansion (short_id, term) VALUES (%s, %s)
                        ON CONFLICT (short_id, term) DO NOTHING;
                        """,
                        (short_id, expansion),
                    )
                    added += cur.rowcount               # 건너뛰면 0 이다
        self._conn.commit()
        logger.info("사전 저장: 축약어 %d개, 확장어 %d개 추가", len(by_term), added)
        return added

    def load_vocab(self) -> dict[str, list[str]]:
        """{축약어: [확장어, ...]} 를 통째로 읽는다.

        질의마다 DB 를 치지 않고 한 번 올려두고 쓴다 — 사전이 십여 개다.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.term AS short, e.term AS expansion
                FROM vocab_short s
                JOIN vocab_expansion e ON e.short_id = s.id
                ORDER BY s.term, e.term;
                """
            )
            vocab: dict[str, list[str]] = {}
            for row in cur.fetchall():
                vocab.setdefault(row["short"], []).append(row["expansion"])
            return vocab

    # ── 조회 도우미 ──────────────────────────────────────────────────────

    def count(self) -> dict:
        """저장 결과를 눈으로 확인할 때 쓴다."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    (SELECT count(*) FROM document)                                  AS documents,
                    (SELECT count(*) FROM parent_chunk)                              AS parents,
                    (SELECT count(*) FROM child_chunk)                               AS children,
                    (SELECT count(*) FROM child_chunk WHERE embedding IS NOT NULL)   AS embedded,
                    (SELECT count(*) FROM child_chunk WHERE lexical   IS NOT NULL)   AS lexical;
                """
            )
            return cur.fetchone()
