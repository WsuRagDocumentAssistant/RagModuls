#================================================
# db_manager_service.py
#================================================
"""
DB 저장/검색 단계 — db-manager 의 저장 프로시저를 부른다.

DbService 와 같은 메서드를 제공하는 어댑터다. 컨트롤러는 둘 중 무엇을 들고 있든
같은 방식으로 부른다.

DbService 와 무엇이 다른가
  DbService 는 이 저장소가 SQL 을 직접 쓴다. 이쪽은 DB 담당이 만든 프로시저를
  이름으로 부른다 — 스키마가 바뀌어도 프로시저가 흡수하므로 여기가 안 깨진다.
  운영은 이쪽이고, DbService 는 이 저장소 안에서 혼자 돌려볼 때 쓴다.

무엇을 변환하는가
  프로시저는 jsonb 를 받는다. 그래서 ChunkedDocument 를 dict 로 펴야 하고,
  numpy 배열인 dense 벡터는 list 로 바꿔야 한다(json 이 numpy 를 모른다).
  sparse 는 {토큰id: 가중치} 인데 json 을 거치면 키가 문자열이 된다 — 프로시저가
  그걸 전제로 만들어져 있다.
"""

import logging
from typing import Any

from ..util import document_to_payload, to_plain_sparse, to_plain_vector

logger = logging.getLogger(__name__)

# sparse 차원. DbService 와 같은 값이어야 한다 — 프로시저에 인자로 넘긴다.
DEFAULT_SPARSE_DIM = 250002


class DbManagerService:

    def __init__(self, manager: Any = None, sparse_dim: int = DEFAULT_SPARSE_DIM) -> None:
        """manager 를 안 주면 DBManager 를 만들어 init() 까지 한다.

        DBManager 는 생성 시 이벤트 루프를 하나 만들어 계속 재사용한다. close() 가
        그 루프까지 닫으므로, LlmService 처럼 자기 루프를 쓰는 것과 종료 순서를
        섞지 않는다 — 각자 자기 것만 닫으면 서로 간섭하지 않는다.
        """
        self.sparse_dim = sparse_dim
        self._owned = manager is None
        if manager is None:
            from db_manager import DBManager

            manager = DBManager()
            manager.init()
            logger.info("DBManager 준비 완료")
        self._manager = manager

    def close(self) -> None:
        """내가 만든 DBManager 만 닫는다. 밖에서 받은 것은 밖에서 닫는다."""
        if self._owned and self._manager is not None:
            self._manager.close()
            self._manager = None

    #------------------------------------------------┌> 저장

    def save_document(self, document) -> int:
        """ChunkedDocument 를 통째로 저장하고 document.id 를 돌려준다.

        같은 source_path 면 프로시저가 RAG 컬럼만 갱신하고 업무 분류값은 보존한다
        (UPSERT). DbService 는 지우고 다시 넣는데, 이쪽이 분류값을 안 잃는다.
        """
        payload = document_to_payload(document)
        document_id = self._manager.call(
            "index_document", document=payload, sparse_dim=self.sparse_dim)
        logger.info("저장 완료: document=%s, parent=%d, child=%d",
                    document_id, len(payload["parents"]),
                    sum(len(p["children"]) for p in payload["parents"]))
        return document_id

    #------------------------------------------------┌> 검색

    def search(self, query_vector, top_k: int = 5, document_id: int | None = None) -> list[dict]:
        """dense 검색."""
        return self._manager.call(
            "search_documents_vector",
            query_vector=to_plain_vector(query_vector), top_k=top_k, document_id=document_id)

    def search_lexical(self, query_weights, top_k: int = 5) -> list[dict]:
        """sparse 검색."""
        if not query_weights:
            return []
        return self._manager.call(
            "search_documents_lexical",
            query_weights=to_plain_sparse(query_weights),
            sparse_dim=self.sparse_dim, top_k=top_k)

    def search_hybrid(self, query_vector, query_weights, top_k: int = 5,
                      document_id: int | None = None, k: int = 60) -> list[dict]:
        """dense + sparse 를 RRF 로 합쳐 검색한다.

        query_weights 가 없으면 dense 단독으로 떨어진다 — DbService 와 같은 동작이다.
        """
        if not query_weights:
            return self.search(query_vector, top_k, document_id)
        return self._manager.call(
            "search_documents_hybrid",
            query_vector=to_plain_vector(query_vector),
            query_weights=to_plain_sparse(query_weights),
            sparse_dim=self.sparse_dim, top_k=top_k,
            document_id=document_id, k=k)

    #------------------------------------------------┌> 축약어 사전

    def save_vocab(self, pairs) -> int:
        """축약어/확장어 짝을 넣고 새로 들어간 확장어 수를 돌려준다."""
        payload = [{"term": p.term, "expansion": p.expansion} for p in pairs]
        added = self._manager.call("save_vocab_pairs", pairs=payload)
        logger.info("사전 저장: 짝 %d개 -> 확장어 %d개 추가", len(payload), added)
        return added

    def load_vocab(self) -> dict[str, list[str]]:
        """{축약어: [확장어, ...]}.

        프로시저가 jsonb 를 돌려주는데 asyncpg 가 그걸 문자열로 준다. VocabRepository
        는 dict 를 반환한다고 적어뒀지만 실제로는 str 이 나온다(실측). 여기서 푼다 —
        문자열을 그대로 넘기면 expand_query 가 dict 처럼 순회하다 깨진다.
        그쪽이 고쳐서 dict 로 오더라도 아래 검사가 그대로 통과한다.
        """
        vocab = self._manager.call("load_vocab") or {}
        if isinstance(vocab, str):
            import json

            vocab = json.loads(vocab or "{}")
        return vocab

    #------------------------------------------------┌> 조회 도우미

    def count(self) -> dict:
        """색인 통계. documents/parents/children/embedded/lexical."""
        return self._manager.call("count_documents") or {}
