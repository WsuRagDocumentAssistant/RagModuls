"""DB 저장/검색 단계. 구현 예정 — Embedde의 DbService를 참고해 이식 예상."""


class DbService:

    def load(self) -> None:
        raise NotImplementedError("TODO: DB 연결 구현")

    def save_document(self, document) -> None:
        """document: ChunkedDocument. file 정보 + parent/child(벡터 포함)를 통째로 받는다.

        관계형으로 넣을 때는 child 행에 parent_id가 필요하다. 메모리 구조는
        중첩이라 그 키가 없으므로 여기서 parent.id를 내려 붙인다.
        """
        raise NotImplementedError("TODO: 저장 구현")

    def search(self, query_vector, top_k: int) -> list:
        raise NotImplementedError("TODO: 검색 구현")
