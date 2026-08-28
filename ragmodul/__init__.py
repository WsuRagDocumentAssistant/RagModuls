#================================================
# __init__.py
#================================================
"""
RAG 처리 모듈.

단계 하나가 메서드 하나다. 순서·재시도·단계 간 데이터 전달은 부르는 쪽 책임이고,
이 모듈은 요청받은 단계만 실행한다. 실패는 예외로 올라온다.

    from ragmodul import RagController

    rag = RagController(
        embedding_model_path="models/bge-m3",
        reranker_model_path="models/bge-reranker-v2-m3",
    )

    # 문서 등록
    parsed = rag.parse_document("문서.hwpx")
    document = rag.chunk_parent_child(parsed)
    rag.embed_bge_m3(document)
    rag.save_to_vector_db(document)

    # 질의 검색
    query = "세부과제 2-2의 추진 내용은?"
    qvec, qweights = rag.embed_query(query)
    hits = rag.hybrid_search(qvec, qweights, top_k=40)   # 조각. 넉넉히 뽑는다
    contexts = rag.build_contexts(hits)                  # LLM 입력 단위로 묶기
    contexts = rag.rerank(query, contexts, top_k=5)      # 최종 순서

build_contexts 에 limit 을 걸지 말 것. 약한 신호(유사도)로 미리 자른 뒤 강한 신호
(리랭커)에게 남은 것만 주면 정답이 잘린다 — 실측으로 recall 이 93%에서 83%로
떨어졌다. 후보를 다 넘기고 리랭커가 top_k 로 줄이게 한다.
"""

from .controller import RagController
from .models.chunk_model import ChildChunk, ChunkedDocument, ParentChunk
from .models.search_model import RetrievedChild, RetrievedContext
from .service.chunker_service import chunk
from .service.embedded_service import EmbeddedService
from .service.ocr_service import OcrService
from .service.parser_service import parse
from .service.reranker_service import RerankerService

__all__ = [
    # 진입점
    "RagController",
    # 데이터 모델 — 청킹 결과
    "ChunkedDocument",
    "ParentChunk",
    "ChildChunk",
    # 데이터 모델 — 검색 결과
    "RetrievedContext",
    "RetrievedChild",
    # 단계별로 따로 쓰고 싶을 때
    "parse",
    "chunk",
    "EmbeddedService",
    "RerankerService",
    "OcrService",
]
