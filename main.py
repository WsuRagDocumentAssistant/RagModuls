"""
ragmodul 사용 예.

설정을 어디서 읽을지는 애플리케이션(여기)이 정한다. ragmodul 은 환경변수나
.env 를 보지 않고 인자로 받은 값만 쓴다.
"""

import logging
import os

from dotenv import load_dotenv

from ragmodul import RagController

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


if __name__ == "__main__":
    rag = RagController(
        embedding_model_path=os.environ.get("EMBEDDING_MODEL_PATH", "models/bge-m3"),
        reranker_model_path=os.environ.get("RERANKER_MODEL_PATH", "models/bge-reranker-v2-m3"),
        device=os.environ.get("DEVICE") or None,
        unpack_dir=os.environ.get("HWPX_UNPACK_DIR", "unpacked"),
    )
    print("embedding:", rag.embedding_model_path)
    print("reranker :", rag.reranker_model_path)
    print("device   :", rag.device)
