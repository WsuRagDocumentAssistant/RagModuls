"""OCR 단계 — 이미지/스캔 PDF에서 텍스트를 뽑아낸다. 구현 예정.

실제 구현체는 보통 엔진/모델(Tesseract, PaddleOCR 등) 또는 클라우드 API
클라이언트를 로드해서 들고 있어야 하므로 클래스로 유지한다
(EmbeddedService/RerankerService와 같은 이유).
"""


class OcrService:

    def run(self, file_path: str) -> str:
        raise NotImplementedError("TODO: OCR 구현")
