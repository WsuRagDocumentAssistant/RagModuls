#================================================
# parser_service.py
#================================================
"""
파싱 단계 - 문서를 구조화된 DocumentModel 로 만든다.

hwpx·docx·xlsx·pdf 를 통합 파서(wsu-document-parser, Parser 저장소)가 읽는다. 제목 계층·
제목상자·표 판정·그림 위치까지 거기서 계산하므로 여기서는 호출하고 결과를 우리 규칙에 맞춘다.

예전에는 hwpx 전용 HwpxParser_sub 를 썼다. 같은 문서로 비교해 제목·단락 구조가 같은 수준인
것을 확인하고 바꿨다(2026-10-07, 제목 수 RISE 319/319, 2주기 206/202). 통합 파서와 그대로
쓰면 달라지는 세 가지를 여기서 예전과 같게 맞춘다.

  - 문서 제목: 통합 파서는 문서 내부 메타 제목을 쓰는데, 다른 문서에서 복사해 온 엉뚱한 값인
    경우가 있다(실측: RISE 사업계획서의 내부 제목이 '대전RISE 평가체계 구축 연구'). 답변 출처에
    그 이름이 뜨므로 파일명(확장자 없이)으로 덮는다. 저장 키(source_path)도 이 값이라 예전 행과 맞는다.
  - 그림 파일: 통합 파서는 <image_dir>/<문서명>/images/<SHA-256>.<확장자> 로 저장한다. 이미지
    목록·편집기에 해시가 보이지 않게 예전처럼 <image_dir>/<문서명>/<ref>.<확장자> 로 옮긴다.
  - 그림 목록 타입: ragmodul 의 DocumentImage 로 바꿔 담는다(필드는 같고 클래스만 다르다).
"""

import logging
import shutil
import tempfile
from pathlib import Path

from ..models.image_model import DocumentImage

logger = logging.getLogger(__name__)

# 통합 파서가 읽는 형식. 업로드 단계가 이 목록으로 먼저 거른다(색인까지 가서 실패하지 않게).
SUPPORTED_SUFFIXES = (".hwpx", ".docx", ".xlsx", ".pdf")


def parse(file_path: str, unpack_dir: str = "unpacked", image_dir: str | None = None):
    """문서를 DocumentModel 로 만든다. 형식은 확장자로 고른다.

    image_dir 를 주면 본문 그림 블록의 이미지를 image_dir/<문서명>/ 에 저장하고
    model.document_images 에 DocumentImage 목록(저장 경로·문서 순서·제목 경로·캡션)을 붙인다.
    안 주면 그림 파일을 쓰지 않고 목록은 비어 있다.

    unpack_dir 는 그림을 옮기기 전에 잠깐 받아 두는 자리다. 이 문서 몫의 임시 폴더만 만들고
    끝나면 지운다 — 같은 순간에 다른 문서를 파싱하는 스레드의 산출물은 건드리지 않는다.

    표 셀 안의 그림은 목록에 넣지 않는다(예전과 같다). 그 파일은 임시 폴더와 함께 지워지므로
    셀의 그림 참조(table.cells[].images)에는 경로가 없다.
    """
    from document_parser import parse as parse_document

    path = Path(file_path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"지원하지 않는 문서 형식입니다: {path.suffix or '(확장자 없음)'}. "
                         f"{', '.join(SUPPORTED_SUFFIXES)} 만 등록할 수 있습니다.")

    if image_dir:
        Path(unpack_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=unpack_dir) as staging:
            model = parse_document(path, image_dir=staging)
            model.document_images = _keep_images(model, Path(image_dir) / path.stem)
    else:
        model = parse_document(path)
        model.document_images = []

    # 파일명을 제목·저장 키로 쓴다(모듈 설명 참고).
    model.file.title = model.file.filename = path.stem
    logger.info("파싱: %s — 블록 %d개, 그림 %d개", path.name, len(model.blocks), len(model.document_images))
    return model


#------------------------------------------------┌> 이미지

def _keep_images(model, target: Path) -> list[DocumentImage]:
    """임시 폴더에 받은 본문 그림을 target/<ref>.<확장자> 로 옮기고 DocumentImage 목록을 만든다.

    같은 그림이 문서에 여러 번 놓이면 통합 파서가 파일 하나를 같이 쓰므로 한 번만 옮긴다.
    모델 안의 경로(model.images, 그림 블록의 image.path)도 옮긴 자리로 바꾼다 — 그렇지 않으면
    임시 폴더가 지워진 뒤 없는 경로를 가리킨다.
    """
    images = list(getattr(model, "document_images", None) or [])
    if not images:
        return []
    target.mkdir(parents=True, exist_ok=True)

    moved: dict[str, str] = {}                  # ref -> 옮긴 경로
    for image in images:
        if image.ref in moved:
            continue
        source = Path(image.path)
        if not source.is_file():
            logger.warning("그림 파일이 없다: %s (%s)", image.ref, source)
            continue
        dest = target / f"{image.ref}{source.suffix.lower()}"
        shutil.copy2(source, dest)
        moved[image.ref] = str(dest)

    for ref, saved in model.images.items():
        saved.path = moved.get(ref, "")
    for block in model.blocks:
        image = getattr(getattr(block, "figure", None), "image", None)
        if image is not None:
            image.path = moved.get(image.ref)
    for table in model.tables():
        for cell in table.cells:
            for image in cell.images:
                image.path = moved.get(image.ref)

    kept = [DocumentImage(ref=i.ref, path=moved[i.ref], order=i.order, section=i.section,
                          media_type=i.media_type, heading_path=list(i.heading_path),
                          caption=i.caption)
            for i in images if i.ref in moved]
    logger.info("이미지 저장: %d개 -> %s (캡션 %d개)", len(kept), target,
                sum(1 for i in kept if i.caption))
    return kept
