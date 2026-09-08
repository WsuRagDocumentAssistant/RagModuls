#================================================
# parser_service.py
#================================================
"""
파싱 단계 - hwpx 패키지로 문서를 구조화된 DocumentModel로 만든다.

hwpx.run_pipeline()이 depth(제목 계층)/heading_path(제목 경로)/표 구조/
이미지 위치까지 이미 다 계산해주므로, 여기서는 그걸 호출만 한다.

두 가지만 더 한다.
  - 라이브러리가 '제외:OCR'로 비워둔 표를 필터 전 원본으로 되살린다.
  - image_dir을 주면 문서 이미지를 그 폴더로 빼낸다.
"""

import logging
import shutil
from pathlib import Path

import hwpx
from hwpx.analysis.build_document_model import table_markdown
from hwpx.analysis.table_filter import cell_text, index_tables, state_view

logger = logging.getLogger(__name__)


def parse(file_path: str, unpack_dir: str = "unpacked", recover_excluded: bool = True,
          image_dir: str | None = None):
    """hwpx를 DocumentModel로 만든다.

    image_dir를 주면 문서 이미지를 image_dir/<문서명>/ 으로 복사한다. 안 주면
    복사하지 않는다 - 이미지는 unpack_dir 안에도 풀려 있으므로 필요한 쪽만 켜면 된다.
    """
    parser, result = hwpx.run_pipeline(file_path, out_root=unpack_dir)
    model = hwpx.build_document_model(result)
    if recover_excluded:
        _recover_excluded_tables(model, result)
    if image_dir:
        _save_images(model, image_dir, unpack_dir)
    return model


def _save_images(model, out_dir: str, unpack_dir: str) -> dict[str, str]:
    """문서 이미지를 out_dir/<문서명>/ 으로 복사하고 {ref: 저장경로}를 돌려준다.

    unpack_dir 안에도 이미지가 있지만 그건 파싱 산출물이라 언제 지워도 되는 곳이다.
    오래 두고 쓸 이미지는 우리가 정한 곳으로 옮긴다.

    문서마다 하위 폴더를 만든다. 이미지 ref가 문서 안에서만 유일해서(image1, image2...)
    문서 두 개를 처리하면 image1.jpg가 서로 덮어쓴다.

    BMP는 PNG로 바꿔 저장한다(_to_png). 여기서 한 번 바꾸면 그 뒤로는 아무도 BMP를
    보지 않는다 - DB에 적히는 경로도, 클라이언트가 받는 파일도 PNG다.
    """
    if not model.images:
        return {}

    stem = Path(getattr(model.file, "filename", "") or "document").stem
    target = Path(out_dir) / stem
    target.mkdir(parents=True, exist_ok=True)

    source_root = _unpacked_root(unpack_dir, stem)
    saved, missing, converted = {}, [], 0
    for ref, image in model.images.items():
        src = source_root / image.path if source_root else None
        if src is None or not src.is_file():
            missing.append(ref)
            continue

        dst = target / Path(image.path).name
        png = _to_png(src, dst)
        if png is not None:
            dst = png                   # 확장자가 .png 로 바뀌었다
            converted += 1
        else:
            shutil.copy2(src, dst)      # 수정시각까지 보존
        saved[ref] = str(dst)

    logger.info("이미지 저장: %d개 -> %s (PNG 변환 %d개)", len(saved), target, converted)
    if missing:
        logger.warning("원본을 못 찾은 이미지 %d개: %s", len(missing), missing[:5])
    return saved


# BMP만 바꾼다. jpg는 이미 압축돼 있어 PNG로 바꾸면 오히려 커지고, PNG는 바꿀 게 없다.
_CONVERT_TO_PNG = {".bmp", ".dib"}


def _to_png(src: Path, dst: Path) -> Path | None:
    """BMP면 PNG로 저장하고 그 경로를 준다. 아니면 아무것도 안 하고 None.

    왜 바꾸나(실측, 문서 하나 243장 기준) -
      BMP   39개  합계 106.6MB  최대 15.8MB  평균 2,799KB
      그 외 42개  합계   7.5MB  최대  1.1MB  평균   182KB
    BMP는 무압축이라 장수는 비슷한데 용량이 14배다. LLM에 보낼 때 base64로 33%가 더
    붙어서, 15.8MB짜리는 21MB가 되어 Gemini 인라인 한도(약 20MB)를 넘긴다.

    PNG는 무손실이라 화질이 그대로다. 도표의 가는 선과 작은 글자가 안 뭉개진다 -
    JPEG로 바꿨다면 그게 망가져서 그림 속 글자를 읽는 작업에 나빴을 것이다.

    형식 제약도 같이 풀린다(실측):
      gpt / claude   BMP를 400으로 거부
      gemini / local BMP를 읽음
    바꿔두면 provider를 자유롭게 고를 수 있다.

    돌려주는 경로는 확장자가 .png다. 부르는 쪽이 그걸 그대로 기록하므로 DB의
    image_path도 .png가 되고, 클라이언트가 그 경로로 파일을 찾는다.

    Pillow가 없거나 읽지 못하는 파일이면 None을 주고 부르는 쪽이 그냥 복사한다 -
    변환은 최적화지 필수가 아니다. 그림을 잃는 것보다 큰 채로 두는 편이 낫다.
    """
    if src.suffix.lower() not in _CONVERT_TO_PNG:
        return None
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow가 없어 BMP를 그대로 둔다: %s", src.name)
        return None

    dst_png = dst.with_suffix(".png")
    try:
        with Image.open(src) as img:
            img.save(dst_png, format="PNG", optimize=True)
    except Exception as e:                          # noqa: BLE001
        logger.warning("PNG 변환 실패, 원본을 쓴다: %s (%s)", src.name, e)
        return None

    logger.debug("PNG 변환: %s %.1fMB -> %.1fMB", src.name,
                 src.stat().st_size / 1048576, dst_png.stat().st_size / 1048576)
    return dst_png


def _unpacked_root(unpack_dir: str, stem: str) -> Path | None:
    """ImageFile.path의 기준이 되는 폴더를 찾는다.

    path가 'BinData/image1.jpg' 같은 상대경로인데, 그 기준이 되는 폴더를 파서가
    돌려주지 않는다. 지금은 <unpack_dir>/unpacked/<문서명>/ 이지만 그 규칙에 기대면
    라이브러리가 바뀔 때 조용히 깨지므로, BinData를 가진 폴더를 찾는다.
    """
    root = Path(unpack_dir)
    candidates = [p.parent for p in root.rglob("BinData") if p.is_dir()] if root.is_dir() else []
    if not candidates:
        logger.warning("BinData 폴더를 못 찾았다: %s", root)
        return None
    if len(candidates) == 1:
        return candidates[0]
    # 문서 여러 개가 풀려 있으면 이름으로 고른다
    for path in candidates:
        if path.name == stem:
            return path
    logger.warning("문서 폴더가 %d개다. 첫 번째를 쓴다.", len(candidates))
    return candidates[0]


def _recover_excluded_tables(model, result) -> int:
    """'제외:OCR'로 비워진 표 자리를 필터 전 원본 셀 내용으로 채운다.

    라이브러리는 격자로 서지만 레코드가 완전하지 않은 표를 일부러 비우고
    'OCR 결과가 들어올 자리'로 남긴다(table_filter.classify의 S5c 단계).
    구조를 못 믿겠다는 판단 자체는 타당하지만, 이 문서에서는 자율성과지표
    정의서·달성도와 요약표 등 핵심 수치가 전부 그 표들에 있어서 비워두면
    수치 질문에 원천적으로 답할 수 없다.

    그래서 구조 신뢰도를 포기하는 대신 내용은 살린다. 필터는 사본에만
    적용되므로 PipelineResult에는 셀 텍스트가 그대로 남아 있다.
    """
    excluded = [b for b in model.blocks if b.excluded_table is not None]
    if not excluded:
        return 0

    tables = index_tables(state_view(result))
    recovered = 0
    for block in excluded:
        node = tables.get(str(block.excluded_table.table_id))
        if node is None:
            continue
        markdown = table_markdown(node, cell_text)
        if markdown:
            block.text = markdown
            recovered += 1
    return recovered
