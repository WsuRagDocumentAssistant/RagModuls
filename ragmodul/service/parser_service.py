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
    """
    if not model.images:
        return {}

    stem = Path(getattr(model.file, "filename", "") or "document").stem
    target = Path(out_dir) / stem
    target.mkdir(parents=True, exist_ok=True)

    source_root = _unpacked_root(unpack_dir, stem)
    saved, missing = {}, []
    for ref, image in model.images.items():
        src = source_root / image.path if source_root else None
        if src is None or not src.is_file():
            missing.append(ref)
            continue
        shutil.copy2(src, target / Path(image.path).name)   # 수정시각까지 보존
        saved[ref] = str(target / Path(image.path).name)

    logger.info("이미지 저장: %d개 -> %s", len(saved), target)
    if missing:
        logger.warning("원본을 못 찾은 이미지 %d개: %s", len(missing), missing[:5])
    return saved


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
