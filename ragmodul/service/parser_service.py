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

    image_dir를 준 경우 model.document_images 에 DocumentImage 목록이 붙는다.
    저장 경로·문서 순서·제목 경로·캡션이 들어 있어서, 등록할 때 폴더를 훑지 않아도
    된다 - 폴더 훑기는 지난번 잔재까지 등록하는 문제가 있었다.

    DocumentModel 은 남의 dataclass 인데 속성을 붙인다. 반환 타입을 튜플로 바꾸면
    기존 호출부가 전부 깨지고, 파싱 결과의 일부라 모델에 있는 것이 자연스럽다.
    image_dir 를 안 주면 이 속성은 없다(hasattr 로 확인한다).
    """
    parser, result = hwpx.run_pipeline(file_path, out_root=unpack_dir)
    model = hwpx.build_document_model(result)
    if recover_excluded:
        _recover_excluded_tables(model, result)
    if image_dir:
        saved = _save_images(model, image_dir, unpack_dir)
        model.document_images = _collect_images(model, saved, result)
    return model


# 그림 블록 다음에 오는 이 역할의 블록을 캡션으로 본다. BLOCK_ROLE 에 caption 이
# 없어서 영어가 그대로 남는 경우까지 넣는다.
_CAPTION_ROLES = frozenset({"캡션", "caption"})


def _collect_images(model, saved: dict[str, str], result=None) -> list:
    """{ref: 저장경로} 와 블록을 엮어 DocumentImage 목록을 만든다. 문서 순서대로.

    블록에서 가져오는 것 -
      order          목록을 문서 순서로 보여줄 때 쓴다. 파일명순으로 정렬하면
                     image1, image10, image11, image2 로 섞인다.
      heading_path   그림이 있던 자리의 제목 경로. 앞 세 개가 대·중·소제목이다.
      caption        아래 두 경로에서 찾는다.

    저장 경로를 못 찾은 그림은 건너뛴다. 원본이 없어서 복사가 안 된 경우다.
    """
    from ..models.image_model import DocumentImage

    by_ref = _captions_by_ref(result)
    blocks = sorted(model.blocks, key=lambda b: b.order)
    images = []
    for index, block in enumerate(blocks):
        figure = getattr(block, "figure", None)
        ref = getattr(getattr(figure, "image", None), "ref", None)
        if ref is None or ref not in saved:
            continue
        images.append(DocumentImage(
            ref=ref,
            path=saved[ref],
            order=block.order,
            section=getattr(block, "section", 0),
            media_type=getattr(figure.image, "media_type", None),
            heading_path=list(getattr(block, "heading_path_text", []) or []),
            # ref 로 찾은 것을 먼저 쓴다. 그쪽이 확정이고 인접은 추측이다.
            caption=by_ref.get(ref) or _caption_after(blocks, index),
        ))

    images.extend(_captioned_leftovers(images, saved, by_ref))

    captioned = sum(1 for i in images if i.caption)
    logger.info("이미지 목록: %d개 (캡션 %d개)", len(images), captioned)
    return images


def _captioned_leftovers(images: list, saved: dict[str, str],
                         by_ref: dict[str, str]) -> list:
    """블록에 없지만 캡션이 달린 그림을 뒤에 붙인다.

    표 셀 안의 그림은 블록으로 안 올라와서 위의 순회에 안 잡힌다. 파일은 저장돼
    있고(manifest 기준으로 복사한다) 캡션도 ref 로 짝지어져 있는데, 문서 어디에
    있었는지만 모른다.

    캡션이 달린 것만 넣는다. manifest 에는 표 안 아이콘이나 장식까지 다 들어 있어서
    전부 넣으면 목록이 쓰레기로 찬다(실측: 그림 블록 30개인데 manifest 293개).
    캡션이 달렸다는 건 문서를 쓴 사람이 설명할 값이 있다고 본 그림이라는 뜻이다.

    [주의] order 는 진짜 위치가 아니다. 맨 뒤 번호를 이어 붙인 값이라 목록에서
    '위치 미상' 묶음처럼 끝에 몰린다. heading_path 도 비어 있다. 파서가 셀 안
    그림을 블록으로 올려주면 이 함수는 필요 없어진다.
    """
    from ..models.image_model import DocumentImage

    have = {image.ref for image in images}
    leftovers = [ref for ref in by_ref if ref not in have and ref in saved]
    if not leftovers:
        return []

    start = max((image.order for image in images), default=0)
    logger.info("블록에 없는 캡션 그림 %d개를 뒤에 붙인다(위치 미상): %s",
                len(leftovers), leftovers[:5])
    return [
        DocumentImage(ref=ref, path=saved[ref], order=start + offset,
                      caption=by_ref[ref])
        for offset, ref in enumerate(leftovers, 1)
    ]


def _captions_by_ref(result) -> dict[str, str]:
    """파서의 표 분석 결과에서 {이미지 ref: 캡션} 을 모은다.

    왜 여기서 가져오나: 표 셀 안의 그림에 달린 캡션은 블록으로 올라오지 않는다.
    블록 목록은 문서를 위에서 아래로 읽는 한 줄기라 셀 내부의 중첩 구조가 들어갈
    자리가 없고, 파서가 그걸 '기록된 정책' 으로 남겨두었다(nested_control_skipped).
    실측(SW중심대학 단계보고서): 캡션 14개 중 8개가 그렇게 빠졌고 그중 6개가 그림
    캡션이었다 - 블록만 보면 3개, 여기까지 보면 9개다.

    TableParser 가 셀마다 captions 를 수집하면서 binary_item_id_ref 를 함께
    담아준다. 그래서 어느 그림의 캡션인지 확정된다 - 인접으로 짚을 때처럼
    '그림이 연달아 있으면 어긋나는' 문제가 없다.

    model 의 Cell 에는 이 값이 안 올라온다(build_document_model 이 captions 를
    옮기지 않는다). 그래서 model 이 아니라 result 를 본다. 그쪽이 고쳐지면
    이 함수는 지우고 Cell.captions 를 보면 된다.

    구조를 훑는 이유는 표가 중첩되기 때문이다(표 안의 표 안의 그림). 깊이만
    제한하고 이름으로 찾는다 - 파서 내부 구조에 경로를 박아두면 그쪽이 바뀔 때
    조용히 빈 dict 가 된다.
    """
    if result is None:
        return {}

    found: dict[str, str] = {}
    seen: set[int] = set()

    def walk(node, depth: int = 0) -> None:
        if depth > 10 or id(node) in seen:
            return
        seen.add(id(node))

        if isinstance(node, dict):
            ref = node.get("binary_item_id_ref")
            text = (node.get("text") or "").strip() if ref else ""
            if ref and text:
                found.setdefault(ref, text)     # 먼저 나온 것을 쓴다
                return
            for value in node.values():
                walk(value, depth + 1)
            return

        if isinstance(node, (list, tuple)):
            for value in node:
                walk(value, depth + 1)
            return

        for name in ("raw", "analyzed", "tables", "rows", "cells", "child_tables",
                     "captions"):
            child = getattr(node, name, None)
            if child is not None:
                walk(child, depth + 1)

    walk(getattr(result, "tables", None))
    if found:
        logger.debug("표에서 찾은 이미지 캡션 %d개", len(found))
    return found


def _caption_after(blocks: list, index: int) -> str | None:
    """그림 블록 바로 다음이 캡션이면 그 글. 아니면 None.

    _captions_by_ref 가 못 찾았을 때만 쓴다. 본문에 바로 놓인 그림은 캡션이 블록으로
    올라오는데, 그건 ref 가 안 붙어 있어서 위치로 짚어야 한다.

    떠 있는 그림이 여러 개 겹친 구간에서는 어긋날 수 있다. 빈 문단은 건너뛴다 -
    그림과 캡션 사이에 빈 줄이 들어가는 문서가 있다(실측: 그림 > 빈문단 > 캡션).
    """
    for block in blocks[index + 1: index + 4]:
        role = getattr(block, "role", None)
        if role in _CAPTION_ROLES:
            text = (getattr(block, "text", None) or "").strip()
            return text or None
        if role not in ("빈문단", "empty_paragraph"):
            return None                     # 캡션이 아닌 내용이 먼저 나왔다
    return None


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
