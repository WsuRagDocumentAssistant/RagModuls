#================================================
# parser_service.py
#================================================
"""
파싱 단계 - hwpx 패키지로 문서를 구조화된 DocumentModel로 만든다.

hwpx.run_pipeline()이 depth(제목 계층)/heading_path(제목 경로)/표 구조/
이미지 위치까지 이미 다 계산해주므로, 여기서는 그걸 호출만 한다.

세 가지만 더 한다.
  - 라이브러리가 '제외:OCR'로 비워둔 표를 필터 전 원본으로 되살린다.
  - image_dir을 주면 본문 그림 블록의 이미지를 그 폴더로 빼낸다.
  - 압축을 푼 자리(unpack_dir/<문서명>/)를 지운다. 파싱이 끝나면 볼 일이 없다.
"""

import logging
import shutil
from pathlib import Path

import hwpx
from hwpx.analysis.build_document_model import table_markdown
from hwpx.analysis.table_filter import cell_text, index_tables, state_view

logger = logging.getLogger(__name__)


def parse(file_path: str, unpack_dir: str = "unpacked", recover_excluded: bool = True,
          image_dir: str | None = None, cleanup: bool = True):
    """hwpx를 DocumentModel로 만든다.

    image_dir를 주면 본문 그림 블록(block.figure.image)의 이미지를
    image_dir/<문서명>/ 으로 복사한다. 안 주면 복사하지 않는다.

    cleanup 이면 압축을 푼 자리(unpack_dir/<문서명>/)를 지운다. 파서가 푼 폴더를
    알고 있으므로 이 문서 것만 정확히 지운다 - 부르는 쪽이 unpack_dir 을 통째로
    비우면 같은 순간에 다른 문서를 파싱하는 워커의 산출물까지 지운다. 파싱이
    실패하면 남긴다. 무엇을 받았는지 열어봐야 하는 경우가 그때다(hwp 를 hwpx 로
    올려서 zip 이 아니었던 일이 있었다). 풀린 파일을 들여다보려면 cleanup=False.

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
        model.document_images = _extract_images(model, parser, image_dir, unpack_dir)
    if cleanup:
        _remove_unpacked(parser, unpack_dir)
    return model


def _remove_unpacked(parser, unpack_dir: str) -> None:
    """파서가 푼 문서 폴더를 지운다. 못 지워도 파싱을 실패로 만들지 않는다.

    unpack_dir 바깥은 절대 지우지 않는다. 파서가 엉뚱한 경로를 들고 있어도(라이브러리가
    바뀌어서) 피해가 unpack_dir 안에 갇힌다.
    """
    target = getattr(parser, "unpacked_dir_path", None)
    if target is None:
        return
    target = Path(target)
    root = Path(unpack_dir).resolve()
    try:
        inside = target.resolve().is_relative_to(root) and target.resolve() != root
    except OSError:
        inside = False
    if not inside or not target.is_dir():
        logger.warning("압축 해제분을 안 지운다(unpack_dir 밖이거나 없음): %s", target)
        return
    try:
        shutil.rmtree(target)
        logger.info("압축 해제분 삭제: %s", target)
    except OSError as e:
        logger.warning("압축 해제분 삭제 실패(%s): %s - %s", target, type(e).__name__, e)


#------------------------------------------------┌> 이미지

# 그림 블록 다음에 오는 이 역할의 블록을 캡션으로 본다. BLOCK_ROLE 에 caption 이
# 없어서 영어가 그대로 남는 경우까지 넣는다.
_CAPTION_ROLES = frozenset({"캡션", "caption"})


def _extract_images(model, parser, image_dir: str, unpack_dir: str) -> list:
    """본문 그림 블록의 이미지만 복사하고 DocumentImage 목록으로 만든다. 문서 순서대로.

    기준은 block.figure.image 다. hwpx 는 hp:pic 이 최상위 문단의 run 바로 아래에
    있을 때만 그림 블록으로 올린다. 표 셀 안·묶음 개체 안·도형 안의 그림과
    셀 배경 채우기(fillBrush) 이미지는 블록이 안 되고, 여기서도 다루지 않는다.

    manifest(model.images) 전체를 복사하지 않는 이유: 그림 블록은 그 일부다
    (실측 - 2주기 보고서 81개 중 13개, SW중심대학 293개 중 29개). 전부 복사하면
    등록도 설명도 안 되는 파일이 디스크에만 쌓인다(문서당 수십~수백 개).

    [알아둘 것] 이 보고서들은 레이아웃용 표 안에 그래프·절차도를 넣는 식이라,
    실제 그림의 74~85% 가 표 셀 안에 있다(중앙값 750KB, 아이콘이 아니다). 그건
    block.table.cells[].images 로 ref 와 위치가 잡히므로, 필요해지면 여기에 그
    경로를 더하면 된다. 지금은 일부러 넣지 않는다.
    """
    blocks = _figure_blocks(model)
    saved = _save_images(model, [ref for ref, _ in blocks], image_dir,
                         _unpacked_root(parser, unpack_dir))
    return _collect_images(model, blocks, saved)


def _figure_blocks(model) -> list[tuple[str, object]]:
    """[(ref, 그림 블록)] 을 문서 순서로. 같은 ref 가 두 번 놓였으면 첫 자리만."""
    seen: set[str] = set()
    out = []
    for block in sorted(model.blocks, key=lambda b: b.order):
        figure = getattr(block, "figure", None)
        ref = getattr(getattr(figure, "image", None), "ref", None)
        if ref is None or ref in seen:
            continue
        seen.add(ref)
        out.append((ref, block))
    return out


def _collect_images(model, blocks: list, saved: dict[str, str]) -> list:
    """그림 블록과 {ref: 저장경로} 를 엮어 DocumentImage 목록을 만든다.

    블록에서 가져오는 것 -
      order          목록을 문서 순서로 보여줄 때 쓴다. 파일명순으로 정렬하면
                     image1, image10, image11, image2 로 섞인다.
      heading_path   그림이 있던 자리의 제목 경로. 앞 세 개가 대·중·소제목이다.
      caption        그림 블록 바로 뒤의 캡션 블록.

    저장 경로를 못 찾은 그림은 건너뛴다. 원본이 없어서 복사가 안 된 경우다.
    """
    from ..models.image_model import DocumentImage

    ordered = sorted(model.blocks, key=lambda b: b.order)
    index_of = {b.id: i for i, b in enumerate(ordered)}

    images = []
    for ref, block in blocks:
        if ref not in saved:
            continue
        figure = block.figure
        images.append(DocumentImage(
            ref=ref,
            path=saved[ref],
            order=block.order,
            section=getattr(block, "section", 0),
            media_type=getattr(figure.image, "media_type", None),
            heading_path=list(getattr(block, "heading_path_text", []) or []),
            caption=_caption_after(ordered, index_of[block.id]),
        ))

    captioned = sum(1 for i in images if i.caption)
    logger.info("이미지 목록: %d개 (캡션 %d개)", len(images), captioned)
    return images


def _caption_after(blocks: list, index: int) -> str | None:
    """그림 블록 바로 다음이 캡션이면 그 글. 아니면 None.

    본문에 바로 놓인 그림은 캡션이 블록으로 올라오는데, ref 가 안 붙어 있어서
    위치로 짚어야 한다.

    떠 있는 그림이 여러 개 겹친 구간에서는 어긋날 수 있다. 빈 문단은 건너뛴다 -
    그림과 캡션 사이에 빈 줄이 들어가는 문서가 있다(실측: 그림 > 빈문단 > 캡션).

    대개 None 이다. 한글은 그림마다 캡션 자리를 만들어두지만 사용자가 채우지 않으면
    빈 요소로 남는다(실측한 보고서: 그림 39개 중 <hp:caption> 6개, 그 6개도 전부
    텍스트 없음).
    """
    for block in blocks[index + 1: index + 4]:
        role = getattr(block, "role", None)
        if role in _CAPTION_ROLES:
            text = (getattr(block, "text", None) or "").strip()
            return text or None
        if role not in ("빈문단", "empty_paragraph"):
            return None                     # 캡션이 아닌 내용이 먼저 나왔다
    return None


def _save_images(model, refs: list[str], out_dir: str,
                 source_root: Path | None) -> dict[str, str]:
    """refs 에 든 이미지만 out_dir/<문서명>/ 으로 복사하고 {ref: 저장경로}를 돌려준다.

    source_root 는 ImageFile.path('BinData/image1.jpg')의 기준 폴더다. 압축을 푼
    자리는 parse() 가 끝나며 지우므로, 오래 두고 쓸 이미지는 여기서 옮겨둬야 한다.

    문서마다 하위 폴더를 만든다. 이미지 ref가 문서 안에서만 유일해서(image1, image2...)
    문서 두 개를 처리하면 image1.jpg가 서로 덮어쓴다.
    """
    if not refs or not model.images:
        return {}

    stem = Path(getattr(model.file, "filename", "") or "document").stem
    target = Path(out_dir) / stem
    target.mkdir(parents=True, exist_ok=True)

    saved, missing = {}, []
    for ref in refs:
        image = model.images.get(ref)
        src = source_root / image.path if (image is not None and source_root) else None
        if src is None or not src.is_file():
            missing.append(ref)
            continue
        shutil.copy2(src, target / Path(image.path).name)   # 수정시각까지 보존
        saved[ref] = str(target / Path(image.path).name)

    logger.info("이미지 저장: %d개 -> %s (manifest %d개 중)",
                len(saved), target, len(model.images))
    if missing:
        logger.warning("원본을 못 찾은 이미지 %d개: %s", len(missing), missing[:5])
    return saved


def _unpacked_root(parser, unpack_dir: str) -> Path | None:
    """ImageFile.path('BinData/image1.jpg')의 기준이 되는 폴더.

    파서가 BinData 경로(image_dir_path)를 들고 있으므로 그 부모를 쓴다. zip 안에
    문서 폴더가 한 겹 더 들어간 경우도 파서가 이미 풀어서 잡아둔 값이다.

    그 속성이 없거나 폴더가 없으면(라이브러리가 바뀐 경우) unpack_dir 에서 BinData 를
    가진 폴더를 찾는 예전 방식으로 떨어진다.
    """
    bin_dir = getattr(parser, "image_dir_path", None)
    if bin_dir is not None and Path(bin_dir).is_dir():
        return Path(bin_dir).parent

    stem = getattr(parser, "filename", "")
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


#------------------------------------------------┌> 제외 표 복구

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
