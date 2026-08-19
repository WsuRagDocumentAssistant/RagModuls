"""
파싱 단계 - hwpx 패키지로 문서를 구조화된 DocumentModel로 만든다.

hwpx.run_pipeline()이 depth(제목 계층)/heading_path(제목 경로)/표 구조/
이미지 위치까지 이미 다 계산해주므로, 여기서는 그걸 호출만 한다.

한 가지만 더 한다: 라이브러리가 '제외:OCR'로 비워둔 표를 필터 전 원본으로 되살린다.
"""

import hwpx
from hwpx.analysis.build_document_model import table_markdown
from hwpx.analysis.table_filter import cell_text, index_tables, state_view


def parse(file_path: str, unpack_dir: str = "unpacked", recover_excluded: bool = True):
    parser, result = hwpx.run_pipeline(file_path, out_root=unpack_dir)
    model = hwpx.build_document_model(result)
    if recover_excluded:
        _recover_excluded_tables(model, result)
    return model


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
