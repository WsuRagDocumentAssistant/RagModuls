"""
청킹 단계 - 목차(제목 계층)를 기준으로 자르고, 크면 실제 존재하는 하위 제목으로 내려간다.

원칙
  1. 목차 기준으로 자른다. 글자수로 무작정 자르지 않는다.
  2. 그룹이 크면 그 안에 '실제로 존재하는' 다음 제목으로 내려가 다시 나눈다.
  3. 더 내려갈 제목이 없을 때만 문단 -> 문장 순으로 기계 분할한다.
  4. 서술 문단은 레벨을 갖지 않는다. 자기 위 제목에 내용으로 합쳐진다.
  5. parent = 맥락용(LLM 전달), child = 검색용. child가 parent_id로 parent를 가리킨다.
  6. 표는 절대 중간에서 자르지 않는다. 표 하나가 통째로 크면 헤더를 반복해서 행 단위로 나눈다.
  7. child에는 제목을 붙여 조각만 남지 않게 한다.

제목은 두 형태로 온다
  - 문단 제목    role == '제목',            텍스트는 block.text        (□, ○ 등)
  - 제목상자 표  table.kind == '제목상자',  텍스트는 table.title 리스트 (1, 3.1, Ⅱ-1 등)
"""

import re

from ..models.chunk_model import ChildChunk, ChunkedDocument, ParentChunk

MAX_PARENT_CHARS = 5000
CHILD_CHUNK_SIZE = 500

# 표를 만나면 크기와 무관하게 청크를 끊어 표를 독립 청크로 만든다.
# 표 조각이 본문 조각과 검색 순위를 다투다 밀리는 것을 막는다.
TABLE_ALWAYS_BREAK = True

# 'Ⅱ-1' 처럼 로마숫자로 시작하는 파트 제목. hwpx가 그 아래 '1', '2'와 같은
# depth를 주기 때문에, 한 단계 위로 올려야 하위 절이 파트에 매달린다.
_ROMAN_PART = re.compile(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]")
_SEPARATOR = re.compile(r"^[\s|:-]+$")


def _heading_text(b):
    """이 블록이 제목이면 표시용 텍스트를, 아니면 None."""
    if b.role == "제목" and b.text:
        return b.text
    table = getattr(b, "table", None)
    if table is not None and getattr(table, "kind", None) == "제목상자" and table.title:
        return " ".join(table.title)
    return None


def _heading_depth(b, heading: str) -> int:
    depth = b.depth if b.depth is not None else 0
    if _ROMAN_PART.match(heading):
        return depth - 1
    return depth


def _body_text(b):
    """제목이 아닌 블록이 기여하는 본문 텍스트."""
    table = getattr(b, "table", None)
    if table is not None and getattr(table, "markdown", None):
        return table.markdown
    return b.text or None


def chunk(parsed) -> ChunkedDocument:
    """parsed: hwpx DocumentModel (parser_service.parse()의 반환값)."""
    blocks = [b for b in parsed.blocks
              if b.searchable or b.role == "제목" or _heading_text(b) is not None]
    entries = _with_paths(blocks)
    parents: list[ParentChunk] = []
    _split_group(entries, prefix_len=1, parents=parents, seq=[0])
    return ChunkedDocument(file=parsed.file, parents=parents)


def _with_paths(blocks):
    """각 블록에 (제목상자까지 반영한) 제목 경로를 붙인다.

    같은 제목 문구가 문서에 여러 번 나오므로(예: '□ 자율성과지표 정의서')
    그룹 키는 문구가 아니라 블록 id로 잡는다.
    """
    stack: list[tuple] = []
    out = []
    for b in blocks:
        heading = _heading_text(b)
        if heading is not None:
            depth = _heading_depth(b, heading)
            while stack and stack[-1][0] >= depth:
                stack.pop()
            stack.append((depth, b.id, heading))
            text = heading
        else:
            text = _body_text(b)
        if text:
            out.append((tuple((bid, txt) for _, bid, txt in stack), text))
    return out


def _split_group(entries, prefix_len: int, parents: list, seq: list) -> None:
    groups: dict[tuple, list] = {}
    order: list[tuple] = []
    for path, text in entries:
        key = tuple(bid for bid, _ in path[:prefix_len])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((path, text))

    for key in order:
        members = groups[key]
        crumbs = [txt for _, txt in members[0][0][:prefix_len]]
        heading = crumbs[-1] if crumbs else None
        breadcrumb = " > ".join(crumbs)
        content = "\n".join(text for _, text in members)

        can_go_deeper = any(len(path) > prefix_len for path, _ in members)
        if len(content) <= MAX_PARENT_CHARS or not can_go_deeper:
            _emit(content, heading, breadcrumb, parents, seq)
        else:
            _split_group(members, prefix_len + 1, parents, seq)


def _emit(content: str, heading, breadcrumb: str, parents: list, seq: list) -> None:
    content = content.strip()
    if not content:
        return
    # 제목만 있고 본문이 없는 껍데기는 만들지 않는다. 그 제목은 하위 청크의
    # breadcrumb에 이미 남아 있어 검색에서 잃는 정보가 없다.
    if heading and content == heading.strip():
        return

    parent = ParentChunk(
        id=f"parent::{seq[0]}", content=content,
        heading=heading, breadcrumb=breadcrumb,
    )
    seq[0] += 1
    # breadcrumb를 자른 뒤에 붙이므로 그만큼 미리 빼둬야 최종 길이가 한도를 지킨다.
    room = max(CHILD_CHUNK_SIZE - len(breadcrumb) - 1, 100)
    for piece in _split_text(content, room):
        parent.children.append(ChildChunk(
            id=f"child::{seq[0]}", content=_with_context(piece, breadcrumb),
        ))
        seq[0] += 1
    parents.append(parent)


def _with_context(piece: str, breadcrumb: str) -> str:
    """조각만 남지 않도록 제목 경로 전체를 앞에 붙인다.

    직속 제목만 붙이면 상위 제목('3.2 학생 지원 및 관리 체계' 등)이 임베딩되는
    텍스트에 한 번도 안 들어가서, 그 문구로 묻는 질문이 검색되지 않는다.
    """
    if not breadcrumb:
        return piece
    return f"{breadcrumb}\n{piece}"


# ── 분할 ────────────────────────────────────────────────────────────────

def _units(text: str) -> list[tuple]:
    """텍스트를 단위로 나눈다. 연속된 표 줄은 한 덩어리로 묶어 자르지 않는다."""
    units: list[tuple] = []
    buf: list[str] = []
    in_table = False
    for line in text.split("\n"):
        is_table = "|" in line
        if buf and is_table != in_table:
            units.append(("table" if in_table else "text", "\n".join(buf)))
            buf = []
        in_table = is_table
        buf.append(line)
    if buf:
        units.append(("table" if in_table else "text", "\n".join(buf)))
    return units


def _split_text(text: str, size: int) -> list[str]:
    if len(text) <= size:
        return [text]

    pieces: list[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            pieces.append(buf.strip())
        buf = ""

    for kind, unit in _units(text):
        if kind == "table" and TABLE_ALWAYS_BREAK:
            flush()
            pieces.extend(_split_table(unit, size) if len(unit) > size else [unit])
            continue
        if len(unit) > size:
            flush()
            parts = _split_table(unit, size) if kind == "table" else _split_prose(unit, size)
            pieces.extend(parts)
            continue
        candidate = f"{buf}\n{unit}" if buf else unit
        if len(candidate) <= size:
            buf = candidate
        else:
            flush()
            buf = unit
    flush()
    return [p for p in pieces if p.strip()]


def _header_body(lines: list[str]) -> tuple[list[str], list[str]]:
    """머리글 구간과 본문 행을 가른다.

    라이브러리가 header_rows마다 구분선을 넣으므로, 머리글이 여러 행이면
    (행, 구분선)이 반복된다. 그 반복이 끝나는 지점까지가 머리글이다.
    """
    end = 0
    i = 0
    while i + 1 < len(lines) and _SEPARATOR.match(lines[i + 1]):
        end = i + 2
        i += 2
    if not end:
        end = 1 if lines else 0
    return lines[:end], lines[end:]


def _split_table(markdown: str, size: int) -> list[str]:
    """표가 통째로 너무 크면 행 단위로 나누되 머리글을 매 조각에 반복한다."""
    lines = [ln for ln in markdown.split("\n") if ln.strip()]
    if not lines:
        return []
    header, body = _header_body(lines)
    if not body:
        return ["\n".join(header)]

    out: list[str] = []
    buf: list[str] = []

    def flush():
        nonlocal buf
        if buf:
            out.append("\n".join(header + buf))
            buf = []

    for row in body:
        # 행 하나만으로도 한도를 넘으면 그 행을 잘라서 조각낸다.
        if len("\n".join(header + [row])) > size:
            flush()
            out.extend(_split_row(header, row, size))
            continue
        if buf and len("\n".join(header + buf + [row])) > size:
            flush()
        buf.append(row)
    flush()
    return out


def _split_row(header: list[str], row: str, size: int) -> list[str]:
    """행 하나가 한도를 넘으면 그 행을 잘라 조각내되, 머리글을 매 조각에 붙인다."""
    head = "\n".join(header)
    room = max(size - len(head) - 1, 100)
    return [f"{head}\n{part}" if head else part for part in _wrap(row, room)]


def _wrap(text: str, room: int) -> list[str]:
    """한 줄짜리 긴 텍스트를 room 길이로 자른다. 가능하면 구분점에서 끊는다."""
    out: list[str] = []
    rest = text.strip()
    while len(rest) > room:
        window = rest[:room]
        cut = max(window.rfind(" ·"), window.rfind("| "), window.rfind(" "))
        if cut < room // 2:
            cut = room
        out.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        out.append(rest)
    return out


def _split_prose(text: str, size: int) -> list[str]:
    """문단 -> 문장 순으로 자른다."""
    out: list[str] = []
    buf = ""
    for paragraph in [p for p in text.split("\n") if p.strip()]:
        if len(paragraph) > size:
            if buf:
                out.append(buf)
                buf = ""
            sentences = re.split(r"(?<=[.!?다요함음])\s+", paragraph)
            for sentence in sentences:
                candidate = f"{buf} {sentence}".strip() if buf else sentence
                if len(candidate) <= size:
                    buf = candidate
                else:
                    if buf:
                        out.append(buf)
                    buf = sentence
            continue
        candidate = f"{buf}\n{paragraph}" if buf else paragraph
        if len(candidate) <= size:
            buf = candidate
        else:
            out.append(buf)
            buf = paragraph
    if buf:
        out.append(buf)
    return out
