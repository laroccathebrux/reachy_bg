"""Split a Fantasy Flight rules PDF into titled sections using its typography and layout.

The official Eldritch Horror rulebook and reference guide have no PDF outline, but their
headings are set in the small-caps display face ``UglyQua`` at 16-18 pt (sections) and
12-14 pt (subsections), while body text is Adobe Garamond at 9-10 pt. Small caps split every
word into several spans of different sizes (the capital at heading size, the rest smaller),
so headings are detected per *line*, using the largest span on the line.

Both documents are laid out in two columns with occasional full-width banners. Reading
order is rebuilt per page: full-width blocks split the page into horizontal bands, and
inside a band blocks are read column by column, top to bottom. Running headers and footers
(page numbers, "Con-El" glossary ranges) are ignored.

Public API:

    sections = extract_sections(Path("data/rulebooks/eh01_rulebook.pdf"), HeadingRules())
    for s in sections:
        print(s.path, s.page_start, len(s.text))
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf


@dataclass(frozen=True)
class HeadingRules:
    """Font-based heuristics; defaults fit the FFG Eldritch Horror PDFs."""

    heading_font_prefix: str = "UglyQua"
    section_min_size: float = 15.5  # 16 pt and above = top-level section
    subsection_min_size: float = 11.5  # 12-15 pt = subsection
    min_heading_chars: int = 3
    max_heading_chars: int = 90
    skip_pages: tuple[int, ...] = ()  # 1-based pages to ignore (covers, index)
    header_margin: float = 0.06  # fraction of page height treated as running header
    footer_margin: float = 0.05  # fraction of page height treated as footer
    full_width_ratio: float = 0.6  # a block wider than this fraction of the page spans columns
    column_block_ratio: float = 0.3  # blocks at least this wide define where columns start
    column_gap_ratio: float = 0.08  # x0 gap (fraction of page width) that starts a new column
    min_body_size: float = 8.0  # smaller text is illustration labels, not rules
    heading_join_gap: float = 12.0  # points between two heading blocks that form one title


@dataclass
class Section:
    title: str
    level: int  # 1 = section, 2 = subsection
    parent: str | None
    page_start: int
    page_end: int
    text: str = ""
    paragraphs: list[str] = field(default_factory=list)

    @property
    def section(self) -> str:
        return self.parent if self.level == 2 and self.parent else self.title

    @property
    def subsection(self) -> str:
        return self.title if self.level == 2 else ""

    @property
    def path(self) -> str:
        return f"{self.section} > {self.subsection}" if self.subsection else self.section


_WS = re.compile(r"[ \t ]+")
_DOT_LEADERS = re.compile(r"\.{4,}")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_RUNNING_HEAD = re.compile(r"^[A-Z][a-z]{0,3}-[A-Z][a-z]{0,3}$")  # glossary ranges like "Con-El"


def _clean_line(text: str) -> str:
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("’", "'")
    return _WS.sub(" ", text).strip()


def _span_text(span: dict[str, Any]) -> str:
    """Text of a span, with symbol fonts mapped to something readable."""
    font = span.get("font", "")
    if "Ornament" in font:  # bullet glyphs ("^") from BodoniOrnaments
        return " "
    if "Icons" in font:  # skill / reckoning icons from EldritchHorrorIcons
        return " [icon] "
    return span.get("text", "")


def _line_text_and_size(line: dict[str, Any]) -> tuple[str, float]:
    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
    if not spans:
        return "", 0.0
    text = "".join(_span_text(s) for s in line["spans"])
    sizes = [float(s["size"]) for s in spans if "Ornament" not in s.get("font", "")]
    return _clean_line(text), max(sizes or [float(spans[0]["size"])])


def _is_heading_face(line: dict[str, Any], prefix: str) -> bool:
    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
    total = sum(len(s["text"]) for s in spans)
    if total == 0:
        return False
    heading_chars = sum(len(s["text"]) for s in spans if s.get("font", "").startswith(prefix))
    return heading_chars / total >= 0.6


def _looks_like_heading(text: str, rules: HeadingRules) -> bool:
    if not (rules.min_heading_chars <= len(text) <= rules.max_heading_chars):
        return False
    if _DOT_LEADERS.search(text):  # table of contents lines
        return False
    if re.fullmatch(r"[\d\W]+", text):  # page numbers, ornaments
        return False
    if _RUNNING_HEAD.match(text):
        return False
    return True


def _adjacent(previous: tuple[float, ...], current: tuple[float, ...], rules: HeadingRules) -> bool:
    """True when ``current`` sits directly under ``previous`` in the same column."""
    if previous == current:  # two lines of the same block
        return True
    vertical_gap = current[1] - previous[3]
    previous_center = (previous[0] + previous[2]) / 2
    current_center = (current[0] + current[2]) / 2
    return (
        -rules.heading_join_gap <= vertical_gap <= rules.heading_join_gap
        and abs(previous_center - current_center) < 80
    )


_SENTENCE_END_CHARS = ".!?:;\"')]"


def _merge_continuations(paragraphs: list[str]) -> list[str]:
    """Join a block that merely continues the previous one (PyMuPDF splits the reference
    guide's bullets into 'restricted to resolving each action only once per' / 'round.')."""
    merged: list[str] = []
    for paragraph in (p.strip() for p in paragraphs if p and p.strip()):
        if merged:
            previous = merged[-1]
            continues = paragraph[0].islower() or previous[-1] not in _SENTENCE_END_CHARS
            if continues and not paragraph[0].isdigit():
                merged[-1] = f"{previous} {paragraph}"
                continue
        merged.append(paragraph)
    return merged


def _finish_paragraphs(paragraphs: list[str]) -> str:
    text = "\n\n".join(_merge_continuations(paragraphs))
    return _HYPHEN_BREAK.sub(r"\1\2", text).strip()


def _ordered_blocks(page: pymupdf.Page, rules: HeadingRules) -> list[dict[str, Any]]:
    """Text blocks of a page in reading order (bands of columns), minus headers and footers."""
    width, height = page.rect.width, page.rect.height
    top, bottom = height * rules.header_margin, height * (1 - rules.footer_margin)
    blocks = [
        b
        for b in page.get_text("dict")["blocks"]
        if b.get("type") == 0 and b["bbox"][3] > top and b["bbox"][1] < bottom
    ]
    full = [b for b in blocks if (b["bbox"][2] - b["bbox"][0]) >= width * rules.full_width_ratio]
    narrow = [b for b in blocks if b not in full]
    boundaries = sorted({b["bbox"][1] for b in full})

    def band_of(block: dict[str, Any]) -> int:
        y = block["bbox"][1]
        return sum(1 for boundary in boundaries if y >= boundary)

    def column_of(block: dict[str, Any], columns: list[float]) -> int:
        center = (block["bbox"][0] + block["bbox"][2]) / 2
        return max((i for i, start in enumerate(columns) if center >= start), default=0)

    ordered: list[dict[str, Any]] = []
    for band in range(len(boundaries) + 1):
        members = [b for b in narrow if band_of(b) == band]
        # Column starts come from the wide body blocks only; centred headings and short
        # fragments are then assigned to the column that contains their centre.
        wide = [b for b in members if (b["bbox"][2] - b["bbox"][0]) >= width * rules.column_block_ratio]
        starts = {round(b["bbox"][0]) for b in (wide or members)}
        if members:
            starts.add(round(min(b["bbox"][0] for b in members)))  # the leftmost column always exists
        starts = sorted(starts)
        columns: list[float] = []
        for x in starts:
            if not columns or x - columns[-1] > width * rules.column_gap_ratio:
                columns.append(x)
        members.sort(key=lambda b: (column_of(b, columns), b["bbox"][1], b["bbox"][0]))
        banner = [b for b in full if band_of(b) == band]
        # A full-width block opens its band (it is the boundary the band starts at).
        ordered.extend(sorted(banner, key=lambda b: b["bbox"][1]))
        ordered.extend(members)
    return ordered


def extract_sections(pdf_path: Path, rules: HeadingRules | None = None) -> list[Section]:
    """Walk the PDF in reading order and return one Section per heading found.

    Text before the first heading is attached to a synthetic "Front matter" section so
    nothing is lost. Sections without body text are dropped.
    """
    rules = rules or HeadingRules()
    doc = pymupdf.open(pdf_path)
    sections: list[Section] = []
    current: Section | None = None
    current_section_title: str | None = None

    def open_section(title: str, level: int, page: int) -> None:
        nonlocal current, current_section_title
        if current is not None:
            current.text = _finish_paragraphs(current.paragraphs)
            sections.append(current)
        parent = current_section_title if level == 2 else None
        if level == 1:
            current_section_title = title
        current = Section(title=title, level=level, parent=parent, page_start=page, page_end=page)

    for page_index, page in enumerate(doc):
        page_no = page_index + 1
        if page_no in rules.skip_pages:
            continue
        # A heading may wrap onto a second line that PyMuPDF puts in a separate block;
        # keep the pending heading (text, level, bbox) across blocks and join when the next
        # heading block of the same level sits right below it.
        pending: tuple[str, int, tuple[float, ...]] | None = None
        for block in _ordered_blocks(page, rules):
            paragraph_lines: list[str] = []
            bbox = tuple(block["bbox"])
            for line in block.get("lines", []):
                text, size = _line_text_and_size(line)
                if not text or _RUNNING_HEAD.match(text):
                    continue
                level = 0
                if _is_heading_face(line, rules.heading_font_prefix):
                    if size >= rules.section_min_size:
                        level = 1
                    elif size >= rules.subsection_min_size:
                        level = 2
                if level and _looks_like_heading(text, rules):
                    if paragraph_lines and current is not None:
                        current.paragraphs.append(" ".join(paragraph_lines))
                        paragraph_lines = []
                    if pending and pending[1] == level and _adjacent(pending[2], bbox, rules):
                        pending = (f"{pending[0]} {text}", level, bbox)
                    else:
                        if pending:
                            open_section(pending[0], pending[1], page_no)
                        pending = (text, level, bbox)
                    continue
                if size < rules.min_body_size:
                    continue  # labels inside illustrations
                if pending:
                    open_section(pending[0], pending[1], page_no)
                    pending = None
                if current is None:
                    open_section("Front matter", 1, page_no)
                paragraph_lines.append(text)
            if paragraph_lines and current is not None:
                current.paragraphs.append(" ".join(paragraph_lines))
                current.page_end = page_no
        if pending:
            open_section(pending[0], pending[1], page_no)
    if current is not None:
        current.text = _finish_paragraphs(current.paragraphs)
        sections.append(current)
    return [s for s in sections if s.text]


__all__ = ["HeadingRules", "Section", "extract_sections"]
