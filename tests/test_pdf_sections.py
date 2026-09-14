"""The section extractor is exercised on a synthetic two-column PDF that mimics the FFG
typography (display face for headings, serif body), so the test needs no copyrighted file."""

from pathlib import Path

import pymupdf
import pytest

from src.rag.pdf_sections import HeadingRules, extract_sections

# Fonts available in every PyMuPDF build. The extractor keys on the *font name prefix*,
# so the synthetic rules use Helvetica-Bold ("helv" -> "Helvetica-Bold") as the heading face.
HEADING_FONT = "hebo"  # Helvetica-Bold
BODY_FONT = "tiro"  # Times-Roman


def _write_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    left, right = 54, 312
    # Running header that must be ignored.
    page.insert_text((280, 30), "Con-El", fontsize=12, fontname=HEADING_FONT)
    # Left column: a section with two subsections; one subsection title wraps onto 2 lines.
    page.insert_text((left, 90), "Phase 1: Action Phase", fontsize=17, fontname=HEADING_FONT)
    page.insert_text(
        (left, 115), "Each investigator performs up to two actions.", fontsize=10, fontname=BODY_FONT
    )
    page.insert_text((left, 145), "Travel Action", fontsize=13, fontname=HEADING_FONT)
    page.insert_text((left, 165), "Move to an adjacent space.", fontsize=10, fontname=BODY_FONT)
    page.insert_text((left, 195), "Prepare for", fontsize=13, fontname=HEADING_FONT)
    page.insert_text((left, 211), "Travel Action", fontsize=13, fontname=HEADING_FONT)
    page.insert_text((left, 231), "Gain one travel ticket.", fontsize=10, fontname=BODY_FONT)
    page.insert_text((left, 260), "tiny label in an illustration", fontsize=5, fontname=BODY_FONT)
    # Right column: next section (centred title, like the FFG layout).
    page.insert_text((right + 40, 90), "Phase 2: Encounter Phase", fontsize=17, fontname=HEADING_FONT)
    page.insert_text(
        (right, 115), "Each investigator resolves one encounter.", fontsize=10, fontname=BODY_FONT
    )
    page.insert_text((right, 145), "Combat Encounters", fontsize=13, fontname=HEADING_FONT)
    page.insert_text(
        (right, 165), "Resolve a Will test, then a Strength test.", fontsize=10, fontname=BODY_FONT
    )
    # Footer page number.
    page.insert_text((300, 775), "7", fontsize=11, fontname=BODY_FONT)
    doc.save(path)


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic.pdf"
    _write_pdf(path)
    return path


def test_sections_follow_column_reading_order(pdf: Path):
    rules = HeadingRules(
        heading_font_prefix="Helvetica-Bold", section_min_size=15.5, subsection_min_size=11.5
    )
    sections = extract_sections(pdf, rules)
    paths = [s.path for s in sections]
    assert paths == [
        "Phase 1: Action Phase",
        "Phase 1: Action Phase > Travel Action",
        "Phase 1: Action Phase > Prepare for Travel Action",
        "Phase 2: Encounter Phase",
        "Phase 2: Encounter Phase > Combat Encounters",
    ]


def test_body_text_and_pages_are_captured(pdf: Path):
    rules = HeadingRules(heading_font_prefix="Helvetica-Bold")
    by_path = {s.path: s for s in extract_sections(pdf, rules)}
    travel = by_path["Phase 1: Action Phase > Travel Action"]
    assert travel.text == "Move to an adjacent space."
    assert travel.page_start == travel.page_end == 1
    combat = by_path["Phase 2: Encounter Phase > Combat Encounters"]
    assert "Will test" in combat.text


def test_running_heads_footers_and_tiny_labels_are_ignored(pdf: Path):
    rules = HeadingRules(heading_font_prefix="Helvetica-Bold")
    all_text = "\n".join(s.text for s in extract_sections(pdf, rules))
    assert "Con-El" not in all_text
    assert "tiny label" not in all_text
    assert "\n7" not in all_text and not all_text.endswith("7")


def test_continuation_blocks_are_merged_into_one_paragraph():
    from src.rag.pdf_sections import _finish_paragraphs

    text = _finish_paragraphs(
        [
            "Each investigator is restricted to resolving each action only once per",
            "round.",
            "If an investigator cannot or does not wish to perform an action, he",
            "may choose not to.",
            "Related Topics: Acquire Assets Action",
        ]
    )
    assert text == (
        "Each investigator is restricted to resolving each action only once per round.\n\n"
        "If an investigator cannot or does not wish to perform an action, he may choose not to.\n\n"
        "Related Topics: Acquire Assets Action"
    )
