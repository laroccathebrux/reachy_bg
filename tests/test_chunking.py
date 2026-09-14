from src.rag.chunking import chunk_text


def test_short_text_is_a_single_chunk():
    assert chunk_text("Roll one die. A 5 or 6 is a success.") == ["Roll one die. A 5 or 6 is a success."]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n\n  ") == []


def test_long_text_is_split_with_overlap():
    sentence = "An investigator may perform up to two actions during the Action Phase. "
    text = sentence * 40  # ~2800 chars
    chunks = chunk_text(text, target=800, overlap=120)
    assert len(chunks) >= 3
    assert all(len(c) <= 800 + 120 + len(sentence) for c in chunks)
    # Overlap: the start of chunk 2 repeats the tail of chunk 1.
    tail = chunks[0][-60:]
    assert tail.split()[-1] in chunks[1][:200]


def test_paragraph_breaks_are_preferred_split_points():
    paragraph_a = "First paragraph about the Travel action. " * 12
    paragraph_b = "Second paragraph about the Rest action. " * 12
    chunks = chunk_text(f"{paragraph_a}\n\n{paragraph_b}", target=600, overlap=0)
    assert chunks[0].strip().startswith("First paragraph")
    assert any(c.strip().startswith("Second paragraph") for c in chunks[1:])


def test_tiny_trailing_piece_is_merged_into_previous_chunk():
    text = ("Sentence number one is here. " * 30) + "Tail."
    chunks = chunk_text(text, target=400, overlap=0, minimum=200)
    assert chunks[-1].endswith("Tail.")
    assert len(chunks[-1]) >= 200
