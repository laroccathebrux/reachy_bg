from src.rag.sessions import session_text


def test_session_text_keeps_context_in_front():
    text = session_text("20260914-2030", "Cthulhu", 3, "  Lily closed the gate in Shanghai.  ")
    assert text == "Session 20260914-2030 vs Cthulhu, round 3: Lily closed the gate in Shanghai."


def test_session_text_handles_unknown_ancient_one():
    assert session_text("s", None, 1, "x").startswith("Session s vs an unknown Ancient One, round 1:")
