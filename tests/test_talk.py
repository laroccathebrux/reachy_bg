def test_the_camera_is_held_only_while_the_voice_plays_and_not_on_every_tick():
    """The watcher loop ticks ten times a second; the hold is refreshed about once a second and
    expires by itself, so a crash mid-sentence cannot leave the camera switched off."""
    from src.integration.talk import Table

    table = Table.__new__(Table)
    table.held_camera_at = 0.0
    table.audio = type("A", (), {"speaking": False})()

    calls: list[float] = []
    import src.vision.board_link as link

    original = link.hold_still
    link.hold_still = lambda s=1.5: calls.append(s) or True
    try:
        assert table.hold_camera_while_speaking() is False  # quiet table, no calls at all
        assert calls == []

        table.audio.speaking = True
        assert table.hold_camera_while_speaking(hold_s=2.0) is True
        assert calls == [2.0]

        # The next tick, 100 ms later, must not send another.
        assert table.hold_camera_while_speaking(hold_s=2.0) is False
        assert calls == [2.0]

        table.held_camera_at -= 5.0  # a second has passed
        assert table.hold_camera_while_speaking(hold_s=2.0) is True
        assert calls == [2.0, 2.0]
    finally:
        link.hold_still = original


def test_the_agent_clock_is_not_started_by_whispers_own_noise():
    """"E aí" on near-silence, 73 times in one day: forwarded to the agent, rightly unanswered,
    and reported as the cloud session having died. The audio still goes up - only the clock
    does not start, because filtering speech is the one thing that must not happen here."""
    from src.integration.talk import Table

    table = Table.__new__(Table)
    table.waiting_since = None
    table.waiting_text = ""

    table.released("E aí", voice=0.31)  # the median of the noise measured on 2026-09-17
    assert table.waiting_since is None and table.waiting_text == ""

    table.released("Quantos dados eu tenho que usar?", voice=0.61)  # the median of real speech
    assert table.waiting_since is not None
    assert table.waiting_text == "Quantos dados eu tenho que usar?"


def test_a_quiet_real_sentence_only_costs_a_warning_never_the_sentence_itself():
    """Missing one is cheap: the next sentence arms the clock, and nothing was dropped."""
    from src.integration.talk import Table

    table = Table.__new__(Table)
    table.waiting_since = None
    table.waiting_text = ""

    table.released("uma frase baixinha", voice=0.45)
    assert table.waiting_since is None

    table.released("e a próxima, normal", voice=0.7)
    assert table.waiting_text == "e a próxima, normal"
