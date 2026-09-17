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
