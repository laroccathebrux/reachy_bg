"""The bridge from the camera process to the conversation process."""

import json

from src.strategy.state import BoardState
from src.vision.board_link import board_now, summary


class FakeSighting:
    def __init__(self, space, kind="piece", name=None):
        self.space = space
        self.near = None
        self.kind = kind
        self.name = name
        self.x = 0.0
        self.y = 0.0
        self.score = 1.0


def _saved(tmp_path, spaces, named=None):
    state = BoardState()
    state.seed([FakeSighting(s) for s in spaces])
    for piece_id, name in (named or {}).items():
        state.name(piece_id, name)
    return state.save(tmp_path / "board_state.json")


DEAD = "http://127.0.0.1:1"  # nothing listens here, so the file is the only source


def test_the_file_is_read_when_the_camera_is_not_running(tmp_path):
    path = _saved(tmp_path, ["Rome", "Shanghai"], {1: "investigator:Lily Chen"})
    board = board_now(url=DEAD, path=path)
    assert board["seen"] and board["source"] == "file" and board["count"] == 2
    assert "may have moved" in board["note"]  # never passed off as live
    line = summary(board)
    assert "Lily Chen at Rome" in line  # label speaks the name, not the "investigator:" prefix
    assert "last seen before the camera stopped" in line


def test_nothing_at_all_is_said_plainly(tmp_path):
    board = board_now(url=DEAD, path=tmp_path / "nothing.json")
    assert board["seen"] is False and board["count"] == 0
    assert summary(board) == "the camera is not running"


def test_an_empty_board_is_not_the_same_as_no_camera(tmp_path):
    path = _saved(tmp_path, [])
    board = board_now(url=DEAD, path=path)
    assert board["seen"] is True and board["count"] == 0
    assert "has not found any pieces" in board["note"] or "Nothing is on the board" in board["note"]
    assert summary(board) == "the camera sees no pieces on the board"


def test_the_live_preview_wins_over_the_file(tmp_path, monkeypatch):
    path = _saved(tmp_path, ["Rome"])
    live = {"text": "2 piece(s): a piece at Tokyo, a piece at London", "count": 2, "pieces": []}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return live

    import httpx

    monkeypatch.setattr(httpx, "get", lambda *a, **k: Response())
    board = board_now(url="http://127.0.0.1:8090", path=path)
    assert board["source"] == "preview" and board["count"] == 2
    assert board["note"] == "This is the board right now."


def test_unnamed_pieces_are_counted_not_listed(tmp_path):
    path = _saved(tmp_path, ["Rome", "Tokyo", "London"])
    assert "none of them named yet" in summary(board_now(url=DEAD, path=path))


def test_the_preview_reads_back_the_board_it_left(tmp_path):
    """Saving on every change and loading on no start is how a scan was thrown away by a
    restart of the camera - the robot then reported an empty board it had already read."""
    path = _saved(tmp_path, ["Rome"], {1: "investigator:Lily Chen"})
    again = BoardState.load(path)
    assert len(again) == 1
    assert again.on_board[0].name == "investigator:Lily Chen"
    assert json.loads(path.read_text())["pieces"][0]["space"] == "Rome"


def test_the_camera_is_told_to_hold_still_and_a_missing_preview_is_not_fatal(monkeypatch):
    """The camera is in the head, and the speech-synced sway moves it while the agent talks.
    On 2026-09-17 that read as three pieces leaving three different spaces in one instant and
    emptied a board nobody had touched."""
    import httpx

    from src.vision.board_link import hold_still

    sent = {}

    def post(url, json=None, timeout=None):
        sent.update(url=url, json=json, timeout=timeout)
        return type("R", (), {})()

    monkeypatch.setattr(httpx, "post", post)
    assert hold_still(2.0, url="http://127.0.0.1:8090") is True
    assert sent["url"].endswith("/still") and sent["json"] == {"seconds": 2.0}
    assert sent["timeout"] and sent["timeout"] < 1.5  # never block the voice on the camera

    def dead(*a, **k):
        raise httpx.ConnectError("nobody home")

    monkeypatch.setattr(httpx, "post", dead)
    assert hold_still(2.0) is False  # a preview that is down must not stop the robot talking
