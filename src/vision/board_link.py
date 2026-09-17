"""What the camera sees, read from the conversation process.

The camera lives in the preview (``src/vision/preview.py``): it owns the board reference, the
piece tracker and the motion watcher, and it keeps a :class:`BoardState` of where every piece
stands. The conversation lives in ``src/integration/talk.py``. They are two processes, and
until now they were strangers - the preview wrote ``board_state.json`` and nobody read it,
while the conversation carried its own, always empty ``GameState.board`` and saved it as "the
board is empty". The robot then told the table it could not see the board, which was true of
the process it was speaking from and false of the robot.

This is the link, and it prefers the living one:

* the preview's ``GET /state`` while it is up - the board as it is right now;
* ``board_state.json`` when it is not - the board as the last preview left it, which is worth
  saying with the caveat that nothing is watching it.

    board = board_now()
    board["text"]      # "3 piece(s): investigator:Lily Chen at Shanghai, ..."
    summary(board)     # the one line game_state carries on every turn
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import GAME_LOG_DIR, VISION_URL
from src.logger import get_logger

log = get_logger(__name__)

STATE_FILE = GAME_LOG_DIR / "board_state.json"
MAX_PIECES = 20  # what one sentence can carry; the rest are counted, not listed


def _from_preview(url: str, timeout: float) -> dict[str, Any] | None:
    import httpx

    try:
        response = httpx.get(f"{url.rstrip('/')}/state", timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        log.debug("the preview is not answering at %s (%s); reading the file instead", url, exc)
        return None
    if not isinstance(data, dict) or "pieces" not in data:
        return None
    return data


def _from_file(path: Path) -> dict[str, Any] | None:
    from src.strategy.state import BoardState

    try:
        state = BoardState.load(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        log.debug("no board file at %s (%s)", path, exc)
        return None
    return {
        "text": state.describe(),
        "count": len(state),
        "pieces": [
            {"id": p.id, "label": p.label, "where": p.where, "kind": p.kind, "named": bool(p.name)}
            for p in state.on_board
        ],
    }


def board_now(*, url: str = VISION_URL, path: Path = STATE_FILE, timeout: float = 1.5) -> dict[str, Any]:
    """The board as the camera has it, live if the preview is up and from its file if not."""
    data = _from_preview(url, timeout)
    source = "preview"
    if data is None:
        data = _from_file(path)
        source = "file"
    if data is None:
        return {
            "seen": False,
            "source": "",
            "text": "",
            "count": 0,
            "pieces": [],
            "note": "The camera is not running, so I cannot see the board. Ask the table to describe it.",
        }
    pieces = list(data.get("pieces") or [])
    count = int(data.get("count") or len(pieces))
    note = (
        "This is the board right now."
        if source == "preview"
        else "The camera is not running; this is where the pieces were when it last looked, "
        "so say that it may have moved since."
    )
    if not count:
        note = (
            "Nothing is on the board yet, or nobody has scanned it. That is not the same as an "
            "empty game: say the camera has not found any pieces, and ask the table."
        )
    return {
        "seen": True,
        "source": source,
        "text": str(data.get("text") or "the board is empty"),
        "count": count,
        "pieces": pieces[:MAX_PIECES],
        "note": note,
    }


def summary(board: dict[str, Any] | None = None) -> str:
    """One short line for ``game_state``, which is read on every turn and paid for in latency.

    The named pieces are the ones worth carrying: a piece the camera tracks but nobody has
    claimed is "a piece at Rome", which is noise in a state summary and detail for the tool.
    """
    board = board_now() if board is None else board
    if not board.get("seen"):
        return "the camera is not running"
    count = int(board.get("count") or 0)
    if not count:
        return "the camera sees no pieces on the board"
    named = [p for p in board.get("pieces") or [] if p.get("named")]
    where = ", ".join(f"{p['label']} at {p['where']}" for p in named[:6])
    stale = "" if board.get("source") == "preview" else " (last seen before the camera stopped)"
    if not named:
        return f"the camera sees {count} piece(s), none of them named yet{stale}"
    return f"the camera sees {count} piece(s){stale}; known: {where}"


__all__ = ["board_now", "summary", "STATE_FILE", "MAX_PIECES"]
