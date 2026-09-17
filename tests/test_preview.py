"""The camera preview server on synthetic frames: page, status, JPEG frame, MJPEG stream, head control."""

import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from src.vision.preview import BOUNDARY, FakeCamera, Preview, crop_frame, encode_jpeg, serve


def test_encode_jpeg_downscales_and_keeps_full_size_otherwise():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    small = encode_jpeg(frame, max_width=320)
    full = encode_jpeg(frame)
    assert small[:2] == b"\xff\xd8" and full[:2] == b"\xff\xd8"
    import io

    from PIL import Image

    assert Image.open(io.BytesIO(small)).size == (320, 180)
    assert Image.open(io.BytesIO(full)).size == (640, 360)


def test_server_end_to_end(tmp_path):
    camera = FakeCamera(160, 90)
    preview = Preview(camera, stream_width=120, fps=20.0, capture_dir=tmp_path).start()
    server = serve(preview, port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "<canvas" in page and "/stream" in page
        frame = urllib.request.urlopen(base + "/frame.jpg").read()
        assert frame[:2] == b"\xff\xd8"
        status = json.loads(urllib.request.urlopen(base + "/status").read())
        assert status["frames"] >= 1 and status["width"] == 160 and status["height"] == 90
        assert status["warning"] == ""
        with urllib.request.urlopen(base + "/stream") as stream:
            assert stream.headers["Content-Type"].endswith(BOUNDARY)
            first = stream.readline()
            assert first.strip() == f"--{BOUNDARY}".encode()
            headers = b""
            while True:
                line = stream.readline()
                if line in (b"\r\n", b""):
                    break
                headers += line
            assert b"Content-Type: image/jpeg" in headers
        req = urllib.request.Request(
            base + "/look",
            data=json.dumps({"pitch": 90, "yaw": -12}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        reply = json.loads(urllib.request.urlopen(req).read())
        assert reply == {"ok": True, "pitch": 60.0, "yaw": -12.0, "body": 0.0}  # pitch clamped
        assert camera.looks[-1] == (60.0, -12.0, None)
        req = urllib.request.Request(
            base + "/sweep",
            data=json.dumps({"body_yaws": [30, 0], "pitch": 35}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        assert json.loads(urllib.request.urlopen(req).read()) == {"ok": True, "views": 2}
        preview._sweep_thread.join(10.0)
        assert preview.sweep_state.startswith("sweep done: 2")
        gallery = urllib.request.urlopen(base + "/gallery").read().decode()
        assert "left30" in gallery and "/captures/sweep_" in gallery
        first_img = gallery.split('src="')[1].split('"')[0]
        assert urllib.request.urlopen(base + first_img).read()[:2] == b"\xff\xd8"
        try:
            urllib.request.urlopen(base + "/captures/../../etc/passwd")
            raise AssertionError("a path outside the capture folder was served")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
        snap = json.loads(
            urllib.request.urlopen(urllib.request.Request(base + "/snapshot", method="POST")).read()
        )
        assert snap["path"].startswith(str(tmp_path)) and snap["path"].endswith(".jpg")
        assert status["pitch"] == 35.0
    finally:
        server.shutdown()
        preview.stop()


def test_warning_when_no_frames_arrive():
    class Dark:
        pitch = yaw = 0.0

        def get_frame(self):
            return None

    clock = {"t": 0.0}
    preview = Preview(Dark(), clock=lambda: clock["t"])
    assert preview.status()["warning"] == ""
    clock["t"] = 10.0
    assert "permission" in preview.status()["warning"]


def test_crop_frame_stays_inside_the_frame():
    frame = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
    assert crop_frame(frame, 1, 0.5, 0.5) is frame
    centre = crop_frame(frame, 2, 0.5, 0.5)
    assert centre.shape == (50, 100, 3) and np.array_equal(centre, frame[25:75, 50:150])
    corner = crop_frame(frame, 4, 0.0, 1.0)
    assert corner.shape == (25, 50, 3) and np.array_equal(corner, frame[75:100, 0:50])


def test_zoom_endpoint_changes_the_stream_size(tmp_path):
    import io

    from PIL import Image

    camera = FakeCamera(160, 90)
    preview = Preview(camera, stream_width=400, fps=20.0, capture_dir=tmp_path).start()
    server = serve(preview, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        req = urllib.request.Request(
            base + "/zoom",
            data=json.dumps({"factor": 2, "cx": 0.25, "cy": 0.25}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        assert json.loads(urllib.request.urlopen(req).read()) == {"ok": True, "zoom": [2.0, 0.25, 0.25]}
        seq = preview.frames
        preview.wait_jpeg(seq + 1, 2.0)
        _, jpeg = preview.wait_jpeg(seq + 1, 2.0)
        assert Image.open(io.BytesIO(jpeg)).size == (80, 45)  # a 2x crop of 160x90, below the stream width
        assert json.loads(urllib.request.urlopen(base + "/status").read())["zoom"] == 2.0
    finally:
        server.shutdown()
        preview.stop()


def test_board_endpoints_with_a_synthetic_board(tmp_path):
    pytest.importorskip("cv2")
    from src.vision.board_map import BoardReference
    from tests.test_board_map import perspective_frame, synthetic_board

    board = synthetic_board()
    frame, _ = perspective_frame(board)

    class Still:
        pitch = yaw = body = 0.0

        def get_frame(self):
            return frame.copy()

    preview = Preview(Still(), stream_width=320, fps=10.0, capture_dir=tmp_path)
    preview.board = BoardReference(image=board, min_inliers=20)
    preview.start()
    server = serve(preview, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 10.0
        answer = {}
        while time.monotonic() < deadline and not answer.get("inliers"):
            answer = json.loads(urllib.request.urlopen(base + "/board").read())
            time.sleep(0.2)
        assert answer["inliers"] >= 20 and len(answer["outline"]) == 4
        assert any(s["name"] == "London" for s in answer["spaces"])
        assert all(0 <= s["x"] <= 1 for s in answer["spaces"] if s["name"] == "London")
        top = urllib.request.urlopen(base + "/rectified.jpg").read()
        assert top[:2] == b"\xff\xd8"
        assert json.loads(urllib.request.urlopen(base + "/status").read())["board"] >= 20
    finally:
        server.shutdown()
        preview.stop()


def test_scan_learns_the_empty_board_then_finds_a_token_from_every_view(tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    import src.vision.detect as detect_module

    monkeypatch.setattr(detect_module, "CLOSER_LOOK_MARGIN", 1000.0)  # every piece gets a closer look
    from src.vision.board_map import BoardReference
    from src.vision.spaces import BY_NAME
    from tests.test_board_map import perspective_frame, synthetic_board

    board = synthetic_board()
    empty, true_h = perspective_frame(board)
    with_token = board.copy()
    rome = BY_NAME["Rome"].pixel(board.shape[1], board.shape[0])
    cv2.circle(with_token, (int(rome[0]), int(rome[1])), 14, (30, 30, 230), -1)
    busy = cv2.warpPerspective(with_token, true_h, (empty.shape[1], empty.shape[0]))
    current = {"frame": empty}

    class Camera:  # the same picture from every pose: three views that fully overlap
        pitch = yaw = body = 0.0
        looks = []

        def get_frame(self):
            return current["frame"].copy()

        def look(self, pitch, yaw, body_yaw=None):
            self.looks.append((pitch, yaw, body_yaw))
            if body_yaw is not None:
                self.body = body_yaw

        def look_at(self, u, v):
            self.looks.append(("at", u, v))

    camera = Camera()
    preview = Preview(camera, stream_width=320, fps=10.0, capture_dir=tmp_path / "board")
    preview.board = BoardReference(image=board, min_inliers=20)
    preview.start()
    server = serve(preview, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def post(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            base + path, data=data, headers={"content-type": "application/json"}, method="POST"
        )
        return json.loads(urllib.request.urlopen(req).read())

    def wait_scan():
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not preview.scan_state.startswith("scan done"):
            time.sleep(0.1)
        preview._scan_thread.join(5.0)
        return json.loads(urllib.request.urlopen(base + "/scan_result").read())

    try:
        preview.wait_jpeg(0, 3.0)
        assert "error" in post("/detect")  # no baseline yet
        assert post("/scan", {"mode": "baseline"})["ok"]
        learned = wait_scan()
        assert (
            learned["ok"]
            and learned["mode"] == "baseline"
            and sorted(preview.baselines.views) == ["centre", "left", "right"]
        )
        assert (preview.baseline_dir() / "views.json").exists() and "London" in learned["seen"]
        assert camera.looks[-1] == (35.0, 0.0, 0.0)  # back to the centre at the end
        current["frame"] = busy
        preview.wait_jpeg(preview.frames, 3.0)
        assert post("/scan", {"mode": "detect"})["ok"]
        found = wait_scan()
        assert found["ok"] and len(found["pieces"]) == 1, found
        piece = found["pieces"][0]
        assert piece["space"] == "Rome" and sorted(piece["views"]) == ["centre", "left", "right"]
        assert [r["slot"] for r in found["reserve"]] == [1, 2, 3, 4] and not any(
            r["occupied"] for r in found["reserve"]
        )
        assert piece["confirmed"] is True  # with the margin forced high every piece gets a closer look
        assert piece["name"] is None  # no gallery wired in this test
        # Labelling a crop shown on the page puts it in the gallery; a later scan names the piece.
        from src.vision.gallery import Gallery
        from tests.test_gallery import ColourEmbedder

        preview.gallery = Gallery(tmp_path / "gallery", embedder=ColourEmbedder(), threshold=0.9)
        saved = post("/label", {"crop": piece["crop"], "label": "token:red disc"})
        assert saved["ok"] and saved["counts"] == {"token:red disc": 1}
        assert "error" in post("/label", {"crop": "/captures/../../etc/passwd", "label": "x"})
        assert post("/scan", {"mode": "detect"})["ok"]
        named = wait_scan()["pieces"][0]
        assert named["name"] == "token:red disc" and named["name_score"] > 0.9
        assert any(look[0] == "at" for look in camera.looks)
        assert "Rome" in found["text"] and found["centre_pieces"][0]["space"] == "Rome"
        crop = urllib.request.urlopen(base + piece["crop"]).read()
        assert crop[:2] == b"\xff\xd8"
        single = post("/detect")
        assert single["ok"] and single["view"] == "centre" and single["pieces"][0]["space"] == "Rome"
        assert preview.load_baseline()
    finally:
        server.shutdown()
        preview.stop()


def test_moves_json_reports_the_watcher_state_and_the_tail(tmp_path):
    """The board log the page polls: newest first, capped, with the watcher's own state."""
    preview = Preview(FakeCamera(160, 90), capture_dir=tmp_path)
    empty = preview.moves_json()
    assert empty["moves"] == [] and empty["count"] == 0
    assert empty["state"] == "starting"

    preview.motion_state = "watching"
    for i in range(20):
        preview.moves.append(
            {
                "at": float(i),
                "time": f"10:00:{i:02d}",
                "seconds": 1.0,
                "text": f"move {i}",
                "departed": ["Rome"],
                "arrived": ["London"],
            }
        )
    out = preview.moves_json()
    assert out["state"] == "watching" and out["count"] == 20
    assert len(out["moves"]) == 12, "the page is served a bounded tail"
    assert out["moves"][0]["text"] == "move 19", "newest first"
    assert out["moves"][-1]["text"] == "move 8"


def test_the_watcher_is_dropped_while_the_robot_is_moving(tmp_path):
    """A scan or a sweep turns the robot, so the registered view is gone until it stops."""
    preview = Preview(FakeCamera(160, 90), capture_dir=tmp_path)
    assert preview.busy_with_robot() is False

    running = threading.Event()
    thread = threading.Thread(target=running.wait, daemon=True)
    thread.start()
    preview._scan_thread = thread
    try:
        assert preview.busy_with_robot() is True
    finally:
        running.set()
        thread.join(timeout=2)
    assert preview.busy_with_robot() is False


def test_the_page_carries_the_board_log_and_polls_it():
    from src.vision.preview import PAGE

    assert "BOARD LOG" in PAGE and 'id="logitems"' in PAGE
    assert "pollMoves" in PAGE and "'/moves'" in PAGE
    assert "setInterval(pollMoves, 1000)" in PAGE


def test_the_camera_is_never_letterboxed_under_the_overlay():
    """The overlay canvas covers the element, so the picture must fill the element exactly.

    A redesign once set `width: 100%` with `object-fit: contain`, which letterboxes the image on
    a wide window: the canvas kept drawing over the whole element, so every space was pushed
    sideways the further it sat from the centre, and the board spaces looked misregistered when
    the registration was in fact good to about 1.4 px.
    """
    from src.vision.preview import PAGE

    style = PAGE[PAGE.index("<style>") : PAGE.index("</style>")]
    cam_rule = [line for line in style.splitlines() if line.strip().startswith("#cam {")]
    assert cam_rule, "no rule for the camera image"
    assert "object-fit" not in cam_rule[0], "object-fit letterboxes the image under the canvas"
    # The wrapper's width is capped by the height left over, so the image always fills it.
    assert "#wrap {" in style and "min(100%," in style


# --------------------------------------------------------------- a camera that dies under us


class DyingCamera:
    """Frames, then the same object for ever - which is what a dead GStreamer pipeline looks like."""

    def __init__(self, alive: int = 2):
        self.alive = alive
        self.stale = np.zeros((8, 8, 3), dtype=np.uint8)
        self.reopens = 0

    def get_frame(self):
        if self.alive > 0:
            self.alive -= 1
            return np.full((8, 8, 3), self.alive + 1, dtype=np.uint8)
        return self.stale  # the same object: `frame is not last` is False

    def reopen(self) -> bool:
        self.reopens += 1
        self.alive = 2
        return True

    def look(self, *a, **k):
        return None


def test_a_dead_camera_is_reported_and_not_rebuilt_behind_the_owners_back(monkeypatch):
    """The rebuild is off by default, and for a measured reason: on 2026-09-17 one bound to a
    different 1080p camera on this Mac (there are three) and reported success, so the board
    matcher produced a confident homography over a photo of the room and the robot told the
    table there were no pieces. The drought is still logged - that half was always the useful
    one - but nothing silently changes camera."""
    from src.vision.preview import CAMERA_DEAD_S, Preview

    now = [1000.0]
    camera = DyingCamera(alive=0)
    preview = Preview(camera)
    preview.clock = lambda: now[0]
    preview.grab_once()  # the first stale frame counts; the drought starts here

    now[0] += CAMERA_DEAD_S + 1
    assert preview.grab_once() is False
    assert camera.reopens == 0, "not without CAMERA_REBUILD"
    assert preview.reopens == 0


def test_a_repeated_frame_is_not_a_frame_and_the_camera_is_rebuilt(monkeypatch):
    """2026-09-17: talk.py asks the SDK for no_media, and on that branch the SDK tells the
    *daemon* to release the camera this process reads from. The pipeline died mid-startup and
    get_frame went on handing back the same object, so the preview captured one frame in three
    minutes - a dark page, and a scan that reported an empty table instead of no camera."""
    import src.vision.preview as mod
    from src.vision.preview import CAMERA_DEAD_S, Preview

    monkeypatch.setattr(mod, "CAMERA_REBUILD", True)  # opt in, as an owner would
    now = [1000.0]
    camera = DyingCamera(alive=2)
    preview = Preview(camera)
    preview.clock = lambda: now[0]
    preview._fresh_at = now[0]

    assert preview.grab_once() is True and preview.grab_once() is True  # the two live frames
    assert preview.frames == 2

    # The first stale frame cannot be told from a new one, so it counts and the drought starts
    # after it. Only the repeats that follow say the pipeline has stopped.
    assert preview.grab_once() is True
    assert preview.frames == 3

    now[0] += 1.0
    assert preview.grab_once() is False
    assert camera.reopens == 0, "a slow camera is not a broken one"

    now[0] += CAMERA_DEAD_S
    assert preview.grab_once() is False
    assert camera.reopens == 1 and preview.reopens == 1
    assert preview.grab_once() is True  # the rebuilt pipeline delivers again


def test_a_camera_that_will_not_come_back_is_not_hammered(monkeypatch):
    import src.vision.preview as mod
    from src.vision.preview import CAMERA_DEAD_S, CAMERA_RETRY_S, Preview

    monkeypatch.setattr(mod, "CAMERA_REBUILD", True)
    now = [1000.0]

    class Hopeless(DyingCamera):
        def reopen(self) -> bool:
            self.reopens += 1
            return False

    camera = Hopeless(alive=0)
    preview = Preview(camera)
    preview.clock = lambda: now[0]
    preview.grab_once()  # the first stale frame counts; the drought starts here

    now[0] += CAMERA_DEAD_S + 1
    preview.grab_once()
    assert camera.reopens == 1

    now[0] += 1.0  # still dead, but the retry window has not passed
    preview.grab_once()
    assert camera.reopens == 1

    now[0] += CAMERA_RETRY_S
    preview.grab_once()
    assert camera.reopens == 2


def test_the_camera_is_not_rebuilt_while_a_scan_is_turning_the_head():
    """A scan turns the head for seconds at a time and the frames it takes are its own business."""
    from src.vision.preview import CAMERA_DEAD_S, Preview

    now = [1000.0]
    camera = DyingCamera(alive=0)
    preview = Preview(camera)
    preview.clock = lambda: now[0]
    preview.grab_once()
    preview.hold_still(30.0)  # what talk.py does while the voice plays

    now[0] += CAMERA_DEAD_S + 1
    preview.grab_once()
    assert camera.reopens == 0
