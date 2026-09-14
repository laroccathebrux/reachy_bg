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


def test_baseline_and_detect_endpoints(tmp_path):
    cv2 = pytest.importorskip("cv2")
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

    class Camera:
        pitch = yaw = body = 0.0

        def get_frame(self):
            return current["frame"].copy()

    preview = Preview(Camera(), stream_width=320, fps=10.0, capture_dir=tmp_path / "board")
    preview.board = BoardReference(image=board, min_inliers=20)
    preview.start()
    server = serve(preview, port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        preview.wait_jpeg(0, 3.0)

        def post(path):
            return json.loads(
                urllib.request.urlopen(urllib.request.Request(base + path, method="POST")).read()
            )

        assert "error" in post("/detect")  # no baseline yet
        answer = post("/baseline")
        assert answer["ok"] and preview.baseline_path().exists()
        current["frame"] = busy
        preview.wait_jpeg(preview.frames, 3.0)
        time.sleep(0.3)
        found = post("/detect")
        assert found["ok"] and found["pieces"], found
        assert found["pieces"][0]["space"] == "Rome" and "Rome" in found["text"]
        crop = urllib.request.urlopen(base + found["pieces"][0]["crop"]).read()
        assert crop[:2] == b"\xff\xd8"
        assert preview.load_baseline()
    finally:
        server.shutdown()
        preview.stop()
