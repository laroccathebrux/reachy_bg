"""The camera preview server on synthetic frames: page, status, JPEG frame, MJPEG stream, head control."""

import json
import threading
import urllib.request

import numpy as np

from src.vision.preview import BOUNDARY, FakeCamera, Preview, encode_jpeg, serve


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
        assert reply == {"ok": True, "pitch": 60.0, "yaw": -12.0}  # pitch clamped
        assert camera.looks[-1] == (60.0, -12.0)
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
