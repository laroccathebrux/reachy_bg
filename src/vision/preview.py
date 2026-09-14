"""Live view of the robot's camera in the browser, to place the robot and the board.

    uv run python -m src.vision.preview                 # http://127.0.0.1:8090 , robot looking at the table
    uv run python -m src.vision.preview --fake          # synthetic frames, no robot (plumbing test)
    uv run python -m src.vision.preview --port 8091 --pitch 40

The page shows the camera as an MJPEG stream with framing guides (centre cross, thirds, a
dashed board guide), lets you move the head (pitch and yaw sliders) and save full-resolution
snapshots to ``data/captures/board/``. Frames come from the daemon through the SDK's local
IPC backend; the daemon must be allowed to use the camera (macOS: System Settings > Privacy
& Security > Camera for the app that launched it; its log says "video access permission has
been denied" otherwise).

Connection note: the SDK's "auto" and "localhost_only" modes dial ``localhost``, which
resolves to ``::1`` first; when another service publishes port 8000 on IPv6 (a Docker
container did) that attempt gets a 403 and "auto" falls back to the network mode. The
preview therefore connects to ``REACHY_HOST`` (127.0.0.1) in network mode and asks for the
``local`` media backend explicitly.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from src.config import CAPTURE_DIR, REACHY_HOST, REACHY_PORT
from src.logger import get_logger

log = get_logger(__name__)

STREAM_WIDTH = 960  # the stream is downscaled; snapshots keep the full resolution
STREAM_FPS = 10.0
BOUNDARY = "reachyframe"
BOARD_CAPTURE_DIR = CAPTURE_DIR / "board"
TABLE_PITCH = 35.0

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Reachy camera</title>
<style>
  body { margin: 0; background: #111; color: #ddd; font: 14px system-ui, sans-serif; }
  #wrap { position: relative; display: inline-block; max-width: 100vw; }
  #cam { display: block; max-width: 100vw; max-height: 80vh; }
  #guides { position: absolute; left: 0; top: 0; pointer-events: none; }
  #bar { padding: 8px 12px; display: flex; flex-wrap: wrap; gap: 14px; align-items: center; }
  #bar label { display: flex; gap: 6px; align-items: center; }
  button { background: #333; color: #eee; border: 1px solid #666; padding: 4px 10px; cursor: pointer; }
  #msg { padding: 0 12px 8px; color: #f6c343; white-space: pre-wrap; }
  #info { color: #9ad; }
</style>
</head>
<body>
<div id="wrap">
  <img id="cam" src="/stream" alt="camera">
  <canvas id="guides"></canvas>
</div>
<div id="bar">
  <span id="info">connecting...</span>
  <label>pitch <input id="pitch" type="range" min="-10" max="60" value="__PITCH__"> <span id="pitchv"></span>&deg;</label>
  <label>yaw <input id="yaw" type="range" min="-45" max="45" value="0"> <span id="yawv"></span>&deg;</label>
  <button id="table">Look at the table</button>
  <button id="snap">Snapshot</button>
  <label><input id="grid" type="checkbox" checked> guides</label>
  <label>board guide margin <input id="margin" type="range" min="0" max="30" value="8">%</label>
</div>
<div id="msg"></div>
<script>
const cam = document.getElementById('cam'), canvas = document.getElementById('guides');
const info = document.getElementById('info'), msg = document.getElementById('msg');
const pitch = document.getElementById('pitch'), yaw = document.getElementById('yaw');
const pitchv = document.getElementById('pitchv'), yawv = document.getElementById('yawv');
const grid = document.getElementById('grid'), margin = document.getElementById('margin');
function draw() {
  const w = cam.clientWidth, h = cam.clientHeight;
  if (!w || !h) return;
  canvas.width = w; canvas.height = h;
  const c = canvas.getContext('2d');
  c.clearRect(0, 0, w, h);
  if (!grid.checked) return;
  c.strokeStyle = 'rgba(255,255,255,0.45)'; c.lineWidth = 1;
  for (const f of [1/3, 2/3]) {
    c.beginPath(); c.moveTo(w*f, 0); c.lineTo(w*f, h); c.stroke();
    c.beginPath(); c.moveTo(0, h*f); c.lineTo(w, h*f); c.stroke();
  }
  c.strokeStyle = 'rgba(255,80,80,0.9)'; c.lineWidth = 2;
  c.beginPath(); c.moveTo(w/2-20, h/2); c.lineTo(w/2+20, h/2); c.stroke();
  c.beginPath(); c.moveTo(w/2, h/2-20); c.lineTo(w/2, h/2+20); c.stroke();
  const m = margin.value / 100;
  c.setLineDash([10, 8]); c.strokeStyle = 'rgba(80,220,120,0.9)';
  c.strokeRect(w*m, h*m, w*(1-2*m), h*(1-2*m));
  c.setLineDash([]);
  c.fillStyle = 'rgba(80,220,120,0.9)'; c.font = '13px system-ui';
  c.fillText('board should fill the dashed area', w*m + 6, h*m + 16);
}
async function look() {
  pitchv.textContent = pitch.value; yawv.textContent = yaw.value;
  await fetch('/look', {method: 'POST', headers: {'content-type': 'application/json'},
                        body: JSON.stringify({pitch: +pitch.value, yaw: +yaw.value})});
}
let lookTimer = null;
function lookSoon() { clearTimeout(lookTimer); pitchv.textContent = pitch.value; yawv.textContent = yaw.value; lookTimer = setTimeout(look, 150); }
pitch.oninput = lookSoon; yaw.oninput = lookSoon;
document.getElementById('table').onclick = () => { pitch.value = __PITCH__; yaw.value = 0; look(); };
document.getElementById('snap').onclick = async () => {
  const r = await fetch('/snapshot', {method: 'POST'}); const j = await r.json();
  msg.textContent = j.path ? 'saved ' + j.path : (j.error || 'no frame yet');
};
grid.onchange = draw; margin.oninput = draw;
new ResizeObserver(draw).observe(cam);
cam.onload = draw;
async function poll() {
  try {
    const s = await (await fetch('/status')).json();
    info.textContent = s.frames ? `${s.width}x${s.height}  ${s.fps.toFixed(1)} fps  frame age ${s.age_s.toFixed(1)} s  head pitch ${s.pitch} yaw ${s.yaw}` : `waiting for frames (${s.waited_s.toFixed(0)} s)`;
    msg.textContent = s.warning || msg.textContent;
    if (s.warning) msg.textContent = s.warning;
  } catch (e) { info.textContent = 'server unreachable'; }
}
pitchv.textContent = pitch.value; yawv.textContent = yaw.value;
setInterval(poll, 1000); poll(); draw();
</script>
</body>
</html>
"""

NO_FRAMES_HINT = (
    "No frames from the daemon after {seconds:.0f} s. If its log (data/daemon.log) says "
    "'video access permission has been denied', allow the camera for the app that launched the "
    "daemon (System Settings > Privacy & Security > Camera) and restart the daemon."
)


def encode_jpeg(frame_bgr: np.ndarray, *, max_width: int | None = None, quality: int = 80) -> bytes:
    """JPEG bytes of a BGR frame, optionally downscaled to ``max_width``."""
    from PIL import Image

    image = Image.fromarray(np.ascontiguousarray(frame_bgr[:, :, ::-1]))
    if max_width and image.width > max_width:
        image = image.resize((max_width, round(image.height * max_width / image.width)))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


class FakeCamera:
    """Synthetic frames (a moving pattern) so the page and the server can be tested without a robot."""

    def __init__(self, width: int = 640, height: int = 360):
        self.width, self.height = width, height
        self.pitch, self.yaw = TABLE_PITCH, 0.0
        self.started = time.monotonic()
        self.looks: list[tuple[float, float]] = []

    def get_frame(self) -> np.ndarray | None:
        t = time.monotonic() - self.started
        y, x = np.mgrid[0 : self.height, 0 : self.width]
        wave = (np.sin(x / 40 + t * 2) + np.cos(y / 30 - t)) * 60 + 128
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :, 0] = wave.astype(np.uint8)
        frame[:, :, 1] = (x * 255 // max(1, self.width)).astype(np.uint8)
        frame[:, :, 2] = (y * 255 // max(1, self.height)).astype(np.uint8)
        return frame

    def look(self, pitch: float, yaw: float) -> None:
        self.pitch, self.yaw = pitch, yaw
        self.looks.append((pitch, yaw))

    def close(self) -> None:
        pass


class RobotCamera:
    """Frames and head control through the daemon (SDK local IPC backend)."""

    def __init__(self, host: str = REACHY_HOST, port: int = REACHY_PORT, timeout: float = 10.0):
        from reachy_mini import ReachyMini

        from src.robot.reachy import Robot

        self._mini = ReachyMini(
            host=host, port=port, connection_mode="network", media_backend="local", timeout=timeout
        )
        self._mini.__enter__()
        self.robot = Robot(self._mini)
        self.robot.wake()
        self.pitch, self.yaw = 0.0, 0.0

    def get_frame(self) -> np.ndarray | None:
        return self._mini.media.get_frame()

    def look(self, pitch: float, yaw: float) -> None:
        self.robot.look(pitch=pitch, yaw=yaw, duration=0.6)
        self.pitch, self.yaw = pitch, yaw

    def close(self) -> None:
        try:
            self.robot.rest()
        finally:
            self._mini.__exit__(None, None, None)


class Preview:
    """Grabs frames from a camera on a thread and keeps the latest JPEG for the HTTP handlers."""

    def __init__(
        self,
        camera: Any,
        *,
        stream_width: int = STREAM_WIDTH,
        fps: float = STREAM_FPS,
        capture_dir: Path = BOARD_CAPTURE_DIR,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.camera = camera
        self.stream_width = stream_width
        self.fps = fps
        self.capture_dir = capture_dir
        self.clock = clock
        self.jpeg: bytes = b""
        self.frame: np.ndarray | None = None
        self.frame_at = 0.0
        self.frames = 0
        self.started = clock()
        self._times: list[float] = []
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> Preview:
        self._thread = threading.Thread(target=self._grab_loop, name="camera-grab", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _grab_loop(self) -> None:
        period = 1.0 / self.fps
        last = None
        while not self._stop.is_set():
            started = self.clock()
            try:
                frame = self.camera.get_frame()
            except Exception as exc:
                log.warning("get_frame failed: %s", exc)
                frame = None
            if frame is not None and frame is not last:
                last = frame
                self.publish(frame)
            time.sleep(max(0.0, period - (self.clock() - started)))

    def publish(self, frame: np.ndarray) -> None:
        jpeg = encode_jpeg(frame, max_width=self.stream_width)
        now = self.clock()
        with self._changed:
            self.frame, self.jpeg, self.frame_at = frame, jpeg, now
            self.frames += 1
            self._times = [t for t in self._times if now - t <= 3.0] + [now]
            self._changed.notify_all()

    def wait_jpeg(self, after: int, timeout: float) -> tuple[int, bytes]:
        """Block until a frame newer than sequence ``after`` exists (or ``timeout``)."""
        with self._changed:
            if self.frames <= after:
                self._changed.wait(timeout)
            return self.frames, self.jpeg

    def status(self) -> dict[str, Any]:
        now = self.clock()
        with self._lock:
            times = [t for t in self._times if now - t <= 3.0]
            frame = self.frame
            status: dict[str, Any] = {
                "frames": self.frames,
                "fps": len(times) / 3.0,
                "age_s": (now - self.frame_at) if self.frames else 0.0,
                "waited_s": now - self.started,
                "width": 0 if frame is None else int(frame.shape[1]),
                "height": 0 if frame is None else int(frame.shape[0]),
                "pitch": getattr(self.camera, "pitch", None),
                "yaw": getattr(self.camera, "yaw", None),
                "warning": "",
            }
        if not self.frames and status["waited_s"] > 8.0:
            status["warning"] = NO_FRAMES_HINT.format(seconds=status["waited_s"])
        return status

    def snapshot(self) -> Path | None:
        with self._lock:
            frame = self.frame
        if frame is None:
            return None
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        path = self.capture_dir / f"board_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        path.write_bytes(encode_jpeg(frame, quality=92))
        log.info("snapshot %s (%dx%d)", path, frame.shape[1], frame.shape[0])
        return path

    def look(self, pitch: float, yaw: float) -> None:
        pitch = max(-10.0, min(60.0, float(pitch)))
        yaw = max(-45.0, min(45.0, float(yaw)))
        if any(math.isnan(v) for v in (pitch, yaw)):
            raise ValueError("pitch and yaw must be numbers")
        self.camera.look(pitch, yaw)


def make_handler(preview: Preview, table_pitch: float = TABLE_PITCH) -> type[BaseHTTPRequestHandler]:
    page = PAGE.replace("__PITCH__", str(int(table_pitch))).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # quiet: one line per frame otherwise
            if not self.path.startswith(("/stream", "/status", "/frame")):
                log.info("%s %s", self.command, self.path)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict[str, Any], status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path.startswith("/status"):
                self._json(preview.status())
            elif self.path.startswith("/frame"):
                _, jpeg = preview.wait_jpeg(0, 2.0)
                if jpeg:
                    self._send(200, jpeg, "image/jpeg")
                else:
                    self._json({"error": "no frame yet"}, 503)
            elif self.path.startswith("/stream"):
                self._stream()
            else:
                self._send(404, b"not found", "text/plain")

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seen = 0
            try:
                while True:
                    seq, jpeg = preview.wait_jpeg(seen, 1.0)
                    if seq == seen or not jpeg:
                        continue
                    seen = seq
                    self.wfile.write(
                        f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            if self.path.startswith("/look"):
                try:
                    body = json.loads(raw or b"{}")
                    preview.look(float(body.get("pitch", 0.0)), float(body.get("yaw", 0.0)))
                except (ValueError, TypeError) as exc:
                    self._json({"error": str(exc)}, 400)
                    return
                except Exception as exc:
                    log.warning("look failed: %s", exc)
                    self._json({"error": str(exc)}, 500)
                    return
                self._json({"ok": True, "pitch": preview.camera.pitch, "yaw": preview.camera.yaw})
            elif self.path.startswith("/snapshot"):
                path = preview.snapshot()
                self._json({"path": str(path)} if path else {"error": "no frame yet"})
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


def serve(preview: Preview, host: str = "127.0.0.1", port: int = 8090, table_pitch: float = TABLE_PITCH):
    server = ThreadingHTTPServer((host, port), make_handler(preview, table_pitch))
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--fake", action="store_true", help="synthetic frames, no robot")
    parser.add_argument(
        "--pitch", type=float, default=TABLE_PITCH, help="initial head pitch (down is positive)"
    )
    parser.add_argument("--no-look", action="store_true", help="do not move the head at start")
    args = parser.parse_args(argv)

    if args.fake:
        camera: Any = FakeCamera()
    else:
        try:
            camera = RobotCamera()
        except Exception as exc:
            log.error("cannot connect to the robot: %s", exc)
            return 1
    if not args.no_look:
        try:
            camera.look(args.pitch, 0.0)
        except Exception as exc:
            log.warning("could not move the head: %s", exc)
    preview = Preview(camera).start()
    server = serve(preview, port=args.port, table_pitch=args.pitch)
    log.info("camera preview at http://127.0.0.1:%d  (Ctrl+C to stop)", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        preview.stop()
        camera.close()
    return 0


__all__ = ["Preview", "FakeCamera", "RobotCamera", "encode_jpeg", "make_handler", "serve", "NO_FRAMES_HINT"]


if __name__ == "__main__":
    sys.exit(main())
