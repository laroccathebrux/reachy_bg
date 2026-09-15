"""Live view of the robot's camera in the browser, to place the robot and the board.

    uv run python -m src.vision.preview                 # http://127.0.0.1:8090 , robot looking at the table
    uv run python -m src.vision.preview --fake          # synthetic frames, no robot (plumbing test)
    uv run python -m src.vision.preview --port 8091 --pitch 40

The page shows the camera as an MJPEG stream with framing guides (centre cross, thirds, a
dashed board guide), lets you move the head (pitch and yaw sliders) and save full-resolution
snapshots to ``data/captures/board/``. Frames come from the daemon through the SDK's local
IPC backend; on macOS the daemon must have been started from a terminal application that is
allowed to use the camera (iTerm, Terminal). A daemon started from a Claude Code session gets
"video access permission has been denied" in its log and never delivers a frame.

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
# name, body yaw in degrees: the Reserve (bottom-left card slots) needs its own view, centred at 75.
# name, body yaw in degrees. The Reserve (bottom-left card slots) is read from the left view:
# a view turned further (60-75) shows too much room to register, and the base stalls at 64.
SCAN_VIEWS = (("centre", 0.0), ("left", 45.0), ("right", -45.0))
RESERVE_VIEW = "left"
SCAN_SETTLE_S = 0.6
BODY_YAW_MAX = 90.0  # degrees either way; the base turns further but the table is in front
BODY_YAW_SPEED = 60.0  # deg/s asked of the base when turning between views
FRAME_WIDTH, FRAME_HEIGHT = 1920, 1080
LOOK_PX_PER_DEG_BODY = 17.5  # measured: frame pixels a map point moves per degree of body yaw
LOOK_CENTRE_TOLERANCE = 250.0  # px (horizontal): a closer look is refined once beyond this
LOOK_RAISE_DEG = 8.0  # head raised by this much for a closer look at the far edge of the board
MAX_CLOSER_LOOKS = 4  # per scan: bounds the time when the board is full of ambiguous blobs
MIN_VIEW_INLIERS = (
    30  # a scan view registered with fewer inliers is not trusted (the left view gives 35-53 in daylight)
)

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
  <label>body <input id="body" type="range" min="-90" max="90" value="0"> <span id="bodyv"></span>&deg;</label>
  <button id="table">Look at the table</button>
  <button id="snap">Snapshot</button>
  <label>zoom <input id="zoom" type="range" min="1" max="4" step="0.5" value="1"> <span id="zoomv"></span>x (click the image to centre)</label>
  <button id="unzoom">Reset zoom</button>
  <label>sweep body angles <input id="angles" type="text" value="60,30,0,-30,-60" size="16"></label>
  <button id="sweep">Sweep</button>
  <a href="/gallery" target="_blank" style="color:#9ad">gallery</a>
  <label><input id="grid" type="checkbox" checked> guides</label>
  <label><input id="overlay" type="checkbox" checked> board overlay</label>
  <a href="/rectified.jpg" target="_blank" style="color:#9ad">top-down view</a>
  <button id="baseline">Baseline sweep (empty board)</button>
  <button id="scan">Scan the board</button>
</div>
<div id="pieces" style="padding: 0 12px 8px; display: flex; flex-wrap: wrap; gap: 10px;"></div>
<div id="bar2" style="display:none">
  <label>board guide margin <input id="margin" type="range" min="0" max="30" value="8">%</label>
</div>
<div id="msg"></div>
<script>
const cam = document.getElementById('cam'), canvas = document.getElementById('guides');
const info = document.getElementById('info'), msg = document.getElementById('msg');
const pitch = document.getElementById('pitch'), yaw = document.getElementById('yaw');
const pitchv = document.getElementById('pitchv'), yawv = document.getElementById('yawv');
const grid = document.getElementById('grid'), margin = document.getElementById('margin');
const body = document.getElementById('body'), bodyv = document.getElementById('bodyv');
const overlay = document.getElementById('overlay');
let board = null;  // last /board answer: outline and spaces in frame fractions
let found = [];    // last /detect answer: pieces with frame boxes in fractions
let scanning = false;
let lastShown = '';  // the scan result already rendered (a reload in the middle of a scan must not lose it)
function labelForm(crop, current) {
  const id = 'lbl' + Math.random().toString(36).slice(2, 8);
  return `<div style="font-size:12px"><input id="${id}" placeholder="kind:name, e.g. investigator:Akachi Onyele" value="${current || ''}" size="22"> <button onclick="saveLabel('${crop}', '${id}')">Save</button></div>`;
}
async function saveLabel(crop, id) {
  const label = document.getElementById(id).value.trim();
  if (!label) return;
  const j = await (await fetch('/label', {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify({crop, label})})).json();
  msg.textContent = j.error ? j.error : `saved as ${j.label}; gallery now: ` + Object.entries(j.counts).map(([k, v]) => `${k} x${v}`).join(', ');
}
async function startScan(mode) {
  const j = await (await fetch('/scan', {method: 'POST', headers: {'content-type': 'application/json'},
                                        body: JSON.stringify({mode, pitch: +pitch.value})})).json();
  msg.textContent = j.error || `${mode} scan started: centre, left, right`;
  scanning = !j.error;
}
document.getElementById('baseline').onclick = () => startScan('baseline');
document.getElementById('scan').onclick = () => startScan('detect');
async function showScan() {
  const j = await (await fetch('/scan_result')).json();
  if (!j.ok) return;
  found = j.centre_pieces || [];
  const skipped = (j.views || []).filter(v => v.error).map(v => `${v.view}: ${v.error}`);
  msg.textContent = j.text + (j.unseen.length ? `  |  not covered by any view: ${j.unseen.join(', ')}` : '  |  every space covered') + (skipped.length ? `  |  skipped ${skipped.join('; ')}` : '');
  const box = document.getElementById('pieces'); box.innerHTML = '';
  for (const r of (j.reserve || [])) {
    const fig = document.createElement('figure'); fig.style.margin = '0';
    const label = !r.seen ? 'not in view' : (r.occupied ? 'card' : 'empty');
    const named = r.occupied && r.name ? ` ${r.name.replace(':', ' ')} (${Math.round(r.name_score * 100)}%)` : (r.occupied ? ' unknown card' : '');
    fig.innerHTML = `<img src="${r.crop || ''}" style="height:160px;border:2px solid ${r.occupied ? '#fd5' : '#555'};display:block"><figcaption style="font-size:12px;color:#ccc">Reserve slot ${r.slot}: ${label}${named} (${Math.round(r.fraction * 100)}%)</figcaption>` + (r.occupied && r.crop ? labelForm(r.crop, r.name) : '');
    box.appendChild(fig);
  }
  for (const p of j.pieces) {
    const fig = document.createElement('figure'); fig.style.margin = '0';
    let where = p.space || (p.near ? 'near ' + p.near : 'between spaces');
    if (p.kind === 'die') where = `die showing ${p.value ?? '?'}${p.kind_confidence < 0.6 ? ' (unsure)' : ''} at ${where}`;
    else if (p.name) where = `${p.name.replace(':', ' ')} (${Math.round(p.name_score * 100)}%) at ${where}`;
    else where = `unknown piece at ${where}`;
    fig.innerHTML = `<img src="${p.crop}" style="height:160px;border:1px solid #555;display:block;cursor:zoom-in"><figcaption style="font-size:12px;color:#ccc">${where} (seen from ${p.views.join(', ')}${p.confirmed ? ', confirmed by a closer look' : ''})</figcaption>` + (p.kind === 'die' ? '' : labelForm(p.crop, p.name));
    fig.querySelector('img').onclick = () => { if (p.views.includes('centre') && p.box) { zoom.value = 3; setZoom(3, (p.box[0] + p.box[2]) / 2, (p.box[1] + p.box[3]) / 2); } };
    box.appendChild(fig);
  }
  draw();
}
const zoom = document.getElementById('zoom'), zoomv = document.getElementById('zoomv');
let crop = {factor: 1, cx: 0.5, cy: 0.5};
async function setZoom(factor, cx, cy) {
  crop = {factor, cx, cy}; zoomv.textContent = factor;
  await fetch('/zoom', {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(crop)});
}
zoom.oninput = () => setZoom(+zoom.value, crop.cx, crop.cy);
document.getElementById('unzoom').onclick = () => { zoom.value = 1; setZoom(1, 0.5, 0.5); };
cam.onclick = (e) => {
  const r = cam.getBoundingClientRect();
  const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
  const size = 1 / crop.factor;  // the visible crop as a fraction of the frame
  const x0 = Math.min(Math.max(crop.cx - size / 2, 0), 1 - size), y0 = Math.min(Math.max(crop.cy - size / 2, 0), 1 - size);
  setZoom(+zoom.value, x0 + fx * size, y0 + fy * size);
};
function draw() {
  const w = cam.clientWidth, h = cam.clientHeight;
  if (!w || !h) return;
  canvas.width = w; canvas.height = h;
  const c = canvas.getContext('2d');
  c.clearRect(0, 0, w, h);
  if (!grid.checked) { drawBoard(); return; }
  c.strokeStyle = 'rgba(255,255,255,0.45)'; c.lineWidth = 1;
  for (const f of [1/3, 2/3]) {
    c.beginPath(); c.moveTo(w*f, 0); c.lineTo(w*f, h); c.stroke();
    c.beginPath(); c.moveTo(0, h*f); c.lineTo(w, h*f); c.stroke();
  }
  c.strokeStyle = 'rgba(255,80,80,0.9)'; c.lineWidth = 2;
  c.beginPath(); c.moveTo(w/2-20, h/2); c.lineTo(w/2+20, h/2); c.stroke();
  c.beginPath(); c.moveTo(w/2, h/2-20); c.lineTo(w/2, h/2+20); c.stroke();
  drawBoard();
  const m = margin.value / 100;
  c.setLineDash([10, 8]); c.strokeStyle = 'rgba(80,220,120,0.9)';
  c.strokeRect(w*m, h*m, w*(1-2*m), h*(1-2*m));
  c.setLineDash([]);
  c.fillStyle = 'rgba(80,220,120,0.9)'; c.font = '13px system-ui';
  c.fillText('board should fill the dashed area', w*m + 6, h*m + 16);
}
function toView(p) {  // frame fraction -> canvas pixels, through the current zoom crop
  const size = 1 / crop.factor;
  const x0 = Math.min(Math.max(crop.cx - size / 2, 0), 1 - size), y0 = Math.min(Math.max(crop.cy - size / 2, 0), 1 - size);
  return [(p[0] - x0) / size * canvas.width, (p[1] - y0) / size * canvas.height];
}
function drawBoard() {
  const c = canvas.getContext('2d');
  for (const p of found) {
    const [x0, y0] = toView([p.box[0], p.box[1]]), [x1, y1] = toView([p.box[2], p.box[3]]);
    c.strokeStyle = '#f44'; c.lineWidth = 3; c.strokeRect(x0, y0, x1 - x0, y1 - y0);
    c.fillStyle = '#f44'; c.font = 'bold 14px system-ui'; c.fillText(p.space || (p.near ? 'near ' + p.near : '?'), x0, y0 - 4);
  }
  if (!overlay.checked || !board || !board.inliers) return;
  c.strokeStyle = 'rgba(0,220,80,0.9)'; c.lineWidth = 2; c.beginPath();
  board.outline.forEach((p, i) => { const [x, y] = toView(p); i ? c.lineTo(x, y) : c.moveTo(x, y); });
  c.closePath(); c.stroke();
  c.font = '12px system-ui';
  for (const s of board.spaces) {
    const [x, y] = toView([s.x, s.y]);
    c.strokeStyle = s.kind === 'sea' ? '#4af' : (s.kind === 'wilderness' ? '#6d6' : '#fd5');
    c.beginPath(); c.arc(x, y, 7, 0, 2 * Math.PI); c.stroke();
    c.fillStyle = 'rgba(0,0,0,0.6)'; c.fillRect(x + 9, y - 14, c.measureText(s.name).width + 6, 16);
    c.fillStyle = '#fff'; c.fillText(s.name, x + 12, y - 2);
  }
}
function labels() { pitchv.textContent = pitch.value; yawv.textContent = yaw.value; bodyv.textContent = body.value; zoomv.textContent = zoom.value; }
async function look() {
  labels();
  await fetch('/look', {method: 'POST', headers: {'content-type': 'application/json'},
                        body: JSON.stringify({pitch: +pitch.value, yaw: +yaw.value, body: +body.value})});
}
let lookTimer = null;
function lookSoon() { clearTimeout(lookTimer); labels(); lookTimer = setTimeout(look, 150); }
pitch.oninput = lookSoon; yaw.oninput = lookSoon; body.oninput = lookSoon;
document.getElementById('table').onclick = () => { pitch.value = __PITCH__; yaw.value = 0; body.value = 0; look(); };
document.getElementById('sweep').onclick = async () => {
  const angles = document.getElementById('angles').value.split(',').map(Number).filter(n => !isNaN(n));
  const r = await fetch('/sweep', {method: 'POST', headers: {'content-type': 'application/json'},
                                  body: JSON.stringify({body_yaws: angles, pitch: +pitch.value})});
  const j = await r.json();
  msg.textContent = j.error || ('sweep started: ' + angles.join(', ') + ' (see gallery when done)');
};
document.getElementById('snap').onclick = async () => {
  const r = await fetch('/snapshot', {method: 'POST'}); const j = await r.json();
  msg.textContent = j.path ? 'saved ' + j.path : (j.error || 'no frame yet');
};
grid.onchange = draw; margin.oninput = draw; overlay.onchange = draw;
async function pollBoard() {
  try { board = await (await fetch('/board')).json(); } catch (e) { board = null; }
  draw();
}
setInterval(pollBoard, 1000);
new ResizeObserver(draw).observe(cam);
cam.onload = draw;
async function poll() {
  try {
    const s = await (await fetch('/status')).json();
    info.textContent = s.frames ? `${s.width}x${s.height}  ${s.fps.toFixed(1)} fps  frame age ${s.age_s.toFixed(1)} s  head pitch ${s.pitch} yaw ${s.yaw} body ${s.body}  zoom ${s.zoom}x` : `waiting for frames (${s.waited_s.toFixed(0)} s)`;
    if (s.warning) msg.textContent = s.warning;
    if (s.sweep) msg.textContent = s.sweep;
    if (s.scan) { msg.textContent = s.scan; if (s.scan.startsWith('scan done') && s.scan !== lastShown) { lastShown = s.scan; scanning = false; showScan(); } }
    if (s.board !== undefined) info.textContent += s.board ? `  board: ${s.board} inliers` : '  board: not found';
  } catch (e) { info.textContent = 'server unreachable'; }
}
labels();
setInterval(poll, 1000); poll(); draw();
</script>
</body>
</html>
"""

GALLERY = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Reachy sweeps</title>
<style>
  body { margin: 0; padding: 12px; background: #111; color: #ddd; font: 14px system-ui, sans-serif; }
  h2 { margin: 18px 0 6px; font-size: 16px; color: #9ad; }
  .row { display: flex; flex-wrap: wrap; gap: 10px; }
  figure { margin: 0; }
  figure img { width: 440px; max-width: 95vw; display: block; border: 1px solid #444; }
  figcaption { font-size: 12px; color: #aaa; padding: 3px 0; }
  a { color: #ddd; }
</style></head><body>
<p><a href="/">back to the camera</a></p>
__SWEEPS__
</body></html>
"""

NO_FRAMES_HINT = (
    "No frames from the daemon after {seconds:.0f} s. If its log (data/daemon.log) says "
    "'video access permission has been denied', restart reachy-mini-daemon from a terminal "
    "application that has the camera permission (iTerm or Terminal), not from a Claude Code session."
)


def crop_frame(frame: np.ndarray, factor: float, cx: float, cy: float) -> np.ndarray:
    """The ``1/factor`` window of ``frame`` centred at (``cx``, ``cy``) in [0, 1], kept inside the frame."""
    factor = max(1.0, float(factor))
    if factor == 1.0:
        return frame
    h, w = frame.shape[:2]
    cw, ch = max(16, int(round(w / factor))), max(9, int(round(h / factor)))
    x0 = int(round(min(max(cx * w - cw / 2, 0), w - cw)))
    y0 = int(round(min(max(cy * h - ch / 2, 0), h - ch)))
    return frame[y0 : y0 + ch, x0 : x0 + cw]


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
        self.pitch, self.yaw, self.body = TABLE_PITCH, 0.0, 0.0
        self.started = time.monotonic()
        self.looks: list[tuple[float, float, float | None]] = []

    def get_frame(self) -> np.ndarray | None:
        t = time.monotonic() - self.started
        y, x = np.mgrid[0 : self.height, 0 : self.width]
        wave = (np.sin(x / 40 + t * 2) + np.cos(y / 30 - t)) * 60 + 128
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :, 0] = wave.astype(np.uint8)
        frame[:, :, 1] = (x * 255 // max(1, self.width)).astype(np.uint8)
        frame[:, :, 2] = (y * 255 // max(1, self.height)).astype(np.uint8)
        return frame

    def look(self, pitch: float, yaw: float, body_yaw: float | None = None) -> None:
        self.pitch, self.yaw = pitch, yaw
        if body_yaw is not None:
            self.body = body_yaw
        self.looks.append((pitch, yaw, body_yaw))

    def look_at(self, u: float, v: float) -> None:
        self.looks.append(("at", float(u), float(v)))

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
        self.pitch, self.yaw, self.body = 0.0, 0.0, 0.0

    def get_frame(self) -> np.ndarray | None:
        return self._mini.media.get_frame()

    def look(self, pitch: float, yaw: float, body_yaw: float | None = None) -> None:
        """Move, giving the base about 60 deg/s, then wait until the body has actually arrived.

        ``goto_target`` returns when its interpolation ends, not when the motors got there;
        a capture taken right away is smeared by the base still turning.
        """
        delta = 0.0 if body_yaw is None else abs(body_yaw - self.body)
        duration = max(0.8, delta / BODY_YAW_SPEED)
        self.robot.look(pitch=pitch, yaw=yaw, body_yaw=body_yaw, duration=duration)
        self.pitch, self.yaw = pitch, yaw
        if body_yaw is not None:
            self.body = body_yaw
            self.wait_for_body(body_yaw)

    def look_at(self, u: float, v: float) -> None:
        """Centre pixel (u, v) of the current frame horizontally by turning the body.

        Measured on the robot (2026-09-14): one degree of body yaw moves a point of the map
        by LOOK_PX_PER_DEG_BODY pixels at 1080p, while one degree of head pitch moves it by
        under 3 pixels (the head tilts around the camera), so the vertical position is left
        alone. The SDK's ``look_at_image`` is not used: it reads the calibration at the
        sensor's full size and resets the body yaw, which sent the head to the ceiling.
        """
        d_yaw = (u - FRAME_WIDTH / 2) / LOOK_PX_PER_DEG_BODY
        body = max(-BODY_YAW_MAX, min(BODY_YAW_MAX, self.body - d_yaw))
        pitch = self.pitch
        if v < FRAME_HEIGHT * 0.25:  # far edge of the board: raise the head a little (owner's suggestion)
            pitch = max(-10.0, self.pitch - LOOK_RAISE_DEG)
        self.look(pitch, 0.0, body)

    def body_angle(self) -> float | None:
        """Measured body yaw in degrees (the first head joint), None when unavailable."""
        try:
            joints, _ = self._mini.get_current_joint_positions()
            return math.degrees(float(joints[0]))
        except Exception:
            return None

    def wait_for_body(self, target: float, tolerance: float = 2.0, timeout: float = 4.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            measured = self.body_angle()
            if measured is None:
                return False
            if abs(measured - target) <= tolerance:
                return True
            time.sleep(0.05)
        log.warning("body yaw did not reach %.0f (at %s)", target, self.body_angle())
        return False

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
        self.sweep_state = ""
        self._sweep_thread: threading.Thread | None = None
        self.zoom = (1.0, 0.5, 0.5)  # digital zoom of the stream only: factor, centre x, centre y
        self.board: Any = None  # BoardReference, when the reference picture exists
        self.gallery: Any = None  # Gallery of labelled crops, when torch is available
        self.baselines: Any = None  # BaselineSet of the empty board, one per scan view
        self.scan_state = ""
        self.last_scan: dict[str, Any] | None = None
        self._scan_thread: threading.Thread | None = None
        self._look_crops: dict[int, str] = {}
        self.registration: Any = None  # latest Registration of the latest frame
        self.registered_at = 0.0
        self._board_thread: threading.Thread | None = None

    def start(self) -> Preview:
        self._thread = threading.Thread(target=self._grab_loop, name="camera-grab", daemon=True)
        self._thread.start()
        if self.board is not None:
            self._board_thread = threading.Thread(target=self._board_loop, name="board-locate", daemon=True)
            self._board_thread.start()
        return self

    def _board_loop(self, period_s: float = 1.0) -> None:
        """Register the latest frame to the board picture about once a second."""
        last_seen = 0
        while not self._stop.is_set():
            time.sleep(0.05)
            with self._lock:
                frame, seq = self.frame, self.frames
            if frame is None or seq == last_seen:
                continue
            last_seen = seq
            started = self.clock()
            try:
                registration = self.board.locate(frame, scale=0.5)
            except Exception as exc:
                log.warning("board registration failed: %s", exc)
                registration = None
            self.registration, self.registered_at = registration, self.clock()
            time.sleep(max(0.0, period_s - (self.clock() - started)))

    def board_json(self) -> dict[str, Any]:
        """Outline and space centres of the last registration, as fractions of the frame."""
        registration = self.registration
        if registration is None:
            return {"inliers": 0, "outline": [], "spaces": []}
        fw, fh = registration.frame_size
        from src.vision.spaces import BY_NAME

        spaces = [
            {"name": name, "kind": BY_NAME[name].kind, "x": x / fw, "y": y / fh}
            for name, (x, y) in registration.space_pixels(margin=40).items()
        ]
        outline = [[float(x) / fw, float(y) / fh] for x, y in registration.outline()]
        return {
            "inliers": registration.inliers,
            "matches": registration.matches,
            "seconds": registration.seconds,
            "age_s": round(self.clock() - self.registered_at, 2),
            "outline": outline,
            "spaces": spaces,
        }

    def baseline_dir(self) -> Path:
        return self.capture_dir.parent / "board_baseline"

    def load_baseline(self) -> bool:
        from src.vision.detect import BaselineSet

        try:
            self.baselines = BaselineSet.load(self.baseline_dir())
        except (FileNotFoundError, OSError, ValueError):
            return False
        return bool(self.baselines.views)

    def _register_now(self) -> tuple[Any, np.ndarray | None]:
        """A fresh full-resolution registration of the latest frame (for baselines and detections)."""
        with self._lock:
            frame = self.frame
        if frame is None or self.board is None:
            return None, None
        return self.board.locate(frame), frame

    def _more_frames(self, first: np.ndarray, count: int = 4) -> list[np.ndarray]:
        """A few more frames of the same view (the median of them removes sensor noise)."""
        extra: list[np.ndarray] = []
        seq = self.frames
        while len(extra) < count:
            seq, _ = self.wait_jpeg(seq, 1.0)
            with self._lock:
                more = self.frame
            if more is None or more is first or (extra and more is extra[-1]):
                break
            extra.append(more)
        return extra

    def _pieces_json(
        self, registration: Any, pieces: list[Any], stamp: str, view: str
    ) -> list[dict[str, Any]]:
        """Records with the frame box (fractions) and the saved crop of every piece."""
        fw, fh = registration.frame_size
        folder = self.capture_dir / "pieces"
        folder.mkdir(parents=True, exist_ok=True)
        out = []
        for i, piece in enumerate(pieces):
            name = f"{stamp}_{view}_{i}_{(piece.space or piece.near or 'between').replace(' ', '_')}.jpg"
            if piece.crop is not None and piece.crop.size:
                (folder / name).write_bytes(encode_jpeg(piece.crop, quality=92))
            x0, y0, x1, y1 = piece.frame_box
            out.append(
                {
                    **piece.record(),
                    "box": [x0 / fw, y0 / fh, x1 / fw, y1 / fh],
                    "crop": f"/captures/pieces/{name}",
                }
            )
        return out

    def detect_pieces(self) -> dict[str, Any]:
        """Find what is on the board in the current view only (the scan does every view)."""
        from src.vision.detect import describe, find_pieces

        view = self.current_view()
        baseline = None if self.baselines is None else self.baselines.views.get(view)
        if baseline is None:
            return {"error": f"no empty-board baseline for the {view} view: run the baseline sweep first"}
        registration, frame = self._register_now()
        if registration is None or frame is None:
            return {"error": "board not registered: is the map in view?"}
        pieces = find_pieces(registration, frame, baseline)
        out = self._pieces_json(registration, pieces, time.strftime("%Y%m%d_%H%M%S"), view)
        text = describe(pieces)
        log.info("pieces (%s view): %s", view, text)
        return {"ok": True, "view": view, "inliers": registration.inliers, "pieces": out, "text": text}

    def _closer_looks(self, merged: list[Any], pitch: float, stamp: str) -> None:
        """Point the camera straight at every ambiguous piece and take the verdict from there.

        At the frame centre the homography is exact (the wide lens distorts the borders) and
        the piece is at its largest, so the base lands on the right space. The comparison uses
        the baseline of the view the piece was seen from (both are in map coordinates).
        """
        from src.vision.capture import capture_sharpest
        from src.vision.detect import DIE_SURE_CONFIDENCE, closest_to, find_pieces

        bodies = dict(SCAN_VIEWS)
        looks = 0
        # Dice first: their value depends on the closer look most, and the looks are capped.
        order = sorted(range(len(merged)), key=lambda i: (merged[i].kind != "die", i))
        for index in order:
            piece = merged[index]
            if not piece.needs_closer_look():
                continue
            if looks >= MAX_CLOSER_LOOKS:
                log.info(
                    "closer looks capped at %d; %s left as seen", MAX_CLOSER_LOOKS, piece.space or piece.near
                )
                continue
            looks += 1
            view = next((v for v in piece.views if v in bodies), None)
            baseline = None if view is None else self.baselines.views.get(view)
            if baseline is None:
                continue
            if piece.kind == "die" and piece.kind_confidence < DIE_SURE_CONFIDENCE:
                why = f"die value {piece.value} at {piece.kind_confidence:.2f}"
            elif piece.space is None:
                why = "between spaces"
            elif piece.edge:
                why = "frame edge"
            else:
                why = f"strength {piece.strength:.0f} (threshold {piece.threshold:.0f})"
            self.scan_state = f"scan: closer look at {piece.space or piece.near or '?'} ({why})"
            try:
                self.camera.look(pitch, 0.0, bodies[view])
                time.sleep(SCAN_SETTLE_S)
                target = piece.frame_base
                frame = registration = None
                for attempt in range(2):  # point, check where the piece landed, correct once
                    self.camera.look_at(*target)
                    time.sleep(SCAN_SETTLE_S)
                    frame, _ = capture_sharpest(self.camera.get_frame, frames=3)
                    registration = None if frame is None else self.board.locate(frame)
                    if registration is None:
                        break
                    scale = registration.reference_size[0] / baseline.image.shape[1]
                    landed = registration.to_frame([(piece.x * scale, piece.y * scale)])[0]
                    off = abs(float(landed[0]) - FRAME_WIDTH / 2)  # horizontal only: pitch cannot centre
                    log.info(
                        "closer look at %s: attempt %d, %d inliers, piece %.0f px from the centre",
                        piece.space or piece.near,
                        attempt + 1,
                        registration.inliers,
                        off,
                    )
                    if off <= LOOK_CENTRE_TOLERANCE:
                        break
                    target = (float(landed[0]), float(landed[1]))
                if frame is not None:  # kept for offline analysis of the closer looks
                    looks_dir = self.capture_dir / "looks"
                    looks_dir.mkdir(parents=True, exist_ok=True)
                    (looks_dir / f"{stamp}_{index}.jpg").write_bytes(encode_jpeg(frame, quality=92))
                if registration is None:
                    log.info("closer look at %s: board not found", piece.space or piece.near)
                    continue
                # Compare with the baseline of the scan view nearest to where the body ended up:
                # the piece sits near that view's frame centre, where its baseline is accurate.
                body_now = float(getattr(self.camera, "body", bodies[view]) or 0.0)
                nearest_view = min(bodies, key=lambda name: abs(bodies[name] - body_now))
                baseline = self.baselines.views.get(nearest_view, baseline)
                candidates = find_pieces(registration, frame, baseline)
                seen_again = closest_to(candidates, piece.x, piece.y)
            except Exception as exc:
                log.warning("closer look failed: %s", exc)
                continue
            if seen_again is None:
                log.info("closer look at %s: nothing there any more", piece.space or piece.near)
                continue
            before = piece.space or f"near {piece.near}"
            for attr in (
                "x",
                "y",
                "width",
                "height",
                "area",
                "strength",
                "space",
                "near",
                "distance",
                "radius_ratio",
                "crop",
            ):
                setattr(piece, attr, getattr(seen_again, attr))
            piece.confirmed = True
            piece.edge = False
            if seen_again.crop is not None and seen_again.crop.size:
                piece.sighting_crops = piece.sighting_crops + [seen_again.crop]
            name = f"{stamp}_look_{index}_{(piece.space or piece.near or 'between').replace(' ', '_')}.jpg"
            if piece.crop is not None and piece.crop.size:
                (self.capture_dir / "pieces").mkdir(parents=True, exist_ok=True)
                (self.capture_dir / "pieces" / name).write_bytes(encode_jpeg(piece.crop, quality=92))
                self._look_crops[id(piece)] = f"/captures/pieces/{name}"
            log.info(
                "closer look: %s -> %s (strength %.0f)",
                before,
                piece.space or f"near {piece.near}",
                piece.strength,
            )

    def _name_pieces(self, pieces: list[Any]) -> None:
        """Name every non-die piece by the gallery (best of its crops, the closer look last)."""
        if self.gallery is None:
            return
        for piece in pieces:
            if piece.kind == "die":
                continue
            crops = [c for c in piece.sighting_crops if c is not None and c.size] or (
                [piece.crop] if piece.crop is not None and piece.crop.size else []
            )
            best = None
            for crop in crops:
                try:
                    found = self.gallery.match(crop)
                except Exception as exc:
                    log.warning("gallery match failed: %s", exc)
                    continue
                if found and (best is None or found.score > best.score):
                    best = found
            if best is not None:
                piece.name, piece.name_score = best.label, best.score

    def label_crop(self, crop_url: str, label: str) -> dict[str, Any]:
        """Add a crop shown on the page (``/captures/...``) to the gallery under ``label``."""
        import cv2

        if self.gallery is None:
            return {"error": "no gallery (torch missing?)"}
        relative = crop_url.split("?")[0].removeprefix("/captures/")
        target = (self.capture_dir / relative).resolve()
        if self.capture_dir.resolve() not in target.parents or not target.exists():
            return {"error": "unknown crop"}
        image = cv2.imread(str(target))
        if image is None:
            return {"error": "unreadable crop"}
        try:
            path = self.gallery.add(label, image, source=relative)
        except ValueError as exc:
            return {"error": str(exc)}
        return {"ok": True, "label": label, "path": str(path), "counts": self.gallery.counts()}

    def current_view(self) -> str:
        """Name of the scan view the body is closest to."""
        body = float(getattr(self.camera, "body", 0.0) or 0.0)
        return min(SCAN_VIEWS, key=lambda v: abs(v[1] - body))[0]

    def start_scan(self, mode: str, pitch: float = TABLE_PITCH) -> bool:
        """Sweep the scan views on a thread: ``baseline`` learns the empty board, ``detect`` finds pieces."""
        if self._scan_thread is not None and self._scan_thread.is_alive():
            return False
        pitch = max(-10.0, min(60.0, float(pitch)))
        self._scan_thread = threading.Thread(target=self._scan, args=(mode, pitch), name="scan", daemon=True)
        self._scan_thread.start()
        return True

    def _scan(self, mode: str, pitch: float) -> None:
        from src.vision.capture import capture_sharpest
        from src.vision.detect import (
            LIGHT_CHANGE_MEDIAN,
            Baseline,
            BaselineSet,
            classify_pieces,
            describe,
            find_pieces,
            light_change,
            merge_pieces,
        )
        from src.vision.spaces import SPACES

        if self.board is None:
            self.scan_state = "scan failed: no board reference picture"
            return
        if mode == "baseline":
            self.baselines = BaselineSet()
        elif self.baselines is None or not self.baselines.views:
            self.scan_state = "scan failed: run the baseline sweep on the empty board first"
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self._look_crops = {}
        candidate_names: dict[int, str | None] = {}  # crop file names use the space at sighting time
        sightings: list[tuple[str, list[Any]]] = []
        views_out: list[dict[str, Any]] = []
        seen: set[str] = set()
        centre_pieces: list[dict[str, Any]] = []
        reserve_out: list[dict[str, Any]] = []
        reserve_text = ""
        try:
            for name, body in SCAN_VIEWS:
                self.scan_state = f"scan: looking {name}"
                self.camera.look(pitch, 0.0, body)
                time.sleep(SCAN_SETTLE_S)
                frame, sharpness = capture_sharpest(self.camera.get_frame, frames=3)
                if frame is None:
                    views_out.append({"view": name, "error": "no frame"})
                    continue
                registration = self.board.locate(frame)
                if registration is None:
                    views_out.append({"view": name, "error": "board not found"})
                    continue
                covered = registration.space_pixels(margin=-30)
                seen.update(covered)
                if mode == "baseline":
                    self.baselines.views[name] = Baseline.capture(
                        registration, frame, extra=self._more_frames(frame)
                    )
                    views_out.append({"view": name, "inliers": registration.inliers, "spaces": len(covered)})
                    continue
                baseline = self.baselines.views.get(name)
                if baseline is None:
                    views_out.append({"view": name, "error": "no baseline for this view"})
                    continue
                # A view whose frame does not line up with its baseline (few inliers, or the
                # whole board differing: light change or a bad homography) would turn the map
                # into false pieces; it is skipped and reported instead.
                change = light_change(registration, frame, baseline)
                if registration.inliers < MIN_VIEW_INLIERS or change > LIGHT_CHANGE_MEDIAN:
                    reason = f"median difference {change:.0f} (limit {LIGHT_CHANGE_MEDIAN:.0f}), {registration.inliers} inliers"
                    views_out.append({"view": name, "error": f"baseline mismatch: {reason}"})
                    log.warning("scan: %s view skipped, %s", name, reason)
                    if name == "centre":
                        self.scan_state = (
                            f"scan failed: the centre view does not match its baseline ({reason}); "
                            "if the light changed, clear the board and run the baseline sweep"
                        )
                        return
                    self.scan_state = f"scan: {name} view skipped ({reason})"
                    continue
                if name == RESERVE_VIEW:
                    from src.vision.reserve import describe_reserve, read_reserve

                    slots = read_reserve(registration, frame, baseline)
                    folder = self.capture_dir / "reserve"
                    folder.mkdir(parents=True, exist_ok=True)
                    for slot in slots:
                        record = slot.record()
                        if slot.crop is not None and slot.crop.size:
                            file = (
                                folder
                                / f"{stamp}_slot{slot.index}_{'card' if slot.occupied else 'empty'}.jpg"
                            )
                            file.write_bytes(encode_jpeg(slot.crop, quality=92))
                            record["crop"] = f"/captures/reserve/{file.name}"
                        if (
                            slot.occupied
                            and slot.crop is not None
                            and slot.crop.size
                            and self.gallery is not None
                        ):
                            try:
                                found = self.gallery.match(slot.crop)
                            except Exception as exc:
                                log.warning("gallery match failed: %s", exc)
                                found = None
                            record["name"] = found.label if found else None
                            record["name_score"] = found.score if found else 0.0
                        reserve_out.append(record)
                    reserve_text = describe_reserve(slots)
                    self.scan_state = f"scan: {reserve_text}"
                pieces = find_pieces(registration, frame, baseline)
                for piece in pieces:
                    candidate_names[id(piece)] = piece.space or piece.near
                records = self._pieces_json(registration, pieces, stamp, name)
                if name == "centre":
                    centre_pieces = records
                sightings.append((name, pieces))
                views_out.append(
                    {
                        "view": name,
                        "inliers": registration.inliers,
                        "spaces": len(covered),
                        "pieces": len(pieces),
                    }
                )
                self.scan_state = f"scan: {name} view, {describe(pieces)}"
            if mode == "detect":
                merged = merge_pieces(sightings)
                classify_pieces(merged)  # dice with a divided vote are worth a closer look
                self._closer_looks(merged, pitch, stamp)
                classify_pieces(merged)  # again, with the closer look's crop weighing twice
                self._name_pieces(merged)
        finally:
            try:
                self.camera.look(pitch, 0.0, 0.0)
            except Exception as exc:
                log.warning("could not return to the centre view: %s", exc)
        unseen = [space.name for space in SPACES if space.name not in seen]
        if mode == "baseline":
            self.baselines.save(self.baseline_dir())
            self.last_scan = {
                "ok": True,
                "mode": mode,
                "views": views_out,
                "seen": sorted(seen),
                "unseen": unseen,
            }
            self.scan_state = (
                f"scan done: baseline of {len(self.baselines.views)} views saved; "
                f"{len(seen)} spaces covered, {len(unseen)} not"
            )
            return
        merged_out = []
        for piece in merged:
            record = piece.record()
            if id(piece) in self._look_crops:
                record["crop"] = self._look_crops[id(piece)]
            for view_name, pieces in sightings:
                for i, candidate in enumerate(pieces):
                    if candidate is piece and "crop" not in record:
                        record["crop"] = (
                            f"/captures/pieces/{stamp}_{view_name}_{i}_"
                            f"{(candidate_names.get(id(candidate)) or 'between').replace(' ', '_')}.jpg"
                        )
                        if view_name == "centre" and i < len(centre_pieces):
                            record["box"] = centre_pieces[i]["box"]
            merged_out.append(record)
        text = describe(merged)
        if reserve_text:
            text = f"{text} {reserve_text}"
        self.last_scan = {
            "ok": True,
            "mode": mode,
            "views": views_out,
            "pieces": merged_out,
            "centre_pieces": centre_pieces,
            "reserve": reserve_out,
            "seen": sorted(seen),
            "unseen": unseen,
            "text": text,
        }
        log.info("scan: %s (unseen: %s)", text, ", ".join(unseen) or "none")
        self.scan_state = f"scan done: {text}"

    def rectified_jpeg(self, width: int = 1200) -> bytes:
        registration = self.registration
        with self._lock:
            frame = self.frame
        if registration is None or frame is None:
            return b""
        return encode_jpeg(registration.rectify(frame, width=width), quality=85)

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
        factor, cx, cy = self.zoom
        jpeg = encode_jpeg(crop_frame(frame, factor, cx, cy), max_width=self.stream_width)
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
                "body": getattr(self.camera, "body", None),
                "sweep": self.sweep_state,
                "scan": self.scan_state,
                "zoom": self.zoom[0],
                "board": None if self.registration is None else self.registration.inliers,
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

    def look(self, pitch: float, yaw: float, body: float | None = None) -> None:
        pitch = max(-10.0, min(60.0, float(pitch)))
        yaw = max(-45.0, min(45.0, float(yaw)))
        body = None if body is None else max(-BODY_YAW_MAX, min(BODY_YAW_MAX, float(body)))
        if any(math.isnan(v) for v in (pitch, yaw, body if body is not None else 0.0)):
            raise ValueError("pitch, yaw and body must be numbers")
        self.camera.look(pitch, yaw, body)

    def set_zoom(self, factor: float, cx: float = 0.5, cy: float = 0.5) -> None:
        values = (
            max(1.0, min(8.0, float(factor))),
            min(max(float(cx), 0.0), 1.0),
            min(max(float(cy), 0.0), 1.0),
        )
        if any(math.isnan(v) for v in values):
            raise ValueError("zoom values must be numbers")
        self.zoom = values

    def start_sweep(self, body_yaws: list[float], pitch: float) -> bool:
        """Run a sweep on its own thread; False when one is already running."""
        from src.vision.capture import sweep, yaw_views

        if self._sweep_thread is not None and self._sweep_thread.is_alive():
            return False
        angles = [max(-BODY_YAW_MAX, min(BODY_YAW_MAX, float(a))) for a in body_yaws]
        views = yaw_views(angles, pitch=max(-10.0, min(60.0, float(pitch))))

        def run() -> None:
            self.sweep_state = f"sweep running: {len(views)} views"
            try:
                captures = sweep(
                    self.camera,
                    views,
                    out_dir=self.capture_dir,
                    on_view=lambda c: setattr(
                        self, "sweep_state", f"sweep: {c.view.name} done ({c.sharpness:.0f})"
                    ),
                )
                self.sweep_state = f"sweep done: {len(captures)} views saved, see the gallery"
            except Exception as exc:
                log.warning("sweep failed: %s", exc)
                self.sweep_state = f"sweep failed: {exc}"

        self._sweep_thread = threading.Thread(target=run, name="sweep", daemon=True)
        self._sweep_thread.start()
        return True

    def gallery_html(self) -> str:
        """The saved sweeps, newest first, with their views side by side."""
        sections = []
        folders = sorted((d for d in self.capture_dir.glob("sweep_*") if d.is_dir()), reverse=True)
        for folder in folders[:10]:
            index = folder / "views.json"
            views: list[dict[str, Any]] = []
            if index.exists():
                try:
                    views = json.loads(index.read_text(encoding="utf-8")).get("views", [])
                except (OSError, ValueError):
                    views = []
            figures = "".join(
                f'<figure><img src="/captures/{folder.name}/{Path(v["path"]).name}">'
                f"<figcaption>{v['name']}: body {v['body_yaw']:.0f}, pitch {v['pitch']:.0f}, "
                f"sharpness {v['sharpness']:.0f}, {v['width']}x{v['height']}</figcaption></figure>"
                for v in views
            )
            sections.append(f"<h2>{folder.name}</h2><div class='row'>{figures or '(no views)'}</div>")
        return GALLERY.replace("__SWEEPS__", "\n".join(sections) or "<p>no sweep yet</p>")


def make_handler(preview: Preview, table_pitch: float = TABLE_PITCH) -> type[BaseHTTPRequestHandler]:
    page = PAGE.replace("__PITCH__", str(int(table_pitch))).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # quiet: one line per frame otherwise
            if not self.path.startswith(("/stream", "/status", "/frame", "/board")):
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
            elif self.path.startswith("/board"):
                self._json(preview.board_json())
            elif self.path.startswith("/scan_result"):
                self._json(preview.last_scan or {"ok": False, "error": "no scan yet"})
            elif self.path.startswith("/rectified"):
                jpeg = preview.rectified_jpeg()
                if jpeg:
                    self._send(200, jpeg, "image/jpeg")
                else:
                    self._json({"error": "board not registered yet"}, 503)
            elif self.path.startswith("/gallery"):
                self._send(200, preview.gallery_html().encode("utf-8"), "text/html; charset=utf-8")
            elif self.path.startswith("/captures/"):
                self._capture_file(self.path[len("/captures/") :])
            else:
                self._send(404, b"not found", "text/plain")

        def _capture_file(self, relative: str) -> None:
            target = (preview.capture_dir / relative.split("?")[0]).resolve()
            root = preview.capture_dir.resolve()
            if root not in target.parents or target.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                self._send(404, b"not found", "text/plain")
                return
            try:
                self._send(200, target.read_bytes(), "image/jpeg")
            except OSError:
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
                    turn = body.get("body")
                    preview.look(
                        float(body.get("pitch", 0.0)),
                        float(body.get("yaw", 0.0)),
                        None if turn is None else float(turn),
                    )
                except (ValueError, TypeError) as exc:
                    self._json({"error": str(exc)}, 400)
                    return
                except Exception as exc:
                    log.warning("look failed: %s", exc)
                    self._json({"error": str(exc)}, 500)
                    return
                self._json(
                    {
                        "ok": True,
                        "pitch": preview.camera.pitch,
                        "yaw": preview.camera.yaw,
                        "body": getattr(preview.camera, "body", None),
                    }
                )
            elif self.path.startswith("/scan"):
                try:
                    body = json.loads(raw or b"{}")
                    mode = str(body.get("mode", "detect"))
                    pitch = float(body.get("pitch", TABLE_PITCH))
                except (ValueError, TypeError) as exc:
                    self._json({"error": str(exc)}, 400)
                    return
                if mode not in ("baseline", "detect"):
                    self._json({"error": "mode must be baseline or detect"}, 400)
                elif preview.start_scan(mode, pitch):
                    self._json({"ok": True, "mode": mode, "views": [v[0] for v in SCAN_VIEWS]})
                else:
                    self._json({"error": "a scan is already running"}, 409)
            elif self.path.startswith("/detect"):
                try:
                    self._json(preview.detect_pieces())
                except Exception as exc:
                    log.warning("detection failed: %s", exc)
                    self._json({"error": str(exc)}, 500)
            elif self.path.startswith("/look_at"):
                try:
                    body = json.loads(raw or b"{}")
                    preview.camera.look_at(float(body["u"]), float(body["v"]))
                except (ValueError, TypeError, KeyError) as exc:
                    self._json({"error": str(exc)}, 400)
                    return
                except Exception as exc:
                    self._json({"error": str(exc)}, 500)
                    return
                self._json({"ok": True, "body": getattr(preview.camera, "body", None)})
            elif self.path.startswith("/label"):
                try:
                    body = json.loads(raw or b"{}")
                    self._json(preview.label_crop(str(body["crop"]), str(body["label"])))
                except (ValueError, TypeError, KeyError) as exc:
                    self._json({"error": str(exc)}, 400)
            elif self.path.startswith("/zoom"):
                try:
                    body = json.loads(raw or b"{}")
                    preview.set_zoom(body.get("factor", 1.0), body.get("cx", 0.5), body.get("cy", 0.5))
                except (ValueError, TypeError) as exc:
                    self._json({"error": str(exc)}, 400)
                    return
                self._json({"ok": True, "zoom": preview.zoom})
            elif self.path.startswith("/sweep"):
                try:
                    body = json.loads(raw or b"{}")
                    angles = [float(a) for a in body.get("body_yaws", [])]
                    pitch = float(body.get("pitch", TABLE_PITCH))
                except (ValueError, TypeError) as exc:
                    self._json({"error": str(exc)}, 400)
                    return
                if not angles:
                    self._json({"error": "no body angles"}, 400)
                elif preview.start_sweep(angles, pitch):
                    self._json({"ok": True, "views": len(angles)})
                else:
                    self._json({"error": "a sweep is already running"}, 409)
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
    parser.add_argument(
        "--sweep", default="", help="run one sweep at these body angles (e.g. 60,30,0,-30,-60) and exit"
    )
    parser.add_argument("--no-board", action="store_true", help="do not register frames to the board picture")
    parser.add_argument("--no-gallery", action="store_true", help="do not name pieces with the gallery")
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
    preview = Preview(camera)
    if not args.no_board:
        try:
            from src.vision.board_map import BoardReference

            preview.board = BoardReference()
            if not args.no_gallery:
                try:
                    from src.vision.gallery import Gallery

                    preview.gallery = Gallery()
                except Exception as exc:
                    log.warning("gallery off: %s", exc)
            if preview.load_baseline():
                log.info(
                    "empty-board baselines loaded from %s: %s",
                    preview.baseline_dir(),
                    sorted(preview.baselines.views),
                )
        except Exception as exc:
            log.warning("board registration off: %s", exc)
    preview.start()
    if args.sweep:
        from src.vision.capture import sweep, yaw_views

        angles = [float(a) for a in args.sweep.split(",") if a.strip()]
        time.sleep(3.0)  # first frames
        captures = sweep(camera, yaw_views(angles, pitch=args.pitch), out_dir=preview.capture_dir)
        for c in captures:
            print(f"{c.view.name:10s} body {c.view.body_yaw:5.0f}  sharpness {c.sharpness:7.0f}  {c.path}")
        preview.stop()
        camera.close()
        return 0
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


__all__ = [
    "Preview",
    "FakeCamera",
    "RobotCamera",
    "encode_jpeg",
    "crop_frame",
    "make_handler",
    "serve",
    "NO_FRAMES_HINT",
]


if __name__ == "__main__":
    sys.exit(main())
