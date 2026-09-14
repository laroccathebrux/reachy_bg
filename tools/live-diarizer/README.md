# Live diarizer sidecar

Streaming "who is speaking" for reachy_bg: [diart](https://github.com/juanmc2005/diart) on
`pyannote/segmentation-3.0` + `pyannote/embedding`, reading the same Mac input device as the
main process and publishing anonymous speaker turns over a local WebSocket.

It lives in its own virtual environment because diart needs `numpy < 2` and the robot SDK
needs `numpy >= 2.2` (see `docs/SPEECH_PIPELINE.md`).

```bash
cd tools/live-diarizer
uv sync
uv run python -m live_diarizer --device "MacBook Pro Microphone" --verbose
```

The gated pyannote models need a Hugging Face login once (`hf auth login`, or `HF_TOKEN` in
the repository `.env`) after accepting the terms on the model pages.

Messages (`ws://127.0.0.1:8765`, one JSON object per frame): `hello` once per connection,
`turn` while a speaker talks (`final: false`, extended in place) and when the turn closes
(`final: true`), `step` after every inference step with the list of active speakers.
`start`/`end` are seconds into the stream; `wall_start`/`wall_end` are `time.time()` seconds.

Tuning: `--latency` (default 1 s; `--latency 0.5` is faster and noisier), `--step`,
`--gap` (silence that ends a turn), `--max-speakers`, and diart's `--tau-active`,
`--rho-update`, `--delta-new` (defaults are diart's tuned values for `segmentation-3.0`).
