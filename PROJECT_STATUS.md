# Project Status

Updated 2026-09-14 (Phase 2a).

## Done: Phase 0, foundation

- Repository restarted from scratch in English with correct facts: Eldritch Horror 2013 base
  game, Reachy Mini Lite, broken robot microphone, shared work Mac. `CLAUDE.md` carries the
  rules for every coding session.
- Python 3.12 environment managed by `uv` (`pyproject.toml`, `uv.lock`); `reachy-mini` 1.10
  installs and imports on the Mac.
- Configuration (`src/config.py`) and logging (`src/logger.py`) with tests.
- Retrieval layer `src/rag/`:
  - `embeddings.py` (bge-m3 via Ollama), `store.py` (Qdrant lifecycle, upsert, search),
    `collections.py` (payload contracts).
  - `pdf_sections.py` + `chunking.py`: typography- and layout-aware splitter for the FFG PDFs
    (two-column reading order, running heads removed, wrapped headings joined, icon fonts
    mapped).
  - `ingest_rules.py`: official English rulebook + reference guide -> `bg_rules`
    (71 + 80 sections, 216 chunks).
  - `migrate_knowledge.py`: legacy Portuguese-labelled knowledge -> `bg_knowledge` in English
    with structured investigator sheets and verified Ancient One records (322 points).
  - `sessions.py`: `bg_sessions` round memory with `record_round` / `recall`.
- Documentation rewritten: design, game reference, setup protocol, physical setup, speech
  pipeline, model sizing, learning, getting started, workflow.
- 31 unit tests passing; none require external services.

## Done: step 1, end-to-end smoke test on the physical robot (2026-09-14)

`uv run python -m src.integration.smoke` runs typed question -> retrieval (`bg_rules` +
`bg_knowledge`) -> Qwen 3.6 via Ollama -> ElevenLabs with the native voice for the detected
language -> playback on the Reachy Mini speaker, with a nod and antenna gestures. New modules:
`src/llm/` (Ollama client, prompts), `src/rag/retrieve.py`, `src/speech/` (typed-text language
detection, TTS), `src/robot/reachy.py` (SDK wrapper with a speaker-only simulation mode).

Measured on the work Mac under heavy load (load average ~70, 8 GB swap in use):

| Stage | English question | Portuguese question |
|---|---|---|
| Embedding + retrieval (8 passages) | 2.4 s (cold) | 0.2 s |
| LLM answer, 3 sentences | 12.4 s at 4.3 tok/s | 7.8 s at 7.7 tok/s |
| First LLM call of the session (model load) | 41 s | - |
| ElevenLabs TTS (16 kHz PCM) | 1.5 s | 1.4 s |
| Spoken answer length | 16 s | 17 s |

Findings:
- The robot daemon works on this macOS with SDK 1.10.0 (control loop 49 Hz), but only after
  `scripts/venv_postinstall.py`: the venv's `.pth` files get the macOS `hidden` flag re-applied
  within a minute by something on this machine, and Python 3.12.13 skips hidden `.pth` files,
  so the GStreamer bindings were invisible. The daemon takes about 3 minutes to open port 8000.
- macOS denied camera access to the daemon started from the terminal; grant it in
  System Settings > Privacy & Security > Camera before Phase 1.
- LLM throughput is well below the 30-50 tok/s expected for a 3B-active model because the
  machine was saturated; answers of 3 sentences still land in 8-12 s. Re-measure on a quiet
  machine and consider `think=False` plus shorter answers for in-game replies.
- Language detection and voice selection behaved correctly in both languages; game terms
  stay in English inside Portuguese answers as agreed.

## Done: Phase 2a, listening and transcribing on the physical robot (2026-09-14)

`uv run python -m src.integration.listen` runs Mac microphone -> energy VAD -> mlx-whisper ->
retrieval -> Qwen 3.6 -> ElevenLabs native voice -> robot speaker, in a loop, with an
interruptible `Robot.say`. New modules: `src/speech/microphone.py` (device selection,
`Segmenter` VAD, capture thread, WAV captures in `data/captures/audio/`), `src/speech/asr.py`
(`Transcriber`: language identification restricted to `SPOKEN_LANGUAGES`, then decoding with
the language pinned), `src/integration/answering.py` (retrieval + LLM + TTS shared with the
smoke test), `src/integration/listen.py`. 58 tests; the Whisper test runs only when the model
is in the Hugging Face cache and uses macOS `say` to make its clip.

Validated with the owner speaking freely at the MacBook (three questions in Portuguese and
English, all transcribed verbatim, language identified with confidence 0.99 or better) and
with clips played through the Mac speakers. Measured on the work Mac, mostly under heavy
load:

| Stage | Best (quiet Mac) | Typical (loaded Mac) |
|---|---|---|
| VAD tail (silence that ends the utterance) | 0.72 s | 0.72 s |
| Whisper large-v3-turbo, 2-4 s utterance (language id + decode) | 1.35 s | 2.9-3.1 s |
| Whisper warm-up at start (weights cached on disk) | 2.8 s | 5.8 s |
| Embedding + retrieval | 0.1-0.25 s | 0.7-2 s (cold) |
| LLM answer, 3 sentences | 1.5 s at 28.6 tok/s | 7-16 s at 2.6-8.6 tok/s; 31 s when the model reloads |
| ElevenLabs TTS | 1.1 s | 1.7 s |
| Ear to mouth (person stops talking -> robot starts) | 4.9 s | 10-21 s |

Findings:
- The HyperX QuadCast, the Mac's default input, delivered digital silence (-96 dBFS) in every
  test: it is muted on its touch sensor or its gain is at zero. The MacBook microphone works
  and is what `--device "MacBook Pro Microphone"` used. Set `AUDIO_INPUT_DEVICE` accordingly.
- Without echo cancellation the robot's own voice reaches the MacBook microphone at
  -21 to -31 dBFS peak, the same level as a person talking normally at the table (-28 to
  -31 dBFS). Energy-only barge-in therefore needs a raised voice next to the Mac; the
  owner's "para, para, para" at normal volume did not cross the bar. Whisper transcribes the
  robot's own voice perfectly, which suggests a cheap fix: compare each transcript with the
  last spoken answer and drop the echo, then lower `BARGE_IN_MARGIN_DB`. A reference-based
  echo canceller (the played clip is known) is the proper fix. See docs/SPEECH_PIPELINE.md.
- Whisper likes to hear "Rich" for "Reachy"; the addressee rules in Phase 2b must accept
  the common misspellings.
- Utterances captured while the robot thinks are currently discarded before it speaks; a
  queue with a "still relevant?" check belongs to the conversation state machine.
- The Portuguese answers translated "Action Phase" once ("fase de ação") despite the prompt;
  worth a few-shot example in the prompt.
- Background runs must be stopped with SIGTERM (a `&` job ignores SIGINT); `listen.py` now
  treats SIGTERM like Ctrl+C. Two instances left running answered each other for a minute.

## Decisions taken

| Topic | Decision | Where |
|---|---|---|
| Reasoning model | Qwen 3.6 35B-A3B (`qwen3.6:35b-mlx`), not Qwen 2.5 72B | docs/MODEL_SIZING.md |
| Embeddings | bge-m3 through Ollama, 1024-d | src/rag/collections.py |
| Vector store | Qdrant in the existing Docker container; collections `bg_rules`, `bg_knowledge`, `bg_sessions` | docs/DESIGN_DOCUMENT.md |
| Speech input | Mac microphone; mlx-whisper; diart + pyannote 3 live in a sidecar; pyannote 4 + WhisperX offline | docs/SPEECH_PIPELINE.md |
| Speech output | ElevenLabs by default, local TTS optional | src/config.py |
| Game setup | verbal briefing + knowledge base, no card OCR | docs/SETUP_PROTOCOL.md |
| Vision | YOLO-World + image-embedding gallery, SAM not used | docs/DESIGN_DOCUMENT.md |
| Prior project | read for lessons only; no code copied | CLAUDE.md |

## Next

### Phase 1: vision and board state (weeks 1-2)

Blocked on the owner deciding where the robot sits at the table.

- [ ] Grant camera permission to the terminal/daemon (macOS Privacy settings).
- [ ] Calibration session with the checklist in docs/PHYSICAL_SETUP.md; reference photos of every token type.
- [ ] `src/vision/capture.py`: gaze to table pose, sharpest-of-three capture, save to `data/captures/`.
- [ ] `src/vision/detect.py`: YOLO-World with the game's prompt list; gallery matching.
- [ ] `src/vision/board_map.py`: board corners -> canonical map -> space assignment.
- [ ] `src/strategy/state.py`: pydantic game state + patch application (needed by everything after).
- [ ] Angle and lighting robustness test with recorded frames.

### Phase 2: speech and turn-taking (weeks 2-3)

- [x] Mac microphone capture (`sounddevice`) with device selection from `AUDIO_INPUT_DEVICE`.
- [x] mlx-whisper transcription per utterance, Portuguese and English, language id per utterance.
- [x] Listening loop with interruptible speech (`src/integration/listen.py`).
- [ ] Echo handling so barge-in works at normal voice level (transcript match, then a
      reference-based canceller).
- [ ] `tools/live-diarizer/`: diart sidecar publishing speaker turns over WebSocket.
- [ ] Speaker enrolment and voiceprint matching.
- [ ] Rule-based addressee classifier + speak/silence log; annotate a recorded session.

### Phase 3: strategic reasoning (weeks 3-4)

- [ ] Ollama client with JSON-schema outputs; prompt templates in English.
- [ ] Candidate generation, legality checks against `bg_rules` and the board map, evaluation, justification.
- [ ] Rules Q&A tool returning section and page.

### Phase 4: end-to-end (weeks 4-5)

- [ ] Conversation state machine, gaze states, TTS through the robot.
- [ ] Setup protocol implemented; first full 2-player game logged.
- [ ] Human feedback captured into the trajectory.

### Phase 5: world models (weeks 5-6)

- [ ] Trajectory dataset builder; baseline next-state model; Dreamer-style model.
- [ ] Rollouts in the decision pipeline; evaluation on held-out rounds.

### Phase 6: research (week 6+)

- [ ] Multi-player sessions, annotation study of turn-taking, paper draft.

## Open items

- Calibration numbers (riser height, head pitch, exposure) are placeholders until measured.
- Latency figures in docs/SPEECH_PIPELINE.md come from published benchmarks, not this Mac.
- The two unverified facts in docs/GAME_REFERENCE.md (Doom track maximum, token counts).
- Repairing the robot microphone cable would add direction-of-arrival to addressee detection.
