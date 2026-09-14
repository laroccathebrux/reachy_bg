# Project Status

Updated 2026-09-14.

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
- 21 unit tests passing; none require external services.

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

- [ ] Calibration session with the checklist in docs/PHYSICAL_SETUP.md; reference photos of every token type.
- [ ] `src/vision/capture.py`: gaze to table pose, sharpest-of-three capture, save to `data/captures/`.
- [ ] `src/vision/detect.py`: YOLO-World with the game's prompt list; gallery matching.
- [ ] `src/vision/board_map.py`: board corners -> canonical map -> space assignment.
- [ ] `src/strategy/state.py`: pydantic game state + patch application (needed by everything after).
- [ ] Angle and lighting robustness test with recorded frames.

### Phase 2: speech and turn-taking (weeks 2-3)

- [ ] Mac microphone capture (`sounddevice`) with device selection from `AUDIO_INPUT_DEVICE`.
- [ ] mlx-whisper streaming transcription in Portuguese.
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
