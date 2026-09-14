# Reachy Mini plays Eldritch Horror

An embodied, cooperative, learning board-game agent. A Reachy Mini desktop robot sits at the
table, controls its own investigator in **Eldritch Horror (Fantasy Flight Games, 2013, base
game)**, decides its moves, explains them by voice, answers rules questions from the official
rulebook, and learns from every game it plays with its human teammates.

It is also a research platform on three questions: turn-taking in a multi-party conversation
without wake words, emergence of cooperative strategy between a human and an AI, and how
many games a world model needs before it becomes useful.

## Status

Phase 0 (retrieval and knowledge base) and Phase 2a (listening) are done: the robot hears a
question at the table in Portuguese or English, looks it up, and answers with a native voice.

| Piece | State |
|---|---|
| `bg_rules` | official English rulebook + reference guide, 216 chunks, searchable |
| `bg_knowledge` | 12 investigators, 4 Ancient Ones, monsters, gates, 142 FAQ entries |
| `bg_sessions` | created, fills up as games are played |
| Speech | ElevenLabs agent for the conversation (0.2-2.5 s turns) with local rules tools; fully local Whisper + Ollama path kept for offline work |
| Who is speaking | diart sidecar (`tools/live-diarizer`), voiceprints per player, rule-based "is it for me?" with a per-utterance log |
| Vision, strategy, learning | designed, not built (Phases 1, 3-5) |

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the roadmap.

## How it works

```
 Mac microphone --> ASR + speaker diarization --> "who said what, and was it for me?"
                                                        |
 Reachy camera --> board detection --> game state <-----+-----> Ollama (Qwen 3.6 35B-A3B)
                                            |                    + Qdrant (rules, knowledge, memory)
                                            v
                              decision + justification --> TTS --> Reachy speaker
                                            |
                                            v
                              trajectory log --> world model (learning between games)
```

Everything runs locally on one Mac except the optional cloud voice. Details:
[docs/DESIGN_DOCUMENT.md](docs/DESIGN_DOCUMENT.md).

## Quick start

```bash
git clone ssh://git@github.com/laroccathebrux/reachy_bg.git
cd reachy_bg
uv sync                                  # Python 3.12 venv with the core dependencies
cp .env.example .env                     # then fill in the secrets you need
uv run pytest                            # unit tests, no services needed
```

With Docker (Qdrant) and Ollama running:

```bash
uv run python -m src.rag.ingest_rules    # embed data/rulebooks/*.pdf into bg_rules
uv run python -m src.rag.migrate_knowledge
```

With the robot daemon on `:8000` and the ElevenLabs key in `.env`:

```bash
uv sync --extra speech --extra tts-cloud
uv run python scripts/venv_postinstall.py       # macOS: make the venv's .pth files visible
uv run python -m src.integration.smoke          # typed question -> spoken answer
uv run python -m src.integration.talk           # table conversation on the ElevenLabs agent (Ctrl+C to stop)
uv run python -m src.integration.talk --players "Ana,Bruno"     # enrol voices first
uv run python -m src.integration.talk --humans 2                 # the gate: only what is for the robot reaches the agent
uv run python -m src.integration.talk --always-answer            # no gate: the agent hears everything
uv run python scripts/gate_replay.py data/captures/audio/*.wav --humans 2   # replay captures through the gate, offline
uv run python -m src.vision.preview                              # camera preview at http://127.0.0.1:8090 to place the board
uv run python -m src.vision.preview --sweep 60,30,0,-30,-60      # one sweep of the board with the body, saved to data/captures/board/
# The preview draws the board outline and the space names over the video when data/imgs/World_Map.webp
# (a top-down picture of the real board) is present; see src/vision/board_map.py.
uv run python -m src.integration.listen         # fully local pipeline (Whisper + Ollama), slower
uv run python -m src.integration.listen --list-devices
```

Optional live "who is speaking" (separate venv, see [tools/live-diarizer](tools/live-diarizer/README.md)):

```bash
cd tools/live-diarizer && uv sync && uv run python -m live_diarizer --device "MacBook Pro Microphone"
```

Full setup, services and the robot daemon: [GETTING_STARTED.md](GETTING_STARTED.md).

## Documentation

| Document | What it covers |
|---|---|
| [docs/DESIGN_DOCUMENT.md](docs/DESIGN_DOCUMENT.md) | architecture, game state, decision pipeline, conversation policy |
| [docs/GAME_REFERENCE.md](docs/GAME_REFERENCE.md) | verified Eldritch Horror base-game facts |
| [docs/SETUP_PROTOCOL.md](docs/SETUP_PROTOCOL.md) | how a game session starts |
| [docs/PHYSICAL_SETUP.md](docs/PHYSICAL_SETUP.md) | table, robot hardware, gaze control, calibration |
| [docs/SPEECH_PIPELINE.md](docs/SPEECH_PIPELINE.md) | microphone, ASR, diarization, addressee detection, TTS |
| [docs/MODEL_SIZING.md](docs/MODEL_SIZING.md) | why Qwen 3.6 35B-A3B, measured on the development Mac |
| [docs/FULL_CONTEXT_LEARNING.md](docs/FULL_CONTEXT_LEARNING.md) | trajectory format, online and offline learning |
| [WORKFLOW.md](WORKFLOW.md) | day-to-day development with Claude Code and git |
| [CLAUDE.md](CLAUDE.md) | rules every coding session must follow |

## Tech stack

| Layer | Choice |
|---|---|
| Robot | Reachy Mini Lite, `reachy-mini` SDK, daemon on port 8000 |
| Reasoning | Qwen 3.6 35B-A3B through Ollama (local) |
| Embeddings and retrieval | bge-m3 through Ollama, Qdrant (Docker) |
| Speech in | Mac microphone, mlx-whisper (Metal), diart + pyannote (live), pyannote 4 + WhisperX (offline) |
| Speech out | ElevenLabs (default) or local TTS, played on the robot |
| Vision | OpenCV, YOLO-World, image-embedding gallery |
| Learning | PyTorch, Dreamer-style world model |
| Language | Python 3.12, `uv`, pytest, ruff |

## Conventions

- The repository is **English only**: code, comments, docs, tests, commits, example dialogue.
  The robot's spoken language is a runtime setting.
- Base game only, 2013 edition, no Arkham Horror content.
- Work is committed directly to `main`.

## License

MIT. Eldritch Horror is a trademark of Fantasy Flight Games; the rulebook PDFs are not part of
this repository.
