# CLAUDE.md

Project conventions for Claude Code sessions in this repository. Read this before touching anything.

## What this project is

Reachy Mini (Pollen Robotics / Hugging Face desktop robot) plays **Eldritch Horror (Fantasy Flight Games, 2013, base game only)** as a real cooperative player: it controls its own investigator, decides its moves, explains them by voice, and the human moves the pieces. It is also a research platform for multi-party turn-taking without wake words, cooperative human/AI strategy, and world models learned from play. See [README.md](README.md) and [docs/DESIGN_DOCUMENT.md](docs/DESIGN_DOCUMENT.md).

## Hard rules

1. **English only in the repository.** Code, comments, docstrings, docs, tests, commit messages, example dialogues, prompts sent to the LLM. Conversation with the project owner happens in Portuguese, but nothing Portuguese is committed. The robot's spoken language is a runtime setting (`SPOKEN_LANGUAGE`), never hardcoded.
2. **Eldritch Horror 2013, base game.** Never use Arkham Horror content (Roland Banks, Joe Diamond, elder signs, terror track, Arkham locations). Use the 12 base investigators, the 4 base Ancient Ones, skills Lore / Influence / Observation / Strength / Will, Mysteries, the world map. Facts live in [docs/GAME_REFERENCE.md](docs/GAME_REFERENCE.md).
3. **The robot is a Reachy Mini Lite**, driven from this Mac through the `reachy-mini` SDK and its daemon on `http://127.0.0.1:8000`. Not the full-size Reachy, no gRPC, no `reachy-sdk`.
4. **The robot microphone is broken** (torn flat cable). Audio input always comes from a Mac input device; audio output still goes to the robot speaker. See [docs/SPEECH_PIPELINE.md](docs/SPEECH_PIPELINE.md).
5. **This Mac is the owner's work computer.** Model choices must leave memory headroom for their work apps. See [docs/MODEL_SIZING.md](docs/MODEL_SIZING.md).
6. **Commit and push directly to `main`** after every completed task. No branches, no PRs. Run `pytest` before committing.

## Workflow for every task

```bash
cd "/Users/alessandrolaroccasilveira/Documents/Documents - USSILVEIRAAXWR2/reachy_bg"
git pull origin main
uv sync                      # creates/updates .venv (Python 3.12)
# ... do the work ...
uv run pytest
git add -A && git commit -m "Clear message in English" && git push origin main
git log --oneline -3 && git status
```

The remote is `ssh://git@github.com/laroccathebrux/reachy_bg.git`. Keep the `ssh://` form: the global gitconfig rewrites `git@github.com:` to HTTPS, and the HTTPS credential in the keychain is a different account.

## Prior work: read, never copy

`../reachy/boardgames/` is an earlier attempt at this idea, built on Pollen's conversation app with a cloud voice agent. It is useful to read for lessons (SDK usage, macOS pitfalls, what worked for board vision), but **no code, script, profile or config from `../reachy` is copied into this repository.** Everything here is written from scratch. The only migration that happened was data: its Qdrant knowledge collection was copied into `bg_knowledge` with an English payload. Do not modify `../reachy` from here.

## Services this project expects

| Service | Where | Notes |
|---|---|---|
| Qdrant | Docker container `reachy-books-qdrant`, `http://127.0.0.1:6333` | Ours: `bg_rules` (English rulebook + reference guide), `bg_knowledge` (investigators, Ancient Ones, monsters, gates, FAQ), `bg_sessions` (round memory). The other collections in that container (`boardgames`, `conhecimento`, `partidas`, `livros`) belong to `../reachy`: read-only, never modify. |
| Ollama | `http://127.0.0.1:11434` | `qwen3.6:35b-mlx` for reasoning (35B MoE, 3B active), `bge-m3` for embeddings (1024-d; every `bg_*` collection uses it). |
| Reachy daemon | `reachy-mini-daemon`, `http://127.0.0.1:8000` | Started per session; robot on USB `/dev/cu.usbmodem*`. |

## Layout

```
src/            Python package (config, logger, then vision / speech / strategy / learning / integration)
tests/          pytest
docs/           design and reference documents
data/           runtime data (git-ignored): captures, game logs, models
```
