# Getting Started

Everything needed to run the project on the development Mac, from a fresh clone to a
searchable rulebook and a robot that answers to Python.

## 1. Prerequisites

| Tool | Why | Check |
|---|---|---|
| `uv` | Python 3.12 environment and lockfile | `uv --version` |
| Docker Desktop | Qdrant vector database | `docker ps` |
| Ollama | local LLM and embeddings | `ollama list` |
| Homebrew `portaudio`, `ffmpeg`, `libsndfile` | audio (Phase 2 only) | `brew list` |
| Reachy Mini Lite on USB with its power supply | the robot | `ls /dev/cu.usbmodem*` |

Models to have in Ollama:

```bash
ollama pull bge-m3            # embeddings, required
ollama pull qwen3.6:35b-a3b   # reasoning; the dev Mac already has it as qwen3.6:35b-mlx
ollama pull qwen2.5:3b        # optional: small model for tests
```

## 2. Clone and install

```bash
git clone ssh://git@github.com/laroccathebrux/reachy_bg.git
cd reachy_bg
uv sync                       # core deps: robot SDK, Ollama/Qdrant clients, PDF tools
cp .env.example .env
```

Optional extras, installed on demand:

```bash
uv sync --extra vision        # OpenCV, YOLO-World, torch
uv sync --extra speech        # mlx-whisper, sounddevice, pyannote 4 (offline diarization)
uv sync --extra tts-cloud     # ElevenLabs client
```

The live diarizer runs in its own venv (`tools/live-diarizer/`, Phase 2) because its
dependencies conflict with the robot SDK.

## 3. Start the services

```bash
# Qdrant: the container is shared with another local project; start it if it is down
docker start reachy-books-qdrant || docker run -d --name reachy-books-qdrant -p 127.0.0.1:6333:6333 -p 127.0.0.1:6334:6334 qdrant/qdrant
curl -s http://127.0.0.1:6333/healthz

# Ollama (usually already running as a menu-bar app)
curl -s http://127.0.0.1:11434/api/tags | head -c 200
```

## 4. Build the knowledge base

Download the official English PDFs into `data/rulebooks/` (they are copyrighted and not
committed):

```bash
mkdir -p data/rulebooks
curl -L -o data/rulebooks/eh01_rulebook.pdf "https://images-cdn.fantasyflightgames.com/ffg_content/eldritch-horror/EH01%20Rulebook.pdf"
curl -L -o data/rulebooks/eh01_reference_guide.pdf "https://images-cdn.fantasyflightgames.com/ffg_content/eldritch-horror/EH01%20Reference%20Guide.pdf"
```

Then embed them and check retrieval:

```bash
uv run python -m src.rag.ingest_rules --dry-run     # sections and chunks, no writes
uv run python -m src.rag.ingest_rules               # writes bg_rules (216 chunks)
uv run python -m src.rag.migrate_knowledge          # writes bg_knowledge from the legacy collection
```

```bash
uv run python -c "
from src.rag.store import get_client, search
from src.rag.embeddings import embed_text
for h in search(get_client(), 'bg_rules', embed_text('how many actions per round?'), limit=3):
    print(round(h['score'],3), h['payload']['path'], '|', h['payload']['text'][:90])
"
```

The Qdrant dashboard at http://127.0.0.1:6333/dashboard shows the three `bg_*` collections.

## 5. Talk to the robot

```bash
uv run reachy-mini-daemon                 # separate terminal; wait for "Uvicorn running on :8000"
```

```bash
uv run python -c "
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
with ReachyMini(media_backend='no_media') as mini:
    mini.enable_motors(); mini.wake_up()
    mini.goto_target(head=create_head_pose(pitch=20, degrees=True), duration=1.0)
    print('ok', mini.get_current_joint_positions()[0][:3])
"
```

If nothing moves, the robot is asleep or its motors are disabled; the snippet above handles
both. See [docs/PHYSICAL_SETUP.md](docs/PHYSICAL_SETUP.md) for the rest.

## 6. Run the tests

```bash
uv run pytest                 # unit tests only; nothing here needs Qdrant, Ollama or the robot
uv run ruff check src tests   # lint
```

## 7. Project layout

```
CLAUDE.md            rules for coding sessions
README.md            overview
GETTING_STARTED.md   this file
WORKFLOW.md          git and Claude Code routine
PROJECT_STATUS.md    roadmap and what is done
pyproject.toml       dependencies (uv), pytest and ruff settings
.env.example         every setting with its default
src/                 config, logger, rag/ (retrieval layer); vision/ speech/ strategy/ integration/ learning/ come next
tests/               pytest
docs/                design, game reference, setup, physical, speech, model sizing, learning
data/                git-ignored runtime data: rulebooks/, captures/, game_logs/, models/
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `uv sync` tries to build PyGObject | It should not: `pyproject.toml` limits resolution to macOS. Run `uv lock --upgrade` if the lock is stale. |
| `ModuleNotFoundError: src` | Run from the repository root with `uv run python -m src....`; tests add the root to `sys.path`. |
| Embedding requests hang | Ollama is loading `bge-m3` for the first time; wait, or check `ollama ps`. |
| Qdrant `Connection refused` | Start the container (section 3). |
| Push rejected with 403 for another GitHub user | The remote must stay `ssh://git@github.com/...`; see [WORKFLOW.md](WORKFLOW.md). |
