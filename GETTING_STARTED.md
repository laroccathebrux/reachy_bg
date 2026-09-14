# Getting Started Guide

## 🚀 Quick Start

### Option 1: Using Claude Code (Recommended for Development)

The project is already initialized in Claude Code. You can:

1. **Use Claude Code directly** to develop and test
2. **Push to GitHub** when ready (see below)
3. **Clone on your Mac** for local development

### Option 2: Clone to Your Mac

#### Prerequisites
- Python 3.9+
- Git
- M1 Max 64GB RAM (your setup)
- Ollama installed locally
- Qdrant running locally

#### Steps

```bash
# 1. Navigate to your Documents folder
cd "/Users/alessandrolaroccasilveira/Documents/Documents - USSILVEIRAAXWR2/"

# 2. Clone the repository (once it's pushed to GitHub)
git clone https://github.com/your-username/eldritch-horror-reachy.git
cd eldritch-horror-reachy

# 3. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Copy and configure environment
cp .env.example .env
# Edit .env with your settings

# 6. Verify services are running
# Terminal 1: Ollama
ollama serve

# Terminal 2: Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Terminal 3: Run tests
pytest

# Terminal 4: Start game loop (when ready)
python src/integration/game_loop.py
```

---

## 📁 Project Structure

```
eldritch-horror-reachy/
├── src/
│   ├── __init__.py              # Main package
│   ├── config.py                # Configuration management
│   ├── logger.py                # Logging utilities
│   ├── vision/                  # Board recognition (Phase 1)
│   ├── speech/                  # STT, TTS, turn-taking (Phase 2)
│   ├── strategy/                # LLM reasoning, planning (Phase 3)
│   ├── learning/                # World models, training (Phase 5)
│   └── integration/             # Game loop, end-to-end (Phase 4)
├── docs/
│   ├── DESIGN_DOCUMENT.md       # Architecture overview
│   ├── PHYSICAL_SETUP.md        # Table setup & calibration
│   ├── FULL_CONTEXT_LEARNING.md # Learning strategy
│   └── SETUP_PROTOCOL.md        # Game initialization
├── tests/                       # Unit & integration tests
├── data/
│   ├── qdrant_setup/            # Vector DB initialization
│   ├── game_logs/               # Game trajectories
│   ├── models/                  # Trained world models
│   └── captures/                # Board images
├── config/                      # Configuration files
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── README.md                    # Project overview
├── .env.example                 # Environment template
└── .gitignore                   # Git ignore rules
```

---

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
# Qdrant Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Ollama LLM
OLLAMA_MODEL=qwen2.5:72b
OLLAMA_BASE_URL=http://localhost:11434

# ElevenLabs TTS
ELEVENLABS_API_KEY=sk_your_key_here
ELEVENLABS_VOICE_ID=your_voice_id

# Reachy Robot
REACHY_HOST=localhost
REACHY_PORT=50051

# Directories
GAME_LOG_DIR=./data/game_logs
MODEL_DIR=./data/models
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_setup_manager.py

# Run with coverage
pytest --cov=src

# Run specific test with verbose output
pytest -v tests/test_*.py
```

---

## 📊 Development Phases

### Phase 0: Setup & Configuration ✅
- [x] Project structure
- [x] Documentation
- [x] Configuration management
- [ ] Qdrant data initialization
- [ ] Ollama setup verification

### Phase 1: Vision & Game State (Weeks 1-2)
- [ ] Board capture from Reachy camera
- [ ] SAM segmentation pipeline
- [ ] Game state parser (image → JSON)
- [ ] Angle robustness testing

### Phase 2: Speech & Turn-Taking (Weeks 2-3)
- [ ] Continuous STT setup
- [ ] Speaker diarization
- [ ] Addressee detection
- [ ] Conversation state machine

### Phase 3: Strategic Reasoning (Weeks 3-4)
- [ ] Ollama inference pipeline
- [ ] Multi-option evaluation
- [ ] RAG integration
- [ ] Action justification

### Phase 4: End-to-End Integration (Weeks 4-5)
- [ ] Full game loop
- [ ] Voice communication
- [ ] Feedback incorporation
- [ ] Testing with Alessandro

### Phase 5: World Models (Weeks 5-6)
- [ ] Data collection
- [ ] Open Dreamer training
- [ ] Simulation-based planning
- [ ] Continuous learning

### Phase 6: Research & Publication (Weeks 6+)
- [ ] Multi-player scenarios
- [ ] Emergent behavior analysis
- [ ] Research paper

---

## 🔄 Git Workflow

### Make Changes in Claude Code

```bash
# Claude Code terminal
cd /home/claude/eldritch-horror-reachy

# Make changes
# Test locally
pytest

# Commit
git add .
git commit -m "Clear message about changes"
```

### Push to GitHub

```bash
# Once repository is on GitHub:
git remote add origin https://github.com/your-username/eldritch-horror-reachy.git
git branch -M main
git push -u origin main

# Future pushes:
git push
```

### Pull on Your Mac

```bash
# On your Mac
cd "/Users/alessandrolaroccasilveira/Documents/Documents - USSILVEIRAAXWR2/"
git clone https://github.com/your-username/eldritch-horror-reachy.git
cd eldritch-horror-reachy

# Later, pull updates
git pull origin main
```

---

## 🚨 Troubleshooting

### Python Import Errors

```bash
# Make sure venv is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Qdrant Connection Error

```bash
# Check Qdrant is running
curl http://localhost:6333/health

# If not running:
docker run -p 6333:6333 qdrant/qdrant
```

### Ollama Connection Error

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# If not running:
ollama serve

# Verify model is loaded
ollama list
```

### Tests Failing

```bash
# Run with verbose output
pytest -v

# Check Python version
python --version  # Should be 3.9+

# Reinstall in venv
pip install -e ".[dev]"
```

---

## 📚 Documentation

Read these in order:

1. **README.md** — Project overview
2. **DESIGN_DOCUMENT.md** — Architecture details
3. **PHYSICAL_SETUP.md** — Table setup & constraints
4. **SETUP_PROTOCOL.md** — Game initialization protocol
5. **FULL_CONTEXT_LEARNING.md** — Learning strategy

---

## 🎯 Next Steps

1. **Clone locally** or continue in Claude Code
2. **Read DESIGN_DOCUMENT.md** thoroughly
3. **Set up Qdrant** with Eldritch Horror data
4. **Verify Ollama** runs Qwen 2.5 locally
5. **Start Phase 1** — Vision module

---

## 💬 Questions?

Refer to relevant documentation or discuss in issues/PRs:
- Architecture questions → DESIGN_DOCUMENT.md
- Setup questions → PHYSICAL_SETUP.md
- Learning strategy questions → FULL_CONTEXT_LEARNING.md

---

Happy coding! 🚀🤖
