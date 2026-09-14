# Eldritch Horror + Reachy Mini: Embodied Conversational AI Agent

Research project combining embodied AI, game-playing agents, and cooperative multi-agent learning.

## Overview

Building **Reachy Mini** (a humanoid desktop robot) into an intelligent game-playing agent for **Eldritch Horror**, a cooperative board game.

### Key Features

- **Embodied Conversational AI**: Robot that understands multi-party conversation pragmatics (turn-taking without wake words)
- **Game-Playing Agent**: Makes strategic decisions using local LLMs (Qwen 2.5) + RAG
- **Cooperative Learning**: Learns from human players through multi-agent world models
- **Research Platform**: Novel work on embodied AI, pragmatic understanding, and emergent cooperation

## Project Structure

```
eldritch-horror-reachy/
├── src/
│   ├── vision/           # Board recognition, SAM segmentation
│   ├── speech/           # STT, TTS, turn-taking, addressee detection
│   ├── strategy/         # LLM reasoning, action generation, evaluation
│   ├── learning/         # Online learning, world model training
│   └── integration/      # End-to-end game loop
├── docs/                 # Design documents, research notes
├── config/               # Configuration files
├── data/
│   ├── qdrant_setup/     # Vector DB initialization
│   ├── game_logs/        # Game trajectories for learning
│   ├── models/           # Trained world models
│   └── captures/         # Board images for analysis
├── tests/                # Unit and integration tests
├── requirements.txt      # Python dependencies
├── setup.py             # Package setup
└── README.md            # This file
```

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/eldritch-horror-reachy.git
cd eldritch-horror-reachy
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Start Services

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Terminal 3: Run Reachy Agent
python src/integration/game_loop.py
```

## Documentation

- **[DESIGN_DOCUMENT.md](docs/DESIGN_DOCUMENT.md)** - Complete architecture overview
- **[PHYSICAL_SETUP.md](docs/PHYSICAL_SETUP.md)** - Table setup, camera calibration, mechanical constraints
- **[FULL_CONTEXT_LEARNING.md](docs/FULL_CONTEXT_LEARNING.md)** - Multi-agent cooperative learning strategy
- **[SETUP_PROTOCOL.md](docs/SETUP_PROTOCOL.md)** - Game initialization (verbal + RAG)

## Architecture at a Glance

```
┌────────────────────────────────────────────────────┐
│            REACHY MINI GAME AGENT                  │
├────────────────────────────────────────────────────┤
│                                                    │
│  PERCEPTION          COGNITION       COMMUNICATION│
│  ├─ Vision          ├─ Strategy      ├─ Speech    │
│  ├─ Speech          ├─ Planning      └─ Movement  │
│  └─ Board State     └─ Reasoning                  │
│                                                    │
└────────────────────────────────────────────────────┘
        ↓                   ↓                ↓
    Reachy Camera      Ollama (Qwen)    ElevenLabs
    SAM Segmentation   Qdrant (RAG)     Natural Voice
                       World Models
```

## Tech Stack

- **LLM**: Qwen 2.5 72B via Ollama (local, zero API cost)
- **Voice**: ElevenLabs TTS
- **Vision**: SAM (Segment Anything), OpenCV, PyTorch
- **RAG**: Qdrant vector DB with Sentence Transformers
- **Robot**: Reachy Mini (M1 Max, 64GB RAM)
- **Language**: Python 3.9+

## Research Focus

### 1. Multi-Party Conversational Pragmatics
- Turn-taking without wake words
- Addressee detection via embedding similarity
- Conversational context understanding

### 2. Cooperative Multi-Agent Gameplay
- Human-AI alignment in cooperative games
- Strategy emergence through shared context
- Communication patterns in mixed teams

### 3. World Models for Games
- Learning game dynamics from observation
- Predicting consequences of actions
- Cooperative vs isolated strategy

## Development Roadmap

### Phase 0: Setup & Configuration
- [x] Project structure
- [ ] Qdrant initialization
- [ ] Ollama setup with Qwen 2.5
- [ ] Config management

### Phase 1: Vision & Game State (Weeks 1-2)
- [ ] Board capture & SAM segmentation
- [ ] Game state parser (image → JSON)
- [ ] Angle robustness testing

### Phase 2: Speech & Turn-Taking (Weeks 2-3)
- [ ] Continuous STT
- [ ] Speaker diarization
- [ ] Addressee detection
- [ ] Conversation state machine

### Phase 3: Strategic Reasoning (Weeks 3-4)
- [ ] Ollama inference pipeline
- [ ] Multi-option evaluation
- [ ] RAG integration for rules
- [ ] Action justification

### Phase 4: End-to-End Integration (Weeks 4-5)
- [ ] Full game loop
- [ ] Voice communication
- [ ] Human feedback incorporation
- [ ] Testing with Alessandro

### Phase 5: World Models (Weeks 5-6)
- [ ] Data collection during games
- [ ] Open Dreamer training
- [ ] Simulation-based planning
- [ ] Continuous learning

### Phase 6: Research & Publication (Weeks 6+)
- [ ] Multi-player scenarios
- [ ] Emergent behavior analysis
- [ ] Research paper drafting

## Configuration

See `.env.example` for all configuration options:

```
QDRANT_HOST=localhost
QDRANT_PORT=6333
OLLAMA_MODEL=qwen2.5:72b
ELEVENLABS_API_KEY=sk_...
REACHY_HOST=localhost
```

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src

# Specific test file
pytest tests/test_setup_manager.py
```

## Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes and test: `pytest`
3. Commit: `git commit -m "Add feature X"`
4. Push: `git push origin feature/your-feature`
5. Open PR

## License

MIT License - See LICENSE file

## Citation

If you use this project in your research, please cite:

```bibtex
@software{laroccasilveira2025eldritch,
  title={Embodied Conversational AI for Cooperative Board Games},
  author={La Rocca Silveira, Alessandro},
  year={2025},
  url={https://github.com/your-username/eldritch-horror-reachy}
}
```

## Contact

Alessandro La Rocca Silveira  
AI Engineer @ Point of Rental  
[Your Email] | [LinkedIn] | [GitHub]

---

*Building the future of embodied AI through games* 🎲🤖
