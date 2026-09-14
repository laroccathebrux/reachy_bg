# Transferring Project to Your Mac

Your project has been initialized in Claude Code. Here's how to get it on your Mac and start developing.

---

## Option 1: Using GitHub (Recommended)

This is the cleanest way to get everything on your Mac and stay synchronized.

### Step 1: Create a GitHub Repository

1. Go to https://github.com/new
2. Repository name: `eldritch-horror-reachy`
3. Description: "Embodied Conversational AI Agent for Eldritch Horror"
4. Choose **Private** (if you prefer)
5. Click "Create repository"
6. Copy the repository URL (should look like `https://github.com/your-username/eldritch-horror-reachy.git`)

### Step 2: Push from Claude Code

In Claude Code terminal:

```bash
cd /home/claude/eldritch-horror-reachy

# Add GitHub as remote
git remote add origin https://github.com/YOUR-USERNAME/eldritch-horror-reachy.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Clone on Your Mac

On your Mac:

```bash
# Navigate to your Documents
cd "/Users/alessandrolaroccasilveira/Documents/Documents - USSILVEIRAAXWR2/"

# Clone the repository
git clone https://github.com/YOUR-USERNAME/eldritch-horror-reachy.git
cd eldritch-horror-reachy

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env from example
cp .env.example .env
# Edit .env with your actual API keys and settings
```

### Step 4: Keep Synchronized

When you make changes in Claude Code:

```bash
# In Claude Code
cd /home/claude/eldritch-horror-reachy
git add .
git commit -m "Your message"
git push origin main
```

Then on your Mac:

```bash
git pull origin main
```

---

## Option 2: Manual Transfer (Without GitHub)

If you prefer not to use GitHub yet:

### Step 1: Download Files from Claude Code

In Claude Code, create a tar archive:

```bash
cd /home/claude
tar -czf eldritch-horror-reachy.tar.gz eldritch-horror-reachy/
```

Then download the file from Claude Code's file browser.

### Step 2: Extract on Your Mac

```bash
# Navigate to your Documents
cd "/Users/alessandrolaroccasilveira/Documents/Documents - USSILVEIRAAXWR2/"

# Extract archive
tar -xzf eldritch-horror-reachy.tar.gz

# Navigate to project
cd eldritch-horror-reachy

# Initialize git locally
git init
git add .
git commit -m "Initial commit from Claude Code"
```

### Step 3: Optional - Add to GitHub Later

```bash
# Add remote after creating GitHub repo
git remote add origin https://github.com/YOUR-USERNAME/eldritch-horror-reachy.git
git branch -M main
git push -u origin main
```

---

## Setup Your Mac Environment

### Prerequisites

Make sure you have these installed:

```bash
# Check Python version (should be 3.9+)
python3 --version

# Install Homebrew if needed (for other tools)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Install Ollama

1. Download from https://ollama.ai
2. Install and run
3. Pull Qwen 2.5:
   ```bash
   ollama pull qwen2.5:72b
   ```

### Install Qdrant

Using Docker (recommended):

```bash
# If you don't have Docker, install from https://docs.docker.com/desktop/install/mac-install/

# Run Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### Create .env File

```bash
cd /Users/alessandrolaroccasilveira/Documents/Documents - USSILVEIRAAXWR2/eldritch-horror-reachy

# Copy template
cp .env.example .env

# Edit with your values
nano .env
# Or use VSCode
code .env
```

Your .env should look like:

```
QDRANT_HOST=localhost
QDRANT_PORT=6333
OLLAMA_MODEL=qwen2.5:72b
OLLAMA_BASE_URL=http://localhost:11434
ELEVENLABS_API_KEY=sk_your_actual_key_here
ELEVENLABS_VOICE_ID=your_voice_id
REACHY_HOST=localhost
REACHY_PORT=50051
```

### Verify Everything Works

```bash
cd /Users/alessandrolaroccasilveira/Documents/Documents - USSILVEIRAAXWR2/eldritch-horror-reachy

# Activate venv
source venv/bin/activate

# Test imports
python -c "import ollama; print('Ollama: OK')"
python -c "from qdrant_client import QdrantClient; print('Qdrant: OK')"
python -c "from elevenlabs import ElevenLabs; print('ElevenLabs: OK')"

# Run tests
pytest
```

---

## Project Structure on Your Mac

```
~/Documents/Documents - USSILVEIRAAXWR2/eldritch-horror-reachy/
├── src/
│   ├── config.py          # Configuration management
│   ├── logger.py          # Logging setup
│   ├── vision/            # Vision module (Phase 1)
│   ├── speech/            # Speech module (Phase 2)
│   ├── strategy/          # Strategy module (Phase 3)
│   ├── learning/          # Learning module (Phase 5)
│   └── integration/       # Integration (Phase 4)
├── docs/
│   ├── DESIGN_DOCUMENT.md
│   ├── PHYSICAL_SETUP.md
│   ├── FULL_CONTEXT_LEARNING.md
│   └── SETUP_PROTOCOL.md
├── tests/                 # Test files
├── data/                  # Data directories (auto-created)
├── requirements.txt
├── setup.py
├── README.md
├── GETTING_STARTED.md     # ← Read this next!
└── .env                   # ← Create this from .env.example
```

---

## Next Steps After Transfer

1. **Read GETTING_STARTED.md**
   ```bash
   cat GETTING_STARTED.md
   ```

2. **Read DESIGN_DOCUMENT.md**
   ```bash
   code docs/DESIGN_DOCUMENT.md
   ```

3. **Set up your Qdrant database** with Eldritch Horror data

4. **Verify Ollama** runs Qwen 2.5:
   ```bash
   curl http://localhost:11434/api/tags
   ```

5. **Start developing Phase 1** (Vision Module)

---

## Workflow While Developing

### In Claude Code
```bash
cd /home/claude/eldritch-horror-reachy
# Make changes
git add .
git commit -m "Clear message"
git push origin main
```

### On Your Mac
```bash
cd ~/Documents/Documents\ -\ USSILVEIRAAXWR2/eldritch-horror-reachy
git pull origin main
# Test locally
pytest
# Continue developing
```

---

## Troubleshooting Transfer

### "Repository already exists"
```bash
rm -rf /home/claude/eldritch-horror-reachy/.git
cd /home/claude/eldritch-horror-reachy
git init
# Then push to GitHub as Step 2 above
```

### Files not transferring
```bash
# Double-check all files are committed
cd /home/claude/eldritch-horror-reachy
git status  # Should show "nothing to commit, working tree clean"
git log     # Should show your commits
```

### Permission denied on Mac
```bash
# Make scripts executable
chmod +x /Users/alessandrolaroccasilveira/Documents/Documents\ -\ USSILVEIRAAXWR2/eldritch-horror-reachy/venv/bin/*
```

---

## You're Ready! 🚀

Once you've followed these steps:

1. ✅ Project is on your Mac
2. ✅ Virtual environment is set up
3. ✅ Dependencies are installed
4. ✅ .env is configured
5. ✅ Services are running (Ollama, Qdrant)
6. ✅ Git is initialized and synced

**Now go read GETTING_STARTED.md and start Phase 1!**

---

## Git Cheat Sheet

```bash
# Check what changed
git status

# See your commits
git log

# Make a new branch for features
git checkout -b feature/your-feature-name

# Switch back to main
git checkout main

# Pull latest from GitHub
git pull origin main

# Push your changes
git push origin main

# View changes before committing
git diff

# Undo last commit (if not pushed)
git reset --soft HEAD~1
```

---

Questions? Check the docs or reach out! 💬
