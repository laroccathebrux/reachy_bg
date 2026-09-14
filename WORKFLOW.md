# Development Workflow

## 📍 Workspace Setup

**Primary Development**: Your Mac  
```
~/Documents/Documents - USSILVEIRAAXWR2/reachy_bg/
```

**Claude Code Development**: Sync repository  
```
/home/claude/reachy_bg_dev/
```

**Repository**: https://github.com/laroccathebrux/reachy_bg

---

## 🔄 Daily Development Cycle

### 1️⃣ Start Your Day

**On your Mac:**
```bash
cd ~/Documents/Documents\ -\ USSILVEIRAAXWR2/reachy_bg
source venv/bin/activate
git pull origin main
```

**On Claude Code:**
```bash
cd /home/claude/reachy_bg_dev
git pull origin main
```

---

### 2️⃣ Request New Feature from Claude Code

**In this conversation:**
```
Create [feature/module], add tests, then commit and push
```

**Claude Code will:**
- Pull latest from main
- Create/modify files
- Test locally
- Commit with clear message
- Push to GitHub

---

### 3️⃣ Sync Changes to Your Mac

**On your Mac:**
```bash
git pull origin main
```

**Files appear automatically**

---

### 4️⃣ Local Development & Testing

**On your Mac:**
```bash
# Edit files
code src/

# Run tests
pytest tests/

# Commit locally
git add .
git commit -m "Feature: clear description"
git push origin main
```

**Claude Code gets update:**
```bash
cd /home/claude/reachy_bg_dev
git pull origin main
```

---

## 📋 Common Commands

### Your Mac (Development)

```bash
# Navigate to project
cd ~/Documents/Documents\ -\ USSILVEIRAAXWR2/reachy_bg

# Activate environment
source venv/bin/activate

# Pull latest from GitHub
git pull origin main

# Make changes and test
code src/
pytest

# Commit and push
git add .
git commit -m "Feature: description"
git push origin main

# Check status
git status
git log --oneline -5
```

### Claude Code (Generation & Sync)

```bash
# Navigate to dev copy
cd /home/claude/reachy_bg_dev

# Sync with your Mac
git pull origin main

# After making changes
git add .
git commit -m "Feature: description"
git push origin main
```

---

## 🎯 Typical Workflow Session

### Request → Generate → Test → Sync → Develop

```
1. You (Mac):      git pull origin main
2. You:            Request feature to Claude Code
3. Claude Code:    git pull origin main
4. Claude Code:    Create module + tests
5. Claude Code:    git add . && git commit && git push
6. You (Mac):      git pull origin main
7. You:            Run pytest, test locally
8. You:            Make adjustments if needed
9. You:            git push origin main
10. Claude Code:   git pull origin main (ready for next feature)
```

---

## ⚡ Quick Checklist

### Before Starting Claude Code Task
- [ ] `git pull origin main` on Mac
- [ ] Venv activated on Mac
- [ ] Clear description of what you want

### Before Starting Local Work
- [ ] `git pull origin main` on Mac
- [ ] Venv activated
- [ ] Latest from Claude Code fetched

### Before Committing
- [ ] Tests pass: `pytest`
- [ ] Code follows structure: `ls -la src/*/`
- [ ] Clear commit message
- [ ] No `.env` secrets in commit

### Before Requesting Feature
- [ ] Describe clearly what you need
- [ ] Mention which phase (if applicable)
- [ ] Mention if you want tests included

---

## 🔀 Git Branch Strategy

For now: Single branch `main`

When scaling, use:
```
main          (stable)
├── phase-1   (features for Phase 1)
├── phase-2   (features for Phase 2)
└── feature/* (experimental)
```

---

## 📤 Push & Pull Reminders

### Your Mac Pushes To GitHub
```bash
git push origin main
```

### Claude Code Pulls From GitHub
```bash
git pull origin main
```

### Your Mac Receives Updates
```bash
git pull origin main
```

### Rule: Always Pull Before You Push
```bash
git pull origin main
git add .
git commit -m "Feature: ..."
git push origin main
```

---

## 🆘 If Conflicts Occur

```bash
# On your Mac
git status  # Shows conflicts

# Choose yours or theirs
git checkout --ours src/file.py    # Keep your version
git checkout --theirs src/file.py  # Keep Claude Code version

# Then commit
git add .
git commit -m "Resolve conflict: description"
git push origin main
```

---

## 📊 Check Project Status

```bash
# On your Mac
cd ~/Documents/Documents\ -\ USSILVEIRAAXWR2/reachy_bg

# See recent activity
git log --oneline -10

# Check current branch
git branch

# See uncommitted changes
git status

# See what changed in last commit
git show HEAD
```

---

## 🚀 Ready to Start?

1. **On your Mac:**
   ```bash
   cd ~/Documents/Documents\ -\ USSILVEIRAAXWR2/reachy_bg
   git status
   ```

2. **Request Phase 1 or specific feature** to Claude Code

3. **Claude Code creates → You test → You push**

4. **Iterate!**

---

## 📚 Related Files

- `CLAUDE_CODE_SETUP_PROMPT.md` - Setup prompt for Claude Code
- `GETTING_STARTED.md` - Quick start guide
- `PROJECT_STATUS.md` - Current project status
- `DESIGN_DOCUMENT.md` - Architecture overview

---

*Last Updated: September 14, 2026*  
*Workflow: Mac (Primary) ← → Claude Code (Generation)*
