# Development Workflow

One workspace, one branch, two ways of editing it: the owner in an editor, and Claude Code
sessions opened in the same directory. GitHub is the sync point between machines and the
backup.

## Workspace

```
/Users/alessandrolaroccasilveira/Documents/Documents - USSILVEIRAAXWR2/reachy_bg
remote: ssh://git@github.com/laroccathebrux/reachy_bg.git   (branch: main)
```

Claude Code reads [CLAUDE.md](CLAUDE.md) automatically when a session starts in this
directory, so no setup prompt has to be pasted. A task request is enough:

```
Sync and build Phase 1.1: board capture with sharpest-of-three selection, with tests.
```

## What a session does, every time

```bash
git pull origin main
uv sync
# ... implement, in English, with tests ...
uv run pytest
uv run ruff check src tests
git add -A
git commit -m "Short imperative message in English"
git push origin main
git log --oneline -3 && git status
```

Commits go straight to `main`. No feature branches or pull requests while the project is a
one-person effort; if that changes, branch per phase.

## The owner's side

```bash
cd "/Users/alessandrolaroccasilveira/Documents/Documents - USSILVEIRAAXWR2/reachy_bg"
git pull origin main
source .venv/bin/activate      # or prefix commands with `uv run`
```

Edit, test, commit, push the same way. Always pull before starting and before pushing.

## Rules that are easy to forget

- **English only in the repository.** Conversation with Claude may be in Portuguese; files
  are not. Check a diff for stray Portuguese before committing.
- **Never copy code from `../reachy`.** Read it for lessons, write ours from scratch.
- **Keep the SSH remote form.** The global git config rewrites `git@github.com:` URLs to
  HTTPS, and the HTTPS credential in the keychain belongs to a different GitHub account, so
  pushes over HTTPS are rejected with 403. `ssh://git@github.com/...` bypasses the rewrite.
- **Do not commit** `.env`, `data/`, `app.log`, the rulebook PDFs, or model weights.
- **Do not upgrade the robot SDK casually**; it pins numpy and GStreamer and is checked
  against a physical robot.

## Conflicts

Happen only if both sides edit without pulling. Resolve in the editor, run the tests, commit
the merge, push. When unsure which side is right, keep the version with passing tests.

## Checking state

```bash
git log --oneline -10
git status
uv run pytest -q
curl -s http://127.0.0.1:6333/collections | python3 -m json.tool | grep name
```
