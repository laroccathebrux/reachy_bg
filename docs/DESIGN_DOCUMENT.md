# Design Document

Reachy Mini as a cooperative Eldritch Horror player and as a research platform for embodied
conversational AI. This is the architecture the code follows; game facts are in
[GAME_REFERENCE.md](GAME_REFERENCE.md), audio in [SPEECH_PIPELINE.md](SPEECH_PIPELINE.md),
model choice in [MODEL_SIZING.md](MODEL_SIZING.md), learning in
[FULL_CONTEXT_LEARNING.md](FULL_CONTEXT_LEARNING.md).

## 1. Goals

1. The robot **controls its own investigator** and decides every one of its actions; the
   human moves the pieces, rolls the dice and draws the cards for it.
2. It **talks like a player**: announces and justifies decisions, answers rules questions from
   the rulebook, proposes coordination, and stays quiet when not addressed.
3. It **learns**: during a game from outcomes and human corrections, between games from
   full trajectories that include the human's moves.
4. It is a **research platform** on three questions: turn-taking without wake words in a
   multi-party conversation, emergence of cooperative strategy between a human and an AI,
   and how many games a world model needs before it is useful.

## 2. System overview

```
                 +-----------------------------------------------------------+
                 |                     REACHY MINI AGENT                     |
                 +-----------------------------------------------------------+
 PERCEPTION      |  vision/    board capture, piece detection, board state   |
                 |  speech/    Mac mic -> ASR + diarization -> addressee     |
 COGNITION       |  strategy/  game state, rules RAG, candidate actions, LLM |
                 |  learning/  trajectory logging, world model, replay       |
 COMMUNICATION   |  integration/  conversation state machine, TTS, gaze,    |
                 |                game loop                                  |
                 +-----------------------------------------------------------+
        |                    |                     |                    |
   Reachy camera        Mac microphone        Ollama (Qwen 3.6)     Reachy speaker
   Reachy head (gaze)   diarizer sidecar      Qdrant (bg_*)          ElevenLabs / local TTS
```

Every module talks to the others through plain Python data (pydantic models) and an event
queue; no module imports another module's internals. External processes: the `reachy-mini`
daemon (robot), Ollama (LLM + embeddings), Qdrant (vector store), the live diarizer sidecar.

## 3. Game state

The single source of truth during a session, kept as a pydantic model in memory and
persisted to `data/game_logs/<session_id>/state.json` after every change. All names follow
[GAME_REFERENCE.md](GAME_REFERENCE.md).

```json
{
  "session_id": "20260914-2030",
  "ancient_one": "Cthulhu",
  "doom": 11,
  "omen": "green",
  "round": 3,
  "phase": "action",
  "lead_investigator": "Jacqueline Fine",
  "active_investigator": "Lily Chen",
  "active_mystery": "The Deep One Heretic",
  "mysteries_solved": 0,
  "investigators": {
    "Lily Chen": {
      "controller": "robot",
      "space": "Shanghai",
      "health": 5, "health_max": 6, "sanity": 6, "sanity_max": 6,
      "skills": {"lore": 2, "influence": 2, "observation": 2, "strength": 4, "will": 3},
      "improvements": {"strength": 1},
      "assets": ["Protective Amulet", "Lucky Rabbit's Foot"],
      "artifacts": [], "spells": [], "conditions": [],
      "clues": 1, "train_tickets": 0, "ship_tickets": 1,
      "delayed": false
    },
    "Jacqueline Fine": {"controller": "Alessandro", "space": "Space 5", "...": "..."}
  },
  "board": {
    "gates": [{"space": "Tokyo", "omen": "blue"}],
    "monsters": [{"name": "Deep One", "space": "Tokyo", "damage_taken": 0}],
    "clues": ["Space 10"],
    "expedition": "The Himalayas",
    "rumors": [],
    "eldritch_tokens": {"Space 3": 1},
    "reserve": ["Kerosene", ".38 Revolver", "Personal Assistant", "Arcane Manuscripts"]
  },
  "pending": null,
  "notes": []
}
```

Changes arrive as **patches** (what changed, not the whole state): "Lily loses 1 Health",
"a Gate opened in Tokyo with a Deep One". The vision module proposes patches for tokens it
can see; the speech module proposes patches from what players say; the human confirms
anything ambiguous.

## 4. The robot's decision space

The robot decides, for its investigator only:

- which action to take (Travel, Rest, Trade, Prepare for Travel, Acquire Assets, Component
  Action) and its parameters (destination, tickets, purchase, target);
- which encounter to take when there is a choice, and how to handle combat choices;
- whether to spend Clues on rerolls, use Spells or Assets;
- what to ask of the human ("can you cover Tokyo?"), and when to say nothing.

The robot never decides dice results, card draws, or other players' moves, but it tracks all of
them.

## 5. Strategic decision pipeline

Runs when it is the robot's turn or when it is asked what it would do.

```
game state + last rounds (bg_sessions) + active Mystery
        |
        v
 1. Candidate generation (LLM, structured output): 3-5 legal-looking actions
        |
        v
 2. Legality check: rules retrieved from bg_rules for each candidate
    ("can Lily travel two spaces with one ship ticket?") + hard checks in code
    (adjacency from the board map, once-per-round actions, Monster on space)
        |
        v
 3. Evaluation
    Phase 3: LLM scoring against explicit objectives (doom pressure, clue progress,
             health/sanity margin, coordination with the human's stated plan)
    Phase 5: world-model rollouts 1-3 turns ahead replace or complement the LLM score
        |
        v
 4. Justification (LLM): two sentences a teammate would say
        |
        v
 5. Speak (TTS through the robot), log decision + rationale to the trajectory
```

Structured outputs are enforced with JSON schemas; the LLM never mutates the state directly.

## 6. Conversation: when does the robot speak?

The hardest problem. There is no wake word; several people talk; some sentences are about
the robot without being for it.

```
utterance (text, speaker, timing)
   |
   +--> addressee classifier -> robot | other player | self-talk | table (open)
   +--> utterance type       -> question | request | statement | thinking aloud
   +--> game context         -> whose turn, current phase, pending question from robot
   |
   v
 policy:
   robot's turn and open/robot addressee     -> announce or answer
   addressed to robot, question or request   -> answer
   addressed to another player               -> stay silent, update context
   thinking aloud about the robot's plan     -> maybe one short remark, at most once
   rules dispute at the table                -> offer the rule (bg_rules) when asked or when
                                                two players disagree twice
```

The classifier starts rule-based (name mentions, second person while it is the robot's turn,
questions right after the robot spoke) and is replaced by an LLM classifier over the last
few utterances once we have logged conversations to evaluate it. Every decision to speak or
stay silent is logged with its inputs; this log is the research dataset.

The table is bilingual (Portuguese and English). Each utterance carries its detected
language; the robot answers in that language and speaks it with the native voice for it
(see [SPEECH_PIPELINE.md](SPEECH_PIPELINE.md), "Bilingual operation").

Example (English for documentation; runtime language follows the speaker):

```
Alessandro: "I'll go research in Space 10."          -> other player's move: silent
Alessandro: "Reachy, are you fighting that Deep One?" -> robot addressed: answer
Alessandro: "Hmm, maybe I should keep the Clue..."    -> thinking aloud: silent
Alessandro: "What do you think?"                       -> follow-up to robot: answer
```

## 7. Vision

The camera in the robot's head looks down at the table when the robot needs to check the
board and looks at the speaker otherwise (gaze is part of communication). Board reading:

1. **Gaze**: turn face tracking off, tilt the head to the table pose, wait for settling,
   grab frames, choose the sharpest.
2. **Detection**: open-vocabulary detector (YOLO-World) with text prompts for the game's
   pieces (board, cards, dice, tokens, investigator markers), plus a reference gallery of
   photographed pieces matched by image embedding to name specific tokens. A small custom
   detector trained on photos of the real pieces is the upgrade path.
3. **Board mapping**: a homography from the detected board corners to a canonical map lets
   detected tokens be assigned to spaces.
4. **State patches**: detections become proposed patches; anything below confidence is
   confirmed verbally ("I see a gate in Tokyo, right?").

Vision is assistive: the verbal channel is always able to correct or replace it.

## 8. Learning

Two loops, detailed in [FULL_CONTEXT_LEARNING.md](FULL_CONTEXT_LEARNING.md):

- **Online**: after every round the robot writes a summary to `bg_sessions` and updates a
  short "what worked / what did not" note used by the strategy prompt.
- **Offline**: every session produces a trajectory of (state, robot action, human actions,
  events, next state). A world model is trained on these trajectories to predict the next
  state and the human's likely actions; the strategy pipeline uses it for rollouts.

## 9. Repository layout

```
src/config.py, src/logger.py       settings and logging
src/rag/                           embeddings, Qdrant store, rules ingestion, knowledge
                                   migration, session memory  (Phase 0, done)
src/vision/                        capture, detection, board mapping        (Phase 1)
src/speech/                        mic capture, ASR, diarizer client, addressee, TTS (Phase 2)
src/strategy/                      game state, rules checks, candidate generation, evaluation (Phase 3)
src/integration/                   conversation state machine, gaze, game loop (Phase 4)
src/learning/                      trajectory logging, world model, replay  (Phase 5)
tools/live-diarizer/               sidecar process with its own venv       (Phase 2)
tests/                             pytest; anything touching Ollama/Qdrant is skipped when offline
docs/                              this document and its companions
data/                              runtime data, git-ignored (rulebook PDFs, captures, logs, models)
```

## 10. Research questions and how they will be measured

| Question | Metric | Data |
|---|---|---|
| Turn-taking without wake words | precision / recall of "should speak" decisions against human annotation | speak/silence log per utterance |
| Cooperative strategy emergence | synergy score per round, win rate, number of coordination requests over sessions | trajectories in `bg_sessions` and game logs |
| World model usefulness | next-state prediction accuracy vs number of games; win rate with vs without rollouts | offline dataset |

## 11. Success criteria

- [ ] The robot plays a full 2-player game making only legal moves.
- [ ] 80% or more of utterances directed at the robot get a response, and fewer than 10% of
      the others do.
- [ ] Rules questions are answered from `bg_rules` with the section and page.
- [ ] A game is won together.
- [ ] World model next-state prediction above 70% on held-out rounds.
- [ ] A paper draft on pragmatic turn-taking in embodied game play.
