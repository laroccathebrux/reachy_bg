# Eldritch Horror + Reachy Mini: Design Document

## Executive Summary

Building **Reachy Mini** into a cooperative game-playing agent for Eldritch Horror that:
1. Controls its own investigators (user executes moves)
2. Makes strategic decisions using local LLMs + world models
3. Communicates naturally, understanding multi-party conversation pragmatics
4. Learns during gameplay and fine-tunes post-game
5. Is a research platform for embodied conversational AI in games

---

## 1. System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        REACHY MINI AGENT                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────┐      │
│  │   PERCEPTION │  │  COGNITION    │  │   COMMUNICATION│      │
│  └──────────────┘  └───────────────┘  └────────────────┘      │
│        ↓                  ↓                     ↓              │
│   ┌─────────────────────────────────────────────────────┐     │
│   │ Vision         │ Strategy    │ Speech Out (TTS)    │     │
│   │ Speech (STT)   │ Planning    │ Natural Language    │     │
│   │ Board State    │ Reasoning   │ Personality         │     │
│   └─────────────────────────────────────────────────────┘     │
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐    │
│  │          CORE COMPONENTS (DETAILED)                   │    │
│  ├───────────────────────────────────────────────────────┤    │
│  │                                                        │    │
│  │ VISION SUBSYSTEM                                      │    │
│  │ ├─ Reachy Camera → SAM Segmentation                  │    │
│  │ ├─ Board State Parser (JSON)                         │    │
│  │ └─ State Synchronizer (with game reality)           │    │
│  │                                                        │    │
│  │ SPEECH SUBSYSTEM                                      │    │
│  │ ├─ Continuous STT (Whisper)                          │    │
│  │ ├─ Speaker Diarization (who speaks?)                 │    │
│  │ ├─ Addressee Detection (for Reachy?)                 │    │
│  │ ├─ Conversation State Tracker                         │    │
│  │ └─ Turn-Taking Decision (when to respond?)           │    │
│  │                                                        │    │
│  │ STRATEGIC REASONING                                   │    │
│  │ ├─ LLM (Ollama Qwen 2.5)                             │    │
│  │ ├─ RAG Engine (Qdrant)                               │    │
│  │ ├─ Action Generator (N candidates)                    │    │
│  │ ├─ World Model Simulator (Phase 5)                    │    │
│  │ └─ Evaluator (which action is best?)                 │    │
│  │                                                        │    │
│  │ SPEECH OUTPUT                                          │    │
│  │ ├─ Decision → Natural Language                        │    │
│  │ ├─ ElevenLabs TTS                                     │    │
│  │ └─ Personality Injection                              │    │
│  │                                                        │    │
│  │ LEARNING SYSTEM                                        │    │
│  │ ├─ Online: Collect (state, action, outcome)          │    │
│  │ ├─ Offline: Train world model on trajectories        │    │
│  │ └─ Feedback: Human corrections → context update      │    │
│  │                                                        │    │
│  └───────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐    │
│  │              PERSISTENT STATE                         │    │
│  ├───────────────────────────────────────────────────────┤    │
│  │                                                        │    │
│  │ Game State (current board)                           │    │
│  │ Game History (trajectory of moves)                    │    │
│  │ World Model (learned game dynamics)                   │    │
│  │ Conversation Context (who said what, when)           │    │
│  │ Learning Logs (for offline analysis)                  │    │
│  │                                                        │    │
│  └───────────────────────────────────────────────────────┘    │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. Game State Representation

Reachy needs to track the entire game state. Define it as JSON:

```json
{
  "timestamp": "2025-09-14T15:30:45Z",
  "round": 3,
  "turn": "reachy",
  
  "reachy_investigators": [
    {
      "name": "Roland Banks",
      "location": "Arkham_University",
      "sanity": 5,
      "health": 3,
      "items": ["Revolver", "Diary"],
      "status": "healthy"
    },
    {
      "name": "Daisy Walker",
      "location": "Miskatonic_River",
      "sanity": 2,
      "health": 1,
      "items": ["Dog Bone"],
      "status": "injured"
    }
  ],
  
  "human_investigators": [
    {
      "name": "Joe Diamond",
      "location": "Graveyard",
      "sanity": 4,
      "health": 2,
      "items": []
    }
  ],
  
  "board_state": {
    "doom_track": 5,
    "gates_open": 2,
    "gate_locations": ["Innsmouth", "Black_Goat"],
    "terror_track": 3,
    "investigation_progress": 8
  },
  
  "recent_events": [
    "Gate opened in Innsmouth",
    "Roland gained Revolver",
    "Daisy's sanity decreased by 1"
  ]
}
```

### Why This Matters:
- **Reachy's Decision Space**: Which of MY investigators moves? Where?
- **Shared State**: Reachy tracks what other players did
- **History**: Used for RAG context ("Last turn, Joe was attacked...")

---

## 3. Reachy's Decision Space (Action Space)

Reachy makes decisions about:

```python
class RearchyAction:
    """What Reachy can decide to do"""
    
    investigator: str  # "Roland Banks" or "Daisy Walker"
    action_type: str   # "move", "fight", "research", "heal"
    location: str      # Where to move/act
    item_usage: str    # What to use (optional)
    
    # Example: RearchyAction(
    #   investigator="Roland",
    #   action_type="fight",
    #   location="Innsmouth Gate",
    #   item_usage="Revolver"
    # )
```

### Reachy Does NOT Decide:
- Dice rolls (those are random/physical)
- Card draws (you handle)
- Other players' moves

### Reachy DOES Decide:
- Which investigator to move
- When to fight vs retreat
- Resource allocation ("Use the gun now or save it?")
- Cooperation strategy ("Joe, you investigate while I fight")

---

## 4. Conversational Pragmatics: The Turn-Taking Problem

This is the **hardest research problem**. Example scenario:

```
Turn 1: Alessandro: "Vou investigar em Arkham"
        → About Alessandro's move, not for Reachy → Reachy listens silently

Turn 2: Reachy's turn
        Reachy: "Vou lutar o gate em Innsmouth"
        
Turn 3: Alessandro: "Boa estratégia"
        → Opinion about Reachy, but not directed at Reachy → Reachy calmly confident
        
Turn 4: Alessandro: "E agora, você investe em defesa?"
        → Explicitly asking Reachy (implicit "E agora, você...") → Reachy responds
        
Turn 5: Alessandro: "Acho que aquele investigador..."
        → Alessandro thinking aloud, not asking Reachy → Reachy stays quiet
        
Turn 6: Alessandro: "Reachy, concordas que devíamos—"
        → Wake word + question → Reachy definitely responds
```

### Solution Architecture:

```python
class ConversationState:
    """Track who's saying what"""
    
    last_speaker: str           # "Alessandro"
    addressee_for_last: str     # "reachy", "self", "other"
    conversation_topic: str     # "strategy", "status", "question"
    reachy_is_active: bool      # Is it Reachy's turn to decide?
    
    # Decision: Should Reachy respond NOW?
    def should_reachy_respond(utterance: str) -> bool:
        """
        Logic:
        1. Is Reachy's turn? → YES (must decide on action)
        2. Is addressee Reachy? → YES (being asked)
        3. Is it question about Reachy's strategy? → MAYBE (depends on context)
        4. Is it addressing someone else? → NO
        """
```

### Pragmatic Understanding (not keyword matching):

We need to distinguish:

```
"Reachy deveria lutar" 
  → Suggestion (Reachy can disagree)

"Reachy, deveria lutar?"
  → Question (Reachy should respond)

"Deveria lutar?" (when Reachy is active player)
  → Implicit question (Reachy should respond)

"Aquele investigador..." (about Reachy's investigator)
  → Observation (Reachy might comment if relevant)
```

---

## 5. Strategic Decision Pipeline

When it's Reachy's turn:

```
GAME STATE OBSERVED
        ↓
┌─────────────────────────────────────┐
│ 1. CANDIDATE GENERATION (LLM)       │
│ Input: Current state + History      │
│ Output: 3 candidate actions         │
│                                     │
│ "Given current state, what are     │
│  reasonable actions for Reachy?"    │
└──────────────┬──────────────────────┘
               ↓
        ┌──────────────────┐
        │ Action A         │
        │ Action B         │
        │ Action C         │
        └──────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 2a. RULE CHECKING (RAG)             │
│ "Is this move legal?"               │
│ (Can this investigator move there?) │
└──────────────┬──────────────────────┘
               ↓
        ┌──────────────────┐
        │ Action A ✓       │
        │ Action B ✗ (ilegal)
        │ Action C ✓       │
        └──────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 2b. SIMULATION (World Model)        │
│ Phase 5: "What happens next?"       │
│                                     │
│ For each action:                    │
│  - Predict board state 1-3 turns    │
│  - Estimate doom/danger             │
│  - Calculate win probability        │
└──────────────┬──────────────────────┘
               ↓
        ┌──────────────────────────┐
        │ Action A: Doom +1        │
        │ Action C: Doom +3 (bad)  │
        └──────────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 3. EVALUATION (LLM + scoring)       │
│ Input: Candidates + predictions     │
│ "Which is best for team?"           │
│                                     │
│ Scoring:                            │
│  - Reduces doom?                    │
│  - Protects sanity?                 │
│  - Progresses toward victory?       │
└──────────────┬──────────────────────┘
               ↓
        ┌──────────────────┐
        │ BEST: Action A   │
        └──────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 4. JUSTIFICATION (LLM)              │
│ Input: Chosen action + reasoning    │
│ Output: Natural language explanation│
│                                     │
│ "Vou colocar o Roland em Arkham     │
│  porque preciso de pistas. Daisy    │
│  está ferida então fico defensivo." │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 5. COMMUNICATION (TTS)              │
│ Speak the decision with ElevenLabs  │
└──────────────┬──────────────────────┘
               ↓
        ACTION EXECUTED
```

---

## 6. Learning Pipeline: Online + Offline

### Online Learning (During Game)

```
Turn 1:
  Reachy state: { Roland: Arkham, Doom: 2 }
  Reachy action: "Move Roland to Graveyard"
  
  [Alessandro executes]
  
  Observed outcome: { Roland: Graveyard, Doom: 3, Found Clue }
  
  → Store: (state_1, action, state_2, outcome="found clue")
  
Turn 2:
  NEW state: { Roland: Graveyard, Doom: 3 }
  Reachy action: "Fight the gate"
  
  Observed outcome: { Roland: Defeated (sanity=0), Gate closes }
  
  → Store: (state_2, action, state_3, outcome="won fight")

[Continue until end of game]
```

**What Reachy learns online:**
- "That move worked!" or "That was bad"
- Immediate feedback loops
- Can adjust strategy mid-game

### Offline Learning (After Game)

```
After game ends:
  
  Trajectory = [
    (state_1, action_1, state_2, outcome_1),
    (state_2, action_2, state_3, outcome_2),
    ...
    (state_N, action_N, state_final, outcome_N)
  ]
  
  Open Dreamer Training:
    Input:  state_t, action_t
    Learn:  predict state_t+1
    Loss:   |predicted_state - actual_state|
  
  After training:
    Model now predicts: "If you fight here, you lose X sanity"
                        "If you investigate, you find Y pistes"
```

**What Reachy learns offline:**
- General game dynamics (not one-off)
- Emergent strategies ("investigate first, then fight")
- Cross-game patterns

---

## 7. Online vs Offline Learning in Practice

### Example: The Sanity Loss Puzzle

**Game 1:**
```
Reachy fights gate → loses 3 sanity
Online: "Ouch, that cost sanity"
Offline: Nothing special (one data point)
```

**Game 2:**
```
Reachy fights gate → loses 3 sanity
Reachy investigates → loses 1 sanity
Online: "Research is safer than fighting"
Offline: Two data points
```

**Game 3:**
```
Reachy fights gate → loses 1 sanity (different gate, different investigator)
Online: "Wait, this time only 1? Why?"
Offline: Three data points - model starts inferring "investigator strength matters"
```

**After 5-10 games:**
```
Offline fine-tuning:
World Model learns: "Roland (strength 4) loses less sanity fighting than Daisy (strength 1)"

Next game:
Reachy decides: "Roland should fight (only 1-2 sanity loss)"
               "Daisy should investigate (safer for her)"
```

**This is emergent strategy learning.**

---

## 8. Data Collection Format

For offline learning, we store:

```python
{
  "game_id": "20250914_001",
  "timestamp": "2025-09-14T15:30:00Z",
  "human_players": ["Alessandro"],
  "result": "victory",  # "victory", "defeat", "draw"
  "rounds_played": 5,
  
  "trajectory": [
    {
      "round": 1,
      "turn_order": ["reachy", "alessand"],
      
      "reachy_turn": {
        "state_before": {...},  # Full game state as JSON
        "action": "move Roland to Arkham",
        "action_type": "move",
        "investigator": "Roland",
        "location": "Arkham",
        
        "state_after": {...},   # Board after move
        "immediate_outcome": "found_clue",  # What happened
        
        "image_before": "path/to/board_before.png",
        "image_after": "path/to/board_after.png",
      },
      
      "human_turn": {
        "state_before": {...},
        "actions": [  # Multiple players possible
          {"player": "Alessandro", "action": "fight gate"}
        ],
        "state_after": {...},
        "immediate_outcome": "defeated_gate"
      }
    },
    
    {...more turns...}
  ],
  
  "game_summary": {
    "total_doom_prevented": 3,
    "total_clues_found": 5,
    "key_decisions": [
      "Early sanity preservation saved the game",
      "Coordinated investigation in mid-game was key"
    ]
  }
}
```

---

## 9. Conversation State Machine

Pseudo-code for when Reachy should speak:

```python
class ConversationManager:
    
    def process_utterance(self, speaker: str, text: str):
        """Decide if Reachy should respond"""
        
        # Step 1: Identify speaker
        diarized_speaker = self.diarizer.identify(text)  # "Alessandro"
        
        # Step 2: Who is being addressed?
        addressee = self.addressee_detector(text, history)
        # Returns: "reachy", "other_player", "self_reflection", "no_target"
        
        # Step 3: Is Reachy's turn?
        is_reachy_turn = (self.game_state.current_turn == "reachy")
        
        # Step 4: Classify utterance type
        utterance_type = self.classify_utterance(text)
        # Returns: "question", "statement", "opinion", "order", "observation"
        
        # Step 5: DECISION LOGIC
        
        if is_reachy_turn and utterance_type == "question" and addressee == "reachy":
            # "Reachy, qual é o seu movimento?"
            return self.respond_to_decision_question(text)
        
        elif addressee == "reachy" and utterance_type == "question":
            # "Reachy, você concorda?" (even if not Reachy's turn)
            return self.respond_to_strategy_question(text)
        
        elif is_reachy_turn and addressee in ["reachy", "no_target"]:
            # Reachy's turn, should announce decision
            return self.announce_decision()
        
        elif addressee == "reachy" and utterance_type == "order":
            # "Reachy, faz X" - acknowledge but maybe disagree
            return self.respond_to_order(text)
        
        else:
            # Not for Reachy → stay silent, listen
            self.update_conversation_context(speaker, text)
            return None
```

---

## 10. World Model: Online Inference Loop

Phase 5 - during game, Reachy uses world model to evaluate:

```python
class StrategicPlanner:
    
    def evaluate_action(self, action: RearchyAction, world_model):
        """Simulate what happens if we take this action"""
        
        # 1. Encode current state
        state_latent = world_model.encoder(self.game_state)
        
        # 2. Simulate 2-3 turns ahead
        # (current turn + next human turn + event/doom escalation)
        trajectory = world_model.rollout(
            state_latent,
            actions=[action, "human_actions", "random_event"],
            horizon=3
        )
        
        # 3. Decode final state
        predicted_state = world_model.decoder(trajectory[-1])
        
        # 4. Score: is final state better or worse?
        score = self.evaluate_outcome(
            current=self.game_state,
            predicted=predicted_state,
            objectives=[
                "minimize_doom",
                "preserve_sanity",
                "progress_victory"
            ]
        )
        
        return score
```

---

## 11. Dataset for World Model

During gameplay, collect:

```
(image_board_t, action_vector) → (image_board_t+1)

E.g.:
( Image("Roland in Arkham", "Doom=2"), 
  Action("move_Roland_to_Graveyard") )
  → Image("Roland in Graveyard", "Doom=2", "found_clue")
```

After ~50-100 game transitions:
- World model starts learning board dynamics
- Can predict "If you investigate here, ~60% chance of clue"
- Can predict "If you fight, doom +2"

---

## 12. Key Research Questions

1. **Turn-taking without wake words:**
   - How much context do we need to detect addressee?
   - Can embeddings + conversation history do it?

2. **Cooperative strategy emergence:**
   - What strategies emerge from world model training?
   - Do they match human intuitions?

3. **Learning efficiency:**
   - How many games until world model is useful?
   - When does online feedback outweigh offline?

---

## 13. Implementation Phases (Detailed)

### Phase 1: Vision (1-2 weeks)
- Reachy camera → SAM segmentation
- JSON game state parser
- Angle robustness

### Phase 2: Speech Understanding (2-3 weeks)
- STT (Whisper continuous)
- Diarization (who speaks?)
- Addressee detection (embedding-based)

### Phase 3: Strategy (3-4 weeks)
- Ollama Qwen inference
- Multi-option evaluation
- RAG for rules

### Phase 4: Voice Loop (4-5 weeks)
- End-to-end integration
- Play test with Alessandro
- Feedback incorporation

### Phase 5: World Models (5-6 weeks)
- Data collection during games
- Open Dreamer training
- Simulation-based planning

### Phase 6: Research & Iteration (6+ weeks)
- Multi-player handling
- Analysis & publication
- Emergent behavior documentation

---

## 14. Success Criteria

- [ ] Reachy makes coherent moves (legaland strategic)
- [ ] Reachy understands 80%+ of directedrequests
- [ ] Reachy communicates clearly via voice
- [ ] Reachy and Alessandro win a full cooperative game
- [ ] World model predictions match reality >70%
- [ ] Research paper drafted on pragmatic turn-taking in embodied AI

---

## 15. Next Steps

1. Define game state JSON schema precisely
2. Map out Reachy's exact investigators (which characters?)
3. Design conversation state machine flow
4. Prototype vision pipeline (Phase 1.1)
5. Build STT pipeline (Phase 2.1)

Ready?
