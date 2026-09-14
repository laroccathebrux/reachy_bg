# Full-Context Learning: Cooperative Strategy Emergence

The robot does not learn alone. It learns the dynamics of the whole table: what it did,
what the human did, what the Mythos deck did, and how those interacted. This is what makes
cooperative strategy emerge rather than solitary optimisation.

## Three levels of context

| Level | What is recorded | What can be learned |
|---|---|---|
| 1. Robot only | (state, robot action, next state) | "Travelling to a Gate space and fighting costs Sanity." |
| 2. Robot + immediate human response | + the human's actions in the same round | "When Jacqueline researches while I fight, the round nets a Clue and closes a Gate." |
| 3. Full context (ours) | + reasoning, requests, Mythos events, multi-round outcome | "Asking for cover *before* fighting changes what the human does; coordinated rounds prevent Doom." |

## Trajectory format

One JSON document per session under `data/game_logs/<session_id>/trajectory.jsonl`, one
line per round. Names and values follow [GAME_REFERENCE.md](GAME_REFERENCE.md).

```json
{
  "session_id": "20260914-2030",
  "round": 2,
  "ancient_one": "Cthulhu",
  "state_before": { "doom": 12, "omen": "green", "...": "..." },

  "robot_turn": {
    "investigator": "Lily Chen",
    "candidates": [
      {"action": "travel", "to": "Tokyo", "then": "combat"},
      {"action": "rest"},
      {"action": "acquire_assets"}
    ],
    "chosen": {"action": "travel", "to": "Tokyo", "then": "combat"},
    "reasoning": "A Deep One guards the Tokyo gate; Strength 4 makes the fight cheap and closing the gate protects the blue omen.",
    "coordination_request": "Jacqueline, can you pick up the Clue on Space 10 while I clear Tokyo?",
    "state_after": { "doom": 12, "...": "..." }
  },

  "human_turns": [
    {
      "player": "Alessandro",
      "investigator": "Jacqueline Fine",
      "actions": [{"action": "travel", "to": "Space 10"}, {"action": "rest"}],
      "stated_intent": "Getting that clue so we can start the mystery.",
      "responded_to_request": true,
      "state_after": { "...": "..." }
    }
  ],

  "encounters": [
    {"investigator": "Lily Chen", "type": "combat", "monster": "Deep One", "result": "defeated", "health_lost": 1, "sanity_lost": 0},
    {"investigator": "Jacqueline Fine", "type": "research", "result": "pass", "clues_gained": 1}
  ],

  "mythos": {
    "card": "Mysterious Lights",
    "icons": ["advance_omen", "spawn_clues"],
    "effects": ["omen -> blue", "doom advanced 1 (gate in Tokyo)", "clue spawned on Space 17"]
  },

  "state_after": { "doom": 11, "...": "..." },

  "round_outcome": {
    "doom_delta": -1,
    "clues_gained": 1,
    "gates_closed": 0,
    "health_lost": 1,
    "sanity_lost": 0,
    "mystery_progress": 1
  },

  "cooperation": {
    "request_made": true,
    "request_honoured": true,
    "roles": {"Lily Chen": "combat", "Jacqueline Fine": "research"},
    "synergy_score": 8.0,
    "note": "Complementary positioning; the omen advance was not covered."
  },

  "human_feedback": [
    {"speaker": "Alessandro", "text": "Next time close the gate before the omen turns blue.", "about": "robot_turn"}
  ]
}
```

The `cooperation.synergy_score` (0-10) starts as a rule-based score (complementary roles,
requests honoured, no duplicated actions, Doom prevented) and is later replaced by a learned
estimate.

## What the robot learns from full context

- **Location synergies**: which pairs of positions cover Gates and Clues at the same time.
- **Timing synergies**: fight first, research second, or the reverse, given the Omen.
- **Resource synergies**: which Assets and Spells combine across investigators.
- **Communication**: when a request changes the human's action and when it does not.
- **Failure modes**: both investigators on the same task, nobody near a matching Gate when
  the Omen advances.
- **The human's style**: research-heavy early, combat late, willingness to take Debt, and
  so on. Different humans produce different learned policies.

## Online loop (during a game)

After every round:

1. Write the round summary to `bg_sessions` (`src/rag/sessions.py`) so it is searchable
   during the game ("what happened the last time we faced Cthulhu?").
2. Update a short **lessons** note in the game state: two or three sentences the strategy
   prompt reads ("fighting the Deep One cost 1 Health, cheaper than expected"; "the human
   answers cover requests when asked one round ahead").
3. If the human corrected the robot, store the correction with the state it applied to.

## Offline loop (between games)

1. Build the dataset from every `trajectory.jsonl`: (state_t, robot action, human actions,
   mythos) -> state_t+1, plus the round outcome.
2. Train the **world model**: an encoder for the structured state, a dynamics model that
   predicts the next state and the round outcome, and a **human policy head** that predicts
   the human's likely actions given the state and the robot's request. A Dreamer-style
   latent model is the target; a gradient-boosted or small transformer baseline comes first
   because the dataset will be tiny for many games.
3. Evaluate on held-out rounds: next-state accuracy, outcome accuracy, human-action
   accuracy.
4. Plug the model into the strategy pipeline as a rollout simulator (Phase 5).

## Expected progression

| Games | Behaviour |
|---|---|
| 1-3 | Plays by rules + LLM judgement; asks for nothing. |
| 4-6 | Notices the human's positioning; lessons notes start shaping choices. |
| 7-9 | Makes targeted coordination requests; predicts whether they will be honoured. |
| 10+ | Proposes plans the human had not considered; world-model rollouts change decisions. |

## Research questions this answers

- How do AI agents learn to cooperate with humans in complex games from observed actions
  and outcomes, without explicit programming of the cooperation?
- Does the robot learn *when* to communicate, not only *how*?
- How many games until cooperation emerges, and does it transfer to a new human?
