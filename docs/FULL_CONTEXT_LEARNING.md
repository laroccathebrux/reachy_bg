# Full Context Learning: Cooperative Strategy Emergence

## The Core Innovation

Instead of learning in isolation, Reachy learns the **full game dynamics** including how human players' actions affect Reachy's success and vice versa.

This is **multi-agent cooperative learning**.

---

## 1. What "Full Context" Means

### Level 1 (Incomplete): Reachy-Only

```python
trajectory = [
  (state_t, 
   reachy_action="move Roland to Arkham",
   state_t+1,
   outcome="found clue")
]

Reachy learns: "If I move to Arkham, I find clues"
Problem: Ignores why Alessandro positioned Joe in Graveyard
```

### Level 2 (Better): Reachy + Immediate Human Response

```python
trajectory = [
  (state_t, 
   reachy_action="move Roland to Arkham",
   human_actions={"Alessandro": "move Joe to Graveyard"},
   state_t+1,
   outcome_reachy="found clue",
   outcome_human="found artifact",
   combined_outcome="both made progress")
]

Reachy learns: "My move + Alessandro's move = synergy"
Problem: Still doesn't capture long-term strategy
```

### Level 3 (Complete): Full Cooperative Context

```python
trajectory = [
  {
    "round": 1,
    "phase": "reachy_turn",
    "state_before": {...full game state...},
    "reachy_investigator": "Roland Banks",
    "reachy_action": "move to Arkham",
    "reachy_reasoning": "High probability of clues in Arkham",
    
    "state_after_reachy": {...},
    
    "human_turns": [
      {
        "player": "Alessandro",
        "investigator": "Joe Diamond",
        "action": "move to Graveyard",
        "reasoning": "Cover Reachy's weakness (Joe's lore 4)",
        "state_after": {...}
      }
    ],
    
    "events": [
      "Reachy found clue in Arkham",
      "Alessandro found ally in Graveyard",
      "Doom track: no increase (avoided by cooperation)"
    ],
    
    "combined_outcome": {
      "doom_prevented": 1,
      "synergy_score": 8.5,  # How well coordinated?
      "cooperativeness": "high",
      "strategic_alignment": "both protecting each other's weakness"
    },
    
    "long_term_impact": {
      "next_turn_reachy_options": [...],  # Affected by what just happened
      "probability_victory_next_3_turns": 0.72
    }
  },
  
  {
    "round": 1,
    "phase": "event_phase",
    "event": "Gate opens in Innsmouth",
    "state_after_event": {...},
    "players_affected": ["reachy", "Alessandro"],
    "how_synergy_helped": "Joe's ally allows re-roll, saves Reachy's position"
  },
  
  {...more rounds...}
]
```

**What Reachy learns:**
- "When I move to location X, and Alessandro covers position Y, doom +2 happens"
- "Joe's lore synergizes with Roland's combat" 
- "If we both position defensively, we prevent 3 doom per round"
- "Alessandro's strategy predicts what threats come next"

---

## 2. World Model with Cooperative Context

Instead of predicting only game consequences, world model learns **human behavior too**.

```python
class CooperativeWorldModel:
    """Learn both game dynamics AND player strategies"""
    
    def predict_next_state(self, 
                          current_state,
                          reachy_action,
                          human_actions,  # ← NEW
                          game_events):
        """
        Predict state after:
          1. Reachy's move
          2. Human players' moves
          3. Random events
          4. Doom escalation
        """
        
        # Encode everything
        state_latent = self.encoder(current_state)
        reachy_emb = self.action_encoder(reachy_action)
        human_embs = [self.action_encoder(a) for a in human_actions]
        
        # Multi-agent dynamics
        # "If Reachy does this AND Alessandro does that, result is..."
        combined_action = self.fuse_actions([reachy_emb] + human_embs)
        
        # Predict next state
        next_state_latent = self.dynamics_model(
            state_latent,
            combined_action,
            sequence_length=3  # Next 3 turns
        )
        
        # Decode
        predicted_state = self.decoder(next_state_latent)
        
        return predicted_state
    
    def extract_cooperation_patterns(self, trajectory):
        """
        Analyze trajectory to find synergies:
        - Which combinations of moves work well?
        - How does human strategy enable Reachy's success?
        - What patterns repeat?
        """
        
        patterns = []
        
        for i in range(len(trajectory) - 3):
            reachy_move = trajectory[i]["reachy_action"]
            human_moves = trajectory[i]["human_actions"]
            outcome = trajectory[i]["combined_outcome"]
            
            # Pattern: This combo of moves → this outcome
            pattern = {
                "reachy_action": reachy_move,
                "human_coverage": human_moves,
                "synergy_score": outcome["synergy_score"],
                "success": outcome["doom_prevented"] > 0
            }
            
            patterns.append(pattern)
        
        return patterns
```

---

## 3. Concrete Example: Learning Synergy

### Game 1: Reachy Learns Nothing (No Context)

```
Round 2, Reachy's turn:
  State: Doom 4, Roland health 3, Joe health 2
  
  Reachy: "I'll fight the gate"
  Result: Wins fight, Roland takes 2 sanity loss
  
  Learning: "Fighting gates damages sanity"
  
No context about:
  - Alessandro positioning Joe for defense
  - How Joe's health affects Reachy's positioning
  - That Reachy should have retreated earlier
```

### Game 5: Reachy Learns Patterns (Full Context)

```
Round 2, Reachy's turn (similar state as Game 1):
  State: Doom 4, Roland health 3, Joe health 2
  
  Reachy: "I'll fight the gate"
  
  [Meanwhile, Alessandro: "I'll move Joe to defend"]
  
  Result: Wins fight, Roland takes 1 sanity loss (Joe's protection worked)
  
  Context stored:
  {
    "reachy_action": "fight",
    "human_support": {"Alessandro": "defend Joe"},
    "outcome": {
      "sanity_loss": 1,  # Less than normal!
      "synergy": "Joe's positioning reduced damage"
    }
  }
  
  Learning: "Fighting while Alessandro defends = optimal"
```

### Game 10: Reachy Predicts & Suggests (Emergent Strategy)

```
Round 2, Reachy's turn:
  Reachy sees: Doom 4, Roland health 3, Joe health 2
  
  World model simulates:
    - "If I fight (alone)": 2 sanity loss, gate closes
    - "If I fight (Alessandro defends)": 1 sanity loss, gate closes
  
  Reachy PROACTIVELY says:
  "Alessandro, se você defender Joe aqui, eu ganho força para lutar.
   Você consegue? Vou tentar fechar o gate se você me cobrir."
  
  Alessandro: "Sim, vou mover Joe para cá."
  
  Result: Perfectly coordinated combat, minimal damage
  
  Learning: "Cooperative communication > isolated optimization"
```

---

## 4. Data Format: Full Context Trajectory

```json
{
  "game_id": "20250914_cthulhu_001",
  "outcome": "victory",
  "total_rounds": 5,
  
  "player_team": {
    "reachy": {
      "investigator": "Roland Banks",
      "stats": {"strength": 4, "lore": 2, "will": 2}
    },
    "human": {
      "investigator": "Joe Diamond",
      "controller": "Alessandro",
      "stats": {"strength": 3, "lore": 4, "will": 3}
    }
  },
  
  "rounds": [
    {
      "round_number": 1,
      "doom_at_start": 2,
      
      "reachy_turn": {
        "state_before": {
          "investigator_positions": {"Roland": "Arkham", "Joe": "Library"},
          "sanity": {"Roland": 5, "Joe": 4},
          "items": {"Roland": ["Gun"], "Joe": ["Tome"]},
          "gates": [],
          "doom": 2
        },
        
        "decision_process": {
          "candidates": [
            "move Roland to Graveyard",
            "move Roland to Innsmouth", 
            "move Roland to Miskatonic"
          ],
          "chosen": "move Roland to Graveyard",
          "reasoning": "Joe covers Library, I cover Graveyard for defense"
        },
        
        "reachy_action": {
          "action_type": "move",
          "investigator": "Roland",
          "destination": "Graveyard",
          "intention": "Position defensively while Joe investigates"
        },
        
        "state_after_reachy_action": {
          "investigator_positions": {"Roland": "Graveyard", "Joe": "Library"},
          "doom": 2,
          "immediate_event": "no gate opened"
        }
      },
      
      "human_turn": {
        "player": "Alessandro",
        "state_before": {...same as reachy saw...},
        
        "human_actions": [
          {
            "investigator": "Joe",
            "action_type": "investigate",
            "location": "Library",
            "intention": "Strong lore (4) makes sense for Library research"
          }
        ],
        
        "state_after_human_action": {
          "investigator_positions": {"Roland": "Graveyard", "Joe": "Library"},
          "clues": {"Joe": 2},
          "doom": 2,
          "health_effects": {}
        }
      },
      
      "cooperation_analysis": {
        "reachy_positioning": "defensive",
        "human_positioning": "offensive (investigation)",
        "alignment": "complementary",
        "synergy_score": 8.0,
        "why_synergistic": "Roland covers defense, Joe uses high lore for research"
      },
      
      "events": [
        "No gates opened",
        "Joe successfully researched in Library"
      ],
      
      "round_outcome": {
        "clues_found": 2,
        "sanity_lost": 0,
        "health_lost": 0,
        "doom_increase": 0,
        "net_progress": "high (no losses, clues gained)"
      }
    },
    
    {
      "round_number": 2,
      "doom_at_start": 2,
      
      "reachy_turn": {
        "state_before": {...},
        
        "world_model_simulation": {
          "description": "Before deciding, Reachy uses world model to predict",
          "simulations": [
            {
              "candidate_action": "fight gate in Miskatonic",
              "predicted_outcome": {
                "victory": true,
                "sanity_loss": 2,
                "allies_needed": "Joe should defend Roland",
                "total_doom_after": 2
              }
            },
            {
              "candidate_action": "investigate further",
              "predicted_outcome": {
                "clues": 1,
                "sanity_loss": 1,
                "total_doom_after": 2
              }
            }
          ],
          "chosen_based_on": "Fighting + Joe's defense = lowest overall risk"
        },
        
        "reachy_action": {
          "action_type": "fight",
          "location": "Miskatonic",
          "investigator": "Roland",
          "coordination_request": "Joe, can you defend me here? I'll fight the gate."
        },
        
        "state_after_reachy_action": {
          "combat_initiated": true,
          "gate_threatened": "Miskatonic"
        }
      },
      
      "human_turn": {
        "player": "Alessandro",
        "decision": "Respond to Reachy's request",
        
        "human_actions": [
          {
            "investigator": "Joe",
            "action_type": "support",
            "target": "Roland",
            "mechanism": "Use Tome to provide magical support",
            "intention": "Reachy asked for backup, my tome helps in mystical combat"
          }
        ],
        
        "state_after_human_action": {
          "combat_resolution": {
            "Roland": "wins",
            "sanity_loss": 1,  # Lower because Joe helped!
            "gate_closed": true
          }
        }
      },
      
      "cooperation_analysis": {
        "proactivity": "Reachy communicated need BEFORE acting",
        "human_response": "Immediate + optimal (Tome is perfect)",
        "synergy_score": 9.5,
        "outcome": "Combat succeeded with minimal losses"
      }
    },
    
    {...more rounds...}
  ],
  
  "game_summary": {
    "total_synergy_score": 8.7,
    "emergent_strategy": "Reachy learns to ask for help preemptively",
    "human_learning": "Alessandro learns when Reachy needs support",
    "key_insight": "Proactive communication > reactive support"
  }
}
```

---

## 5. What Reachy Learns from Full Context

### Pattern Recognition

```
After 5-10 games with full context, Reachy discovers:

PATTERN 1: Location Synergies
  "When I'm in Arkham (combat-heavy) and Alessandro is in Library 
   (research-heavy), we cover each other's weakness"

PATTERN 2: Timing Synergies
  "If I fight on turn 1 and Alessandro investigates on turn 2,
   doors are open for clues (he finds more)"

PATTERN 3: Resource Synergies
  "Joe's Tome + my combat = better outcomes
   Daisy's dog + Investigation = better clues"

PATTERN 4: Communication Synergies
  "When I ask 'Can you defend?', Alessandro responds 80% of time
   When I don't ask, he doesn't position defensively"

PATTERN 5: Failure Modes
  "When both of us investigate same area, we waste actions
   When neither of us fights gates, doom increases"
```

### Emergent Strategies

```
Game 1-3: Reachy plays independently
Game 4-6: Reachy notices human positioning patterns
Game 7-8: Reachy starts requesting specific support
Game 9+: Reachy makes strategic suggestions

By Game 10:
  Reachy: "Alessandro, I notice you always put Joe in research.
           What if we rotate? Daisy (lore 3) could research turn 1,
           Joe (lore 4) turn 2. Gives us flexibility."
           
  Alessandro: "Interessante! Vamos tentar."
  
  Result: Better gate coverage, more clues found
  
  Learning: "Reachy discovered a strategy that human hadn't thought of"
```

---

## 6. Online Learning: During Game

```python
class OnlineCooperativeLearner:
    """Learn about Alessandro during gameplay"""
    
    def after_human_action(self, 
                          alessandros_action,
                          outcome,
                          how_it_affected_me):
        """
        Immediate feedback loop:
        "Alessandro did X → Y happened → affects my next move"
        """
        
        # Store pattern
        self.pattern_memory.append({
            "human_action": alessandros_action,
            "outcome": outcome,
            "impact_on_me": how_it_affected_me
        })
        
        # Adjust next decision
        if how_it_affected_me == "positive":
            # Encourage similar patterns
            self.suggest_collaborative_action()
        elif how_it_affected_me == "negative":
            # Suggest adjustment
            self.proactively_communicate()
```

Example:

```
Turn 1: Reachy moves to Arkham
Turn 2: Alessandro moves Joe to Library (research)
Turn 3: Reachy learns "Oh, he's going for research synergy"
Turn 4: Reachy proactively: "Mantenho defesa em Arkham?"

Online learned: "Alessandro's strategy + my suggestion = cooperation"
```

---

## 7. Offline Learning: After Game

```python
class OfflineCooperativeTrainer:
    """Fine-tune world model with full trajectory"""
    
    def train_on_game(self, full_trajectory):
        """
        Input: Entire game transcript with all actions
        Learn: Multi-agent dynamics
        """
        
        # 1. Extract action sequences
        # (reachy_action, human_actions) → (state_t, state_t+1)
        
        # 2. Learn cooperation patterns
        # "When we do (X, Y) together, outcome is Z"
        
        # 3. Learn Alessandro's strategy
        # "Alessandro prefers research early, combat late"
        
        # 4. Learn emergent synergies
        # "Combination of our items/stats/positioning = this outcome"
        
        # 5. Update world model
        # Next game, Reachy can predict human actions
```

---

## 8. The Research Gold: Emergent AI-Human Coordination

This is **novel research**:

```
RESEARCH QUESTION:
"How do AI agents learn to cooperate with humans in complex games
 through observation of actions and outcomes rather than 
 explicit programming?"

KEY FINDINGS (expected):

1. Communication Learning
   Reachy learns WHEN to communicate
   (Not just HOW to communicate, but WHEN it's needed)

2. Adaptive Strategy
   Reachy's strategy adapts to Alessandro's playstyle
   (Different humans → different emergent strategies)

3. Efficiency
   How many games until cooperation emerges?
   (Probably 5-10 games)

4. Generalization
   Does cooperation learned with Alessandro transfer to new human?
   (Multi-human cooperative learning)
```

---

## 9. Implementation: Full Context Data Collection

```python
class FullContextGameTracker:
    """Track EVERYTHING for learning"""
    
    def record_turn(self, player_type, action_data):
        """Record both Reachy and human turns"""
        
        if player_type == "reachy":
            self.record_reachy_turn(
                state_before=self.game_state.copy(),
                decision_process=reachy_decision,
                action=reachy_action,
                state_after=self.game_state.copy()
            )
        
        elif player_type == "human":
            self.record_human_turn(
                player=player_name,
                action=human_action,
                state_before=self.game_state.copy(),
                state_after=self.game_state.copy()
            )
    
    def analyze_synergy(self):
        """After each round, score cooperation"""
        
        reachy_move = self.turns[-2]  # Reachy's action
        human_move = self.turns[-1]   # Human's response
        
        synergy = calculate_synergy(reachy_move, human_move)
        # Returns: 0-10 score based on complementarity
        
        self.trajectory.append({
            "reachy_action": reachy_move,
            "human_action": human_move,
            "synergy_score": synergy
        })
```

---

## 10. What Makes This Different from Single-Agent Learning

| Aspect | Single-Agent | Full Context |
|--------|-------------|--------------|
| **Data** | Only Reachy's actions | Reachy's + humans' actions |
| **Patterns** | "Move X → result Y" | "Move X + human Y → result Z" |
| **Strategy** | Isolated optimization | Cooperative optimization |
| **Communication** | N/A | Learned when to ask for help |
| **Adaptation** | Learns game rules | Learns human preferences |
| **Emergence** | Tactical improvements | Collaborative strategies |

---

## 11. Next Steps to Implement

1. **Expand JSON format** to include human actions alongside Reachy's
2. **Synergy calculator** (how well coordinated were moves?)
3. **Offline trainer** that learns multi-agent dynamics
4. **Online updater** that adjusts strategy mid-game
5. **Communication module** that suggests when to ask for help

---

## 12. The Payoff

By game 10-15:

```
Reachy not just PLAYS the game.
Reachy UNDERSTANDS the human.

"Alessandro, você sempre favorece investigação.
 Mas nesse round, o Doom está aumentando.
 Se você lutar comigo agora, evitamos 2 pontos.
 Vou ficar defensivo se você me apoiar."

Result: Coordinated gameplay that neither could achieve alone.

This is human-AI cooperation emerging from learning.
This is publishable research.
```

---

Ready to implement full context learning?
