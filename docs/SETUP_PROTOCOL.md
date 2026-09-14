# Game Setup Protocol: Verbal + RAG Hybrid

## The Elegant Solution

You told me: **Verbal setup + RAG enrichment + Visual input**

This is **pragmatic and elegant**. No OCR needed, no complex vision, just conversational.

---

## Setup Flow (What Actually Happens)

```
┌────────────────────────────────────────────────────────────┐
│ PHASE 0: SETUP (5-10 minutes total)                        │
├────────────────────────────────────────────────────────────┤
│                                                             │
│ YOU: "Vamos jogar com Roland Banks e Joe Diamond.          │
│       Roland é seu. Eu jogo com Joe."                      │
│                                                             │
│ └─→ REACHY HEARS (STT)                                      │
│     "investigadores: Roland Banks (para mim), Joe Diamond" │
│                                                             │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ REACHY ACTION: Query Qdrant                          │   │
│ │                                                      │   │
│ │ Query: "investigador: Roland Banks"                 │   │
│ │ Returns:                                             │   │
│ │   {                                                  │   │
│ │     "name": "Roland Banks",                          │   │
│ │     "strength": 4,                                   │   │
│ │     "lore": 2,                                       │   │
│ │     "craft": 1,                                      │   │
│ │     "common_sense": 3,                               │   │
│ │     "will": 2,                                       │   │
│ │     "starting_location": "Arkham",                   │   │
│ │     "starting_items": ["Revolver"],                  │   │
│ │     "unique_ability": "Combat Specialist",           │   │
│ │     "description": "..."                             │   │
│ │   }                                                  │   │
│ │                                                      │   │
│ │ Query: "investigador: Joe Diamond"                  │   │
│ │ Returns: {...similar structure...}                  │   │
│ │                                                      │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                             │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ REACHY ACTION: Analyze team composition              │   │
│ │                                                      │   │
│ │ LLM reasoning:                                       │   │
│ │ "Roland: strong in combat (strength 4)              │   │
│ │  Joe: strong in lore (lore 4)                       │   │
│ │  Together: combat + investigation synergy"          │   │
│ │                                                      │   │
│ │ Generated context:                                   │   │
│ │ "Gosto dessa combinação. Vou focar em combate       │   │
│ │  enquanto você investiga com Joe."                  │   │
│ │                                                      │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                             │
│ YOU: [Shows image of the Abomination on phone/tablet]      │
│                                                             │
│ └─→ REACHY SEES (Camera or image upload)                    │
│                                                             │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ REACHY ACTION: Analyze Creature                      │   │
│ │                                                      │   │
│ │ Vision model (Claude) + LLM reasoning:               │   │
│ │ "Image shows: Cthulhu                               │   │
│ │  Characteristics: Tentacles, enormous size, flight" │   │
│ │                                                      │   │
│ │ Query Qdrant: "Cthulhu - rules, weaknesses"         │   │
│ │ Returns: {...fighting rules, combat modifiers...}   │   │
│ │                                                      │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                             │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ REACHY ACTION: Query 2-Player Rules                  │   │
│ │                                                      │   │
│ │ Query: "2 players - changes to game flow"           │   │
│ │ Returns:                                             │   │
│ │ {                                                    │   │
│ │   "doom_track_modifier": -1,  # Easier with 2 ppl  │   │
│ │   "elder_signs_needed": 13,   # Instead of 15      │   │
│ │   "player_count_effect": "Faster escalation",       │   │
│ │   "gate_handling": "2 investigators can cover more" │   │
│ │ }                                                    │   │
│ │                                                      │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                             │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ REACHY OUTPUT: Setup Summary                         │   │
│ │                                                      │   │
│ │ "Pronto! Vamos jogar contra Cthulhu.                │   │
│ │                                                      │   │
│ │  Você: Joe Diamond (lore forte, investigação)       │   │
│ │  Eu: Roland Banks (combate forte)                   │   │
│ │                                                      │   │
│ │  Cthulhu é poderoso mas com dois investigadores     │   │
│ │  focados, temos chance. Você investiga enquanto    │   │
│ │  eu defendo.                                        │   │
│ │                                                      │   │
│ │  Começamos em Arkham e Miskatonic?                  │   │
│ │  Você está pronto?"                                 │   │
│ │                                                      │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## What This Solves

### ✅ Advantages

1. **Conversational** — Feels natural
2. **Fast** — No OCR, no complex vision, no physical card placement
3. **Reliable** — Qdrant queries are exact (no recognition errors)
4. **Extensible** — Works with ANY investigators (just name them)
5. **Emergent** — LLM can reason about synergies
6. **Research-clean** — Focuses on strategy/conversation, not engineering

### ❌ What We're NOT Doing

- No OCR of investigator cards
- No special card-reading zone on table
- No image analysis of character stats
- No visual recognition of abilities

---

## Implementation: Setup Module

```python
class GameSetupManager:
    """Handle game initialization through conversational RAG"""
    
    def __init__(self, qdrant_client, llm, tts):
        self.qdrant = qdrant_client
        self.llm = llm
        self.tts = tts
    
    def parse_investigator_setup(self, utterance: str):
        """
        User says: "Vamos jogar com Roland Banks e Joe Diamond"
        
        Extract investigator names via NER or LLM
        Returns: ["Roland Banks", "Joe Diamond"]
        """
        
        # Could use regex or LLM NER
        investigators = self.llm.extract_names(
            text=utterance,
            context="game investigators from Eldritch Horror"
        )
        
        return investigators
    
    def fetch_investigator_profile(self, name: str) -> dict:
        """Query Qdrant for investigator details"""
        
        results = self.qdrant.search(
            collection_name="eldritch_horror_investigators",
            query_text=name,
            limit=1,
            score_threshold=0.8
        )
        
        if not results:
            return {"error": f"Investigador {name} não encontrado"}
        
        return results[0].payload
    
    def analyze_team_composition(self, 
                                 reachy_inv: dict, 
                                 human_invs: list[dict]) -> str:
        """
        LLM analyzes team synergy
        
        Input: Reachy's investigator + human's investigators
        Output: Natural language analysis of strengths/weaknesses
        """
        
        prompt = f"""
        Você está analisando um time de investigadores para Eldritch Horror.
        
        Investigador do Reachy: {reachy_inv['name']}
          - Força: {reachy_inv['strength']}
          - Lore: {reachy_inv['lore']}
          - Habilidade: {reachy_inv['unique_ability']}
        
        Investigadores do jogador humano:
        {json.dumps(human_invs, indent=2, ensure_ascii=False)}
        
        Analise:
        1. Qual é a força do time combinado?
        2. Qual é o ponto fraco?
        3. Como você sugere que joguem juntos?
        
        Responda em 2-3 frases, como se fosse uma conversa.
        """
        
        analysis = self.llm.generate(prompt=prompt, max_tokens=150)
        
        return analysis
    
    def process_abomination_image(self, image) -> dict:
        """
        User shows image of creature
        
        Reachy analyzes:
        1. What creature is it? (via vision)
        2. Get rules from Qdrant
        3. Analyze implications for strategy
        """
        
        # Vision: identify creature
        vision_context = self.llm.analyze_image(image)
        # Returns: "Image shows Cthulhu with tentacles, flying, enormous"
        
        # Query Qdrant for rules
        creature_name = vision_context['creature_name']
        
        creature_rules = self.qdrant.search(
            collection_name="eldritch_horror_creatures",
            query_text=creature_name,
            limit=1
        )
        
        return {
            "identified_as": creature_name,
            "visual_description": vision_context,
            "rules": creature_rules[0].payload if creature_rules else {},
        }
    
    def query_player_count_rules(self, player_count: int) -> dict:
        """Get rules for 2-player (or N-player) variant"""
        
        results = self.qdrant.search(
            collection_name="eldritch_horror_rules",
            query_text=f"{player_count} player variant",
            limit=1
        )
        
        return results[0].payload if results else {}
    
    def generate_setup_summary(self,
                               reachy_inv: dict,
                               human_invs: list[dict],
                               creature: dict,
                               rules: dict) -> str:
        """
        Generate full setup summary for TTS
        
        Reachy will speak this, establishing game context
        """
        
        prompt = f"""
        Você é um estrategista experiente em Eldritch Horror.
        O jogo vai começar. Resuma o setup em 4-5 frases naturais.
        
        SEU investigador: {reachy_inv['name']} ({reachy_inv['unique_ability']})
        Investigadores do oponente: {', '.join([i['name'] for i in human_invs])}
        Inimigo: {creature['identified_as']}
        
        Você está pronto? Qual é sua estratégia inicial?
        
        IMPORTANTE: Responda como se estivesse falando (conversacional).
        """
        
        summary = self.llm.generate(prompt=prompt, max_tokens=200)
        
        return summary

# Usage
setup_manager = GameSetupManager(qdrant, llm, tts)

# 1. User speaks
user_input = "Vamos jogar com Roland Banks e Joe Diamond. Roland é seu."

# 2. Parse
invs = setup_manager.parse_investigator_setup(user_input)
# ["Roland Banks", "Joe Diamond"]

# 3. Fetch profiles
reachy_profile = setup_manager.fetch_investigator_profile("Roland Banks")
human_profiles = [
    setup_manager.fetch_investigator_profile("Joe Diamond")
]

# 4. Analyze team
team_analysis = setup_manager.analyze_team_composition(
    reachy_profile,
    human_profiles
)

# 5. Get creature (user shows image)
creature_data = setup_manager.process_abomination_image(creature_image)

# 6. Get 2-player rules
rules_2p = setup_manager.query_player_count_rules(2)

# 7. Generate summary
summary = setup_manager.generate_setup_summary(
    reachy_profile,
    human_profiles,
    creature_data,
    rules_2p
)

# 8. Speak it
tts.speak(summary, voice="reachy")
```

---

## Data Structure: What's in Qdrant

### Collection: `eldritch_horror_investigators`

```json
{
  "id": "roland_banks",
  "name": "Roland Banks",
  "text": "Roland Banks - detective, strength 4, common_sense 3, revolver...",
  "metadata": {
    "strength": 4,
    "lore": 2,
    "craft": 1,
    "common_sense": 3,
    "will": 2,
    "starting_location": "Arkham",
    "starting_items": ["Revolver"],
    "unique_ability": "Combat specialist - +2 to combat checks",
    "health": 3,
    "sanity": 5,
    "description": "Police detective from Boston..."
  }
}
```

### Collection: `eldritch_horror_creatures`

```json
{
  "id": "cthulhu",
  "name": "Cthulhu",
  "text": "Cthulhu - ancient entity, flying, tentacles, high difficulty...",
  "metadata": {
    "difficulty": "very hard",
    "combat_strength": 6,
    "special_powers": ["Flight", "Tentacle attacks", "Madness aura"],
    "weakness": "Requires elder signs to seal",
    "elder_signs_needed": 15,
    "starting_location": "Deep Ones Rising",
    "rules": "..."
  }
}
```

### Collection: `eldritch_horror_rules`

```json
{
  "id": "rules_2_player",
  "text": "Two player variant - doom track modified, easier sealing...",
  "metadata": {
    "player_count": 2,
    "doom_track_max": 13,
    "elder_signs_needed": 13,
    "gate_handling": "2 investigators cover different areas",
    "sanity_effects": "Scaled for 2 players"
  }
}
```

---

## Conversation Flow During Setup

```
TURN 1:
You: "Vamos jogar com Roland Banks e Joe Diamond. 
      Roland é seu. Eu jogo com Joe."

Reachy (internal): 
  ├─ Identify investigadores: ["Roland", "Joe"]
  ├─ Fetch from Qdrant: profiles
  ├─ Analyze: "Roland combate, Joe lore → bom balanço"
  └─ Confirm understanding

Reachy (TTS): "Entendi! Você com Joe Diamond (força em lore), 
              eu com Roland Banks (força em combate). 
              Boa combinação. Próximo: quem é o inimigo?"

TURN 2:
You: [Shows image of Cthulhu on phone]

Reachy (internal):
  ├─ Vision: "This is Cthulhu"
  ├─ Fetch creature rules from Qdrant
  ├─ Query 2-player rules
  └─ Synthesize strategy implications

Reachy (TTS): "Cthulhu! Muito poderoso, vai ser difícil.
              Mas com dois investigadores, temos chance se nos 
              coordenarmos bem. Você cobre investigação com Joe,
              eu vou focar em selagem?"

TURN 3:
You: "Sim, vamos começar!"

Reachy (internal):
  ├─ Initialize game state
  ├─ Set starting positions based on investigator profiles
  ├─ Begin board observation

Reachy (TTS): "Pronto! Começamos em Arkham. 
              Você quer jogar ou você decide a estratégia?"

[Game begins]
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────┐
│ USER SPEECH                         │
│ "Roland e Joe. Roland é seu."       │
└────────────┬────────────────────────┘
             │
             ↓
    ┌─────────────────────────────────┐
    │ NER/LLM Parser                  │
    │ Extract: ["Roland", "Joe"]      │
    └────────────┬────────────────────┘
                 │
      ┌──────────┴──────────┐
      ↓                     ↓
┌─────────────┐     ┌──────────────┐
│ Qdrant      │     │ Qdrant       │
│ "Roland..." │     │ "Joe..."     │
└──────┬──────┘     └──────┬───────┘
       │                   │
       └───────┬───────────┘
               ↓
        ┌─────────────────────┐
        │ LLM Analysis        │
        │ (Synergy, strategy) │
        └────────────┬────────┘
                     │
                     ↓
            ┌───────────────────┐
            │ ElevenLabs TTS    │
            │ (Natural speech)  │
            └───────────────────┘

[USER SHOWS CREATURE IMAGE]
            ↓
    ┌─────────────────────┐
    │ Vision Model        │
    │ Identify creature   │
    └────────────┬────────┘
                 ↓
        ┌─────────────────────┐
        │ Qdrant Query        │
        │ "Cthulhu rules..."  │
        └────────────┬────────┘
                     ↓
        ┌─────────────────────┐
        │ LLM: Rules analysis │
        │ + Strategy adapt    │
        └────────────┬────────┘
                     ↓
            ┌───────────────────┐
            │ TTS Summary       │
            └───────────────────┘
```

---

## What This Enables

1. **Fast setup** (5-10 min)
2. **Natural conversation** (no special commands)
3. **Emergent strategy** (LLM reasons about team)
4. **Reliable** (Qdrant queries, no vision errors)
5. **Scalable** (works with any investigators/creatures)
6. **Clean research** (focuses on AI, not engineering)

---

## Next Step

This becomes **Phase 0: Setup Module** in the roadmap.

Then we move to **Phase 1: Vision** (board observation during game).

Sound good?
