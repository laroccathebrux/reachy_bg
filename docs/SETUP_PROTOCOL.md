# Game Setup Protocol: Verbal Briefing + Knowledge Base

How a game session starts. The goal is a setup that takes a couple of minutes, needs no card
recognition, and leaves the robot with a complete, structured picture of the table before
the first Action Phase.

## Design choice

Setup is **verbal**: the human names the investigators and the Ancient One, the robot fetches
the sheets from `bg_knowledge` and confirms. No optical reading of Investigator sheets, no
special reader zone on the table. Visual confirmation of the board (gates, clues, tokens)
comes later from the vision pipeline during play, not during setup.

Why: the knowledge base already holds every base-game sheet with exact values; a spoken
name resolves to it with a single lookup, which is faster and more reliable than any camera
read. The research value is in conversation and strategy, not in card OCR.

## Flow

```
1. Greeting and player enrolment
   Human:  "Hi Reachy, we're playing tonight. I'm Alessandro."
   Robot:  (stores a voiceprint from that sentence; see SPEECH_PIPELINE.md)
           "Good to see you. Who else is at the table?"
   -> one sentence per additional player, each enrolled as a speaker.

2. Investigators
   Human:  "You control Lily Chen. I take Jacqueline Fine."
   Robot:  lookup bg_knowledge kind=investigator for both names
           -> Investigator sheets (health, sanity, skills, abilities, start space, possessions)
           "Lily Chen, the Martial Artist: Strength 4, Health 6, Sanity 6, starts in Shanghai
            with a Protective Amulet and a Lucky Rabbit's Foot. Jacqueline Fine, the Psychic:
            Lore 4, starts on Space 5 with Flesh Ward and a Clue. Correct?"

3. Ancient One
   Human:  "We're facing Cthulhu."
   Robot:  lookup bg_knowledge kind=ancient_one
           -> starting Doom 12, Mythos deck 0/2/2 - 1/3/0 - 3/4/0, set-aside monsters
           "Cthulhu: Doom starts at 12. Set aside one Deep One, one Star Spawn and the Cthulhu
            Special Encounters. Mythos deck: stage one 0 green, 2 yellow, 2 blue; ..."

4. Player count and Reference card
   Robot:  2 players -> 1 Gate, 1 Clue, 1 Monster per surge.
           "Two of us, so the two-player reference card: one gate, one clue."

5. Starting effects
   Robot:  walks the rulebook setup steps 8 and 9 from bg_rules if asked
           ("Build the Mythos deck", "Resolve starting effects": spawn gates and clues,
            place the Expedition token, draw the first Mystery).
   Human:  reads the drawn Mystery aloud; robot stores it as the active Mystery.

6. Team analysis (LLM)
   Robot:  "Lily is our fighter, Jacqueline our researcher. I'll hunt gates and monsters;
            you chase clues. Shall we?"

7. Game state initialised
   -> data/game_logs/<session_id>/state.json (see DESIGN_DOCUMENT.md, "Game state")
   -> bg_sessions receives its first entry at the end of round 1.
```

Every robot line above is a paraphrase generated at runtime in `SPOKEN_LANGUAGE`; the
examples are English for documentation only.

## Lookups used during setup

| Need | Collection | Filter |
|---|---|---|
| Investigator sheet by name | `bg_knowledge` | `kind = investigator`, `base_game = true`, name match |
| Ancient One sheet | `bg_knowledge` | `kind = ancient_one` |
| Gate colors, monster list | `bg_knowledge` | `kind = gates` / `kind = monster`, `base_game = true` |
| Setup steps, deck building, player-count table | `bg_rules` | semantic search, `source = rulebook` then `reference_guide` |
| Rules questions during setup | `bg_rules` | semantic search |

## What the robot must have before round 1

- Its own investigator: name, current and maximum Health and Sanity, five skills, abilities,
  space, possessions, tickets, Clues, Conditions.
- The same for every human investigator (controller = the enrolled player name).
- Ancient One, Doom value, Omen position, active Mystery text, Reserve contents.
- Board tokens: Gates with colors and spaces, Monsters with spaces, Clues, Active Expedition
  space, Rumors.
- Player order and who holds the Lead Investigator token.

Anything the human does not state is asked, one item at a time, before the first action.

## Failure modes and answers

| Situation | Behaviour |
|---|---|
| Name not found in `bg_knowledge` | Ask again; offer the closest matches by name. Expansion content is refused politely ("only the base game is set up"). |
| Ambiguous player count | Ask. The Reference card depends on it. |
| Human changes an investigator mid-setup | Replace the sheet, re-run the team analysis. |
| Knowledge base offline | Say so, continue with what the human dictates, mark the state as unverified. |
