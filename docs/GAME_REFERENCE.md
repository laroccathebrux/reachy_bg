# Eldritch Horror (2013) Base Game Reference

Ground truth for everything in this repository that touches the game. Only the 2013 base box
is in scope: no expansions, no Arkham Horror content. When code, prompts or the knowledge base
disagree with this page, this page wins until it is corrected against the rulebook.

Sources: the official FFG Rulebook and Reference Guide (embedded in the `bg_rules` collection),
the Eldritch Horror wiki and the Order of Gamers summary v5.6. Values marked *(unverified)*
could not be confirmed against the official PDFs.

## Investigators (12)

Skills: L = Lore, I = Influence, O = Observation, St = Strength, W = Will. Health + Sanity = 12
for every base investigator.

| Name | Occupation | Health | Sanity | L | I | O | St | W | Starting space | Starting possessions |
|---|---|---|---|---|---|---|---|---|---|---|
| Akachi Onyele | The Shaman | 5 | 7 | 3 | 2 | 2 | 2 | 4 | Space 15 (South Africa) | Mists of Releh (Spell), 1 Clue |
| Charlie Kane | The Politician | 4 | 8 | 2 | 4 | 3 | 2 | 2 | San Francisco | Personal Assistant (Asset) |
| Diana Stanley | The Redeemed Cultist | 7 | 5 | 4 | 2 | 3 | 3 | 1 | Space 7 (Central America) | Arcane Manuscripts (Asset), Wither (Spell) |
| Jacqueline Fine | The Psychic | 4 | 8 | 4 | 2 | 3 | 1 | 3 | Space 5 (American Heartland) | Flesh Ward (Spell), 1 Clue |
| Jim Culver | The Musician | 7 | 5 | 3 | 3 | 2 | 2 | 3 | Space 6 (Deep South) | Shriveling (Spell), 1 Clue |
| Leo Anderson | The Expedition Leader | 6 | 6 | 2 | 2 | 3 | 3 | 3 | Buenos Aires | Hired Muscle (Asset) |
| Lily Chen | The Martial Artist | 6 | 6 | 2 | 2 | 2 | 4 | 3 | Shanghai | Protective Amulet (Asset), Lucky Rabbit's Foot (Asset) |
| Lola Hayes | The Actress | 5 | 7 | 2 | 4 | 2 | 2 | 3 | Tokyo | .18 Derringer (Asset), improve 1 skill of choice |
| Mark Harrigan | The Soldier | 8 | 4 | 1 | 2 | 2 | 4 | 4 | Space 14 (Northern Europe) | .38 Revolver (Asset), Kerosene (Asset) |
| Norman Withers | The Astronomer | 5 | 7 | 3 | 1 | 3 | 2 | 4 | Arkham | Feed the Mind (Spell) |
| Silas Marsh | The Sailor | 8 | 4 | 1 | 3 | 3 | 3 | 3 | Sydney | Fishing Net (Asset) |
| Trish Scarborough | The Spy | 7 | 5 | 1 | 3 | 4 | 3 | 2 | Space 16 (Central Russia) | .45 Automatic (Asset) |

Each sheet has one **action ability** (used as a Component Action, once per round) and one
**passive ability**:

| Name | Action ability | Passive ability |
|---|---|---|
| Akachi | Look at the top 2 Gates of the Gate stack; put one on top and one on the bottom. | After closing a Gate during an Other World Encounter, may move to any space with a Clue or a Gate. |
| Charlie | Another investigator may immediately perform 1 additional action. | When performing Acquire Assets, may give the purchased cards to other investigators. |
| Diana | If a Cultist is on her space, discard all Monsters there or move the Cultist to any space. | Horror of Monsters she encounters is reduced to 1. |
| Jacqueline | Trade any number of Clues with an investigator on any space. | Once per round, when another investigator gains a non-Common Condition, look at its back and gain 1 Clue. |
| Jim | Each investigator on his space recovers 1 Sanity. | Investigators on his space roll 1 extra die on Combat Encounter tests. |
| Leo | Test Influence; on a pass gain an Ally Asset from the reserve or discard pile. | On a Wilderness space, investigators there roll 1 extra die on tests. |
| Lily | Spend any Health or Sanity, then recover the same amount of the other. | When she improves a skill she may improve it again immediately. |
| Lola | Spend Improvement tokens to improve any skills, one per token (a +2 token counts as 2). | Once per round an investigator on her space rolls 1 extra die on a test. |
| Mark | He and 1 Monster on his space each lose 1 Health. | Cannot become Delayed or Detained unless he chooses to. |
| Norman | Spend 2 Clues to discard 1 Monster on a space with a Gate. | Once per round may spend 1 Sanity instead of 1 Clue. |
| Silas | Move 1 space along a Ship path, then perform 1 additional action (does not count toward the 2-action limit; no tickets on that move). | On a Sea space, investigators there roll 1 extra die on tests. |
| Trish | If she has no Clues, gain 1 Clue. | When an investigator on her space spends a Clue to reroll, he may reroll 2 dice. |

## Ancient Ones (4)

All four require **3 solved Mysteries** to win before the Ancient One awakens. Mythos deck
composition is given as green / yellow / blue cards per stage.

| Ancient One | Title | Starting Doom | Stage I | Stage II | Stage III | Special setup |
|---|---|---|---|---|---|---|
| Azathoth | The Daemon Sultan | 15 | 1/2/1 | 2/3/1 | 2/4/0 | 1 Eldritch token on the green Omen space |
| Cthulhu | The Madness from the Sea | 12 | 0/2/2 | 1/3/0 | 3/4/0 | Set aside 1 Deep One, 1 Star Spawn, Cthulhu Special Encounters |
| Shub-Niggurath | The Black Goat of the Woods | 13 | 1/2/1 | 3/2/1 | 2/4/0 | Set aside 2 Ghouls, 2 Goat Spawn, 1 Dark Young |
| Yog-Sothoth | The Lurker at the Threshold | 14 | 0/2/1 | 2/3/1 | 3/4/0 | Set aside Yog-Sothoth Special Encounters |

Awakening: Azathoth ends the game immediately (loss). Cthulhu and Shub-Niggurath spawn as Epic
Monsters (Space 3 and The Heart of Africa) that must be defeated as the Final Mystery.
Yog-Sothoth's Final Mystery is resolved through "The Key and the Gate" Special Encounter on a
Gate space. Details are stored per Ancient One in `bg_knowledge`.

## Round structure

Every round: **Action Phase -> Encounter Phase -> Mythos Phase**. The Lead Investigator acts
first, then clockwise; at the end of the Mythos Phase the Lead Investigator token may be passed.

### Action Phase

Each investigator performs **up to 2 actions, each distinct action at most once per round**,
fully resolving one before starting the next. A Delayed investigator stands the token up
instead of acting.

| Action | Rule |
|---|---|
| Travel | Move to an adjacent space. Then spend any number of tickets: each gives 1 extra move; Train tickets only along Train paths, Ship tickets only along Ship paths. |
| Prepare for Travel | City only. Gain 1 Train or Ship ticket matching a path that touches the space. Maximum 2 tickets. |
| Acquire Assets | City without a Monster. Test Influence; gain Reserve cards whose total value is at most the successes. Bank Loan: gain a Debt Condition for +2 successes (not if already in Debt). |
| Rest | No Monster on the space. Recover 1 Health and 1 Sanity. |
| Trade | With an investigator on the same space: exchange possessions (Assets, Artifacts, Spells, Clues, tickets). Never Conditions, Health, Sanity or Improvement tokens. |
| Component Action | An "Action:" printed on the Investigator sheet, a possession or a Condition; each component once per round. "Local Action:" may be used by anyone on that space. |

The Focus action and Focus tokens **do not exist in the base game** (Mountains of Madness).

### Encounter Phase

Each investigator resolves exactly one encounter (Detained replaces it):

1. Monsters on the space: a **Combat Encounter** against each, one at a time. Will test
   (horror minus successes = Sanity lost), then Strength test (damage minus successes =
   Health lost; the Monster loses Health equal to successes). If no Monster remains, one
   additional encounter may follow.
2. Otherwise one of: **Location** (the city's regional deck: America, Europe, Asia/Australia),
   **General** (City / Wilderness / Sea paragraph), **Research** (Clue token on the space;
   Ancient One's Research deck), **Other World** (Gate on the space; may close the Gate),
   **Expedition** (Active Expedition token), **Rumor** (Rumor token; text on the Mythos card),
   **Special** (Ancient One specific), or **Defeated Investigator** (a defeated token on the
   space).

Expedition, Other World and Special encounters are **complex**: initial effect with a test,
then a pass or fail effect. An **Ambush** draws a random Monster, fights it and discards it
afterwards, with no bonus encounter.

### Mythos Phase

The Lead Investigator draws the top Mythos card and resolves its icons left to right, then its
text:

| Icon | Effect |
|---|---|
| Advance Omen | Move the Omen token 1 space clockwise; advance Doom 1 per Gate matching the new Omen. |
| Reckoning | Resolve Reckoning effects in order: Monsters, Ancient One sheet, Ongoing Mythos cards, investigator possessions and Conditions. |
| Spawn Gates | Spawn the Reference-card number of Gates, each with 1 Monster. Empty stack and discard pile: advance Doom 1 instead. |
| Monster Surge | On each Gate matching the current Omen, spawn the Reference-card number of Monsters; no matching Gate: spawn 1 Gate. |
| Spawn Clues | Spawn the Reference-card number of Clues on their printed spaces. |
| Place Rumor token / Place Eldritch tokens | As printed on the card. |
| Text | Event: resolve and discard. Ongoing: stays in play. Rumor: Ongoing with its own solve condition; its Rumor and Eldritch tokens and Epic Monsters are discarded when solved. |

If no Mythos card can be drawn, the investigators lose. The Mythos deck is never replenished.

## Board

- World map with **9 cities** in three regions: America (San Francisco, Arkham, Buenos Aires),
  Europe (London, Rome, Istanbul), Asia/Australia (Shanghai, Sydney, Tokyo). City encounters
  tend to improve a skill (San Francisco: Observation; Rome: Will; Istanbul: Influence;
  Shanghai: Lore; Sydney: Strength) or grant Spells (Arkham: Incantation; Buenos Aires: Ritual).
- **6 Expedition spaces**: The Amazon, The Pyramids, The Heart of Africa, The Himalayas,
  Tunguska (Wilderness) and Antarctica (Sea).
- **21 numbered spaces** (City: 1, 5, 6, 7, 14, 15, 16, 17, 20; Sea: 2, 3, 8, 11, 12, 13, 18;
  Wilderness: 4, 9, 10, 19, 21).
- Path types: Train, Ship, Uncharted. Paths leaving one edge of the board continue on the
  opposite edge.
- **Gates** (9 tokens): San Francisco and Istanbul (green), Buenos Aires, London and Tokyo
  (blue), Arkham, Rome, Shanghai and Sydney (red).
- **Omen track**: 4 spaces, clockwise from the top: green comet, blue stars, red eclipse, blue
  stars. The token starts on green.
- **Doom track**: starts at the Ancient One's value and advances toward 0.
- Other tokens: Monsters and Epic Monsters (opaque cup), Clues (facedown pool, each names a
  space; also a resource: spend 1 Clue to reroll 1 die), Active Expedition, Rumor, Eldritch,
  Mystery, Travel tickets (Train / Ship, max 2), Health, Sanity, Improvement (+1 / +2 per
  skill), Lead Investigator.
- **Reserve**: 4 faceup Asset cards, refilled after an Acquire Assets action fully resolves.
- Card types in the base game: Assets, Artifacts, Spells, Conditions, Mysteries, Research /
  Other World / Expedition / General / Location / Special Encounters, Mythos. Unique Assets
  and Impairment tokens are expansion content.

## Skill tests

1. Dice = skill value +/- test modifier + Improvement tokens + at most one "+X" bonus (the
   highest) + all "additional die" effects. Minimum 1 die.
2. A **5 or 6 is a success**. Any success = pass (Blessed: 4s count; Cursed: only 6s).
3. Rerolls: spend 1 Clue per die, unlimited; other rerolls from Assets and abilities.
4. Improvement tokens: +1, flip to +2, maximum +2 per skill; not tradeable.

## Winning and losing

- **Win**: solve 3 Mysteries; if the Ancient One has awakened, also solve its Final Mystery.
  Simultaneous win and loss counts as a win.
- **Lose**: Doom reaches 0 and the Ancient One's awakened side ends the game (Azathoth
  immediately); all players eliminated; the Mythos deck runs out; a card effect says so.
- A defeated investigator (Health or Sanity at 0) advances Doom by 1, is laid on its side in
  the nearest city with its possessions, and the player picks a new unused investigator at the
  end of the Mythos Phase (before the awakening). After the awakening, defeat means
  elimination.

## Player-count scaling (Reference card)

| Players | Spawn Gates | Spawn Clues | Monster Surge per matching Gate |
|---|---|---|---|
| 1 - 2 | 1 | 1 | 1 |
| 3 - 4 | 1 | 2 | 2 |
| 5 - 6 | 2 | 3 | 2 |
| 7 - 8 | 2 | 4 | 3 |

The robot plus one human is a **2-player game**: 1 Gate, 1 Clue, 1 Monster per surge. Solo
rules recommend controlling 2 investigators with the 2-player card. Difficulty is adjusted by
removing "hard" (easier) or "easy" (harder) Mythos cards before building the deck, or by
starting with a Rumor in play.

## Unverified

- Printed maximum of the Doom track (believed 0 to 15).
- Exact counts of Clue and Monster tokens in the base box.
