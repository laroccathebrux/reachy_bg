"""Names and payload contracts of the project's Qdrant collections.

Every collection uses bge-m3 embeddings (1024 dimensions, cosine distance) so that
one embedding call can query any of them.

Payload contracts (all keys and values in English):

``bg_rules`` - one point per text chunk of an official rules document
    game_id        "eldritch-horror"
    source         "rulebook" | "reference_guide"
    section        top-level heading, e.g. "Phase 1: Action Phase"
    subsection     second-level heading or "" when the chunk sits directly under the section
    path           "section > subsection" (search-friendly breadcrumb)
    page_start     first page of the chunk (1-based)
    page_end       last page of the chunk
    chunk_index    position of the chunk inside its section (0-based)
    text           the chunk itself (roughly 400-900 characters)

``bg_knowledge`` - one point per named entity or FAQ entry known before a game starts
    game_id        "eldritch-horror"
    kind           "investigator" | "ancient_one" | "monster" | "gates" | "faq"
                   | "card_composition" | "flowchart" | "starting_possessions" | "strategy"
    name           entity name or FAQ question
    expansion      product the entity comes from, e.g. "Eldritch Horror (base game)"
    base_game      True when it ships in the 2013 base box
    text           English description used for the embedding
    ...            kind-specific structured fields (see ``migrate_knowledge.py``)

    A ``strategy`` point is player advice rather than a game fact, so it carries where the
    advice came from and never reads as a rule. See ``ingest_strategy.py``.
    topic          "investigator_choice" | "ancient_one_matchup" | "monsters" | "assets" | "habits"
    players        number of investigators the advice is about, or 0 when it is general
    ancient_one    the Ancient One it is about, or "" when it is general
    source_kind    "community_guide"
    source_url     where it came from
    source_author  who wrote it
    confidence     "stated" when the source asserts it plainly, "partial" when this project
                   could only read part of the passage

``bg_sessions`` - one point per round summary of a played game
    game_id        "eldritch-horror"
    session_id     "YYYYMMDD-HHMM" of the game start
    round          round number being closed
    ancient_one    name or null
    investigators  list of investigator names in play
    summary        2-4 sentence English summary of the round
    text           "Session <id> vs <ancient one>, round <n>: <summary>" (embedded)
    recorded_at    unix timestamp
"""

from typing import Final

GAME_ID: Final = "eldritch-horror"

RULES: Final = "bg_rules"
KNOWLEDGE: Final = "bg_knowledge"
SESSIONS: Final = "bg_sessions"

VECTOR_DIM: Final = 1024  # bge-m3

RULE_SOURCES: Final = ("rulebook", "reference_guide")

KNOWLEDGE_KINDS: Final = (
    "investigator",
    "ancient_one",
    "monster",
    "gates",
    "faq",
    "card_composition",
    "flowchart",
    "starting_possessions",
    "strategy",
)

# Payload fields that get a keyword index so filters stay fast.
KEYWORD_INDEXES: Final[dict[str, tuple[str, ...]]] = {
    RULES: ("game_id", "source", "section"),
    KNOWLEDGE: ("game_id", "kind", "name", "expansion"),
    SESSIONS: ("game_id", "session_id", "ancient_one"),
}

BOOL_INDEXES: Final[dict[str, tuple[str, ...]]] = {
    KNOWLEDGE: ("base_game",),
}

INTEGER_INDEXES: Final[dict[str, tuple[str, ...]]] = {
    SESSIONS: ("round",),
}
