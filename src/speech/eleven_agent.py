"""ElevenLabs conversation agent: configuration, creation and the local client tools.

The agent runs the spoken loop (ASR, LLM, TTS, turn-taking, interruptions) in the cloud; the
robot's knowledge stays here as *client tools* the agent calls per turn:

* ``game_rules(question)``      -> passages from ``bg_rules`` + ``bg_knowledge`` (Qdrant)
* ``game_knowledge(name)``      -> a specific investigator, Ancient One, monster or FAQ entry

Bilingual output keeps one native voice per language: the agent speaks Brazilian Portuguese
with the Brazilian voice by default and switches to the American voice for English through
a language preset plus the ``language_detection`` system tool.

    agent_id = ensure_agent()          # create once (id kept in data/eleven_agent.json) or update
    tools = client_tools()             # ClientTools for the SDK conversation
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from src.config import (
    DATA_DIR,
    DEFAULT_LANGUAGE,
    ELEVEN_AGENT_ID,
    ELEVEN_AGENT_LLM,
    ELEVEN_AGENT_NAME,
    ELEVEN_AGENT_VOICES,
    ELEVEN_MAX_DURATION_S,
    ELEVEN_TURN_TIMEOUT_S,
    ELEVENLABS_API_KEY,
    ELEVENLABS_MODEL_ID,
    LANGUAGE_LOCK,
    SPOKEN_LANGUAGES,
)
from src.llm.prompts import format_passages
from src.logger import get_logger
from src.speech.language import language_name

log = get_logger(__name__)

API = "https://api.elevenlabs.io/v1/convai"
AGENT_FILE = DATA_DIR / "eleven_agent.json"
SAMPLE_RATE = 16_000

# The robot decides locally who each sentence was for (src/speech/gatekeeper.py). With that gate on,
# only what is for the robot ever reaches the agent, and the agent keeps its own quiet rule as a second
# opinion. With the gate off everything reaches it, and the quiet rule has to go with the gate: leaving it
# in is a second gate in the cloud, and on 2026-09-17 it swallowed a question asked straight at the robot
# ("Tu nao consegue ver o tabuleiro?") in a session started with ADDRESSEE_GATE=false.
ADDRESSING_GATED = (
    "Several people sit at the table and talk to each other. Answer when you are addressed (by name, "
    '"Reachy", or with a question to you), when someone asks the table a rules question, or when you are '
    "asked to continue. Otherwise stay quiet."
)
ADDRESSING_OPEN = (
    "Everything said at this table reaches you, with the robot's own echo already thrown out. Answer what "
    "you hear - a question, a statement, a correction, an aside about the game - and never stay silent "
    "because your name was not said. Stay quiet only for a sentence that plainly names another player as "
    "the person being spoken to."
)

AGENT_PROMPT = """You are Reachy, a small desktop robot sitting at a table where people play the board game Eldritch Horror (Fantasy Flight Games, 2013, base game only). You are a fellow player: you control your own investigator, you know the game, and you help the table when asked. You cannot see the board; ask people to describe it when it matters.

Golden rule: you do not know the rules by heart. For any question about rules, cards, investigators, Ancient Ones, monsters, phases or numbers, call the game_rules tool (or game_knowledge for one named thing) and answer ONLY from what it returns. Never invent rules, numbers or card texts. If the tool has nothing, say so in one sentence and suggest checking the Reference Guide. While a tool runs, say a short filler first ("Let me check the Reference Guide...").

How you talk: you are speaking out loud, so keep it to one to three short sentences; no lists, no markdown, no exclamation marks; warm and a little dry. You speak {language} and only {language}, for this whole session, whatever language you are spoken to in - if somebody says a sentence in another language, you still answer in {language}. The table plays the English edition, so keep every game term in English exactly as printed (investigator, card and Ancient One names; Doom, Omen, Clue, Gate, Mystery; Action Phase, Encounter Phase, Mythos Phase; Travel, Rest, Trade, Acquire Assets; Delayed, Detained; Lore, Influence, Observation, Strength, Will; Health, Sanity), and everything else in {language}.

{addressing} If someone says "stop", "wait", "hold on" or talks over you, stop at once, without finishing the sentence, and only say you are listening.

When somebody says a card came up - "saiu a carta 8, lê a parte de Rome" - call encounter_card with that number and that space, and read back what it returns, as it is. If it says nobody has read that card, ask them to read that part of it out once; it is remembered afterwards. Never make up what a card says.

You do not remember the game; the robot does, in its own state file. game_state is how you read it and remember_setup and remember_note are how you write to it. Never say you have written something down, noted it or will remember it unless one of those two tools has just returned - if somebody tells you something about this game and asks you to keep it, call remember_note before you answer. Call game_state BEFORE asking the table anything about the setup - the Ancient One, who plays which investigator, whose turn it is, the Mystery, the Reserve - and before answering any question about "our game". Never ask for something game_state already knows, and never contradict it. If game_state says something is missing, that is the one thing worth asking for."""

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "client",
        "name": "game_rules",
        "description": (
            "Look up the official Eldritch Horror rules and reference guide, plus the FAQ and the "
            "component knowledge base, for a question. Returns numbered passages with their source "
            "and page. Call it for EVERY rules, card, phase, action, token or number question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The rules question, in the player's own words (any language).",
                }
            },
            "required": ["question"],
        },
        "expects_response": True,
        "response_timeout_secs": 20,
        "pre_tool_speech": "auto",
    },
    {
        "type": "client",
        "name": "game_state",
        "description": (
            "The game in progress as the robot itself remembers it: the Ancient One and doom, the "
            "investigators and who plays each, the robot's own investigator and where it stands, "
            "the Mystery, the Reserve, the round, and what the robot still has to be told. Call it "
            "before asking the table about the setup and before answering anything about this game."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
        "expects_response": True,
        "response_timeout_secs": 10,
    },
    {
        "type": "client",
        "name": "take_turn",
        "description": (
            "Use when the table hands the robot's own investigator its turn - 'é a vez da Lily "
            "Chen', 'your turn', 'pode jogar'. The robot works out its move from the board and "
            "the rules and returns the move and the sentence to say. Say that sentence as it "
            "comes back, in your own voice but without changing what it decided, and then let "
            "the table move the piece. Never invent a move."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
        "expects_response": True,
        # The local model has to think about the whole board; cold it takes half a minute.
        "response_timeout_secs": 45,
        "pre_tool_speech": "auto",
    },
    {
        "type": "client",
        "name": "remember_setup",
        "description": (
            "Use when somebody tells the robot how this game is set up or corrects it - the "
            "Ancient One, who plays which investigator, the Mystery, what is in the Reserve. "
            "Pass what they said, word for word. The robot checks the names against the box, "
            "writes down what it can and returns what it wrote and what it still needs. Say that "
            "back. Do not use it for rules questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "said": {
                    "type": "string",
                    "description": "What the person said about the setup, in their own words.",
                }
            },
            "required": ["said"],
        },
        "expects_response": True,
        "response_timeout_secs": 30,
        "pre_tool_speech": "auto",
    },
    {
        "type": "client",
        "name": "remember_note",
        "description": (
            "Use when the table tells the robot something about this game that the setup does not "
            "cover and asks it to keep it - where a Gate is open, which monster stands on which "
            "space, a Clue somebody picked up, what the table agreed to do next, 'grava isso'. "
            "Pass what they said, word for word. It is written into the robot's state file and it "
            "is still there in the next session. Never claim you wrote something down without "
            "calling this. Not for rules questions and not for the setup - that is remember_setup."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "said": {
                    "type": "string",
                    "description": "What the person said, in their own words.",
                }
            },
            "required": ["said"],
        },
        "expects_response": True,
        "response_timeout_secs": 10,
        "pre_tool_speech": "auto",
    },
    {
        "type": "client",
        "name": "encounter_card",
        "description": (
            "Read back an Encounter card somebody has already read to the robot, by the number "
            "printed on it and the city it was drawn for: 'saiu a carta 8, a parte de Rome' is "
            "number 8, space Rome. Returns the card's text, or says nobody has read that one yet "
            "- in which case ask the table to read it once and it is remembered."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "number": {"type": "integer", "description": "The number printed on the card."},
                "space": {"type": "string", "description": "The city whose part is wanted, in English."},
                "deck": {
                    "type": "string",
                    "description": "America, Europe, Asia/Australia, General, Other World, Expedition, Research or Special. Leave empty when the city says it.",
                },
            },
            "required": ["number"],
        },
        "expects_response": True,
        "response_timeout_secs": 10,
    },
    {
        "type": "client",
        "name": "game_knowledge",
        "description": (
            "Facts about one named investigator, Ancient One, monster, gate location or FAQ topic "
            "of Eldritch Horror (stats, abilities, starting possessions, mystery). Use the exact "
            "English name."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Exact English name, e.g. 'Lily Chen' or 'Azathoth'.",
                },
                "question": {
                    "type": "string",
                    "description": "What about it, e.g. 'starting possessions' or 'how does it awaken'.",
                },
            },
            "required": ["name"],
        },
        "expects_response": True,
        "response_timeout_secs": 20,
        "pre_tool_speech": "auto",
    },
    {
        "type": "system",
        "name": "language_detection",
        "description": (
            "Switch the conversation language and voice. Call it BEFORE answering whenever the person "
            "speaks a supported language different from the current one, with the code: 'en' for "
            "English, 'pt' for Portuguese."
        ),
        "params": {"system_tool_type": "language_detection", "only_at_conversation_start": False},
    },
]


def _lang_code(tag: str) -> str:
    return tag.split("-")[0].lower()


def tools_for(language_lock: bool = LANGUAGE_LOCK) -> list[dict[str, Any]]:
    """The tools the agent gets. With the language locked, the switching tool is not one of them."""
    if not language_lock:
        return _TOOLS
    return [t for t in _TOOLS if t.get("name") != "language_detection"]


def agent_config(
    *,
    prompt: str = AGENT_PROMPT,
    voices: dict[str, str] | None = None,
    default_language: str = DEFAULT_LANGUAGE,
    languages: tuple[str, ...] = SPOKEN_LANGUAGES,
    llm: str = ELEVEN_AGENT_LLM,
    tts_model: str = ELEVENLABS_MODEL_ID,
    turn_timeout_s: float = ELEVEN_TURN_TIMEOUT_S,
    max_duration_s: int = ELEVEN_MAX_DURATION_S,
    state_text: str = "",
    language_lock: bool = LANGUAGE_LOCK,
    answer_everything: bool = False,
) -> dict[str, Any]:
    """The agent's ``conversation_config`` (pure; unit-tested).

    ``answer_everything`` says the local addressee gate is off, so every sentence at the table
    reaches the agent; its own "otherwise stay quiet" rule then has to go with the gate, or the
    robot is gated twice and ignores questions asked straight at it.

    ``language_lock`` is the owner's rule after a live session where the robot answered in
    English in the middle of a Portuguese game: the language is whatever the session starts in
    and it does not change. Locked, the agent gets no ``language_detection`` tool and no
    language presets to switch into - there is nothing to switch with - and its prompt says
    which language it speaks. Unlocked, it behaves as it used to.
    """
    voices = ELEVEN_AGENT_VOICES if voices is None else voices
    default_voice = voices.get(default_language, "")
    if not default_voice:
        raise ValueError(f"no native voice configured for {default_language}")
    prompt = prompt.replace("{language}", language_name(default_language))
    prompt = prompt.replace("{addressing}", ADDRESSING_OPEN if answer_everything else ADDRESSING_GATED)
    full_prompt = f"{prompt}\n\n{state_text}".rstrip() if state_text else prompt
    agent: dict[str, Any] = {
        "first_message": "",
        "language": _lang_code(default_language),
        "prompt": {"prompt": full_prompt, "tools": tools_for(language_lock), "temperature": 0.3},
    }
    if llm:
        agent["prompt"]["llm"] = llm
    presets: dict[str, Any] = {}
    for tag in () if language_lock else languages:
        if tag == default_language:
            continue
        voice = voices.get(tag)
        if not voice:
            raise ValueError(f"no native voice configured for {tag}")
        presets[_lang_code(tag)] = {
            "overrides": {
                "tts": {"voice_id": voice},
                "agent": {"language": _lang_code(tag)},
            }
        }
    return {
        "agent": agent,
        "tts": {
            "voice_id": default_voice,
            "model_id": tts_model,
            "agent_output_audio_format": f"pcm_{SAMPLE_RATE}",
            "optimize_streaming_latency": 3,
        },
        "asr": {
            "user_input_audio_format": f"pcm_{SAMPLE_RATE}",
            "keywords": ["Reachy", "Eldritch", "Doom", "Omen"],
        },
        "turn": {"turn_timeout": turn_timeout_s},
        "conversation": {"max_duration_seconds": max_duration_s},
        "language_presets": presets,
    }


# --------------------------------------------------------------------------- create / update
# The client may override language and voice per session (a language switch restarts the
# session with the other native voice); everything else stays as configured.
PLATFORM_SETTINGS: dict[str, Any] = {
    "overrides": {
        "conversation_config_override": {
            "agent": {"language": True, "first_message": True, "prompt": {"prompt": False}},
            "tts": {"voice_id": True},
        }
    }
}


def session_override(language: str, voices: dict[str, str] | None = None) -> dict[str, Any]:
    """``conversation_config_override`` that starts a session in ``language`` with its native voice."""
    voices = ELEVEN_AGENT_VOICES if voices is None else voices
    voice = voices.get(language)
    if not voice:
        raise ValueError(f"no native voice configured for {language}")
    return {"agent": {"language": _lang_code(language)}, "tts": {"voice_id": voice}}


def _headers(api_key: str) -> dict[str, str]:
    return {"xi-api-key": api_key, "Content-Type": "application/json"}


def load_agent_id(path: Path = AGENT_FILE) -> str:
    if ELEVEN_AGENT_ID:
        return ELEVEN_AGENT_ID
    if path.exists():
        return str(json.loads(path.read_text(encoding="utf-8")).get("agent_id", ""))
    return ""


def save_agent_id(agent_id: str, path: Path = AGENT_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"agent_id": agent_id}, indent=2), encoding="utf-8")


def ensure_agent(
    *,
    api_key: str = ELEVENLABS_API_KEY,
    state_text: str = "",
    name: str = ELEVEN_AGENT_NAME,
    language: str = DEFAULT_LANGUAGE,
    language_lock: bool = LANGUAGE_LOCK,
    answer_everything: bool = False,
    timeout: float = 30.0,
) -> str:
    """Create the agent once, then keep it in sync with :func:`agent_config` at every start.

    ``language`` is the one this session speaks: with the lock on it is written into the prompt
    and the agent is given no way to change it, so the agent has to be updated when a session
    starts in the other language - which is exactly what this function already does every time.
    """
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is empty")
    config = agent_config(
        state_text=state_text,
        default_language=language,
        language_lock=language_lock,
        answer_everything=answer_everything,
    )
    agent_id = load_agent_id()
    with httpx.Client(headers=_headers(api_key), timeout=timeout) as http:
        if agent_id:
            response = http.patch(
                f"{API}/agents/{agent_id}",
                json={"conversation_config": config, "platform_settings": PLATFORM_SETTINGS, "name": name},
            )
            if response.status_code == 404:
                agent_id = ""
            else:
                _raise_for(response, "update")
                log.info("agent %s updated", agent_id)
        if not agent_id:
            response = http.post(
                f"{API}/agents/create",
                json={"conversation_config": config, "platform_settings": PLATFORM_SETTINGS, "name": name},
            )
            _raise_for(response, "create")
            agent_id = str(response.json()["agent_id"])
            save_agent_id(agent_id)
            log.info("agent %s created", agent_id)
    return agent_id


def _raise_for(response: httpx.Response, action: str) -> None:
    if response.is_error:
        raise RuntimeError(f"agent {action} failed ({response.status_code}): {response.text[:600]}")


# --------------------------------------------------------------------------- client tools
def rules_lookup(question: str, *, retriever: Any = None) -> dict[str, Any]:
    """``game_rules`` implementation: passages plus a note telling the model how to use them."""
    from src.rag.retrieve import retrieve

    passages = (retriever or retrieve)(question, rules_limit=4, knowledge_limit=2)
    return {
        "passages": format_passages(passages),
        "note": (
            "Answer using ONLY these passages, in the language the player used, in one to three "
            "sentences, naming the source (Rulebook or Reference Guide, page) in a few words. "
            "Keep game terms in English as printed. Never invent rules, numbers or card texts."
            if passages
            else "No passage matched. Say you could not find it and suggest the Reference Guide."
        ),
    }


def knowledge_lookup(name: str, question: str = "", *, retriever: Any = None) -> dict[str, Any]:
    """``game_knowledge`` implementation: the knowledge base only, keyed by an exact name."""
    from src.rag.retrieve import retrieve

    query = f"{name} {question}".strip()
    passages = (retriever or retrieve)(query, rules_limit=1, knowledge_limit=4)
    return {
        "passages": format_passages(passages),
        "note": "Answer ONLY from these passages, in the player's language, one to three sentences.",
    }


CLOSING_RESULT: dict[str, Any] = {
    "passages": "",
    "note": "This voice session is closing; do not answer, the question is being re-asked elsewhere.",
}


def game_state_report(game: Any) -> dict[str, Any]:
    """The state the agent is allowed to speak from: what is known, and what is missing.

    Written as data rather than prose so the agent cannot mistake a summary for a rule, and kept
    short: it is read on every call and every token of it is paid for in latency.
    """
    if game is None:
        return {"known": False, "note": "No game has been set up yet. Ask the table to describe it."}
    mine = game.robot_investigator
    return {
        "known": True,
        "ancient_one": game.ancient_one.name if game.ancient_one else "",
        "doom": game.doom,
        "mystery": game.mystery,
        "round": game.round,
        "my_investigator": None
        if mine is None
        else {
            "name": mine.name,
            "space": mine.space,
            "health": mine.health,
            "sanity": mine.sanity,
            "clues": mine.clues,
            "possessions": list(mine.possessions),
        },
        "investigators": [
            {"name": i.name, "player": "me" if i.is_robot else i.controller, "space": i.space}
            for i in game.investigators
        ],
        "reserve": list(game.reserve),
        # What the table asked the robot to keep: gates, monsters, whatever has no field of its
        # own. Without this the robot could be told where a Gate was and never read it back.
        "notes": list(game.notes),
        "still_missing": game.missing(),
        "note": (
            "This is what the robot remembers. Speak from it, do not ask for anything in it, and "
            "ask only for what still_missing lists. 'notes' is what the table asked it to write "
            "down; answer from it instead of saying you do not know."
        ),
    }


def encounter_report(number: int, space: str = "", deck: str = "") -> dict[str, Any]:
    """One Encounter card as the table calls it, or an honest miss.

    The encounter decks are not in this repository and will not be: they are the box's own text,
    hundreds of cards of it. A card the table has read once is remembered
    (``src/rag/dictate.py``), and one nobody has read is a question, never an invention.
    """
    from src.rag.dictate import encounter_key, find_encounter
    from src.strategy.reference import region_of

    deck = deck or region_of(space)
    card = find_encounter(number, space=space, deck=deck)
    if card is None:
        return {
            "known": False,
            "asked_for": encounter_key(deck, number, space) or f"card {number}",
            "note": (
                "Nobody has read that card to me. Ask the table to read it out once - the part "
                "for that space - and I will have it from then on."
            ),
        }
    return {
        "known": True,
        "card": card.get("name", ""),
        "deck": card.get("deck", ""),
        "number": card.get("number"),
        "space": card.get("space", ""),
        "text": card.get("text", ""),
        "read_by": card.get("source_author", ""),
        "note": "Read this out as it is. It is the card as somebody at this table read it.",
    }


def client_tools(
    *,
    retriever: Any = None,
    on_call: Any = None,
    active: Callable[[], bool] | None = None,
    game: Callable[[], Any] | None = None,
    session: Callable[[], Any] | None = None,
) -> Any:
    """SDK ``ClientTools`` with the local implementations registered.

    ``game`` and ``session`` are read at call time, not at registration, because the robot learns
    the setup while the session is already running. ``session`` is the
    ``src/integration/game_session.GameSession``: it is how the agent takes the robot's turn and
    writes down a briefing, since the robot itself has no voice at the table - the agent is the
    one mouth, and these tools are its hands.

    ``active`` says whether the session these tools belong to is still the live one; a session
    being closed for a language switch gets :data:`CLOSING_RESULT` instead of a search, so the
    rules lookup does not run twice (once per session) for the same question.
    """
    from elevenlabs.conversational_ai.conversation import ClientTools

    tools = ClientTools()

    def game_rules(parameters: dict) -> str:
        if active is not None and not active():
            return json.dumps(CLOSING_RESULT, ensure_ascii=False)
        question = str(parameters.get("question", ""))
        result = rules_lookup(question, retriever=retriever)
        if on_call:
            on_call("game_rules", parameters, result)
        return json.dumps(result, ensure_ascii=False)  # the orchestrator validates the result as text

    def game_knowledge(parameters: dict) -> str:
        if active is not None and not active():
            return json.dumps(CLOSING_RESULT, ensure_ascii=False)
        result = knowledge_lookup(
            str(parameters.get("name", "")), str(parameters.get("question", "")), retriever=retriever
        )
        if on_call:
            on_call("game_knowledge", parameters, result)
        return json.dumps(result, ensure_ascii=False)  # the orchestrator validates the result as text

    def game_state(parameters: dict) -> str:
        result = game_state_report(game() if game is not None else None)
        if on_call:
            on_call("game_state", parameters, result)
        return json.dumps(result, ensure_ascii=False)

    def take_turn(parameters: dict) -> str:
        playing = session() if session is not None else None
        if playing is None:
            result = {
                "took_a_turn": False,
                "say": "",
                "note": "No game is loaded, so there is no turn for me to take.",
            }
        else:
            result = playing.turn_report()
        if on_call:
            on_call("take_turn", parameters, result)
        return json.dumps(result, ensure_ascii=False)

    def remember_setup(parameters: dict) -> str:
        said = str(parameters.get("said", "")).strip()
        playing = session() if session is not None else None
        if playing is None or not said:
            result = {
                "noted": [],
                "say": "",
                "note": "There is nothing to write it down in yet."
                if playing is None
                else "I need their words.",
            }
        else:
            result = playing.setup_report(said)
        if on_call:
            on_call("remember_setup", parameters, result)
        return json.dumps(result, ensure_ascii=False)

    def remember_note(parameters: dict) -> str:
        said = str(parameters.get("said", "")).strip()
        playing = session() if session is not None else None
        if playing is None or not said:
            result = {
                "written": "",
                "note": "There is nothing to write it down in yet."
                if playing is None
                else "I need their words.",
            }
        else:
            result = playing.note_report(said)
        if on_call:
            on_call("remember_note", parameters, result)
        return json.dumps(result, ensure_ascii=False)

    def encounter_card(parameters: dict) -> str:
        try:
            number = int(parameters.get("number"))
        except (TypeError, ValueError):
            return json.dumps({"known": False, "note": "I need the number printed on the card."})
        result = encounter_report(number, str(parameters.get("space", "")), str(parameters.get("deck", "")))
        if on_call:
            on_call("encounter_card", parameters, result)
        return json.dumps(result, ensure_ascii=False)

    tools.register("take_turn", take_turn)
    tools.register("remember_setup", remember_setup)
    tools.register("remember_note", remember_note)
    tools.register("encounter_card", encounter_card)
    tools.register("game_rules", game_rules)
    tools.register("game_knowledge", game_knowledge)
    tools.register("game_state", game_state)
    return tools


def language_of(code: str) -> str:
    """Whisper/agent two-letter code -> configured tag (``pt`` -> ``pt-BR``)."""
    for tag in SPOKEN_LANGUAGES:
        if _lang_code(tag) == (code or "").lower():
            return tag
    return DEFAULT_LANGUAGE


__all__ = [
    "AGENT_PROMPT",
    "ADDRESSING_GATED",
    "ADDRESSING_OPEN",
    "AGENT_FILE",
    "agent_config",
    "tools_for",
    "client_tools",
    "CLOSING_RESULT",
    "ensure_agent",
    "knowledge_lookup",
    "language_name",
    "language_of",
    "load_agent_id",
    "rules_lookup",
    "save_agent_id",
    "session_override",
]
