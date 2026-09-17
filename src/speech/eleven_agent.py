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

AGENT_PROMPT = """You are Reachy, a small desktop robot sitting at a table where people play the board game Eldritch Horror (Fantasy Flight Games, 2013, base game only). You are a fellow player: you control your own investigator, you know the game, and you help the table when asked. You have a camera on the table and look_at_board is how you use it, so call it before saying anything about where pieces are - never say you cannot see the board. It sees pieces and where they stand, not what is printed on them, so a piece it has not been told the name of is just a piece; ask the table who it is.

Golden rule: you do not know the rules by heart. For any question about rules, cards, investigators, Ancient Ones, monsters, phases or numbers, call the game_rules tool (or game_knowledge for one named thing) and answer ONLY from what it returns. Never invent rules, numbers or card texts. If the tool has nothing, say so in one sentence and suggest checking the Reference Guide. While a tool runs, say a short filler first ("Let me check the Reference Guide...").

How you talk: you are speaking out loud, so keep it to one to three short sentences; no lists, no markdown, no exclamation marks; warm and a little dry. You speak {language} and only {language}, for this whole session, whatever language you are spoken to in - if somebody says a sentence in another language, you still answer in {language}. The table plays the English edition, so keep every game term in English exactly as printed (investigator, card and Ancient One names; Doom, Omen, Clue, Gate, Mystery; Action Phase, Encounter Phase, Mythos Phase; Travel, Rest, Trade, Acquire Assets; Delayed, Detained; Lore, Influence, Observation, Strength, Will; Health, Sanity), and everything else in {language}.

{addressing} If someone says "stop", "wait", "hold on" or talks over you, stop at once, without finishing the sentence, and only say you are listening.

Whose investigator is whose: yours is yours. When your own investigator takes the damage, loses the Sanity, rolls the dice or has the encounter, say "I" - "I lose 1 Sanity", "my Observation test" - in the language of the table, and never "your Sanity" or "how many dice do you have", which hands your own turn to the table and makes them explain their own game back to you. game_state marks your investigator; read it before you speak about anybody's sheet.

Somebody announcing what they are about to do is not a question. "Vou ler a carta de Shanghai", "deixa eu rolar os dados", "espera que eu estou lendo" - answer with one short line that you are listening, and nothing else: no rule, no tool, no summary of how encounters work. They will tell you the result, and the result is when you speak. While somebody is reading a card out loud, stay quiet until they stop, even through their pauses.

You are a player, not an assistant. Your investigator's moves are yours: decide them and say what you are doing, never ask whether you may. "Shall I buy the Bull Whip?", "May I carry on?", "Do you want me to do that now?" - none of those are yours to ask; say "I am buying the Bull Whip" and let the table resolve it. Never end a turn offering more help: when you are done, say whose turn it is next and stop. Ask the table for exactly two things - something only they can see (a die result, a card nobody has read to you, where a piece is) and a decision that is theirs (another investigator's move, whether the round moves on). If the answer would not change what you do, do not ask at all.

The game has a shape and you keep it: every round is Action Phase, then Encounter Phase, then Mythos Phase. game_state tells you the round, the phase and who the table is waiting on - read it, never ask. take_turn is the robot's own two actions and it records them, so once it has acted it says so instead of deciding again: do not ask it twice in one round and do not repeat the plan back a second time. next_phase is how the table moves the game on. If somebody reads out what a Reserve card costs, call card_value at once - asking for the same number twice is the fastest way to look like you were not listening.\n\nWhen somebody says a card came up - "saiu a carta 8, lê a parte de Rome" - call encounter_card with that number and that space, and read back what it returns, as it is. If it says nobody has read that card, ask them to read that part of it out once; it is remembered afterwards. Never make up what a card says.

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
            "The Reserve changes during play: when cards are bought and new ones replace them, "
            "say so through this, not as a note, or the robot goes on offering cards that are "
            "gone. Their values go to card_value in the same breath. "
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
            "Use when the table tells the robot something about this game that no other tool "
            "holds - where a Gate is open, which monster stands on which space, a Clue somebody "
            "picked up, what the table agreed to do next. Pass their own words, and only what "
            "they actually said: never write down something you worked out yourself from "
            "game_state, and never write down what you just did or decided - that is already "
            "recorded. Anything that has its own tool goes there instead, because a note is text "
            "nobody can act on: what a Reserve card costs is card_value, what is in the Reserve "
            "and who plays which investigator is remember_setup, a rules question is game_rules. "
            "Never claim you wrote something down without calling this."
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
        "name": "look_at_board",
        "description": (
            "Look at the table through the robot's camera: every piece it is tracking and the "
            "space each one stands on. Call it whenever the table asks what is on the board, "
            "where something is, or whether you can see it. It reports positions, not identities "
            "- a piece nobody has named is 'a piece', so ask the table who it is rather than "
            "guessing. If it says the camera is not running, say that instead of pretending."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
        "expects_response": True,
        "response_timeout_secs": 10,
        "pre_tool_speech": "auto",
    },
    {
        "type": "client",
        "name": "card_value",
        "description": (
            "Use the moment somebody reads out what a Reserve card costs - 'Lucky Cigarette Case "
            "é dois', 'o Kerosene custa um'. Pass every card and number they said. The robot "
            "knows what each card does and never what it costs, so without this it asks for the "
            "same numbers again on the next turn. Call it before answering, then say you wrote "
            "them down, in one sentence."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cards": {
                    "type": "array",
                    "description": "One entry per card whose value was read out.",
                    "items": {
                        "type": "object",
                        "description": "One card and the number printed on it.",
                        "properties": {
                            "name": {"type": "string", "description": "The card name, in English."},
                            "value": {"type": "integer", "description": "The number printed on it."},
                        },
                        "required": ["name", "value"],
                    },
                }
            },
            "required": ["cards"],
        },
        "expects_response": True,
        "response_timeout_secs": 10,
        "pre_tool_speech": "auto",
    },
    {
        "type": "client",
        "name": "listening",
        "description": (
            "Somebody says they are about to read something out loud - a card, an encounter, a "
            "rule from the box - or asks you to wait because they are still talking. Call this "
            "first, then say one short line that you are listening and nothing else. It keeps "
            "you quiet through the pauses people take while reading, which is when you were "
            "interrupting them. You will hear the whole thing at once when they stop."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
        "expects_response": True,
        "response_timeout_secs": 10,
    },
    {
        "type": "client",
        "name": "skill_test",
        "description": (
            "Somebody read out the dice of a skill test: pass every number they said and this "
            "counts the successes. Always call it - never count the dice yourself, and never say "
            "how many successes there were before it answers. It knows what counts as a success, "
            "Blessed and Cursed included."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dice": {
                    "type": "array",
                    "items": {"type": "integer", "description": "One die face, 1 to 6."},
                    "description": "Every die face they read out, in any order.",
                },
                "investigator": {
                    "type": "string",
                    "description": "Whose test it is, in English. Leave it out for your own.",
                },
            },
            "required": ["dice"],
        },
        "expects_response": True,
        "response_timeout_secs": 10,
    },
    {
        "type": "client",
        "name": "apply_effect",
        "description": (
            "What a resolved encounter, card or Mythos effect did to an investigator - Health or "
            "Sanity lost or recovered, Clues gained or spent, a card gained or discarded, a "
            "Condition taken or removed. Give the change, not the total: losing 1 Sanity is "
            "sanity -1. Call it the moment the effect is resolved; saying you will write it down "
            "without calling it leaves the sheet wrong for the rest of the game."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "investigator": {
                    "type": "string",
                    "description": "Whose sheet, in English. Leave it out for your own.",
                },
                "health": {"type": "integer", "description": "Health gained (+) or lost (-)."},
                "sanity": {"type": "integer", "description": "Sanity gained (+) or lost (-)."},
                "clues": {"type": "integer", "description": "Clues gained (+) or spent (-)."},
                "gain": {
                    "type": "array",
                    "items": {"type": "string", "description": "One name, in English."},
                    "description": "Cards gained, English names.",
                },
                "lose": {
                    "type": "array",
                    "items": {"type": "string", "description": "One card name, in English."},
                    "description": "Cards discarded.",
                },
                "conditions_gained": {
                    "type": "array",
                    "items": {"type": "string", "description": "One name, in English."},
                    "description": "Conditions taken.",
                },
                "conditions_lost": {
                    "type": "array",
                    "items": {"type": "string", "description": "One name, in English."},
                    "description": "Conditions removed.",
                },
            },
            "required": [],
        },
        "expects_response": True,
        "response_timeout_secs": 10,
        "pre_tool_speech": "auto",
    },
    {
        "type": "client",
        "name": "next_phase",
        "description": (
            "Move the game on one phase when the table says so - 'acabou a Action Phase', 'vamos "
            "pro Mythos', 'próxima rodada', 'terminou o turno'. Every round is Action Phase, then "
            "Encounter Phase, then Mythos Phase, and after the Mythos Phase a new round opens. "
            "Returns where the game now is; say that in one sentence. Do not call it to answer a "
            "question about where the game is - game_state already says that."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
        "expects_response": True,
        "response_timeout_secs": 10,
        "pre_tool_speech": "auto",
    },
    {
        "type": "client",
        "name": "encounter_done",
        "description": (
            "The table says an investigator has resolved its encounter in the Encounter Phase. "
            "Give the investigator's name; leave it out for the robot's own."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "investigator": {"type": "string", "description": "Whose encounter it was, in English."}
            },
            "required": [],
        },
        "expects_response": True,
        "response_timeout_secs": 10,
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


def _card_by_name(name: str) -> str:
    """The printed text of a base-game card, matched on the name alone, or "".

    The 76 base Assets ship in this repository with what each one does, and only the planner
    ever read them: the agent went to the vector search, where an exact name is exactly what
    embeddings are worst at. "Witch Doctor" came back as Diana Stanley and the Witch monster
    while the card sat in the file, and "Double Barreled Shotgun" missed because the box spells
    it with a hyphen. So the name is tried here first, ignoring punctuation and case.
    """
    from src.strategy.moves import _cards

    def key(text: str) -> str:
        return "".join(c for c in text.lower() if c.isalnum())

    wanted = key(name)
    if not wanted:
        return ""
    cards = _cards()
    for card_name, card in cards.items():
        if key(card_name) == wanted:
            return str(card.get("effect") or "")
    return ""


def knowledge_lookup(name: str, question: str = "", *, retriever: Any = None) -> dict[str, Any]:
    """``game_knowledge`` implementation: the card list first, then the knowledge base."""
    from src.rag.retrieve import retrieve

    printed = _card_by_name(name)
    query = f"{name} {question}".strip()
    passages = (retriever or retrieve)(query, rules_limit=1, knowledge_limit=4)
    text = format_passages(passages)
    if printed:
        text = f"{name} (printed on the card): {printed}\n\n{text}".strip()
    return {
        "passages": text,
        "note": "Answer ONLY from these passages, in the player's language, one to three sentences.",
    }


CLOSING_RESULT: dict[str, Any] = {
    "passages": "",
    "note": "This voice session is closing; do not answer, the question is being re-asked elsewhere.",
}


def game_state_report(game: Any, board: str = "") -> dict[str, Any]:
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
        "where_we_are": game.where_we_are(),
        "my_investigator": None
        if mine is None
        else {
            "name": mine.name,
            "space": mine.space,
            "health": mine.health,
            "sanity": mine.sanity,
            "clues": mine.clues,
            "possessions": list(mine.possessions),
            "conditions": list(mine.conditions),
            # The skills, so the number of dice in a test is never a question for the table.
            "skills": {
                "lore": mine.sheet.lore,
                "influence": mine.sheet.influence,
                "observation": mine.sheet.observation,
                "strength": mine.sheet.strength,
                "will": mine.sheet.will,
            },
        },
        # Health, Sanity and Clues for everybody, not only for the robot's own. apply_effect
        # wrote a Clue that Jacqueline Fine had spent and this line did not carry it back, so the
        # robot said it had nothing recorded and asked the table for a number it already held.
        "investigators": [
            {
                "name": i.name,
                "player": "me" if i.is_robot else i.controller,
                "space": i.space,
                "health": i.health,
                "sanity": i.sanity,
                "clues": i.clues,
                "conditions": list(i.conditions),
            }
            for i in game.investigators
        ],
        "reserve": list(game.reserve),
        # What the table asked the robot to keep: gates, monsters, whatever has no field of its
        # own. Without this the robot could be told where a Gate was and never read it back.
        "notes": list(game.notes),
        # One line, because game_state is read on every turn. look_at_board has the detail.
        "board": board,
        "still_missing": game.missing(),
        "note": (
            "This is what the robot remembers. Speak from it, do not ask for anything in it, and "
            "ask only for what still_missing lists. 'notes' is what the table asked it to write "
            "down; answer from it instead of saying you do not know. 'board' is what the camera "
            "sees; call look_at_board for which piece is where. 'where_we_are' is the round and "
            "the phase: never ask the table what phase it is, and never take a turn out of the "
            "Action Phase."
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
    keeper: Callable[[], Any] | None = None,
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
        from src.vision.board_link import summary

        try:
            board = summary()
        except Exception as exc:  # the camera must never take the conversation down
            log.warning("reading the board failed: %s", exc)
            board = ""
        result = game_state_report(game() if game is not None else None, board)
        if on_call:
            on_call("game_state", parameters, result)
        return json.dumps(result, ensure_ascii=False)

    def look_at_board(parameters: dict) -> str:
        from src.vision.board_link import board_now

        try:
            result = board_now()
        except Exception as exc:
            log.warning("looking at the board failed: %s", exc)
            result = {"seen": False, "note": f"I could not use the camera: {exc}"}
        if on_call:
            on_call("look_at_board", parameters, result)
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

    def card_value(parameters: dict) -> str:
        from src.rag.dictate import remember_value

        cards = parameters.get("cards") or []
        written, refused = [], []
        for item in cards if isinstance(cards, list) else []:
            name = str((item or {}).get("name", "")).strip()
            try:
                value = int((item or {}).get("value"))
            except (TypeError, ValueError):
                refused.append(name or "a card with no number")
                continue
            try:
                remember_value(name, value, read_by="the table")
            except Exception as exc:
                log.warning("could not keep the value of %s: %s", name, exc)
                refused.append(name)
                continue
            written.append({"name": name, "value": value})
        result = {
            "written": written,
            "not_written": refused,
            "note": "The values are kept and I will not ask for them again."
            if written
            else "I did not catch a card and a number; ask them to say it again.",
        }
        if on_call:
            on_call("card_value", parameters, result)
        return json.dumps(result, ensure_ascii=False)

    def listening(parameters: dict) -> str:
        result = (
            keeper().hold_floor()
            if keeper is not None and keeper() is not None
            else {"listening": False, "note": "Say that you are listening and wait."}
        )
        if on_call:
            on_call("listening", parameters, result)
        return json.dumps(result, ensure_ascii=False)

    def skill_test(parameters: dict) -> str:
        playing = session() if session is not None else None
        dice = parameters.get("dice") or []
        result = (
            {"note": "There is no game loaded, so I do not know whose test this is."}
            if playing is None
            else playing.test_report(list(dice), str(parameters.get("investigator", "")))
        )
        if on_call:
            on_call("skill_test", parameters, result)
        return json.dumps(result, ensure_ascii=False)

    def apply_effect(parameters: dict) -> str:
        playing = session() if session is not None else None
        if playing is None:
            result: dict = {"applied": False, "note": "There is no game to write it in yet."}
        else:
            fields = ("health", "sanity", "clues", "gain", "lose", "conditions_gained", "conditions_lost")
            deltas = {k: parameters[k] for k in fields if parameters.get(k) not in (None, "", [])}
            result = playing.effect_report(str(parameters.get("investigator", "")), **deltas)
        if on_call:
            on_call("apply_effect", parameters, result)
        return json.dumps(result, ensure_ascii=False)

    def next_phase(parameters: dict) -> str:
        playing = session() if session is not None else None
        result = {"note": "There is no game to move on yet."} if playing is None else playing.phase_report()
        if on_call:
            on_call("next_phase", parameters, result)
        return json.dumps(result, ensure_ascii=False)

    def encounter_done(parameters: dict) -> str:
        playing = session() if session is not None else None
        result = (
            {"recorded": False, "note": "There is no game to write it in yet."}
            if playing is None
            else playing.encounter_report(str(parameters.get("investigator", "")))
        )
        if on_call:
            on_call("encounter_done", parameters, result)
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
    tools.register("card_value", card_value)
    tools.register("listening", listening)
    tools.register("skill_test", skill_test)
    tools.register("apply_effect", apply_effect)
    tools.register("next_phase", next_phase)
    tools.register("encounter_done", encounter_done)
    tools.register("encounter_card", encounter_card)
    tools.register("game_rules", game_rules)
    tools.register("game_knowledge", game_knowledge)
    tools.register("game_state", game_state)
    tools.register("look_at_board", look_at_board)
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
