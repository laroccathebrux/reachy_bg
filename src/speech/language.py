"""Language identification for short *typed* text (Portuguese vs English).

Spoken input gets its language from Whisper; this module covers typed input (the smoke test,
tests, a chat fallback) with a cheap stop-word vote. It only decides between the configured
``SPOKEN_LANGUAGES`` and falls back to ``DEFAULT_LANGUAGE`` when the text carries no signal.
"""

from __future__ import annotations

import re
import unicodedata

from src.config import DEFAULT_LANGUAGE, SPOKEN_LANGUAGES

_WORD = re.compile(r"[a-zà-ú]+", re.IGNORECASE)

# Frequent function words that rarely appear in the other language.
_MARKERS: dict[str, frozenset[str]] = {
    "pt-BR": frozenset(
        "o a os as um uma de do da dos das em no na nos nas por para com sem que não sim é são "
        "eu você voce ele ela nós nos eles elas meu minha seu sua isso isto aqui ali como quando "
        "onde qual quais quanto quantos pode posso podemos vou vai vamos tem tenho temos está "
        "estou estamos foi ser ter fazer faz jogar jogo rodada fase ação acao turno regra regras "
        "porque porquê então entao também tambem mas ou já ja ainda muito pouco bem mal".split()
    ),
    "en-US": frozenset(
        "the a an of to in on at for with without that this these those is are was were be been "
        "i you he she we they it my your his her our their me him them what which who whom when "
        "where why how can could may might should would will do does did have has had not no yes "
        "and or but so if then than there here about into from as by up down out over under "
        "round phase action turn rule rules game play".split()
    ),
}


def _strip_accents(word: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")


def detect_language(text: str) -> str:
    """Return the BCP-47 code of the most likely configured language for ``text``."""
    words = [w.lower() for w in _WORD.findall(text or "")]
    if not words:
        return DEFAULT_LANGUAGE
    scores: dict[str, int] = {}
    for lang in SPOKEN_LANGUAGES:
        markers = _MARKERS.get(lang, frozenset())
        scores[lang] = sum(1 for w in words if w in markers or _strip_accents(w) in markers)
    # Accented characters are a strong Portuguese signal on their own.
    if "pt-BR" in scores and any(ord(c) > 127 for c in text):
        scores["pt-BR"] += 1
    best = max(scores.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else DEFAULT_LANGUAGE


LANGUAGE_NAMES = {"pt-BR": "Brazilian Portuguese", "en-US": "American English"}


def language_name(code: str) -> str:
    """Human-readable name for prompts, e.g. 'Brazilian Portuguese'."""
    return LANGUAGE_NAMES.get(code, code)


__all__ = ["detect_language", "language_name", "LANGUAGE_NAMES"]
