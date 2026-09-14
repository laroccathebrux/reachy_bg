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

# Frequent function words that rarely appear in the other language. Game vocabulary (Action
# Phase, Travel, Doom, Gate...) is deliberately absent: the table plays the English edition,
# so those words appear inside Portuguese sentences too and must not count as English.
_MARKERS: dict[str, frozenset[str]] = {
    "pt-BR": frozenset(
        "o a os as um uma de do da dos das em no na nos nas por para com sem que não nao sim é são "
        "sao eu você voce ele ela nós eles elas meu minha seu sua isso isto esse essa este esta aqui "
        "ali como quando onde qual quais quanto quantos quanta quantas pode posso podemos consigo "
        "dá da vou vai vamos tem tenho temos está estou estamos foi ser ter fazer faz feito jogar "
        "jogo rodada fase ação acao turno vez vezes duas dois mesma mesmo mesmas outra outro regra "
        "regras porque porquê então entao também tambem mas ou já ja ainda muito pouco bem mal "
        "precisa preciso precisamos devo deve devemos quero queremos viajar viajo mover descansar "
        "comprar lutar ganhar perder".split()
    ),
    "en-US": frozenset(
        "the an of to in at for with without that this these those is are was were be been am "
        "i you he she we they it my your his her our their me him them what which who whom when "
        "where why how can could may might should would will do does did have has had not no yes "
        "and or but so if then than there here about into from as by up down out over under "
        "twice again same once during each every any many much more".split()
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
