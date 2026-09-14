from src.rag.migrate_knowledge import (
    ANCIENT_ONES,
    ancient_one_record,
    find_residue,
    parse_investigator,
    transform,
    translate_text,
)

LEGACY_INVESTIGATOR = {
    "jogo_id": "eldritch-horror",
    "tipo": "investigador",
    "nome": "Lily Chen",
    "expansao": "Eldritch Horror (caixa base)",
    "base": True,
    "texto": (
        "Investigador: Lily Chen (The Martial Artist). Expansão: Eldritch Horror (caixa base) — DISPONÍVEL (caixa base).\n"
        "Vida (Health) 6, Sanidade (Sanity) 6. Atributos: Conhecimento (Lore) 2, Influência (Influence) 2, "
        "Observação (Observation) 2, Força (Strength) 4, Determinação (Will) 3.\n"
        "Ação especial: Spend any number of Health or Sanity, then recover an equal number of Health or Sanity.\n"
        "Habilidade passiva: When you improve a skill, you may immediately improve that skill again.\n"
        "Itens iniciais: 1 Protective Amulet Asset, 1 Lucky Rabbit's Foot Asset. Local inicial: Shanghai, China. "
        "Papel sugerido: Combat / All-Rounder. Classe(s): Guardião (Guardian)."
    ),
}

LEGACY_FAQ = {
    "tipo": "faq",
    "nome": "Is discarding a gate the same as closing a gate?",
    "expansao": "FAQ oficial (todas)",
    "base": True,
    "texto": "Pergunta: Is discarding a gate the same as closing a gate? Resposta: No.",
}

LEGACY_MONSTER = {
    "tipo": "monstro",
    "nome": "Sand Dweller",
    "expansao": "Under the Pyramids",
    "base": False,
    "texto": "Monstro: Sand Dweller (Normal). Expansão: Under the Pyramids — expansão, NÃO disponível.",
}


def test_investigator_is_translated_and_parsed():
    record = transform(LEGACY_INVESTIGATOR)
    assert record["kind"] == "investigator"
    assert record["expansion"] == "Eldritch Horror (base game)"
    assert record["base_game"] is True
    assert record["health"] == 6 and record["sanity"] == 6
    assert record["skills"] == {"lore": 2, "influence": 2, "observation": 2, "strength": 4, "will": 3}
    assert record["starting_location"] == "Shanghai, China"
    assert record["classes"] == ["Guardian"]
    assert record["action_ability"].startswith("Spend any number of Health")
    assert find_residue(record) == []


def test_faq_labels_become_q_and_a():
    record = transform(LEGACY_FAQ)
    assert record["kind"] == "faq"
    assert record["text"] == "Q: Is discarding a gate the same as closing a gate? A: No."
    assert record["expansion"] == "Official FAQ (all products)"


def test_expansion_monster_is_marked_not_owned():
    record = transform(LEGACY_MONSTER)
    assert record["kind"] == "monster"
    assert record["base_game"] is False
    assert "not owned" in record["text"]
    assert find_residue(record) == []


def test_unknown_kind_is_dropped():
    assert transform({"tipo": "livro", "nome": "x", "texto": "y"}) is None


def test_ancient_ones_are_rebuilt_from_verified_data():
    for name in ANCIENT_ONES:
        record = ancient_one_record(name)
        assert record["kind"] == "ancient_one"
        assert record["mysteries_to_solve"] == 3
        assert record["starting_doom"] in (12, 13, 14, 15)
        assert name in record["text"]
        assert find_residue(record) == []
    assert transform({"tipo": "anciao", "nome": "Cthulhu", "texto": "whatever"})["starting_doom"] == 12
    assert transform({"tipo": "anciao", "nome": "Nyarlathotep", "texto": "expansion"}) is None


def test_translate_text_leaves_english_untouched():
    english = "Q: Can Focus be used twice? A: No."
    assert translate_text(english) == english


def test_parse_investigator_returns_none_for_free_text():
    assert parse_investigator("Investigator: Someone. Not a sheet.") is None
