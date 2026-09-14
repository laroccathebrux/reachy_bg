import json

from src.speech.addressee import (
    Decision,
    TurnLogger,
    about_the_game,
    decide,
    is_question,
    looks_like_echo,
    mentions_person,
    mentions_robot,
    needs_rules,
)


def test_mentions_robot_accepts_whisper_misspellings():
    assert mentions_robot("Reachy, quantas ações eu tenho?")
    assert mentions_robot("Hey Rich, can you hear me?")
    assert mentions_robot("Ritchie what do you think")
    assert not mentions_robot("Vamos viajar para Londres")
    assert mentions_robot("O que você acha, Reachy?")
    assert not mentions_robot("The rich merchant card is expensive")  # not a vocative
    assert not mentions_robot("enriched")


def test_is_question_in_both_languages():
    assert is_question("Quantas ações eu posso fazer?", "pt-BR")
    assert is_question("quantas ações eu posso fazer", "pt-BR")
    assert is_question("Can I travel twice", "en-US")
    assert is_question("E posso viajar duas vezes na mesma ação", "pt-BR")
    assert not is_question("E eu vou viajar duas vezes", "pt-BR")  # a statement without intonation stays one
    assert is_question("so can I rest there", "en-US")
    assert not is_question("Eu vou viajar para Londres", "pt-BR")
    assert not is_question("I will rest this round", "en-US")


def test_about_the_game():
    assert about_the_game("posso fazer duas ações?", "pt-BR")
    assert about_the_game("is that action allowed?", "en-US")
    assert not about_the_game("quer um café?", "pt-BR")


def test_decisions_in_order_of_rules():
    assert decide("Reachy, o que você acha?", "pt-BR") == Decision(True, "name", 0.95)
    assert (
        decide("E na fase de Mythos?", "pt-BR", seconds_since_robot_spoke=3.0).reason == "follow_up_question"
    )
    assert decide("Ok, obrigado", "pt-BR", seconds_since_robot_spoke=2.0).reason == "follow_up_reply"
    assert decide("Ok, obrigado", "pt-BR", seconds_since_robot_spoke=30.0).addressed is False
    assert decide("Can I take two actions?", "en-US", robot_turn=True).reason == "robot_turn_question"
    late = decide(
        "Can I take two actions?", "en-US", seconds_since_robot_spoke=30.0, answer_game_questions=True
    )
    assert late == Decision(True, "game_question", 0.5)
    quiet = decide(
        "Can I take two actions?", "en-US", seconds_since_robot_spoke=30.0, answer_game_questions=False
    )
    assert quiet == Decision(False, "game_question", 0.5)
    assert decide("Quer um café?", "pt-BR").reason == "not_addressed"
    assert decide("Vou viajar para Londres", "pt-BR", seconds_since_robot_spoke=3.0).reason == "not_addressed"


def test_turn_logger_appends_json_lines(tmp_path):
    logger = TurnLogger(tmp_path / "log.jsonl")
    logger.log(Decision(True, "name", 0.95), text="Reachy?", speaker="Alessandro", language="pt-BR")
    logger.log(Decision(False, "not_addressed", 0.7), text="café", speaker="", language="pt-BR")
    lines = (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["addressed"] is True and first["reason"] == "name" and first["speaker"] == "Alessandro"
    assert "ts" in first


def test_looks_like_echo():
    answer = "Each investigator may perform up to two actions during the Action Phase."
    assert looks_like_echo("investigator may perform up to two actions", answer)
    assert looks_like_echo(
        "Isso se aplica ao número de Monstros", "Isso se aplica tanto ao número de Monstros gerados"
    )
    assert not looks_like_echo("Can I travel twice in a round?", answer)
    assert not looks_like_echo("anything", "")
    assert looks_like_echo(
        "Deixa eu verificar o vídeo.", "Deixe-me verificar o Guia de Referência."
    )  # misheard echo
    assert looks_like_echo(
        "Você deve...", "nessa fase você deve resolver um encontro"
    )  # short echo, every word spoken
    assert not looks_like_echo("Então eu posso pegar mais de um?", "Deixe-me verificar o Guia de Referência.")
    assert not looks_like_echo("wait, what about Rest?", answer)
    # A player's real sentence shares only function words with the robot's speech.
    robot = "Durante a Encounter Phase você deve resolver um encontro, não é possível descansar nessa fase."
    assert not looks_like_echo("Não, eu não falei. Eu não te interrompi, por que que tu parou?", robot)
    assert not looks_like_echo(
        "A minha pergunta foi se eu posso descansar durante a fase de encontros.", robot
    )


def test_second_person_counts_only_with_a_single_human():
    assert (
        decide("Você não parece disposta a conversar.", "pt-BR", humans_present=1).reason
        == "second_person_solo"
    )
    assert decide("Tu não me responde?", "pt-BR", humans_present=1).addressed
    assert decide("Can you hear me?", "en-US", humans_present=1).addressed
    assert not decide("Você não parece disposta a conversar.", "pt-BR", humans_present=2).addressed
    assert not decide("Você não parece disposta a conversar.", "pt-BR").addressed
    assert decide("Estou com uma dúvida aqui no jogo.", "pt-BR", humans_present=1).reason == "solo"
    assert not decide("Hmm", "pt-BR", humans_present=1).addressed
    assert not decide("Eu vou viajar para Londres.", "pt-BR", humans_present=2).addressed


def test_follow_up_needs_the_same_speaker():
    assert (
        decide("E depois?", "pt-BR", seconds_since_robot_spoke=3.0, follow_up_ok=True).reason
        == "follow_up_question"
    )
    assert not decide("E depois?", "pt-BR", seconds_since_robot_spoke=3.0, follow_up_ok=False).addressed
    assert not decide("Cough, cough.", "en-US", seconds_since_robot_spoke=2.0, follow_up_ok=False).addressed


def test_needs_rules_separates_game_talk_from_banter():
    assert needs_rules("Quantas ações eu posso fazer por rodada?", "pt-BR")
    assert needs_rules("What happens when Doom reaches zero?", "en-US")
    assert needs_rules("Posso usar o Flesh Ward contra um Monster?", "pt-BR")
    assert not needs_rules("Então esse é o seu nome.", "pt-BR")
    assert not needs_rules("Hey Rich, can you hear me?", "en-US")


def test_mentions_person_is_a_vocative_of_another_player():
    assert mentions_person("Bruno, o que você acha?", ["Alessandro", "Bruno"])
    assert mentions_person("what do you think, Ana?", ["Ana Paula"])
    assert mentions_person("Ei Alessandro, sua vez", ["Alessandro"])
    assert not mentions_person("o Bruno já viajou para Londres", ["Bruno"])  # mentioned, not addressed
    assert not mentions_person("Bruno, o que você acha?", [])


def test_naming_another_player_is_not_for_the_robot():
    others = ["Bruno"]
    assert decide("Bruno, quantas ações eu tenho?", "pt-BR", other_names=others) == Decision(
        False, "other_person", 0.85
    )
    assert (
        decide("Bruno, quantas ações eu tenho?", "pt-BR", other_names=others, humans_present=2).addressed
        is False
    )
    # The robot's name wins over another name in the same sentence.
    assert decide("Reachy, o Bruno pode viajar?", "pt-BR", other_names=others).reason == "name"
