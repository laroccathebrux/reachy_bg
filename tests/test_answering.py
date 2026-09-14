from src.integration.answering import retrieval_query, split_sentences


def test_split_sentences_returns_complete_ones_and_the_rest():
    sentences, rest = split_sentences(
        "Each investigator has two actions. You may Travel twice? Not in one rou"
    )
    assert sentences == ["Each investigator has two actions.", "You may Travel twice?"]
    assert rest == "Not in one rou"


def test_split_sentences_final_flushes_everything():
    sentences, rest = split_sentences(
        "Cada investigator tem duas ações. Consulte o Reference Guide", final=True
    )
    assert sentences == ["Cada investigator tem duas ações.", "Consulte o Reference Guide"]
    assert rest == ""


def test_short_fragments_are_glued():
    sentences, rest = split_sentences("Yes. It is allowed once per round. No. ", final=True)
    assert sentences == ["Yes. It is allowed once per round. No."]
    assert rest == ""
    sentences, rest = split_sentences("Dr. Chen may Rest. Then travel.")
    assert sentences == ["Dr. Chen may Rest."]
    assert rest == "Then travel."


def test_empty_buffer():
    assert split_sentences("") == ([], "")
    assert split_sentences("   ", final=True) == ([], "")


def test_retrieval_query_borrows_the_previous_question_for_follow_ups():
    recent = [("Se eu comprar duas cartas da reserva, posso escolher a mesma opção?", "Não.")]
    assert retrieval_query("E eu preciso repor essas cartas?", recent).startswith("Se eu comprar duas cartas")
    assert retrieval_query("Então eu posso pegar mais de um.", recent).startswith("Se eu comprar")
    long = "Quando o Ancient One acorda, o que acontece com os investigadores que estão em Gates abertos?"
    assert retrieval_query(long, recent) == long
    assert retrieval_query("E depois?", None) == "E depois?"
