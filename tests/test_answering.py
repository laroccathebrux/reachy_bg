from src.integration.answering import split_sentences


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
