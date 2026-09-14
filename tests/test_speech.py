import wave

from src.llm.prompts import conversation_messages, format_passages, rules_question_messages
from src.speech.language import detect_language, language_name
from src.speech.tts import SAMPLE_RATE, elevenlabs_request, pcm16_to_wav


def test_detects_portuguese_and_english():
    assert detect_language("Quantas ações eu posso fazer por rodada?") == "pt-BR"
    assert detect_language("How many actions can I take per round?") == "en-US"
    assert detect_language("O que acontece quando o Doom chega a zero?") == "pt-BR"
    assert detect_language("What happens when Doom reaches zero?") == "en-US"


def test_game_terms_alone_do_not_flip_the_language():
    # English component names inside a Portuguese sentence stay Portuguese.
    assert detect_language("Eu vou usar o Flesh Ward na Jacqueline Fine agora") == "pt-BR"


def test_empty_or_signal_free_text_falls_back_to_default():
    assert detect_language("") == "pt-BR"
    assert detect_language("42") == "pt-BR"


def test_language_names():
    assert language_name("pt-BR") == "Brazilian Portuguese"
    assert language_name("en-US") == "American English"
    assert language_name("xx-YY") == "xx-YY"


def test_prompt_carries_language_and_passages():
    passages = [
        {
            "source": "rulebook",
            "path": "Phase 1: Action Phase",
            "page_start": 7,
            "text": "Up to two actions.",
        },
        {"kind": "investigator", "name": "Lily Chen", "text": "Strength 4."},
    ]
    messages = rules_question_messages("How many actions?", passages, "en-US")
    assert messages[0]["role"] == "system"
    assert "American English" in messages[0]["content"]
    assert "[1] (rulebook: Phase 1: Action Phase, page 7)" in messages[1]["content"]
    assert "[2] (investigator: Lily Chen)" in messages[1]["content"]
    assert (
        "Question from a player (it may continue the conversation above): How many actions?"
        in messages[1]["content"]
    )
    assert "Reply in American English, keeping the game terms in English" in messages[1]["content"]


def test_format_passages_handles_empty_list():
    assert format_passages([]) == "(no passages found)"


def test_elevenlabs_request_asks_for_pcm_16k():
    url, params, body = elevenlabs_request("Hello", "voice-x", model_id="eleven_flash_v2_5")
    assert url.endswith("/v1/text-to-speech/voice-x")
    assert params == {"output_format": "pcm_16000"}
    assert body == {"text": "Hello", "model_id": "eleven_flash_v2_5"}


def test_pcm16_to_wav_reports_duration(tmp_path):
    one_second = bytes(SAMPLE_RATE * 2)  # 16-bit mono silence
    path = tmp_path / "clip.wav"
    assert pcm16_to_wav(one_second, path) == 1.0
    with wave.open(str(path)) as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getnframes() == SAMPLE_RATE


def test_english_game_terms_inside_portuguese_stay_portuguese():
    assert detect_language("Posso viajar duas vezes na mesma Action Phase?") == "pt-BR"
    assert detect_language("Posso fazer Travel duas vezes no mesmo round?") == "pt-BR"
    assert detect_language("Quantos Mysteries precisamos resolver para vencer?") == "pt-BR"
    assert detect_language("Can I do the Travel action twice in the same round?") == "en-US"


def test_prompt_ends_with_language_reminder():
    messages = rules_question_messages("Posso descansar?", [], "pt-BR")
    assert "Reply in Brazilian Portuguese, keeping the game terms in English" in messages[1]["content"]


def test_format_passages_truncates_long_chunks_on_a_word_boundary():
    long = {"source": "rulebook", "path": "X", "text": "word " * 400}
    out = format_passages([long], max_chars=100)
    body = out.split("\n", 1)[1]
    assert body.endswith(" ...")
    assert len(body) <= 105


def test_conversation_messages_carry_recent_exchanges():
    messages = conversation_messages("E aí?", "pt-BR", recent=[("Oi", "Olá."), ("Tudo bem?", "Tudo.")])
    assert messages[0]["role"] == "system" and "Brazilian Portuguese" in messages[0]["content"]
    assert [m["role"] for m in messages[1:]] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[-1]["content"].startswith("E aí?")
    assert "Reply in Brazilian Portuguese" in messages[-1]["content"]


def test_rules_messages_carry_history_before_the_question():
    recent = [("Se eu comprar duas cartas?", "Não, " + "x" * 400)]
    messages = rules_question_messages("E preciso repor?", [], "pt-BR", recent)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "Se eu comprar duas cartas?"
    assert messages[2]["content"].endswith(" ...") and len(messages[2]["content"]) < 320
    assert "E preciso repor?" in messages[3]["content"]
