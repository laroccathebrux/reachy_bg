"""Replay saved captures through the addressee gate, without the microphone or the agent.

    uv run python scripts/gate_replay.py data/captures/audio/20260914_16*.wav --humans 2

Each WAV (one utterance as the local VAD cut it) is transcribed by the local Whisper, matched
against the enrolled voiceprints and judged by the gatekeeper exactly as at the table; the
verdicts and timings are printed and appended to ``data/game_logs/addressee_replay.jsonl``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import GAME_LOG_DIR  # noqa: E402
from src.speech.addressee import TurnLogger  # noqa: E402
from src.speech.asr import Transcriber  # noqa: E402
from src.speech.gatekeeper import Gatekeeper  # noqa: E402
from src.speech.microphone import Utterance, read_wav  # noqa: E402
from src.speech.speakers import SpeakerRegistry  # noqa: E402


class RecordingAudio:
    """Stands in for RobotAudioInterface: counts what the gate would have sent or dropped."""

    last_played_at = -1000.0
    gate_tail_s = 0.3

    def release_utterance(self, until: float) -> int:
        return int(until * 4)

    def discard_utterance(self, until: float) -> int:
        return int(until * 4)

    def pass_through(self) -> int:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("wavs", nargs="+", type=Path)
    parser.add_argument("--humans", type=int, default=2)
    parser.add_argument("--language", default="pt-BR", help="voice language of the session")
    parser.add_argument("--last-answer", default="", help="what the robot said last (echo check)")
    args = parser.parse_args()

    registry = SpeakerRegistry.load()
    transcriber = Transcriber()
    print(f"whisper warm-up {transcriber.warm_up():.1f}s; voices: {', '.join(registry.names) or 'none'}")
    audio = RecordingAudio()
    keeper = Gatekeeper(
        audio,
        transcriber,
        TurnLogger(GAME_LOG_DIR / "addressee_replay.jsonl"),
        humans=args.humans,
        names=tuple(registry.names),
        spoken_recently=lambda: args.last_answer,
        voice_language=lambda: args.language,
        on_switch=lambda lang, text, source: print(f"    -> would restart the session in {lang} ({source})"),
    )
    whisper = []
    for path in args.wavs:
        samples = read_wav(path)
        duration = samples.size / 16000
        now = time.monotonic()
        utterance = Utterance(samples, 16000, now - duration, now - 0.6, now, -20.0, path=path)
        name, score = registry.identify(samples) if registry.names else ("", 0.0)
        verdict = keeper.judge(utterance, name, score)
        whisper.append(verdict.whisper_ms)
        print(
            f"{path.name}  {duration:4.1f}s  [{name or '?'} {score:.2f}]  {verdict.route:9s} "
            f"{verdict.decision.reason:18s} whisper {verdict.whisper_ms:4d} ms  decided {verdict.decision_ms:4d} ms"
            f"\n    {verdict.language}: {verdict.text}"
        )
    if whisper:
        whisper.sort()
        print(f"\nwhisper ms: min {whisper[0]}  median {whisper[len(whisper) // 2]}  max {whisper[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
