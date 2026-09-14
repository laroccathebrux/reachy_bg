# Speech Pipeline: Listening, Knowing Who Speaks, Speaking

This document fixes the audio design for Phase 2 and records the study it is based on.

## Constraints that shape the design

1. **The robot's microphone is broken.** The flat cable of the Reachy Mini microphone array is
   torn. All speech input comes from a Mac input device; the robot keeps its speaker and camera.
   Consequences: no direction-of-arrival from the robot, no on-board echo cancellation, and
   the Mac's own echo cancellation must be used so the robot can be interrupted while it talks.
2. **Several people around one table, one microphone, no wake word.** The research question is
   *when* the robot should speak, so the pipeline must tell speakers apart and detect who is
   being addressed.
3. **Bilingual table.** Players switch between Brazilian Portuguese and English mid-session.
   The robot must understand both, answer in the language it was spoken to, and use a
   **native voice for each language**: a Brazilian voice never reads English and an American
   voice never reads Portuguese (see "Bilingual operation" below). Every model in the chain
   must be language-agnostic or support both languages.
4. **Dependency conflict.** The robot SDK (`reachy-mini`) needs `numpy >= 2.2`; the best
   streaming diarizer (`diart`) needs `numpy < 2`. They cannot share a virtual environment.

## Architecture

```
Mac input device (default: built-in mic; a USB mic such as the HyperX QuadCast also works)
        |
        v
 +--------------------------+      speaker events (JSON over WebSocket)
 | live diarizer sidecar    | ----------------------------------------+
 | own venv, Python 3.12    |                                         |
 | diart + pyannote 3.x     |                                         v
 +--------------------------+                              +----------------------+
        |                                                  | src/speech (in-proc) |
        | same audio stream (shared ring buffer / 2nd tap) | - mlx-whisper ASR    |
        +------------------------------------------------> | - speaker enrolment  |
                                                           | - addressee detector |
                                                           | - conversation FSM   |
                                                           +----------------------+
                                                                      |
                                                     text + speaker + "is it for me?"
                                                                      v
                                                             strategy / dialogue layer
                                                                      |
                                                                      v
                                                    TTS  ->  robot speaker (reachy-mini media)
```

- **Transcription**: `mlx-whisper` with `whisper-large-v3-turbo` runs on the Metal GPU. It is the
  only Whisper implementation that uses the Apple GPU from Python; `faster-whisper` is CPU-only
  on macOS.
- **Live diarization**: `diart` on top of `pyannote/segmentation-3.0` and `pyannote/embedding`,
  language-independent, no hard speaker cap, 0.5-1 s latency for "who is speaking", running
  as a separate process (`tools/live-diarizer/`, its own `pyproject.toml`) that publishes
  speaker turns over a local WebSocket (`DIARIZER_URL`).
- **Who is who**: at game start each player says one sentence; the embedding of that sentence
  (`pyannote/wespeaker-voxceleb-resnet34-LM`, ungated) becomes their voiceprint. Diarized
  clusters are matched to voiceprints by cosine distance (threshold about 0.7).
- **Addressee detection**: rule-based first (robot's name, second-person questions while it is
  the robot's turn), then an LLM classifier over the last few utterances. This is the research
  core; see [DESIGN_DOCUMENT.md](DESIGN_DOCUMENT.md).
- **Post-game processing** (offline, best accuracy): `pyannote.audio 4` with the
  `speaker-diarization-community-1` pipeline in exclusive mode, plus `WhisperX` for word-level
  speaker attribution, used to build the learning dataset.
- **Speech output**: ElevenLabs by default (the voice the owner already uses; good Portuguese),
  played through the robot speaker via the `reachy-mini` media API. A local TTS path is an
  optional later swap (`TTS_PROVIDER=local`).

## Bilingual operation

```
utterance audio
   |
   v
 mlx-whisper (WHISPER_LANGUAGE=auto) -> text + language id (pt / en) + confidence
   |
   v
 language tracker: per speaker, last language; the table language switches after
 LANGUAGE_SWITCH_THRESHOLD consecutive utterances in another supported language
   |
   v
 LLM prompt (always English) + instruction "reply in <language of the utterance>"
   |
   v
 TTS voice = voice_for(language):  pt-BR -> Brazilian native voice
                                   en-US -> American native voice
   |
   v
 robot speaker
```

Rules:

- **Reply in the language of the utterance being answered**, not in the table's majority
  language. A question in English gets an English answer with the American voice even if
  the last ten minutes were in Portuguese.
- **One native voice per language, never a multilingual voice reading a foreign language.**
  With ElevenLabs that means two voice ids (`ELEVENLABS_VOICE_ID_PT_BR`,
  `ELEVENLABS_VOICE_ID_EN_US`) on a multilingual model (`eleven_flash_v2_5` or
  `eleven_multilingual_v2`); the voice's native accent decides the sound, the model decides
  the language. With local TTS (`TTS_PROVIDER=local`) the same applies with per-language
  voices (Piper `pt_BR-faber-medium` / `en_US-lessac-medium`, or any other engine that ships
  native voices for both).
- **Game vocabulary stays in English inside Portuguese speech** (investigator names, card
  names, "Doom", "Omen") because that is how the physical components are printed; the
  Portuguese voice pronounces them as loan words, which is what players do too.
- **Language identification is per utterance**, from Whisper's own language head; a single
  English word inside a Portuguese sentence does not switch anything. When confidence is
  low, the previous language of that speaker wins.
- `validate_config()` refuses a setup where a supported language has no native voice, so a
  missing voice is caught at start-up instead of being heard as a bad accent at the table.
- Speech input works the same way for both languages: Whisper, pyannote and diart are
  language-independent; speaker voiceprints do not change with the language spoken.

## Phase 2a as built (2026-09-14)

```
Mac input device (AUDIO_INPUT_DEVICE; opened at 16 kHz when the device allows it, else resampled)
   |  30 ms int16 frames, capture thread
   v
Segmenter (src/speech/microphone.py): RMS level per frame in dBFS
   speech  = level > max(VAD_THRESHOLD_DBFS, noise floor + VAD_NOISE_MARGIN_DB [+ barge-in margin])
   start   = 3 consecutive speech frames, with VAD_PRE_ROLL_MS of audio kept from before
   end     = VAD_SILENCE_MS of silence (or VAD_MAX_UTTERANCE_S); < VAD_MIN_SPEECH_MS is dropped
   noise floor = exponential average that drops fast and rises 25x slower; frozen while the robot speaks
   |  Utterance (int16 audio, timestamps, peak level, WAV in data/captures/audio/)
   v
Transcriber (src/speech/asr.py): mlx-whisper large-v3-turbo, fp16, temperature 0
   1. language id on the mel spectrogram, argmax restricted to SPOKEN_LANGUAGES (pt / en)
   2. decode with that language pinned, WHISPER_PROMPT as vocabulary hint
   3. Whisper's no-speech / log-prob gate plus a short list of known hallucinations
   |  Transcript (text, pt-BR / en-US, confidence, latency)
   v
think() + voice() (src/integration/answering.py) -> Robot.say(clip, interrupt=...)
```

Barge-in: while the robot speaks, the segmenter runs with `BARGE_IN_MARGIN_DB` added to the
threshold and `Robot.say` polls `Microphone.voice_ms`; a voice above the bar for
`BARGE_IN_MIN_MS` stops the clip and the head wobbler, and that voice is transcribed as the
next utterance. When the clip ends normally, everything the microphone caught meanwhile is
discarded as echo.

Measured levels at the MacBook microphone (robot 60 cm away, owner at the keyboard):

| Signal | Level (RMS per 30 ms frame) |
|---|---|
| Room floor | -45 to -52 dBFS |
| Owner talking normally | peaks -28 to -31 dBFS |
| Robot's own voice (echo) | median -44, p90 -35, peaks -21 to -31 dBFS |
| `say` clip through the Mac speakers | peaks -31 dBFS |

So the echo peaks at the same level as a person talking: energy alone cannot separate the
two, and the default `BARGE_IN_MARGIN_DB=16` only lets a raised voice through. Two fixes,
in order of effort: (1) transcribe what was heard during playback and drop it when it
matches the answer being spoken (Whisper transcribes the robot's voice verbatim), which
allows a much lower margin; (2) a reference-based acoustic echo canceller, since the clip
being played is known sample by sample (NLMS in numpy, or the daemon's own audio path if it
ever exposes one). The HyperX QuadCast has a cardioid pattern and would also reject the
robot better than the MacBook's array, once it is unmuted.

Latencies measured on this Mac are in [PROJECT_STATUS.md](../PROJECT_STATUS.md).

## Why not the alternatives

| Option | Why not (for now) |
|---|---|
| NVIDIA Streaming Sortformer | Best streaming accuracy on English, but hard cap of 4 speakers, English-trained, CPU-only on Mac and NeMo does not install cleanly on Apple Silicon today. Worth a re-test later. |
| senko (CoreML) | Very fast offline diarization, no token, but no streaming API and weaker on meeting-style audio. Good candidate for the post-game step if speed matters more than accuracy. |
| pyannote 4 alone | No streaming mode. Used offline only. |
| Robot microphone array | Broken cable. If repaired, its direction-of-arrival would be a strong extra signal for addressee detection. |

## Pinned stacks

In-process (`uv sync --extra speech`): `mlx-whisper`, `sounddevice`, `torch >= 2.8`,
`torchaudio >= 2.8`, `pyannote.audio >= 4.0.7` (offline diarization).

Live diarizer sidecar (separate venv, Python 3.12, `brew install portaudio ffmpeg libsndfile`):

```
torch==2.8.0
torchaudio==2.8.0
numpy==1.26.4
pyannote.audio==3.4.0
diart==0.9.2
sounddevice>=0.4.6
```

Both need a Hugging Face token once (`HF_TOKEN`) to download the gated pyannote models, after
accepting their terms on the model pages.

## Expected latencies on the M1 Max (targets; measured values are in PROJECT_STATUS.md)

| Signal | Target |
|---|---|
| "Someone is speaking" (VAD) | < 0.3 s |
| "Speaker X is speaking" | 0.5 - 1 s |
| Portuguese text with speaker label | 1.5 - 3 s |
| Robot starts answering (after LLM) | < 4 s total |

Numbers above come from published benchmarks on comparable hardware, not from this machine.
Phase 2 starts by measuring them with a 10-minute recording of a real table.

## Sources

- pyannote community-1 model card: https://huggingface.co/pyannote/speaker-diarization-community-1
- diart: https://github.com/juanmc2005/diart and latency study https://arxiv.org/pdf/2407.04293
- Target-speaker streaming pipeline (diart + verification): https://arxiv.org/abs/2608.17972
- WhisperX: https://github.com/m-bain/whisperX
- NVIDIA Streaming Sortformer: https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1
- senko: https://github.com/narcotic-sh/senko
- mlx-whisper: https://pypi.org/project/mlx-whisper/
- faster-whisper has no Metal backend: https://github.com/SYSTRAN/faster-whisper/issues/911
