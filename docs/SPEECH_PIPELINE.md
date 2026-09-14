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
3. **Portuguese at runtime.** Players speak Brazilian Portuguese; the language is a setting
   (`SPOKEN_LANGUAGE`), and every model used must be language-agnostic or support Portuguese.
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

## Expected latencies on the M1 Max (to be measured)

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
