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

## Phase 2b as built (2026-09-14)

```
                    +----------------------------------+
Mac input device -->| tools/live-diarizer (own venv)    |--- ws://127.0.0.1:8765 --+
      |             | diart: segmentation-3.0+embedding |   hello / turn / step     |
      |             +----------------------------------+                           v
      v                                                            src/speech/diarization.py
Microphone + Segmenter (Phase 2a) --> Utterance                    DiarizerClient (optional)
      |                                                                            |
      v                                                                            |
Transcriber (Whisper) --> text, language                                           |
      |                                                                            |
      v                                                                            |
src/speech/speakers.py  SpeakerRegistry.identify(audio) -> (name, cosine)          |
      |                 wespeaker-voxceleb-resnet34-LM, enrolled at game start     |
      v                                                                            |
src/speech/addressee.py decide(text, language, since_robot_spoke, robot_turn)      |
      |                 name | follow_up_question | follow_up_reply |              |
      |                 robot_turn_question | game_question | not_addressed        |
      |                 (+ self_echo when the text matches the robot's last answer)|
      v                                                                            |
TurnLogger -> data/game_logs/addressee.jsonl  <-- diart label joined here  <-------+
      |
      v  only when addressed
think() + voice() -> Robot.say()
```

- **Enrolment** (`listen.py --players "Ana,Bruno"`): the robot asks each player for one
  sentence in the default language, keeps the utterance's embedding, and saves the registry
  to `data/speakers/voiceprints.json`. More clips per name can be added later; matching uses
  the best clip. Threshold `SPEAKER_MATCH_THRESHOLD=0.30` sits between the measured
  same-voice (0.36-0.84) and different-voice (< 0.15) similarities.
- **Diart label vs voiceprint**: the sidecar's label arrives 1.3-1.6 s after the utterance
  ends and tells "who is speaking now" with overlap handling; the voiceprint names the
  utterance once it is complete. The decision uses the voiceprint; the label is logged next
  to it so the two can be reconciled offline (and, later, online by votes).
- **The dataset**: every heard utterance becomes one JSON line with text, language, speaker
  name and score, diart label, seconds since the robot last spoke, the decision, its rule
  and confidence, and the path of the captured WAV. Annotating the `addressed` column of
  a few real sessions gives the training set for the learned classifier.

Sidecar pins that were not in the plan: `matplotlib<3.9`, `huggingface_hub<1.0`, and a
`torch.load(weights_only=False)` shim (pyannote 3 checkpoints are full pickles; torch 2.6+
refuses them by default). The Hugging Face token must be passed explicitly to pyannote 3;
`huggingface_hub`'s cached login (`hf auth login`) is read for that.

## Turn timing as built (2026-09-14)

```
person stops talking
  | VAD tail 0.6 s
  | voiceprint 0.03 s, Whisper 0.7-1.5 s (language id restricted to pt/en, fallback to the
  |   speaker's last language below 0.5 confidence)
  | addressee rules (name / solo / follow-up / game question) -> JSONL log
  | needs_rules? yes: retrieve 5 capped passages (0.1-0.2 s)   no: chat prompt + last 3 turns
  | LLM streamed; sentence 1 -> ElevenLabs (0.6-0.9 s) -> robot speaker   (sentences 2..n
  |   are generated and synthesized while sentence 1 plays)
robot starts talking: 5-6 s after the person stopped, on a busy Mac
  | while speaking: mic margin +4 dB; voice > 600 ms -> Whisper on that audio ->
  |   most words in the spoken sentences? echo, keep going : player -> stop clip + stream
  |   (reaction 1.4 s); the player's words are handled as the next utterance
```

## The conversation as built: ElevenLabs agent + local ears (2026-09-14)

```
Mac microphone --250 ms pcm16k--> RobotAudioInterface --echo gate--> addressee gate --> ElevenLabs agent
      |                                   |  held while the speaker   |  held per utterance   (ASR, LLM, TTS, turns)
      |  tap                              |  is busy (+0.3 s)         |  until the local        |  client_tool_call
      v                                   |                           |  decision               v
LocalEar: Segmenter (VAD) -> voiceprint  |                           |            game_rules / game_knowledge (Qdrant)
      |        -> diarizer label          |                           |                         |
      |  utterance closed                 |  <---- pcm16k audio + transcripts ----------------------+
      v                                   v
Gatekeeper: Whisper (single pass) ->     USB "Reachy Mini Audio" speaker + HeadSway
  decide() -> release_utterance()        ^
  (burst + 1 s silence) or discard       |  EchoAwareBargeIn (Whisper on the echo-held voice):
  or restart the session in the other    |  player's words -> release_gate(): forward held 2 s,
  language with the text                 |  cut playback, pass_through() for the rest
NameSpotter (Whisper on the voice so far): robot's name -> pass_through() at once
every decision -> addressee.jsonl (shadow=False when the gate controlled the audio); every event -> conversation.jsonl
```

The addressee gate is what makes "stay quiet" free: an utterance the rules judge as table
talk is dropped before any byte reaches the agent, so there is no cloud turn and no LLM
tokens for it. The cost is latency for the utterances that are released: the VAD tail
(0.6 s) plus the local Whisper (0.7 s on this Mac with the single-encoder-pass path), unless
the robot's name was heard in the partial transcript, in which case the gate opens while the
person is still talking. Measured numbers and the rules are in PROJECT_STATUS.md; the rules
themselves live in `src/speech/addressee.py` and the gate in `src/speech/gatekeeper.py`.
With one person at the table (`--humans 1`) or `--always-answer` the gate is off and the
decisions are logged as shadow decisions only.

One voice, two speakers: when a player starts talking over the end of the robot's sentence
the energy never drops, so the local VAD hands the gatekeeper a single utterance whose first
half is the robot's own echo. Judged whole it is echo - most of the words are the robot's - and
`self_echo` is the one verdict that survives `answer_everything`, so the player's sentence was
discarded with it (2026-09-17, "Dá uma olhada, porque eu já te falei isso"). `_after_playback`
now cuts at `last_played_at + gate_tail_s`: only what was said after the robot fell silent is
transcribed, and `release_utterance(..., since=...)` forwards only those frames, so the agent
never hears the robot's half either. A voice that ends before the robot does has no player half
and is still dropped whole by the `echo_gate` branch. A voiceprint that matched a player also
overrules the text: the robot's own voice measured at most 0.25 against an enrolled player and a
player 0.45 and up, so a name there means a person spoke, whatever the words look like.

There is a second gate, and it is easy to forget: the agent's own prompt used to end with
"Answer when you are addressed... otherwise stay quiet". That rule is the right one while the
local gate is on - the agent is a second opinion on what already got through - but with the
gate off it gates the robot twice in the cloud, where nothing here can see it. On 2026-09-17 a
question asked straight at the robot ("Tu não consegue ver o tabuleiro?") was forwarded by a
gate that was off and answered by nobody. So `ensure_agent(answer_everything=...)` now picks
between `ADDRESSING_GATED` and `ADDRESSING_OPEN` in `src/speech/eleven_agent.py`, and
`talk.py` settles the gate before it configures the agent rather than after.

What the agent may see: the camera is in the preview process and the conversation is in
`talk.py`, and for two days they were strangers - the preview wrote `board_state.json` and
nobody read it, while `GameState.board` sat empty in the other process and was saved as "the
board is empty". The prompt still opened with "You cannot see the board", so on 2026-09-17 the
robot answered "Não consigo ver o tabuleiro" to a direct question. `src/vision/board_link.py`
is the bridge: the preview's `GET /state` while it is up, `board_state.json` when it is not
(said as stale, never as live), nothing claimed when there is neither. `game_state` carries one
line of it on every turn and `look_at_board` has the piece list. The preview also reads
`board_state.json` back at start now, so restarting the camera no longer throws away the scan.

Holding the floor: `VAD_SILENCE_MS` is 600 ms, and somebody reading a card out loud pauses
for longer than that. The VAD closed the reader's sentence mid-card, the gate released it and the
agent answered over them (2026-09-17). `hold_floor()` stops releasing until the reader has been
quiet for `LISTEN_SILENCE_S`, or until `LISTEN_MAX_S` runs out; their audio still accumulates in
the utterance buffer and goes up as one turn when the reading ends, so the robot hears the whole
card once instead of interrupting it. The robot's own name still cuts through, because a person
who stops to call it is deliberately talking to it. The agent enters the mode through the
`listening` tool. The buffer caps at `HOLD_MAX_FRAMES` (30 s), so a very long reading loses its
oldest frames.

The camera is in the head, so speaking moves it. The speech-synced sway runs at 10 Hz while
the agent talks, which is small enough to slip under the motion watcher's "view moved" check and
large enough to shift where every space sits: on 2026-09-17 it read as three pieces leaving
three different spaces in the same instant, and emptied a board nobody had touched - the watcher
even attributed the departure to space 5 while the tracked piece stood on Rome. The preview
already drops its watcher while a scan turns the robot; it now also honours `hold_still`, and
`talk.py` refreshes that hold about once a second while the voice plays. Only the conversation
can say when that is, because the agent's audio goes straight to the USB speaker and the daemon
never sees it. It is a deadline rather than a flag, so a crash mid-sentence cannot leave the
camera switched off, and a preview that is down never blocks the voice.

Scanning on request: the sweep used to be a button on the preview page, so a table that
asked the robot to look at the board got a robot that could not. `scan_board` starts the
preview's three-view sweep and waits for `/scan_result`. It pauses the speech sway first,
because the sway commands the head at 10 Hz from the conversation process while the sweep turns
it from the camera process, and two hands on one motor blur every view.

What the agent may write: `remember_setup` for the setup (Ancient One, investigators, Mystery,
Reserve) and `remember_note` for everything else the table asks it to keep - where a Gate is
open, which monster stands where, what was agreed. Both land in the same `game_state.json`,
and `game_state` reads `notes` back, so a fact told on one evening is there the next. Before
`remember_note` existed the agent had nowhere to put such a fact and said "Anotado" anyway,
having called the read-only `game_state`; the prompt now forbids claiming a note without the
tool.

Language and voice: the agent's own `language_detection` tool is unreliable (the LLM often
answers in English with the Portuguese voice, tested live). So the switch is local and
deterministic: when the transcript (or, if the agent's ASR mangled the words, our Whisper on
the captured audio) says the player changed language, `talk.py` ends the session and opens a
new one with `conversation_config_override` = that language plus its native voice (the agent
has `platform_settings.overrides` enabled for `agent.language` and `tts.voice_id`), sends the
last three robot lines as context and re-asks the question. Measured: 1.5 s from detection to
the new session, correct voice in both directions (verified by voiceprint against reference
clips of Michelle and Lara).

Why the gate: the MacBook microphone has no acoustic echo cancellation and the robot's voice
arrives at -21..-31 dBFS, the same as a player. Without the gate the agent transcribed its own
sentences and answered them (verified). With the gate alone there is no voice interruption
(the earlier projects accepted that with the robot microphone); the local Whisper check on the
held audio restores it at the cost of about 1.4 s of reaction. A reference-based canceller
would make the gate unnecessary and is still the better long-term fix.

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
