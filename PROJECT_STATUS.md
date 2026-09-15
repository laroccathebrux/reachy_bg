# Project Status

Updated 2026-09-14 (Phase 2b).

## Done: Phase 0, foundation

- Repository restarted from scratch in English with correct facts: Eldritch Horror 2013 base
  game, Reachy Mini Lite, broken robot microphone, shared work Mac. `CLAUDE.md` carries the
  rules for every coding session.
- Python 3.12 environment managed by `uv` (`pyproject.toml`, `uv.lock`); `reachy-mini` 1.10
  installs and imports on the Mac.
- Configuration (`src/config.py`) and logging (`src/logger.py`) with tests.
- Retrieval layer `src/rag/`:
  - `embeddings.py` (bge-m3 via Ollama), `store.py` (Qdrant lifecycle, upsert, search),
    `collections.py` (payload contracts).
  - `pdf_sections.py` + `chunking.py`: typography- and layout-aware splitter for the FFG PDFs
    (two-column reading order, running heads removed, wrapped headings joined, icon fonts
    mapped).
  - `ingest_rules.py`: official English rulebook + reference guide -> `bg_rules`
    (71 + 80 sections, 216 chunks).
  - `migrate_knowledge.py`: legacy Portuguese-labelled knowledge -> `bg_knowledge` in English
    with structured investigator sheets and verified Ancient One records (322 points).
  - `sessions.py`: `bg_sessions` round memory with `record_round` / `recall`.
- Documentation rewritten: design, game reference, setup protocol, physical setup, speech
  pipeline, model sizing, learning, getting started, workflow.
- 31 unit tests passing; none require external services.

## Done: step 1, end-to-end smoke test on the physical robot (2026-09-14)

`uv run python -m src.integration.smoke` runs typed question -> retrieval (`bg_rules` +
`bg_knowledge`) -> Qwen 3.6 via Ollama -> ElevenLabs with the native voice for the detected
language -> playback on the Reachy Mini speaker, with a nod and antenna gestures. New modules:
`src/llm/` (Ollama client, prompts), `src/rag/retrieve.py`, `src/speech/` (typed-text language
detection, TTS), `src/robot/reachy.py` (SDK wrapper with a speaker-only simulation mode).

Measured on the work Mac under heavy load (load average ~70, 8 GB swap in use):

| Stage | English question | Portuguese question |
|---|---|---|
| Embedding + retrieval (8 passages) | 2.4 s (cold) | 0.2 s |
| LLM answer, 3 sentences | 12.4 s at 4.3 tok/s | 7.8 s at 7.7 tok/s |
| First LLM call of the session (model load) | 41 s | - |
| ElevenLabs TTS (16 kHz PCM) | 1.5 s | 1.4 s |
| Spoken answer length | 16 s | 17 s |

Findings:
- The robot daemon works on this macOS with SDK 1.10.0 (control loop 49 Hz), but only after
  `scripts/venv_postinstall.py`: the venv's `.pth` files get the macOS `hidden` flag re-applied
  within a minute by something on this machine, and Python 3.12.13 skips hidden `.pth` files,
  so the GStreamer bindings were invisible. The daemon takes about 3 minutes to open port 8000.
- macOS denied camera access to the daemon started from the terminal; grant it in
  System Settings > Privacy & Security > Camera before Phase 1.
- LLM throughput is well below the 30-50 tok/s expected for a 3B-active model because the
  machine was saturated; answers of 3 sentences still land in 8-12 s. Re-measure on a quiet
  machine and consider `think=False` plus shorter answers for in-game replies.
- Language detection and voice selection behaved correctly in both languages; game terms
  stay in English inside Portuguese answers as agreed.

## Done: Phase 2a, listening and transcribing on the physical robot (2026-09-14)

`uv run python -m src.integration.listen` runs Mac microphone -> energy VAD -> mlx-whisper ->
retrieval -> Qwen 3.6 -> ElevenLabs native voice -> robot speaker, in a loop, with an
interruptible `Robot.say`. New modules: `src/speech/microphone.py` (device selection,
`Segmenter` VAD, capture thread, WAV captures in `data/captures/audio/`), `src/speech/asr.py`
(`Transcriber`: language identification restricted to `SPOKEN_LANGUAGES`, then decoding with
the language pinned), `src/integration/answering.py` (retrieval + LLM + TTS shared with the
smoke test), `src/integration/listen.py`. 58 tests; the Whisper test runs only when the model
is in the Hugging Face cache and uses macOS `say` to make its clip.

Validated with the owner speaking freely at the MacBook (three questions in Portuguese and
English, all transcribed verbatim, language identified with confidence 0.99 or better) and
with clips played through the Mac speakers. Measured on the work Mac, mostly under heavy
load:

| Stage | Best (quiet Mac) | Typical (loaded Mac) |
|---|---|---|
| VAD tail (silence that ends the utterance) | 0.72 s | 0.72 s |
| Whisper large-v3-turbo, 2-4 s utterance (language id + decode) | 1.35 s | 2.9-3.1 s |
| Whisper warm-up at start (weights cached on disk) | 2.8 s | 5.8 s |
| Embedding + retrieval | 0.1-0.25 s | 0.7-2 s (cold) |
| LLM answer, 3 sentences | 1.5 s at 28.6 tok/s | 7-16 s at 2.6-8.6 tok/s; 31 s when the model reloads |
| ElevenLabs TTS | 1.1 s | 1.7 s |
| Ear to mouth (person stops talking -> robot starts) | 4.9 s | 10-21 s |

Findings:
- The HyperX QuadCast, the Mac's default input, delivered digital silence (-96 dBFS) in every
  test: it is muted on its touch sensor or its gain is at zero. The MacBook microphone works
  and is what `--device "MacBook Pro Microphone"` used. Set `AUDIO_INPUT_DEVICE` accordingly.
- Without echo cancellation the robot's own voice reaches the MacBook microphone at
  -21 to -31 dBFS peak, the same level as a person talking normally at the table (-28 to
  -31 dBFS). Energy-only barge-in therefore needs a raised voice next to the Mac; the
  owner's "para, para, para" at normal volume did not cross the bar. Whisper transcribes the
  robot's own voice perfectly, which suggests a cheap fix: compare each transcript with the
  last spoken answer and drop the echo, then lower `BARGE_IN_MARGIN_DB`. A reference-based
  echo canceller (the played clip is known) is the proper fix. See docs/SPEECH_PIPELINE.md.
- Whisper likes to hear "Rich" for "Reachy"; the addressee rules in Phase 2b must accept
  the common misspellings.
- Utterances captured while the robot thinks are currently discarded before it speaks; a
  queue with a "still relevant?" check belongs to the conversation state machine.
- The Portuguese answers translated "Action Phase" once ("fase de ação") despite the prompt;
  worth a few-shot example in the prompt.
- Background runs must be stopped with SIGTERM (a `&` job ignores SIGINT); `listen.py` now
  treats SIGTERM like Ctrl+C. Two instances left running answered each other for a minute.

## Done: Phase 2b, who is speaking and is it for me (2026-09-14)

- `tools/live-diarizer/`: diart 0.9.2 + pyannote.audio 3.4 + torch 2.8 + numpy 1.26 in its own
  `uv` project, reading the same Mac input device and publishing anonymous speaker turns
  over `ws://127.0.0.1:8765` (`hello` / `turn` / `step` messages, stream and wall-clock
  times). Three pins were needed on top of the planned stack: `matplotlib<3.9`
  (pyannote.core still imports `get_cmap`), `huggingface_hub<1.0` (pyannote 3 passes
  `use_auth_token`), and a `torch.load(weights_only=False)` shim for pyannote's pickled
  checkpoints. The token must be passed explicitly (`hf auth login` cache or `HF_TOKEN`).
- `src/speech/diarization.py`: optional WebSocket client with reconnection; the loop runs
  without it. `src/speech/speakers.py`: voiceprints with
  `pyannote/wespeaker-voxceleb-resnet34-LM` (256-d) in the main venv, one or more clips per
  name, cosine matching, persisted in `data/speakers/voiceprints.json`.
  `src/speech/addressee.py`: rule-based "is it for me?" plus the self-echo check and the
  per-utterance JSONL log (`data/game_logs/addressee.jsonl`), the dataset for the learned
  classifier.
- `listen.py` now enrols players (`--players "Ana,Bruno"`: the robot asks each one for a
  sentence), names every utterance, applies the rules and answers only when addressed
  (`--always-answer` restores the Phase 2a behaviour). 75 tests across the two projects (71 + 4).

Measured with synthetic voices (macOS `say`) through the Mac speakers and the MacBook mic:

| Signal | Value |
|---|---|
| diart first label after speech starts | 2.0 s (latency 1 s + step 0.5 s + CPU) |
| diart final turn after speech ends | 1.3-1.6 s |
| diart turn boundaries vs the played clip | within 0.1-0.3 s |
| wespeaker load / embedding of a 2-4 s clip | 0.4 s / 30-90 ms |
| same synthetic voice, enrolment vs later sentences | cosine 0.77-0.84 |
| different voice | cosine -0.02 |
| owner's real voice, five far-field captures | 0.36-0.59 between captures |

Findings:
- Whisper does not hear "Reachy" reliably: "Rich" from the owner, "Reaxi" and "E assim"
  from the synthetic Portuguese voice. The alias list covers the first two; a name the
  players can pronounce (or an audio keyword spotter) would be more robust than text.
- The three `say` clips (two voices) were clustered by diart as one speaker; real voices at
  the table are the test that matters (the owner's session is the next step).
- A rules question asked to the table without the robot's name is answered by default
  (`ANSWER_GAME_QUESTIONS=true`); the log records the decision either way so the policy can
  be learned from the annotated sessions.

## Done: fluid conversation on the physical robot (2026-09-14, second live session)

The owner's verdict on the first live run was "the delay is too long and it does not notice
when I interrupt it". Three changes fixed most of it, all measured live afterwards:

1. **Sentence streaming.** The LLM reply is split into sentences as it streams; each one is
   synthesized in a background thread while the previous one plays. The robot starts talking
   after the first sentence instead of after the whole answer (`answer_streaming` in
   `listen.py`, `think_sentences` in `answering.py`).
2. **Chat routing.** Utterances without game vocabulary skip retrieval and use a short
   conversation prompt with the last three exchanges (`needs_rules`, `conversation_messages`);
   rules questions get five capped passages instead of eight full ones (1000-1200 prompt
   tokens instead of ~2000). Prompt processing was the largest share of the wait on a busy Mac.
3. **Echo-aware barge-in** (`src/speech/barge_in.py`). Energy cannot separate a player from
   the robot's echo at the Mac microphone, so while the robot speaks any voice longer than
   600 ms is transcribed and compared with the sentences being spoken: echo continues, a
   player's words cut the clip and the LLM stream, and the interrupting utterance becomes
   the next question.

Also from that session: the exponential noise floor learned the robot's voice and cut the
owner's sentences in half (now a rolling 10 s minimum, frozen while the robot speaks);
low-confidence language ids decoded Portuguese as English nonsense (now the speaker's last
language wins below 0.5); a colleague's cough became a follow-up (follow-ups need the same
speaker; coughs and stretched noises are filtered); "Reachy" arrives as Rich, Reed, Ritch,
Reaxi (aliases); and with one enrolled player at the table everything said out loud is for
the robot (solo mode).

| Measured live, owner's voice, busy Mac | Before | After |
|---|---|---|
| Ear to mouth, rules question | 10-21 s | 5-6 s (12 s on a cold prompt) |
| Ear to mouth, chat | 9-13 s | 5 s (LLM first sentence 2 s) |
| Barge-in at normal voice level | never | 1.4 s after the player starts talking |
| False interruptions | - | 1 in 4 (a misheard fragment) |

Prompt tokens per rules answer: 1000-1200. The remaining wait is Whisper (1.4 s) plus the
LLM prefill at 2-6 tok/s on the loaded Mac; the same machine idle does 28 tok/s.

## Done: the conversation moved to an ElevenLabs agent (2026-09-14, third live session)

Even with streaming, the local chain answered in 5-13 s and the 3B-active model invented
rules ("Você só precisa repor se... exceder o limite máximo de dez"). The owner's verdict:
"horrível... pior comportamento desde julho". The three earlier projects in `../reachy` all
ended in the same place after benchmarking their local stacks: a cloud realtime agent for the
conversation, everything else local. `src/integration/talk.py` does that, written from
scratch:

- **ElevenLabs Agents** (`src/speech/eleven_agent.py`): the agent is created once and
  updated at every start from `agent_config()` (English persona prompt, golden rule "rules
  come from the tool", one native voice per language through `language_presets` plus the
  `language_detection` system tool, pcm 16 kHz both ways, 300 s turn timeout, 2 h sessions).
  Voices with live moderation (Mariana M) are refused by agents, so the agent uses Michelle
  for Portuguese (`ELEVEN_AGENT_VOICE_ID_PT_BR`) and Lara for English.
- **Local client tools**: `game_rules` and `game_knowledge` run our Qdrant retrieval and
  return numbered passages plus a note on how to use them (0.7 s per call). Tool results
  must be sent as text (the orchestrator rejects a JSON object).
- **Audio** (`src/speech/agent_audio.py`): Mac microphone in; the robot speaker is the USB
  output device "Reachy Mini Audio", written directly from a playback thread (the SDK's
  `push_audio_sample` is silent on macOS, like its `play_sound`). Head motion comes from
  `src/robot/sway.py`, a loudness-driven sway, because the daemon's wobbler cannot see USB audio.
- **Echo gate with local barge-in**: the Mac microphone hears the robot at a player's level,
  and the first run had the agent answering itself in a loop. Now microphone audio is held
  back while the speaker is busy (+0.3 s); the local VAD + Whisper check transcribes what is
  held and, when it is a player's words rather than the robot's own text, forwards the last
  2 s to the agent and cuts playback. A player talking over the robot gets a full answer.
- **Language switch by session restart**: the agent's `language_detection` tool was called
  with the wrong argument by Gemini and not at all by gpt-4.1-mini in live use, so a change
  of language (transcript text, or local Whisper on the audio) restarts the session with the
  other native voice, re-sends context and the question; 1.5 s, verified by voiceprint.
- **Shadow turn-taking**: the local ear (VAD, voiceprints, diarizer label) still names every
  utterance and the addressee rules still decide; the decision is logged with
  `shadow=True` next to what the agent did (`addressee.jsonl`), and every heard/said/tool
  event goes to `conversation.jsonl`. That is the research dataset; the agent itself answers
  whatever it hears, as the earlier projects did.

Measured with synthetic voices through the Mac speakers:

| Signal | Value |
|---|---|
| Question end -> filler line ("Let me check the Reference Guide") | 0.2 s |
| Question end -> answer with passages (tool 0.7 s inside) | 2.5 s |
| Language switch pt -> en with the American voice | same turn |
| Player talking over the robot -> playback cut | 1.4 s |
| False interruption from the robot's own echo | 1 (fixed: echo compared with the last 3 lines) |

`listen.py` (Whisper + Ollama) stays as the offline path and for the research components.

## Done: "stay quiet" decided locally, no cloud turn for table talk (2026-09-14)

The agent used to answer everything the microphone heard. Now the local ear decides first and
the agent only hears what is for the robot (`src/speech/gatekeeper.py`,
`src/speech/agent_audio.py`), so a sentence between players costs no turn and no tokens:

- **Addressee gate.** After the echo gate, every 250 ms frame is held in a time-stamped
  buffer instead of going to the agent. When the local VAD closes the utterance (600 ms of
  silence), the gatekeeper transcribes it with the local Whisper, names the voice, applies
  the rules and either releases the frames of that utterance in a burst (plus 1 s of silence
  so the agent's turn detector closes the turn) or drops them. The agent sees audio without
  timestamps, so a held-then-burst utterance is just a late one; the barge-in path already
  relied on that.
- **No extra wait when the name is heard.** While a voice is running (from 0.9 s, then every
  1.2 s of new voice) the same Whisper transcribes the audio so far; if the robot's name is in
  it, the gate opens at once and the rest of the sentence streams live. A barge-in that
  passed the echo check opens the gate the same way.
- **Language switch before the audio leaves.** When Whisper says the utterance is in the
  other language (confidence >= 0.8, three words or more), the audio is dropped, the session
  restarts with the native voice and the text is sent there: the rules search now runs once
  per question. In `--always-answer` mode the old session's tools also stop searching once it
  is being closed (`client_tools(active=...)`).
- **Rules added**: another enrolled player named as the vocative ("Bruno, o que você acha?")
  is not for the robot; a question after a leading "e"/"então"/"so"/"and" is still a question;
  the self-echo label applies only to utterances that overlapped the robot's playback (an
  English rules question 24 s after the answer was marked `self_echo` before); Whisper's
  looping keyword hallucination ("omen omen omen...") counts as no speech.
- **Whisper twice as fast for the gate**: `mlx_whisper.transcribe` runs the encoder once
  for language id and again to decode, on a 30 s padded window either way; the gate path
  (`Transcriber.transcribe(fast=True)`) encodes once and reuses the features for both.
- **Flags**: `--humans N` (people at the table; default the enrolled voices; 1 = solo,
  everything is for the robot and the gate is off), `--always-answer` (gate off, decisions
  logged as shadow). The diart sidecar is started automatically when
  `tools/live-diarizer/.venv` exists and nothing listens on :8765 (with a clean environment;
  our venv variables made its torch import from the wrong site-packages).
- `scripts/gate_replay.py` replays saved captures through the gate (no microphone, no agent),
  the way to test rule changes on real voices without spending agent minutes.
- 114 tests in the main project (+4 in the sidecar).

Measured on the owner's real-voice captures of this day (`scripts/gate_replay.py`, Mac loaded
with work apps):

| Signal | Value |
|---|---|
| Whisper on a 1-4 s utterance, `mlx_whisper.transcribe` (two encoder passes) | 1.1-1.8 s, median 1.26 s |
| Same with the single-pass gate path | 0.64-0.73 s, median 0.70 s (first call after load 1.2 s) |
| Gate decision after the VAD closes the utterance | 0.67-0.78 s |
| Extra delay before the agent gets a released utterance | about 1.3 s (0.6 s VAD tail + 0.7 s Whisper), minus what the agent saves by receiving the audio in a burst |
| Utterances that name the robot (>= 0.9 s of voice) | no extra delay once the partial transcript has the name |
| Decisions on the 20 captures (two humans assumed) | 4 released (name, rules questions), 3 switched to English, 11 discarded, 2 no speech; 0 wrong |

Not yet done: the live test with the owner speaking (one sentence with "Reachy", one table
sentence without it, one rules question), checking in `conversation.jsonl` that only the
table sentence produced no `heard`/`said` turn. Run it with `--humans 2` (one enrolled voice
otherwise means solo mode, where everything is for the robot).

## Done: camera preview page for placing the robot and the board (2026-09-14)

`uv run python -m src.vision.preview` serves http://127.0.0.1:8090: the robot camera as an
MJPEG stream (960 px wide, ~10 fps) with framing guides (thirds, centre cross, a dashed
"the board should fill this" rectangle with adjustable margin), pitch/yaw sliders that move
the head, a "look at the table" button (pitch 35) and full-resolution snapshots to
`data/captures/board/`. `--fake` runs it on synthetic frames. Frames come through the SDK's
local IPC camera backend (`src/vision/preview.py`); 3 tests cover the page, the stream, the
status, the head control and the no-frames warning.

Two things found while wiring it, both outside the code:

- **The daemon must be started from a terminal app, not from a Claude Code session.** The
  daemon running since 13:16 had been launched by a Claude Code Bash (its environment carried
  `CLAUDE_CODE_*` variables, no tty); macOS attributes the camera request to the Claude Code
  helper binary, which has no camera usage description, so the request is denied silently
  with no prompt and no entry in Privacy > Camera. The daemon log said `Device video access
  permission has been denied` (`avfvideosrc`), neither the IPC frames nor the WebRTC producer
  existed and `get_frame()` stayed `None`. Restarting it outside the Bash sandbox changed
  nothing; starting it from iTerm (which has the permission) fixed it at once: 1920x1080 at
  9.3 fps in the preview page, the first board snapshot in `data/captures/board/`. Motors and
  audio never needed the permission, which is why `talk.py` worked all day.
- **`localhost` was not the daemon.** A Docker container of another project published port
  8000 on IPv6 too; `localhost` resolves to `::1` first, so the SDK's "auto" and
  "localhost_only" modes got a 403 from the container. "auto" then falls back to
  `REACHY_HOST` (127.0.0.1) in network mode, which is why `talk.py` kept working; the preview
  connects that way explicitly with `media_backend="local"`. The owner is moving that
  container to port 8005.

## Done: the board in pieces, sweeps with the body (2026-09-14)

The board does not fit one view at a distance that keeps card text readable, so the robot
looks at it in pieces: `src/vision/capture.py` defines views (body yaw, head pitch, head yaw),
takes the sharpest of three frames per view (variance of the Laplacian) and writes each sweep
to `data/captures/board/sweep_<stamp>/` with a `views.json` index. The preview page got a
body slider, a "Sweep" button (list of body angles), a gallery of the saved sweeps, and a
digital zoom (1-4x, click to centre) to judge legibility; `--sweep 60,30,0,-30,-60` runs one
sweep from the command line. `Robot.look()` takes `body_yaw` (degrees, positive left).

Measured on the robot:

| Signal | Value |
|---|---|
| Base turn, head at pitch 35 | 60 deg in 1.3 s, 120 deg in 1.5 s, settles within 1.6 deg |
| Base turn, head at pitch 40-50 | unreliable beyond about 45 deg: stopped at 33-49 for 60, at -32..-54 for -60 |
| Five-view sweep (60, 30, 0, -30, -60) | 14 s, 2.8 s per view |
| The four Reserve cards | all visible in the "left60" view at pitch 35, small and slanted |
| Camera capture resolution | fixed by the daemon at 1920x1080 (the module offers 3840x2592, no daemon option) |

Consequences: keep the head near pitch 35 and let the base do the turning (the owner will
tilt the whole robot a few degrees instead of pitching the head further); the sweep waits
until the measured body angle is within 2 deg of the target before capturing, because
`goto_target` returns when its interpolation ends, not when the base has arrived. "Zoom" is
a crop of the 1080p frame; the far edge of the board stays soft at 3x. Reading text there
would need the camera at 4K, which means capturing outside the daemon (`--no-media` and our
own GStreamer pipeline), or a second, closer position for the robot.

## Done: the robot recognises the board by its own art (2026-09-14)

No markers: the printed map is the marker. `src/vision/board_map.py` keeps the SIFT features
of a top-down picture of the board (`BOARD_REFERENCE_IMAGE`, `data/imgs/World_Map.webp`,
1200x790, the real 2013 board) and registers every camera frame to it (Lowe ratio test,
RANSAC homography, accepted from 25 inliers). `src/vision/spaces.py` is the table of the 36
spaces (9 cities, 6 expedition sites, 21 numbered) in normalised map coordinates, checked by
drawing them over the picture. A registration maps any frame pixel to a space
(`space_at`), projects the space centres into the frame (`space_pixels`) and warps the frame
to the top-down map (`rectify`). The preview page draws the board outline and the space
names over the live video (`/board`, once a second, at half resolution) and serves the
top-down view (`/rectified.jpg`).

| Signal | Value |
|---|---|
| Centre view, full resolution | 188 ratio matches, 106 inliers, 0.2 s |
| Views turned 45 deg left / right | 43 / 38 inliers, 0.1 s |
| Live loop at half resolution | 73-81 inliers, 0.05 s per frame |
| The fan "total conversion" picture tried first | 6 inliers (different continents and cities): unusable |
| Space labels on the live frame | on the printed spaces for all 33 spaces in view |

The board picture is runtime data (git-ignored, copyrighted art); the owner drops it in
`data/imgs/`. Anything with the same art works after one rescale of the space table.

## Done: first piece found and named on the real board (2026-09-14)

`src/vision/detect.py`: at the start of a session the empty board is captured as a baseline
(the frame rectified to the map, median of five frames); afterwards "what is on the board
that is not the board" is the Lab difference between the current rectified frame and that
baseline, cleaned by morphology, split into blobs and named by the space under the blob's
*base* (standees rise towards the far edge in perspective; the centroid put the Istanbul
marker between spaces). Each piece comes with a full-resolution crop of the frame (the zoom)
saved under `data/captures/board/pieces/`. The preview page has the two buttons and draws
the boxes.

Live test with the owner moving an investigator standee: Istanbul -> "something at
Istanbul"; Buenos Aires -> found (dark standee on a dark green circle, strength 53, the
weakest case); Arkham -> "something at Arkham" twice; empty board -> nothing, three times.
Tuned on the real board: threshold 45 (empty-board maximum 44-78 before the median
baseline), minimum 300 px and 12 px on each side (a sliver along the frame edge), top 9 %
of the map ignored (Doom track and hands beyond the far edge), bottom 6 % (Reserve).
Sea spaces 12 and 13 were swapped in the first space table; fixed by the owner's reading.

Still open on this path: naming *what* the piece is (gallery of the real pieces, next),
several pieces close together (one blob), and pieces on the Reserve or the far right edge
(a sweep view instead of the centre view).

## Decisions taken

| Topic | Decision | Where |
|---|---|---|
| Reasoning model | Qwen 3.6 35B-A3B (`qwen3.6:35b-mlx`), not Qwen 2.5 72B | docs/MODEL_SIZING.md |
| Embeddings | bge-m3 through Ollama, 1024-d | src/rag/collections.py |
| Vector store | Qdrant in the existing Docker container; collections `bg_rules`, `bg_knowledge`, `bg_sessions` | docs/DESIGN_DOCUMENT.md |
| Speech input | Mac microphone; mlx-whisper; diart + pyannote 3 live in a sidecar; pyannote 4 + WhisperX offline | docs/SPEECH_PIPELINE.md |
| Speech output | ElevenLabs by default, local TTS optional | src/config.py |
| Conversation | ElevenLabs agent (cloud) + local client tools; local Whisper + Ollama kept as fallback | docs/SPEECH_PIPELINE.md |
| Agent LLM | `gpt-4.1-mini` inside ElevenLabs (owner's decision, 2026-09-14): follows the prompt better than the Gemini default; billed through the ElevenLabs account, no OpenAI key | src/config.py |
| Turn-taking | the local ear decides speak/stay-quiet before any audio reaches the agent (rules on the local Whisper transcript; a local LLM classifier only if the rules prove insufficient) | src/speech/gatekeeper.py |
| Game setup | verbal briefing + knowledge base, no card OCR | docs/SETUP_PROTOCOL.md |
| Vision | YOLO-World + image-embedding gallery, SAM not used | docs/DESIGN_DOCUMENT.md |
| Prior project | read for lessons only; no code copied | CLAUDE.md |

## Next

### Phase 1: vision and board state (weeks 1-2)

Blocked on the owner deciding where the robot sits at the table.

- [ ] Grant camera permission to the terminal/daemon (macOS Privacy settings).
- [ ] Calibration session with the checklist in docs/PHYSICAL_SETUP.md; reference photos of every token type.
- [ ] `src/vision/capture.py`: gaze to table pose, sharpest-of-three capture, save to `data/captures/`.
- [ ] `src/vision/detect.py`: YOLO-World with the game's prompt list; gallery matching.
- [x] `src/vision/board_map.py`: SIFT registration to the board picture -> space assignment.
- [ ] `src/strategy/state.py`: pydantic game state + patch application (needed by everything after).
- [ ] Angle and lighting robustness test with recorded frames.

### Immediate next steps (decided 2026-09-14 with the owner)

1. ~~Local "stay quiet" gate, no LLM tokens~~ done (section above); pending its live test
   with the owner speaking, and a local LLM classifier only if the rules prove insufficient
   in real sessions.
2. **Phase 1, board vision**: the owner places the robot with the camera preview page
   done up to the board map: preview page, sweeps, `board_map.py` and the space table. Next:
   `detect.py` (YOLO-World candidates + a gallery of the real pieces photographed by this
   camera, built in a calibration session with the owner) and the state patches.
3. A recorded 2-3 player session (the annotated dataset) waits until players are available.
4. ~~Small fixes: the rules tool ran twice across a language switch; the shadow label marked
   an English question as echo; start the diart sidecar by default when present~~ done.

### Phase 2: speech and turn-taking (weeks 2-3)

- [x] Mac microphone capture (`sounddevice`) with device selection from `AUDIO_INPUT_DEVICE`.
- [x] mlx-whisper transcription per utterance, Portuguese and English, language id per utterance.
- [x] Listening loop with interruptible speech (`src/integration/listen.py`).
- [x] Self-echo rejection by transcript match (the robot ignores its own answers).
- [x] Barge-in at normal voice level (transcription against the spoken text; an acoustic canceller would cut its 1.4 s reaction).
- [x] `tools/live-diarizer/`: diart sidecar publishing speaker turns over WebSocket.
- [x] Speaker enrolment and voiceprint matching.
- [x] Rule-based addressee classifier + speak/silence log.
- [x] The classifier gates the audio to the agent (no cloud turn unless addressed).
- [ ] Record and annotate a real 2-3 player session; measure the rules against the annotation.
- [ ] Map diart labels to enrolled names over time (label -> name votes) for overlap cases.

### Phase 3: strategic reasoning (weeks 3-4)

- [ ] Ollama client with JSON-schema outputs; prompt templates in English.
- [ ] Candidate generation, legality checks against `bg_rules` and the board map, evaluation, justification.
- [ ] Rules Q&A tool returning section and page.

### Phase 4: end-to-end (weeks 4-5)

- [ ] Conversation state machine, gaze states, TTS through the robot.
- [ ] Setup protocol implemented; first full 2-player game logged.
- [ ] Human feedback captured into the trajectory.

### Phase 5: world models (weeks 5-6)

- [ ] Trajectory dataset builder; baseline next-state model; Dreamer-style model.
- [ ] Rollouts in the decision pipeline; evaluation on held-out rounds.

### Phase 6: research (week 6+)

- [ ] Multi-player sessions, annotation study of turn-taking, paper draft.

## Open items

- Calibration numbers (riser height, head pitch, exposure) are placeholders until measured.
- Latency figures in docs/SPEECH_PIPELINE.md come from published benchmarks, not this Mac.
- The two unverified facts in docs/GAME_REFERENCE.md (Doom track maximum, token counts).
- Repairing the robot microphone cable would add direction-of-arrival to addressee detection.
