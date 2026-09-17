# Project Status

Updated 2026-09-16 (Phase 3: the robot plays its own turn).

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

## Done: the board scan with a closer look (2026-09-14, evening)

At the start of a session the robot scans the board in three views (centre, body +45,
body -45): a *baseline sweep* on the empty board learns one baseline per view; a *scan*
detects the pieces in every view, merges the sightings by map position and returns one
board state (piece -> space, which views saw it) plus the spaces no view covers (none, with
this framing). Ambiguous sightings (between spaces, base off-centre in its space, frame edge,
or difference strength below 60, the owner's rule) get a *closer look*: the body turns to
put the piece at the frame centre and the verdict is taken there, against the baseline of
the nearest scan view. Buttons "Baseline sweep (empty board)" and "Scan the board" in the
preview; `POST /scan`, `GET /scan_result`.

Measured and learned on the robot:

| Signal | Value |
|---|---|
| Three-view scan without closer looks | 14 s |
| Each closer look (turn, settle, three frames, register) | 4-5 s |
| Body yaw: frame pixels per degree at 1080p | 17.5 (a point moves right when the body turns left) |
| Head pitch: frame pixels per degree | 2.7 (the head tilts around the camera; pitch cannot centre a piece) |
| Investigator on San Francisco, first pass from the left view | "1" or "near 2" (standee smeared by perspective, board corner) |
| Same, after the closer look with the nearest view's baseline | "San Francisco" |
| Investigator on Sydney (right view only) | "Sydney", confirmed |
| Card lying in the Reserve | ignored (the Reserve and the legend are masked) |

Two dead ends: the SDK's `look_at_image` sent the head to the ceiling (it reads the camera
calibration at the sensor's full size and resets the body yaw); comparing a closer-look frame
with the *centre* baseline at a board corner gave a false blob offset (the corner is in the
lens distortion zone of the centre view), hence the nearest-view baseline. The piece's base
is the extreme of its blob towards the camera, computed per view.

## Done: Reserve, dice, and the morning after (2026-09-15)

- **Reserve** (`src/vision/reserve.py`): the four card slots (measured on the board picture)
  are judged on the upper part of each slot, where a card's picture differs strongly from the
  cream slot (cards 0.6-0.9 of the window above 20, empty slots 0.00). Read from the *left*
  scan view: a view turned further (60-75) shows too much room to register, and the base
  stalls at 64 with the head at pitch 35.
- **Dice** (`src/vision/dice.py`): a die is the black, roughly square blob nearest the centre
  of a piece's crop (25-140 px). Seen from the head at 35 degrees above the table the top
  face is the compressed upper part of the outline (about 45 % of its height, pips wider than
  tall) and the front face the larger lower part with rounder pips; the first reader counted
  the front face. Now only the spots above 55 % of the body height count (in the 40-55 %
  band only wide, full-size ones), a spot 1.6 x the median pip area counts as two touching
  pips, and the value is voted across the views with the closer look weighing twice. Live
  with the owner's values: 5 read as 5 (confidence 0.8) and 4 as 4 (0.33, flagged unsure);
  per view the centre and the closer look were right every time, the side views half the time.
- **Daylight**: the evening baseline was useless in the morning (board brightness 138 -> 86,
  median difference 28 against 2 the night before) and the scan invented pieces. Now every
  scan view is checked against its baseline (at least 40 inliers, median difference at most
  15) and skipped, or the scan refused, otherwise; the difference threshold adapts to the
  frame's own noise (3 x the 95th percentile, clamped to 25-45: 42 in the morning, 45 at
  night); pieces are found by hysteresis (a core above the threshold, the footprint down to
  60 % of it), which recovers the standees that differ from the board by only 33-40 in flat
  daylight; a standing piece's base is its lowest point in the frame, pushed a little lower
  because its bottom vanishes against dark art; closer looks are capped at four per scan.
- Live this morning after a new baseline sweep: two standees (Tunguska, Buenos Aires), two
  dice, one card in the Reserve, nothing false.

Rules that fell: comparing with the *centre* baseline at a corner; a map-space "towards the
camera" direction (frame pixels above the horizon flip it); a fourth scan view for the
Reserve; locally normalised differences (noise as high as the signal); the SDK's
`look_at_image`.

## Done: what just moved, and what a VLM is worth here (2026-09-16)

The scan compares the board with a baseline of the *empty* board, and that baseline ages: the
evening one was useless the next morning. `src/vision/motion.py` answers a narrower question
that does not age — "what changed just now" — by comparing consecutive frames, 0.1 s apart,
where the light is the same light.

Measured on the live stream at 960x540 with the board untouched:

| Frames apart | Median difference | 95th percentile | Max |
|---|---|---|---|
| 1 (0.10 s) | 2.24 | 4.58 | 14.7 |
| 9 (0.94 s) | 2.24 | 4.58 | 14.9 |

A piece is worth 33-60 by the same measure: the signal is about 7x the noise, and the noise
does not grow with the interval (it is the sensor, not drifting light).

`MotionWatcher` keeps an *anchor* frame while the board is quiet. A hand reaching in raises the
activity far above anything a piece does, which is the cue; when the board is still again for
five frames the anchor is compared with the new frame by `find_pieces`, so the whole detector
applies (adaptive threshold, hysteresis, masked Reserve and legend, naming by the space under
the base). The occlusion is the trigger, not an obstacle.

The rule that fell on the way: a frame difference is **symmetric**, so using either frame as
the other's baseline returns the same blobs and cannot say which one holds the piece — the
first version reported `['Rome', 'London']` as having both departed. `stands_out` breaks the
tie by asking how far a blob's colour sits from the board immediately around it in each frame:
a piece stands out against the board, a vacated space does not.

This also means identity can come from **tracking instead of recognition**: a piece followed
across a move carries its own name, and the name itself is given once, by voice, through the
conversation that already exists.

### The VLM, measured rather than assumed

`qwen3.6:35b-mlx` turns out to have a working vision tower (it read `ZXQ-4718-KTMR` off a
synthetic image; a blind control answered nonsense). `qwen3-vl:8b` was pulled and compared with
the ResNet gallery on the 13 labelled crops, leave-one-out:

| | correct |
|---|---|
| ResNet gallery (threshold 0.72) | 2/13 |
| Qwen3-VL 8B, piece name | 0/13 |
| Qwen3-VL 8B, kind only | 4/13 (the four standees) |

The VLM does not fail quietly: it produced "Edward Carnby" (a character from another game),
"Bernard Aizen", "Ritual Gate", "Knyfe". On the full board photo it placed the investigator on
The Amazon (it was on Buenos Aires) and invented six gates out of the red-bordered cities
*printed* on the board, three of them Arkham Horror spaces that do not exist in this game.

The conclusion is not "VLM or ResNet" but that **the pixels are the bottleneck**: the crops run
116x57 to 277x255 and are out of focus. The single large, sharp crop in the set (460x274) is the
only one read correctly, and it was read by the 35B. No recogniser fixes that.

Two facts found the hard way, both worth keeping:

- **The gallery is contaminated**: `Card/20260915_125442_001.jpg` and
  `asset:Bull Whip/20260915_125411_001.jpg` are byte-identical with contradicting labels, so the
  ResNet always ties at 1.000 between those two labels.
- **Ollama pre-allocates the model's native context**: `qwen3-vl:8b` is 6.1 GB on disk and took
  **45 GB of memory** because it reserved its 256K window. `num_ctx` must be set explicitly on
  this Mac; `num_predict` must *not* be, because the model spends the budget on internal
  reasoning and returns an empty string.
- **MLX is the cheaper road on this machine**, as the owner expected: the same model as
  `lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit` through `mlx-vlm` runs at **1.6 s per crop
  and 6.3 GB peak**, against 16-25 s and 10-45 GB through Ollama. `mlx-vlm` is installed in the
  venv but is not in `pyproject.toml` yet: it upgrades `transformers` to 5.x, and the 160 tests
  pass with it.

## Done: the robot sees a move as it happens (2026-09-16, live with the owner)

`src/vision/motion.py` watches one fixed view and reports moves as they happen; the watcher runs
inside the preview, so nothing competes for the camera, and the page carries a **BOARD LOG**
panel fed by `GET /moves`. Live on the real board, six moves in a row, no false verdict:

| Time | Reported |
|---|---|
| 10:45:36 | a piece moved from Rome to Istanbul |
| 10:46:26 | a piece moved from Istanbul to Tunguska |
| 10:46:55 | a piece moved from The Himalayas to The Heart of Africa |
| 10:47:06 | a piece moved from The Heart of Africa to The Amazon |

Four of six came out as one clean sentence; two were the same move split into two verdicts
(2.8 s and 3.6 s of disturbance) with one spurious blob, which is the settling rule being
generous rather than a detection failure.

Measured, not guessed (960x540 live stream):

| Signal | Value |
|---|---|
| Frame-to-frame noise, board untouched | median 2.24, p95 4.58, max 14.7 (a piece is 33-60) |
| Activity, board untouched | exactly 0.0000 (no pixel clears 30 grey levels) |
| Activity, a hand moving pieces | 0.006 to 0.039 |
| Activity, a body sweep | 0.036 to 0.100 (overlaps the hand: magnitude cannot tell them apart) |
| Registration accuracy, rectified vs reference art | 1.4 px median over 27 tiles |
| Registration jitter, board corners | up to **145.6 px** (extrapolated far outside the frame) |
| Registration jitter, visible space centres | up to **6.5 px** |

Four rules fell on the way, each after it broke something live:

- **The trigger was guessed and never fired.** `DISTURBED_FRACTION` was 0.04 from an estimate
  that "an arm covers an order of magnitude more of the frame than a piece"; a real move topped
  out at 0.0386. The floor is what matters and the floor is zero, so it sits at 0.005 now.
- **A frame difference is symmetric**, so neither direction can say which frame holds the piece.
  Local contrast (`stands_out`) gets it wrong over the board's own art: a piece landing on the
  dark "The Heart of Africa" card was called a departure. `occupancy` asks the absolute question
  against the empty-board baseline instead. The baseline only *arbitrates* and never triggers, so
  a stale one degrades direction but cannot make the watcher fire.
- **The stale-view guard measured pure noise.** Comparing board *corners* between registrations
  refused every real move ("the view moved by 112 px") because those corners sit outside the
  frame; it compares the visible space centres now.
- **Half a second of stillness is not enough.** A hand pauses mid-move, and those pauses read as
  quiet; settling takes a full second.

The preview page was also rebuilt to the owner's teleop-console design. One bug came with it:
the image was stretched with `width:100%` + `object-fit:contain`, which letterboxes it on a wide
window while the overlay canvas kept covering the whole element, pushing every space sideways in
proportion to its distance from the centre. The board looked misregistered when the registration
was good to 1.4 px. The wrapper's width is capped by the height left over now.

## Done: the robot knows the game it is in (2026-09-16, afternoon)

The vision could say "a piece moved from Rome to Istanbul" and nothing could say *which* piece
or what the game around it was. Three layers closed that.

**`src/strategy/state.py`** - `BoardState`, the pieces the camera can follow. A scan seeds it,
every `motion.Move` updates it, and identity comes from **continuity, not recognition**:
departures pair with arrivals by distance, so a piece that leaves Rome and arrives at Istanbul
keeps whatever it was called. A gallery match is stored as `guess`, never as `name` - it
measured 2 of 13, and storing it as a name once made `claim_pieces` skip the piece it had
mislabelled, so a 2/13 guess was outranking a fact.

**`src/strategy/game.py` + `reference.py`** - `GameState` adds everything the camera can never
see: doom, Mystery, Omen, health, sanity, cards, round, phase, and who controls which
investigator. `reference.py` holds the 12 sheets and 4 Ancient Ones as checkable data, with
lookup tolerant of how people speak ("Lily", "Chen", "lily chan" -> suggestions) and expansion
names refused rather than guessed.

**`src/strategy/setup.py`** - the spoken briefing becomes that state. The local model extracts
the fields; `reference.py` decides whether they are real, because the model is good at "who did
they mean by 'you'" and unreliable at "is Nyarlathotep in the base game". A briefing can arrive
in pieces, and what is missing comes back as one question at a time.

The two halves meet at `claim_pieces`: the person says "Lily Chen starts in Shanghai", the scan
finds an unnamed piece at Shanghai, and the piece becomes Lily Chen. **No image recognition is
on the critical path.** Two pieces on one space is a question, not a guess.

Live on the owner's real two-player setup (Azathoth, Lily Chen with the robot at Shanghai,
Jacqueline Fine at space 5, four cards in the reserve, a Portal and a Monster at Rome):

| Step | Result |
|---|---|
| Narration -> state | every field right, both languages, 26 s cold / 2.7 s warm |
| Scan | Lily at Shanghai, Portal and Monster at Rome, 4 reserve slots |
| `claim_pieces` | the Shanghai piece became Lily Chen from the briefing alone |
| Jacqueline | **not** claimed - she was detected at space 4, not 5, so it asked instead |

Two false positives (glare along the printed banners at San Francisco and Arkham) exposed the
filter that was missing: every real piece was either seen from two views or confirmed by a
closer look, and those two were the only sightings that were neither. `detect.doubtful()` also
refuses a blob more than 2.5x longer than wide. Doubtful sightings are still *mentioned* but
kept out of the state: a token that is not there is worse than a gap.

**The ear knows the turn is its own.** "É a vez da Lily Chen" now reaches the robot at the same
confidence as calling its name, because it knows which investigator it plays; a turn call
naming somebody else's investigator stays refused. The `Gatekeeper` takes it as a callable,
since setup happens while the ear is already listening.

### The cards, and why the camera is not involved

A list of 391 cards was researched elsewhere and filtered on the way in: **76 are the 2013 box**
and 315 were expansion content, refused with the reason logged. They live versioned at
`src/rag/cards/eldritch_base_game.json` and carry the **printed wording** - "Bull Whip: +1
Strength during Combat Encounters, reroll 1 die" is actionable, a paraphrase is not. 65 are
assets, 11 are Conditions under their own kind.

The camera is deliberately not asked to read them. It can see *that* a reserve slot holds a
card, never *which* one from 250 blurred pixels, and never a card in a hand at all. The name
comes from the person, as investigator names do; what the card *does* comes from here.

`retrieve.card()` looks a card up by its **indexed name, not by embedding**: asked "what does
Bull Whip do?", the vector search answered Vatican Missionary at 0.32, because a card's name
says almost nothing about its meaning while its effect text dominates the embedding.

Community strategy is also in `bg_knowledge` under `kind: "strategy"` - 11 entries from a
Ludopedia dossier, in English, each carrying its source and phrased as advice, with a test that
fails if one starts reading like a rule.

## Done: the robot chooses a turn and says why (2026-09-16, evening)

Everything before this was perception and memory. The robot could see the board, follow a move
and hold the whole game state, and it had no turn of its own. Four pieces closed that, and the
first of them was a hole nobody had noticed.

**`src/strategy/map_graph.py`** - the paths. `spaces.py` knew *where* the 36 spaces are and
nothing about which touches which, so Travel, tickets and Prepare for Travel were all
impossible. The table is 56 paths of the three printed kinds (Train, Ship, Uncharted), with the
three that cross the edge of the map marked as wrapping: 1/19 the Bering Strait, 2/Tokyo the
northern crossing, 3/Sydney the southern one. `travel_options()` is the printed rule and nothing
more - one free move along any path, then one extra move per ticket spent, a ticket only along
its own kind, and an Uncharted path walked but never bought.

Every path was traced on `data/imgs/World_Map.webp` at 4-7x, region by region; nothing came from
memory, and colour segmentation was tried first and abandoned (in that photo a Train dot is hue
12 and an Uncharted dot hue 17). `scripts/draw_map_graph.py` draws the table back over the
picture, which is how the owner checks it; `VERIFIED` stays False until he has, and while it is
False `decide` refuses any turn that moves a piece.

Two things the tracing found: space 8 was typed `city` in `spaces.py` and the board prints a
blue Sea token (`docs/GAME_REFERENCE.md` had it right all along), which would have offered
Acquire Assets and Prepare for Travel where the rules forbid them; and three spaces are
dead ends with a single path each (9, 13, 21), which is worth the owner's eye.

**`src/strategy/moves.py`** - every legal turn. The unit is the whole turn, up to two distinct
actions, the second judged from where the first one leaves the investigator: "Travel to Tokyo,
then Acquire Assets at Tokyo" is an idea a one-action-at-a-time generator cannot have. The six
actions and their conditions live in a table, because the reference card cannot change while the
box is the 2013 box; everything that does vary is read rather than written, and a Component
Action is simply a card whose printed text begins "Action:", straight from
`src/rag/cards/eldritch_base_game.json`, offline.

What the robot cannot know stays a question: an unnamed piece on the space is "is any of them a
Monster?", not a Monster, and the Reserve cards are known by what they do and not by the value
printed on them. Only a piece somebody named as a Monster removes Rest and Acquire Assets.
`refusals()` explains what is missing in the rule's own words, because a silent omission looks
like a mistake to the table.

**`src/strategy/decide.py`** - the choice. An ordinary turn has 40 to 200 legal plans. A
readable score (every term in `Weights`) prunes to eight, keeping the best plan of *every* first
action so no whole idea is lost; the model then picks one number from that shortlist and writes
the sentence. Anything that is not a number from the list is asked again once, and a model that
does not answer at all leaves the best-scored plan standing with the score's own words: the
robot always plays. Every plan that was cut stays in `Decision.considered`.

**`src/strategy/plan.py`** - the plan for the whole game, written after the setup and revised
when something actually happened, so the turns add up to something. It is labelled everywhere as
the robot's intention, never a rule. The eleven community strategy entries are fetched here
rather than per turn: advice about which investigator suits which Ancient One is advice about a
game, not about a move.

`uv run python -m src.strategy.turn --demo --plan` runs the whole thing on the screen, no
microphone and no ElevenLabs.

Measured on the work Mac, model warm, the owner's two-player setup:

| Signal | Value |
|---|---|
| Candidates and plans at Shanghai, no tickets | 11 actions, 40 turn plans (57 when hurt) |
| Turn decision, `think=False` | 1.9-8.5 s |
| Turn decision, `think=True` | 114 s, and it was the answer that invented a rule |
| Plan for the game (once, with advice) | 15 s |
| First call of the session (model load) | +40 s |

`think` is therefore off by default. Three faults the live runs exposed, all fixed: "doom at 15
of 15" was read as "time is nearly up" (the brief now says doom moves towards 0 and what that
means); a Gate rendered as "Rome: Rome" (now "Rome has a Gate and the Monster Cultist"); and the
model explained a choice by the Clues it would spend, so the printed rule of every offered action
now travels with the shortlist, including that a Clue rerolls a die and buys nothing.

What is still wrong at this size of model: it translates a term now and then ("Mistérios" for
Mystery) even with an explicit example against it, and once said "Roma" and "Cultista". The
decision itself has been sound in every run so far.

## Done: the turn is spoken, and the agent never hears it (2026-09-16, night)

"É a vez da Lily Chen" already reached the robot at full confidence, because the addressee rules
know which investigator it plays (`reason="my_turn"`). Now something happens when it does.

`src/integration/turn_taking.py` takes the turn: the gate drops the audio instead of releasing it
to the ElevenLabs agent, `decide()` chooses the move locally, and the robot says the reason that
came out of it - word for word, through its own TTS, on the same output stream the agent uses, so
the echo gate, the barge-in check and the head sway treat it as the same voice.

**The agent is never asked to say a move.** Not only for the minutes, though a turn now costs
none: the sentence the decision produced is already finished, in the table's language, with the
game terms in English, and handing it to a conversational model to "say" is handing it a chance
to say something the robot did not decide.

Only the model's own question is spoken with it (`Decision.ask`, written in the table's
language). The other unknowns are the generator's English notes for the log, and a Portuguese
table should not be read an English sentence. Two lines are fixed text in both languages -
"I cannot act this turn" and "I do not know which investigator I am playing" - because there is
no decision behind them to explain.

`scripts/speak_turn.py` rehearses the whole thing without the agent and without the microphone:
the local model decides, the local TTS speaks, the audio goes to the robot. A rehearsal costs a
few hundred characters of TTS instead of a session, which is why the live test below could be run
as often as it needed to be.

Measured on the robot this evening (`--demo`, Portuguese):

| Signal | Value |
|---|---|
| Turn call -> decision (model warm) | 2.8 s |
| Decision -> 17.5 s of speech synthesized | 1.9 s |
| Turn call -> the robot starts talking | about 5 s |
| Agent minutes spent on a turn | none |

Two faults found on the way, both outside this feature: `AUDIO_INPUT_DEVICE` in `.env` held the
example's own comment (`# empty = system default input...`), so the microphone went looking for a
device by that name - settings now drop a trailing `# comment` the way a .env is meant to read;
and `RobotAudioInterface` could not open without a microphone, which is all a rehearsal needs
(`input_device=None` now means "speak, do not listen").

Still to do live, with the owner at the table: say it out loud to the robot and check in
`conversation.jsonl` that the turn produced no agent turn at all.

## Done: a person can tell the robot something, not only ask it (2026-09-16, first live turn)

The first live session with the robot playing ended with the owner's verdict: "ele descartou
muita instrução que era relevante... ele não tá sabendo reconhecer o próprio nome". The log said
otherwise about the name - "Ritch", "Ei Rich", "Allie, Rich", "Reach", "Gate reach" were all
heard as the name, and "É a vez da Lily Chang" was taken as its own turn - but he was right about
the instructions, and the reason was in the rules rather than in the ears.

The gate answered questions and name-calls and dropped **statements**. A person explaining a
board does not ask anything:

    17:22:18  Reach.                                                    -> released (name)
    17:22:20  Eu comprei uma carta.                                     -> discarded
    17:22:23  Chamada de Deep Ones Attack.                              -> discarded
    17:22:24  Quando ela entra em jogo...                               -> released (a question)
    17:22:29  Vou botar um Eldritch Token no espaço 18 e outro no 8.    -> discarded
    17:22:30  Eu coloquei lá, ok?                                       -> discarded

So the speaker now **keeps the floor**: once somebody has said something the robot took as its
own, their next sentences are for it too, statements included, until 30 seconds pass in silence
or they name another player. The voiceprint identifies the voice; with nobody enrolled the
diarizer's anonymous label still keeps two speakers apart, and with neither the floor stays shut
rather than opening for the table.

`scripts/rules_replay.py` is how the window was chosen: it replays the addressee rules over the
transcripts already in `addressee.jsonl`, instantly and with no audio, and prints the lines a
rule change flips. On that session: 20 s recovered 7 of the 9 dropped instructions, 25 s
recovered 8, **30 s recovered all 9, and 40 s added nothing**. Nothing that should have been
ignored was released at any of those windows.

Other things that session showed, not yet addressed: Whisper writes "Lily Chang", "Lily Shane",
"AsaTot", "Azatov", "Eldritch Tolkien", "Old Jonah" (the setup extraction tolerates it, the
knowledge lookups may not); and the owner reading the Reserve out loud right after the robot said
those same card names was marked `self_echo`, which is the one case where the echo check and a
person quoting the robot are genuinely the same signal.

## Done: the robot remembers the game between runs (2026-09-16, night)

The owner asked the obvious question after the first live turn: "se eu fechar e rodar
novamente, ele vai manter o que já conversamos?" The answer was no, and worse than no -
`src/strategy/setup.py` could turn a spoken briefing into a `GameState` and **nothing in the
running conversation ever called it**. Everything he narrated lived in the ElevenLabs session
and died with it; the only game the robot could play was one loaded from a file by hand.

`src/integration/game_session.py` is the memory. While the robot does not yet know what it needs
to play - the Ancient One, or which investigator is its own - a sentence that is not a question
goes to the local extractor instead of to the agent, the reference checks the names, the robot
says back what it wrote down and asks for the next missing thing, and the state is written to
`data/game_logs/game_state.json` at once. Opening again reads it back and the robot says what it
remembers, which is also how a stale game announces itself.

Live, with the sentences exactly as Whisper wrote them in that session:

    "Nós vamos jogar contra o ancião Azatov."
      -> Não conheço o Ancient One "Azatov". Era Azathoth? Qual é?
    "Ok, eu vou controlar Jacqueline Fine e tu vai controlar a Lily Shane."
      -> Anotado. Com você: Jacqueline Fine (Alessandro). Não conheço o investigador
         "Lily Shane". Era Lily Chen ou Lola Hayes ou Charlie Kane? Qual é?
    "É a Lily Chen mesmo."
      -> Anotado. Eu jogo com Lily Chen, começando em Shanghai. [...]
    (closed, reopened)
      -> Continuando a partida. Eu jogo com Lily Chen, começando em Shanghai. Com você:
         Jacqueline Fine (Alessandro). O Mystery é Find the Way In. Qual Ancient One a gente
         vai enfrentar?

8.7 s for the first reading (cold), 3-4 s afterwards. Nothing is guessed: a name the reference
refuses is asked about with the names it might have been (`closest_ancient_ones` is new, the
investigator one already existed), and the question is built from data so it is asked in the
language of the table rather than read out of an English sentence.

Three faults this exposed, all of them older:

- **Every Portuguese sentence starting with "o" was a question.** The interrogative list held
  "o que" split on whitespace, which left a bare "o": "o ancião é Azathoth" and "O Mystery é
  esse" were both classified as questions and answered as such. Interrogatives are now words
  plus pairs ("o que", "por que", "how many").
- **The speaker's own name never reached the extractor**, so "eu vou controlar a Jacqueline"
  ended up with a controller called "me". The gate knows whose voice it is and now passes it.
- The gate's hook was specific to taking a turn; it is now "what the robot handled itself",
  which is what let the setup ear sit beside the turn taker without a second path through the
  audio.

## Done: the briefing reaches the robot even when nobody says its name (2026-09-16, late)

The next live session put the same complaint one step further on. The game file already had the
Ancient One and the investigator, so the setup ear was closed - and the owner was telling the
robot the Mystery:

    17:54:41  O mistério atual diz o seguinte, ele é o The Deepest One Attack.  -> discarded
    17:54:43  Conhece ele?                                                      -> discarded

Two things were wrong, both of them ours:

- **The ear was open for the wrong condition.** It waited on `game.ready`, which the Ancient One
  and its own investigator already satisfy. It now waits on what the robot can only be *told* -
  the Ancient One, its investigator, the Mystery, and any name the reference refused - and not on
  `game.missing()`, which includes "which piece on the board is Lily Chen" and would hold the ear
  open all game for something the camera answers.
- **A statement never reached the ear at all.** The ear only sees what the gate already judged
  to be for the robot, and a briefing has no name, no question and no floor behind it. So the
  rules gained one: while the robot is missing something it can only be told, a statement that
  carries the words of a briefing is for it (`setup_talk`).

That rule is a guess, so it costs nothing when it is wrong: the extraction runs **before** the
audio is dropped, and a sentence that turns out not to be a briefing goes on to the agent as if
nothing had happened. "Não, tu não entendeu, esse foi o mistério que eu comprei" has the word in
it and is not a briefing; the robot stays quiet and the agent answers it.

Live, with the state from that session and the sentence as Whisper wrote it:

    "O mistério atual diz o seguinte, ele é o The Deepest One Attack."
      -> setup in 3.6 s: "Anotado. O Mystery é The Deepest One Attack."

Only what changed is read back now. Confirming with the whole state was twenty seconds of speech
for one new fact, and a table stops listening to that.

Replayed over both live sessions (`scripts/rules_replay.py`): 58 utterances, 14 reached the robot
then, 29 would now, and every one of the 15 newly heard is an instruction the owner gave it.

## Done: the gate can be switched off, and then nothing is ever dropped (2026-09-16, late)

The owner's call after three sessions of tuning rules: "vamos remover os discarded por completo
por enquanto". Fair - a rule that drops a sentence is only worth having once the rest works, and
until then every drop is a thing he has to notice and complain about.

``ADDRESSEE_GATE=false`` (already set in his ``.env``) or ``--no-gate`` on the command line: the
robot answers everything it hears. What does **not** change with the gate off, and is the reason
this is not simply "pass the audio through":

- **The utterance is still held** for the moment it takes to transcribe it. Without that the
  agent has already heard a turn call by the time the robot decides to take it, and both of them
  answer the same sentence. The delay is the one already measured, about 1.3 s.
- **The robot's own echo is still dropped.** Answering itself is not answering the table, and
  an earlier session had exactly that loop.
- **The turn and the briefing are still the robot's own**: a turn call and a setup narration are
  handled locally and never reach the agent.
- **The rules still run and are still logged.** ``addressee.jsonl`` is the dataset the learned
  classifier will be trained on, and a session with the gate off has to stay comparable with one
  with it on; each line now carries ``gate_off`` next to the verdict the rules would have given.

The old ``--always-answer`` flag stays as an alias for ``--no-gate``. What it used to do - leave
the gate watching but let the audio stream straight through - is what caused the double answer,
so it is gone.

## Done: the agent reads the game the robot remembers (2026-09-16, night)

With the gate off, the first session ended with the agent asking "me diga qual é o Ancient One
que vocês escolheram", while the robot's own greeting two minutes earlier had said "vamos
enfrentar Azathoth". Both are true from where each of them sits: the state lives in the robot's
file and the agent had never been shown it.

The agent now has a third client tool, ``game_state``, which returns exactly what the robot
remembers - Ancient One and doom, who plays which investigator and where, its own investigator's
Health, Sanity, Clues and possessions, the Mystery, the Reserve, the round, and what is still
missing - as data rather than prose, so it cannot be mistaken for a rule. The prompt makes it an
obligation: call it before asking the table anything about the setup, never ask for something it
already holds, never contradict it, and ask only for what ``still_missing`` lists. The tool reads
the game at call time, because the robot learns the setup while the session is already running.

The other direction was also shut. The robot's own ear only saw utterances the rules had already
judged to be for it, so with the gate off - where the rules stop deciding - the local memory
never updated from conversation at all, and a Mystery Whisper had written as "Q-Mystery" could
not be corrected, because the ear only listened while something was missing. Now the ear looks at
every non-echo utterance that sounds like a briefing, keeps it only when it learnt something, and
otherwise hands it straight to the agent.

**Why there are two at all**, since the question came up: the agent is the conversation and
nothing else. It is a cloud model that invents when it is out of its depth (which is why the
rules live in a tool), its session dies after two hours and is restarted deliberately on every
language switch, and its context cannot be read by the vision, the turn decision, the plan or the
logs. The game state has to outlive the conversation, be checked against the reference, and be
usable by things that never speak. So: one memory, the robot's, and the agent reads it.

## Done: the Mysteries, and a way to read the rest of the box in (2026-09-16, night)

"Nós colocamos como conhecimento todas as cartas pequenas... mas precisamos também das cartas de
mistério, de portal, de encontros nos locais... enfim, de todo o game."

**The Mysteries are in**, all 16, and they are the ones that matter most: three solved is how the
table wins, and until now the Mystery in ``GameState`` was free text, so "The Deep Ones Attack!"
arrived as "Q-Mystery" and meant nothing. They live in
``src/rag/cards/eldritch_base_mysteries.json`` as **checkable data** - name, Ancient One, which of
the five kinds it is, one line on what solving it takes, the spaces the card names, the Epic
Monster it spawns - each with the page it was read from, and the import refuses the list unless
every Ancient One has exactly four and every space exists on the board.

That last field is why this was worth doing: **the Mystery is now a destination**. "Rituals in
the Wild" puts Eldritch tokens on 4, 10, 21 and Tunguska; ``decide.goals`` reads those spaces and
the turn changes because of it (live: with that Mystery the best plan becomes "Prepare for Travel
Train ticket, then Travel to Tunguska - closer to Tunguska, where the Mystery Rituals in the Wild
is"). The setup ear also checks a spoken Mystery against the 16 now: "The Deepest One Attack" gets
"Era The Deep Ones Attack!?" back, and an unrecognised one is a question rather than a stored
guess.

**The rest of the box is not scraped, it is read.** The encounter decks, the Mythos deck and the
Other World cards are hundreds of cards of the game's own text; the two places that hold them
(the Fandom wiki, BoardGameGeek) are the publisher's work reproduced by fans, and copying them
wholesale into this repository is not something this project will do. ``src/rag/dictate.py`` and
``scripts/dictate_cards.py`` take them from the box instead: somebody reads a card, the local
model puts it into fields, the reference checks the names, and it is stored in ``data/cards/``
(git-ignored, like the picture of the board) and in ``bg_knowledge``.

An Encounter card is filed the way the table calls it - **by its number and its city**. "Saiu a
carta 8, lê a parte de Rome" is number 8, space Rome, and the deck follows from the city
(``reference.region_of``). The agent has a new tool, ``encounter_card``, which reads it back
exactly as it was read in, or says nobody has read that one yet and asks for it once. Live:

    dictate: "Carta de encontro número 8, a parte de Rome: ... Teste Influence ..."
      -> Encounter card: Europe 8, Rome. You search for a contact in the catacombs. Test
         Influence. If you succeed, gain 2 Clues; if you fail, lose 1 Sanity.
    encounter_card(8, "Rome") -> that text, with who read it

Three smaller things fell out of it: a dictated card **corrects** the imported one instead of
sitting beside it (one id convention, ``store.point_id(game, kind, name)``, and the old payload is
kept underneath), the knowledge base stays English even when the card is read in Portuguese, and
the printed **value** of an Asset finally has somewhere to live - ``moves.card_value`` reads it,
so Acquire Assets stops asking for a number somebody has already read out.

## Done: one voice, one language (2026-09-16, after the session was stopped)

The owner stopped a live session: "tem agente demais ouvindo... cada um responde o que acha que
deve responder... até em inglês falou". He was right on both counts, and both were ours.

**Two mouths.** The ElevenLabs agent answered what it heard, and the robot's own TTS answered
for the turn, for a briefing it wrote down, and with a greeting when the session opened. From a
chair at the table that is two things talking, sometimes about the same sentence.

There is one now, and it is the agent's. The robot still decides its own turn and still writes
down its own setup - that is local, checked against the box, and unchanged - but it says neither.
Two new client tools hand it over:

- ``take_turn``: the table gives the robot's investigator its turn, the agent calls this, the
  local model works the move out and returns the sentence to say.
- ``remember_setup``: somebody describes or corrects the setup, the agent passes their words,
  the reference checks the names, and what was written down comes back as one sentence.

Both return a ``say`` field with the note "say this as it is". The greeting is a log line now, not
speech: the agent reads the same state through ``game_state`` whenever it needs it. The only local
speech left is voice enrolment (``--players``), which happens before the agent session starts.

**One language.** ``LANGUAGE_LOCK`` (on by default) and ``--lang``: the session starts in a
language and stays there. Locked, the agent has no ``language_detection`` tool to switch with, no
second voice preset to switch into, and a prompt that says which language it speaks; the gate's
language-switch restart is not wired at all. ``--free-language`` restores what it did before.

Also: the reasoning model is loaded in the background when a game is opened, because the agent
gives a client tool 45 seconds and a cold ``qwen3.6:35b-mlx`` spends 30 of them being read off
disk (measured tonight: 31.6 s cold for a turn, 2-8 s warm).

Live check of both tools, on the owner's saved game:

    take_turn       -> "Vou fazer Acquire Assets em Shanghai ... Qual é o valor total das cartas
                       Lucky Cigarette Case, Private Investigator e Kerosene no Reserve?"
    remember_setup("o Mystery é The Deep Ones Attack")
                    -> noted: mystery: The Deep Ones Attack!   (spelled as the box prints it)
                    -> "Anotado. O Mystery é The Deep Ones Attack!."
    game_state      -> mystery: "The Deep Ones Attack!"

## Done: the state the robot writes to, and a camera that admits what it is (2026-09-17, a long day with the owner)

Eleven fixes, in two halves. The first half closed holes that had been there for days; the
second half was mostly undoing damage this session caused, and the lesson from it is worth more
than the code.

### The holes that were already there

Every one of them had the same shape: **the state existed and there was no door to it.**

- **Barge-in judged the whole voice segment.** While the robot speaks the segment opens on its
  own echo and never closes, so after two seconds it is mostly the robot whatever the player
  says, and `looks_like_echo` ruled echo every time. A player's sentence over the robot was
  heard and ignored. Checks now judge the newest two seconds, and by voiceprint where players
  are enrolled - 0.02 s against Whisper's 1.4 s, and it does not care what proportion of the
  segment is the robot. Measured separation: the robot's own echo scores at most 0.25 against
  an enrolled player, the player 0.45 and up.
- **The addressee gate could be left stuck open.** `release_gate` opens it and only
  `utterance_ended` closes it, which fires when the local VAD closes a voice - so a barge-in
  decided after the voice had ended armed nothing. A whole sentence then streamed live instead
  of being held, and the release that should have closed the agent's turn reported
  `forwarded=0`. The turn-closing silence also went out only when frames had been held, and
  holding the microphone is not silence to the agent: it is no audio at all, which its turn
  detector waits on for ever.
- **A player talking over the end of a sentence was thrown out with the echo.** One unbroken
  voice, first half the robot's; judged whole it read as echo, and `self_echo` is the one
  verdict that survives `answer_everything`. `_after_playback` now cuts at the moment the
  speaker fell silent.
- **There was a second gate, in the cloud.** The agent's prompt ended with "answer when you are
  addressed... otherwise stay quiet", which is right while the local gate is on and a second
  gate when it is off. A question asked straight at the robot went unanswered in a session
  started with `ADDRESSEE_GATE=false`.
- **"Anotado" without writing anything.** Asked to remember where a Gate was, the agent called
  the read-only `game_state` and said it had noted it. `remember_note` is the write door;
  `game_state` returns the notes; the prompt forbids claiming a note without the tool.
- **The camera never reached the conversation.** The preview owned it and wrote
  `board_state.json`; `talk.py` carried its own empty `GameState.board`. The prompt opened with
  "you cannot see the board". `src/vision/board_link.py` is the bridge, `look_at_board` and
  `scan_board` are the tools, and the preview reads its board file back at startup instead of
  throwing away every scan on restart.
- **The game had no round.** `take_turn` derived a move, logged it to the diary and wrote
  nothing, so `round` sat at 0 for a whole session and every call re-derived the same move.
  `GameState` now holds the round structure from GAME_REFERENCE.md, `take_turn` records its
  actions, and `moves.plans` stops offering an action already spent.
- **Card values had only an offline door.** `card_value()` reads what
  `scripts/dictate_cards.py` dictated, so a value said out loud was heard and forgotten and
  Acquire Assets asked again on the next turn. `dictate.remember_value` is the live door, and
  it drops the `lru_cache` that hid the write from the process that made it.
- **Two 4s became two successes**, ninety seconds after the robot quoted the rule correctly.
  Counting is arithmetic: `GameState.test_result` does it, Blessed and Cursed included.
- **Nothing could write Health, Sanity or Clues.** "Vou atualizar isso aqui" left the sheet
  untouched, because none of the eleven tools wrote them - the whole Encounter Phase happened
  outside the game state. `apply_effect` takes deltas, because the table says "perde 1 de
  Sanity" and never "fica com 5".
- **An exact card name is what embeddings are worst at.** "Witch Doctor" came back as Diana
  Stanley and the Witch monster while the card sat in the repository with its text.
  `game_knowledge` tries the 76-asset list by name first, ignoring case and punctuation.

### The camera, and reading a log wrong for three hours

The scan reported nothing all afternoon and the robot honestly told the table the board was
empty. This session read that as a board problem - framing, light, stale baselines - and
reasoned confidently from `frames: 215`, `unseen: none` and `board: 72 inliers`.

All three were true and meaningless. The owner said the page looked dark, a frame was fetched
and looked at, and it was **a different camera**: a wide view of the room with the board on a
table off to one side. This Mac has three cameras that do 1080p. The board matcher had found a
confident homography with 26 inliers over a photograph of a sitting room.

The cause is ours: `talk.py` asks the SDK for `media_backend="no_media"`, and on that branch the
SDK tells the **daemon** to release camera and audio - which is where the preview reads its
frames. A pipeline already running survives it (at 15:10 the release landed 17 s after the
preview was up and the camera kept working); one still being built dies with an "Internal data
stream error" and never comes back. This morning those two GStreamer lines were dismissed here
as harmless, which they are when `talk.py` goes first.

Three faults were then added trying to fix it, and each is a better lesson than the fix:

- A rebuild that **bound to the wrong camera and reported success**. Frames arriving is not
  evidence of the right camera, and nothing the SDK or the daemon exposes says which device was
  taken - so `CAMERA_REBUILD` is off by default. The drought warning stays, because that half
  is what turned "the scan finds nothing" into "the camera is gone".
- `scan_board` **read the scan that was already there** as its own and came back in a second:
  the robot answered before it had looked, talked over its own sweep, and the `finally` resumed
  the speech sway mid-sweep, blurring the views the pause existed to protect. The result now
  carries the time it was made.
- The tool **claimed a piece the vision layer would not claim**. The scan withholds what it is
  unsure of ("not reporting 8: seen from one view only") while `believed` still carries it.

The head moving is its own problem, and the owner worked it out from the log: the camera is in
the head and the speech sway runs at 10 Hz, small enough to slip under the "view moved" check
and large enough to shift where every space sits. It read as three pieces leaving three
different spaces in one instant, and emptied a board nobody had touched. `hold_still` is
refreshed while the voice plays.

The first legitimate scan of the day, at 15:58: *Investigator at Shanghai, confirmed by a closer
look; Portal at Rome, confirmed by a closer look; Monster at Rome*, and it declined to report
what it had seen from one view only. It agreed with a note the table had dictated the day before.

### What the model turns out not to be doing

`scripts/decide_replay.py` measures what the reasoning model adds over the score that already
ranks the turns. Over 24 decisions each, warm: `qwen3.6:35b-mlx` agrees with the score **21
times (88%)**, median 1.9 s and 8.9 s at worst; `qwen2.5:3b` agrees 18 times. The one turn the
35B changed, it changed for the worse - the score took the Trade that was only possible because
another investigator stood on that space, and the model took the plan that would serve any turn,
explaining that acquiring assets was risky before choosing a plan whose second action is
acquiring assets.

Its failure mode is consistent and worth naming: **the facts are grounded and the causal links
are not.** It justified a move by the Protective Amulet's +1 Will, which is printed on the card
and in the prompt, welded to Prepare for Travel, which has nothing to do with it. Pick first,
justify from whatever is in the context window. An agent loop with tools would address exactly
that, and cannot sit in the turn path: `think=True` was measured at 114 s against 4 s and
rejected, because a table does not wait two minutes for a move. The dead time while humans play
is where a slow loop belongs.

### Still open

- **`listening` is never called.** The prompt makes the robot say "estou ouvindo" and stop,
  which is the visible half; the tool that actually holds the gate was not called once in a
  whole session, so nothing protects a card reading from being interrupted.
- **The watchdog fires on Whisper's own noise.** `'E aí'` on near-silence, 73 times in one day
  against 422 real utterances. Language confidence separates them cleanly (0.28 against 1.00)
  but 16 of 422 real sentences fall under any useful threshold - the ones with English names in
  Portuguese, which are the worst to lose. Better to not start the clock than to filter speech.
- **A recorded action cannot be undone.** Told "you resolved that wrong, do it again", the
  machine refuses and the robot restates instead of saying it cannot.
- **Startup order still matters.** Let one of the preview and `talk.py` be fully up before
  starting the other; what breaks is the overlap.
- **The baselines are from 2026-09-16 09:42.** A new sweep needs the board empty, or the pieces
  on it become part of "empty board" and detection never sees them again.

## Done: the gate holds itself, the turn can be taken back, and the model stops choosing (2026-09-17, evening)

Three of the items left open this morning, and the two decisions that were waiting for the
owner. Every one of them was measured before it was changed.

### The three that were open

- **`listening` was never called.** The prompt produced the visible half - the robot said "estou
  ouvindo" - and the tool that actually holds the floor went a whole session without being
  called once, so nothing stopped the next breath of a card reading from being answered over.
  The rule is local now: `announces_reading` runs on the transcript the gate already has,
  arms the hold *after* the announcement has been routed (so the agent still says its one short
  line), and refreshes it when the reader says "espera" again. A question never holds the floor,
  because "espera, quantos dados eu rolo?" is a question with a word in front of it. Measured
  against the 462 transcribed utterances in the log: **2 fire, both right, no false positives** -
  and one of the two is a sentence that produced a false "the agent is dead" warning the same
  afternoon. The agent keeps the tool; it is no longer the only way in.

- **The watchdog fired on Whisper's own noise.** The fix is not to filter speech - 16 of 422 real
  sentences fail every test worth having and they are the Portuguese ones with English names -
  but to not start the clock for what does not look like a voice. `voiced_fraction` measures how
  much of a clip stands above *its own* quiet floor, so a change of microphone gain does not
  retune it. Over the 218 clips forwarded on 2026-09-17 (51 of them `'E aí'` over room noise,
  167 real): at 0.60, **2 of 51 noise clips still arm the clock and 96 of 167 real sentences
  still arm it**. Absolute peak level was measured too and separates worse - noise p95 -25.3 dBFS
  against speech p05 -31.5 - so the scale-free measure is the one kept. Nothing is dropped:
  the audio reaches the agent exactly as before, and missing one costs a warning, never a
  sentence, because the next real sentence arms it.

- **A recorded action had no undo.** `GameState.checkpoint`/`undo` snapshot the state at each
  write door, `undo_that` is the tool, and the prompt now tells the robot that being corrected is
  part of playing - never that it cannot be changed, and never to argue that it already did it,
  because the table is looking at the board and it is not. Two things had to be got right:
  a checkpoint that changed nothing is skipped, so one "refaz" always reaches one real change;
  and investigators are restored **in place** rather than replaced, because undo runs mid-session
  where the turn taker is holding those objects and a fresh list would leave it updating a sheet
  nobody reads.

### The model stops choosing turns

`scripts/decide_replay.py` was run again before anything was decided. Over 16 more decisions
`qwen3.6:35b-mlx` agreed with the score **13 times (81%)** and **all three disagreements made the
turn worse**: it dropped the Rest of a hurt investigator, it added an Acquire Assets against an
empty Reserve, and it gave up a Trade that was only possible because another investigator was
standing on that space. It also **answered differently on the same situation in 3 of the 8**, so
the same table state did not produce the same turn twice.

So the score takes the best turn and the model is handed that turn, and the score's own words for
why it won, and asked only for the sentence. `NARRATION_PROMPT` names its measured failure - the
facts are real and the links are invented - and answers it by handing over the real reason and
forbidding the rest: a fact may be used only when it is about an action in *this* turn. Measured
after the change, on the game as it stands: **the same turn on all six runs of three situations,
1.3-1.6 s warm against a 4.4 s median**, and a model that cannot answer now costs a plainer
sentence instead of a different move. `decide(model_picks=True)` keeps the old arrangement
reachable, and `decide_replay.py` uses it, so this can be re-measured when a model changes.

### The loop in the dead time

`src/strategy/reflect.py` thinks between rounds, on its own thread, while the humans take their
turns. It is not in the turn path and must not be put there.

What it may hand over is not prose: `src/strategy/intent.py` defines three shapes - reach a
space, prefer an action, avoid a space - and the code checks every one before the score sees it.
The space has to be on the map table, the action one of the six on the reference card, the weight
is clamped, and no round gets more than four. The score's term for an intent is the same size as
one Monster on the space a turn ends on, so a priority **re-ranks turns that were nearly level
and cannot overturn a fight**.

Two of the checks exist because it was run against the real game instead of reasoned about. On
the first live run it **committed on step one without looking anything up**, and explained a Rest
by *"o combate iminente no espaço do Mar"* - there is no such space, there was no combat, and it
had read neither. Having tools is not the same as using them, and a prompt asking nicely is not a
mechanism. So: at least one lookup is required **in code**, and every intent has to cite words
that actually appear in something it read. After that it looked up the Mystery in play, quoted
the card - *"An Eldritch token goes on the Sea space nearest each investigator"* - and its
priorities followed from the quote.

**What is checked is that the fact is real and was read. What is not checked is whether the link
from that fact to the target holds** - "prefer Acquire Assets" citing "spend a Clue to take the
token" is still a leap. The bounded weight is the safety net, and every round's thinking,
refusals included, goes into the diary as a `thinking` line for the owner to argue with.

### The game state

The note said San Francisco and the sheet said space 5. The owner settled it: she moved from 5 to
San Francisco, and the game is at the Mythos Phase. Both were written, and note 4 now records the
move rather than standing as a second, competing record of where she is.

### Still open

- **The setup ear and the local turn are not wired into `talk.py`.** `Gatekeeper` takes
  `on_addressed` and `wants_setup`, both of them are exercised in the tests, and `main()` passes
  neither - so `setup_talk` never fires live and the local `TurnTaker.handle` path never runs.
  The turn reaches the table through the `take_turn` tool, which is the "one mouth" decision, so
  this may be deliberate; the setup half looks like it is not. Not touched, because it changes
  what the robot does with an utterance in a live session.
- **The empty-board baselines are still from 2026-09-16 09:42.**
- **The path table has still not been checked against the board**, so the robot still will not
  move a piece.

## Decisions taken

| Topic | Decision | Where |
|---|---|---|
| Reasoning model | Qwen 3.6 35B-A3B (`qwen3.6:35b-mlx`), not Qwen 2.5 72B | docs/MODEL_SIZING.md |
| Embeddings | bge-m3 through Ollama, 1024-d | src/rag/collections.py |
| Vector store | Qdrant in the existing Docker container; collections `bg_rules`, `bg_knowledge`, `bg_sessions` | docs/DESIGN_DOCUMENT.md |
| Speech input | Mac microphone; mlx-whisper; diart + pyannote 3 live in a sidecar; pyannote 4 + WhisperX offline | docs/SPEECH_PIPELINE.md |
| Speech output | ElevenLabs by default, local TTS optional | src/config.py |
| Conversation | ElevenLabs agent (cloud) + local client tools; local Whisper + Ollama kept as fallback. **One voice**: the agent speaks, and everything the robot works out locally (its turn, the setup it writes down) reaches the table through a tool | docs/SPEECH_PIPELINE.md |
| Language | one per session, chosen at the start and locked (`LANGUAGE_LOCK`, `--lang`); the agent is given no way to switch | src/speech/eleven_agent.py |
| Agent LLM | `gpt-4.1-mini` inside ElevenLabs (owner's decision, 2026-09-14): follows the prompt better than the Gemini default; billed through the ElevenLabs account, no OpenAI key | src/config.py |
| Turn-taking | the local ear decides speak/stay-quiet before any audio reaches the agent (rules on the local Whisper transcript; a local LLM classifier only if the rules prove insufficient). Switchable: with `ADDRESSEE_GATE=false` the robot answers everything and the rules only watch, which is how the owner is running it for now | src/speech/gatekeeper.py |
| Game setup | verbal briefing + knowledge base, no card OCR | docs/SETUP_PROTOCOL.md |
| Card text | the 76 base assets and the 16 Mysteries ship with the repository (effects, and what solving one takes); the encounter, Mythos and Other World decks are read from the owner's own box when the table needs one, never scraped | src/rag/dictate.py |
| Choosing a turn | the **score** chooses; the reasoning model only says it out loud. Measured twice with `scripts/decide_replay.py` (88% and 81% agreement, every disagreement worse, unstable across repeats); `decide(model_picks=True)` keeps the old arrangement measurable | src/strategy/decide.py |
| Slow reasoning | a loop with tools between rounds, never in the turn path, writing checked `Intent`s the score reads - bounded so they re-rank near-ties and cannot overturn a fight | src/strategy/reflect.py, src/strategy/intent.py |
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

### Next: the robot still does not play

Everything above is perception and memory. The robot can see the board, follow a move, hold the
whole game state and answer questions about it - and it has no investigator turn of its own. It
never chooses an action, never justifies one, never acts. That is Phase 3, and it is the only
thing between here and a real game.

What is already in place for it: the `GameState` (its own investigator's sheet, skills, space,
health, sanity, possessions), `bg_rules` (the rulebook), `bg_knowledge` (76 cards with printed
effects, 122 monsters, 4 Ancient Ones, 11 strategy entries), and `llm.ollama_client.chat()` with
JSON-schema output.

### Re-ordered after the 2026-09-16 measurements

Naming a piece from a crop was going to come first (normalise the gallery labels, tune the
0.72 threshold, decide ResNet vs a stronger embedder). The measurement says that order is
wrong: on the current crops the gallery gets 2/13 and a VLM 0/13, because the crops are too
small and soft for either. Pixels first, recogniser second.

1. ~~Live test of `motion.py`~~ done (section above): six moves, four as one clean sentence.
2. **Bigger, sharper crops**: a piece is about 250 px of 1920 today. The closer look should
   fill the frame with the piece before anything tries to name it.
3. **Clean the gallery**: remove the byte-identical crop filed under two labels, then
   normalise to `kind:Name`.
4. Only then re-measure recognisers, with `mlx-vlm` as the cheap way to run one.
5. Pending from before: the live test of the addressee gate.

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

- [x] Ollama client with JSON-schema outputs; prompt templates in English.
- [x] Candidate generation, legality checks, evaluation, justification (`map_graph`, `moves`,
      `decide`, `plan`, `turn`). Legality is code and data, not a vector search: the six actions
      are printed on the reference card and fixed, card effects come from the versioned card
      file, and `bg_rules` is cited rather than queried per candidate.
- [ ] The owner checks `scripts/draw_map_graph.py`'s picture against the board, then `VERIFIED`
      turns True and the robot may move a piece.
- [x] The spoken turn: "é a vez da Lily Chen" -> `decide` -> the robot's own voice, with the
      agent never hearing the turn call (`src/integration/turn_taking.py`); the live test with
      the owner speaking is still pending.
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

- ~~`listening` is never called by the agent~~ done: the ear arms the hold itself
  (`announces_reading`), and the tool stays as a second way in.
- ~~The agent-quiet watchdog fires on Whisper's hallucinations~~ done: the clock is not started
  for audio that does not look like a voice (`voiced_fraction`), and nothing is filtered.
- ~~A recorded action has no undo~~ done: `undo_that`, one real change per call.
- **`on_addressed` and `wants_setup` are never wired in `talk.py`**, so the local setup ear and
  the local turn path do not run in a live session. Worth the owner's decision: the turn half
  follows from "one mouth at the table", the setup half looks like an oversight.
- **An intent's citation is checked; the link from it is not.** "prefer Acquire Assets" citing
  "spend a Clue to take the token" passes every check there is. The bounded weight is what keeps
  that cheap, and the `thinking` lines in the diary are where it would be caught.
- **CAMERA_REBUILD is off by default** until there is a way to tell which camera the pipeline
  bound to. Frames arriving is not evidence of the right one.
- **The empty-board baselines are from 2026-09-16 09:42.** A new sweep needs the board clear.
- **The path table has not been checked against the board** (`scripts/draw_map_graph.py`).
  Until it is, the robot will not choose a turn that moves a piece. Worth the owner's eye:
  spaces 9, 13 and 21 came out with a single path each, and London with no Train path at all.
- Calibration numbers (riser height, head pitch, exposure) are placeholders until measured.
- Latency figures in docs/SPEECH_PIPELINE.md come from published benchmarks, not this Mac.
- The two unverified facts in docs/GAME_REFERENCE.md (Doom track maximum, token counts).
- Repairing the robot microphone cable would add direction-of-arrival to addressee detection.
