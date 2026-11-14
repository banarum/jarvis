# Status

## Telegram line (`tgline.py`)

- 2026-11-13 — Speech rate on the Telegram line slowed by 20%: it was following the shared
  `speed` in `~/.config/jarvis/config.json` (1.5); the line now has its own `speed` in
  `~/.config/jarvis/telegram.json`, set to 1.2. The call page keeps the shared value.
  Applies to the `speak` and `local` pipelines (ffmpeg `atempo` / `say -r`); the `gemini`
  pipeline speaks at Gemini Live's own pace and has no rate setting. Read at start, so a
  change needs a restart of `tgline.py`.
- 2026-11-13 — After the first real calls on the `speak` pipeline (answers 20 s late and out
  of order, no way to interrupt, phantom transcripts, silence on pickup): added `phone.py`,
  which wraps the `speak`/`local` call classes on this line only — latest phrase wins,
  reliable barge-in, streamed replies with a "one second" filler before tools, instant
  greeting, transcript filter plus Whisper fallback, brain reasoning off, news kept across
  hang-ups, per-turn latency timing. `server.py` and `local.py` are untouched. Offline
  scenario: 2.0–3.5 s from end of phrase to first sound. Not yet confirmed on a real call.
- 2026-11-13 — Third real call (4 min, `speak` + `phone.py`): in step for 3 minutes
  (median 2.2 s to first sound), then the brain provider went quiet for about a minute and
  the call's media connection dropped. Added: brain requests are abandoned after 7 s of
  silence and retried (twice), the end of a reply is spoken without waiting for the stream's
  tail, greeting and filler are synthesised at startup, and the log now records media
  connection state, loss of the caller's audio and why a call ended. Why the connection
  dropped is not known; the new logging is there to show it next time.

## Open problems and next steps (Telegram line)

- Confirm on a real call: brain stall retry, instant greeting, drop diagnostics.
- Cause of the media connection drop after ~4 minutes: unknown. Look in `logs/tgline.log`
  for "call media connection", "no audio from the caller", "telegram says the call ended".
- Possible: call the user back automatically when a call drops (not decided).
- Tool turns take ~8 s to the real answer (`send_to_agent` waits 3 s, two model round-trips).
- No guard against a slow speech-synthesis start (seen once: 4.6 s).
- The transcriber mishears project names; feed it the glossary more aggressively.
- Unverified on real calls: echo-triggered false barge-ins, the Whisper fallback.
- Not done: work chat configured, menu-bar app, autostart, Linux, outgoing calls.
- Nothing is committed yet: `tgline.py`, `phone.py`, `TELEGRAM.md`, `STATUS.md`,
  `telegram.example.json`, `requirements-telegram.txt`, plus edits to `.gitignore`, `README.md`.

## Real-call scores so far

| Call | Pipeline | Score | Why |
|---|---|---|---|
| 1 | gemini | ~7 | in step, 0.2–0.8 s to first sound; short call |
| 2 | speak, before `phone.py` | 3 | answers 20 s late and out of order, no barge-in, phantom transcripts |
| 3 | speak + `phone.py` | 5 | 3 good minutes (median 2.2 s), then a 48 s brain stall and a dropped connection |

- 2026-11-13 (evening) — Review of the long calls: 5/10 (delegates instead of answering,
  guesses stated as fact, long answers, a 1.5-minute brain provider outage that also wiped
  the call's history). Added a conversation that persists across calls
  (`phone-context.json`), compacted into a summary a few minutes after a hang-up once it
  is long; a brain failure no longer wipes the history. Offline test: a fact told in call 1
  was recalled in calls 2 and 3, and after compaction. Still open from the reviews: backup
  brain, consent rule, mishearing list, length cap, announce-before-delegating.

- 2026-11-13 (late) — Two changes modelled on how LiveKit/Pipecat agents work.
  (1) Delegation no longer blocks: `send_to_agent` returns at once, the assistant says whom
  it is asking in the same reply as the tool call, answers are followed in a ledger
  (`phone-ledger.json`, `ledger` tool). Offline with a simulated agent: first sound 2.2 s
  after the caller stopped (was 8–20 s on real calls); "what are we waiting for" answered
  from the ledger. (2) Turn detection: Smart Turn v3 (`turn.py`) may end a phrase after
  320 ms of quiet instead of the fixed 800 ms; on synthetic speech it was right 7 times
  out of 10, so the fallback silence is capped at 1000 ms until real phrases
  (now recorded in `logs/phrases/`) show how far it can be trusted. Neither is confirmed
  on a real call yet.

- 2026-11-13 (night) — Read the LiveKit Agents and Pipecat sources (not adopted as
  frameworks: Telegram audio would need a custom transport either way) and copied their
  mechanisms: two-level end-of-turn wait (answer at once when the turn model is sure, up
  to 2.5 s when it is not, probes at 240/800/1600 ms); barge-in pauses instead of dropping,
  resumes after 2 s without words or on a backchannel; history trimmed to the sentences
  actually heard; backup brain after a 5 s stall; one agent answer closes every question
  waiting on that agent; tool-specific fillers; "Claude Code" removed from the
  transcriber's hints; read-back rule for product-changing instructions. Offline: noise
  over an answer paused and resumed it, "угу" resumed it, "стоп" cut the history to 2 of 6
  sentences; with the main brain unreachable the backup answered. Not yet on a real call.
  Not taken yet: speculative generation, the LLM's own complete/unfinished markers,
  per-speaker adaptive delay.

## Plan: main Jarvis process on the remote Mac mini (asked 2026-11-13)

Host names, addresses and keys are deliberately not written here (public repo).

### Verdict
Technically possible, but **not onto that mini as it is**: it is the machine that was
root-compromised in August and it has never been reimaged (system install date predates
the incident, 85 days of uptime). Whatever runs there must be assumed readable by a third
party: the Telegram user session, the model keys, the assistant's memory, and an SSH key
that reaches the laptop and every agent. Recommendation: staged, and only after a reimage.

### What makes "one Jarvis, agents on both machines" work
- `herdr --machine <label>` drives another machine's agents over SSH; `server.py` already
  uses it for every machine in `machines`, reads remote session transcripts over SSH and
  copies attachments with `scp`. Nothing new has to be written: on the mini, "local" would
  be the mini's agents and the laptop becomes a `machines` entry.
- Today it works in one direction only: laptop → mini. The reverse needs Remote Login
  enabled on the laptop (it is off) and a key on the mini that is authorised there.
- The laptop must be awake and on the tailnet to be seen; with the lid closed its agents
  simply disappear from the board until it is back.

### What does not survive the move by itself
| Thing | After the move |
|---|---|
| Telegram user session | Do not copy the session file: the same session from two hosts gets it revoked. Log in again on the mini (`tgline.py login`: phone number, code from the account's Telegram app, 2FA password) and stop the laptop's line first. |
| Voice pipeline | `gemini` and `speak` are cloud and move as is. `local` needs Apple Silicon (the mini is) plus the Whisper model download. Turn model re-downloads. Call audio now crosses the Atlantic: expect roughly 0.1–0.2 s more per turn (not measured). |
| `~/.config/jarvis/` | `env` (keys), `config.json`, `telegram.json`, `memory.json` (notes), `phone-context.json`, `phone-ledger.json`, `token` must be copied; `uploads/` (attachment queue) is a day's cache and can be dropped. |
| Repo state | Nothing of the Telegram line is committed yet; commit and push first, then clone on the mini. |
| Tailscale serve link | The browser call page gets a new host name; `tailscale serve` must be set up on the mini and the bookmark on the phone changed. |
| `logs/` | Call transcripts, phrase recordings, latency and usage logs stay on the laptop unless copied. |
| `research` tool | Needs `opencode` on the mini. |
| Process supervision | Currently `nohup` by hand; on an unattended host it needs launchd. |

### What the mini needs (checked read-only on 2026-11-13)
- Reachable over the tailnet now; SSH with the laptop's key works; 32 GB RAM, plenty of disk.
- System Python is 3.9: too old (the call library needs 3.10+). No `ffmpeg`, no `opencode`
  on the non-interactive PATH. Install Python 3.10+, ffmpeg, opencode, then the two
  requirements files.
- It is a CI host: a build was using three cores during the check (load 8–10). Real-time
  audio and CI builds will compete; either move CI or accept worse latency during builds.
- Two SSH keys are authorised there: the laptop's, and one deploy key to confirm.

### Staged plan
0. **Blocker: reimage the mini** (provider), new admin password, rotate every key that was
   on it, re-join the tailnet as a new node, re-register CI runners with new tokens.
1. Commit and push the Telegram line; on the clean mini: Python 3.10+, ffmpeg, herdr,
   opencode, clone, venv, copy `~/.config/jarvis/` (without `telegram.session`, `uploads/`).
2. On the laptop: enable Remote Login for one dedicated key from the mini, restricted to
   the tailnet interface; add the laptop as a herdr machine on the mini; put it in
   `machines`. Check `list_agents` on the mini shows both.
3. Dry run on the mini with the browser page only (`server.py`, `tailscale serve`), laptop
   line still answering calls.
4. Cut over the phone line: stop `tgline.py` on the laptop, `tgline.py login` on the mini,
   start it under launchd, one test call, compare latency with `logs/latency.jsonl`.
5. Keep the laptop as a worker; its copy of the line stays installed but stopped, as the
   fallback.

### If always-on is needed before the reimage
Keep the main process on the laptop (it is kept awake) and do not add the mini to
`machines`: its agents' output would flow into the assistant's context from a machine
that cannot be trusted.
- 2026-11-13 (20:00) — 27-minute real call on the copied mechanisms: 5.5/10. Found and
  fixed: the "[interrupted …]" markers written into the assistant's own history were read
  aloud (now a "…" plus a system note); the backup brain was refused by Gemini because of
  unsigned tool calls in the history (history is flattened for it); a Whisper rescue could
  repeat a phrase already heard; agent news was delivered on top of a fresh question (now
  waits 8 s after the caller's last words). Added: mishearing aliases (`aliases` in
  `telegram.json`) applied before the brain, and `turn_min_ms` 480 — the turn model was
  ending phrases after 240 ms while the caller was still mid-thought, so replies were cut
  before a word of them was heard (30 times in that call). Still open: the model does not
  read back product-changing instructions; tool turns of 15–30 s; Pipecat's
  complete/unfinished marker from the brain is the next candidate for the split phrases.
- 2026-11-13 (20:30) — For the problems of the 27-minute call, taken from the frameworks
  rather than invented: Pipecat's turn-completion protocol (the brain marks the caller's
  turn complete / cut off / needs time; text copied with its BSD notice into
  `turn_protocol.py`), LiveKit's rule that the agent never starts speaking into the caller's
  speech, LiveKit's dynamic endpointing (the minimum quiet learns this caller's pauses,
  480–900 ms), and a code-level yes-gate for product-changing messages modelled on the
  existing approve-dialog gate. Text-driven test: three unfinished fragments got silence,
  the finished question got an answer that picked the thread back up; a product-changing
  message was held, read back, and sent only after "да". Not yet on a real call.
- 2026-11-13 (20:20) — Gemini removed from every thinking role on the line (owner's call:
  too weak): the backup brain is now `gpt-5.4-mini` on the main gateway. Brain candidates
  benchmarked with the full prompt (first token, protocol marker, delegation): main model
  1.1–2.2 s, `gpt-5.4-mini` 1.0–1.3 s, `claude-haiku-5-5` 0.9–1.6 s, `minimax-m3` 1.2–1.7 s;
  the rest 2–14 s. A faster front model buys ~0.3 s; hand-off text is now kept to the request
  itself with the standing preamble added in code (~1 s saved per delegation).
- 2026-11-13 (20:25) — The browser call page is removed: the line is Telegram only. Gone:
  `static/`, `run.sh`, the HTTP server, upload endpoint and page token in `server.py`, the
  Tailscale link to it. `server.py` stays as the core (call classes, tools, agent monitor).
  A copy of the removed files and of the uncommitted page edits is kept outside the repo.
- 2026-11-13 (20:40) — The Gemini Live pipeline is deleted from the code (`server.py`): no
  model does listening, thinking and speaking in one any more. Pipelines left: `speak`
  (default) and `local`. Gemini remains only as the transcriber and the voice of `speak`.
- 2026-11-14 — Text channel built: results by message after a hang-up (batched 90 s, one
  per 3 min, urgent ones at once), recipient = who asked, chat sessions on the same
  conversation, voice notes via Whisper, silence in the chat during a call, conversations
  per caller. Tested without Telegram (faked agent and message sender): the hand-off was
  stamped with the caller, one result message arrived, a second allowlisted person got
  their own conversation, nothing was written during a simulated call. Not yet exercised
  on the real account.
