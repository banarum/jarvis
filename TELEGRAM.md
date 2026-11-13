# Jarvis on Telegram

`tgline.py` is the line. It logs into Telegram as a dedicated **user** account (bots
cannot receive calls), answers voice calls from whitelisted users and runs them through
the call classes and tools in `server.py` and `local.py`. There is no browser page any more.
It also listens to one work chat.

```
caller ⇄ Telegram call (py-tgcalls / ntgcalls, Telethon session)
           ⇅ 48 kHz PCM, 10 ms frames
         Line: resample, VAD / turn-taking, barge-in, latency timer
           ⇅ frames and pauses, the interface the call classes expect
         server.Call (gemini) | local.SpeakCall (speak) | local.LocalCall (local)
```

## Setup

```sh
.venv/bin/pip install -r requirements.txt -r requirements-telegram.txt   # plus mlx-whisper for "local"
```

1. Create an app at https://my.telegram.org → API development tools, and add to
   `~/.config/jarvis/env`:
   ```
   TELEGRAM_API_ID=...
   TELEGRAM_API_HASH=...
   ```
2. Log in once as the dedicated account (asks for phone, code, 2FA password):
   ```sh
   .venv/bin/python tgline.py login
   ```
   This writes `~/.config/jarvis/telegram.session`. Alternatively put a Telethon string
   session into `TELEGRAM_SESSION=` in the env file.
3. `cp telegram.example.json ~/.config/jarvis/telegram.json` and fill in:
   - `whitelist`: numeric user ids allowed to call and message. Empty = refuses to start.
   - `work_chat`: chat id, `@username` or invite link. `tgline.py chats` lists ids.
     With an invite link or a public username the account joins by itself.
   - `pipeline`: `gemini`, `speak` or `local` (default: `JARVIS_PIPELINE`).
   - `speed`: speech rate of the `speak`/`local` pipelines on this line only (1.0 = the voice's
     own pace); without it `config.json`'s `speed` applies. The `gemini` pipeline has no rate control.
   - `hangover_ms`: silence that ends the caller's phrase (lower = snappier, more cut-offs).
4. In the account's Telegram privacy settings, allow calls from the whitelisted users
   (Privacy → Calls). Peer-to-peer can stay off.
5. `.venv/bin/python tgline.py`

`tgline.py selftest` speaks a question into the pipeline without Telegram and prints
the latency; the answer is saved to `logs/selftest.wav`.

## What it does

- **Calls**: a call from a whitelisted id is accepted; anyone else, or a second call
  while one is live, is declined. Speaking over the assistant cuts it off.
- **Messages**: a whitelisted user's private text goes into their live call, or into a
  text conversation with the same brain and tools. Everyone else is ignored.
- **Work chat**: every message is kept (`read_chat` tool, last 200) and every photo or
  file is downloaded into the file queue (`~/.config/jarvis/uploads`), where
  `list_files` / `send_to_agent(files=…)` see it. Files sent privately by whitelisted
  users land there too. Chat text is given to the model as data, not as instructions.

## Turn detection (`turn.py`)

A phrase ends when Pipecat's Smart Turn v3 model (ONNX, runs locally, downloaded once
from Hugging Face) judges from the audio that the caller has finished: it is asked after
240 ms, 800 ms and 1.6 s of quiet, and a confident yes (`turn_threshold`) ends the phrase
there. Otherwise the caller is given `hangover_max_ms` (2.5 s) to finish the thought: the two
delays LiveKit uses. `smart_turn: false` goes
back to the plain `hangover_ms` timer. Each phrase's audio and the model's verdicts are
kept for a week in `logs/phrases/` and `logs/turns.jsonl` (`record_phrases`), to tune
this and to compare transcribers; they are the caller's voice, so they stay out of git
with the rest of `logs/`.

## Whether the caller has finished: three layers

1. `turn.py` (audio): ends a phrase early when it sounds finished, but never before
   `turn_min_ms` of quiet; that minimum grows towards `turn_min_max_ms` each time the caller
   carries on right after an early ending (LiveKit's dynamic endpointing).
2. `turn_protocol.py` (meaning): the brain starts every reply with a marker — complete,
   cut off mid-sentence, or needs time. For the last two it says nothing and judges the next
   words together with these; if none come it is asked to speak after 5 or 10 s. The protocol
   text is Pipecat's (BSD 2-Clause, notice kept in the file).
3. The mouth never starts a sentence while the caller is speaking.

Messages to agents that change a product are held by `send_to_agent` until the caller says
yes to a read-back (`confirm_send`), unless the caller's last phrase was itself a clear go.

## Phone manners (`phone.py`)

On this line the `speak` and `local` pipelines are wrapped by `phone.py`, because on a real
call the page's one-phrase-at-a-time loop drifts out of step with the caller:

- a new phrase cancels the reply being written and is answered together with the earlier ones;
- sound over his voice only pauses him (as LiveKit's agents do): words that follow make it
  an interruption, "угу" or nothing lets him carry on where he stopped two seconds later;
- his history keeps only the sentences the caller heard to the end; a reply that was cut
  off is marked so, and he does not assume the rest is known (Pipecat's rule);
- if the brain goes quiet for 5 s the turn is retried on a backup model
  (`JARVIS_BACKUP_MODEL`, default `gpt-5.4-mini` on the same gateway; Gemini is used for
  audio only, never for thinking),
  which is then used for 90 s;
- the reply is streamed and spoken from its first finished sentence; "one second" is said
  when it turns out to need a tool;
- a fixed greeting (`greeting`) is spoken the moment the call connects;
- transcripts in a script or language nobody speaks here are dropped, and a phrase the
  transcriber returns nothing for is transcribed by Whisper instead;
- the brain's hidden reasoning is off on calls (`brain_extra`; `{}` turns it back on);
- agent news that arrives after a hang-up is told at the start of the next call;
- handing something to an agent does not block: he says whom he is asking in the same
  breath, the send returns at once, and the answer is told when it lands. What was asked
  and what came back is kept in a ledger (`~/.config/jarvis/phone-ledger.json`, the
  `ledger` tool), so "what are we waiting for" is answered from it, not by asking again;
- one conversation across calls: a call picks up the history of the previous ones
  (`~/.config/jarvis/phone-context.json`). `compact_after_min` minutes after a hang-up,
  if nobody has called back and it has passed `compact_over_chars`, the older part is
  folded into a summary and the last messages are kept word for word. Delete the file
  to start from a blank slate.

## Secrets — never commit

All in `~/.config/jarvis/` (mode 600), covered by `.gitignore` in case they stray:

| File | Why |
|---|---|
| `telegram.session` (or `TELEGRAM_SESSION`) | a full login to the account |
| `env` | `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, model keys |
| `telegram.json` | whitelist ids, work chat |
| `telegram-chat.jsonl`, `uploads/` | work chat contents |
| `phone-context.json` | the running conversation of the calls |

`logs/` holds call transcripts and `latency.jsonl`.

## Latency

Time from the VAD deciding the phrase is over to the first sound of the answer, measured
with `selftest` (one question that needs a `list_agents` tool call). Add `hangover_ms`
(800) for the time from the caller actually going quiet, and the Telegram network leg,
which is not measured yet.

| pipeline | answer starts after |
|---|---|
| gemini (Gemini Live) | ~0.5 s |
| speak (Gemini ears/mouth, text brain) | ~2.5 s (first words of the answer; ~5.5 s when a tool is needed, with "one second" at ~2 s) |
| local (Whisper, text brain, macOS `say`) | ~4.1 s |

Every answer in a real call is timed the same way into `logs/latency.jsonl`, and a call
ends with a median/worst line in the log.

## Still missing

- **Not yet tested on a real Telegram call.** Login, accept, decline and the audio frames
  are written against py-tgcalls 3.0 / ntgcalls 3.0.7 but need the account to verify;
  the frame format (48 kHz mono, 10 ms) is the first thing to check if audio is silent.
- No shell/file tools of its own: the agent can do what the Jarvis stack exposes (herdr
  agents, research, memory, file queue), nothing more.
- The VAD is the page's energy threshold; a model VAD (Silero) would handle noisy lines.
- No menu-bar UI, no autostart (launchd), no reconnect supervision; Mac only so far
  (the `local` pipeline needs Apple Silicon; `gemini`/`speak` should run on Linux).
- Outgoing calls (the assistant ringing you) and group voice chats.
- Work chat images are downloadable by agents but not looked at by the voice model itself.
