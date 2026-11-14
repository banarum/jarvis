# Changelog

## 2026-11-14 — The chat becomes part of the conversation

### Added
- Results by text: when the person who asked has hung up, an agent's answer reaches them as
  one short Telegram message, batched, project name first. The recipient is whoever asked;
  no id is configured for it.
- Chat sessions: a whitelisted person can write to the account and work in text with the
  same conversation, tools and rules as on a call. Voice notes are transcribed and treated
  as messages.
- Nothing is written to someone while they are on a call.

### Changed
- Conversations and open questions are kept per caller.

## 2026-11-13 — The line moves to Telegram

Jarvis is now reached by calling a Telegram account. The browser call page is gone.

### Added
- `tgline.py`: a Telegram user-account client that answers voice calls from whitelisted
  users, declines everyone else, and feeds the call into the existing call classes and
  tools. Also handles private text from whitelisted users and one work chat, whose messages
  the assistant can read (`read_chat`) and whose files land in its queue.
- `phone.py`: call behaviour for the text-brain pipelines.
  - A new phrase cancels the reply being written and is answered with the earlier ones.
  - Sound over the assistant pauses it; words make that an interruption, a backchannel or
    silence lets it resume.
  - The history keeps only what the caller heard to the end.
  - Replies are streamed and spoken sentence by sentence, with a short filler when a tool
    call starts and an instant greeting on pickup.
  - Messages to agents no longer block; a ledger records what was asked and what came back.
  - Messages that change a product are held until the caller says yes to a read-back.
  - One conversation carries across calls and is summarised a few minutes after a hang-up
    once it is long.
  - A backup model takes over when the main one goes quiet for 5 seconds.
  - Mis-heard names can be mapped to the real ones (`aliases`).
- `turn.py`: end-of-turn detection with Pipecat's Smart Turn v3 audio model, with a minimum
  quiet that adapts to the caller's pauses.
- `turn_protocol.py`: Pipecat's turn-completion protocol (BSD 2-Clause, notice retained):
  the model marks the caller's turn complete, cut off, or needing time.
- `local.py`: the text-brain pipelines (`speak`: cloud transcription and speech, `local`:
  Whisper and the system voice).
- `TELEGRAM.md` (setup, configuration, secrets, what is missing), `STATUS.md` (history,
  measurements, open problems), `telegram.example.json`, `requirements-telegram.txt`.

### Changed
- `server.py` is now the core only: call classes, tools and the agent monitor.
- Hand-offs to agents carry just the request; the standing "answer briefly" part is added
  in code.
- `README.md` rewritten for the Telegram line.

### Removed
- The Gemini Live pipeline, where one model listened, thought and spoke, and its web-search
  tool. Gemini is now used for transcription and speech only; a text model does the thinking.
- The browser call page: `static/`, `run.sh`, the HTTP server, the upload endpoint and the
  page token.

### Known gaps
- Project names are still mis-heard by the transcriber; aliases are a stopgap.
- Tool turns can take many seconds before the real answer.
- No menu-bar app, no autostart, macOS only so far.
- Several of the newest mechanisms have only been tested offline; see `STATUS.md`.
