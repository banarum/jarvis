# Jarvis

A phone line to the coding agents (Claude Code, opencode) you run in
[herdr](https://herdr.dev). Call a Telegram account from your phone, ask what the agents
are doing, tell them what to do next, approve their prompts — without looking at a screen.

`tgline.py` logs into Telegram as a dedicated user account and answers voice calls from
whitelisted users. The caller's speech is transcribed, a text model with tools that drive
the `herdr` CLI decides what to do, and the answer is spoken back. Setup, configuration,
secrets and what is still missing: [TELEGRAM.md](TELEGRAM.md). History and open problems:
[STATUS.md](STATUS.md).

## What it can do

- **Look**: list agents by project, read an agent's screen, answer questions from a whole
  session or a whole project.
- **Act**: message an agent (changes to a product wait for a spoken yes), start a new agent
  in a project, answer approval dialogs, keep a ledger of what was asked and what came back.
- **Search**: a sandboxed opencode researcher for questions that need the web.
- **Tell you things**: agents that finish or get blocked are mentioned in pauses.
- **Remember**: one conversation carried across calls, plus notes the model keeps for itself.
- **Files**: photos and documents sent to the account, or posted in its work chat, land in
  a queue the assistant can hand to agents.

## Requirements

- macOS or Linux with `herdr` installed and agents running in it
- Python 3.10+, `ffmpeg`
- A Telegram account for the line, and API credentials from my.telegram.org
- A Gemini API key (speech in and out only) and a key for an OpenAI-compatible text model
- `opencode` on `PATH` (only for the `research` tool)

## Pipelines

- **speak** (the one in use): Gemini transcribes and speaks, a text model thinks
  (`JARVIS_BRAIN_KEY`, `JARVIS_BRAIN_MODEL`, `JARVIS_BRAIN_URL`), with a backup model when it
  stalls (`JARVIS_BACKUP_MODEL`).
- **local**: Whisper on this Mac (`pip install mlx-whisper`) transcribes, the same text
  model thinks, macOS `say` speaks.

Everything personal lives in `~/.config/jarvis/`, never in the repo. Every model call is
priced into `logs/usage.jsonl`.

## Safety notes

- Agent output, session transcripts and web pages go into the model's context. The prompt
  tells it to treat them as data, and actions on agents are only allowed right after the
  caller's own speech, but this is mitigation, not a guarantee. Only list machines in
  `machines` whose agents you trust.
- The researcher runs in `research/`, whose `opencode.json` denies shell, edits and
  access outside that folder.
- `logs/` contains transcripts and recordings of what the caller said.
