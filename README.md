# Jarvis

A phone-call style voice line to the coding agents (Claude Code, opencode) you run in
[herdr](https://herdr.dev). Call it from your phone, ask what the agents are doing, tell
them what to do next, approve their prompts — without looking at a screen.

It is a small aiohttp server that bridges a browser microphone to the Gemini Live API
and gives the model tools that drive the `herdr` CLI.

## What it can do

- **Look**: list agents by project, read an agent's screen, answer questions from a whole
  session or a whole project.
- **Act**: message an agent, run a multi-round "find out and report" job, start a new
  agent in a project, answer approval dialogs (the only action that needs a spoken or
  tapped yes).
- **Search**: Google Search for quick facts, and a sandboxed opencode researcher for
  questions that need digging.
- **Tell you things**: agents that finish or get blocked are mentioned in pauses.
- **Remember**: the model keeps its own notes between calls.

## Requirements

- macOS or Linux with `herdr` installed and agents running in it
- Python 3.10+
- A Gemini API key with access to the Live API
- `opencode` on `PATH` (only for the `research` tool)

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

mkdir -p ~/.config/jarvis
echo 'GEMINI_API_KEY=your-key' > ~/.config/jarvis/env
chmod 600 ~/.config/jarvis/env
cp config.example.json ~/.config/jarvis/config.json   # then edit it

./run.sh
```

The server listens on `127.0.0.1:8765` and logs a URL containing an access token
(generated once into `~/.config/jarvis/token`). Open that URL in a browser, press the
call button and talk. Use headphones, or the assistant will hear itself.

### From a phone

Browsers only allow the microphone over HTTPS. The simplest private way is Tailscale:

```sh
tailscale serve --bg 8765
```

Then open `https://<your-machine>.<tailnet>.ts.net/?k=<token>` on the phone. Do not
expose the port to the public internet: whoever has the URL can message your agents.

## Configuration

Everything personal lives in `~/.config/jarvis/`, never in the repo:

| File | What |
|---|---|
| `env` | `GEMINI_API_KEY` |
| `token` | access token for the page URL |
| `config.json` | your name, the assistant's name, languages, voice, herdr machines, project glossary |
| `memory.json` | notes the assistant saved for itself |

Environment overrides: `JARVIS_MODEL`, `JARVIS_DEEP_MODEL`, `JARVIS_THINKING`, `JARVIS_PORT`.

## Safety notes

- Agent output, session transcripts and web pages go into the model's context. The prompt
  tells it to treat them as data, and the server only lets it message an agent right
  after the caller's own speech, but this is mitigation, not a guarantee. Only list
  machines in `machines` whose agents you trust.
- The researcher runs in `research/`, whose `opencode.json` denies shell, edits and
  access outside that folder.
- Call logs (`logs/`) contain transcripts of what you and your agents said.
