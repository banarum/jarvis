"""Print an agent session as compact plain text.

Usage: python3 transcript.py <claude|opencode> <session_id>

Standalone (stdlib only) so it can also be piped over ssh to a remote machine:
    ssh host python3 - claude <id> < transcript.py
"""

import glob
import json
import os
import sqlite3
import sys

TOOL_IN, TOOL_OUT, MAX = 200, 300, 400_000
NOISE = ("<local-command", "<command-", "<system-reminder", "Caveat:")


def clip(s, n):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + "…"


def brief_input(inp):
    if not isinstance(inp, dict):
        return clip(inp, TOOL_IN)
    for k in ("command", "file_path", "path", "pattern", "url", "description", "prompt"):
        if k in inp:
            return clip(inp[k], TOOL_IN)
    return clip(json.dumps(inp, ensure_ascii=False), TOOL_IN)


def result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return ""


def claude(sid):
    files = glob.glob(os.path.expanduser(f"~/.claude/projects/*/{sid}.jsonl"))
    if not files:
        sys.exit(f"no Claude Code transcript for {sid}")
    out = []
    for line in open(files[0], encoding="utf-8", errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        role = d.get("type")
        if role not in ("user", "assistant"):
            continue
        content = (d.get("message") or {}).get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for b in content or []:
            t = b.get("type")
            if t == "text":
                text = b.get("text", "").strip()
                if text and not text.startswith(NOISE):
                    out.append(f"[{role}] {text}")
            elif t == "tool_use":
                out.append(f"[tool] {b.get('name')}: {brief_input(b.get('input'))}")
            elif t == "tool_result":
                out.append(f"[result] {clip(result_text(b.get('content')), TOOL_OUT)}")
    return out


def opencode(sid):
    db = sqlite3.connect(f"file:{os.path.expanduser('~/.local/share/opencode/opencode.db')}?mode=ro", uri=True)
    rows = db.execute(
        "select type, data from session_message where session_id=? order by seq", (sid,)
    ).fetchall()
    if not rows:
        sys.exit(f"no opencode session {sid}")
    out = []
    for role, data in rows:
        d = json.loads(data)
        if role == "user":
            out.append(f"[user] {d.get('text', '').strip()}")
            continue
        for c in d.get("content", []):
            t = c.get("type")
            if t == "text" and c.get("text", "").strip():
                out.append(f"[assistant] {c['text'].strip()}")
            elif t == "tool":
                st = c.get("state") or {}
                out.append(f"[tool] {c.get('name')}: {brief_input(st.get('input'))}")
                out.append(f"[result] {clip(result_text(st.get('content')), TOOL_OUT)}")
    return out


if __name__ == "__main__":
    kind, sid = sys.argv[1], sys.argv[2]
    lines = {"claude": claude, "opencode": opencode}[kind](sid)
    text = "\n".join(lines)
    # Keep the end: that's where the current state of the work is.
    sys.stdout.write(text[-MAX:])
