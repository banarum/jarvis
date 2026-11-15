#!/bin/zsh
# Restart the Telegram line safely: never during a call, and never with code that fails the self-test.
cd "$(dirname "$0")" || exit 1
set -a; [ -f ~/.config/jarvis/env ] && . ~/.config/jarvis/env; set +a
mkdir -p logs

live() {
  local last=$(grep -E "answered in|lasted [0-9]+ s" logs/tgline.log 2>/dev/null | tail -1)
  [[ -n "$last" && "$last" != *lasted* ]] && kill -0 "$(cat logs/tgline.pid 2>/dev/null)" 2>/dev/null
}

if live; then echo "REFUSED: a call is live. Try again after it ends."; exit 2; fi

echo "self-test (about a minute)…"
if ! .venv/bin/python tgline.py selftest > logs/selftest.log 2>&1; then
  echo "REFUSED: the self-test failed; the running line was left alone."
  grep -E "SELF-TEST FAILED|Traceback|Error" logs/selftest.log | tail -5
  exit 1
fi
grep -E "^-- " logs/selftest.log | tail -2

if live; then echo "REFUSED: a call started meanwhile. Try again after it ends."; exit 2; fi
[ -f logs/tgline.log ] && cp logs/tgline.log "logs/tgline-$(date +%F-%H%M%S).log"
[ -f logs/tgline.pid ] && kill "$(cat logs/tgline.pid)" 2>/dev/null && sleep 3
nohup .venv/bin/python tgline.py > logs/tgline.log 2>&1 &
echo $! > logs/tgline.pid
for i in {1..40}; do
  sleep 1
  grep -q "on the line as" logs/tgline.log && { echo "line is up (pid $(cat logs/tgline.pid))"; exit 0; }
  kill -0 "$(cat logs/tgline.pid)" 2>/dev/null || break
done
echo "FAILED: the line did not come up:"; tail -5 logs/tgline.log; exit 1
