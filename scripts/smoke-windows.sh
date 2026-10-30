#!/bin/sh
set -u
Xvfb :99 -screen 0 1024x768x24 -nolisten tcp >/dev/null 2>&1 &
export DISPLAY=:99
wine /app/LocalRAG.exe --serve 8765 2>&1 | grep -v fixme &
TRIES=${SMOKE_TRIES:-150}
rc=1
i=0
while [ "$i" -lt "$TRIES" ]; do
  if body=$(curl -sf http://127.0.0.1:8765/health); then
    printf '%s\n\nSMOKE_OK\n' "$body"
    rc=0
    break
  fi
  i=$((i + 1))
  sleep 2
done
[ "$rc" -eq 0 ] || echo SMOKE_FAIL
wineserver -k
exit "$rc"
