#!/bin/sh
set -u
cd "$(dirname "$0")/.."
rm -rf dist/windows-x64
mkdir -p dist
docker build -f Dockerfile.windows --output dist/windows-x64 . > dist/build-windows.log 2>&1
# The Dockerfile cannot see a Wine crash (Rosetta), so the log is the only proof the build worked.
log=dist/build-windows.log
line=$(grep -m1 'rosetta error' "$log") && { echo "BUILD_FAIL rosetta: $line"; exit 1; }
grep -q 'Successfully installed' "$log" || { echo "BUILD_FAIL pip did not finish"; exit 1; }
line=$(grep -m1 'ERROR: Hidden import' "$log") && { echo "BUILD_FAIL $line"; exit 1; }
[ -f dist/windows-x64/LocalRAG.exe ] || { echo "BUILD_FAIL LocalRAG.exe is missing"; exit 1; }
docker run --rm --platform linux/amd64 -v "$PWD/dist/windows-x64:/app:ro" -v "$PWD/scripts/smoke-windows.sh:/smoke.sh:ro" tobix/pywine:3.12 sh /smoke.sh > dist/smoke-windows.log 2>&1
grep -q SMOKE_OK dist/smoke-windows.log || { echo SMOKE_FAIL; exit 1; }
(cd dist/windows-x64 && uv run --project ../.. python -m zipfile -c ../LocalRAG-windows-x64.zip LocalRAG.exe) || exit 1
echo BUILD_OK
