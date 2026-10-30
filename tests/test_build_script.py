import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX sh script, run on the Docker host")

ROOT = Path(__file__).resolve().parent.parent
GOOD_LOG = "Successfully installed x\nBuilding EXE completed\n"

FAKE_DOCKER = """#!/bin/sh
if [ "$1" = build ]; then
  printf '%s' "$FAKE_BUILD_LOG"
  if [ -n "${FAKE_BUILD_EXE:-}" ]; then
    while [ $# -gt 0 ]; do
      if [ "$1" = --output ]; then mkdir -p "$2" && echo exe > "$2/LocalRAG.exe"; fi
      shift
    done
  fi
else
  printf '%s\\n' "$FAKE_RUN_OUT"
fi
"""


def _fake(bin_dir, name, body):
    path = bin_dir / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)


@pytest.fixture
def repo(tmp_path):
    shutil.copytree(ROOT / "scripts", tmp_path / "scripts")
    for name in ("Dockerfile.windows", "pyproject.toml", "uv.lock", "localrag.spec"):
        shutil.copy(ROOT / name, tmp_path / name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake(bin_dir, "docker", FAKE_DOCKER.split("\n", 1)[1])
    return tmp_path


def _build(repo, log, exe=False, run_out=""):
    env = dict(os.environ, PATH=f"{repo / 'bin'}:{os.environ['PATH']}",
               FAKE_BUILD_LOG=log, FAKE_BUILD_EXE="1" if exe else "", FAKE_RUN_OUT=run_out)
    return subprocess.run(["sh", str(repo / "scripts" / "build-windows.sh")],
                          capture_output=True, text=True, env=env)


def test_tr2_build_script_fails_on_rosetta_error(repo):
    r = _build(repo, "rosetta error: invalid gdt selector index 5\n", exe=True)
    assert r.returncode == 1 and "BUILD_FAIL rosetta" in r.stdout


def test_tr2_build_script_fails_when_pip_did_not_finish(repo):
    r = _build(repo, "Collecting x\n", exe=True)
    assert r.returncode == 1 and "pip did not finish" in r.stdout


def test_tr2_build_script_fails_on_missing_hidden_import(repo):
    r = _build(repo, "Successfully installed x\n11961 ERROR: Hidden import 'webview' not found\n", exe=True)
    assert r.returncode == 1 and "Hidden import" in r.stdout


def test_tr2_build_script_fails_when_smoke_fails(repo):
    r = _build(repo, GOOD_LOG, exe=True, run_out="SMOKE_FAIL")
    assert r.returncode == 1 and "SMOKE_FAIL" in r.stdout
    assert not (repo / "dist" / "LocalRAG-windows-x64.zip").exists()


def test_tr2_build_script_zips_the_exe_after_smoke_ok(repo):
    r = _build(repo, GOOD_LOG, exe=True, run_out="SMOKE_OK")
    assert r.returncode == 0 and "BUILD_OK" in r.stdout, r.stdout + r.stderr
    assert zipfile.ZipFile(repo / "dist" / "LocalRAG-windows-x64.zip").namelist() == ["LocalRAG.exe"]


def _smoke(tmp_path, curl_rc, tries=None):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake(bin_dir, "Xvfb", "exit 0\n")
    _fake(bin_dir, "wineserver", "exit 0\n")
    _fake(bin_dir, "wine", "sleep 1\n")
    _fake(bin_dir, "curl", f"echo '{{\"ok\":true}}'\nexit {curl_rc}\n")
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    if tries:
        env["SMOKE_TRIES"] = tries
    return subprocess.run(["sh", str(ROOT / "scripts" / "smoke-windows.sh")],
                          capture_output=True, text=True, env=env)


def test_tr2_smoke_script_polls_health_and_reports(tmp_path):
    r = _smoke(tmp_path, 0)
    assert r.returncode == 0 and '{"ok":true}' in r.stdout and "SMOKE_OK" in r.stdout


def test_tr2_smoke_script_reports_fail_when_health_never_answers(tmp_path):
    r = _smoke(tmp_path, 1, tries="1")
    assert r.returncode == 1 and "SMOKE_FAIL" in r.stdout
