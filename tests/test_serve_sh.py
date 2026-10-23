# SPDX-License-Identifier: Apache-2.0
"""No-GPU checks for serve.sh's speech-input switch, with a stub `vllm` that records its arguments.

python3 -m pytest tests/test_serve_sh.py -v --tb=short
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SERVE_SH = Path(__file__).resolve().parent.parent / "serve.sh"
OVERRIDE = '{"asr_enabled": false}'


def _run(tmp_path: Path, extra_env: dict[str, str], *args: str, asr_enabled: bool = True):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "vllm"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$STUB_OUT"\n')
    stub.chmod(0o755)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"asr_enabled": asr_enabled}))
    out = tmp_path / "args.txt"
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        "MODEL_DIR": str(model_dir),
        "TENSOR_PARALLEL_SIZE": "1",
        "STUB_OUT": str(out),
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_CACHE": str(tmp_path / "empty-cache"),
        "SYSTEMONE_PREFLIGHT_AUDIO_MODULES": "",  # these tests are about the speech model, not the audio libraries
        **extra_env,
    }
    done = subprocess.run(["bash", str(SERVE_SH), *args], env=env, capture_output=True, text=True)
    return done, (out.read_text().splitlines() if out.exists() else [])


def test_speech_input_is_off_by_default_and_needs_no_speech_model(tmp_path):
    done, argv = _run(tmp_path, {})
    assert done.returncode == 0, done.stderr
    assert argv[argv.index("--hf-overrides") + 1] == OVERRIDE


def test_speech_input_on_without_the_model_stops_before_vllm(tmp_path):
    done, argv = _run(tmp_path, {"SYSTEMONE_SPEECH_INPUT": "1"})
    assert done.returncode == 1
    assert argv == []  # vllm never started
    assert "WITH_ASR=1" in done.stderr


def test_speech_input_on_with_the_model_cached_starts_without_the_override(tmp_path):
    cache = tmp_path / "empty-cache" / "models--ibm-granite--granite-speech-5.0-470m-turboctc" / "snapshots" / "abc"
    cache.mkdir(parents=True)
    (cache / "config.json").write_text("{}")
    done, argv = _run(tmp_path, {"SYSTEMONE_SPEECH_INPUT": "1"})
    assert done.returncode == 0, done.stderr
    assert "--hf-overrides" not in argv


def test_speech_input_on_without_the_audio_libraries_stops_before_vllm(tmp_path):
    done, argv = _run(
        tmp_path, {"SYSTEMONE_SPEECH_INPUT": "1", "SYSTEMONE_PREFLIGHT_AUDIO_MODULES": "no_such_audio_module_xyz"}
    )
    assert done.returncode == 1
    assert argv == []  # vllm never started
    assert "no_such_audio_module_xyz" in done.stderr
    assert "HTTP 500" in done.stderr


def test_extra_arguments_still_reach_vllm(tmp_path):
    done, argv = _run(tmp_path, {}, "--api-key", "k")
    assert done.returncode == 0, done.stderr
    assert argv[-2:] == ["--api-key", "k"]


@pytest.mark.parametrize("value", ["2", "yes", ""])
def test_bad_switch_value_is_refused(tmp_path, value):
    done, argv = _run(tmp_path, {"SYSTEMONE_SPEECH_INPUT": value})
    if value == "":
        assert done.returncode == 0  # empty falls back to the default, off
    else:
        assert done.returncode == 2
        assert argv == []
