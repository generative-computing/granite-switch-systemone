# SPDX-License-Identifier: Apache-2.0
"""No-GPU checks for preflight.py, the speech-model check `serve.sh` runs before `vllm serve`.

python3 -m pytest tests/test_preflight.py -v --tb=short
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

from granite_switch_systemone import preflight

DEFAULT = preflight.DEFAULT_ASR_MODEL_ID


def _checkpoint(tmp_path: Path, **config) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config))
    return model_dir


def _cache_model(cache: Path, model_id: str, with_config: bool = True) -> None:
    snap = cache / f"models--{model_id.replace('/', '--')}" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    if with_config:
        (snap / "config.json").write_text("{}")


def _offline(cache: Path) -> dict[str, str]:
    return {"HF_HUB_OFFLINE": "1", "HF_HUB_CACHE": str(cache)}


def test_no_config_json_is_not_checked(tmp_path):
    assert preflight.missing_asr_model(tmp_path, _offline(tmp_path / "cache")) is None


def test_speech_input_disabled_needs_nothing(tmp_path):
    model_dir = _checkpoint(tmp_path, asr_enabled=False, asr_model_id=None)
    assert preflight.missing_asr_model(model_dir, _offline(tmp_path / "cache")) is None


def test_online_will_download_so_nothing_is_missing(tmp_path):
    model_dir = _checkpoint(tmp_path, asr_enabled=True, asr_model_id=None)
    assert preflight.missing_asr_model(model_dir, {"HF_HUB_CACHE": str(tmp_path / "cache")}) is None


def test_offline_with_default_model_missing_names_the_default(tmp_path):
    model_dir = _checkpoint(tmp_path, asr_enabled=True, asr_model_id=None)
    assert preflight.missing_asr_model(model_dir, _offline(tmp_path / "cache")) == DEFAULT


def test_offline_with_named_model_missing_names_that_model(tmp_path):
    model_dir = _checkpoint(tmp_path, asr_enabled=True, asr_model_id="acme/speech-1")
    assert preflight.missing_asr_model(model_dir, _offline(tmp_path / "cache")) == "acme/speech-1"


def test_offline_with_model_cached_passes(tmp_path):
    cache = tmp_path / "cache"
    _cache_model(cache, DEFAULT)
    model_dir = _checkpoint(tmp_path, asr_enabled=True, asr_model_id=None)
    assert preflight.missing_asr_model(model_dir, _offline(cache)) is None


def test_cache_entry_without_a_snapshot_config_counts_as_missing(tmp_path):
    cache = tmp_path / "cache"
    _cache_model(cache, DEFAULT, with_config=False)
    model_dir = _checkpoint(tmp_path, asr_enabled=True, asr_model_id=None)
    assert preflight.missing_asr_model(model_dir, _offline(cache)) == DEFAULT


def test_local_path_model_passes(tmp_path):
    local = tmp_path / "speech"
    local.mkdir()
    model_dir = _checkpoint(tmp_path, asr_enabled=True, asr_model_id=str(local))
    assert preflight.missing_asr_model(model_dir, _offline(tmp_path / "cache")) is None


def test_transformers_offline_also_counts_as_offline(tmp_path):
    model_dir = _checkpoint(tmp_path, asr_enabled=True, asr_model_id=None)
    env = {"TRANSFORMERS_OFFLINE": "true", "HF_HUB_CACHE": str(tmp_path / "cache")}
    assert preflight.missing_asr_model(model_dir, env) == DEFAULT


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"HF_HUB_CACHE": "/a", "HF_HOME": "/b"}, Path("/a")),
        ({"HUGGINGFACE_HUB_CACHE": "/c", "HF_HOME": "/b"}, Path("/c")),
        ({"HF_HOME": "/b"}, Path("/b/hub")),
    ],
)
def test_hub_cache_dir_follows_huggingface_hub_precedence(env, expected):
    assert preflight.hub_cache_dir(env) == expected


def test_main_exits_1_and_says_what_to_do(tmp_path, monkeypatch, capsys):
    model_dir = _checkpoint(tmp_path, asr_enabled=True, asr_model_id=None)
    monkeypatch.setenv("SYSTEMONE_PREFLIGHT_AUDIO_MODULES", "")  # this test is about the speech model
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "cache"))
    assert preflight.main([str(model_dir)]) == 1
    err = capsys.readouterr().err
    assert DEFAULT in err
    assert str(tmp_path / "cache") in err
    assert "WITH_ASR=1" in err
    assert "SYSTEMONE_SPEECH_INPUT" in err
    assert "HF_HUB_OFFLINE=0" in err


def test_audio_modules_missing_are_named_when_speech_is_on(tmp_path, monkeypatch):
    model_dir = _checkpoint(tmp_path, asr_enabled=True)
    monkeypatch.setattr(
        preflight.importlib.util, "find_spec", lambda name: None if name in {"av", "soxr"} else object()
    )
    assert preflight.missing_audio_modules(model_dir) == ["av", "soxr"]


def test_audio_modules_are_not_checked_when_the_checkpoint_has_speech_off(tmp_path, monkeypatch):
    model_dir = _checkpoint(tmp_path, asr_enabled=False)
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda name: None)
    assert preflight.missing_audio_modules(model_dir) == []


def test_audio_modules_are_not_checked_without_a_local_config(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda name: None)
    assert preflight.missing_audio_modules(tmp_path) == []


def test_main_exits_1_and_names_the_missing_audio_libraries(tmp_path, monkeypatch, capsys):
    model_dir = _checkpoint(tmp_path, asr_enabled=True, asr_model_id=None)
    cache = tmp_path / "cache"
    _cache_model(cache, DEFAULT)  # the speech model is fine; only the libraries are missing
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda name: None if name == "soundfile" else object())
    assert preflight.main([str(model_dir)]) == 1
    err = capsys.readouterr().err
    assert "soundfile" in err
    assert "HTTP 500" in err
    assert "vllm[audio]" in err


def test_main_exits_0_quietly_when_nothing_is_missing(tmp_path, monkeypatch, capsys):
    model_dir = _checkpoint(tmp_path, asr_enabled=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert preflight.main([str(model_dir)]) == 0
    assert capsys.readouterr().err == ""


def test_main_rejects_wrong_arguments(capsys):
    assert preflight.main([]) == 2
    assert "usage" in capsys.readouterr().err


def test_default_matches_the_installed_granite_switch():
    """The constant is copied from granite-switch; fail if the installed copy has moved on."""
    spec = importlib.util.find_spec("granite_switch")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("granite-switch is not installed")
    asr = Path(next(iter(spec.submodule_search_locations))) / "vllm" / "audio" / "asr.py"
    if not asr.is_file():
        pytest.skip("this granite-switch has no vllm/audio/asr.py")
    found = re.search(r'^DEFAULT_ASR_MODEL_ID\s*=\s*"([^"]+)"', asr.read_text(), re.MULTILINE)
    assert found, "DEFAULT_ASR_MODEL_ID not found in granite-switch's asr.py"
    assert found.group(1) == DEFAULT
