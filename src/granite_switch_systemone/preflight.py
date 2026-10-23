# SPDX-License-Identifier: Apache-2.0
"""Checks to run before `vllm serve` starts; `serve.sh` calls this module when
SYSTEMONE_SPEECH_INPUT=1 (with speech input off the speech model is never loaded).

Granite Switch's audio processor loads a speech model while the engine profiles memory. The 8b
and 30b checkpoints enable it (`asr_enabled`) without naming a model, so granite-switch falls
back to a default one. The container runs with the Hub offline, so a speech model that is not in
the cache stops the engine deep in startup with a Hub lookup error that does not say what to do.
This check fails earlier and names the cause and the fixes.

    python3 -m granite_switch_systemone.preflight /repository
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

# What vLLM's `audio` extra needs to decode an audio request. Without them the server starts and
# passes /health, then answers every audio request with HTTP 500 "install vllm[audio]".
AUDIO_MODULES = ("av", "scipy", "soundfile", "soxr")
# Test hook: a comma-separated list of the modules to treat as installed (an empty value checks nothing).
_AUDIO_MODULES_ENV = "SYSTEMONE_PREFLIGHT_AUDIO_MODULES"

# granite-switch 0.1.1's DEFAULT_ASR_MODEL_ID (granite_switch/vllm/audio/asr.py), used when the
# checkpoint's `asr_model_id` is null. tests/test_preflight.py compares it with the installed copy.
DEFAULT_ASR_MODEL_ID = "ibm-granite/granite-speech-5.0-470m-turboctc"

_TRUE = {"1", "ON", "YES", "TRUE"}


def _offline(env: Mapping[str, str]) -> bool:
    return any(env.get(name, "").strip().upper() in _TRUE for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))


def hub_cache_dir(env: Mapping[str, str]) -> Path:
    """Where huggingface_hub looks for cached models, by its own precedence."""
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if env.get(name):
            return Path(env[name])
    home = Path(env["HF_HOME"]) if env.get("HF_HOME") else Path.home() / ".cache" / "huggingface"
    return home / "hub"


def _cached(model_id: str, cache: Path) -> bool:
    if Path(model_id).is_dir():  # a local path, as `asr_model_id` may be
        return True
    snapshots = cache / f"models--{model_id.replace('/', '--')}" / "snapshots"
    return snapshots.is_dir() and any((snap / "config.json").is_file() for snap in snapshots.iterdir())


def missing_asr_model(model_dir: Path, env: Mapping[str, str]) -> str | None:
    """The speech model the checkpoint needs but the offline cache lacks, or None."""
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return None  # not a local checkpoint directory; vLLM reports that itself
    config = json.loads(config_path.read_text())
    if not config.get("asr_enabled") or not _offline(env):
        return None  # speech input off, or online so the model downloads on first use
    model_id = config.get("asr_model_id") or DEFAULT_ASR_MODEL_ID
    return None if _cached(model_id, hub_cache_dir(env)) else model_id


def missing_audio_modules(model_dir: Path) -> list[str]:
    """The audio libraries a speech-enabled checkpoint needs but this environment lacks."""
    config_path = model_dir / "config.json"
    if not config_path.is_file() or not json.loads(config_path.read_text()).get("asr_enabled"):
        return []  # not a local checkpoint, or speech input is off in it
    required = AUDIO_MODULES
    if _AUDIO_MODULES_ENV in os.environ:
        required = tuple(n for n in os.environ[_AUDIO_MODULES_ENV].split(",") if n)
    return [name for name in required if importlib.util.find_spec(name) is None]


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python3 -m granite_switch_systemone.preflight MODEL_DIR", file=sys.stderr)
        return 2
    modules = missing_audio_modules(Path(argv[0]))
    if modules:
        print(
            "granite-switch-systemone: speech input is on, but these audio libraries are not installed: "
            f"{', '.join(modules)}.\n"
            "Without them the server starts and passes /health, then answers every audio request with "
            'HTTP 500 "Please install vllm[audio]". Install vllm[audio] (or granite-switch[audio]) into '
            "this environment, or use a release image, which includes them. Leave SYSTEMONE_SPEECH_INPUT "
            "unset or 0 if you do not need audio.",
            file=sys.stderr,
        )
        return 1
    model_id = missing_asr_model(Path(argv[0]), os.environ)
    if model_id is None:
        return 0
    print(
        f"granite-switch-systemone: this checkpoint enables speech input (asr_enabled in config.json) and needs\n"
        f'the speech model "{model_id}", but the Hub is offline and it is not in the cache at\n'
        f"{hub_cache_dir(os.environ)}. Do one of:\n"
        "  - use an image built with --build-arg WITH_ASR=1, which bakes the model in\n"
        "  - leave speech input off (SYSTEMONE_SPEECH_INPUT unset or 0), which decisions do not need\n"
        "  - mount a cache that holds it and point HF_HUB_CACHE at that directory\n"
        "  - set HF_HUB_OFFLINE=0 so the model can be downloaded at startup",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
