#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Speech-input smoke test for a live granite-switch-systemone server started with SYSTEMONE_SPEECH_INPUT=1.

Standard library only. Sends short spoken clips to /v1/chat/completions as `input_audio` and checks
that the model heard them: the reply has to repeat most of the words that were spoken. Decisions take
text only, so this checks the chat route the speech model feeds.

  python3 scripts/speech_check.py --base-url http://localhost:8000
  python3 scripts/speech_check.py --base-url http://localhost:8000 --clip my.wav "the words in it"

Without --clip it makes its own clips at run time with espeak-ng (any platform) or `say` (macOS), so
no audio is stored in this repository. If neither is installed it says so and exits 2; pass --clip
with your own 16-bit WAV file instead. Exit 0 means every check passed.

Checks: a text-only chat request still works; each clip gets HTTP 200 and a reply that repeats at
least --min-overlap of the spoken words; an audio request with no speech model behind it is reported
with its server message (the usual cause is missing `vllm[audio]` libraries: the server passes
/health, then every audio request fails with HTTP 500).
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

# Plain English sentences: the check is that the model heard them, so they avoid names and numbers.
BUILTIN = (
    "the quick brown fox jumps over the lazy dog",
    "please send the invoice to the finance team tomorrow morning",
)
INSTRUCTION = "Repeat the words you heard in the audio exactly, and output nothing else."


def post(base_url: str, body: dict, timeout: float = 180) -> tuple[int, dict | str]:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:400]


def words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def overlap(expected: str, reply: str) -> float:
    want, got = words(expected), set(words(reply))
    return sum(w in got for w in want) / len(want) if want else 0.0


def synthesize(text: str, out: Path) -> bool:
    """Write a 16 kHz mono WAV of `text`; False if no speech synthesizer is installed."""
    if shutil.which("espeak-ng"):
        subprocess.run(["espeak-ng", "-w", str(out), text], check=True, capture_output=True)
        return True
    if shutil.which("say"):
        subprocess.run(["say", "--data-format=LEI16@16000", "-o", str(out), text], check=True, capture_output=True)
        return True
    return False


def ask_audio(base_url: str, wav: Path, model: str) -> tuple[int, str]:
    audio = base64.b64encode(wav.read_bytes()).decode()
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 120,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}},
                    {"type": "text", "text": INSTRUCTION},
                ],
            }
        ],
    }
    status, data = post(base_url, body)
    if status == 200 and isinstance(data, dict):
        return status, data["choices"][0]["message"]["content"] or ""
    return status, str(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="granite-switch")
    ap.add_argument(
        "--clip",
        nargs=2,
        action="append",
        metavar=("WAV", "WORDS"),
        default=[],
        help="your own clip and the words in it (repeatable)",
    )
    ap.add_argument(
        "--min-overlap", type=float, default=0.8, help="share of spoken words the reply must repeat (default 0.8)"
    )
    args = ap.parse_args()

    failures = 0

    def report(ok: bool, name: str, detail: str = "") -> None:
        nonlocal failures
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))

    status, data = post(
        args.base_url,
        {
            "model": args.model,
            "max_tokens": 8,
            "temperature": 0,
            "messages": [{"role": "user", "content": "Say hello."}],
        },
    )
    report(status == 200, "text-only chat still works", f"HTTP {status}")

    with tempfile.TemporaryDirectory() as tmp:
        clips = [(Path(w), words_) for w, words_ in args.clip]
        if not clips:
            for i, text in enumerate(BUILTIN):
                wav = Path(tmp) / f"clip{i}.wav"
                if not synthesize(text, wav):
                    print(
                        "No speech synthesizer found (install espeak-ng, or use macOS `say`), and no --clip given.",
                        file=sys.stderr,
                    )
                    return 2
                clips.append((wav, text))
        for wav, spoken in clips:
            status, reply = ask_audio(args.base_url, wav, args.model)
            if status != 200:
                hint = (
                    " (the audio libraries are probably missing: install vllm[audio])"
                    if "vllm[audio]" in reply
                    else ""
                )
                report(False, f"audio clip {wav.name}", f"HTTP {status}: {reply[:200]}{hint}")
                continue
            share = overlap(spoken, reply)
            report(
                share >= args.min_overlap,
                f"audio clip {wav.name}",
                f"heard {share:.0%} of {len(words(spoken))} words; reply {reply!r}",
            )

    print("speech check:", "FAILED" if failures else "passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
