# Testing

Results of each check are in [STATUS.md](STATUS.md). This page says what each test
covers and how to run it.

## Automated (CI, every push and pull request)

```bash
pip install -e ".[dev]"
ruff format --check src tests scripts ci examples && ruff check src tests scripts ci examples
ty check src
python3 ci/check_headers.py
pytest
```

CI also builds the wheel and sdist (`python -m build`), checks that `LICENSE` and
`THIRD_PARTY_NOTICES.md` ship inside the wheel, installs the wheel into a clean
environment and runs the same suite against the installed package, and runs
[gitleaks](https://github.com/gitleaks/gitleaks) over the full history of all refs
(the same scanner the pre-commit hook runs on staged files).

**What CI deliberately does not cover:** anything needing a GPU, a checkpoint or a
live server. Those run by hand (the tokenizer test, the live-server batteries and the load
probes below) because a CI runner cannot reproduce the model deployments they need.
The commands are in the two Manual sections that follow, and results are recorded in
[STATUS.md](STATUS.md). Locally, `pre-commit run --all-files` (after
`pre-commit install`) runs the same lint, header, DCO and secret checks the CI runs;
it needs `gitleaks` on `PATH` for the last of those.

- `tests/test_plugin.py`: request validation, body limits, the context guard, chunked
  label reads and their usage counts, adapter-name resolution, the concurrency limiter,
  disconnect cancellation, startup checks.
- `tests/test_decisions.py`: the `/v1/decisions` wire format with the engine stubbed,
  including parity with `/v1/systemone` (both routes must build the same internal
  question) and error shapes.
- `tests/test_openai_spec.py`: `/v1/decisions` requests and responses validated with a
  JSON Schema validator against OpenAI's published schema. The schema is downloaded on
  first run from `openai/openai-openapi` (MIT) at a pinned commit, checked against a
  pinned SHA-256 and cached in `~/.cache/granite-switch-systemone`; it is never committed.
  Offline, the test skips and says how to supply a local copy (`SYSTEMONE_OPENAI_SPEC`).
  Deliberate limits are listed as named exceptions.
- `tests/test_systemone_spec.py`: `/v1/systemone` validated against jev's wire models from
  TypeSafe's MIT-licensed `typesafe-sdk` (a dev dependency), which TypeSafe generates
  from its API spec. Those models omit the spec's bounds on choice options (1-255) and
  score levels (at most 10), which are checked against the plugin's own limits.
  Without `typesafe-sdk` the test skips and says what to install.
- `tests/test_renderer.py`: prompt layout, label pool, special-token and marker
  escaping, the experimental system prompt switch.
- `tests/test_plugin.py` also covers the startup activation probe and the label-mass log.

## Manual: needs the checkpoint's tokenizer, no GPU

```bash
pip install "transformers>=5.16"
SYSTEMONE_TOKENIZER=ibm-granite/granite-switch-4.1-8b pytest tests/test_prefix_sharing.py
```

`tests/test_prefix_sharing.py` renders the real prompt with the checkpoint's tokenizer
and chat template and counts which 16-token prefix-cache blocks a decision can share with
other requests. It also checks that the startup activation probe accepts the real chat
template and refuses a wrong marker. It is skipped in CI because it downloads the tokenizer.

## Manual: needs a running server

Start the server (`serve.sh`, or the commands in the README) on a machine with no other
traffic: several checks read vLLM's server-wide `/metrics`. All scripts use only the
Python standard library unless noted.

| Command | What it checks |
|---|---|
| `python3 scripts/e2e_check.py --base-url URL` | Both routes' answer shapes and parity, input flattening, boolean values, error shapes, usage, chat on the same server, prefix-cache hits. Add `--system-prompt` when the server runs with `SYSTEMONE_EXPERIMENTAL_SYSTEM_PROMPT=1`. |
| `python3 scripts/probe_check.py --base-url URL` | Question cap, size guard (no engine work for oversized requests), body limits, special-token and marker escaping, options without descriptions, disconnect cancellation. |
| `python3 scripts/probe_check.py --base-url URL --max-concurrent N` | Adds the concurrency limit: a burst against `SYSTEMONE_MAX_CONCURRENT_REQUESTS=N` gets 200s and 503s. |
| `python3 scripts/probe_check.py --base-url URL --load 30` | Adds a load test at 1, 8 and 32 clients with chat latency alongside. Reported, not asserted. |
| `python3 scripts/longctx_check.py --base-url URL --tokenizer /path/to/granite-switch-4.1-8b` | Long-context caps, on a server started with `--max-model-len 131072` and the 8b: the danger-zone header path (a ~98k-token request answered with the header, rejected without it), the 131,072 backstop, the default 32k/64k caps, and that `/v1/decisions` has no header escape. Unlike the scripts above it needs the checkpoint's tokenizer (transformers). |
| `python3 scripts/accuracy_check.py --base-url URL` | Accuracy, log-loss and Brier score on 200 BoolQ validation rows (public data, downloaded when it runs, needs internet). Fails below 75% accuracy, which catches a wrong invocation marker. To check an upgrade or a changed setting, run it on each deployment with `--save`, then `--compare` the second against the first: it reports items only one run got right and an exact McNemar p-value. |
| `python3 scripts/speech_check.py --base-url URL` | Speech input, on a server started with `SYSTEMONE_SPEECH_INPUT=1` (see the README). Sends spoken clips to chat completions as `input_audio` and checks that the reply repeats at least 80% of the words. Makes its own clips with `espeak-ng` or macOS `say`, so no audio is stored here; `--clip FILE.wav "the words"` uses your own. Exits 2 if it has no clips. A server missing the audio libraries passes `/health` and fails here with HTTP 500. |
| `python3 scripts/load_test.py --base-url URL` | Throughput and latency of `/v1/decisions` at 1, 8, 32, 64 and 128 clients, with the server's prefix-cache hit rate. `--mode unique` (default) gives every request a fresh state; `--mode identical` repeats one body and flatters throughput. Run both and report both. See below. |
| `SYSTEMONE_BASE_URL=URL pytest tests/test_decisions_e2e.py` | `/v1/decisions` and `/v1/systemone` return the same answers for equivalent requests. |

### A short deployment check

On a fresh server, in this order (about five minutes after the model has loaded):

```bash
python3 scripts/e2e_check.py --base-url URL      # wire formats, chat on the same server, prefix cache
python3 scripts/probe_check.py --base-url URL    # limits and escaping
python3 scripts/accuracy_check.py --base-url URL # answers are still good
```

Before upgrading vLLM, granite-switch or this plugin, run `accuracy_check.py --save before.json`
on the old deployment and `--compare before.json` on the new one.

### Load testing

`scripts/load_test.py` is a standard-library client. Every client loops on one keep-alive connection and
sends three questions (a predicate, a 20-way choice, a 5-level score) over a state of about 230 tokens.

```bash
python3 scripts/load_test.py --base-url http://HOST:PORT                       # unique state per request
python3 scripts/load_test.py --base-url http://HOST:PORT --mode identical      # one body, cache-flattered
python3 scripts/load_test.py --base-url http://HOST:PORT --levels 1,8,32 --seconds 30
```

How to read it:

- Run the client on a different machine from the server, and run it on a server with no other traffic.
  The prefix-cache hit rate comes from vLLM's `/metrics` and counts all traffic.
- `identical` is an upper bound. The server answers almost every token from its prefix cache, so its
  throughput is not what real, varied states will see. Report `unique` as the realistic figure.
- Throughput levels off once the GPU is saturated; past that point more clients only add latency.
  Where it levels off depends on the GPU, the model and the length of the state, so measure on your
  hardware instead of reusing a figure from another deployment.
- With `SYSTEMONE_MAX_CONCURRENT_REQUESTS` set, requests above the cap get HTTP 503 and are not queued.
  The client resends at once by default, which makes a capped server look worse than a caller that backs
  off would see. Pass `--backoff SECONDS` to wait after a non-200. The cap protects chat latency on a
  shared server; it does not raise decision throughput.
- `--max-num-seqs` is not worth tuning for decisions: they are prefill-only, and vLLM's default was not the
  limit in our runs.
- Each level is one run of 45 seconds by default, so there is no measure of spread. Repeat a level before
  drawing conclusions from a difference of a few per cent.

### Startup checks and logs

- The server refuses to start if a probe decision prompt does not carry exactly one adapter
  control token inside the question turn. The usual cause is a wrong
  `SYSTEMONE_INVOCATION_MARKER` or `SYSTEMONE_ADAPTER_NAME`; the error names both. The probe
  logs nothing when it passes.
- `serve.sh` switches the checkpoint's speech input off by default, which decisions do not use, so
  the 8b and 30b start without a speech model. With `SYSTEMONE_SPEECH_INPUT=1` it first runs
  `python3 -m granite_switch_systemone.preflight`: if the Hub is offline and the speech model is not
  in the cache, it stops with a message naming the model and the ways to fix it, instead of failing
  later inside the engine. The image sets `HF_HUB_OFFLINE=1`.
- Each question logs the probability mass the option labels hold. Below 0.9 it logs a warning
  (a wrong layout or marker makes the model want a token outside the labels); otherwise it
  logs at debug level. Answers are never refused on it.

## Optional: a benchmark run with the Decision Index

The [Decision Index](https://github.com/apolinario/decision-index) is a public benchmark for
typed decision engines. Its `http` engine speaks the `/v1/systemone` wire format, so it can
drive this plugin as it is. Use it as a broad check that a deployment answers well across
many kinds of question, or to compare two configurations of the same server (for example the
system prompt on and off) on the same rows. Nothing from the benchmark is stored in this
repository: you fetch the kit, build the suite yourself, and run it against your server.

What you need:

- A running server (see [Quick start](../README.md#quick-start)). Start it
  with `--max-model-len 131072` if you want the longest rows answered instead of rejected.
- The kit, which is MIT licensed. Install it and build the suite by following its README
  ("Quickstart (local)"). The suite is not redistributed, so you rebuild it from public sources
  (about 7 GB of downloads and 17 GB of working space), and you need a Hugging Face login with
  the terms of the gated HLE dataset accepted. Check each source's own licence before reusing
  any of the data.
- Time. A sample is quick, a full run is not.

Run a sample, then score it (commands as in the kit's README; check them against the version
you installed):

```bash
python -m decision_index suite sample --n 1000 --out sample.jsonl.gz
python -m decision_index run --engine http \
    --option base_url=http://localhost:8000 --option model=granite-switch \
    --rows sample.jsonl.gz --out runs/my-server
```

To compare two servers, run the same `sample.jsonl.gz` against each and compare the scored
results row by row. A sample is a small fraction of the suite, so differences of a point or
so between two runs are within sampling noise unless you pair them on the same rows.

Do not read a headline number from `decision_index score` on a sample. It divides by the full
suite's request counts, so a sampled run comes out far too low (a sample that answered every
row can score under 5). To get a number comparable with a full run, subclass the kit's `Suite`
so that `rows()` yields only the sampled rows, and pass that to the kit's `score_run`. Compare
two runs of the same sample with each other; do not compare a sample's score with a published
full-run score without allowing for the sampling noise.

Treat the scores as a validation signal for your own deployment. The model card's published
figures are the reference for the model itself, and a score from this plugin does not by
itself show that the serving path matches the one that produced them.

## Third-party schemas

Nothing in this repository is copied from another project's API specification. See
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for what is used and under which
licence. TypeSafe's own `openapi.json` carries no licence and is not used.

## Not in this repository

`scripts/accuracy_check.py` is a small public version of the accuracy comparison (BoolQ
only). The larger comparison on labelled public datasets (BoolQ, SST-5, Banking77): the
`/v1/decisions` route against the same prompt sent over `/v1/chat/completions`, against
the layout with the system prompt, and FlashAttention 2 against 3. That harness and its
results are not published here; STATUS.md says what was compared. The Decision Index
reproduction (kit, suite build, runs and scores) likewise lives outside this repository.

The Decision Index runs described in STATUS.md use the public reproduction kit
(`apolinario/decision-index`, MIT) driven against `/v1/systemone`; see the Decision Index
section above for how to run it yourself.
