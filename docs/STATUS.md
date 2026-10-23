# Status: what is tested, and what is not

How to run each test is in [TESTING.md](TESTING.md).

Last updated 2026-10-20. "Live" means a real server on H100s: granite-switch 0.1.1,
transformers 5.17, vLLM 0.26.0 / 0.27.1 / 0.28.0 / 0.29.0 / 0.30.0 as noted per row,
and `ibm-granite/granite-switch-4.1-8b` unless a row names another checkpoint.

## Tested

| Area | How | Result |
|---|---|---|
| Unit and route tests | `pytest`, engine stubbed, in CI | pass |
| `/v1/decisions` against OpenAI's OpenAPI schema | `tests/test_openai_spec.py` (42 cases), plus the same request set live | pass, with the exceptions listed in the README |
| `/v1/systemone` against jev's wire format | `tests/test_systemone_spec.py` (24 cases) against `typesafe-sdk`'s generated models, stubbed | pass; the plugin also rejects unknown fields, which jev's format allows |
| Live wire battery | `scripts/e2e_check.py` (11 checks) | 11/11 |
| Prompt rendering | server prompt token counts against an offline render with the checkpoint's tokenizer and chat template | identical |
| Prefix-cache sharing | `tests/test_prefix_sharing.py`, and `cached_tokens` live, cross-checked with `/metrics` | as documented in the README |
| Route parity | 800 labelled items through `/v1/decisions` and through the same prompt over chat completions | no item answers differently |
| Escaping, size guard, disconnect cancellation, error shapes | live probes, now `scripts/probe_check.py` | as documented in the README |
| Question cap and `SYSTEMONE_MAX_CONCURRENT_REQUESTS` | live: 200 questions accepted, 201 rejected; a 32-request burst against a cap of 8 gave 8 accepted and 24 rejected with 503 | pass |
| `SYSTEMONE_EXPERIMENTAL_SYSTEM_PROMPT=1` | live: the 800-item comparison and `e2e_check.py --system-prompt` | answers identical to the same prompt over chat completions; decisions about one state still share the cache, chat no longer does |
| FlashAttention 3 | FA2 and FA3 servers side by side: the 800-item comparison, 40 cached one-token reads (the case in granite-switch#139), load at 8 and 32 clients | no difference beyond batching noise in answers or speed; FA2 kept in `serve.sh` while #139 is open |
| Load | 1 to 128 concurrent clients, three-question requests | about 140 requests/s plateau, no errors |
| Plugin-only install | the README's steps in a fresh environment, then its `vllm serve` command and the e2e battery | pass |
| Install from this repository | `pip install git+https://github.com/generative-computing/granite-switch-systemone` into a fresh venv, over HTTPS and SSH with authenticated access while the repository was private, with the entry-point check and the unit and spec suites against the installed copy from a neutral directory (184 tests, pass); then anonymously (no git config, credentials or SSH agent) once public, at v0.1.2 | the anonymous install resolved and registered the `vllm.endpoint_plugins` entry point; the suites were not rerun on it |
| vLLM 0.26.0, 0.27.1, 0.28.0, 0.29.0 live (at `v0.1.0` only; not repeated for later changes) | one server per version on one H100 (torch 2.11.0+cu130 on 0.26, torch 2.13.0+cu130 on the rest): the e2e battery, `probe_check`, `test_decisions_e2e`, `test_prefix_sharing`, and the unit suite under each version's own venv | 11/11 e2e and every suite passes on all four versions |
| The 30b checkpoint | `granite-switch-4.1-30b` on vLLM 0.30.0, `--tensor-parallel-size 2` on two H100s: the full battery | pass |
| Tensor parallelism | the 8b with `--tensor-parallel-size 2`: the full battery | pass |
| Long contexts | the 8b with `--max-model-len 131072`, `scripts/longctx_check.py` (9 checks) | 9/9: a 98k-token request answered in one engine read with the danger-zone header and rejected without it; the 131,072 backstop and the default 32k/64k caps enforced; `/v1/decisions` has no header escape |
| Docker image build | `docker build --platform linux/amd64`, default and `WITH_ASR=1` variants | both build; the stack inside is package-identical to the live-tested one (PyPI vLLM 0.30.0, torch 2.13.0+cu130, transformers 5.17.0, granite-switch 0.1.1); entry points and imports verified in-container; the `WITH_ASR=1` image holds the pinned speech-model revision as its only snapshot, with `refs/main` pointing at it |
| Serving from the image | the default image (no speech model inside) under enroot (no Docker daemon on the GPU nodes), through `serve.sh`, only the checkpoint mounted, Hub offline: the 8b on one H100 and the 30b on two H100s (`--tensor-parallel-size 2`, set from the GPU count). On each: `e2e_check`, `probe_check`, `accuracy_check` | pass on both: 11/11 and 6/6, accuracy as on the pip path; the log shows `hf_overrides: {'asr_enabled': False}` and never names the speech model Repeated with the published `ghcr.io/generative-computing/granite-switch-systemone:0.1.3` image (image manifest digest `sha256:77a56fb19768727de6e714d160d32842ddd1715964d51b9a5da5acba604e7971`, every layer's digest verified on download): same results on the 8b and the 30b, and in the speech-input row below |
| Speech input on (`SYSTEMONE_SPEECH_INPUT=1`) | the 8b on the default image with the speech model supplied from a mounted Hub cache (`HF_HUB_CACHE`); and, as a negative control, the same switch with no speech model anywhere | the 8b starts and passes the same three checks; the control stops in `serve.sh` with the preflight message naming the model and the fixes, before vLLM starts. The `WITH_ASR=1` image (model baked in) was served on the 8b in an earlier run |
| Decision Index runs | samples of the Decision Index suite (edition 0.3) through `/v1/systemone` at `--max-model-len 131072`, on the 8b and the 30b, as an end-to-end validation of the endpoint | completes with no errors; scores are not reported here while the methodology is verified, and the model card's published figures are the reference (see the caveats below) |

## Not tested

- **Serving the image under Docker.** The image was served under enroot, on the 8b (one H100)
  and the 30b (two H100s; see the table). That runs the same filesystem and environment as the image, but not
  `docker run` itself: its `ENTRYPOINT` (enroot ignores it, so `serve.sh` was started explicitly),
  its `--gpus all` handling and `--ipc=host` were not exercised.
- **The 3b.** The only 3b checkpoint, `granite-switch-4.1-3b-preview`, ships twelve
  adapters (`citations`, `query_rewrite`, `query_clarification`, `hallucination_detection`,
  `answerability`, `factuality-detection`, `policy-guardrails`, `factuality-correction`,
  `guardian-core`, `uncertainty`, `requirement-check`, `context-attribution`), none named
  `decision`, so this plugin refuses to start on it. No servable 3b exists; the 8b and
  30b are the released checkpoints this plugin can serve.
- **The full 262,144-token danger-zone cap.** The cap clamps to `--max-model-len`, and
  every released Switch checkpoint tops out at 131,072 tokens of context; the header path
  is exercised to 131,072 (see the long-contexts row above).

Decision Index caveats: IBM's exact 1,068 fast-subset rows are unpublished, so the run
uses the kit's own stratified sample of the same size (the card itself reports ~0.6
points of serving-path noise on identical rows); the kit's rebuilt release-v1 rows
differ in bytes from the lab's pinned file (per-benchmark counts match; the added and
GSM8K rows are byte-identical), a known kit condition also hit by other public runs,
worked around by running the v2 cut on the rebuilt rows; and 4 sample rows over 64k
tokens are rejected by the documented default caps (scored as wrong by the kit's rules).
These runs are validation that the endpoint works end to end on a realistic benchmark.
They do not show the serving path is identical to the one that produced the card's figures.

## Known gaps

- **Speculative decoding is untested and may crash the engine.** Decision reads use
  `logprob_token_ids`; vLLM issue #42592 (fix #44727, open) crashes on that whenever a
  speculative-decoding request shares the batch. We have not run MTP, so whether the decision
  route is hit is unverified; the plugin warns at startup when `speculative_config` is set.
- **Decision load slows chat.** Chat p50 for a 32-token reply went from 235 ms idle to
  1,088 ms with 128 concurrent decision clients and no cap. With
  `SYSTEMONE_MAX_CONCURRENT_REQUESTS=8` and 32 clients it was 362 ms, the excess decision
  requests getting 503s. The cap is off by default, and there is no priority between the
  two kinds of traffic.
- **No system prompt by default.** The model card's published score used one; this plugin
  omits it so decisions can share the prefix cache with chat. How it affects accuracy is
  still being evaluated. `SYSTEMONE_EXPERIMENTAL_SYSTEM_PROMPT=1` turns it on (tested
  live); the default stays off until that is settled.
