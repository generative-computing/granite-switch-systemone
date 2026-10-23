# granite-switch-systemone

Decision APIs for [Granite Switch](https://github.com/generative-computing/granite-switch), as a
vLLM plugin. Ask a question about some text and get back probabilities, from the same server and
GPU cache that already serves `/v1/chat/completions`.

- `POST /v1/decisions` follows [OpenAI's Decisions API](https://developers.openai.com/api/docs/guides/decisions).
  The official OpenAI SDKs work once you change `base_url`.
- `POST /v1/systemone` follows [jev's system-one API](https://docs.typesafe.ai/api).

Each question is answered with one engine read of the adapter's single-token option labels, so
callers do not need a prompt format, an invocation marker or a label scheme.

You can run it as a container or add it to a vLLM install you already have. Start with
[Quick start](#quick-start). What has and has not been tested is in
[docs/STATUS.md](docs/STATUS.md), and how to run the tests is in [docs/TESTING.md](docs/TESTING.md).

## Quick start

You need a Linux machine with an NVIDIA GPU. The 8b checkpoint fits on one 80 GB GPU.

### Container

Release images are published to GitHub Container Registry as
`ghcr.io/generative-computing/granite-switch-systemone` for linux/amd64 and linux/arm64, so one tag
works on both. `latest` is the newest release, and each release is also tagged with its version. The
pull is about 9 GB and the image takes about 30 GB on disk (amd64 figures). You also need Docker and the NVIDIA container toolkit.

```bash
# Download the checkpoint (public, not gated, about 18 GB)
hf download ibm-granite/granite-switch-4.1-8b --local-dir /srv/models/granite-switch-4.1-8b

# Run the image
docker run -d --name gss --gpus all --ipc=host -p 8000:8000 \
    -v /srv/models/granite-switch-4.1-8b:/repository:ro \
    ghcr.io/generative-computing/granite-switch-systemone:0.1.3
```

Loading takes a few minutes; follow it with `docker logs -f gss`. `--ipc=host` is vLLM's usual
setting for shared memory, and it matters most when you run with tensor parallelism.

The ghcr package is currently private, so `docker pull` needs `docker login ghcr.io` with an account
that has access. Until it is public, build the image yourself: clone this repository at a release tag
and run `docker build -t granite-switch-systemone .`.

The image is `vllm/vllm-openai:v0.30.0` plus `granite-switch[audio]==0.1.1`, `transformers==5.17.0` and
this plugin. It has no weights in it. It reads the checkpoint from `/repository` (override with
`MODEL_DIR`), so the same image serves the 8b and the 30b. We tested the 30b on two 80 GB GPUs; the
container uses every GPU it can see unless you set `TENSOR_PARALLEL_SIZE`.

### Existing vLLM environment

You need Python 3.11 or later and PyTorch 2.11 or later (CUDA 13 builds are tested). **Use vLLM
0.30.0**: it is the version the live tests ran on. granite-switch 0.1.1 accepts `vllm>=0.26,<0.31`,
and `v0.1.0` was also served on 0.26 to 0.29, but nothing since has been checked there.

```bash
pip install "granite-switch[vllm]==0.1.1" "transformers>=5.16"
pip install "git+https://github.com/generative-computing/granite-switch-systemone@v0.1.3"
export VLLM_PLUGINS=register_granite_switch,granite_switch_systemone
export SYSTEMONE_INVOCATION_MARKER="### DECISION"
vllm serve ibm-granite/granite-switch-4.1-8b --served-model-name granite-switch \
    --dtype bfloat16 --attention-config.flash_attn_version=2 --enable-prompt-tokens-details
```

List both plugins in `VLLM_PLUGINS`. vLLM loads endpoint plugins only when they are named, and once
the variable is set it loads nothing else, so leaving out `register_granite_switch` stops the Switch
model class from registering.

### Hugging Face Inference Endpoints

Create an endpoint for the model repository and choose a custom container. Hugging Face's
[custom container guide](https://huggingface.co/docs/inference-endpoints/en/guides/custom_container)
says the model weights are stored under `/repository`, which is where this image looks. The image
serves on port `8000` and answers `/health`, so those are the values for the endpoint's container port
and health route. The guide lists public images and private ones on Docker Hub, AWS ECR, Azure ACR
and Google GCR; GitHub Container Registry is not in that list, so the package has to be public.
**We have not tried this on Inference Endpoints yet.**

### Check that it works

```bash
curl -fsS http://localhost:8000/health && echo healthy
curl -fsS http://localhost:8000/v1/models         # lists "granite-switch"
python3 scripts/e2e_check.py --base-url http://localhost:8000
```

`e2e_check.py` needs only the standard library and works against any host, so run it from a checkout of
this repository. It exits 0 when the wire formats on both routes, route parity, chat and prefix-cache
sharing all pass. If the server stops at startup, its log names the cause: an unknown
`SYSTEMONE_ADAPTER_NAME`, a checkpoint with no `decision` adapter, or an invocation marker that does not
activate the adapter in the chat template.

## Make a request

```bash
curl http://localhost:8000/v1/decisions -H "Content-Type: application/json" -d '{
  "model": "granite-switch",
  "input": "My payouts have failed three times. The bank says everything is fine.",
  "questions": [
    {"type": "choice", "name": "queue", "instructions": "Which team should handle this?",
     "choices": [{"value": "payments", "description": "Payout failures"},
                 {"value": "account", "description": "Login issues"}]},
    {"type": "predicate", "name": "escalate", "instructions": "Does this need urgent attention?"}
  ]}'
```

[examples/decisions.md](examples/decisions.md) has the OpenAI SDK, all three question types and the
equivalent `/v1/systemone` request. For equivalent questions both routes build the same prompt and
return the same probabilities.

## Secure it before you expose it

The server listens on `0.0.0.0` and, as shipped, asks for no credentials on any route. Put it behind a
gateway that authenticates callers, or turn on vLLM's `--api-key`, which `serve.sh` passes through:

```bash
docker run -d --gpus all --ipc=host -p 8000:8000 -v … granite-switch-systemone --api-key "$KEY"
```

In vLLM 0.30.0 that key check covers every `/v1` route, including `/v1/decisions` and
`/v1/systemone`, and leaves `/health` open (`vllm/entrypoints/serve/middleware/authenticate.py`). We
have not run it against a live server, so after enabling it, confirm that an unauthenticated `curl` to
`/v1/decisions` returns 401.

If the server also handles chat, set `SYSTEMONE_MAX_CONCURRENT_REQUESTS` (8 to 32 is a sensible
range). It is off by default, and without it a burst of decisions slowed chat p50 from 235 ms to
1,088 ms in our load test (see [docs/STATUS.md](docs/STATUS.md)).

## Speech input (optional)

Speech is a chat feature. The decision routes take text only, and the 8b and 30b checkpoints enable
speech input in their configuration. Loading a speech model slows startup and decisions never use it,
so `serve.sh` switches it off (`--hf-overrides '{"asr_enabled": false}'`). The checkpoint then starts
with nothing else to download, and `/v1/chat/completions` accepts no audio.

To turn it on, set `SYSTEMONE_SPEECH_INPUT=1`. The image runs with the Hugging Face Hub offline, so the
speech model
([`ibm-granite/granite-speech-5.0-470m-turboctc`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc),
about 1 GB) has to be available. Any one of these works:

- Use an image with the model built in: the `-asr` tag of a release (releases made with the current
  workflow publish one), or build it yourself:
  `docker build --build-arg WITH_ASR=1 -t granite-switch-systemone:asr .`
- Mount a Hub cache that holds the model and set `HF_HUB_CACHE` to that directory.
- Set `HF_HUB_OFFLINE=0` so the model downloads at startup.

On startup `serve.sh` checks for the speech model and for the audio libraries (`av`, `scipy`,
`soundfile`, `soxr`), and stops with a message that names whatever is missing. The images already
include the libraries. If you run the plugin in your own environment, install `vllm[audio]`. Without
it the server starts and passes `/health`, then answers every audio request with HTTP 500 "Please
install vllm[audio]".

The `-asr` image bakes in the multilingual release of the speech model; its model card lists the
supported languages. To bake in the earlier English-only revision instead, pass
`--build-arg ASR_REVISION=947f59af40db9791170a0628cf0f3f4812d720f1`.

## Configuration

`serve.sh`, the image's entrypoint, runs `vllm serve` with these settings. Everything after the image
name is passed on to `vllm serve`.

| Environment variable | Default | Effect |
|---|---|---|
| `MODEL_DIR` | `/repository` | Where the checkpoint is. |
| `SERVED_MODEL_NAME` | `granite-switch` | The `model` name clients send. |
| `PORT` | `8000` | Listening port. |
| `MAX_MODEL_LEN` | `32768` | Longest prompt plus output. |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Share of GPU memory vLLM may use. |
| `TENSOR_PARALLEL_SIZE` | every visible GPU | GPUs to split the model across. |
| `SYSTEMONE_ADAPTER_NAME` | the only adapter, else `decision` | Which adapter answers. Checked against the checkpoint at startup; an unknown name stops the server. |
| `SYSTEMONE_INVOCATION_MARKER` | none (`### DECISION` in the image) | The text the adapter was trained to activate on. It is required: with a wrong marker the chat template activates the adapter in the wrong place without raising an error. |
| `SYSTEMONE_SPEECH_INPUT` | `0` | `1` keeps speech input; see [Speech input](#speech-input-optional). |
| `SYSTEMONE_MAX_CONCURRENT_REQUESTS` | `0` (no cap) | Decision requests allowed in flight across both routes. Past the cap a request gets 503 at once, so a burst cannot queue ahead of chat. |
| `SYSTEMONE_RENDERER` | `granite_switch_systemone.renderer:render` | A `module:function` that builds the prompt and label ids, for an adapter trained differently. |
| `SYSTEMONE_EXPERIMENTAL_SYSTEM_PROMPT` | `0` | Experimental, may be removed. `1` adds the trained decision-engine system prompt, as used for the model card's published score. It stops chat requests sharing the decision prefix cache. |

Both routes share these settings; nothing is configured per route.

### Build arguments

`GRANITE_SWITCH_VERSION` (default `0.1.1`) is the granite-switch release to install, with its `audio`
extra. `WITH_ASR=1` bakes in the speech model. `ASR_MODEL` and `ASR_REVISION` name the Hub repository
and the commit that is baked in. The default commit is pinned, so a rebuild does not change when the
model repository moves. The checkpoint is never a build argument: you mount it when the container runs.

### Server flags

`serve.sh` passes these non-default flags to vLLM:

| Flag | Why |
|---|---|
| `--dtype bfloat16` | Composed checkpoints carry `torch_dtype: null` ([granite-switch#126](https://github.com/generative-computing/granite-switch/issues/126)), and vLLM would otherwise load float32. |
| `--attention-config.flash_attn_version=2` | Avoids an FA3 schedule bug with CUDA graphs and prefix caching ([granite-switch#139](https://github.com/generative-computing/granite-switch/issues/139), open). With granite-switch 0.1.1 we measured FA3 the same as FA2 for answers and speed, so this is a precaution. |
| `--enable-prompt-tokens-details` | vLLM reports each request's prefix-cache hits, and `/v1/decisions` returns them as `usage.input_tokens_details`. Without the flag both counts are 0. |
| `--max-logprobs 256` | For chat clients that read labels by top-K logprobs (vLLM's default cap is 20). The plugin does not need it. |
| `--enable-auto-tool-choice --tool-call-parser granite4` | Tool calling on the shared chat route. |

There is no server-wide temperature override, so chat requests keep the model's defaults. Prefix
caching stays at vLLM's default, which is on.

On every decision request the plugin itself sets `temperature=0.0`, `max_completion_tokens=1`,
`logprob_token_ids` (the option labels' token ids, read exactly rather than from the top-K) and
`chat_template_kwargs={"adapter_name": <SYSTEMONE_ADAPTER_NAME>}`. Callers cannot change them. vLLM
caps `logprob_token_ids` at 128 ids, so a `choice` with more options runs as several requests over the
same prompt, served from the prefix cache.

## Behaviour and limits

- **Prompt.** Two user turns: `State:\n<state>`, then `### DECISION` with the question and lettered
  options (A to Z, then AA, AB and so on past 26), read as a softmax over the label logits. Predicates
  render as "A: No / false" and "B: Yes / true". An option without a description is described by its
  value.
- **No system prompt by default.** A decision can then share the prefix cache with a chat request
  whose first turn is the same `State:` turn. A chat with its own system prompt, or one that sends the
  state without `State:`, shares nothing. Decisions about one state always share with each other. The
  model card's published score used a system prompt, and its effect on accuracy is still being
  evaluated.
- **Caller text is escaped.** Special tokens and the invocation marker inside the state, instructions
  or descriptions are broken with an invisible U+2060, so they cannot forge turns or move the adapter's
  activation point.
- **Input.** `input` messages must have the role `user`, and images are rejected, because the trained
  layout has one text state turn. `/v1/systemone` rejects unknown fields and `box` questions.
- **Size.** The state, instructions and options of each question are counted before any engine work:
  at most 200 questions, 32k tokens per question and 64k per request (262k with
  `TypeSafe-Danger-Zone-Long-Context: true` on `/v1/systemone`), all capped by `--max-model-len`.
  Bodies over 32 MiB, nesting deeper than 64 and non-finite numbers are rejected.
- **Usage.** A `choice` with more than 128 options runs as two engine requests over the same prompt,
  and `usage.input_tokens` counts the prompt once per request, so such a question reports about twice
  its prompt length. `cached_tokens` shows how much was shared.
- **Speculative decoding.** Every decision read passes `logprob_token_ids`, which crashes vLLM when a
  speculative-decoding request shares the batch ([vllm#42592](https://github.com/vllm-project/vllm/issues/42592),
  fix [#44727](https://github.com/vllm-project/vllm/pull/44727) open). The plugin logs a warning at
  startup if `speculative_config` is set. We have not tested this combination.
- **Cancellation.** A client that disconnects cancels its request's engine work.
- **Errors.** `/v1/decisions` uses OpenAI's error shape and `/v1/systemone` uses jev's
  `HTTPValidationError` for 422s. Engine errors keep vLLM's status code.
- **Not provided.** Authentication (use vLLM's `--api-key`), rate limiting and `refusal` answers.
  Engine reads are not bit-identical between batched and solo requests, so repeated calls can differ in
  the third decimal place.

## Upstream status

vLLM now has its own `/v1/systemone` ([#59299](https://github.com/vllm-project/vllm/pull/59299),
merged 2026-10-08) and `/v1/decisions` ([#60465](https://github.com/vllm-project/vllm/pull/60465),
merged 2026-10-13). Both are on `main` after 0.30.0, so the next vLLM release will carry them. This
package is for vLLM 0.30, and for 0.26 to 0.29 (untested since `v0.1.0`), where those routes are
absent. You can drop it once you move to a release that has them. The upstream endpoints differ in wire
details: this plugin follows jev's format and OpenAI's schema as published.

The plugin refuses to start if either path is already registered, so on a vLLM that has the upstream
endpoints it fails at startup and does not silently pick one. Remove `granite_switch_systemone` from
`VLLM_PLUGINS` there. vLLM 0.32 mounts its own `/v1/systemone` (newer versions add `/v1/decisions`).
Neither knows Switch adapters, and granite-switch 0.1.1 does not support those versions yet.

Still open upstream are native Granite Switch support
([#60232](https://github.com/vllm-project/vllm/pull/60232)) and modular read strategies
([RFC #59365](https://github.com/vllm-project/vllm/issues/59365)). If the RFC is accepted, the natural
migration is a Granite Switch read strategy, and `renderer.render(tok, state, instructions, descs) ->
(messages, label_ids)` is already close to what a strategy needs.

## How well tested is this?

The pip path is what most of the live testing used. The container image is lightly tested: it builds
for amd64 and arm64, with and without the speech model, and the packages and plugin entry points were checked
inside it. The published `0.1.3` image was served on H100 GPUs with only the checkpoint mounted and
speech input off, the 8b on one GPU and the 30b on two with tensor parallelism, and the wire,
robustness and accuracy checks passed. That was run under enroot and not with `docker run`. The arm64 image was also served with `docker run` on an NVIDIA GB10 (the 8b, default image). The arm64 `-asr` image was also served with `docker run` and passed the speech check; see
[docs/STATUS.md](docs/STATUS.md) for exactly what was and was not run. If you run the image, treat the
first start as a test and tell us what you find.

## Development

```bash
pip install -e ".[dev]" && pytest                              # unit and spec tests, no GPU
python3 scripts/e2e_check.py --base-url http://localhost:8000   # live wire checks
python3 scripts/probe_check.py --base-url http://localhost:8000 # live robustness checks
python3 scripts/accuracy_check.py --base-url http://localhost:8000  # answers are still good (needs internet)
```

[docs/TESTING.md](docs/TESTING.md) lists every test, what runs in CI, and what needs a tokenizer or a
live server.

## Licence

Apache-2.0. Parts adapted from MIT-licensed code are listed, with their notices, in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
