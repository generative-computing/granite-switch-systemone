# vLLM server for a composed granite-switch checkpoint: /v1/chat/completions plus the
# /v1/decisions and /v1/systemone plugin routes, on one engine and one KV cache.
# The checkpoint is not in the image; mount it at $MODEL_DIR (default /repository).
#
#   docker build -t <registry>/granite-switch-systemone:<tag> .
#
# The audio libraries are in every build. WITH_ASR=1 also bakes in the speech model (see
# ASR_MODEL and ASR_REVISION); serve.sh keeps speech off unless SYSTEMONE_SPEECH_INPUT=1.
#   docker build --build-arg WITH_ASR=1 -t <registry>/granite-switch-systemone:<tag>-asr .
#
# vLLM 0.30 is the version tested live. The default tag is the CUDA 13 build (torch
# 2.13.0+cu130), the stack the tests ran. The -cu129 tag ships torch 2.14.0, which the
# vLLM wheel's own torch==2.13.0 pin rules out, so granite-switch[vllm] cannot resolve there.
FROM vllm/vllm-openai:v0.30.0
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG GRANITE_SWITCH_VERSION=0.1.1
ARG WITH_ASR=0

# granite-switch is pinned to a PyPI release so the image does not drift with upstream;
# transformers 5.17 is the version it supports. The [audio] extra (av, scipy, soundfile,
# soxr) is in every build: without it audio requests fail with HTTP 500 while /health and
# decisions look fine. The constraint file keeps the base image's vLLM, torch and
# torchcodec as they are.
RUN python3 -m pip list --format=freeze | grep -i -E "^(vllm|torch|torchaudio|torchcodec|torchvision|flashinfer-python)==" > /tmp/keep.txt \
    && uv pip install --system --constraint /tmp/keep.txt \
        "granite-switch[audio]==${GRANITE_SWITCH_VERSION}" \
        "transformers==5.17.0" \
    && python3 -c "import granite_switch.vllm, vllm, transformers, torch, av, scipy, soundfile, soxr; print(vllm.__version__, transformers.__version__, torch.__version__)"

# The speech model the checkpoint's config names, baked in as a Hub cache at the tested
# commit (without the repo's unused .py files). The server runs with the Hub offline, so a
# missing file fails startup instead of fetching something else. A download by commit
# writes no refs/main, which the offline lookup for "main" reads, so it is written here.
ARG ASR_MODEL=ibm-granite/granite-speech-5.0-470m-turboctc
ARG ASR_REVISION=1dd23f46f6d4e6eae2531982ba2e33220086dd97
RUN if [ "$WITH_ASR" = "1" ]; then \
        hf download ${ASR_MODEL} --revision ${ASR_REVISION} --cache-dir /opt/hf-cache --exclude "*.py" > /dev/null \
        && mkdir -p /opt/hf-cache/models--${ASR_MODEL//\//--}/refs \
        && printf '%s' "${ASR_REVISION}" > /opt/hf-cache/models--${ASR_MODEL//\//--}/refs/main \
        && test -d /opt/hf-cache/models--${ASR_MODEL//\//--}/snapshots/${ASR_REVISION}; \
    fi
ENV HF_HUB_CACHE=/opt/hf-cache HF_HUB_OFFLINE=1

# The plugin: a vllm.endpoint_plugins entry point (see src/granite_switch_systemone/plugin.py).
COPY pyproject.toml README.md /opt/granite-switch-systemone/
COPY src /opt/granite-switch-systemone/src
RUN uv pip install --system --no-deps /opt/granite-switch-systemone

COPY serve.sh /usr/local/bin/serve.sh
RUN chmod +x /usr/local/bin/serve.sh
# Endpoint plugins load only when VLLM_PLUGINS names them, and setting it stops vLLM loading
# any others, so register_granite_switch has to be listed too.
ENV VLLM_PLUGINS=register_granite_switch,granite_switch_systemone \
    SYSTEMONE_ADAPTER_NAME=decision \
    SYSTEMONE_INVOCATION_MARKER="### DECISION"
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/serve.sh"]
