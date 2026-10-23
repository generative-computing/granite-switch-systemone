# SPDX-License-Identifier: Apache-2.0 AND MIT
"""No-GPU checks for plugin.py's validation helpers, renderer resolution and adapter-name
resolution. They do not touch vLLM or a model; the engine call (`_answer_question`,
`_fetch_label_logprobs`) is covered by the live scripts in scripts/.

  python3 -m pytest tests/test_plugin.py -v -s --tb=short
"""

import asyncio
import json as _json
import math

import pytest
from pydantic import ValidationError as PydanticValidationError

from granite_switch_systemone import plugin as sp

# --------------------------------------------------------------------------------------------
# Question / SystemOneRequest schema validation (jev's 422s)
# --------------------------------------------------------------------------------------------


def test_question_choice_valid():
    q = sp.Question(type="choice", instructions="Which?", criteria={"a": "A", "b": "B"})
    assert q.criteria == {"a": "A", "b": "B"}


def test_question_choice_single_option_accepted():
    """jev's real OpenAPI spec (ChoiceQuestion.criteria) says minProperties=1, not 2."""
    q = sp.Question(type="choice", instructions="Which?", criteria={"a": "A"})
    assert q.criteria == {"a": "A"}


def test_question_choice_zero_options_rejected():
    with pytest.raises(Exception):
        sp.Question(type="choice", instructions="Which?", criteria={})


def test_question_choice_too_many_options_rejected():
    criteria = {f"k{i}": f"d{i}" for i in range(256)}
    with pytest.raises(Exception):
        sp.Question(type="choice", instructions="Which?", criteria=criteria)


def test_question_score_bounds():
    """jev's real OpenAPI spec (ScoreQuestion.criteria) says minItems=1, not 2."""
    sp.Question(type="score", instructions="How much?", criteria=["lo", "hi"])
    sp.Question(type="score", instructions="How much?", criteria=["only one"])
    with pytest.raises(Exception):
        sp.Question(type="score", instructions="How much?", criteria=[])
    with pytest.raises(Exception):
        sp.Question(type="score", instructions="How much?", criteria=[str(i) for i in range(11)])


def test_question_instructions_optional():
    """jev's real OpenAPI spec: `instructions` is NOT in any question type's
    `required` list, so a question can rely on well-named criteria alone."""
    q = sp.Question(type="choice", criteria={"a": "A", "b": "B"})
    assert q.instructions is None


def test_question_noul_criteria_optional():
    q = sp.Question(type="noul", instructions="Is it?")
    assert q.criteria is None
    q2 = sp.Question(type="noul", instructions="Is it?", criteria={"true": "Yes"})
    assert q2.criteria == {"true": "Yes"}


def test_system_one_request_requires_nonempty_questions():
    q = sp.Question(type="noul", instructions="Is it?")
    sp.SystemOneRequest(state="s", model="m", questions={"q1": q})
    with pytest.raises(Exception):
        sp.SystemOneRequest(state="s", model="m", questions={})


def test_question_instructions_accepts_object_and_array():
    q1 = sp.Question(type="noul", instructions={"question": "Is it urgent?", "ref": "ticket #42"})
    assert q1.instructions["question"] == "Is it urgent?"
    q2 = sp.Question(type="noul", instructions=["Is it urgent?", "context line"])
    assert q2.instructions == ["Is it urgent?", "context line"]


def test_question_choice_criteria_values_accept_object_array_and_null():
    q = sp.Question(
        type="choice",
        instructions="Which?",
        criteria={"a": "plain string", "b": {"desc": "an object"}, "c": ["an", "array"], "d": None},
    )
    assert q.criteria["d"] is None


def test_question_score_criteria_entries_accept_object_and_array():
    q = sp.Question(
        type="score",
        instructions="How much?",
        criteria=["plain string", {"desc": "an object"}, ["an", "array"]],
    )
    assert q.criteria[1] == {"desc": "an object"}


def test_question_score_criteria_entry_cannot_be_null():
    """Unlike choice, jev does not document score levels as nullable."""
    with pytest.raises(Exception):
        sp.Question(type="score", instructions="How much?", criteria=["lo", None])


def test_validation_error_response_matches_jev_shape():
    """jev's real OpenAPI spec documents the 422 body as
    `{"detail": [{"loc", "msg", "type", ...}]}`, pydantic's own
    ValidationError.errors() shape already."""
    try:
        sp.SystemOneRequest(state="s", model="m", questions={})
    except PydanticValidationError as e:
        resp = sp._validation_error_response(e)
    else:
        raise AssertionError("expected a validation error")
    assert resp.status_code == 422
    body = _json.loads(resp.body)
    assert "detail" in body and isinstance(body["detail"], list) and body["detail"]
    item = body["detail"][0]
    assert {"loc", "msg", "type"} <= item.keys()
    assert isinstance(item["loc"], list)


# --------------------------------------------------------------------------------------------
# Body-level validation: nesting depth, finite numbers, BOM, Content-Encoding/Length
# --------------------------------------------------------------------------------------------


def _nested_state(levels):
    value = {}
    cur = value
    for _ in range(levels - 1):
        cur["x"] = {}
        cur = cur["x"]
    return {"state": value}


def test_check_json_value_nesting_depth_ok_at_limit():
    sp._check_json_value(_nested_state(sp.MAX_NESTING_DEPTH))  # must not raise


def test_check_json_value_rejects_excess_nesting():
    with pytest.raises(ValueError, match="nesting"):
        sp._check_json_value(_nested_state(sp.MAX_NESTING_DEPTH + 5))


def test_check_json_value_rejects_non_finite_floats():
    with pytest.raises(ValueError, match="finite"):
        sp._check_json_value({"a": [1.0, float("nan")]})
    with pytest.raises(ValueError, match="finite"):
        sp._check_json_value({"a": float("inf")})


def test_check_json_value_accepts_finite_numbers():
    sp._check_json_value({"a": [1, 2.5, -3, {"b": 0.0}]})  # must not raise


def test_parse_strict_json_rejects_bom():
    with pytest.raises(ValueError, match="BOM"):
        sp._parse_strict_json(b"\xef\xbb\xbf{}")


def test_parse_strict_json_rejects_non_finite_literals():
    with pytest.raises(ValueError, match="non-finite"):
        sp._parse_strict_json(b'{"a": Infinity}')
    with pytest.raises(ValueError, match="non-finite"):
        sp._parse_strict_json(b'{"a": NaN}')


def test_parse_strict_json_rejects_overflowing_literal():
    with pytest.raises(ValueError, match="finite"):
        sp._parse_strict_json(b'{"a": 1e400}')


def test_parse_strict_json_accepts_valid_body():
    data = sp._parse_strict_json(b'{"state": "s", "model": "m", "questions": {}}')
    assert data == {"state": "s", "model": "m", "questions": {}}


def test_check_content_headers_rejects_encoding():
    result = sp._check_content_headers({"content-encoding": "gzip"})
    assert result is not None and result[0] == 415


def test_check_content_headers_rejects_oversized_length():
    result = sp._check_content_headers({"content-length": str(sp.MAX_BODY_BYTES + 1)})
    assert result is not None and result[0] == 413


def test_check_content_headers_accepts_plain_request():
    assert sp._check_content_headers({"content-length": "10"}) is None
    assert sp._check_content_headers({}) is None


# --------------------------------------------------------------------------------------------
# softmax / confidence (jev's own formulas, not our invention)
# --------------------------------------------------------------------------------------------


def test_softmax_sums_to_one_and_prefers_higher_logprob():
    probs = sp._softmax([math.log(0.1), math.log(0.9)])
    assert abs(sum(probs) - 1.0) < 1e-9
    assert probs[1] > probs[0]


@pytest.mark.parametrize(
    "fn_name,probs,expected",
    [
        ("_score_confidence", [0.2] * 5, 0.0),
        ("_score_confidence", [0.04] * 5, 0.0),
        ("_score_confidence", [0.01, 0.02, 0.07, 0.3, 0.6], 0.55),
        ("_choice_confidence", [0.5, 0.5], 0.0),
        ("_choice_confidence", [0.2, 0.2], 0.0),
        ("_choice_confidence", [0.82, 0.18], 0.64),
        ("_score_confidence", [1.0], 1.0),
        ("_choice_confidence", [1.0], 1.0),
    ],
)
def test_confidence_matches_official_test_vectors(fn_name, probs, expected):
    """Test vectors from typesafe-ai/system-one-adapter-python's
    tests/utils/test_confidence_metrics.py (MIT, Copyright (c) 2026 TypeSafe AI); see
    THIRD_PARTY_NOTICES.md."""
    fn = getattr(sp, fn_name)
    assert fn(probs) == pytest.approx(expected)


# --------------------------------------------------------------------------------------------
# _chunk_label_ids: pure list math for a choice with more than MAX_LABEL_OPTIONS options.
# --------------------------------------------------------------------------------------------


def test_chunk_label_ids_single_chunk_under_the_cap():
    ids = list(range(50))
    assert sp._chunk_label_ids(ids) == [ids]


def test_chunk_label_ids_exactly_at_the_cap():
    ids = list(range(sp.MAX_LABEL_OPTIONS))
    assert sp._chunk_label_ids(ids) == [ids]


def test_chunk_label_ids_splits_above_the_cap():
    ids = list(range(200))
    chunks = sp._chunk_label_ids(ids)
    assert len(chunks) == 2
    assert chunks[0] == ids[:128]
    assert chunks[1] == ids[128:]
    assert [i for c in chunks for i in c] == ids


def test_chunk_label_ids_at_jevs_max_choice_options():
    """jev's own ceiling for choice: 255 options -> ceil(255/128) = 2 chunks."""
    ids = list(range(255))
    chunks = sp._chunk_label_ids(ids)
    assert len(chunks) == 2
    assert sum(len(c) for c in chunks) == 255


# --------------------------------------------------------------------------------------------
# _cancel_pending: pure asyncio, no vLLM needed.
# --------------------------------------------------------------------------------------------


def test_cancel_pending_cancels_unfinished_tasks_only():
    async def scenario():
        async def never_finishes():
            await asyncio.sleep(1000)

        async def finishes_immediately():
            return "done"

        running = asyncio.ensure_future(never_finishes())
        already_done = asyncio.ensure_future(finishes_immediately())
        await asyncio.sleep(0)
        assert already_done.done()
        assert not running.done()

        await sp._cancel_pending([running, already_done])

        assert running.cancelled()
        assert already_done.done() and not already_done.cancelled()

    asyncio.run(scenario())


def test_cancel_pending_is_a_noop_on_already_finished_tasks():
    async def scenario():
        async def finishes_immediately():
            return "done"

        tasks = [asyncio.ensure_future(finishes_immediately()) for _ in range(3)]
        await asyncio.gather(*tasks)
        await sp._cancel_pending(tasks)
        assert all(t.done() and not t.cancelled() for t in tasks)

    asyncio.run(scenario())


# --------------------------------------------------------------------------------------------
# _resolve_adapter_name: cross-checks a candidate decision-adapter name against the loaded
# checkpoint's REAL GraniteSwitchConfig.adapter_names, never a hardcoded name list.
# --------------------------------------------------------------------------------------------


class _FakeHfConfig:
    def __init__(self, adapter_names):
        self.adapter_names = adapter_names


class _FakeModelConfig:
    def __init__(self, adapter_names):
        self.hf_config = _FakeHfConfig(adapter_names)


class _FakeChat:
    def __init__(self, adapter_names):
        self.model_config = _FakeModelConfig(adapter_names)


def test_resolve_adapter_name_auto_resolves_single_adapter():
    chat = _FakeChat(["decision"])
    assert sp._resolve_adapter_name(chat) == "decision"


def test_resolve_adapter_name_prefers_decision_among_several(monkeypatch):
    """The released Switch checkpoints carry guardian-core and decision."""
    monkeypatch.delenv("SYSTEMONE_ADAPTER_NAME", raising=False)
    chat = _FakeChat(["guardian-core", "decision"])
    assert sp._resolve_adapter_name(chat) == "decision"


def test_resolve_adapter_name_requires_env_var_with_several_and_no_decision(monkeypatch):
    monkeypatch.delenv("SYSTEMONE_ADAPTER_NAME", raising=False)
    chat = _FakeChat(["guardian-core", "classifier"])
    with pytest.raises(RuntimeError, match="SYSTEMONE_ADAPTER_NAME"):
        sp._resolve_adapter_name(chat)


def test_resolve_adapter_name_requires_env_var_with_zero_adapters():
    chat = _FakeChat([])
    with pytest.raises(RuntimeError, match="SYSTEMONE_ADAPTER_NAME"):
        sp._resolve_adapter_name(chat)


def test_resolve_adapter_name_honors_env_override(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_ADAPTER_NAME", "decision")
    chat = _FakeChat(["guardian-core", "decision"])
    assert sp._resolve_adapter_name(chat) == "decision"


def test_resolve_adapter_name_env_override_still_validated(monkeypatch):
    """An override naming an adapter the checkpoint doesn't have must still fail
    loudly, not silently fall back to anything."""
    monkeypatch.setenv("SYSTEMONE_ADAPTER_NAME", "typo-adapter-name")
    chat = _FakeChat(["decision"])
    with pytest.raises(RuntimeError, match="typo-adapter-name"):
        sp._resolve_adapter_name(chat)


# --------------------------------------------------------------------------------------------
# _load_renderer: resolves "module:function" by name, so plugin.py never imports a
# renderer module directly.
# --------------------------------------------------------------------------------------------


def test_load_renderer_default_resolves_to_this_packages_own_renderer():
    import os as _os

    _os.environ.pop(sp._RENDERER_ENV, None)
    fn = sp._load_renderer()
    from granite_switch_systemone.renderer import render

    assert fn is render


def test_load_renderer_rejects_a_path_without_a_colon(monkeypatch):
    monkeypatch.setenv(sp._RENDERER_ENV, "not_a_valid_path")
    with pytest.raises(RuntimeError, match="module:function"):
        sp._load_renderer()


def test_load_renderer_rejects_a_missing_attribute(monkeypatch):
    monkeypatch.setenv(sp._RENDERER_ENV, "granite_switch_systemone.renderer:nonexistent_fn")
    with pytest.raises(RuntimeError, match="nonexistent_fn"):
        sp._load_renderer()


def test_load_renderer_resolves_a_custom_path(monkeypatch):
    monkeypatch.setenv(sp._RENDERER_ENV, "math:sqrt")
    fn = sp._load_renderer()
    import math as _math

    assert fn is _math.sqrt


# --------------------------------------------------------------------------------------------
# _enforce_context_limits: its ValueError messages must contain a phrase from a third-party
# benchmark harness's own capacity-rejection marker list, or a legitimately-oversized request
# gets scored as a hard "error" instead of "unsupported" by that harness.
# --------------------------------------------------------------------------------------------

_CAPACITY_MARKERS = (
    "maximum context length",
    "maximum model length",
    "longer than the maximum model length",
    "context window",
    "too many tokens",
)


class _FakeTok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_enforce_context_limits_per_question_message_matches_a_capacity_marker():
    tok = _FakeTok()
    q = sp.Question(type="noul", instructions="word " * 100_000)
    with pytest.raises(ValueError) as exc_info:
        sp._enforce_context_limits(tok, 32768, False, "short state", {"q": q})
    assert any(marker in str(exc_info.value) for marker in _CAPACITY_MARKERS)


def test_enforce_context_limits_total_message_matches_a_capacity_marker():
    tok = _FakeTok()
    questions = {f"q{i}": sp.Question(type="noul", instructions="word " * 5000) for i in range(10)}
    with pytest.raises(ValueError) as exc_info:
        sp._enforce_context_limits(tok, 32768, False, "short state", questions)
    assert any(marker in str(exc_info.value) for marker in _CAPACITY_MARKERS)


def test_enforce_context_limits_accepts_small_request():
    tok = _FakeTok()
    q = sp.Question(type="noul", instructions="a short question")
    sp._enforce_context_limits(tok, 32768, False, "a short state", {"q": q})  # must not raise


# --------------------------------------------------------------------------------------------
# _answer_question's usage accounting, with create_chat_completion's responses faked.
# --------------------------------------------------------------------------------------------


def _fake_chat_response(label_ids, prompt_tokens, cached_tokens, created_cache_tokens):
    from types import SimpleNamespace as NS

    top = [NS(token=f"token_id:{tid}", logprob=-float(i)) for i, tid in enumerate(label_ids)]
    details = (
        None if cached_tokens is None else NS(cached_tokens=cached_tokens, created_cache_tokens=created_cache_tokens)
    )
    return NS(
        choices=[NS(logprobs=NS(content=[NS(top_logprobs=top)]))],
        usage=NS(prompt_tokens=prompt_tokens, completion_tokens=1, prompt_tokens_details=details),
    )


@pytest.mark.parametrize(
    ("first_chunk_cache", "expected_cached", "expected_written"),
    [((40, 60), 140, 60), ((None, None), 100, 0)],
    ids=["details reported", "details disabled"],
)
def test_answer_question_counts_every_chunk_of_a_chunked_prompt(
    monkeypatch, first_chunk_cache, expected_cached, expected_written
):
    """A 200-option choice is two engine calls over one prompt, and usage counts both:
    each call prefills (or reads from cache) its own copy of the prompt."""
    calls = []

    async def fake_fetch(chat, served_model, adapter_name, msgs, label_ids_chunk):
        calls.append(label_ids_chunk)
        cached, written = first_chunk_cache if len(calls) == 1 else (100, 0)
        return _fake_chat_response(label_ids_chunk, 100, cached, written)

    monkeypatch.setattr(sp, "_fetch_label_logprobs", fake_fetch)
    criteria = {f"o{i}": f"option {i}" for i in range(200)}
    q = sp.Question(type="choice", instructions="Which?", criteria=criteria)
    renderer = lambda tok, state, instructions, descs: ([], list(range(1000, 1000 + len(descs))))  # noqa: E731

    _, answer, usage = asyncio.run(
        sp._answer_question("q", q, "state", None, None, "m", "decision", renderer, asyncio.Semaphore(8))
    )

    assert len(calls) == 2
    assert answer["type"] == "choice"
    assert usage == sp.TokenUsage(
        input_tokens=200, output_tokens=2, cached_tokens=expected_cached, cache_write_tokens=expected_written
    )


@pytest.mark.parametrize(
    "module",
    ["vllm.entrypoints.openai.engine.protocol", "vllm.entrypoints.serve.engine.protocol"],
    ids=["vllm 0.26-0.28 layout", "vllm 0.29-0.30 layout"],
)
def test_error_response_type_resolves_on_either_vllm_layout(monkeypatch, module):
    """vLLM 0.29 moved ErrorResponse out of entrypoints/openai/engine/; granite-switch
    0.1.1 supports 0.26-0.30, so the engine call must work on both layouts."""
    import sys
    import types

    fake = types.ModuleType(module)

    class ErrorResponse:
        pass

    fake.ErrorResponse = ErrorResponse
    for name in ("vllm.entrypoints.openai.engine.protocol", "vllm.entrypoints.serve.engine.protocol"):
        monkeypatch.setitem(sys.modules, name, fake if name == module else None)

    assert sp._error_response_type() is ErrorResponse


def test_enforce_context_limits_counts_option_descriptions():
    """Options are part of the prompt: a short state with long descriptions must not
    slip past the guard (measured live: 20 x 255 long options prefilled 1.2M tokens)."""
    tok = _FakeTok()
    criteria = {f"o{i}": "lorem " * 200 for i in range(255)}
    q = sp.Question(type="choice", instructions="Which?", criteria=criteria)
    with pytest.raises(ValueError) as exc_info:
        sp._enforce_context_limits(tok, 32768, False, "short state", {"q": q})
    assert any(marker in str(exc_info.value) for marker in _CAPACITY_MARKERS)


def test_enforce_context_limits_counts_score_level_descriptions():
    tok = _FakeTok()
    q = sp.Question(type="score", instructions="How?", criteria=["lorem " * 4000] * 10)
    with pytest.raises(ValueError):
        sp._enforce_context_limits(tok, 32768, False, "short state", {"q": q})


def test_run_until_disconnect_cancels_work_when_the_client_leaves():
    """A client that disconnects mid-request must cancel the in-flight engine work
    (live: a dropped 40-question request kept prefilling for 30+ s)."""
    cancelled = asyncio.Event()

    async def work():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    class _Request:
        async def receive(self):
            await asyncio.sleep(0.01)
            return {"type": "http.disconnect"}

    async def run():
        result = await sp._run_until_disconnect(_Request(), work())
        return result, cancelled.is_set()

    result, was_cancelled = asyncio.run(run())
    assert result is None and was_cancelled


def test_run_until_disconnect_returns_the_result_when_the_client_stays():
    class _Request:
        async def receive(self):
            await asyncio.sleep(30)

    async def work():
        return "answer"

    assert asyncio.run(sp._run_until_disconnect(_Request(), work())) == "answer"


def test_choice_without_descriptions_renders_its_option_keys(monkeypatch):
    """With no description an option must still say what it is: rendering only its
    label made the answer depend on list order (live: one option scored 0.809 first
    and 0.066 third on the same text)."""
    seen = {}

    def renderer(tok, state, instructions, descs):
        seen["descs"] = descs
        return [], list(range(len(descs)))

    async def fake_fetch(chat, served_model, adapter_name, msgs, label_ids_chunk):
        return _fake_chat_response(label_ids_chunk, 10, None, None)

    monkeypatch.setattr(sp, "_fetch_label_logprobs", fake_fetch)
    q = sp.Question(type="choice", instructions="Which?", criteria={"refund request": None, "login": "Login problems"})
    asyncio.run(sp._answer_question("q", q, "state", None, None, "m", "decision", renderer, asyncio.Semaphore(8)))
    assert seen["descs"] == ["refund request", "Login problems"]


def test_parse_strict_json_rejects_lone_surrogates():
    """JSON allows \\ud800 escapes that are not valid UTF-8 text; reject them at parse time
    rather than letting the tokenizer fail with an internal message."""
    with pytest.raises(ValueError, match="surrogate"):
        sp._parse_strict_json(b'{"state": "\\ud800 x"}')


def test_systemone_rejects_more_questions_than_the_cap():
    """jev's schema sets no question limit, but each question is one engine request."""
    questions = {f"q{i}": {"type": "noul"} for i in range(sp.MAX_QUESTIONS + 1)}
    with pytest.raises(PydanticValidationError):
        sp.SystemOneRequest.model_validate({"model": "m", "state": "x", "questions": questions})
    sp.SystemOneRequest.model_validate(
        {"model": "m", "state": "x", "questions": dict(list(questions.items())[: sp.MAX_QUESTIONS])}
    )


def test_concurrency_limit_rejects_with_503_when_full():
    """SYSTEMONE_MAX_CONCURRENT_REQUESTS bounds decision requests in flight; past it, 503."""

    async def run():
        limiter = sp._Limiter(1)
        release = asyncio.Event()

        async def hold():
            async with limiter.slot() as ok:
                assert ok
                await release.wait()

        holder = asyncio.ensure_future(hold())
        await asyncio.sleep(0)
        async with limiter.slot() as ok:
            second = ok
        release.set()
        await holder
        async with limiter.slot() as ok:
            third = ok
        return second, third

    assert asyncio.run(run()) == (False, True)


def test_concurrency_limit_zero_means_unlimited():
    async def run():
        limiter = sp._Limiter(0)
        async with limiter.slot() as a:
            async with limiter.slot() as b:
                return a and b

    assert asyncio.run(run())


def test_init_state_fails_startup_on_a_bad_system_prompt_flag(monkeypatch):
    from types import SimpleNamespace as NS

    monkeypatch.setenv("SYSTEMONE_INVOCATION_MARKER", "### DECISION")
    monkeypatch.setenv("SYSTEMONE_EXPERIMENTAL_SYSTEM_PROMPT", "yes")
    chat = _FakeChat(["decision"])
    chat.renderer = NS(tokenizer=None)
    chat.model_config.max_model_len = 8192
    state = NS(openai_serving_chat=chat)
    args = NS(served_model_name=["m"], model="m")
    with pytest.raises(RuntimeError, match="SYSTEMONE_EXPERIMENTAL_SYSTEM_PROMPT"):
        asyncio.run(sp.SystemOnePlugin().init_state(None, state, args))


def test_init_state_fails_startup_without_an_invocation_marker(monkeypatch):
    from types import SimpleNamespace as NS

    monkeypatch.delenv("SYSTEMONE_INVOCATION_MARKER", raising=False)
    monkeypatch.delenv("SYSTEMONE_EXPERIMENTAL_SYSTEM_PROMPT", raising=False)
    chat = _FakeChat(["decision"])
    chat.renderer = NS(tokenizer=None)
    chat.model_config.max_model_len = 8192
    state = NS(openai_serving_chat=chat)
    args = NS(served_model_name=["m"], model="m")
    with pytest.raises(RuntimeError, match="SYSTEMONE_INVOCATION_MARKER"):
        asyncio.run(sp.SystemOnePlugin().init_state(None, state, args))


def test_parse_strict_json_rejects_deep_nesting_before_parsing():
    """json.loads recurses past Python's limit on deep input and raises RecursionError."""
    with pytest.raises(ValueError, match="nesting"):
        sp._parse_strict_json(b"[" * 100_000 + b"]" * 100_000)


@pytest.mark.parametrize("where", ["state", "instructions", "criteria"])
def test_nesting_limit_is_64_levels_per_field(where):
    """jev: at most 64 nested levels in each of state, instructions and criteria."""

    def nest(n):
        value = "x"
        for _ in range(n):
            value = [value]
        return value

    def body(n):
        if where == "state":
            return {"state": nest(n)}
        return {"questions": {"q": {where: nest(n)}}}

    sp._check_json_value(body(64))
    with pytest.raises(ValueError, match="nesting"):
        sp._check_json_value(body(65))


def test_engine_error_status_need_not_be_a_standard_http_status():
    assert sp.EngineError("x", 529).status == 529
    assert sp.EngineError("x", 200).status == 500


def test_missing_label_logprob_is_an_engine_error(monkeypatch):
    async def fake_fetch(chat, served_model, adapter_name, msgs, label_ids_chunk):
        return _fake_chat_response(label_ids_chunk[:-1], 10, None, None)

    monkeypatch.setattr(sp, "_fetch_label_logprobs", fake_fetch)
    q = sp.Question(type="noul", instructions="q?")
    renderer = lambda tok, state, instructions, descs: ([], [1, 2])  # noqa: E731
    with pytest.raises(sp.EngineError, match="no logprob"):
        asyncio.run(sp._answer_question("q", q, "s", None, None, "m", "decision", renderer, asyncio.Semaphore(8)))


def test_a_failing_label_chunk_cancels_its_sibling(monkeypatch):
    """The sibling must be cancelled before the error reaches the caller, not left
    running (asyncio.run would cancel it at shutdown anyway, so check inside the loop)."""
    sibling = {}

    async def fake_fetch(chat, served_model, adapter_name, msgs, label_ids_chunk):
        if label_ids_chunk[0] == 0:
            await asyncio.sleep(0)
            raise sp.EngineError("boom", 500)
        sibling["task"] = asyncio.current_task()
        await asyncio.sleep(30)

    monkeypatch.setattr(sp, "_fetch_label_logprobs", fake_fetch)
    q = sp.Question(type="choice", instructions="Which?", criteria={f"o{i}": None for i in range(200)})
    renderer = lambda tok, state, instructions, descs: ([], list(range(len(descs))))  # noqa: E731

    async def run():
        with pytest.raises(sp.EngineError):
            await sp._answer_question("q", q, "s", None, None, "m", "decision", renderer, asyncio.Semaphore(8))
        return sibling["task"].done()

    assert asyncio.run(run())


def test_engine_reads_per_request_are_bounded(monkeypatch):
    in_flight, peak = [0], [0]

    async def fake_fetch(chat, served_model, adapter_name, msgs, label_ids_chunk):
        in_flight[0] += 1
        peak[0] = max(peak[0], in_flight[0])
        await asyncio.sleep(0.01)
        in_flight[0] -= 1
        return _fake_chat_response(label_ids_chunk, 10, None, None)

    monkeypatch.setattr(sp, "_fetch_label_logprobs", fake_fetch)
    from types import SimpleNamespace as NS

    state = NS(
        systemone_chat=None,
        systemone_tokenizer=None,
        systemone_served_model="m",
        systemone_adapter_name="decision",
        systemone_renderer=lambda tok, s, i, descs: ([], list(range(len(descs)))),
    )
    questions = [(str(i), sp.Question(type="noul", instructions="q?")) for i in range(60)]
    asyncio.run(sp._answer_all(state, "s", questions))
    assert peak[0] == sp.MAX_READS_PER_REQUEST


def test_identity_content_encoding_is_accepted():
    assert sp._check_content_headers({"content-encoding": "identity"}) is None
    assert sp._check_content_headers({"content-encoding": "gzip"})[0] == 415


class _ProbeTok:
    """Tokenizer stub: the 'text' is a list of ids, so tests choose where the control id (7) sits."""

    unk_token_id = -1

    def __init__(self, with_gen, without_gen):
        self._out = {True: with_gen, False: without_gen}

    def convert_tokens_to_ids(self, token):
        return 7

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, adapter_name):
        return add_generation_prompt

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._out[text]}


def _probe_renderer(tok, state, instructions, descs):
    return [{"role": "user", "content": "q"}], [1, 2]


def test_check_activation_accepts_one_control_token_inside_the_conversation():
    tok = _ProbeTok(with_gen=[1, 7, 2, 3, 9, 9], without_gen=[1, 7, 2, 3])
    sp._check_activation(tok, _probe_renderer, "decision")


@pytest.mark.parametrize(
    "with_gen, without_gen",
    [
        ([1, 2, 3, 9], [1, 2, 3]),  # no control token
        ([7, 1, 7, 9], [7, 1, 7]),  # two
        ([1, 2, 3, 7, 9], [1, 2, 3]),  # fallback: control just before the assistant header
    ],
)
def test_check_activation_refuses_missing_duplicate_or_misplaced_control_token(with_gen, without_gen):
    with pytest.raises(RuntimeError, match="exactly one"):
        sp._check_activation(_ProbeTok(with_gen, without_gen), _probe_renderer, "decision")


def test_label_mass_warns_below_the_floor_and_never_raises(caplog):
    import math

    with caplog.at_level("DEBUG", logger=sp._log.name):
        sp._log_label_mass("q", [math.log(0.2), math.log(0.2)])
        sp._log_label_mass("q", [math.log(0.6), math.log(0.39)])
    levels = [r.levelname for r in caplog.records]
    assert levels == ["WARNING", "DEBUG"]
