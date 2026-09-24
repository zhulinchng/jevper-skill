"""An offline stand-in for an OpenAI- or Anthropic-compatible client, for jevper code.

Why: jevper is duck-typed, so a plain object is enough to drive it end to end — no HTTP, no API key, no
tokens spent — while the real readout path (logprobs softmax, structured JSON, ``auto``'s fallback and
surface move, the same move under a pinned ``method="logprobs"``, the server-limits ladder, the schema
that travels in the prompt when the request cannot carry one) runs for real.

    from offline_stub import StubClient
    from jevper import Choice, SystemOneClient

    stub = StubClient(scenario="reject_logprobs")        # a provider that 400s the logprob fields
    client = SystemOneClient(stub, model="stub-model")   # method="auto" falls back to structured
    response = client.system_one(state="...", questions={"intent": Choice(criteria={"a": "...", "b": "..."})})

    assert response.debug["methods"]["intent"] == "structured"
    assert stub.requests[0]["logprobs"] is True          # and it did ask for logprobs first

Scenarios:

    logprobs             a provider that returns a real next-token distribution
    structured           a provider that answers in JSON with its own probabilities
    reject_logprobs      a provider that answers 400 to the logprob fields (OpenAI reasoning models,
                         Gemini's OpenAI-compatibility layer)
    no_alternatives      a provider that reports the sampled token and nothing else
    reject_include       a server that refuses the Responses logprob includable and carries the
                         distribution on Chat Completions instead (OpenRouter's shape)
    no_responses_route   a server that answers 404 for /v1/responses — a missing route, not a bad model
    no_messages_route    a server that answers 404 for /v1/messages — the mirror of no_responses_route,
                         and the case where there is no other surface to move to
    reject_schema        a server that answers 400 to a strict JSON schema, and accepts ``json_object``
    reject_format        a server with no format field at all: it refuses ``json_object`` too, so the
                         ladder has one rung fewer to walk
    reject_cache_key     a server that answers 400 to ``prompt_cache_key``
    reject_thinking      a server whose protocol has no Messages ``thinking`` field (SGLang's route)
    reject_budget_value  a server that refuses the thinking budget's *value* (``budget_tokens: must be
                         at least 1024``): a field it knows, so nothing is dropped and the provider's
                         own error travels back
    truncated            a reasoning model that spent the whole output budget thinking: no answer, and a
                         stop reason that says so (``max_tokens`` on the Messages surface)
    refusal              a model that refuses instead of answering: OpenAI's ``refusal`` sibling of a
                         null ``content``, the Messages API's ``stop_reason: "refusal"``
    reasoning            a chat-surface two-step sequence: analysis text, then the answer
    reasoning_only       a server whose reasoning parser put the whole generation in a thinking block:
                         no answer text at all

Knobs: ``surface`` (``chat_completions``, ``responses``, ``messages`` or ``both``, default ``both`` — the
endpoints this client exposes, as ``openai.OpenAI`` exposes the first two and ``anthropic.Anthropic`` the
third), ``winner`` (index into the label alphabet of the option the stub prefers), ``alternatives`` (labels
reported alongside the answer; 1 means "no distribution") and ``cached_tokens`` (what the server reports as
read from its prompt cache; ``None`` models a server that says nothing about it). ``requests`` records what
each call put on the wire — the kwargs with ``extra_body`` merged in, as the SDK merges it — and
``surfaces`` the endpoint each one went to, in the same order.

Self-test with ``python offline_stub.py --check``; probe a real provider with
``python offline_stub.py --live --model <id>``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

SCENARIOS = (
    "logprobs",
    "structured",
    "reject_logprobs",
    "reject_include",
    "no_alternatives",
    "no_responses_route",
    "no_messages_route",
    "reject_schema",
    "reject_format",
    "reject_cache_key",
    "reject_thinking",
    "reject_budget_value",
    "truncated",
    "refusal",
    "reasoning",
    "reasoning_only",
)
SURFACES = ("chat_completions", "responses", "messages", "both")

LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # jevper's single-letter label alphabet
ANSWER_LOGPROB = -0.1  # the sampled label; alternatives trail it, so the softmax is decisive
ANALYSIS = "The message is about a duplicate charge, so it concerns money."
TRACE = "The user says they were charged twice, so this is about money."
REFUSAL = "I cannot help with that."  # a model's own words for a refusal, as OpenAI reports them
SIGNATURE = "sig-stub"  # what an Anthropic thinking block carries for a later replay
SCHEMA_MARKER = "JSON Schema:\n"  # jevper puts the schema in the prompt when the request cannot carry it
PROMPT_TOKENS = 2388  # the measured prompt in jevper's docs/local-servers.md
COMPLETION_TOKENS = 8
CACHED_TOKENS = 1010  # llama.cpp's reuse for a state-varied second call on that prompt


class Rejection(Exception):
    """A refusal jevper classifies instead of retrying: a capability 400, or a 404 for a whole route."""

    def __init__(
        self, message: str = "logprobs are not supported with reasoning models.", status_code: int = 400
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


def _wants_logprobs(kwargs: dict[str, Any]) -> bool:
    return bool(kwargs.get("logprobs") or kwargs.get("top_logprobs"))


def _requested_schema(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """The JSON schema jevper asked for, on either surface."""
    fmt = kwargs.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        return (fmt.get("json_schema") or {}).get("schema")
    text = kwargs.get("text")
    if isinstance(text, dict):
        fmt = text.get("format") or {}
        if fmt.get("type") == "json_schema":
            return fmt.get("schema")
    return None


def _prompt_schema(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """The schema jevper put in the prompt, when the request could not carry one itself.

    Both OpenAI surfaces carry a strict schema in the request where the server accepts it; the Messages
    API has no schema field at all, and a server that refuses the strict schema is re-asked with a plain
    JSON object — in all three cases jevper states the shape in the system prompt instead. A model reads
    it there, so the stub does too: without this a request whose schema is only in the prompt would look
    unanswerable.
    """
    texts: list[str] = []
    system = kwargs.get("system")
    if isinstance(system, str):
        texts.append(system)
    for key in ("messages", "input"):
        for message in kwargs.get(key) or ():
            if isinstance(message, dict) and message.get("role") == "system":
                content = message.get("content")
                if isinstance(content, str):
                    texts.append(content)
    for text in texts:
        _, marker, rest = text.partition(SCHEMA_MARKER)
        if not marker:
            continue
        try:
            schema = json.loads(rest)
        except ValueError:
            continue
        if isinstance(schema, dict):
            return schema
    return None


def _distribution(keys: list[str], winner: int) -> dict[str, float]:
    """A decisive but properly normalized distribution over ``keys``."""
    base = min(0.1, 0.4 / max(1, len(keys) - 1))
    probabilities = {key: base for key in keys}
    probabilities[keys[winner % len(keys)]] = 1.0 - base * (len(keys) - 1)
    return probabilities


def _schema_answer(schema: dict[str, Any], winner: int) -> dict[str, Any] | None:
    """The body a well-behaved model would return for the schema jevper sent."""
    properties = schema.get("properties") or {}
    if "noul" in properties:
        kind = (properties["noul"] or {}).get("type")
        return {"noul": True if kind == "boolean" else 0.8}
    probabilities = (properties.get("probabilities") or {}).get("properties")
    if probabilities:
        return {"probabilities": _distribution(list(probabilities), winner)}
    for field in ("choice", "score"):  # the `discrete` schema
        enum = (properties.get(field) or {}).get("enum")
        if enum:
            return {field: enum[winner % len(enum)]}
    return None


def _usage(cached: int | None = None) -> dict[str, Any]:
    """OpenAI-shaped usage, on both the Chat and the Responses key names.

    ``cached`` is what the server reports as read from its prompt cache; a server that reports nothing
    omits both detail objects, which is how ``usage.cached_tokens`` comes back ``None``.
    """
    usage: dict[str, Any] = {
        "prompt_tokens": PROMPT_TOKENS,
        "completion_tokens": COMPLETION_TOKENS,
        "completion_tokens_details": {"reasoning_tokens": 0},
        "input_tokens": PROMPT_TOKENS,
        "output_tokens": COMPLETION_TOKENS,
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
        usage["input_tokens_details"] = {"cached_tokens": cached}
    return usage


def _chat_body(
    kwargs: dict[str, Any], *, text: str | None = None, body: dict[str, Any] | None = None,
    entries: list[dict[str, Any]] | None = None, stop: str | None = None, cached: int | None = None,
    refusal: str | None = None,
) -> dict[str, Any]:
    if body is not None:
        content = json.dumps(body)
    elif text is not None:
        content = text
    elif entries:
        content = entries[0]["token"]
    else:
        content = ""  # nothing reported: the readout fails on the missing logprobs, as it should
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if refusal is not None:
        # OpenAI's shape for a refusal: the model's words sit beside a null ``content``.
        message = {"role": "assistant", "content": None, "refusal": refusal}
    choice: dict[str, Any] = {
        "index": 0,
        "message": message,
        "finish_reason": stop or "stop",  # "length": the output budget ran out mid-answer
    }
    if entries is not None:
        # An empty list is a provider that ignores the logprob fields: `logprobs: null`, as sent back
        # by compatibility layers that drop the field.
        choice["logprobs"] = {"content": entries} if entries else None
    return {
        "id": "stub-chat", "model": kwargs.get("model"), "choices": [choice], "usage": _usage(cached)
    }


def _responses_body(
    kwargs: dict[str, Any], *, text: str | None = None, body: dict[str, Any] | None = None,
    entries: list[dict[str, Any]] | None = None, stop: str | None = None, cached: int | None = None,
) -> dict[str, Any]:
    if body is not None:
        content = json.dumps(body)
    elif text is not None:
        content = text
    elif entries:
        content = entries[0]["token"]
    else:
        content = ""  # nothing reported: the readout fails on the missing logprobs, as it should
    part: dict[str, Any] = {"type": "output_text", "text": content, "annotations": []}
    if entries:
        part["logprobs"] = entries
    payload: dict[str, Any] = {
        "id": "stub-response",
        "model": kwargs.get("model"),
        "status": "incomplete" if stop else "completed",
        "output": [
            {"id": "msg_stub", "type": "message", "role": "assistant",
             "status": "incomplete" if stop else "completed", "content": [part]}
        ],
        "output_text": content,
        "usage": _usage(cached),
    }
    if stop:
        # This surface reports an early end here rather than in a finish_reason.
        payload["incomplete_details"] = {"reason": "max_output_tokens" if stop == "length" else stop}
    return payload


def _messages_body(
    kwargs: dict[str, Any], *, text: str | None = None, body: dict[str, Any] | None = None,
    thinking: str | None = None, stop: str | None = None, cached: int | None = None,
) -> dict[str, Any]:
    """An Anthropic ``message``: content blocks, and the answer among them rather than beside them.

    A model that was asked to think answers with a thinking block first, as the real ones do, so a request
    carrying ``thinking`` gets one. ``text=None`` with ``thinking`` set models a reasoning parser that put
    the whole generation in the thinking block and returned no text at all.
    """
    content: list[dict[str, Any]] = []
    if thinking is None and kwargs.get("thinking") is not None:
        thinking = TRACE  # asked to think, the model answers with a thinking block first
    if thinking:
        content.append({"type": "thinking", "thinking": thinking, "signature": SIGNATURE})
    if body is not None:
        content.append({"type": "text", "text": json.dumps(body)})
    elif text is not None:
        content.append({"type": "text", "text": text})
    usage: dict[str, Any] = {"input_tokens": PROMPT_TOKENS, "output_tokens": COMPLETION_TOKENS}
    if thinking:
        usage["output_tokens_details"] = {"thinking_tokens": 12}
    if cached is not None:
        # Anthropic's own name for what it read from its prompt cache.
        usage["cache_read_input_tokens"] = cached
    return {
        "id": "stub-message",
        "type": "message",
        "role": "assistant",
        "model": kwargs.get("model"),
        "content": content,
        "stop_reason": stop or "end_turn",
        "stop_sequence": None,
        "usage": usage,
    }


class _Endpoint:
    """One ``create`` method; the surface only decides the body shape."""

    def __init__(self, stub: StubClient, surface: str) -> None:
        self._stub = stub
        self._surface = surface

    def create(self, **kwargs: Any) -> dict[str, Any]:
        return self._stub.reply(kwargs, self._surface)


class _Namespace:
    def __init__(self, **children: Any) -> None:
        self.__dict__.update(children)


class StubClient:
    """A duck-typed client that answers jevper from canned bodies and records every request."""

    def __init__(
        self,
        scenario: str = "structured",
        *,
        surface: str = "both",
        winner: int = 0,
        alternatives: int | None = None,
        cached_tokens: int | None = CACHED_TOKENS,
    ) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")
        if surface not in SURFACES:
            raise ValueError(f"surface must be one of {SURFACES}, got {surface!r}")
        self.scenario = scenario
        self.surface = surface
        self.winner = winner
        self.alternatives = alternatives
        self.cached_tokens = cached_tokens
        self.requests: list[dict[str, Any]] = []
        self.surfaces: list[str] = []
        if surface in ("chat_completions", "both"):
            self.chat = _Namespace(completions=_Endpoint(self, "chat_completions"))
        if surface in ("responses", "both"):
            self.responses = _Endpoint(self, "responses")
        if surface == "messages":
            # What ``anthropic.Anthropic`` exposes, and nothing else: this client has no logprob surface.
            self.messages = _Endpoint(self, "messages")

    # -- internals ---------------------------------------------------------------------------------

    def _entries(self, kwargs: dict[str, Any], alternatives: int | None = None) -> list[dict[str, Any]]:
        count = self.alternatives if alternatives is None else alternatives
        if count is None:
            requested = kwargs.get("top_logprobs")
            count = 20 if requested is None else int(requested)
        count = max(1, min(int(count), len(LABELS)))
        answer = LABELS[self.winner % len(LABELS)]
        tops = [{"token": answer, "logprob": ANSWER_LOGPROB}]
        for index, label in enumerate(LABELS[:count]):
            if label != answer:
                tops.append({"token": label, "logprob": -2.0 - 0.6 * index})
        return [{"token": answer, "logprob": ANSWER_LOGPROB, "top_logprobs": tops}]

    def reply(self, kwargs: dict[str, Any], surface: str) -> dict[str, Any]:
        # The SDK merges ``extra_body`` into the request *after* the typed parameters, so what reaches
        # the wire is the union with the caller's keys winning. Record that rather than the SDK's calling
        # convention: a check asking what the server saw should not have to merge it itself.
        extra = kwargs.pop("extra_body", None)
        if isinstance(extra, dict):
            kwargs = {**kwargs, **extra}
        self.requests.append(kwargs)
        self.surfaces.append(surface)

        if self.scenario == "no_responses_route" and surface == "responses":
            # A server that never implemented the route: a 404 that says nothing about the model.
            raise Rejection("Not Found", status_code=404)
        if self.scenario == "no_messages_route" and surface == "messages":
            # A server that never implemented Anthropic's route. With a Messages-only client there is
            # nowhere to move, so this is the 404 jevper must keep reporting, call after call.
            raise Rejection("Not Found", status_code=404)

        schema = _requested_schema(kwargs) or _prompt_schema(kwargs)
        if self.scenario == "reject_schema" and _requested_schema(kwargs) is not None:
            # The strict schema is refused; ``json_object`` is not, so the ladder has all three rungs.
            raise Rejection("response_format is not supported by this server.")
        if self.scenario == "reject_format" and ("response_format" in kwargs or "text" in kwargs):
            # A server with no format field at all: it refuses ``json_object`` as well as a schema.
            raise Rejection("response_format is not supported by this server.")
        if self.scenario == "reject_cache_key" and kwargs.get("prompt_cache_key") is not None:
            raise Rejection("prompt_cache_key is not supported by this server.")
        if self.scenario == "reject_thinking" and kwargs.get("thinking") is not None:
            # SGLang's protocol has no ``thinking`` field: the request is refused, and jevper drops it.
            raise Rejection("thinking is not supported by this server.")
        if self.scenario == "reject_budget_value" and kwargs.get("thinking") is not None:
            # SGLang again, but refusing the number rather than the field: not a capability verdict.
            raise Rejection("budget_tokens: must be at least 1024")
        if self.scenario == "reject_include" and surface == "responses" and kwargs.get("include"):
            # OpenRouter's Responses API refuses the logprob includable outright, without ever writing
            # the word "logprob"; the same server carries the distribution on Chat Completions.
            raise Rejection('Invalid option: expected one of "chat", "message". at path: ["include", 0]')
        if self.scenario == "reject_logprobs" and _wants_logprobs(kwargs):
            raise Rejection()

        body = {
            "chat_completions": _chat_body,
            "responses": _responses_body,
            "messages": _messages_body,
        }[surface]

        if self.scenario == "truncated":
            # Generation stopped before an answer existed — the stop reason is the actionable fact, and
            # each surface spells it its own way. On the Messages surface thinking is a budget, so this
            # is what a spent budget with thinking on looks like there: a thinking block, no text.
            if surface == "messages":
                return _messages_body(kwargs, thinking=TRACE, stop="max_tokens", cached=self.cached_tokens)
            return body(kwargs, text="", stop="length", cached=self.cached_tokens)

        if self.scenario == "refusal":
            # A refusal instead of an answer. Only the Chat surface carries the model's own words.
            if surface == "chat_completions":
                return _chat_body(kwargs, refusal=REFUSAL, cached=self.cached_tokens)
            if surface == "messages":
                return _messages_body(kwargs, stop="refusal", cached=self.cached_tokens)
            return _responses_body(kwargs, text="", stop="refusal", cached=self.cached_tokens)

        if self.scenario == "reasoning_only":
            # The reasoning parser classified the whole generation as thinking: no text block to read.
            return _messages_body(kwargs, thinking=TRACE, cached=self.cached_tokens)

        if self.scenario == "reasoning" and schema is None and not _wants_logprobs(kwargs):
            return body(kwargs, text=ANALYSIS, cached=self.cached_tokens)  # the analysis pass

        if _wants_logprobs(kwargs):
            if self.scenario == "no_alternatives":
                return body(
                    kwargs, entries=self._entries(kwargs, alternatives=1), cached=self.cached_tokens
                )
            if self.scenario in ("logprobs", "reasoning", "no_responses_route", "reject_include"):
                return body(kwargs, entries=self._entries(kwargs), cached=self.cached_tokens)
            return body(kwargs, entries=[], cached=self.cached_tokens)  # no logprobs to give back

        answer = _schema_answer(schema, self.winner) if schema is not None else None
        if answer is not None:
            return body(kwargs, body=answer, cached=self.cached_tokens)
        return body(kwargs, text="A", cached=self.cached_tokens)  # a request jevper reads no answer out of


# -- self-test ---------------------------------------------------------------------------------------


def _run_checks() -> int:
    from jevper import (
        Choice,
        JevperError,
        LabelReadoutError,
        MalformedAnswerError,
        ProviderError,
        ReasoningConfig,
        SystemOneClient,
        UnsupportedMethodError,
        reasoning_text,
    )

    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  [{detail}]"))

    def question() -> Choice:
        return Choice(
            instructions="Pick the intent of the message.",
            criteria={
                "billing": "money, invoices, refunds, charges",
                "technical": "errors, crashes, login or performance problems",
                "sales": "pricing, plans, purchasing, upgrades",
            },
        )

    state = "I was charged twice for the same subscription this month."

    # A provider with real logprobs: auto stays on logprobs and the distribution is a real one.
    stub = StubClient(scenario="logprobs")
    response = SystemOneClient(stub, model="stub-model").system_one(state=state, questions={"intent": question()})
    answer = response.answers["intent"]
    check("logprobs: auto resolves to logprobs", response.debug["methods"]["intent"] == "logprobs",
          str(response.debug["methods"]))
    check("logprobs: readout comes from the token distribution",
          response.debug["llm_attempts"][-1]["readout"]["source"] == "logprobs")
    check("logprobs: the winner is the first option", answer.choice == "billing", repr(answer.choice))
    check("logprobs: distribution covers every option and sums to 1",
          sorted(answer.probabilities) == ["billing", "sales", "technical"]
          and abs(sum(answer.probabilities.values()) - 1.0) < 1e-9)
    check("logprobs: one provider call", response.usage.n_calls == 1, str(response.usage.n_calls))
    check("logprobs: the request asked for the alternatives",
          stub.requests[0].get("top_logprobs") is not None, str(stub.requests[0]))

    # A provider that only does JSON: same code, structured readout.
    stub = StubClient(scenario="structured")
    response = SystemOneClient(stub, model="stub-model").system_one(state=state, questions={"intent": question()})
    answer = response.answers["intent"]
    check("structured: auto resolves to structured", response.debug["methods"]["intent"] == "structured")
    check("structured: the distribution sums to 1", abs(sum(answer.probabilities.values()) - 1.0) < 1e-9)
    check("structured: no normalization warning on a clean answer",
          response.debug["probability_errors"] == {}, str(response.debug["probability_errors"]))

    # A provider that rejects the logprob fields on both surfaces: auto falls back, and remembers.
    stub = StubClient(scenario="reject_logprobs")
    client = SystemOneClient(stub, model="stub-model")
    response = client.system_one(state=state, questions={"intent": question()})
    check("reject_logprobs: auto falls back to structured",
          response.debug["methods"]["intent"] == "structured", str(response.debug["methods"]))
    check("reject_logprobs: it asked for logprobs first", stub.requests[0].get("top_logprobs") is not None,
          str(stub.requests[0]))
    check("reject_logprobs: the fallback is recorded", bool(response.debug["retry_reasons"]))
    check("reject_logprobs: the surface move is tried first",
          any("retrying the label readout on api='chat_completions'" in reason
              for reason in response.debug["retry_reasons"]), str(response.debug["retry_reasons"]))
    check("reject_logprobs: three requests, two of them rejected",
          len(stub.requests) == 3 and len(response.debug["llm_attempts"]) == 3,
          f"requests={len(stub.requests)} attempts={len(response.debug['llm_attempts'])}")
    check("reject_logprobs: responses refused, chat answered",
          stub.surfaces == ["responses", "chat_completions", "chat_completions"]
          and response.debug["api"] == "chat_completions",
          f"surfaces={stub.surfaces} api={response.debug['api']}")
    check("reject_logprobs: a rejected call is not counted as a call",
          response.usage.n_calls == 1, str(response.usage.n_calls))
    before = len(stub.requests)
    client.system_one(state=state, questions={"intent": question()})
    check("reject_logprobs: the verdict is remembered per model and surface",
          len(stub.requests) == before + 1 and stub.requests[-1].get("logprobs") is None,
          str(stub.requests[-1]))

    # A single-surface client cannot move: the fallback is the method, and the counts drop.
    stub = StubClient(scenario="reject_logprobs", surface="chat_completions")
    response = SystemOneClient(stub, model="stub-model").system_one(state=state, questions={"intent": question()})
    check("reject_logprobs: one surface means two requests and no move",
          len(stub.requests) == 2 and response.usage.n_calls == 1
          and response.debug["methods"]["intent"] == "structured",
          f"requests={len(stub.requests)} n_calls={response.usage.n_calls}")

    # A provider that reports the sampled token and nothing else: no distribution to read.
    stub = StubClient(scenario="no_alternatives")
    response = SystemOneClient(stub, model="stub-model").system_one(state=state, questions={"intent": question()})
    check("no_alternatives: auto falls back to structured",
          response.debug["methods"]["intent"] == "structured", str(response.debug["methods"]))
    check("no_alternatives: three answered requests, every one counted",
          len(stub.requests) == 3 and response.usage.n_calls == 3,
          f"requests={len(stub.requests)} n_calls={response.usage.n_calls}")

    stub = StubClient(scenario="no_alternatives", surface="chat_completions")
    response = SystemOneClient(stub, model="stub-model").system_one(state=state, questions={"intent": question()})
    check("no_alternatives: one surface means two answered requests",
          len(stub.requests) == 2 and response.usage.n_calls == 2,
          f"requests={len(stub.requests)} n_calls={response.usage.n_calls}")

    # A server with no Responses route: auto re-asks on Chat Completions and remembers the route.
    stub = StubClient(scenario="no_responses_route")
    client = SystemOneClient(stub, model="stub-model")

    response = client.system_one(state=state, questions={"intent": question()})
    check("no_responses_route: auto re-asks on chat_completions",
          response.debug["api"] == "chat_completions" and response.usage.n_calls == 1
          and stub.surfaces == ["responses", "chat_completions"],
          f"api={response.debug['api']} n_calls={response.usage.n_calls} surfaces={stub.surfaces}")
    check("no_responses_route: the readout is unharmed by the move",
          response.debug["methods"]["intent"] == "logprobs"
          and response.answers["intent"].choice == "billing",
          str(response.debug["methods"]))
    before = len(stub.requests)
    client.system_one(state=state, questions={"intent": question()})
    check("no_responses_route: the missing route is remembered",
          len(stub.requests) == before + 1 and stub.surfaces[-1] == "chat_completions",
          str(stub.surfaces[-2:]))

    # A Messages-only client whose one route is missing: there is nowhere to move, so every call must
    # keep reporting the 404 the first one found — never an AttributeError for a surface it cannot speak.
    stub = StubClient(scenario="no_messages_route", surface="messages")
    client = SystemOneClient(stub, model="stub-model")
    reported = []
    for _ in range(2):
        try:
            client.system_one(state=state, questions={"intent": question()})
        except ProviderError as exc:
            reported.append((exc.status_code, "no 'messages' route" in str(exc)))
        else:
            reported.append((None, False))
    check("no_messages_route: a client with one surface reports the 404 on every call",
          reported == [(404, True), (404, True)] and stub.surfaces == ["messages", "messages"],
          f"reported={reported} surfaces={stub.surfaces}")

    # OpenRouter's shape: the Responses route refuses the logprob includable, Chat carries it. A pinned
    # method="logprobs" asked for a distribution, not for a surface, so the readout moves and keeps it.
    stub = StubClient(scenario="reject_include")
    client = SystemOneClient(stub, model="stub-model", method="logprobs")
    response = client.system_one(state=state, questions={"intent": question()})
    check("reject_include: a pinned logprobs readout moves to the surface that carries it",
          response.debug["api"] == "chat_completions" and response.debug["method"] == "logprobs"
          and stub.surfaces == ["responses", "chat_completions"]
          and response.answers["intent"].choice == "billing",
          f"api={response.debug['api']} method={response.debug['method']} surfaces={stub.surfaces}")
    before = len(stub.requests)
    client.system_one(state=state, questions={"intent": question()})
    check("reject_include: the move is remembered, so the refusal is paid once",
          len(stub.requests) == before + 1 and stub.surfaces[-1] == "chat_completions",
          str(stub.surfaces[-2:]))

    # With nowhere to move, the provider's refusal is the answer: the method is never swapped for JSON.
    stub = StubClient(scenario="reject_include", surface="responses")
    try:
        SystemOneClient(stub, model="stub-model", method="logprobs").system_one(
            state=state, questions={"intent": question()}
        )
    except LabelReadoutError as exc:
        refusal = "rejected the logprob request" in str(exc)
    else:
        refusal = False
    check("reject_include: with nowhere to move the refusal is reported, not another readout",
          refusal and stub.surfaces == ["responses"], f"surfaces={stub.surfaces}")

    # A grammar request is a Chat Completions convention, so a logprob verdict is no reason to move it.
    stub = StubClient(scenario="reject_logprobs")
    try:
        SystemOneClient(stub, model="stub-model", method="grammar").system_one(
            state=state, questions={"intent": question()}
        )
    except LabelReadoutError:
        stayed = stub.surfaces == ["chat_completions"]
    else:
        stayed = False
    check("reject_logprobs: a grammar readout stays on chat, where the grammar lives",
          stayed and "grammar" in stub.requests[0], str(stub.surfaces))

    # A server that refuses a strict schema: the ladder drops to `json_object` and answers anyway.
    stub = StubClient(scenario="reject_schema")
    response = SystemOneClient(stub, model="stub-model", method="structured").system_one(
        state=state, questions={"intent": question()}
    )
    check("reject_schema: the schema is dropped and the call re-asked",
          len(stub.requests) == 2 and response.usage.n_calls == 1
          and ((stub.requests[1].get("text") or {}).get("format") or {}).get("type") == "json_object",
          f"requests={len(stub.requests)} n_calls={response.usage.n_calls}")
    check("reject_schema: the server's limit is reported",
          response.debug["server_limits"]["structured"] == "object",
          str(response.debug.get("server_limits")))
    check("reject_schema: the answer still arrives",
          response.answers["intent"].choice == "billing", repr(response.answers["intent"].choice))

    # A server that refuses the cache key: optional field, dropped, and every later call leaves it out.
    stub = StubClient(scenario="reject_cache_key")
    response = SystemOneClient(stub, model="stub-model", method="structured").system_one(
        state=state, questions={"intent": question()}
    )
    check("reject_cache_key: the key is dropped and the answer arrives",
          response.debug["server_limits"]["cache_key"] is False
          and response.answers["intent"].choice == "billing",
          str(response.debug.get("server_limits")))
    check("reject_cache_key: the re-ask carries no key",
          len(stub.requests) == 2 and stub.requests[1].get("prompt_cache_key") is None,
          str(stub.requests[1].get("prompt_cache_key")))

    # Caching: every request is routed, the key follows the rubric (not the state), and the usage
    # reports what the provider read from its cache.
    stub = StubClient(scenario="structured", surface="chat_completions")
    client = SystemOneClient(stub, model="stub-model", method="structured")
    first = client.system_one(state="state one", questions={"intent": question()})
    second = client.system_one(state="state two", questions={"intent": question()})
    other = client.system_one(
        state="state one",
        questions={"topic": Choice(criteria={"a": "one", "b": "two"})},
    )
    keys = [request.get("prompt_cache_key") for request in stub.requests]
    check("caching: every request carries a prompt_cache_key",
          len(keys) == 3 and all(isinstance(key, str) and key for key in keys), str(keys))
    check("caching: the key is stable across states and follows the rubric",
          keys[0] == keys[1] != keys[2] and other.answers["topic"].choice == "a", str(keys))
    check("caching: the key is derived, not positional",
          all(key.startswith("jevper-") and len(key) == len("jevper-") + 32 for key in keys), str(keys))
    check("caching: cached_tokens is read from usage",
          first.usage.cached_tokens == CACHED_TOKENS and second.usage.cached_tokens == CACHED_TOKENS,
          f"{first.usage.cached_tokens}/{second.usage.cached_tokens}")

    stub = StubClient(scenario="structured", surface="chat_completions", cached_tokens=None)
    response = SystemOneClient(stub, model="stub-model", method="structured").system_one(
        state="state one", questions={"intent": question()}
    )
    check("caching: a server that reports nothing stays None",
          response.usage.cached_tokens is None, repr(response.usage.cached_tokens))

    # A call with no key of its own: the derived key is what reaches the provider.
    stub = StubClient(scenario="structured", surface="chat_completions")
    SystemOneClient(stub, model="stub-model", method="structured", prompt_cache_key="tenant-a").system_one(
        state="state one", questions={"intent": question()}
    )
    check("caching: a caller's key wins over the derived one",
          stub.requests[0].get("prompt_cache_key") == "tenant-a",
          str(stub.requests[0].get("prompt_cache_key")))

    stub = StubClient(scenario="structured", surface="chat_completions")
    try:
        SystemOneClient(stub, model="stub-model", prompt_cache_key=" ")
    except JevperError:
        check("caching: a blank key is refused before any request", not stub.requests)
    else:
        check("caching: a blank key is refused before any request", False, "no error")

    # The method belongs in the key too: its system prompt and answer shape are part of the cached prefix,
    # so a logprobs request must not be routed into a structured one's bucket.
    stub = StubClient(scenario="logprobs", surface="chat_completions")
    client = SystemOneClient(stub, model="stub-model")
    client.system_one(state=state, questions={"intent": question()}, method="logprobs")
    client.system_one(state=state, questions={"intent": question()}, method="structured")
    keys = [request.get("prompt_cache_key") for request in stub.requests]
    check("caching: the method is part of the key",
          len(keys) == 2 and keys[0] != keys[1] and all(key.startswith("jevper-") for key in keys),
          str(keys))

    # Message order: the state goes last so one rubric's requests share a cacheable prefix — except when
    # the state's own last turn is the assistant's, which no server will read as a question.
    stub = StubClient(scenario="structured", surface="chat_completions")
    SystemOneClient(stub, model="stub-model", method="structured").system_one(
        state=state, questions={"intent": question()}
    )
    turns = stub.requests[0]["messages"]
    check("message order: the state goes last, the question before it",
          turns[-1]["role"] == "user" and state in turns[-1]["content"],
          str([message.get("role") for message in turns]))

    stub = StubClient(scenario="structured", surface="chat_completions")
    SystemOneClient(stub, model="stub-model", method="structured").system_one(
        state=[
            {"role": "user", "content": "Where is my refund?"},
            {"role": "assistant", "content": "Let me look that up."},
        ],
        questions={"intent": question()},
    )
    turns = stub.requests[0]["messages"]
    closing = [index for index, message in enumerate(turns) if message.get("content") == "Let me look that up."]
    check("message order: a state ending on the assistant's turn still ends on the question",
          turns[-1]["role"] == "user" and closing and closing[0] == len(turns) - 2,
          str([message.get("role") for message in turns]))

    # An answer that never arrived: the error names the budget that ran out.
    stub = StubClient(scenario="truncated")
    try:
        SystemOneClient(stub, model="stub-model", method="structured").system_one(
            state=state, questions={"intent": question()}
        )
    except MalformedAnswerError as exc:
        text = str(exc)
        check("truncated: the error names the output budget",
              "ran out of output tokens" in text and "max_tokens" in text, text)
        check("truncated: the corrective retry is spent before raising", len(stub.requests) == 2,
              str(len(stub.requests)))
    except JevperError as exc:  # pragma: no cover - the wrong error type
        check("truncated: the error names the output budget", False, repr(exc))
    else:  # pragma: no cover
        check("truncated: the error names the output budget", False, "no error")

    # Pinning logprobs against that provider is the trap the docs warn about: it raises, spends the one
    # surface move it is allowed, and is never swapped for the structured readout that would have worked.
    stub = StubClient(scenario="structured")
    try:
        SystemOneClient(stub, model="stub-model", method="logprobs").system_one(
            state=state, questions={"intent": question()}
        )
    except LabelReadoutError as exc:
        check("pinned logprobs without provider support raises LabelReadoutError", True)
        check("pinned logprobs spends the surface move and nothing else",
              len(stub.requests) == 2 and stub.surfaces == ["responses", "chat_completions"],
              f"requests={len(stub.requests)} surfaces={stub.surfaces}")
        check("pinned logprobs names the alternatives", "structured" in str(exc), str(exc))
    except JevperError as exc:  # pragma: no cover - the wrong error type
        check("pinned logprobs without provider support raises LabelReadoutError", False, repr(exc))
    else:  # pragma: no cover
        check("pinned logprobs without provider support raises LabelReadoutError", False, "no error")

    # Two-step reasoning on the chat surface: an analysis call, then the answer call.
    stub = StubClient(scenario="reasoning", surface="chat_completions")
    response = SystemOneClient(
        stub, model="stub-model", reasoning=ReasoningConfig(effort="low")
    ).system_one(state=state, questions={"intent": question()})
    check("reasoning: two-step on the chat surface", response.debug["reasoning_mode"] == "two_step",
          str(response.debug["reasoning_mode"]))
    check("reasoning: two provider calls", response.usage.n_calls == 2, str(response.usage.n_calls))
    check("reasoning: the trace is readable", bool(reasoning_text(response.reasoning)))
    check("reasoning: the answer still carries a distribution",
          abs(sum(response.answers["intent"].probabilities.values()) - 1.0) < 1e-9)

    # The schema travels in the prompt wherever the request cannot carry it: always on the Messages
    # surface, and on the OpenAI surfaces once the strict schema has been dropped.
    stub = StubClient(scenario="structured", surface="chat_completions")
    response = SystemOneClient(
        stub, model="stub-model", method="structured", structured_outputs=False
    ).system_one(state=state, questions={"intent": question()})
    check("structured_outputs=False: json_object in the request, the schema in the prompt",
          stub.requests[0].get("response_format") == {"type": "json_object"}
          and "validates against this JSON Schema" in stub.requests[0]["messages"][0]["content"],
          str(stub.requests[0].get("response_format")))
    check("structured_outputs=False: the answer still follows the schema",
          response.answers["intent"].choice == "billing", repr(response.answers["intent"].choice))

    # The Messages surface: no logprobs in the protocol at all, no schema field, a required max_tokens
    # and thinking as a budget. `auto` starts in JSON there rather than paying to discover that.
    stub = StubClient(scenario="structured", surface="messages")
    response = SystemOneClient(stub, model="stub-model").system_one(state=state, questions={"intent": question()})
    request = stub.requests[0]
    check("messages: auto answers in JSON without a probe",
          response.debug["api"] == "messages" and response.debug["methods"]["intent"] == "structured"
          and len(stub.requests) == 1 and response.usage.n_calls == 1,
          f"api={response.debug['api']} requests={len(stub.requests)}")
    check("messages: the schema travels in the system prompt, and no schema field is sent",
          "validates against this JSON Schema" in request.get("system", "")
          and "response_format" not in request and "text" not in request,
          str(request.get("system"))[:80])
    check("messages: the system prompt is a top-level field, not a turn",
          all(message.get("role") != "system" for message in request["messages"]),
          str([message.get("role") for message in request["messages"]]))
    check("messages: max_tokens is sent, 1024 by default",
          request.get("max_tokens") == 1024, str(request.get("max_tokens")))
    check("messages: the answer is read from the text blocks",
          response.answers["intent"].choice == "billing", repr(response.answers["intent"].choice))
    check("messages: cache_read_input_tokens becomes cached_tokens",
          response.usage.cached_tokens == CACHED_TOKENS, repr(response.usage.cached_tokens))

    stub = StubClient(scenario="structured", surface="messages")
    response = SystemOneClient(
        stub,
        model="stub-model",
        reasoning=ReasoningConfig(mode="native", budget_tokens=1024),
        extra_body={"max_tokens": 2048},
    ).system_one(state=state, questions={"intent": question()})
    request = stub.requests[0]
    check("messages: thinking is a budget, and extra_body raises max_tokens",
          request.get("thinking") == {"type": "enabled", "budget_tokens": 1024}
          and request.get("max_tokens") == 2048,
          f"thinking={request.get('thinking')} max_tokens={request.get('max_tokens')}")
    check("messages: a thinking block becomes a reasoning part, signature kept",
          reasoning_text(response.reasoning) == TRACE
          and response.reasoning[0].signature == SIGNATURE,
          str(response.reasoning))

    # A server whose protocol has no thinking field at all (vLLM's route): dropped, re-asked, reported.
    stub = StubClient(scenario="reject_thinking", surface="messages")
    response = SystemOneClient(
        stub, model="stub-model", method="structured",
        reasoning=ReasoningConfig(mode="native", budget_tokens=1024),
    ).system_one(state=state, questions={"intent": question()})
    check("reject_thinking: the field is dropped and the limit reported",
          response.debug["server_limits"]["thinking"] is False
          and response.answers["intent"].choice == "billing",
          str(response.debug.get("server_limits")))
    check("reject_thinking: the re-ask carries no thinking",
          len(stub.requests) == 2 and stub.requests[1].get("thinking") is None,
          str(stub.requests[1].get("thinking")))

    # A server that knows the thinking field and refuses the *value* in it: a bad number is not a missing
    # field, so nothing is dropped, nothing is remembered, and the provider's own error travels back.
    stub = StubClient(scenario="reject_budget_value", surface="messages")
    try:
        SystemOneClient(
            stub,
            model="stub-model",
            method="structured",
            reasoning=ReasoningConfig(mode="native", budget_tokens=1024),
        ).system_one(state=state, questions={"intent": question()})
    except ProviderError as exc:
        check("reject_budget_value: the provider's own error travels back",
              exc.status_code == 400 and "must be at least 1024" in str(exc), f"{exc.status_code}: {exc}")
        check("reject_budget_value: no re-ask is spent and no limit is remembered",
              len(stub.requests) == 1 and stub.requests[0].get("thinking") is not None,
              f"requests={len(stub.requests)}")
    except JevperError as exc:  # pragma: no cover - the wrong error type
        check("reject_budget_value: the provider's own error travels back", False, repr(exc))
    else:  # pragma: no cover
        check("reject_budget_value: the provider's own error travels back", False, "no error")

    # extra_body is merged after jevper's own parameters, so a key named there is what reaches the wire —
    # including a capability field, which is dropped with jevper's own so the re-ask really re-asks.
    stub = StubClient(scenario="structured", surface="messages")
    SystemOneClient(
        stub,
        model="stub-model",
        method="structured",
        reasoning=ReasoningConfig(mode="native", budget_tokens=1024),
    ).system_one(state=state, questions={"intent": question()})
    check("messages: the default max_tokens grows by the thinking budget",
          stub.requests[0].get("max_tokens") == 2048, str(stub.requests[0].get("max_tokens")))

    stub = StubClient(scenario="structured", surface="messages")
    SystemOneClient(
        stub,
        model="stub-model",
        method="structured",
        reasoning=ReasoningConfig(mode="native", budget_tokens=1024),
        extra_body={"max_tokens": 4096},
    ).system_one(state=state, questions={"intent": question()})
    check("messages: a caller's max_tokens in extra_body still wins outright",
          stub.requests[0].get("max_tokens") == 4096, str(stub.requests[0].get("max_tokens")))

    stub = StubClient(scenario="reject_thinking", surface="messages")
    response = SystemOneClient(
        stub,
        model="stub-model",
        method="structured",
        reasoning=ReasoningConfig(mode="native", budget_tokens=1024),
        extra_body={"thinking": {"type": "enabled", "budget_tokens": 512}},
    ).system_one(state=state, questions={"intent": question()})
    check("reject_thinking: the caller's own copy of the field is dropped with jevper's",
          response.debug["server_limits"]["thinking"] is False
          and len(stub.requests) == 2 and stub.requests[1].get("thinking") is None,
          str([request.get("thinking") for request in stub.requests]))

    # A model that refuses instead of answering: the message says so, in the model's own words where the
    # surface carries them, rather than reading as a malformed JSON object.
    stub = StubClient(scenario="refusal", surface="chat_completions")
    try:
        SystemOneClient(stub, model="stub-model", method="structured").system_one(
            state=state, questions={"intent": question()}
        )
    except MalformedAnswerError as exc:
        check("refusal: the error says the model refused, in its own words",
              "refused to answer" in str(exc) and REFUSAL in str(exc), str(exc))
    except JevperError as exc:  # pragma: no cover - the wrong error type
        check("refusal: the error says the model refused, in its own words", False, repr(exc))
    else:  # pragma: no cover
        check("refusal: the error says the model refused, in its own words", False, "no error")

    stub = StubClient(scenario="refusal", surface="messages")
    try:
        SystemOneClient(stub, model="stub-model").system_one(state=state, questions={"intent": question()})
    except MalformedAnswerError as exc:
        check("refusal: the Messages stop_reason reads as a refusal too",
              "refused to answer" in str(exc), str(exc))
    except JevperError as exc:  # pragma: no cover - the wrong error type
        check("refusal: the Messages stop_reason reads as a refusal too", False, repr(exc))
    else:  # pragma: no cover
        check("refusal: the Messages stop_reason reads as a refusal too", False, "no error")

    # The same spent budget on the Messages surface, where this surface calls it "max_tokens".
    stub = StubClient(scenario="truncated", surface="messages")
    try:
        SystemOneClient(stub, model="stub-model").system_one(state=state, questions={"intent": question()})
    except MalformedAnswerError as exc:
        check("truncated: the Messages stop_reason names the output budget too",
              "ran out of output tokens" in str(exc) and "max_tokens" in str(exc), str(exc))
    except JevperError as exc:  # pragma: no cover - the wrong error type
        check("truncated: the Messages stop_reason names the output budget too", False, repr(exc))
    else:  # pragma: no cover
        check("truncated: the Messages stop_reason names the output budget too", False, "no error")

    # With structured_outputs=False the request already carries a plain json_object, so a server that
    # refuses any format field goes straight to "no format field" instead of spending a call re-sending
    # the same bytes — while the default shape walks the whole ladder down to the same place.
    stub = StubClient(scenario="reject_format")
    response = SystemOneClient(
        stub, model="stub-model", method="structured", structured_outputs=False
    ).system_one(state=state, questions={"intent": question()})
    check("structured_outputs=False: a refused format field skips the json_object rung",
          response.debug["server_limits"]["structured"] == "none" and len(stub.requests) == 2,
          f"limits={response.debug.get('server_limits')} requests={len(stub.requests)}")

    stub = StubClient(scenario="reject_format")
    response = SystemOneClient(stub, model="stub-model", method="structured").system_one(
        state=state, questions={"intent": question()}
    )
    check("reject_format: the default shape walks every rung, and the answer still arrives",
          response.debug["server_limits"]["structured"] == "none" and len(stub.requests) == 3
          and response.answers["intent"].choice == "billing",
          f"limits={response.debug.get('server_limits')} requests={len(stub.requests)}")

    # A label readout on this surface is refused before the request: the API has no logprob field.
    stub = StubClient(scenario="structured", surface="messages")
    try:
        SystemOneClient(stub, model="stub-model", method="logprobs", api="messages").system_one(
            state=state, questions={"intent": question()}
        )
    except UnsupportedMethodError as exc:
        check("messages: a pinned logprobs is refused before any request",
              not stub.requests and "structured" in str(exc), str(exc))
    except JevperError as exc:  # pragma: no cover - the wrong error type
        check("messages: a pinned logprobs is refused before any request", False, repr(exc))
    else:  # pragma: no cover
        check("messages: a pinned logprobs is refused before any request", False, "no error")

    # A reasoning parser that put the whole generation in a thinking block: no answer text to read.
    stub = StubClient(scenario="reasoning_only", surface="messages")
    try:
        SystemOneClient(stub, model="stub-model", method="structured").system_one(
            state=state, questions={"intent": question()}
        )
    except MalformedAnswerError as exc:
        check("reasoning_only: the error says the response carried reasoning only",
              "carried reasoning only" in str(exc), str(exc))
        check("reasoning_only: the corrective retry is spent first", len(stub.requests) == 2,
              str(len(stub.requests)))
    except JevperError as exc:  # pragma: no cover - the wrong error type
        check("reasoning_only: the error says the response carried reasoning only", False, repr(exc))
    else:  # pragma: no cover
        check("reasoning_only: the error says the response carried reasoning only", False, "no error")

    print()
    print(f"{'all checks passed' if not failures else f'{failures} check(s) failed'}")
    return 1 if failures else 0


# -- live probe --------------------------------------------------------------------------------------


def _run_live(model: str, api: str) -> int:
    if api == "messages":
        # The Anthropic SDK speaks this one; a local server takes any key, so only the base URL matters.
        try:
            from anthropic import Anthropic
        except ImportError:
            print("--api messages needs the anthropic package: pip install anthropic")
            return 2
        base_url = os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        probe_client: Any = Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY") or "local",
            **({"base_url": base_url} if base_url else {}),
        )
    else:
        try:
            from openai import OpenAI
        except ImportError:
            print("the live probe needs the openai package: pip install openai")
            return 2
        if not os.environ.get("OPENAI_API_KEY"):
            print("the live probe needs OPENAI_API_KEY (set OPENAI_BASE_URL for a self-hosted server)")
            return 2
        probe_client = OpenAI()

    from jevper import Choice, JevperError, SystemOneClient

    question = Choice(
        instructions="Pick the intent of the message.",
        criteria={
            "billing": "money, invoices, refunds, charges",
            "technical": "errors, crashes, login or performance problems",
            "sales": "pricing, plans, purchasing, upgrades",
        },
    )
    client = SystemOneClient(probe_client, model=model, api=api)
    try:
        response = client.system_one(
            state="I was charged twice for the same subscription this month.", questions={"intent": question}
        )
    except JevperError as exc:
        print(f"FAILED  {type(exc).__name__}: {exc}")
        return 1

    print(json.dumps({
        "model": response.model,
        "method_per_question": response.debug.get("methods", response.debug["method"]),
        "api": response.debug["api"],
        "readout_source": response.debug["llm_attempts"][-1]["readout"]["source"],
        "answer": response.answers["intent"].model_dump(mode="json"),
        "n_calls": response.usage.n_calls,
        "cached_tokens": response.usage.cached_tokens,
        "server_limits": response.debug.get("server_limits"),
        "retry_reasons": response.debug["retry_reasons"],
    }, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="run the offline self-test")
    parser.add_argument("--live", action="store_true", help="ask a real provider which method it resolves to")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", ""), help="model id for --live")
    parser.add_argument("--api", default="auto", choices=("auto", "chat_completions", "responses", "messages"))
    args = parser.parse_args(argv)

    if args.live:
        if not args.model:
            print("--live needs --model <id> (or LLM_MODEL)")
            return 2
        return _run_live(args.model, args.api)
    if args.check:
        return _run_checks()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
