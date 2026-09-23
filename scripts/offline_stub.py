"""An offline stand-in for an OpenAI-compatible client, for jevper code.

Why: jevper is duck-typed, so a plain object is enough to drive it end to end — no HTTP, no API key, no
tokens spent — while the real readout path (logprobs softmax, structured JSON, ``auto``'s fallback) runs
for real.

    from offline_stub import StubClient
    from jevper import Choice, SystemOneClient

    stub = StubClient(scenario="reject_logprobs")        # a provider that 400s the logprob fields
    client = SystemOneClient(stub, model="stub-model")   # method="auto" falls back to structured
    response = client.system_one(state="...", questions={"intent": Choice(criteria={"a": "...", "b": "..."})})

    assert response.debug["methods"]["intent"] == "structured"
    assert stub.requests[0]["logprobs"] is True          # and it did ask for logprobs first

Scenarios:

    logprobs          a provider that returns a real next-token distribution
    structured        a provider that answers in JSON with its own probabilities
    reject_logprobs   a provider that answers 400 to the logprob fields (OpenAI reasoning models,
                      Gemini's OpenAI-compatibility layer)
    no_alternatives   a provider that reports the sampled token and nothing else
    reasoning         a chat-surface two-step sequence: analysis text, then the answer

Knobs: ``surface`` (``chat_completions``, ``responses`` or ``both``, default ``both``, mirroring the
OpenAI SDK client), ``winner`` (index into the label alphabet of the option the stub prefers) and
``alternatives`` (labels reported alongside the answer; 1 means "no distribution").

Self-test with ``python offline_stub.py --check``; probe a real provider with
``python offline_stub.py --live --model <id>``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

SCENARIOS = ("logprobs", "structured", "reject_logprobs", "no_alternatives", "reasoning")
SURFACES = ("chat_completions", "responses", "both")

LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # jevper's single-letter label alphabet
ANSWER_LOGPROB = -0.1  # the sampled label; alternatives trail it, so the softmax is decisive
ANALYSIS = "The message is about a duplicate charge, so it concerns money."


class Rejection(Exception):
    """The 400 an OpenAI-compatible provider returns when it cannot do logprobs."""

    status_code = 400  # jevper reads this to tell a capability verdict from a transient failure

    def __init__(self, message: str = "logprobs are not supported with reasoning models.") -> None:
        super().__init__(message)


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


def _usage(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_tokens": 42,
        "completion_tokens": 7,
        "completion_tokens_details": {"reasoning_tokens": 0},
        "input_tokens": 42,
        "output_tokens": 7,
        "output_tokens_details": {"reasoning_tokens": 0},
    }


def _chat_body(
    kwargs: dict[str, Any], *, text: str | None = None, body: dict[str, Any] | None = None,
    entries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if body is not None:
        content = json.dumps(body)
    elif text is not None:
        content = text
    elif entries:
        content = entries[0]["token"]
    else:
        content = ""  # nothing reported: the readout fails on the missing logprobs, as it should
    choice: dict[str, Any] = {
        "index": 0,
        "message": {"role": "assistant", "content": content},
        "finish_reason": "stop",
    }
    if entries is not None:
        # An empty list is a provider that ignores the logprob fields: `logprobs: null`, as sent back
        # by compatibility layers that drop the field.
        choice["logprobs"] = {"content": entries} if entries else None
    return {"id": "stub-chat", "model": kwargs.get("model"), "choices": [choice], "usage": _usage(kwargs)}


def _responses_body(
    kwargs: dict[str, Any], *, text: str | None = None, body: dict[str, Any] | None = None,
    entries: list[dict[str, Any]] | None = None,
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
    return {
        "id": "stub-response",
        "model": kwargs.get("model"),
        "output": [
            {"id": "msg_stub", "type": "message", "role": "assistant", "status": "completed",
             "content": [part]}
        ],
        "output_text": content,
        "usage": _usage(kwargs),
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
    ) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")
        if surface not in SURFACES:
            raise ValueError(f"surface must be one of {SURFACES}, got {surface!r}")
        self.scenario = scenario
        self.surface = surface
        self.winner = winner
        self.alternatives = alternatives
        self.requests: list[dict[str, Any]] = []
        self.chat = _Namespace(completions=_Endpoint(self, "chat_completions"))
        if surface in ("responses", "both"):
            self.responses = _Endpoint(self, "responses")

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
        self.requests.append(kwargs)
        body = _chat_body if surface == "chat_completions" else _responses_body

        if self.scenario == "reject_logprobs" and _wants_logprobs(kwargs):
            raise Rejection()

        schema = _requested_schema(kwargs)
        if self.scenario == "reasoning" and schema is None and not _wants_logprobs(kwargs):
            return body(kwargs, text=ANALYSIS)  # the analysis pass: plain text, no schema, no logprobs

        if _wants_logprobs(kwargs):
            if self.scenario == "no_alternatives":
                return body(kwargs, entries=self._entries(kwargs, alternatives=1))
            if self.scenario in ("logprobs", "reasoning"):
                return body(kwargs, entries=self._entries(kwargs))
            return body(kwargs, entries=[])  # this provider has no logprobs to give back

        answer = _schema_answer(schema, self.winner) if schema else None
        if answer is not None:
            return body(kwargs, body=answer)
        return body(kwargs, text="A")  # a request jevper does not read an answer out of


# -- self-test ---------------------------------------------------------------------------------------


def _run_checks() -> int:
    from jevper import (
        Choice,
        JevperError,
        LabelReadoutError,
        ReasoningConfig,
        SystemOneClient,
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

    # A provider that rejects the logprob fields: auto falls back, and remembers.
    stub = StubClient(scenario="reject_logprobs")
    client = SystemOneClient(stub, model="stub-model")
    response = client.system_one(state=state, questions={"intent": question()})
    check("reject_logprobs: auto falls back to structured",
          response.debug["methods"]["intent"] == "structured", str(response.debug["methods"]))
    check("reject_logprobs: it asked for logprobs first", stub.requests[0].get("top_logprobs") is not None,
          str(stub.requests[0]))
    check("reject_logprobs: the fallback is recorded", bool(response.debug["retry_reasons"]))
    check("reject_logprobs: the provider saw two requests, one of them rejected",
          len(stub.requests) == 2 and len(response.debug["llm_attempts"]) == 2,
          f"requests={len(stub.requests)} attempts={len(response.debug['llm_attempts'])}")
    check("reject_logprobs: a rejected call is not counted as a call",
          response.usage.n_calls == 1, str(response.usage.n_calls))
    before = len(stub.requests)
    client.system_one(state=state, questions={"intent": question()})
    check("reject_logprobs: the verdict is remembered per model and surface",
          len(stub.requests) == before + 1 and stub.requests[-1].get("logprobs") is None,
          str(stub.requests[-1]))

    # A provider that reports the sampled token and nothing else: no distribution to read.
    stub = StubClient(scenario="no_alternatives")
    response = SystemOneClient(stub, model="stub-model").system_one(state=state, questions={"intent": question()})
    check("no_alternatives: auto falls back to structured",
          response.debug["methods"]["intent"] == "structured", str(response.debug["methods"]))
    check("no_alternatives: two provider calls, both answered",
          len(stub.requests) == 2 and response.usage.n_calls == 2,
          f"requests={len(stub.requests)} n_calls={response.usage.n_calls}")

    # Pinning logprobs against that provider is the trap the docs warn about: it raises, and no
    # corrective retry is spent on it.
    stub = StubClient(scenario="structured")
    try:
        SystemOneClient(stub, model="stub-model", method="logprobs").system_one(
            state=state, questions={"intent": question()}
        )
    except LabelReadoutError as exc:
        check("pinned logprobs without provider support raises LabelReadoutError", True)
        check("pinned logprobs is not corrective-retried", len(stub.requests) == 1, str(len(stub.requests)))
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

    print()
    print(f"{'all checks passed' if not failures else f'{failures} check(s) failed'}")
    return 1 if failures else 0


# -- live probe --------------------------------------------------------------------------------------


def _run_live(model: str, api: str) -> int:
    try:
        from openai import OpenAI
    except ImportError:
        print("the live probe needs the openai package: pip install openai")
        return 2
    if not os.environ.get("OPENAI_API_KEY"):
        print("the live probe needs OPENAI_API_KEY (set OPENAI_BASE_URL for a self-hosted server)")
        return 2

    from jevper import Choice, JevperError, SystemOneClient

    question = Choice(
        instructions="Pick the intent of the message.",
        criteria={
            "billing": "money, invoices, refunds, charges",
            "technical": "errors, crashes, login or performance problems",
            "sales": "pricing, plans, purchasing, upgrades",
        },
    )
    client = SystemOneClient(OpenAI(), model=model, api=api)
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
        "retry_reasons": response.debug["retry_reasons"],
    }, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="run the offline self-test")
    parser.add_argument("--live", action="store_true", help="ask a real provider which method it resolves to")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", ""), help="model id for --live")
    parser.add_argument("--api", default="auto", choices=("auto", "chat_completions", "responses"))
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
