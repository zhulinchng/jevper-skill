# jevper features

- [Reasoning](#reasoning) — `ReasoningConfig`, native vs two-step, reading the trace
- [Few-shot examples](#few-shot-examples) — `Example`, the three levels, calibration
- [Async](#async) — `AsyncSystemOneClient`
- [Surfaces](#surfaces) — Chat Completions vs Responses
- [Client knobs](#client-knobs) — the options worth changing

## Reasoning

```python
from jevper import ReasoningConfig, reasoning_text

client = SystemOneClient(OpenAI(), model="gpt-4o", reasoning=ReasoningConfig(effort="medium"))
response = client.system_one(state=..., questions=...)
reasoning_text(response.reasoning)      # the trace, joined; "" when there is none
```

`ReasoningConfig(effort=None, summary=None, context=None, mode="auto")` — `effort` is one of `none`,
`minimal`, `low`, `medium`, `high`, `xhigh`, `max`; `summary` is `auto`/`concise`/`detailed`; `context` is
`auto`/`current_turn`/`all_turns`. `reasoning=None` (the default) disables reasoning entirely.

| `mode` | Responses surface | Chat Completions surface |
| --- | --- | --- |
| `auto` (default) | native provider reasoning | two-step |
| `native` | provider reasoning on the answer call | `reasoning_effort` on the answer call (only when `effort` is set) |
| `two_step` | analysis call, then answer call | analysis call, then answer call |

- **Two-step is two provider calls per question**, so roughly twice the cost: an analysis pass (plain text,
  no schema, no logprobs, few-shot turns included) whose output is replayed as an assistant turn before the
  answer pass. `usage.n_calls` counts both.
- On the chat surface a two-step analysis call sends no reasoning parameters at all — the analysis prompt
  *is* the reasoning step. Use `mode="native"` if you want the provider's own reasoning there.
- `summary` and `context` have no chat equivalent; they only reach the Responses surface.
- `response.debug["reasoning_mode"]` reports `off`, `native` or `two_step`. `reasoning_text()` prefers
  provider summaries over content texts.
- The Responses surface sends `store=false`, so the provider keeps no state: `encrypted_content` on the
  reasoning items is what makes a trace portable into a later request.

## Few-shot examples

```python
Example(state="Charged twice for one order", answer="billing")
Example(state="Login fails after reset", answer="technical",
        probabilities={"billing": 0.05, "technical": 0.9, "sales": 0.05})
```

`answer` may be a label (`"B"`, two letters past 26 options), a `Choice` criteria key, a `Score` level
index, or a bool for `Noul`. Examples are rendered as chat turn pairs — the example state plus the same
question block, then the expected answer **in the format the active method expects** (a label for
`logprobs`/`grammar`/`discrete`, a JSON object for `structured`) — so switching methods needs no change to
your examples.

Three levels, first non-empty wins, no merging:

1. `question.examples` — `Choice(criteria=..., examples=[...])`
2. `system_one(..., examples=[...])` or `examples={"intent": [...]}` for one question id
3. `SystemOneClient(..., examples=[...])` — the house default for every call

A bare sequence applies to every question in the call; a mapping is keyed by question id. An answer that
matches no option raises `InvalidQuestionError` before any request. `probabilities` is only read by
`structured` and defaults to one-hot over `answer` — a one-hot example teaches the model that answers are
certain, so pass explicit numbers when the demonstration should teach calibration. Wrong key sets or
negative values are refused (`InvalidQuestionError` naming the example index).

## Async

`AsyncSystemOneClient` has the same constructor and `system_one` signature (`async def`), and uses an
`asyncio.Semaphore(max_concurrency)` instead of a thread pool:

```python
from openai import AsyncOpenAI

async with AsyncSystemOneClient(AsyncOpenAI(), model="gpt-4o") as client:
    response = await client.system_one(state=..., questions=...)
```

`aclose()` is a no-op — the async client holds no resources. Neither client ever closes the client you
passed in; `SystemOneClient.close()` only shuts down its own thread pool.

## Surfaces

`api="auto"` (the default) prefers the Responses surface, which carries native reasoning and encrypted
content — except for `grammar`, which only Chat Completions can carry. A client missing the attribute the
chosen surface needs raises `ClientCapabilityError` naming the surface to pass explicitly.

| | Chat Completions | Responses |
| --- | --- | --- |
| input | `messages=[...]` | `input=[...]`, `store=false` |
| logprobs | `logprobs=true`, `top_logprobs=N` | `top_logprobs=N`, `include=["message.output_text.logprobs"]` |
| JSON schema | `response_format={"type": "json_schema", ...}` | `text={"format": {"type": "json_schema", ...}}` |
| grammar | `extra_body={"grammar": "..."}` | not available |

Neither builder sends `max_tokens`/`max_output_tokens`: reasoning tokens count against those caps, and a
small cap silently truncates a reasoning model. Any other provider field goes through `extra_body`.

## Client knobs

| Option | Default | Change it when |
| --- | --- | --- |
| `method` | `"auto"` | you have a reason (see SKILL.md) |
| `api` | `"auto"` | a client exposes both surfaces and you want a specific one |
| `temperature` | `None` (not sent) | `0.0` for `structured`/`discrete`; leave unset for `logprobs` |
| `top_logprobs` | `20` | lower it only if the provider rejects the field; `0` leaves no distribution to read |
| `structured_outputs` | `True` | `False` sends `{"type": "json_object"}` with the schema in the prompt, for providers that reject strict schemas |
| `normalize_probabilities` | `True` | `False` returns the model's `structured` numbers verbatim (the error is still recorded in `debug`) |
| `max_concurrency` | `8` | your provider rate-limits per key |
| `n_retry_malformed` | `1` | a model that keeps answering in prose |
| `retry` | `RetryPolicy()` | transient-failure retries: `n_retries=2`, `base_delay=0.5`, `max_delay=8.0` |
| `extra_body`, `extra_headers` | `None` | provider-specific fields |

Constructor misuse (unknown `method`/`api`, `top_logprobs` outside `[0, 20]`, `max_concurrency < 1`, a
negative retry field) raises `JevperError` immediately, so a typo never reaches a provider.
