---
name: jevper
description: Writes and debugs Python code that calls jevper — the Jev (System One) interface that turns a state plus Noul/Choice/Score questions into typed answers with probabilities and confidence, over any OpenAI-compatible model. Use whenever the user mentions jevper, the Jev or System One API, TypeSafe-style classification, or wants an LLM to classify, label, triage or rate text with confidence scores — including choosing between the logprobs, grammar, structured and discrete methods, making it work on reasoning models, Gemini's OpenAI-compatibility endpoint or Claude (which reject logprobs), pointing an anthropic client at a local server with api="messages" or a thinking budget, fixing LabelReadoutError, MalformedAnswerError or ProviderError, cutting cost with prompt caching and prompt_cache_key, wiring in llama.cpp/vLLM/Ollama/SGLang, adding few-shot examples or reasoning, and testing an integration without spending provider tokens.
---

# jevper

`state` in, typed `questions` out. One call sends your text plus `Noul` (yes/no), `Choice` (one of N
options) or `Score` (ordered level) questions and returns one answer per question, each carrying
probabilities and a `confidence`. The client is duck-typed — `OpenAI()`, `AsyncOpenAI()`,
`Anthropic()`, or anything exposing `chat.completions.create` / `responses.create` / `messages.create` — so
hosted models and self-hosted llama.cpp/vLLM/Ollama/SGLang servers work the same way, and the state is
rendered last so a rubric's prompts share a cacheable prefix. `openai` and `anthropic` are not runtime
dependencies; `pydantic>=2.7` is. Python 3.10+.

Written against jevper 0.5.3; if you are on a newer release, check its `docs/` — the library is the
authority.

## Quick start

```python
from openai import OpenAI
from jevper import Choice, SystemOneClient

client = SystemOneClient(OpenAI(), model="gpt-4o")   # method defaults to "auto" — leave it unset

response = client.system_one(
    state="I was charged twice for the same subscription this month.",
    questions={
        "intent": Choice(
            instructions="Pick the intent of the message.",
            criteria={
                "billing": "money, invoices, refunds, charges",
                "technical": "errors, crashes, login or performance problems",
                "sales": "pricing, plans, purchasing, upgrades",
            },
        )
    },
)

answer = response.answers["intent"]
answer.choice         # "billing" — argmax, ties broken by criteria order
answer.probabilities  # {"billing": 0.88, "technical": 0.08, "sales": 0.03} — criteria order
answer.confidence     # 0.83
```

Each question is its own provider call, and questions run concurrently (`max_concurrency`, default 8):
put the questions you need into one `system_one` call instead of looping. `state` may be a string, a chat
message list, `{"messages": [...]}`, or any JSON value (rendered as pretty-printed JSON in one user turn).

## Questions

| Question | Criteria | Answer fields |
| --- | --- | --- |
| `Noul(instructions=..., criteria={"true": ..., "false": ...})` | optional | `noul` (probability of `true`) — no `confidence` |
| `Choice(instructions=..., criteria={"key": "what belongs in it"})` | 2–255 keys | `choice`, `probabilities`, `confidence` |
| `Score(instructions=..., criteria=["level 0", "level 1", ...])` | 2–10 levels | `score`, `legend`, `probabilities`, `confidence` |

- Criteria descriptions are prompts, not labels: write what belongs in each option. They are rendered
  verbatim, and the model picks between them.
- Unknown fields are rejected (`extra="forbid"`); raw mappings (`{"type": "choice", "criteria": {...}}`)
  are parsed and validated exactly like the classes.
- `Score.score` is the probability-weighted level index `Σ i·pᵢ` over zero-based levels; `legend` maps
  level index to your description.
- `confidence` for `choice` is `(max(p) − 1/n) / (1 − 1/n)` — the peak rescaled from uniform (0) to
  certainty (1). For `score` it is `max(0, 1 − MAD/MAD_uniform)`. `noul` has none, and `discrete` (one-hot)
  always yields `1.0`.
- Read answers as `response.answers["id"]`, or the filtered views `response.choices` / `.nouls` /
  `.scores`. `response.model_dump_json()` emits the Jev wire shape (`type`/`choice`/`probabilities`/…).

## Methods: leave `method` unset

`auto` (the default) answers with `logprobs` where the provider returns them and with `structured` where it
does not, so the same code works against a logprob-capable server, a reasoning model, and a provider that
never implemented logprobs.

| Method | Asks for | Distribution |
| --- | --- | --- |
| `logprobs` | one label plus the logprobs of the alternatives | the model's real next-token distribution |
| `grammar` | the same, constrained by a GBNF grammar | same, post-mask |
| `structured` | JSON with a probability per option | the model's stated numbers, normalized |
| `discrete` | one option as JSON | one-hot over the choice |

What `auto` does, so you can rely on it: the verdict is made per `(model, surface)` by observation and
remembered for the life of the client; a `Choice` with more than 26 options goes straight to JSON. Three
things count as "this provider cannot do logprobs" — a 4xx that names the logprob fields, an answer with no
logprobs at all, or an answer token with no alternatives. A rejection that complains only about a *value*
(a server whose `top_logprobs` cap is lower than 20) still falls back for that question but is not
remembered, and a 5xx that survives retries falls back for that question alone without moving anything.
The same distinction decides the capability ladder below: a server that refuses the *number* in a field it
knows — `budget_tokens: must be at least 1024`, `reasoning_effort must be one of low, medium, high` — keeps
its own error instead of having the field dropped, because answering with your reasoning quietly switched
off is worse than failing loudly.

Before giving up on logprobs, `auto` tries the other surface when the client exposes one and the reasoning
mode is not `native`, because a server can implement a route and still not carry logprobs through it
(OpenRouter's Responses API refuses the logprob includable, ollama's `/v1/responses` answers with an empty
list, llama.cpp's refuses the fields outright). A pinned `method="logprobs"` takes that same move — it asked
for a distribution, not for a particular surface to produce one — and keeps its method; with nowhere to
move it reports the provider's refusal rather than answering in JSON. A `grammar` request never moves: it
is a Chat Completions convention the other surface cannot carry. `auto` also reads a 404 that does not name
the model as "no such route" and re-asks on the other surface, but only where the client can speak it — a
Messages-only client keeps re-asking and keeps reporting the 404. All of these verdicts are remembered.
Cost: the first question of the first call pays with one extra provider call, and a dual-surface client
spends up to three requests on it (the surface move, then the method fallback) — requests the provider
rejected are not counted in `usage.n_calls`.

`api="messages"` (an `anthropic.Anthropic` client) is the third surface and the only one with no label
readout: no logprobs exist in that API, so `auto` answers with `structured` there without spending a call,
and a pinned `logprobs`/`grammar` raises `UnsupportedMethodError` before any request. It has no schema field
either, so the JSON Schema travels in the system prompt (set `temperature=0.0` there); `max_tokens` is
required and jevper sends `1024` — plus your thinking budget, since this API wants the budget strictly
*below* `max_tokens` — unless `extra_body={"max_tokens": n}` overrides both. Thinking is a budget here,
`ReasoningConfig(budget_tokens=n)`, which `mode="auto"` selects on its own on this surface; a server that
does not know the field at all (SGLang's) has it dropped and the call re-asked, while one that refuses the
number gets its own error back.

Know the refusals by sight, and let the probe in
[Test without spending tokens](#test-without-spending-tokens) settle a new endpoint in one call:

| Refusal | Where |
| --- | --- |
| `400 logprobs are not supported with reasoning models.` | OpenAI `o`-series and `gpt-5` family |
| `400 Unknown name "logprobs": Cannot find field.` | Gemini's OpenAI-compatibility endpoint |
| no logprob API at all | Anthropic Claude, and every server's Messages route |
| sampled token with no alternatives | OpenAI's Responses surface; OpenRouter routes by price and may drop the field |

Full matrix, surfaces and local-server details: [references/providers.md](references/providers.md).

With `auto` you do not have to know any of it; pin `method` only for a stated reason:

- `grammar` — a self-hosted Chat Completions server that accepts a `grammar` field (llama.cpp and friends).
  It is rejected on the Responses surface, and most hosted providers ignore or refuse the field.
- `discrete` — you want the decision and no probabilities; it is the cheapest and never needs logprobs.
- `structured` with `temperature=0.0` — you specifically want the model's own stated probabilities, or the
  provider has neither logprobs nor a strict-schema mode.
- `logprobs` — only when you know the provider returns alternatives, and you want the token distribution.

`logprobs` and `grammar` read a single label token, so they cap a `Choice` at 26 options and raise
`InvalidQuestionError` past that (they name the two methods that take more). `auto`, `structured` and
`discrete` handle the full 2–255 range with two-letter labels.

Check what actually happened before debugging blind:

```python
response.debug["method"]                                  # what auto resolved to, or what you pinned
response.debug["methods"]                                 # {"intent": "structured"} — auto only
response.debug["api"]                                     # "chat_completions" | "responses" | "messages"
response.debug["server_limits"]                           # fields the *answer's* surface refused, else absent
response.debug["llm_attempts"][-1]["readout"]["source"]   # which readout produced the final answer
response.usage.n_calls                                    # answered calls: +analysis passes, +corrective retries
response.usage.cached_tokens                              # what the provider read from its prompt cache
len(response.debug["llm_attempts"])                       # every provider attempt, including rejected ones
```

## Prompt caching

The prompt is assembled in cache order — system prompt, few-shot turns, question block, state turns — so
the part that changes between calls, the state, comes last and a rubric's calls share everything before it.
That is what makes a re-ask with a different state cheap, and what gets a prompt over the 1024-token
minimum a hosted API caches from. One shape is the exception: a chat-list state whose own last turn is the
assistant's, which no server reads as a question (the llama.cpp engines answer
`400 Failed to initialize samplers`), so the question goes last there and that prefix is not reusable.

Every request carries a `prompt_cache_key`: yours if you passed one (`prompt_cache_key=` on the client or
the call; a non-blank string of at most 256 characters, else `JevperError` before any request is sent), or
one derived per question from the model, the method, the example turns and the question block — the method
belongs in it because its system prompt and answer shape are part of the cached prefix, so a `logprobs`
request must not be routed into a `structured` one's bucket. The state is not part of it, so one rubric's
calls route to the same cache, and both passes of a two-step call share one key.

`usage.cached_tokens` is what the provider read from its cache, `None` when it said nothing, and `0` for a
cold or disabled one: vLLM reports it only with `--enable-prompt-tokens-details`, SGLang's Chat route only
with `--enable-cache-report`, and llama.cpp/ollama always. A server that refuses the field has it dropped
and the call re-asked, reported in `debug["server_limits"]["cache_key"]` on the surface that answered. Each
surface reports it under its own name (`prompt_tokens_details`, `input_tokens_details`, or Anthropic's
`cache_read_input_tokens`), and jevper reads all three into `usage.cached_tokens`.

Per-server reporting, the measured reuse, and isolating a cache with `extra_body={"cache_salt": ...}`:
[references/providers.md](references/providers.md).

## Failures

Three groups, and only the middle one is worth catching for control flow:

| Error | Raised when | What to do |
| --- | --- | --- |
| `InvalidQuestionError`, `UnsupportedMethodError`, `ClientCapabilityError` | before any request: bad question or example (every example is checked up front, before the first call), `grammar` on the wrong surface or `logprobs`/`grammar` on the Messages surface, client missing the attribute a surface needs, or a response with no usable first choice and no explanation | fix the code — these cost nothing and never need a retry |
| `LabelReadoutError`, `MalformedAnswerError` | the answer could not be read; retried once by default (`n_retry_malformed`) with a correction turn | usually leave it alone; `auto` turns the provider-side cases into `structured` instead |
| `ProviderError` | a provider call failed after transient retries; `.attempts` and `.status_code` hold the history, including a status carried inside a `200` body (OpenRouter), and it is also what you get when no surface the client can speak has the route — a missing route is the provider's failure, not a verdict jevper keeps | the only one worth a retry loop of your own, and the one to catch at a service boundary |

`JevperError` is the base class — catch it if you want one handler for everything, including constructor
misuse (a count option that is not an integer, a blank `model`, an unknown `method`/`api`), a bad `state`
message, and content that is not JSON-serializable or carries a non-finite number.
Transient failures (`408`, `429`, `500`, `502`, `503`, `504`, `529`, connection and timeout errors, httpx
transport errors) are retried per call with `RetryPolicy(n_retries=2, base_delay=0.5, max_delay=8.0)`, at
`min(base_delay · 3ⁿ, max_delay)`; a `Retry-After` header is not read.

MLflow traces the same story without jevper's help: `mlflow.openai.autolog()` and `mlflow.anthropic.autolog()`
patch SDK resource classes, so every call through a real OpenAI/Anthropic SDK client — rejected, retried and
fallback attempts included — is a span, and wrapping the jevper call in `@mlflow.trace` groups them under one
parent; a duck-typed client is not autologged.
[references/features.md](references/features.md#tracing-with-mlflow).

A server that refuses a field jevper added for capability does not fail the call: `response_format` (or
`text.format`) walks `json_schema` → `json_object` → nothing, then the reasoning parameters, then the
Responses `include` list, then `prompt_cache_key`, then the Messages `thinking` field. Each rung is
remembered for that surface, and the question is still answered — with a fresh retry budget, since the
attempts the old request shape spent say nothing about the new one. `debug["server_limits"]` reports the
rungs of the surface the *answer* came from, so a refusal on a surface jevper later left is narrated in
`debug["retry_reasons"]` instead — read both when a ladder step seems to be missing.
Only a complaint about the field's *existence* moves the ladder: a refusal of the value
(`budget_tokens: must be at least 1024`) travels back as the provider's own error. A capability field you
named in `extra_body` is dropped with jevper's own — the SDK merges `extra_body` last, so leaving it there
would send the refused bytes again — and with `structured_outputs=False` the request already carried a
plain `json_object`, so that rung is skipped. The ladder is finite, so a server that refuses everything
still ends in `ProviderError`.

Three traps worth knowing:

- **One logprob is not a distribution.** A provider that reports only the sampled token gives nothing to
  compare against: `auto` falls back to `structured`, a pinned `logprobs` raises `LabelReadoutError`. Do not
  "fix" that by pinning harder — raise `top_logprobs`, or accept the fallback.
- **`structured` probabilities are the model's self-report.** They are rescaled when they miss 1 by more
  than `1e-6`, and the model's original numbers are kept in `debug["original_probabilities"]` (with the
  error in `debug["probability_errors"]`). Set `temperature=0.0` there; sampling noise moves them directly.
  With `normalize_probabilities=False` they are handed back exactly as the provider sent them, a value
  above 1 included (a negative or non-finite one is still a malformed answer) — normalize them yourself
  if your code assumes a distribution.
- **An answer that never arrived says why.** A reasoning model can spend the whole output budget thinking,
  and the error text then names the stop reason (`finish_reason: 'length'` /
  `incomplete_details.reason: 'max_output_tokens'` / `stop_reason: 'max_tokens'`) and suggests
  `extra_body={"max_tokens": ...}` — nobody sends those caps, so raise it and turn thinking off too. A
  model that *refused* reads as a refusal rather than as malformed JSON: Chat Completions puts it in a
  `refusal` sibling of a null `content`, Responses in a `refusal` content part, the Messages API in
  `stop_reason: 'refusal'`, and the message carries the model's own words where the surface has them. When
  the reasoning parser swallowed the whole generation into a thinking block and returned no answer text, the
  same error says the response carried reasoning only: a server-side deployment setting, not something
  another retry fixes.

## Test without spending tokens

`scripts/offline_stub.py` is a duck-typed client that answers from canned bodies — no HTTP, no key — while
the real readout path (logprobs softmax, structured JSON, `auto`'s fallback and surface move, the same move
under a pinned `method="logprobs"`, the server-limits ladder and the refusals it does not absorb, the schema
that travels in the prompt when the request cannot carry one, the message a truncated or refused answer
carries) runs end to end. Its `surface=` knob picks which endpoints the fake client exposes:
`chat_completions`, `responses`, `messages` (the Anthropic shape) or `both`.

```python
import sys
sys.path.insert(0, "<skill dir>/scripts")            # or copy the file next to your test
from offline_stub import StubClient
from jevper import Choice, SystemOneClient

stub = StubClient(scenario="reject_logprobs")        # a provider that 400s the logprob fields
client = SystemOneClient(stub, model="stub-model")
response = client.system_one(state="...", questions={"intent": Choice(criteria={"a": "...", "b": "..."})})

assert response.debug["methods"]["intent"] == "structured"   # auto fell back
assert stub.requests[0]["logprobs"] is True                  # and it did ask for logprobs first
```

Scenarios: `logprobs`, `structured`, `reject_logprobs`, `reject_include`, `no_alternatives`,
`no_responses_route`, `no_messages_route`, `reject_schema`, `reject_format`, `reject_cache_key`,
`reject_thinking`, `reject_budget_value`, `truncated`, `refusal`, `reasoning`, `reasoning_only`. Run
`python scripts/offline_stub.py --check` from the skill directory for a self-test, and
`python scripts/offline_stub.py --live --model <id>` (add `--api messages` for an Anthropic-compatible
server, `--extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'` for a local server whose
template thinks) with real credentials to see which method that provider actually resolves to, on which
surface, before writing a line of your own. The live probe asks the server what the model advertises first —
free, outside any request quota — and a `429`, `402` or `401` is reported as the account answer it is, not as
a jevper failure.
[references/providers.md](references/providers.md#does-this-provider-do-logprobs) has the
matrix and what each quota means.

## Checklist

- [ ] Criteria written as descriptions of what belongs in each option, keys as stable identifiers.
- [ ] `method` left unset unless there is a stated reason to pin it.
- [ ] Answers read through the typed views (`response.answers[...]`, `.choices`, `.model_dump_json()`).
- [ ] `JevperError` (or `ProviderError`) caught at the boundary the caller actually cares about.
- [ ] Verified against the stub before spending provider tokens.
- [ ] After the first live call: `debug["methods"]`, `debug["server_limits"]`, `usage.n_calls` and
      `usage.cached_tokens` checked.

## More

- [references/features.md](references/features.md) — reasoning, few-shot examples, async, surfaces, knobs.
- [references/providers.md](references/providers.md) — logprob matrix, the Messages route, local servers, cache reporting.
- [references/troubleshooting.md](references/troubleshooting.md) — error triage, debug keys, symptom → fix.

Inside the jevper repo, `docs/` holds the full reference (`api.md`, `methods.md`, `reasoning.md`,
`few-shot.md`, `local-servers.md`, `internals.md`) and `tests/` drives a real `openai` client against a
stub HTTP server.
