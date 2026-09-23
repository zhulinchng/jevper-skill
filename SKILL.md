---
name: jevper
description: Writes and debugs Python code that calls jevper — the Jev (System One) interface that turns a state plus Noul/Choice/Score questions into typed answers with probabilities and confidence, over any OpenAI-compatible model. Use whenever the user mentions jevper, the Jev or System One API, TypeSafe-style classification, or wants an LLM to classify, label, triage or rate text with confidence scores — including choosing between the logprobs, grammar, structured and discrete methods, making it work on reasoning models, Gemini's OpenAI-compatibility endpoint or Claude (which reject logprobs), fixing LabelReadoutError, MalformedAnswerError or ProviderError, wiring in llama.cpp/vLLM/Ollama, adding few-shot examples or reasoning, and testing an integration without spending provider tokens.
---

# jevper

`state` in, typed `questions` out. One call sends your text plus `Noul` (yes/no), `Choice` (one of N
options) or `Score` (ordered level) questions and returns one answer per question, each carrying
probabilities and a `confidence`. The client is duck-typed — `OpenAI()`, `AsyncOpenAI()`, or anything
exposing `chat.completions.create` / `responses.create` — so hosted models and self-hosted
llama.cpp/vLLM/Ollama servers work the same way. `openai` is not a runtime dependency; `pydantic>=2.7`
is. Python 3.10+.

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
remembered for the life of the client; the first question of the first call pays for it with at most one
extra provider call (two under `reasoning`); a `Choice` with more than 26 options goes straight to JSON.
Three things count as "this provider cannot do logprobs" — a 4xx that names the logprob fields, an answer
with no logprobs at all, or an answer token with no alternatives — and a 5xx that survives retries falls
back for that question only, without being remembered.

| Provider | `logprobs` |
| --- | --- |
| OpenAI `gpt-4o`, `gpt-4.1` | yes |
| OpenAI reasoning models (`o`-series, `gpt-5` family) | no — `400 logprobs are not supported with reasoning models.` |
| OpenAI Responses surface | partial — needs `include`; some models fail outright on `top_logprobs >= 2` |
| Anthropic Claude | no logprob API at all |
| Gemini via the OpenAI-compatibility endpoint | no — `400 Unknown name "logprobs": Cannot find field.` |
| DeepSeek, Together | yes, `top_logprobs` up to 20 |
| Ollama, llama.cpp, vLLM | yes |
| anything else | unknown — reasoning models and thin compatibility layers are the ones that say no |

With `auto` you do not have to know this table; pin `method` only for a stated reason:

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
response.debug["api"]                                     # "chat_completions" | "responses"
response.debug["llm_attempts"][-1]["readout"]["source"]   # which readout produced the final answer
response.usage.n_calls                                    # answered calls: +analysis passes, +corrective retries
len(response.debug["llm_attempts"])                       # every provider attempt, including rejected ones
```

## Failures

Three groups, and only the middle one is worth catching for control flow:

| Error | Raised when | What to do |
| --- | --- | --- |
| `InvalidQuestionError`, `UnsupportedMethodError`, `ClientCapabilityError` | before any request: bad question or example, `grammar` on the wrong surface, client missing the attribute a surface needs | fix the code — these cost nothing and never need a retry |
| `LabelReadoutError`, `MalformedAnswerError` | the answer could not be read; retried once by default (`n_retry_malformed`) with a correction turn | usually leave it alone; `auto` turns the provider-side cases into `structured` instead |
| `ProviderError` | a provider call failed after transient retries; `.attempts` holds the history | the only one worth a retry loop of your own, and the one to catch at a service boundary |

`JevperError` is the base class — catch it if you want one handler for everything. Transient failures
(`429`, `500`, `502`, `503`, `504`, `529`, connection and timeout errors) are retried per call with
`RetryPolicy(n_retries=2, base_delay=0.5, max_delay=8.0)`.

Two traps worth knowing:

- **One logprob is not a distribution.** A provider that reports only the sampled token gives nothing to
  compare against: `auto` falls back to `structured`, a pinned `logprobs` raises `LabelReadoutError`. Do not
  "fix" that by pinning harder — raise `top_logprobs`, or accept the fallback.
- **`structured` probabilities are the model's self-report.** They are rescaled when they miss 1 by more
  than `1e-6`, and the model's original numbers are kept in `debug["original_probabilities"]` (with the
  error in `debug["probability_errors"]`). Set `temperature=0.0` there; sampling noise moves them directly.

## Test without spending tokens

`scripts/offline_stub.py` is a duck-typed client that answers from canned bodies — no HTTP, no key, and
the real readout path (logprobs softmax, structured JSON, `auto`'s fallback) runs end to end.

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

Scenarios: `logprobs`, `structured`, `reject_logprobs`, `no_alternatives`, `reasoning`. Run
`python scripts/offline_stub.py --check` from the skill directory for a self-test, and
`python scripts/offline_stub.py --live --model <id>` with real credentials to see which method that
provider actually resolves to before writing a line of your own.

## Checklist

- [ ] Criteria written as descriptions of what belongs in each option, keys as stable identifiers.
- [ ] `method` left unset unless there is a stated reason to pin it.
- [ ] Answers read through the typed views (`response.answers[...]`, `.choices`, `.model_dump_json()`).
- [ ] `JevperError` (or `ProviderError`) caught at the boundary the caller actually cares about.
- [ ] Verified against the stub before spending provider tokens.
- [ ] After the first live call: `debug["methods"]` and `usage.n_calls` checked.

## More

- [references/features.md](references/features.md) — reasoning, few-shot examples, async, surfaces, knobs.
- [references/troubleshooting.md](references/troubleshooting.md) — provider matrix, error triage, debug recipes.

Inside the jevper repo, `docs/` holds the full reference (`api.md`, `methods.md`, `reasoning.md`,
`few-shot.md`) and `tests/` drives a real `openai` client against a stub HTTP server.
