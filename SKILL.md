---
name: jevper
description: Writes and debugs Python code that calls jevper — the Jev (System One) interface that turns a state plus Noul/Choice/Score questions into typed answers with probabilities and confidence, over any OpenAI-compatible model. Use whenever the user mentions jevper, the Jev or System One API, TypeSafe-style classification, or wants an LLM to classify, label, triage or rate text with confidence scores — including choosing between the logprobs, grammar, structured and discrete methods, making it work on reasoning models, Gemini's OpenAI-compatibility endpoint or Claude (which reject logprobs), pointing an anthropic client at a local server with api="messages" or a thinking budget, fixing LabelReadoutError, MalformedAnswerError, IncompleteAnswerError or ProviderError, cutting cost with prompt caching and prompt_cache_key, wiring in llama.cpp/vLLM/Ollama/SGLang, adding few-shot examples or reasoning, and testing an integration without spending provider tokens.
---

# jevper

`state` in, typed `questions` out. One call sends your text plus `Noul` (yes/no), `Choice` (one of N
options) or `Score` (ordered level) questions and returns one answer per question — `Choice` and `Score`
with probabilities and a `confidence`, `Noul` with its single probability. The client is duck-typed —
`OpenAI()`, `AsyncOpenAI()`, `Anthropic()`, or anything exposing `chat.completions.create` /
`responses.create` / `messages.create` — so hosted models and self-hosted llama.cpp/vLLM/Ollama/SGLang servers
work the same way, and the state is rendered last so a rubric's prompts share a cacheable prefix. `openai`
and `anthropic` are not runtime dependencies; `pydantic>=2.7` is. Python 3.10+.

Written against jevper 0.7.4; if you are on a newer release, check its `docs/` — the library is the
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

Each question is answered by its own provider call — more where reasoning, a fallback or a retry joins in —
and questions run concurrently (`max_concurrency`,
default 8): put the questions you need into one `system_one` call instead of looping. `state` may be a
string, a chat message list, `{"messages": [...]}`, or any other JSON value (rendered as pretty-printed JSON
in one user turn; a list of dicts is read as chat turns, a list of anything else as content).

## Questions

| Question | Criteria | Answer fields |
| --- | --- | --- |
| `Noul(instructions=..., criteria={"true": ..., "false": ...})` | optional | `noul` (probability of `true`) — no `confidence` |
| `Choice(instructions=..., criteria={"key": "what belongs in it"})` | 1–255 keys | `choice`, `probabilities`, `confidence` |
| `Score(instructions=..., criteria=["level 0", "level 1", ...])` | 2–10 levels | `score`, `legend`, `probabilities`, `confidence` |

- Criteria descriptions are prompts, not labels: write what belongs in each option. They are rendered
  verbatim, and the model picks between them.
- Unknown fields are rejected (`extra="forbid"`), and a question is checked where it is built *and* again in
  `system_one`: raw mappings (`{"type": "choice", "criteria": {...}}`) are parsed with a discriminated union and
  refused the same way, with the question id in the message. Both paths raise `InvalidQuestionError`, not
  pydantic's `ValidationError`.
- `Score.score` is the probability-weighted level index `Σ i·pᵢ` over zero-based levels, read off the
  distribution rescaled to 1 — so it stays on the 0..N-1 line even with `normalize_probabilities=False`,
  where the reported probabilities are the model's own — and `legend` maps level index to your description.
- `confidence` for `choice` is `(max(p) − 1/n) / (1 − 1/n)` — the peak rescaled from uniform (0) to
  certainty (1). For `score` it is `max(0, 1 − MAD/MAD_uniform)`. `noul` has none, and `discrete` (one-hot)
  always yields `1.0`.
- Read answers as `response.answers["id"]`, or the filtered views `response.choices` / `.nouls` /
  `.scores`. `response.model_dump_json()` keeps the Jev answer field names and keys
  (`type`/`choice`/`probabilities`/…) and wraps them in jevper's own `model`, `usage`, `reasoning` and
  `debug`.

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
remembered for the life of the client; a `Choice` with more than 26 options goes straight to JSON. Four
kinds of evidence count as "this provider cannot do logprobs" — a `400`/`403`/`422` that names the logprob
fields (any other status is not capability evidence, whatever its text says), a Responses request refused
for its `include` entry without ever saying "logprob" (OpenRouter: `400 Invalid option: expected one of …`
for `path: ["include", 0]`; OpenAI: `400 Unsupported parameter: 'include' is not supported with this
model.`), an answer with no logprobs at all, or an answer token with no usable rival among the options. A
rejection is remembered at once; a response that
merely lacks logprobs falls back for that question but is written off only after a second one. A rejection
complaining only about a *value* (a server whose `top_logprobs` cap is lower than 20) is never remembered,
and a 5xx that survives retries falls back for that question alone, under `auto` — pin a label method and
the same 5xx is the provider's error, not a reason to answer in JSON. The same distinction decides the
capability ladder below: a server that refuses the *number* in a field it knows —
`budget_tokens: must be at least 1024` — keeps its own error instead of having the field dropped, because
answering with your reasoning quietly switched off is worse than failing loudly.

Before giving up on logprobs, `auto` tries the other surface when the client exposes one and the reasoning
mode is not `native`, because a server can implement a route and still not carry logprobs through it
(OpenRouter's Responses API refuses the logprob includable, ollama's `/v1/responses` answers with an empty
list, llama.cpp's refuses the fields outright). A pinned `method="logprobs"` takes that same move — it asked
for a distribution, not for a particular surface to produce one — and keeps its method; with nowhere to
move it reports the provider's refusal rather than answering in JSON. A `grammar` request never moves: it is
a Chat Completions convention no other surface can carry. `auto` also reads a 404 as "no such route" —
unless it says the model does not exist, either by quoting the model id beside `model`/`no such`/`not
exist` or by carrying a `code` like `model_not_found` (a body that fills in only the code is the one field
a provider reliably fills in) — and re-asks on a surface the client can
speak, walking all three in order and skipping the ones already known to be missing; a Messages-only client
keeps re-asking and keeps reporting the 404, with the route remembered. A 404 that arrives *inside* a `200`
body is not a route verdict at all: `ProviderError.embedded` marks a failure the provider put in the body,
and a pinned `logprobs` never moves to Messages, which has no logprobs to carry. Cost: every question
already in flight can pay the discovery, and a dual-surface client spends three requests on it (the other
surface, then the method fallback) — three is the ordinary path, not a cap, since ladder re-asks and
transient retries add more — while rejected requests are not counted in `usage.n_calls`.

`api="messages"` (an `anthropic.Anthropic` client) is the third surface and the only one with no label
readout: no logprobs exist in that API, so `auto` answers with `structured` there without spending a call,
and a pinned `logprobs`/`grammar` raises `UnsupportedMethodError` before any request. Its schema travels as
Anthropic's own `output_config.format` field *and* in the system prompt — a server can accept that field and
drop it without a word, so the prompt is the only place the shape is then stated (set `temperature=0.0`
there), and a server that refuses the field has it dropped and the call re-asked. `max_tokens` is required
and jevper sends `1024` — plus your thinking budget, which this API wants strictly *below* it — unless
`extra_body={"max_tokens": n}` overrides both. Thinking is a budget here, `ReasoningConfig(budget_tokens=n)`,
which `mode="auto"` selects on its own; a server that does not know the field (SGLang's) has it dropped and
the call re-asked, while one that refuses the number gets its own error back.

The `responses` surface carries two dialects at one path: OpenAI's Responses API and the
[OpenResponses](https://www.openresponses.org) specification. LM Studio is a listed implementer and vLLM
says its route aligns with it; llama.cpp, SGLang and ollama answer the measured shapes, and broader
dialect support is unverified here. jevper sends the portable form
— every input turn a typed `{"type": "message", ...}` item, content a plain string, which both dialects
accept — and reads either back: text parts named `text`/`input_text` as well as `output_text`, several
messages per response (`phase: "commentary"` then `final_answer`, only the last read), an own `status` per
output item, logprob tokens carried as raw `bytes`. Two of those cost a call rather than an answer: an item
still `in_progress` under a `completed` response, and an event stream answering a non-streaming request.

Know the refusals by sight, and let the probe in
[Test without spending tokens](#test-without-spending-tokens) settle a new endpoint in one `system_one` call:

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
`discrete` handle the full 1–255 range with two-letter labels, and a one-option `Choice` — a question with
no rival to read — is answered by every method, `logprobs` included, at probability 1.0.

Check what actually happened before debugging blind:

```python
response.debug["method"]                                  # the call-level default: what auto starts with, or what you pinned
response.debug["methods"]                                 # {"intent": "structured"} — what each question resolved to, auto only
response.debug["api"]                                     # "chat_completions" | "responses" | "messages"
response.debug["server_limits"]                           # six flags for the final surface, absent when it refused nothing
response.debug["apis"]                                    # {"intent": "chat_completions"} — per question, when one call used several
response.debug["server_limits_by_api"]                    # the same six flags per surface, when one call used several
response.debug["llm_attempts"][-1]["readout"]["source"]   # which readout produced the final answer
response.usage.n_calls                                    # answered calls: +analysis passes, +corrective retries
response.usage.cached_tokens                              # what the provider read from its prompt cache
len(response.debug["llm_attempts"])                       # every provider attempt, including rejected ones
```

## Prompt caching

The prompt is assembled in cache order — system prompt, few-shot turns, question block, state turns — so
the part that changes between calls, the state, comes last and a rubric's calls share everything before it.
That is what makes a re-ask with a different state cheap, and what gets a prompt over OpenAI's 1024-token
minimum cacheable prefix. One shape is the exception: a chat-list state whose own last turn is the
assistant's, which no server reads as a question, so the question goes last there and that prefix is not
reusable.

Every Chat Completions and Responses request carries a `prompt_cache_key` (the Messages API has no such
field): yours if you passed one (`prompt_cache_key=` on the client or the call; a non-blank string of at
most 256 characters, else `JevperError` before any request is sent), or one derived per question from the
model, the method, the example turns and the question block — the method belongs in it because its system
prompt and answer shape are part of the cached prefix, so a `logprobs` request must not be routed into a
`structured` one's bucket. The state is not part of it, so one rubric's calls route to the same cache, and
both passes of a two-step call share one key. It travels in the **request body**, never as an SDK keyword,
so an `openai` older than the field still carries it. jevper's local ceiling is 256 characters; the
OpenResponses schema documents a 64-character maximum, but OpenAI's own API reference states no length
limit, so a longer key is left for the provider to judge.

`usage.cached_tokens` is what the provider read from its cache, `None` when it said nothing, and `0` for a
cold or disabled one: vLLM reports it only with `--enable-prompt-tokens-details`, SGLang's Chat route only
with `--enable-cache-report`, and llama.cpp/ollama always. A server that refuses the field has it dropped
and the call re-asked, reported in `debug["server_limits"]["cache_key"]` on the surface that answered. Each
surface reports it under its own name (`prompt_tokens_details`, `input_tokens_details`, or Anthropic's
`cache_read_input_tokens`), and jevper reads all three into `usage.cached_tokens`.

Per-server reporting, the measured reuse, and isolating a cache with `extra_body={"cache_salt": ...}`:
[references/providers.md](references/providers.md).

## Failures

Four groups, and only the two middle ones are worth catching for control flow:

| Error | Raised when | What to do |
| --- | --- | --- |
| `InvalidQuestionError`, `UnsupportedMethodError`, `ClientCapabilityError` | before any request: a question or example that is invalid **wherever it is built** — `Choice(criteria={})` raises it, and so does the mapping handed to `system_one`, so one `except JevperError` covers a rubric written in Python and one loaded from data. An example's answer and its own numbers are checked against the question that carries it at construction; `examples=` passed to the client or the call have no question to check against until the call pairs them with one, so they are checked there, still before any request. Then `grammar` on the wrong surface or `logprobs`/`grammar` on the Messages surface, a client missing the attribute a surface needs, or an `extra_headers` name that is not an HTTP token or a value that is not printable ASCII (a CRLF, NUL, non-ASCII or lone surrogate is refused here, not sent) — plus, after one, a JSON body carrying no usable first choice and no explanation of why. An event *stream* is not in this row any more: it is `ProviderError` on all three surfaces | fix the code; the first three cost nothing, and the last is never retried |
| `LabelReadoutError`, `MalformedAnswerError` | the answer came back but could not be read; a model-side failure gets one corrective retry by default (`n_retry_malformed`) with a correction turn, while a provider that does not report logprobs at all is never corrected | usually leave it alone; `auto` turns the provider-side cases into `structured` instead |
| `IncompleteAnswerError`, `ModelRefusalError` | the provider stopped generating before the answer was complete (a spent output budget, or a context window too small) or reported that the model declined or was filtered — `refusal`, `content_filter` alike, since a second attempt is filtered the same way — both are `ProviderError` subclasses raised *before* any readout, because a cut-off, declined or withheld generation is not an answer to correct | fix the request, not the reader: turn thinking off, raise the cap the surface names (`max_output_tokens` on Responses, `max_tokens` on Chat and Messages) or shorten the state; a refusal needs a different request or model, and a retry is refused the same way |
| `ProviderError` | a provider call failed — a transient one after its retries are exhausted, or a non-transient one at once. `.attempts`/`.status_code` hold the history, including a status carried inside a `200` body (OpenRouter: `embedded=True`, and its answer, if any, loses to the error beside it), a Responses status that is neither `completed` nor `incomplete` or an output item still `in_progress`, and a non-streaming request answered with an event stream. In that last case the stream is read for the provider's own failure — `event: error`, OpenAI's typed `response.failed`, or a bare `{"error": …}` — and that error's status decides the retry, so a `429` inside a `200` is retried; a stream carrying no failure is a protocol mismatch whose message names the surface. It is also what you get when no surface the client can speak has the route, and that route verdict is remembered, so a client that can speak neither pays the same 404 on every call | the one to catch at a service boundary; retry it yourself only for the transient set — a missing route, a terminal generation and a stream will all fail again |

`JevperError` is the base class — catch it for every error jevper raises, including constructor
misuse (a count option that is not an integer or out of range, a blank `model`, an unknown `method`/`api`, a
header name or value the HTTP layer could not carry), a bad `state` message, content that is not
JSON-serializable or carries a non-finite number, a string that cannot be encoded as UTF-8 in the state,
the question, the model id or `extra_body`, and an `extra_body` that refers to itself — all named, before any
request, rather than failing later inside the SDK or looping over the cycle. A structure nested deeper than
this interpreter's JSON encoder can write is the same story: a named `JevperError`, not a `RecursionError`
escaping a public call, and the message says where the limit came from — it is a property of the runtime, so
the same state can encode on one Python and not another; flatten it or hand it over as text.
`InvalidQuestionError` is a `JevperError`, so one handler for the base covers a rubric built in Python and one
loaded from data; `ReasoningConfig` is the one caller-facing model that is not a question type, and pydantic's
own `ValidationError` is still what it raises. The constructor
copies `extra_body` and `extra_headers`, so editing your dict afterwards does not change what this client sends.

Transient failures are retried per call with `RetryPolicy(n_retries=2, base_delay=0.5, max_delay=8.0)`, at
`min(base_delay · 3ⁿ, max_delay)`: the statuses `408`, `409`, `429` and *any* `5xx`, plus connection and
timeout errors matched by class name. A provider's own header replaces the backoff — `Retry-After` as
delta-seconds or an HTTP date, `retry-after-ms` as milliseconds, and `x-should-retry`, where `false`
suppresses even a `503` and `true` repeats a `400`. `max_delay` does not cap a header jevper honours (a
24-hour ceiling does), because coming back sooner is another request the server will refuse;
`respect_retry_after=False` keeps the curve alone, and an unreadable value falls back to it. An official
SDK's own retry loop is disabled on the copy jevper makes, so `usage.n_retries` counts every retry.

MLflow traces the same story without jevper's help: `mlflow.openai.autolog()` and `mlflow.anthropic.autolog()`
patch SDK resource classes, so every call through a real OpenAI/Anthropic SDK client — rejected, retried and
fallback attempts included — is a span, and wrapping the jevper call in `@mlflow.trace` groups them under one
parent; a duck-typed client is not autologged.
[references/features.md](references/features.md#tracing-with-mlflow).

A server that refuses a field jevper added for capability does not fail the call. The ladder runs
`include` first (unless the refused entry is the logprob carrier, which is a readout problem, not a
reasoning one — a message naming `include[1]` drops only `reasoning.encrypted_content` and keeps the
logprob carrier, exactly as the server says), then the schema — `response_format`/`text.format` walking
`json_schema` → `json_object` → nothing, or on the Messages surface `output_config.format` dropped whole,
since the prompt is where that schema lived before the field existed — then `reasoning_effort`/`reasoning`,
then `prompt_cache_key`, then the Messages `thinking` field. Each rung is remembered for that *(model,
surface)* pair, so a field one model refused is still sent for the next model you name, and the question is
still answered — with a fresh retry budget, since the attempts the old request shape spent say nothing about
the new one. `debug["server_limits"]` is a six-field snapshot for the surface the *answer* came from, absent
when that surface refused nothing; a call that used more than one surface also carries
`debug["server_limits_by_api"]`, `debug["apis"]` and `debug["reasoning_modes"]`, and a refusal on a surface
jevper later left is in that question's `llm_attempts`, not in `retry_reasons` — read all of them when a
ladder step seems missing.
Only a complaint about a field's *existence* moves the ladder, and the schema is the exception: a
complaint about its *content* moves it too, because the prompt keeps the schema either way. A refusal of
the number in any other field (`budget_tokens: must be at least 1024`) travels back as the provider's own
error. A capability field you named in `extra_body` is dropped with jevper's own — the SDK merges
`extra_body` last, so leaving it there would send the refused bytes again — and with
`structured_outputs=False` the request already carried a plain `json_object`, so that rung is skipped. The
ladder is finite, so a server that refuses everything still ends in `ProviderError`.

Three traps worth knowing:

- **One logprob is not a distribution.** A provider that reports only the sampled token gives a question with
  more than one option nothing to compare against: `auto` falls back to `structured`, a pinned `logprobs`
  raises `LabelReadoutError` — unless the question has a single option, where the sampled token is the
  answer. Reported alternatives that are not a usable rival *among the options*, and a value no log
  probability can be (a positive one), are the same case. Do not "fix" the general case by pinning harder:
  raise `top_logprobs`, or accept the fallback.
- **`structured` probabilities are the model's self-report.** They are rescaled when they miss 1 by more
  than `1e-6`, and the model's original numbers are kept in `debug["original_probabilities"]` (with the
  error in `debug["probability_errors"]`). Set `temperature=0.0` there; sampling noise moves them directly.
  With `normalize_probabilities=False` they are handed back exactly as the provider sent them, a value
  above 1 included (a negative or non-finite one is still a malformed answer) — normalize them yourself
  if your code assumes a distribution.
- **An answer that never arrived is not a malformed one.** A generation the provider cut short — a reasoning
  model spending the whole output budget thinking — is `IncompleteAnswerError`, naming the stop reason
  (`finish_reason: 'length'` / `incomplete_details.reason: 'max_output_tokens'` /
  `stop_reason: 'max_tokens'`) and suggesting the cap *that surface* uses (`max_output_tokens` on
  Responses, `max_completion_tokens` on Chat — which is what OpenAI's route takes, with `max_tokens`
  named beside it for local servers — and `max_tokens` on Messages); nobody sends those caps, so raise it
  and turn thinking off.
  A context window too small is terminal the same way but wants the opposite remedy, so that message says
  to shorten the state or the examples. A refusal is `ModelRefusalError` — a `refusal` beside a null
  `content`, a `refusal` content part, `stop_reason: 'refusal'`, or a safety `content_filter` — with the
  model's own words where the surface has them. None of the three spends a corrective retry, and neither
  does a shape the server misreports: a reasoning parser that swallowed the whole generation into a
  thinking block, or a `completed` response whose answer plainly stops mid-object, is a
  `MalformedAnswerError` — jevper believes the status it was given rather than inventing a budget reason.
  Where two views of one generation disagree it refuses rather than picks: a sampled label contradicting
  the answer text beside it is a `LabelReadoutError`.
- **An answer must carry exactly what was asked for.** A `structured` or `discrete` body with an extra root
  key beside the field it was asked for is a `MalformedAnswerError` — the message names the keys it saw and
  not their values, so a model that pads its JSON fails instead of passing. A `discrete` level may be
  written `"2"`, `"2.0"` or `"2.000"`, but a decimal fraction like `"2.0000000000000000000001"` is malformed
  rather than rounded, and a number no float could hold cannot become a level index. Option keys are
  matched exactly first and case-folded ASCII-only second: `a` is label `A`, while `ı` is not `I` and a
  non-ASCII key is matched as itself.

## Test without spending tokens

`scripts/offline_stub.py` is a duck-typed client that answers from canned bodies — no HTTP, no key — while
the real readout path runs end to end: the logprobs softmax, structured JSON, `auto`'s fallback and surface
move (also under a pinned `method="logprobs"`), the server-limits ladder and the refusals it does not
absorb, the schema in `output_config` and in the prompt, the quoted untrusted state, the retry rules, the
event-stream reader on all three surfaces, and the provider shapes that used to read as an answer. Its
`surface=` knob picks which endpoints the fake client exposes: `chat_completions`, `responses`, `messages`
(the Anthropic shape) or `both`. It needs jevper 0.7.4 or newer.

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
`reject_thinking`, `reject_budget_value`, `reject_output_config`, `truncated`, `truncated_context`,
`failed_response`, `embedded_error`, `content_filter`, `item_in_progress`, `refusal`, `positive_logprob`,
`no_rival_alternative`, `contradicts_text`, `two_objects`, `transient`, `reject_reasoning_include`,
`reasoning`, `reasoning_only`, `stream_error`, `stream_error_chat`, `stream_split_data`, `stream_mismatch`,
`response_failed`, `model_not_found`, `header_crlf`, `self_referential_body`, `extra_root_key`,
`deep_json`, `score_decimal`, `unicode_label`, `redacted_header`. Run `python scripts/offline_stub.py
--check` from the skill directory for a self-test, and
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

Inside the jevper repo, `docs/` holds the full reference (`index.md`, `getting-started.md`, `api.md`,
`methods.md`, `reasoning.md`, `few-shot.md`, `local-servers.md`, `architecture.md`, `internals.md`,
`troubleshooting.md`, `complete-example.md`, `glossary.md`, `mlflow.md`) and `tests/`
drives a real `openai` client against a stub HTTP server.
