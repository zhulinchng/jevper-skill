# jevper features

- [Reasoning](#reasoning) — `ReasoningConfig`, native vs two-step, reading the trace
- [Few-shot examples](#few-shot-examples) — `Example`, the three levels, calibration
- [Async](#async) — `AsyncSystemOneClient`
- [Surfaces](#surfaces) — Chat Completions, Responses, Messages and the Jev wire, and the fallbacks between them
- [Decisions API](#decisions-api) — OpenAI's schema (`client.decisions.create`), limits, answer shapes, usage
- [Prompt caching](#prompt-caching) — message order, `prompt_cache_key`, `usage.cached_tokens`
- [Tracing with MLflow](#tracing-with-mlflow) — autolog spans for calls through real SDK clients, including fallbacks, and MLflow model hosting
- [Client knobs](#client-knobs) — the options worth changing

## Reasoning

```python
from jevper import ReasoningConfig, reasoning_text

client = SystemOneClient(OpenAI(), model="gpt-4o", reasoning=ReasoningConfig(effort="medium"))
response = client.system_one(state=..., questions=...)
reasoning_text(response.reasoning)      # the trace, joined; "" when there is none
```

`ReasoningConfig(effort=None, summary=None, context=None, mode="auto", budget_tokens=None)` — `effort` is one
of `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`; `summary` is `auto`/`concise`/`detailed`;
`context` is `auto`/`current_turn`/`all_turns`. `reasoning=None` (the default) disables reasoning entirely.

| `mode` | Responses surface | Chat Completions surface | Messages surface |
| --- | --- | --- | --- |
| `auto` (default) | native provider reasoning | two-step | two-step, or `native` when `budget_tokens` is set |
| `native` | provider reasoning on the answer call | `reasoning_effort` on the answer call (only when `effort` is set) | `thinking` on the answer call (only when `budget_tokens` is set) |
| `two_step` | analysis call, then answer call | analysis call, then answer call | analysis call, then answer call |

- **Two-step is two provider calls per question**, so roughly twice the cost: an analysis pass (plain text,
  no schema, no logprobs, few-shot turns included) whose output is replayed as an assistant turn before the
  answer pass. `usage.n_calls` counts both.
- On the chat surface a two-step analysis call sends no `reasoning_effort` of its own — the analysis prompt
  *is* the reasoning step — though a `reasoning_effort` you put in `extra_body` still reaches the wire. Use
  `mode="native"` if you want the provider's own reasoning there.
- `budget_tokens` is the Messages surface's thinking budget and is ignored by the other two, which carry
  `reasoning_effort` and `reasoning` instead. `effort` is never translated into a budget — that mapping is
  yours — so a Messages request without `budget_tokens` sends no `thinking` field at all and the model's own
  default applies. A budget is the only reason to ask for this surface's own thinking, so `mode="auto"` with
  a budget resolves to `native` there rather than paying for a two-step pass. Anthropic requires at least
  `1024` and strictly less than `max_tokens`, so jevper's own `max_tokens` default grows by the budget
  (`1024` + n) — the answer keeps the whole default and the thinking is paid for out of the extra. jevper
  only refuses a value below `1` and lets the provider answer for itself: a server that refuses the *number*
  (`budget_tokens: must be at least 1024` on SGLang) gets its own error back, not a silent re-ask without
  thinking.
- `summary` and `context` have no chat or Messages equivalent; they only reach the Responses surface.
- On the Messages surface, Claude 4.6 deprecated the manual `thinking` budget and 4.7 and later reject it
  with a `400`; from 4.6 on, Anthropic's supported path is the adaptive form —
  `extra_body={"thinking": {"type": "adaptive"}}` (optionally `display: "summarized"`) plus
  `output_config={"effort": …}` — which jevper forwards verbatim (a caller-supplied `thinking` is
  authoritative), while 4.5 and earlier keep the manual `budget_tokens`. On Fable 5.1, Mythos 5.1, Fable 5,
  Mythos 5, Mythos Preview, Opus 5.5/5/4.8/4.7 and Sonnet 5, Anthropic rejects a non-default `temperature`,
  `top_p` or `top_k` on every request, thinking or not; older models restrict those only while thinking is
  on. The `anthropic` 1.8 SDK removed all three from `messages.create()`, so jevper sends its own
  `temperature` through `extra_body` when it is configured and no thinking budget is sent.
- `response.debug["reasoning_mode"]` reports `off`, `native` or `two_step`, and a call that used more than
  one surface adds `debug["reasoning_modes"]`, one entry per question. `reasoning_text()` prefers
  provider summaries over content texts.
- The Responses surface sends `store=false`, so the provider keeps no state: `encrypted_content` on the
  reasoning items is what makes a trace portable into a later request. On the Messages surface a thinking
  block's `signature` is kept on the reasoning part for the same reason.

## Few-shot examples

```python
Example(state="Charged twice for one order", answer="billing")
Example(state="Login fails after reset", answer="technical",
        probabilities={"billing": 0.05, "technical": 0.9, "sales": 0.05})
```

`answer` may be a label (`"B"`, two letters past 26 options), a `Choice` criteria key, a `Score` level
index, or a bool for `Noul`. A criteria key is matched exactly first — the raw string, then the stripped one
— and the label second, with ASCII-only case folding: `a` is label `A`, while `ı` is not `I` and a
non-ASCII key is only ever matched as itself. A `discrete` level index is read as a decimal, so `"2"`,
`"2.0"` and `"2.000"` are all level 2 while a fraction like `"2.0000000000000000000001"` is malformed rather
than rounded. An answer body must carry *exactly* the field it was asked for — an extra root key beside
`choice`/`noul`/`score`/`probabilities` is a `MalformedAnswerError` naming the keys it saw, not their
values. With criteria `{"b", "a"}` the answer `"a"` names the option keyed `"a"` rather than the first label.
Examples are rendered as chat turn pairs — the example state plus the same question block, then the expected
answer **in the format the active method expects** (a label for `logprobs`/`grammar`/`discrete`, a JSON object
for `structured`) — so switching methods needs no change to your examples. An example answer is resolved the
same way, ASCII folding included, so a few-shot answer written `a` still matches option `A`.

Three levels, first non-empty wins, no merging:

1. `question.examples` — `Choice(criteria=..., examples=[...])`
2. `system_one(..., examples=[...])` or `examples={"intent": [...]}` for one question id
3. `SystemOneClient(..., examples=[...])` — the house default for every call

A bare sequence applies to every question in the call; a mapping is keyed by question id. Every question's
examples are resolved and validated before the first request — a call that is locally invalid spends
nothing, rather than failing inside one worker after the questions ahead of it have already paid. An
example carried *by its own question* (`Choice(examples=[...])`) is checked there instead, at construction,
because the question that gives the answer its meaning is right there; the message is prefixed
`choice question:` rather than a question id. The container must be a sequence of `Example` objects, and
anything else is `InvalidQuestionError` naming the index rather than an `AttributeError` later. An answer
that matches no option raises `InvalidQuestionError`,
and a `Noul` example's `probabilities` must key each answer one way only (`True`/`False`, not
`{True: 0.2, "true": 0.8}`, which is two spellings of one answer after JSON round-trips). `probabilities` is
only read by `structured` and defaults to one-hot over `answer` — a one-hot example teaches the model that
answers are certain, so pass explicit numbers when the demonstration should teach calibration. Wrong key
sets or negative values are refused (`InvalidQuestionError` naming the example index). A bare `Example` has
no question to be checked against, so it is checked only for what pydantic can see on its own — a
non-finite probability, an extra field — and the rest when it is handed to a question or a call.

## Async

`AsyncSystemOneClient` has the same constructor and `system_one` signature (`async def`), and uses an
`asyncio.Semaphore(max_concurrency)` instead of a thread pool:

```python
from openai import AsyncOpenAI

async with AsyncSystemOneClient(AsyncOpenAI(), model="gpt-4o") as client:
    response = await client.system_one(state=..., questions=...)
```

`aclose()` is a no-op — the async client holds no resources. Neither client ever closes the client you
passed in; `SystemOneClient.close()` only shuts down its own thread pool, joining the workers still running.
The facades check each other before any request: a blocking `SystemOneClient` handed an async SDK client (or
the reverse) raises `ClientCapabilityError` naming the facade to use, so an `AsyncOpenAI` never sits
unawaited and a thread pool never blocks on a coroutine.

An interrupt is not a failed question. A `KeyboardInterrupt` or `SystemExit` reaching a batch cancels the
questions still queued and is re-raised at once, and in the async client any `BaseException` out of the batch
takes the same path — a worker thread already running cannot be cancelled and may still finish, which is
what `close()` joins. An ordinary per-question failure is different: every question still runs and the first
failure in question order is raised.

## Surfaces

`api="auto"` (the default) prefers the Responses surface, which carries native reasoning and encrypted
content — except for `grammar`, which only Chat Completions can carry, and except for a client whose only
surface is `messages`. A client missing the attribute the chosen surface needs raises
`ClientCapabilityError` naming the surface to pass explicitly. The fourth value, `api="systemone"`, is never
chosen by `auto` — it is the Jev wire itself, not a fallback — so it is only ever pinned (see below).

| | Chat Completions | Responses | Messages |
| --- | --- | --- | --- |
| input | `messages=[...]` | `input=[{"type": "message", ...}]`, `store=false` | `messages=[...]` + top-level `system` |
| logprobs | `logprobs=true`, `top_logprobs=N` | `top_logprobs=N`, `include=["message.output_text.logprobs"]` | none — the API has no such field |
| JSON schema | `response_format={"type": "json_schema", ...}` | `text={"format": {"type": "json_schema", ...}}` | `output_config={"format": {"type": "json_schema", ...}}`, sent in the body; the schema also stays in the system prompt |
| grammar | `extra_body={"grammar": "..."}` | not available | not available |
| output cap | server default | server default | `max_tokens` required: jevper sends `1024`, or `1024` + the thinking budget |

Neither OpenAI builder sends `max_tokens`/`max_output_tokens`: reasoning tokens count against those caps, and
a small cap silently truncates a reasoning model. The Messages protocol has no server-side default, so jevper
always sends `max_tokens` there (`1024`, or `1024` plus `ReasoningConfig(budget_tokens=n)` because Anthropic
requires the budget to be strictly below it; `extra_body={"max_tokens": n}` overrides both). Any other
provider field goes through `extra_body` — a key you name there is what reaches the wire, because the SDK
merges `extra_body` after the typed parameters, and a capability field among them is dropped along with
jevper's own when a server has refused it. That is also how `output_config` is sent: it travels in the body
rather than as a typed keyword, so the oldest `anthropic` SDK jevper supports (`>=0.49`, which has no such
parameter and would raise `TypeError` before sending anything) can carry it.

The Responses request is the portable form of two specifications at one path — OpenAI's Responses API and
[OpenResponses](https://www.openresponses.org) — so each input turn is a typed item and its content a plain
string, which both accept (see [providers.md](providers.md#surfaces) for what each server does with that).
The response is read as either dialect: text parts named `output_text`/`text`/`input_text`, reasoning under
`summary_text`/`reasoning_text`/`text`, an item `status` of its own, `phase`-labelled messages where only
`final_answer` is read, and logprob tokens whose `bytes` carry the exact text for a byte-level tokenizer.
Two shapes are failures rather than answers: an output item still `in_progress` under a `completed` response,
and an event stream sent in answer to a non-streaming request — which is now read for the provider's own
failure (`event: error`, OpenAI's typed `response.failed`, or a bare `{"error": …}`) and raised as a
`ProviderError` carrying that failure's status, so a `429` inside a `200` is retried; a stream with no
failure in it is a protocol mismatch, not a malformed answer.

The schema travels in the prompt wherever the request cannot be relied on to state it: always on the Messages
surface, and on the OpenAI surfaces when `structured_outputs=False` or the server refused the strict schema.
On the Messages surface the prompt keeps it even when the field was sent, because a server can accept
`output_config` and drop it without a word — the same silence as a server that never read it. It joins the
leading system message, so the question block keeps its place — and where the request no longer states the
shape, the answer's shape is only as good as the model's instruction-following.

Anthropic's structured outputs implement a documented subset of JSON Schema, and an unsupported keyword is a
`400` rather than a warning. The OpenAI surfaces send the bounds they *can* express — every probability at
`minimum: 0`, a `Noul` at `maximum: 1` — and Anthropic's gets each such keyword moved into the description of
the field it bounded (`Must be at least 0.`) while the prompt keeps the full schema, where text can say what a
constraint says. Sum-to-one stays client-side: JSON Schema cannot express it.

### The Jev wire (`api="systemone"`)

This surface posts the Jev wire itself and is the only one that answers every question from **one request**:
`client.post(path="/systemone", body={state, model, questions}, cast_to=dict)`, which an `OpenAI` object
exposes. The answer is the service's own — its `score`, `confidence`, `choice` and `legend` are read as they
arrived rather than recomputed from a distribution (jevper's formulas agree to within 0.015, and the
service's number is the one the caller gets) — and `usage` counts that one call. Two checks the type system
cannot make still run: the answer's `type` must match the question's, and its distribution must carry
*exactly* the question's own options or levels, both directions, with score levels re-keyed from JSON's text
keys to their integer indices. `client.list_models()` reads `GET /v1/models` in the service's shape —
`{"models": [{"name": …, "description": …, "release_date": …}]}`; an OpenAI-style `{"data": …}` answer is a
`MalformedAnswerError`, because that list is the gateway's, not the service's. The client needs `post` (and
`get` for the model list) — an `OpenAI` object has both, and a `ClientCapabilityError` names a missing one.

The base URL is the **host**: jevper appends `/systemone`, so passing the endpoint itself doubles the path
into a 404 from whatever serves that name. Refused by name before any request: the options the wire has no
field for — `method` other than `auto`, `reasoning`, `examples`, `temperature`, `prompt_cache_key`, in one
`ClientCapabilityError` naming every one you set — a noul carrying neither instructions nor criteria (the
hosted Jev answers `400`; `noul_requires_question=False` sends one anyway for servers that read the question
id instead — Ollaya, CLM and kev), a number or boolean where the wire takes a string, object or array (a
state, an instruction, a noul criterion, an option description, a score level — an `InvalidQuestionError`
naming every offender), and an empty question id. `extras` and `keep_alive` are refused on this route in the
same words unless the client is built with `native=True`, which requires `api="systemone"` itself (refused
with `auto`) — they belong to Ollaya's own `POST /api/decide`, whose answer is a `NativeSystemOneResponse`:
the `routing` report (`router`, `model` — the checkpoint the router chose, while the response's own `model`
stays the alias you asked for —, `route`, and a `reason` that is prose, never parsed), `state_truncated`,
`done_reason`, `created_at`, the service's own nanosecond durations, and each answer's `laya` extras
(`confidence`, `act_probability`) when `extras` asked for them. A `routing` that is not an object — a
gateway answering this path with its own body — is a `MalformedAnswerError`.

`auto` also falls back between the surfaces, and every verdict is remembered for the client's life:

- A **404 is a missing route** unless it says the model does not exist — which it may do in the *code* rather
  than the message: `model_not_found`, `model_not_exist`, `model_does_not_exist`, `unknown_model`,
  `invalid_model` or `unsupported_model` is as decisive as quoting the model id beside `model`/`no such`/
  `not exist`, and a message that merely repeats the model name is not enough (`ollama` and `vLLM` answer a
  bad model id that way). A 404 that arrives *inside* a `200` body never counts: that is
  `ProviderError.embedded`, a failure the provider put in the body, and only the status line can say a route
  is missing. The call is then re-asked on another surface, walking the prompt surfaces in order (Responses,
  Chat Completions, Messages) and skipping the ones already known to be missing. An explicit `api="responses"`
  never falls back, and a pinned `method="logprobs"` never moves to Messages, which has no logprobs to carry.
  The remembered verdict only ever *skips* a route, and only where the client can speak another: a client
  whose only surface is `messages` stays on it, pays the 404 again, and reports it as a `ProviderError` with
  `status_code=404` on every call — the same error the call that learned the verdict raised.
- A **refusal that names the protocol or the route** is the same verdict from a different status: a `400`,
  `404`, `405`, `415` or `422` whose message carries `Model does not support this protocol.`,
  `ModelProtocolUnsupported`, `unsupported endpoint` (case-folded, punctuation-stripped) moves the surface
  too, and is checked before the model markers because such a refusal names the model as well. It is
  remembered like any other, and where no prompt surface is left the `ProviderError` reports status `404`
  whatever the refusal's own status was. A pinned `api=` rotates nothing: the provider's own error travels
  back, and only `auto` moves.
- A **surface that answers without a distribution** falls back for that question and is left behind for that
  model once a second answer confirms it — ollama's Responses route returns an empty logprob list,
  llama.cpp's refuses the fields, OpenRouter's refuses the logprob includable outright — so later calls start
  where the distribution is, and a distribution arriving later on a marked surface clears the mark. With no
  surface left to move to, the readout falls back to `structured`; `reasoning="native"` stops the move
  (a native plan is not Responses-only — the Messages surface resolves to native when `budget_tokens` is
  set), and so does a `grammar` request, which is a Chat Completions
  convention the other surface cannot carry.
- A **pinned `method="logprobs"` moves too**, keeping its method: it asked for a distribution, not for a
  particular surface to produce one, and the surface that refuses is not the method the caller chose. It is
  never *swapped* for another readout — with nowhere to move, the provider's refusal is reported as a
  `LabelReadoutError` whose message says the provider rejected the logprob request.
- The Messages surface is never asked for a label readout at all: `method="logprobs"`/`"grammar"` raise
  `UnsupportedMethodError` before any request, and `auto` starts in JSON there.

The state goes last (`system`, examples, question block, state) so a rubric's calls share the cacheable
prefix; a chat-list `state` carrying its own `system`/`developer` turn has that content folded into jevper's
system prompt, because a system turn after a user turn is a `400` on vLLM and SGLang and a 500 on
llama.cpp's Qwen template. Keep a `state` list to `system`/`user`/`assistant` turns.

Every string that reaches the wire is checked for encodability first: an unpaired surrogate in the state, a
state turn, a question, the model id, `prompt_cache_key` or `extra_body` is a local error naming the field,
rather than a serializer failure or a cache-key hash that fails further in. Ordinary non-ASCII text is
untouched.

The state is the content under judgement, so it is treated as data rather than as instructions: a one-value
state is wrapped in `<document>` markers with its angle brackets escaped to their JSON form, a hoisted
`system`/`developer` turn is quoted the same way, and every system prompt says the state is untrusted and
must not be obeyed. A state cannot close its own wrapper and carry on as prompt text. A list of dicts is a
conversation and keeps its roles; a list of anything else (`[1, 2]`, `["a", "b"]`) is content and is quoted
like any other value, while an empty list is refused. This is prompt hardening, not a sandbox — a model can
still be persuaded.

## Decisions API

`client.decisions.create(*, input, questions, examples=(), model=None, method=None, api=None, reasoning=None,
temperature=None, prompt_cache_key=None, extras=(), keep_alive=None) -> Decision`; the async facade awaits the
same. It is the Jev call above under OpenAI's documented request and response names, so every option that
reaches it reaches `system_one` unchanged, and `Decision.raw` is the `SystemOneResponse` it came from. An
unknown keyword is a `TypeError` — there is no `safety_identifier` parameter, the one field of the wire
request with no counterpart here.

| Wire field | Carried as |
| --- | --- |
| `model`, `reasoning`, `temperature`, `prompt_cache_key`, `extras`, `keep_alive` | the same `system_one` option |
| `input`, a string | `state`, that string |
| `input`, user messages | `state`, the messages collapsed to `{"role": "user", "content": …}` turns (a part list is joined with newlines) |
| `questions[].name` | the answer's `name`; jevper's ids here are positional, so two unnamed or two identically named questions stay distinct |
| `predicate` | `Noul(instructions=…)` |
| `choice` | `Choice(instructions=…, criteria={value: description})` |
| `choices[].value`, a string or a boolean | the option key (`"true"`/`"false"` for booleans) and the value echoed back on the answer |
| `score` | `Score(instructions=…, criteria=["<label>: <description>", …])` |

Limits and refusals, all before any request: `questions` must hold 1..200 entries (`questions must carry
1..200 questions, got N`); a `choice` needs 2–255 `choices` and a `score` 2–10 `levels` (the schema's own
bounds); an option value that is neither a string nor a boolean is refused by name (`must be a string or a
boolean, got int 1`) rather than coerced, and two values that would collapse onto one key are refused
(`question 0: choices carry both 'true' and True, which jevper cannot tell apart — its option keys are text,
so those two would be one option; give them distinct values`); `input` must be a string or a non-empty list
of user messages (`input must be a string or a list of user messages, got memoryview`); and an `input_image`
part is refused (`input message 0: input_image parts are not supported — … an image would be evidence the
model never saw`), because the text surfaces answer from text.

Answers come back in ask order, each `name` null when the question carried none:

- `predicate` (`DecisionPredicateAnswer`) — `probability`;
- `choice` (`DecisionChoiceAnswer`) — `choice`, the option's own value; `probabilities`, one `{value,
  probability}` per option, values typed as the caller wrote them; and `confidence`;
- `score` (`DecisionScoreAnswer`) — `score`, the probability-weighted level index; `probabilities`, one
  `{value: index, label, probability}` per level; and `confidence`;
- `refusal` (`DecisionRefusalAnswer`) — `type` and `name`, which jevper never emits: a refusal it meets
  raises `ModelRefusalError`. The type exists so a payload from the endpoint itself still validates.

Every probability and confidence is checked finite, and a confidence is **not** clamped to `[0, 1]`: on
these surfaces it can be the service's own number rather than jevper's. `decision.usage` (`DecisionUsage`)
holds `input_tokens`, `input_tokens_details{cached_tokens, cache_write_tokens}`, `output_tokens`,
`output_tokens_details{reasoning_tokens}` and `total_tokens`: a count the provider omitted is `None`, not
`0`, `total_tokens` is `None` unless both halves are known, and `cache_write_tokens` is always `None`,
because no surface jevper speaks reports a cache-write counter. On a prompt surface these are the sums over
the per-question calls; on `api="systemone"` they are the one call's own numbers.

`model_dump()` holds exactly `model`, `answers` and `usage` — the shape the endpoint itself returns — while
`decision.raw` carries what the wire has no room for: `system_one`'s `reasoning` and `debug`, and each
answer's `laya` extras. A `system_one` response converts with `response.to_decision()`: names are the
question ids it was asked with, a choice's values the option keys, a score's labels the `legend` texts (the
level index when the legend has no text for it), and a response assembled by hand that is missing one of its
own keys is a `MalformedAnswerError` naming it rather than a `KeyError` escaping the conversion.

`examples` follows `system_one`'s three levels, but a mapping is keyed by question **name** here, because
the ids are positional: an unmatched name (`examples is keyed by question name; no question is named 'x'`)
or a name two questions share (`… the name 'x' names more than one question; pass a sequence of Example
objects instead…`) is an `InvalidQuestionError`, and a call-level sequence outranks any id-keyed mapping,
exactly as on `system_one`. A client built with `examples={"intent": […]}` is re-keyed to this call's ids by
name, so a house default still guides a question named `intent`.

The models: `Decision`, `DecisionPredicateAnswer`, `DecisionChoiceAnswer`, `DecisionScoreAnswer`,
`DecisionRefusalAnswer`, `DecisionChoiceProbability`, `DecisionScoreProbability`, `DecisionUsage`,
`DecisionInputTokensDetails`, `DecisionOutputTokensDetails`, and `Decisions`/`AsyncDecisions`, reached as
`client.decisions`.

## Prompt caching

`assemble()` builds the prompt in cache order: the system prompt, the few-shot example turns, the question
block, then the state. A cached prefix is reused up to the first token that differs, so putting the state
last is what lets two calls about the same rubric share anything — and on a hosted API whose minimum
cacheable prefix is 1024 tokens, it is what gets a prompt carrying a few examples over the line at all. The
one exception is a chat-list state whose own last turn is the assistant's: the question then follows the
state, because a conversation ending on an assistant turn is not a question (the llama.cpp engines answer
`400 Failed to initialize samplers`, and a server that reads it as a prefill continues that turn). That
shape costs the shared prefix, and only that shape.

Every request carries `prompt_cache_key`, on both surfaces:

- the caller's, from `SystemOneClient(..., prompt_cache_key=...)` or `system_one(..., prompt_cache_key=...)`
  (the per-call key wins over the client's). It must be a non-blank string of at most 256 characters,
  checked before any request is sent — anything else raises `JevperError`.
- otherwise one derived per question from the model, the method, the example turns and the question block:
  `"jevper-"` plus 32 hex characters. The state and the system prompt are excluded on purpose, since states
  are what vary between calls about one rubric; the method is included because its system prompt and answer
  shape are part of the cached prefix, so a `logprobs` request and a `structured` one for the same question
  must not be routed into one bucket. Both passes of a two-step call pass the same key, so they land on the
  same machine even though their system prompts differ.

Either way it travels in the **request body**, never as an SDK keyword: `openai` 1.92 — the oldest version
jevper is tested against, and the first whose `responses.create` accepts `top_logprobs` — has no typed
parameter for the field, and sending one anyway raised `TypeError` inside the SDK on every call and every
surface. The wire field is the API's own, so the key is yours to choose without an SDK upgrade. jevper's own
ceiling is 256 characters; the [OpenResponses schema](https://github.com/openresponses/openresponses/blob/main/public/openapi/openapi.json)
documents a 64-character maximum for it, but OpenAI's own API reference states no length limit, so a key past
whatever a provider enforces is that provider's `400` to return, not a local refusal.

`usage.cached_tokens` is what the provider read from its cache, taken from `usage.prompt_tokens_details` on
Chat Completions, `usage.input_tokens_details` on Responses or `usage.cache_read_input_tokens` on Messages.
`None` means the provider said nothing about it — vLLM needs `--enable-prompt-tokens-details`, SGLang's Chat
route needs `--enable-cache-report` — while a reported `0` means the cache was cold or disabled. A server
that refuses the `prompt_cache_key` field has it dropped and the call re-asked, reported as
`debug["server_limits"]["cache_key"] is False`. Like every other limit it is remembered per *(model,
surface)*, so a server that refuses the field for one model still receives it for the next.

`cache_salt` is the isolation control vLLM and SGLang implement — requests sharing a salt share cached
prefixes, different salts cannot see each other's — so pass `extra_body={"cache_salt": tenant_id}` when
tenants share a server. vLLM caps it at 128 characters and rejects `@`, `/`, `\` and NUL; the other servers
ignore the field. Per-server reporting and the measured reuse: [providers.md](providers.md).

## Tracing with MLflow

jevper imports nothing from MLflow and MLflow is not a dependency, so the integration is the SDK's own:
`mlflow.openai.autolog()` and `mlflow.anthropic.autolog()` patch the OpenAI and Anthropic *resource classes*,
so every call jevper makes through a real SDK client becomes a span — the attempts it settles away from
included, which is the part `debug["llm_attempts"]` only summarises. Each SDK call is its own *trace* unless
it happens inside a span you opened: wrap the call in `@mlflow.trace` and every question's span — one per SDK
call, so a fallback costs more than one — hangs under that one parent. MLflow's active run is thread-local
rather than a context variable, so a run started with `mlflow.start_run()` is not attached to spans created on
the client's worker threads; `@mlflow.trace` is the way to get one trace per call.

| jevper surface | Span name |
| --- | --- |
| `api="chat_completions"` | `Completions` (`AsyncCompletions` from an async client) |
| `api="responses"` | `Responses` (`AsyncResponses` async) |
| `api="messages"` | `Messages.create` (`AsyncMessages.create` async) |

On a span: `mlflow.spanInputs`/`mlflow.spanOutputs` hold the kwargs jevper built and the raw provider
response (MLflow also promotes some request fields to span attributes, and which ones depends on the route),
`mlflow.chat.tokenUsage` the token counts `{"input_tokens", "output_tokens", "total_tokens"}` plus
`cache_read_input_tokens` on Responses — absent entirely when the provider sent no `usage` at all, `null` per
token when it sent one with those fields missing — `mlflow.llm.model` the model, `mlflow.llm.provider`
`anthropic` on the Messages route and absent on the OpenAI ones, `mlflow.message.format` `openai` or
`anthropic`, and `mlflow.spanLogLevel` `20` for a call that returned, `40` for one that raised, `10` for a
plain `@mlflow.trace` span of your own. The status follows the *SDK
call*, not jevper's reading of the answer: a refusal or a spent budget arrives as an HTTP `200`, so the span
is `OK` while jevper raises `ModelRefusalError` or `IncompleteAnswerError` — and a 200 the SDK itself cannot
parse is an error span, while one it tolerates and jevper then trips over is not. The span says what the
provider said, the error says whether an answer came out. A duck-typed client of your own is invisible to
autolog; wrap it in `@mlflow.trace` yourself. `mlflow.tracing.disable()` stops recording, an unwritable
tracking store does not break the call, and `mlflow.flush_trace_async_logging()` forces the export out.

Hosting jevper as a model is the other half: MLflow 3.16.1 offers `mlflow.pyfunc.PythonModel` (current),
`ResponsesAgent` (recommended for new code) and the `ChatModel` deprecated since 3.0.0 — `ChatAgent` also
exists, for agents rather than chat. The pyfunc fixtures build the client in `load_context` — a cloudpickled
instance cannot carry one — and log the code with `python_model="<path>"`; the LangChain flavour is a
picklable `SimpleChatModel` that takes its `base_url` and `model` from the `model_config` logged beside it.
`log_model` runs your `input_example` through the model while logging, so an integration that cannot answer
its own example fails there, not later. The standalone `mlflow gateway start` is deprecated in favour of
the server-hosted gateway. Behind that gateway — as opposed to calling a provider directly — two things
change: it serves no `/v1/responses` route, so `api="auto"` pays one 404 and answers on Chat Completions, and
it drops `choices[].logprobs` and `message.reasoning_content` from what it forwards, so a label readout cannot
work through it (`auto` falls back to `structured`) and `reasoning_text()` is empty there. Full detail,
verified against MLflow 3.16.1, is the library's `docs/mlflow.md`; the extra is
`pip install 'mlflow[gateway,langchain]>=3.16'`.

## Client knobs

| Option | Default | Change it when |
| --- | --- | --- |
| `method` | `"auto"` | you have a reason (see SKILL.md) |
| `api` | `"auto"` (prefers Responses, then Chat Completions, then Messages; `systemone` is never chosen by `auto`) | a client exposes several surfaces and you want a specific one |
| `native` | `False` | post to Ollaya's native `POST /api/decide` instead of `/v1/systemone`, and read its `routing` report, its own timings and each answer's `laya` extras. Requires `api="systemone"` — refused with `auto` — and it is what makes the per-call `extras`/`keep_alive` legal, since the TypeSafe route has no field for either |
| `noul_requires_question` | `True` | `False` sends a noul carrying neither `instructions` nor `criteria`, which the System One wire format otherwise requires — for servers that read the question id instead (Ollaya, CLM, kev) |
| `top_logprobs` | `20` | an integer in `[0, 20]`, enforced locally for every method; lower it only if the provider rejects the field. A pinned `logprobs`/`grammar` needs at least 2, since one logprob is not a distribution. A provider whose cap is lower refuses the *value* (`Invalid 'top_logprobs': integer must be between 0 and 5`), which falls back for that question without writing logprobs off for good |
| `structured_outputs` | `True` | `False` sends `{"type": "json_object"}` instead of a strict schema on Chat Completions and Responses, and the schema then travels in the system prompt — for providers that reject strict schemas; a server that refuses the format field outright then skips that rung and is re-asked without one. The Messages route has no softer shape to send: the field is simply not sent there and the prompt carries the schema either way |
| `prompt_cache_key` | `None` (derived per question) | route one rubric's calls to a reusable cache; non-blank, at most 256 characters. It is a routing label, not an isolation boundary — for tenants sharing a server use the provider's own control (`extra_body={"cache_salt": ...}`, below) |
| `normalize_probabilities` | `True` | `False` returns the model's `structured` numbers verbatim — the provider's own values, a mass above 1 included, with the error still recorded in `debug`; a negative or non-finite value is a `MalformedAnswerError` either way. A `Score` is still read off the rescaled distribution, so `score` stays on the 0..N-1 line while the reported probabilities do not |
| `max_concurrency` | `8` | your provider rate-limits per key |
| `n_retry_malformed` | `1` | a model that keeps answering in prose |
| `retry` | `RetryPolicy()` | transient-failure retries: `n_retries=2`, `base_delay=0.5`, `max_delay=8.0`, `respect_retry_after=True`. The transient set is `408`, `409`, `429` and any `5xx`, plus connection and timeout errors matched by class name (a `ConnectionProgrammingError` of your own is not one); an `x-should-retry` header outranks it in both directions. A `Retry-After` (delta-seconds or an HTTP date) or a numeric `retry-after-ms` replaces the backoff, which `max_delay` does not cap and a 24-hour ceiling does; `respect_retry_after=False` keeps the curve alone. An official SDK's own retry loop is disabled on the copy jevper makes, so `usage.n_retries` counts every retry — your client keeps its own setting |
| `extra_body`, `extra_headers` | `None` | provider-specific fields — including `max_tokens` on the Messages surface, where jevper's `1024` default may be too small, and the local servers' `chat_template_kwargs` that turns thinking off. A key named here is what reaches the wire (the SDK merges `extra_body` last), so it also wins over jevper's own value for that field; jevper's own copy of a capability field is dropped with it when a server refuses that field. Both mappings are **copied at construction**, so editing yours afterwards changes nothing. Three keys are refused locally instead: `model` (use `model=`, which is validated and hashed into the cache key), a truthy `stream` (jevper reads one whole non-streaming response), and a value that refers to itself, which no request could carry. A header name must be an HTTP token and its value printable ASCII (a horizontal tab allowed), so a CRLF, NUL, `é` or lone surrogate is a local `JevperError` rather than a provider failure. `{"logprobs": false}` switches the request's logprob fields off on both OpenAI surfaces; `{"logprobs": true}` does not suppress jevper's `top_logprobs`. A header named in a different case replaces the SDK's own rather than adding a second credential — and a value under a credential-sounding name (`authorization`, `api-key`, `token`, `secret`, `cookie`, `password`, `signature`, …) is recorded in `debug` and `attempts` as `<redacted>` while the wire keeps the real one |

Constructor misuse (unknown `method`/`api`, a count option that is not an integer, `top_logprobs` outside
`[0, 20]` or below 2 with a pinned label method, `max_concurrency < 1`, a negative retry field, a blank or
over-long `prompt_cache_key`, a `model` that is not a non-blank string, a header name or value the HTTP layer
could not carry) raises `JevperError` immediately, so a typo never reaches a provider — and so does a
per-call `api=""`/`method=""` or `model=""`,
since an override is only used when it is not `None`. The `systemone` route checks its own two call arguments
the same way: `extras` must be a sequence of names (a bare string is a `JevperError`), and `extras` or
`keep_alive` without `native=True` — including when they arrive through `extra_body` — is a
`ClientCapabilityError`, because the route the wrapper posts to has no field for either; the same applies to
`native=True` with `api="auto"`. A question type is checked *twice* — where it is built
and again in `system_one` — and both paths raise `InvalidQuestionError` (a `JevperError`), naming the field
that was wrong: `Choice: weight: Extra inputs are not permitted`, or with an id,
`question 'intent' is invalid: Choice: bogus: Extra inputs are not permitted`. `ReasoningConfig` is a plain
pydantic model and not one of those, so its own validation (`effort`, `budget_tokens`, `mode`) still raises
`pydantic.ValidationError` at the point you build it, not from the client; `budget_tokens` stays validated if
you assign to it afterwards.
