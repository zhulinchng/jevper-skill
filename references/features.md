# jevper features

- [Reasoning](#reasoning) — `ReasoningConfig`, native vs two-step, reading the trace
- [Few-shot examples](#few-shot-examples) — `Example`, the three levels, calibration
- [Async](#async) — `AsyncSystemOneClient`
- [Surfaces](#surfaces) — Chat Completions, Responses and Messages, and the fallbacks between them
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
- On the chat surface a two-step analysis call sends no reasoning parameters at all — the analysis prompt
  *is* the reasoning step. Use `mode="native"` if you want the provider's own reasoning there.
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
- `response.debug["reasoning_mode"]` reports `off`, `native` or `two_step`. `reasoning_text()` prefers
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
index, or a bool for `Noul`. A criteria key is matched exactly first, then the label: with criteria
`{"b", "a"}`, the answer `"a"` names the option keyed `"a"` rather than the first label. Examples are
rendered as chat turn pairs — the example state plus the same question block, then the expected answer **in
the format the active method expects** (a label for `logprobs`/`grammar`/`discrete`, a JSON object for
`structured`) — so switching methods needs no change to your examples.

Three levels, first non-empty wins, no merging:

1. `question.examples` — `Choice(criteria=..., examples=[...])`
2. `system_one(..., examples=[...])` or `examples={"intent": [...]}` for one question id
3. `SystemOneClient(..., examples=[...])` — the house default for every call

A bare sequence applies to every question in the call; a mapping is keyed by question id. Every question's
examples are resolved and validated before the first request — a call that is locally invalid spends
nothing, rather than failing inside one worker after the questions ahead of it have already paid. An answer
that matches no option raises `InvalidQuestionError`. `probabilities` is only read by
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
content — except for `grammar`, which only Chat Completions can carry, and except for a client whose only
surface is `messages`. A client missing the attribute the chosen surface needs raises
`ClientCapabilityError` naming the surface to pass explicitly.

| | Chat Completions | Responses | Messages |
| --- | --- | --- | --- |
| input | `messages=[...]` | `input=[...]`, `store=false` | `messages=[...]` + top-level `system` |
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

The schema travels in the prompt wherever the request cannot be relied on to state it: always on the Messages
surface, and on the OpenAI surfaces when `structured_outputs=False` or the server refused the strict schema.
On the Messages surface the prompt keeps it even when the field was sent, because a server can accept
`output_config` and drop it without a word — the same silence as a server that never read it. It joins the
leading system message, so the question block keeps its place — and where the request no longer states the
shape, the answer's shape is only as good as the model's instruction-following.

Anthropic's structured outputs implement a documented subset of JSON Schema, and an unsupported keyword is a
`400` rather than a warning. jevper's own schemas bound every probability at `minimum: 0`, so the wire
schema has each such bound moved into the description of the field it bounded (`Must be at least 0.`) while
the prompt keeps the full schema, where text can say what a constraint says.

`auto` also falls back between the surfaces, and both verdicts are remembered for the client's life:

- A **404 that does not name the model** is a missing route: the call is re-asked on the other surface. A
  404 that quotes the model (`ollama` and `vLLM` answer a bad model id that way) is about the model and is
  reported as it stands. An explicit `api="responses"` never falls back. The remembered verdict only ever
  *skips* a route, and only where the client can speak the other one: a client whose only surface is
  `messages` stays on it, pays the 404 again, and reports it as a `ProviderError` with `status_code=404` on
  every call — the same error the call that learned the verdict raised.
- A **surface that answers without a distribution** is marked and left behind for that model — ollama's
  Responses route returns an empty logprob list, llama.cpp's refuses the fields, OpenRouter's refuses the
  logprob includable outright — so later calls start where the distribution is. A distribution arriving later
  on a marked surface clears the mark. With no surface left to move to, the readout falls back to
  `structured`; `reasoning="native"` stops the move (native reasoning exists only on Responses), and so does
  a `grammar` request, which is a Chat Completions convention the other surface cannot carry.
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

The state is the content under judgement, so it is treated as data rather than as instructions: a one-value
state is wrapped in `<document>` markers with its angle brackets escaped to their JSON form, a hoisted
`system`/`developer` turn is quoted the same way, and every system prompt says the state is untrusted and
must not be obeyed. A state cannot close its own wrapper and carry on as prompt text. A list of dicts is a
conversation and keeps its roles; a list of anything else (`[1, 2]`, `["a", "b"]`) is content and is quoted
like any other value, while an empty list is refused. This is prompt hardening, not a sandbox — a model can
still be persuaded.

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

`usage.cached_tokens` is what the provider read from its cache, taken from `usage.prompt_tokens_details` on
Chat Completions, `usage.input_tokens_details` on Responses or `usage.cache_read_input_tokens` on Messages.
`None` means the provider said nothing about it — vLLM needs `--enable-prompt-tokens-details`, SGLang's Chat
route needs `--enable-cache-report` — while a reported `0` means the cache was cold or disabled. A server
that refuses the `prompt_cache_key` field has it dropped and the call re-asked, reported as
`debug["server_limits"]["cache_key"] is False`.

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
| `api` | `"auto"` (prefers Responses, then Chat Completions, then Messages) | a client exposes several surfaces and you want a specific one |
| `top_logprobs` | `20` | an integer in `[0, 20]`, enforced locally for every method; lower it only if the provider rejects the field. A pinned `logprobs`/`grammar` needs at least 2, since one logprob is not a distribution. A provider whose cap is lower refuses the *value* (`Invalid 'top_logprobs': integer must be between 0 and 5`), which falls back for that question without writing logprobs off for good |
| `structured_outputs` | `True` | `False` sends `{"type": "json_object"}` instead of a strict schema on Chat Completions and Responses, and the schema then travels in the system prompt — for providers that reject strict schemas; a server that refuses the format field outright then skips that rung and is re-asked without one. The Messages route has no softer shape to send: the field is simply not sent there and the prompt carries the schema either way |
| `prompt_cache_key` | `None` (derived per question) | route one rubric's calls to a reusable cache; non-blank, at most 256 characters. It is a routing label, not an isolation boundary — for tenants sharing a server use the provider's own control (`extra_body={"cache_salt": ...}`, below) |
| `normalize_probabilities` | `True` | `False` returns the model's `structured` numbers verbatim — the provider's own values, a mass above 1 included, with the error still recorded in `debug`; a negative or non-finite value is a `MalformedAnswerError` either way. A `Score` is still read off the rescaled distribution, so `score` stays on the 0..N-1 line while the reported probabilities do not |
| `max_concurrency` | `8` | your provider rate-limits per key |
| `n_retry_malformed` | `1` | a model that keeps answering in prose |
| `retry` | `RetryPolicy()` | transient-failure retries: `n_retries=2`, `base_delay=0.5`, `max_delay=8.0`, `respect_retry_after=True` — a `Retry-After` header (delta-seconds or an HTTP date) or a numeric `retry-after-ms` replaces the backoff, which `max_delay` does not cap; set it `False` for the curve alone |
| `extra_body`, `extra_headers` | `None` | provider-specific fields — including `max_tokens` on the Messages surface, where jevper's `1024` default may be too small, and the local servers' `chat_template_kwargs` that turns thinking off. A key named here is what reaches the wire (the SDK merges `extra_body` last), so it also wins over jevper's own value for that field; jevper's own copy of a capability field is dropped with it when a server refuses that field |

Constructor misuse (unknown `method`/`api`, a count option that is not an integer, `top_logprobs` outside
`[0, 20]` or below 2 with a pinned label method, `max_concurrency < 1`, a negative retry field, a blank or
over-long `prompt_cache_key`, a `model` that is not a non-blank string) raises `JevperError` immediately, so
a typo never reaches a provider — and so does a per-call `api=""`/`method=""` or `model=""`,
since an override is only used when it is not `None`. `ReasoningConfig` is a pydantic model, so its
own validation (`effort`, `budget_tokens`, `mode`) raises `pydantic.ValidationError` at the point you build
it, not from the client; `budget_tokens` stays validated if you assign to it afterwards.
