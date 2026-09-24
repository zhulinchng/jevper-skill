# jevper providers and surfaces

- [Does this provider do logprobs?](#does-this-provider-do-logprobs) — the observed matrix, and the one-line probe
- [Surfaces](#surfaces) — what each one carries, and how `api="auto"` picks
- [Local servers](#local-servers) — what to pass, how to turn thinking off, what each server ignores
- [The Messages route](#the-messages-route) — the Anthropic API: who implements it, and what it lacks
- [Prompt-cache reporting](#prompt-cache-reporting) — who reports `cached_tokens`, and what the counts look like

## Does this provider do logprobs?

Observed against each provider's live API, as of jevper 0.5.3 — providers move, so treat a row as a
starting point rather than a law, and let `auto` verify it per `(model, surface)` for you.

| Provider | `logprobs` | Note |
| --- | --- | --- |
| OpenAI `gpt-4o`, `gpt-4.1` | yes | |
| OpenAI reasoning models (`o`-series, `gpt-5` family) | no | `400 logprobs are not supported with reasoning models.` |
| OpenAI Responses surface | partial | `include` returns the sampled token and no alternatives; a model with no includable logprobs refuses the list outright (`400 Unsupported parameter: 'include' is not supported with this model.`) — the refusal counts, because on this surface `include` is the carrier, while a Chat Completions error merely *mentioning* `include` is about something else |
| Anthropic Claude | no | no logprob API at all |
| Gemini via the OpenAI-compatibility endpoint | no | `400 Unknown name "logprobs": Cannot find field.` |
| Gemini native API | yes | not reachable through an OpenAI-compatible client |
| DeepSeek | yes | `top_logprobs` up to 20 |
| Together | yes | send `top_logprobs` for alternatives; `logprobs: 1` alone returns the sampled token |
| Ollama | partial | local builds since Nov 2025 return them on Chat Completions; its `/v1/responses` route returns an empty logprob list. Ollama Cloud and older builds report none |
| llama.cpp | yes | Chat Completions only: the `/v1/responses` shim refuses the fields (`400 top_logprobs requires logprobs to be set to true`), so `auto` re-asks on Chat Completions |
| vLLM | yes | caps `top_logprobs` at `--max-logprobs` (20 by default); its `/v1/responses` carries them through `include` |
| SGLang | yes | its `/v1/responses` needs `top_logprobs` sent explicitly (it defaults to 0) — jevper always sends it |
| OpenRouter | per model | it routes by price, and the endpoint it picks may ignore `logprobs`; add `extra_body={"provider": {"require_parameters": True}}` to route only to endpoints that support every field you send. Its Responses API refuses the logprob includable (`400 Invalid option: expected one of …` at `path: ["include", 0]`), so a label readout there moves to Chat Completions, where the distribution arrives — with a pinned `method="logprobs"` as much as with `auto` |
| everything else | unknown | reasoning models and thin compatibility layers are the ones that say no |

With `auto` you do not have to know this table. Probe a new endpoint before writing code, or after
switching models:

```sh
python scripts/offline_stub.py --live --model <model-id>                  # OpenAI-compatible
python scripts/offline_stub.py --live --model <model-id> --api messages   # Anthropic-compatible
```

It prints the resolved method per question, the surface actually used, the readout source, `n_calls`,
`usage.cached_tokens` and `debug["server_limits"]` for one real call. The `messages` probe needs the
`anthropic` package and `ANTHROPIC_BASE_URL` (or `OPENAI_BASE_URL`) for a local server.

The probe asks the server what the model advertises *before* spending a call — `GET {base_url}/models`, and
OpenRouter's per-provider `GET {base_url}/models/<id>/endpoints`, neither of which counts against a request
quota — and prints `preflight` (logprobs? a strict schema? a plain `json_object`? neither?). A model
advertising no logprob field is read with `structured`; one advertising no format field also loses the
schema, and the shape then rests on the prompt alone.

Local servers publish no such metadata, so the probe prints `preflight: null` against them and spends its one
call; that is the unknown case, not a failure. What they do need is in the request, and `--extra-body` is how
the probe carries it — against a server whose template thinks, pass
`--extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'` (vLLM, SGLang, llama.cpp) or
`--extra-body '{"reasoning_effort": "none"}'` (ollama), or the first token the label readout sees is the
reasoning rather than the label. A failure is one screen: the error class, the status if there was one, the
provider's own words truncated, and what to do about it.

A quota, credit or key refusal is reported as what it is — the status, the provider's own words (truncated),
and the next step — because it is not a jevper failure. On OpenRouter that means: `free-models-per-day` is
one **account-wide** cap, 50 requests/day with no credits and 1000 once $10 or more is purchased, reset
midnight UTC, so a different `:free` model id does not get around an exhausted cap; a few `:free` ids answer
`403` because they are reserved for agentic harnesses; and `402` means the account is out of credits.
`429` and `503` may carry `Retry-After`, which the OpenAI and Anthropic SDKs honour on their own — jevper
does not read it.

## Surfaces

`api="auto"` (the default) prefers the Responses surface, which carries native reasoning and encrypted
content — except for `grammar`, which only Chat Completions can carry. The Messages API is the last choice
of the three, and the only one with no label readout. A client missing the attribute the chosen surface
needs raises `ClientCapabilityError` naming the surface to pass explicitly.

| | Chat Completions | Responses | Messages |
| --- | --- | --- | --- |
| messages | `messages=[...]` | `input=[...]`, plus `store=false` | `messages=[...]` + top-level `system` |
| logprobs | `logprobs=true`, `top_logprobs=N` | `top_logprobs=N`, `include=["message.output_text.logprobs"]` | none — the API has no such field |
| JSON schema | `response_format={"type": "json_schema", "json_schema": {...}}` | `text={"format": {"type": "json_schema", ...}}` | none — the schema goes in the system prompt |
| schema fallback (`structured_outputs=False`) | `response_format={"type": "json_object"}` | `text={"format": {"type": "json_object"}}` | the prompt, always |
| grammar | `extra_body={"grammar": "..."}` | not available | not available |
| reasoning | `reasoning_effort` (only when `effort` is set) | `reasoning={effort, summary, context}` | `thinking={"type": "enabled", "budget_tokens": n}` (only when `budget_tokens` is set) |
| output cap | server default | server default | `max_tokens` required: jevper sends `1024` |
| cache key | `prompt_cache_key` | `prompt_cache_key` | not a field of this API |

Neither OpenAI builder ever sends `max_tokens`, `max_completion_tokens` or `max_output_tokens`: reasoning
tokens count against those caps, and a small cap silently truncates a reasoning model. The Messages
protocol is the exception — it has no server-side default, so jevper always sends one there and
`extra_body={"max_tokens": n}` overrides it. Anything else provider-specific goes through `extra_body` too.

OpenRouter's Responses API is stateless — `store: true` or a `previous_response_id` is a `400` — so the
`store=false` jevper already sends is the form it accepts, and the whole history travels in `input` each call.

Where a request cannot state the answer's shape — the Messages API has no schema field, and a server that
refuses the strict schema is re-asked with a plain JSON object — jevper appends the JSON Schema to the
system prompt, so the model still knows which keys to answer with. The answer's shape is then only as good
as its instruction-following: on such a surface expect more `MalformedAnswerError`s, set
`temperature=0.0`, and keep the criteria descriptions unambiguous.

A client object cannot say whether the *server* implements a route — `openai.OpenAI` exposes
`responses.create` either way — so `auto` reads the responses:

- **404 that does not name the model** → the route is missing: the call is re-asked on the other surface
  and remembered for the client's life — but only where the client can speak it. A Messages-only client
  whose host has no `/v1/messages` route keeps re-asking and keeps reporting the 404
  (`ProviderError`, `status_code=404`) rather than moving to an attribute it does not have. `ollama` and
  `vLLM` answer a bad model id with a 404 that quotes the model; those are reported as they stand, on
  either surface.
- **A surface that answers without a distribution** → marked and left behind for that model: ollama's
  Responses route returns an empty logprob list, llama.cpp's refuses the fields and OpenRouter's refuses
  the includable, while Chat Completions on all three carries the full distribution. A distribution
  arriving later on a marked surface clears the mark.
- **`reasoning="native"` pins the surface**, because native reasoning is the reason to prefer Responses and
  switching would turn it into a two-step pass silently — and so does a `grammar` request, which the other
  surface cannot carry. A pinned `method="logprobs"` does move, keeping its method.
- An explicit `api="responses"` is a decision, not a preference: its 404 reaches you unchanged.

When a server refuses a request field jevper added — `response_format`, `text.format`, `reasoning_effort`,
`reasoning`, the `include` list, `prompt_cache_key`, or the Messages `thinking` field — the field is dropped
and the same call re-asked, one step down the ladder at a time (`json_schema` → `json_object` → nothing),
remembered per surface and reported in `debug["server_limits"]` for the surface that answered. None of them
is needed to answer the question.

## Local servers

A local server is the same client with a different `base_url`. Checked on one 12 GB card against ollama
0.34.3 and llama.cpp serving `Qwen3.5-9B-Q4_K_M`, vLLM 0.30.1 and SGLang 0.5.20 serving
`Qwen3.5-9B-AWQ-4bit`, and LM Studio's `llmster` 0.0.25 serving `Qwen3-4B-Instruct-2507`.

| Server | `base_url` | `model` | Thinking off | Notes |
| --- | --- | --- | --- | --- |
| ollama | `http://127.0.0.1:11434/v1` | the tag you pulled | `extra_body={"reasoning_effort": "none"}` | Chat Completions carries logprobs; the Responses route returns an empty logprob list |
| llama.cpp | `http://127.0.0.1:8080/v1` | the `--alias` value | `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` | serve with `--jinja`; the only server that honours `grammar`. Its Responses route accepts `text.format` and ignores it, while Chat Completions turns the schema into an enforced grammar — structured work belongs on Chat |
| vLLM | `http://127.0.0.1:8000/v1` | the `--served-model-name` value | `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` | serve with `--reasoning-parser qwen3`; `top_logprobs` capped by `--max-logprobs` (20) |
| SGLang | `http://127.0.0.1:30000/v1` | the `--served-model-name` value | `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` | serve with `--reasoning-parser qwen3`; its Responses route needs `top_logprobs`, which jevper always sends |
| LM Studio | `http://127.0.0.1:1234/v1` | the id `lms ls` prints | nothing reliably — load an instruct model | all three surfaces on one box, as on the other four; logprobs arrive on both OpenAI surfaces, but its Responses route ignores the schema, so structured answers belong on Chat Completions |

`api="auto"` works against all five: it prefers Responses, and when that route is missing or answers
without a distribution it re-asks on Chat Completions and remembers the verdict. `api="chat_completions"`
skips the discovery entirely. All five also answer the Anthropic Messages route, so `auto` has three
surfaces to choose from on any of them; ollama, llama.cpp, SGLang and LM Studio were exercised through it
directly, and vLLM answers `200` with a thinking block and **no text** for a Qwen3-4B model — jevper reports
an empty answer naming the stop reason, and the cause is the server's template, so use its OpenAI surfaces.

**Thinking is the one decision you must make.** `logprobs` and `grammar` read a one-token
answer, and every one of these servers reports logprobs for *every* generated token — with thinking on,
that is the first token of the reasoning, not the label, and the readout raises `LabelReadoutError`. Turn
it off for classification work: it costs a whole reasoning pass to choose one letter. Llama.cpp also
accepts `reasoning_effort: "none"` (its `--jinja` template) and `reasoning_budget: 0`; ollama's native API
has a per-model `think` setting. `structured` and `discrete` are unaffected by thinking — they read the
answer text, and the trace lands in `response.reasoning`.

Measured on `qwen3:4b-thinking-2507` through ollama 0.34.3: `extra_body={"reasoning_effort": "none"}` does
reach the template and does change the answer — to prose (`First …`), which a label readout cannot read, so
the error names that first token. A thinking model there wants its own `think` setting through ollama's
native API, or a non-thinking model; a `structured` or `discrete` call is unaffected either way.

A label readout also survives thinking when the server separates the trace *and* the token stream ends
exactly with the answer text: jevper anchors on that tail and reads the answer's own first token. The
anchor is strict on purpose, with one allowance — vLLM and SGLang append their end-of-turn token
(`<|im_end|>`) after the answer, so up to two trailing tokens that cannot be part of the answer are
dropped before the tail is tested.

All five ignore unknown request fields, so a field that does not apply is not an error. Two exceptions,
and the silent ones:

- `grammar` is a llama.cpp convention — ollama, vLLM and SGLang ignore it, the model answers
  unconstrained, and the label readout reports a non-label first token instead of a grammar failure.
- `reasoning_effort` reaches the chat template on ollama, llama.cpp and SGLang; vLLM validates it against
  its own enum and answers `400` for a value outside it (`xhigh` and `max` are the usual casualties).
- `strict: true` is ignored by ollama, honoured by vLLM and SGLang.
- `max_completion_tokens` is ignored by ollama, which only knows `max_tokens`.
- A small-context model refuses before inference: vLLM answers
  `400 max_tokens=2048 cannot be greater than max_model_len=max_total_tokens=1024`. jevper forwards that
  400 with the numbers in it; bound the output yourself with `extra_body={"max_tokens": n}`.
- `n` is rejected outright by llama.cpp (`1 <= value <= 1`); vLLM and SGLang accept it and return two
  choices, where jevper reads the first.
- A `developer`-role message is a `400` (`Unexpected message role.`) on SGLang. jevper's own turns never
  use another role and a chat-list `state`'s `system`/`developer` turns are folded into the system prompt,
  so keep `state` to `system`/`user`/`assistant`.
- LM Studio's Responses route accepts `text.format` with a strict `json_schema` and **ignores it**
  (structured output is a Chat Completions feature there; bug-tracker #2403, #1396). jevper sent the
  schema in the request, so — the field being accepted — it is not repeated in the prompt, and
  `method="structured"` on that surface reads whatever the model invents. Use `api="chat_completions"`, or
  `method="logprobs"`, for structured work there.
- An unknown path is not a `404` on LM Studio: it answers `200` with
  `{"error": "Unexpected endpoint or method. (POST /…)"}` (#618), which jevper reads as an embedded
  provider error rather than a missing route — right for a real endpoint failing, but do not rely on route
  discovery to catch a typo'd path there.
- LM Studio's routes disagree about `reasoning_effort`: honoured on `/v1/responses`, ignored on
  `/v1/chat/completions` (#2413). `/v1/responses` also ignores `instructions` (#1154).

The library's [local-servers.md](https://github.com/zhulinchng/jevper/blob/main/docs/local-servers.md)
holds the full measurements: raw HTTP shapes per server, cache inspection and flush endpoints, and what
fits a 12 GB card.

## The Messages route

Every server here also implements the Anthropic Messages API (`POST /v1/messages`), so `api="messages"`
works against each of them: an `anthropic.Anthropic` client pointed at the same host and port as the OpenAI
one, passed in place of it. What differs is how much of the protocol each implements — the version is the
first release that ships the route. LM Studio serves it too, and OpenRouter implements it as its Anthropic
skin (`ANTHROPIC_BASE_URL=https://openrouter.ai/api`, documented by OpenRouter rather than measured here).

| Server | Since | `thinking` field | Thinking blocks back | `usage` cache counts |
| --- | --- | --- | --- | --- |
| LM Studio | 0.4.1 | accepted, answer still separated from it | `thinking` blocks when the model thinks | `cache_read_input_tokens`, including a reported `0` on a cold call |
| llama.cpp | b7187 | accepted, and the budget grows `max_tokens` as below | reported | yes (`cache_read_input_tokens`, measured) |
| vLLM | 0.11.1 | **accepted and ignored**: its request model has no `thinking` field and pydantic drops the extra, so no downgrade fires and the answer comes back with no thinking and no way to tell | `thinking` blocks | yes |
| SGLang | 0.5.9 | **refused**: this version has no `thinking` field at all, so the request is answered `400` and jevper drops the field and re-asks | `thinking` blocks | yes |
| ollama | 0.14.0 | accepted, but `budget_tokens` is **not enforced** | `thinking` blocks | yes (`cache_read_input_tokens`, measured; the Chat route reports nothing) |

Four protocol facts shape what the client does on this surface:

- **No logprobs exist in it** — not withheld by some servers, absent from the API. `method="logprobs"` and
  `"grammar"` raise `UnsupportedMethodError` before any request is sent, and `method="auto"` answers with
  `structured` without spending a call to find out.
- **No schema field either**, so `structured`/`discrete` carry the JSON Schema in the system prompt, and the
  answer's shape is only as good as the model's instruction-following.
- **`max_tokens` is required** by vLLM's and SGLang's implementations and has no default on any of them, so
  jevper always sends one: `1024`, or `1024` plus the caller's `ReasoningConfig(budget_tokens=n)`, because
  this API also requires the thinking budget to be strictly *below* `max_tokens` and would refuse the 1024
  its own documentation calls the floor. `extra_body={"max_tokens": n}` wins outright, and a value too small
  to hold the budget raises `JevperError` locally, naming both numbers. Measured on all five:
  a 1024 budget sends `max_tokens: 2048`, a 2048 budget sends `3072`. Thinking is a budget, not an effort
  name: `ReasoningConfig(budget_tokens=n)` sends `thinking={"type": "enabled", ...}` and `effort` is never
  translated into one. A server that refuses the *value* (`budget_tokens: must be at least 1024`) keeps its
  own error rather than being re-asked with your reasoning silently switched off; one that does not know the
  field at all has it dropped and the call re-asked, reported in `debug["server_limits"]["thinking"]`.
- **`temperature` is left out of a Messages request that enables `thinking`** — the API refuses a
  non-default temperature beside thinking, so jevper omits it there. Set it on the OpenAI surfaces, or turn
  thinking off, if you were counting on it.
- **A `system` role inside `messages` is not part of the API** — Anthropic has since added mid-conversation
  `system` messages, but none of these servers implements them, rendering a `system` turn positionally into
  the chat template instead — so jevper moves it to the top-level `system` field, where it cannot be dropped
  or rejected. All four servers answer `200` for one on this route; the
  `400 System message must be at the beginning.` that vLLM and SGLang give belongs to the *OpenAI* surfaces.

The reasoning parsers matter here too. With thinking left on — vLLM's and SGLang's templates default to it —
the parser can put the whole generation into a thinking block and return no text block at all, so there is
nothing to read: a `structured` call raises `MalformedAnswerError`, whose message says the response carried
reasoning only. Turn thinking off per call exactly as on the other surfaces
(`extra_body={"chat_template_kwargs": {"enable_thinking": False}}`) — but on this surface in 0.5.3 that body
only reaches the wire when a `temperature` is set and jevper's own thinking is off, so pass `temperature=0.0`
beside it or the knob is silently dropped; with that, every scenario on vLLM's
Messages route answers. SGLang needs one more server-side decision: with `--reasoning-parser qwen3` and a
*non-thinking* model, whose template has no `enable_thinking` to set, the parser never sees the closing
marker it waits for and classifies the whole generation as reasoning — dropping `--reasoning-parser` fixes
it. That is a property of serving that model with that parser, not of the client.

Reading a response on this surface: text blocks become the answer, thinking blocks become
`response.reasoning` parts with their `signature` kept, `stop_reason` becomes `CallResult.stop`, and
`usage.cache_read_input_tokens` becomes `usage.cached_tokens`.

## Prompt-cache reporting

All four local servers cache the prompt prefix by default and ignore the hosted APIs' cache-control fields;
what they disagree on is telling you it happened.

| | ollama | llama.cpp | vLLM | SGLang |
| --- | --- | --- | --- | --- |
| `usage.prompt_tokens_details.cached_tokens` (Chat) | always | always | needs `--enable-prompt-tokens-details` | needs `--enable-cache-report` |
| `usage.input_tokens_details.cached_tokens` (Responses) | always | always | always | always |
| `usage.cache_read_input_tokens` (Messages) | yes | yes | yes | yes |

jevper reads all three paths into `usage.cached_tokens` and leaves it `None` when the server says nothing.
The Messages route is reported by all four, so the flags that decide whether you get a number are the Chat
Completions ones — a server that says nothing is not the same as one that reported `0`, which is a cold or
disabled cache and is preserved as `0`. A server that refuses the `prompt_cache_key` field has it dropped
and the call re-asked (`debug["server_limits"]["cache_key"] is False`); on all five local servers the field
is accepted with `200` and ignored. LM Studio reports per route rather than per server: Chat Completions
carries no cached-token count at all (so `usage.cached_tokens` is `None` even when the cache was used),
Responses reports `input_tokens_details.cached_tokens` and Messages `cache_read_input_tokens`.

Reuse depends on message order — the state goes last, so a rubric's calls share everything before it.
Measured on one 2388-token prompt (two examples, a ~1300-token state), second call differing only in the
state:

| Message order | ollama | llama.cpp | vLLM | SGLang |
| --- | --- | --- | --- | --- |
| `state, examples, question` (jevper ≤ 0.3.0) | 0 | 40 | 0 | — |
| `examples, question, state` (jevper ≥ 0.4.0) | 0 | **1010** | **528** | **896** |
| identical repeat of the same request | 2384 | 2384 | 2112 | 2368 |

ollama reports the field but credited no part of a *state-varied* prefix here; its cache is per loaded
model runner, so `keep_alive` (native API only — its OpenAI route ignores the field) is what keeps it warm.
`cache_salt` is the isolation control vLLM and SGLang both implement: requests sharing a salt share cached
prefixes, and different salts cannot see each other's. Pass it per deployment when tenants share a server
(`extra_body={"cache_salt": tenant_id}`); vLLM caps it at 128 characters and rejects `@`, `/`, `\` and NUL,
while llama.cpp and ollama ignore it.
