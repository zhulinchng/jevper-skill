# jevper troubleshooting

- [Auto and the provider](#auto-and-the-provider) — what `auto` resolves on its own, and the probe
- [Error triage](#error-triage) — every error class, its cause, the first move
- [Inspecting a failed call](#inspecting-a-failed-call) — `debug` keys and what they mean
- [Symptom → fix](#symptom--fix) — the recurring ones

## Auto and the provider

`auto` reads the provider for you: it asks for logprobs, and on a `400`/`403`/`422` that names the logprob
fields, an answer with no logprobs, or an answer token with no alternatives, it re-asks with `structured`.
A rejection is remembered per `(model, surface)` for the rest of the client's life; a response that merely
lacks logprobs falls back for that question but is written off only after a second one. Before that
fallback it tries the *other surface* when the client exposes one and the reasoning mode is not `native` — a
pinned `method="logprobs"` takes that move too, keeping its method — and a 404 is read as a missing route,
answered on the other surface where the client has one, unless it quotes the model id and says the model
does not exist. A rejection that names a *value* rather than a field (`top_logprobs must be between 0 and
20`) and a 5xx that survives its retries both fall back for that question alone, without being remembered.
The same distinction holds on the capability ladder: only a complaint about a field's *existence* drops it
— the schema excepted, since the prompt keeps the schema either way — and a server that refuses the number
in a field it knows (`budget_tokens: must be at least 1024`) keeps its own error, because answering with your
reasoning silently switched off would be worse than failing.

The Messages surface is the exception to all of it: the API has no logprob field, so `auto` answers with
`structured` there without spending a call to discover that, and an explicit `method="logprobs"`/`"grammar"`
raises `UnsupportedMethodError` before a request is sent.

The observed logprob matrix, the surface rules and the local-server tables live in
[providers.md](providers.md). Probe a new endpoint or a newly switched model before writing code:

```sh
python scripts/offline_stub.py --live --model <model-id>   # resolved method, surface, readout, cached_tokens
python scripts/offline_stub.py --live --model <model-id> --api messages   # Anthropic-compatible route
python scripts/offline_stub.py --live --model <model-id> --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'   # e.g. thinking off on a local server
```

## Error triage

| Error | Cause | First move |
| --- | --- | --- |
| `InvalidQuestionError` | the question or an example is locally invalid, **and it is now raised on both paths**: building one (`Choice(criteria={})`, an unknown field, `Score` outside 2–10 levels, `Noul` keys other than `true`/`false`, more than 26 options under `logprobs`/`grammar`) and handing one over as data, where the message carries the question id — `question 'intent' is invalid: Choice: bogus: Extra inputs are not permitted`. pydantic's own `ValidationError` no longer escapes, so one `except JevperError` covers a rubric written in Python and one loaded from data. An example answer that matches no option, probabilities whose key set is not the question's, a `Noul` example keying an answer two ways (`{True: 0.2, "true": 0.8}`) or a container that is not a sequence of `Example` objects are the same error — the first three where the question is built, the last when a call pairs them | fix the question — zero requests were sent, and the message names the field that was wrong |
| `UnsupportedMethodError` | `method="grammar"` with a Responses surface, or `method="logprobs"`/`"grammar"` with the Messages surface — that API has no logprobs at all | pass `api="chat_completions"` (or drop `grammar`), or use `structured`/`discrete` |
| `ClientCapabilityError` | the client object lacks `responses.create` / `chat.completions.create` / `messages.create`, the response carried no usable first choice (missing or empty) and no explanation of why, or the facade and the client disagree — a blocking `SystemOneClient` with an async SDK client, or the reverse. An event *stream* is not this error: it is `ProviderError` on every surface | pass the surface the client actually has, match the facade to the client (`AsyncSystemOneClient` for `AsyncOpenAI`), or fix the client |
| `LabelReadoutError` | the first answer token was not a label, or the provider returned no logprobs / no alternatives / no logprob for the answer token / no rival *among the options* — or a value no log probability can be (a positive one), or a sampled token that contradicts the answer text beside it, which is one generation's two views disagreeing | if the message blames the provider, stop pinning `logprobs` — `auto` already answers with `structured`; if it names the token, the model misbehaved and one corrective retry was already spent |
| `MalformedAnswerError` | the JSON answer had missing/extra keys (an extra root key beside the field asked for is malformed, and the message names keys, not values), a non-finite or out-of-range number, an unknown label, a level index that is not an exact decimal, or **more than one decodable object** — or there was no JSON at all, including JSON nested too deeply to parse. The message says so when the response carried reasoning only | usually transient model behaviour; when it names reasoning only, turn thinking off and bound the output with `extra_body={"max_tokens": 512}` |
| `IncompleteAnswerError` | the provider stopped generating before the answer was complete: a spent output budget (`length` / `max_output_tokens` / `max_tokens`), a context window too small (`model_context_window_exceeded`), a Responses output item marked `incomplete` under a response that claims completion, or a stop reason that is not a finished one. A `ProviderError` subclass, raised before any readout, so no corrective retry is spent | raise the cap *that surface* names — `extra_body={"max_output_tokens": ...}` on Responses, `{"max_completion_tokens": ...}` on OpenAI Chat (`{"max_tokens": ...}` is what local servers take), `{"max_tokens": ...}` on Messages — and turn thinking off for a spent budget; a spent context window wants a shorter state or fewer examples |
| `ModelRefusalError` | the model declined to answer, or a safety filter withheld it, and the provider said so: a `refusal` beside a null `content` (Chat), a `refusal` content part (Responses), `stop_reason: "refusal"` (Messages), `content_filter` on any surface. Also a `ProviderError` subclass, raised before any readout | not a parsing bug and not worth a retry — a filter is filtered again the same way. Change the request or the model |
| `ProviderError` | the provider call failed — a transient one after its retries are exhausted, or a non-transient one at once; `.attempts` holds the history, `.status_code` the status, and `.embedded` says it came in the body of a `200` rather than the status line — OpenRouter's upstream-failure shape, whose error outranks any answer beside it. Also a Responses `status` of `failed`/`cancelled`, an output item left `in_progress`, an event stream in answer to a non-streaming request (with the provider's own failure and status read out of it when it carried one), and the case where every surface `api="auto"` could try answered `404` | read `exc.attempts[-1]["error"]`; a 401/403 is credentials, a 404 is the model id or the endpoint path, and an embedded one is the provider's own body |
| `JevperError` | base class; also constructor misuse (unknown `method`/`api`, a count option that is not an integer — `top_logprobs`, `max_concurrency`, `n_retry_malformed` — `top_logprobs` outside `[0, 20]` or below 2 with a pinned label method, a `model` that is not a non-blank string, a blank or over-long `prompt_cache_key`, an `extra_headers` name that is not an HTTP token or a value that is not printable ASCII), a per-call `api=""`/`method=""`/`model=""`, an `extra_body` carrying `model` or a truthy `stream` or a structure that refers to itself, a bad `state` message, non-JSON-serializable or non-finite content, a string that cannot be encoded as UTF-8 anywhere that reaches the wire, and content nested deeper than the interpreter's JSON encoder can write — a named error rather than an escaping `RecursionError` | fix the input — zero requests were sent; where the limit falls is a property of the runtime, so flatten an over-deep state or pass it as text rather than assuming your data is the problem |

Recoverable errors (`LabelReadoutError`, `MalformedAnswerError`) get one corrective retry by default — the
client appends the reason as a correction turn (`Your previous reply was invalid: {reason}. …`) and re-asks,
counting towards `usage.n_calls`. Provider-side logprob failures are *not* corrected, because another turn
cannot make a provider report logprobs it does not have, and neither are `IncompleteAnswerError` or
`ModelRefusalError`: a cut-off or declined generation is reported as the provider's failure before any
readout is attempted, with the attempt history the other provider errors carry.

Not every error is jevper's: constructing the client can fail before any jevper code runs. `anthropic` 1.8
is built on `httpx2` and refuses an `httpx.Client` you pass it — let the SDK build its own client, or hand it
an `httpx2.Client()`. That is a `TypeError`, not a `JevperError`, so a boundary catching only `JevperError`
lets it through.

Transient failures are retried per call: HTTP `408`, `409`, `429` and any `5xx`, plus connection, timeout and
`httpx` transport errors matched by class name, backing off at `min(base_delay · 3ⁿ, max_delay)`. An
`x-should-retry` header outranks the status: `false` suppresses even a `503`, `true` repeats a `400`. A
rate-limited provider's `Retry-After` (delta-seconds or an HTTP date) or `retry-after-ms` (milliseconds)
replaces that backoff and is waited out in full — coming back sooner is another request the server will
refuse — where `max_delay` does not cap it and a 24-hour ceiling does; an unreadable value falls back to
the curve, and `RetryPolicy(respect_retry_after=False)` keeps the curve alone. An official SDK's own retry
loop is disabled on the copy jevper makes, so `usage.n_retries` counts every retry that happened.
`ProviderError` propagates only after every question has settled, in question insertion order.

## Inspecting a failed call

`response.debug` is always populated; `ProviderError.attempts` carries the same records when the call never
produced a response.

| Key | What it tells you |
| --- | --- |
| `method` | the effective method: what `auto` resolved to, or what you pinned |
| `methods` | `{question_id: method}` — present only under `auto`, since it chooses per question |
| `api` | the surface actually used: `chat_completions`, `responses` or `messages` |
| `server_limits` | a six-field snapshot of the limits remembered for the *(model, surface)* the **answer** came from: `{"structured": "schema"|"object"|"none", "reasoning": bool, "include": bool, "cache_key": bool, "output_config": bool, "thinking": bool}` — absent when that surface refused nothing, and only non-default values are refusals. A refusal learned for one model is not carried to the next, and a refusal on a surface jevper later left is in that question's `llm_attempts`, not in `retry_reasons` |
| `apis` | `{question_id: surface}` — present only when one call used more than one surface, which the scalar `api` (the final shared context) cannot show under concurrency |
| `server_limits_by_api` | the same six flags per surface, for every surface attempted, when a call used more than one |
| `reasoning_mode` | `off`, `native` or `two_step`; `reasoning_modes` is the per-question version, present when one call used more than one surface |
| `llm_attempts` | one record per provider call: `question_id`, `surface`, `request` (the exact kwargs sent, or about to be sent), `response`, `error` (`"Type: message"`), `readout` |
| `readout.source` | which readout produced the answer: `logprobs`, `grammar`, `structured` or `discrete` |
| `readout.probabilities` / `missing_labels` | the parsed distribution, and labels the provider reported no logprob for (those get probability `0.0`) |
| `readout.observed_text` | the exact text the readout read — the first place to look when a label came back wrong |
| `retry_reasons` | why corrective retries happened, in order — including `… retrying the label readout on api='chat_completions'` after a surface move, `… answering on api='chat_completions'` after a missing route, and `… answering with method='structured'` after a fallback |
| `probability_errors` / `original_probabilities` | `{question_id: abs(sum − 1)}` for `structured` answers that missed 1 by more than `1e-6`, and the model's raw numbers for those questions |

`method`, `api` and `server_limits` describe the final shared context, so with questions running
concurrently they are the last word rather than the whole story: each `llm_attempts[i]` carries its own
`surface`, `request` and `readout`.

`usage.n_calls` counts the provider results jevper could read, so it is larger than your question count
whenever something else happened: a two-step analysis pass (+1 per question), a corrective retry (+1), an
`auto` discovery call that came back unreadable (+1), or a re-ask on another surface that answered (+1). A
call the provider rejected — with an exception, or with a `200` body jevper's normalizer turns away — is an
attempt but not a call: use `len(response.debug["llm_attempts"])` for every attempt.
`n_retries` counts transient-failure retries only. A token count
is `None` when any constituent call omitted it — a reported `0` is preserved.

## Symptom → fix

| Symptom | Fix |
| --- | --- |
| `400 … logprobs are not supported with reasoning models.` after switching models | drop `method="logprobs"`; `auto` (the default) answers with `structured` on that model and keeps logprobs where they work |
| `400 Unknown name "logprobs"` (Gemini's OpenAI-compatibility endpoint) | same — the endpoint never had logprobs; `auto` falls back |
| `LabelReadoutError: no logprobs returned …` | the provider ignored the fields; check the model id and endpoint, then let `auto` fall back |
| `LabelReadoutError: … no alternatives …` | `top_logprobs` is `0`, or the provider reports only the sampled token: raise it, or accept `structured`. A question with a single option is the exception — the sampled token is the whole answer. Alternatives that are not options (or carry no logprob) are the same case, and so is a positive "logprob": the provider sent something that is not a distribution, and jevper refuses it rather than normalizing it into certainty |
| `400` naming `response_format`, `json_schema`, `text.format`, `reasoning_effort`, `include`, `prompt_cache_key`, `output_config` or `thinking` | absorbed automatically — the field is dropped, the call re-asked, and `debug["server_limits"]` records what the answering surface refused. Nothing to fix; on the Messages route the schema stays in the system prompt, so a server that drops `output_config` still answers |
| `400 budget_tokens: must be at least 1024`, or another complaint about the *number* in a field the server knows | not absorbed, on purpose: the provider's own error travels back rather than the call being re-asked with your reasoning silently switched off. Send a value that server accepts (Anthropic's floor is 1024, and the budget must stay below `max_tokens`) |
| `JevperError: prompt_cache_key must be …` | the key is checked before any request: pass a non-blank string of at most 256 characters |
| `UnsupportedMethodError` on an Anthropic-compatible client | that API has no logprobs: use `structured`/`discrete`, or an OpenAI-compatible client for a label readout. `auto` already answers in JSON there |
| `ClientCapabilityError: client has no messages.create …` | you passed `api="messages"` with an OpenAI client: pass an `anthropic.Anthropic` (or another client exposing `messages.create`), or use one of the OpenAI surfaces |
| `ProviderError` with `status_code=404` and a message about a missing route | the server has no such route and the client has no other surface to try, so the 404 is the answer on every call: check the `base_url` and port, or use a client that speaks the other surface. A 404 whose `code` is `model_not_found` (or `unknown_model`, `invalid_model`, …) is the *model*, reported as it stands rather than rotated away from |
| `LabelReadoutError: the provider rejected the logprob request …` under a pinned `method="logprobs"` | the surface that refused has no alternative to move to, and the readout is never swapped for `structured`: drop the pin, or name a surface that carries the distribution (`api="chat_completions"`). `auto` does both by itself |
| `MalformedAnswerError` mentioning reasoning only | a reasoning parser put the whole generation in a thinking block and returned no answer text: turn thinking off ([providers.md](providers.md)) — on SGLang also drop `--reasoning-parser` when serving a model that never emits the closing marker |
| `MalformedAnswerError` on a provider without strict schema support | `structured_outputs=False` sends `{"type": "json_object"}` instead and the schema travels in the system prompt, so the answer's shape is only as good as instruction-following — keep descriptions unambiguous, set `temperature=0.0`, and raise `n_retry_malformed` |
| `IncompleteAnswerError` naming the output budget | a reasoning model spent the cap thinking: turn thinking off ([providers.md](providers.md)) and raise `extra_body={"max_tokens": ...}`. The error is terminal for that call — no corrective retry is spent, because the same request with the same budget is cut short the same way |
| `IncompleteAnswerError` naming the context window | the request is longer than the model can answer, so raising `max_tokens` makes it worse: shorten the state or the examples, or use a model with a larger context |
| `ModelRefusalError` | the model declined and the provider said so — not a parsing bug, and not worth a retry: each surface puts the refusal somewhere of its own (Chat a `refusal` beside a null `content`, Responses a `refusal` content part, Messages `stop_reason: "refusal"`), and the model's own words travel in the message where the surface has them. Change the request or the model |
| `ProviderError: the provider reported status='failed' …` | a Responses generation the provider did not finish and did not complete: it is raised before any readout, so a failed body that still carried text is never reported as an answer. Read `.attempts`; retrying the same request is the only thing that can help |
| `LabelReadoutError: … contradicts the answer text …` | the sampled token says one label and the text beside it another: neither view is trusted, so the answer is refused. One corrective retry is spent; a provider that does this on every attempt is reporting logprobs from a different generation than its text — check for a proxy or cache in front of it |
| `MalformedAnswerError: more than one JSON object` | the answer carried two decodable objects and jevper will not guess which is the answer. A stray brace in prose is tolerated, and a later valid object is found — only two real objects are refused |
| `ProviderError: the provider left the answer item in status='in_progress'` | OpenResponses gives every output item its own lifecycle: the response claimed completion while the message was still being written. The body is a provider failure, not an empty answer; retrying is the only thing that helps |
| `JevperError: … contains a character that cannot be encoded as UTF-8` | an unpaired surrogate reached the state, a question, the model id, `prompt_cache_key` or `extra_body`: the field is named, and nothing was sent. Replace the character |
| `ProviderError: the provider answered the … request with a streaming event stream` | the server streamed where a whole response was asked for. If the stream carried the provider's own failure (`event: error`, `response.failed`, a bare `{"error": …}`) that failure is what you get, with its status, and it is retried when transient; this message means the stream carried none. It is a `ProviderError` on all three surfaces now — in 0.7.0 a Chat stream surfaced as `ClientCapabilityError` and a Messages one as `MalformedAnswerError`. Either way, no answer is read from a stream: point jevper at a non-streaming route |
| `MalformedAnswerError: the structured answer must be an object with exactly …` | the body carried the field it was asked for *and* something beside it. 0.7.0 ignored extra root keys; now they are malformed, and the message names the keys it saw rather than their values, so a model that pads its JSON fails instead of passing. A `discrete` level may be `"2"`, `"2.0"` or `"2.000"`, but a decimal fraction (`"2.0000000000000000000001"`) is malformed rather than rounded |
| `MalformedAnswerError: … nested too deeply to parse` | the answer's JSON nests past CPython's recursion limit. It is a malformed answer, not a `RecursionError`, so the corrective-retry contract holds; a model that nests that deep on every try wants a simpler instruction |
| `JevperError: extra_headers …` naming a CRLF, NUL, non-ASCII or lone surrogate | header names must be HTTP tokens and values printable ASCII, checked at construction: nothing was sent, and a name in a different case replaces the SDK's own rather than adding a second credential. Both `extra_body` and `extra_headers` are copied at construction, so editing yours afterwards changes nothing the client sends |
| `JevperError: extra_body contains a structure that refers to itself` | a mapping or list that contains itself, which no request could carry and which the encoder would walk forever. Break the cycle before handing it over |
| `KeyboardInterrupt` from a multi-question call | an interrupt is not a failed question: the questions still queued are cancelled and the interrupt is re-raised at once, while a worker thread already running cannot be cancelled and may still finish — `close()` joins it. An ordinary per-question failure instead runs every question and raises the first in question order |
| A `400` on `include[1]` naming `reasoning.encrypted_content` | only the reasoning includable is dropped; the logprob carrier in the same list stays, because that is what the server's own message says. `debug["server_limits"]["include"]` records it |
| A label readout reports the first token of the reasoning | thinking is on: turn it off. The tail anchor only saves you when the server separates the trace and the token stream ends exactly with the answer |
| Probabilities all sit on one option with `confidence: 1.0` | you are on `discrete`, or a `structured` answer was one-hot — check `debug["llm_attempts"][-1]["readout"]["source"]`/`debug["methods"]` |
| `confidence` looks too high for `structured` | that is the model's self-report; set `temperature=0.0`, add calibration examples with explicit `probabilities`, or move to `logprobs` |
| A probability above `1.0`, or numbers that do not add up to 1 | `normalize_probabilities=False` hands the model's own numbers back untouched — the gap is still recorded in `debug["probability_errors"]`, and `confidence` is computed from those numbers. That is the contract, not a bug: normalize them yourself, or leave normalization on. A *negative* or non-finite probability is a `MalformedAnswerError` on either setting |
| A `Score` above `N-1`, or probabilities that do not add up to 1 | `score` is an expected value read off the rescaled distribution, so it stays on the 0..N-1 line whatever `normalize_probabilities` is; the reported probabilities are the ones that may not add up. Normalize them yourself, or leave normalization on |
| A one-option `Choice` under `logprobs`/`grammar` | that is answerable now, and the answer is the single label at probability 1.0 — there is no rival to read. Use a `Noul` if the question is really yes/no |
| Costs doubled unexpectedly | `reasoning` is on with `mode="two_step"` (two calls per question) — `usage.n_calls` shows it |
| A second call about the same rubric reports `cached_tokens` `0` or `None` | `None` means the server does not report it (vLLM and SGLang need a flag); `0` means the shared prefix was not reused — the state must stay last, and OpenAI only caches a prefix of 1024 tokens or more. A chat-list state whose last turn is the assistant's moves the question last by design (no server answers a conversation ending on an assistant turn), which costs that prefix: end the state on a user turn to keep it |
| Wide `Choice` (>26 options) raises `InvalidQuestionError` | use `auto`, `structured` or `discrete`; `logprobs` and `grammar` read one label token |

One `except JevperError` covers every error jevper raises, but not construction errors from the SDK itself:
with **both** `method="logprobs"` and `api` pinned there is no `auto` path left to interpret a provider's
refusal, so it reaches you as a `ProviderError` — the `LabelReadoutError` above is the `auto` reading.
