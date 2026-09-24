# jevper troubleshooting

- [Auto and the provider](#auto-and-the-provider) — what `auto` resolves on its own, and the probe
- [Error triage](#error-triage) — every error class, its cause, the first move
- [Inspecting a failed call](#inspecting-a-failed-call) — `debug` keys and what they mean
- [Symptom → fix](#symptom--fix) — the recurring ones

## Auto and the provider

`auto` reads the provider for you: it asks for logprobs, and on a 4xx that names the logprob fields, an
answer with no logprobs, or an answer token with no alternatives, it re-asks with `structured` and
remembers the verdict per `(model, surface)` for the rest of the client's life. Before that fallback it
tries the *other surface* when the client exposes one and the reasoning mode is not `native` — a pinned
`method="logprobs"` takes that move too, keeping its method — and a 404 that does not name the model is
read as a missing route, answered on the other surface where the client has one. A rejection that names a
*value* rather than a field (`top_logprobs must be between 0 and 20`) and a 5xx that survives its retries
both fall back for that question alone, without being remembered. The same distinction
holds on the capability ladder: only a complaint about a field's *existence* drops it — a server that
refuses the number in a field it knows (`budget_tokens: must be at least 1024`) keeps its own error, because
answering with your reasoning silently switched off would be worse than failing.

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
| `InvalidQuestionError` | question or example is locally invalid: unknown fields, `Choice` outside 1–255 options, `Score` outside 2–10 levels, `Noul` criteria keys other than `true`/`false`, more than 26 options under `logprobs`/`grammar`, an example answer that matches no option | fix the question — zero requests were sent |
| `UnsupportedMethodError` | `method="grammar"` with a Responses surface, or `method="logprobs"`/`"grammar"` with the Messages surface — that API has no logprobs at all | pass `api="chat_completions"` (or drop `grammar`), or use `structured`/`discrete` |
| `ClientCapabilityError` | the client object lacks `responses.create` / `chat.completions.create` / `messages.create`, or the response carried no usable first choice (missing or empty) and no explanation of why | pass the surface the client actually has, or fix the client |
| `LabelReadoutError` | the first answer token was not a label, or the provider returned no logprobs / no alternatives / no logprob for the answer token | if the message blames the provider, stop pinning `logprobs` — `auto` already answers with `structured`; if it names the token, the model misbehaved and one corrective retry was already spent |
| `MalformedAnswerError` | the JSON answer had missing/extra keys, a non-finite or out-of-range number, an unknown label, or a non-integer score — or there was no JSON at all. The message says so when the response carried reasoning only | usually transient model behaviour; when it names reasoning only, turn thinking off and bound the output with `extra_body={"max_tokens": 512}` |
| `IncompleteAnswerError` | the provider stopped generating before the answer was complete: a spent output budget (`length` / `max_output_tokens` / `max_tokens`), a context window too small (`model_context_window_exceeded`), or a stop reason that is not a finished one. A `ProviderError` subclass, raised before any readout, so no corrective retry is spent | raise `extra_body={"max_tokens": ...}` and turn thinking off for a spent budget; a spent context window wants a shorter state or fewer examples — the message names which |
| `ModelRefusalError` | the model declined to answer and the provider said so: a `refusal` beside a null `content` (Chat), a `refusal` content part (Responses), `stop_reason: "refusal"` (Messages). Also a `ProviderError` subclass, raised before any readout | not a parsing bug and not worth a retry — change the request or the model |
| `ProviderError` | the provider call failed after transient retries; `.attempts` holds the history and `.status_code` the status — including a status carried inside a nominal `200` body, which is how OpenRouter reports an upstream failure, and a Responses `status` of `failed`/`cancelled`. Also raised when every surface `api="auto"` could try answered `404`: a missing route is the provider's failure, not a verdict jevper keeps | read `exc.attempts[-1]["error"]`; a 401/403 is credentials, a 404 is the model id or the endpoint path |
| `JevperError` | base class; also constructor misuse (unknown `method`/`api`, a count option that is not an integer — `top_logprobs`, `max_concurrency`, `n_retry_malformed` — `top_logprobs` outside `[0, 20]` or below 2 with a pinned label method, a `model` that is not a non-blank string, a blank or over-long `prompt_cache_key`), a per-call `api=""`/`method=""`/`model=""`, a bad `state` message, non-JSON-serializable or non-finite content | fix the input — zero requests were sent |

Recoverable errors (`LabelReadoutError`, `MalformedAnswerError`) get one corrective retry by default — the
client appends the reason as a correction turn (`Your previous reply was invalid: {reason}. …`) and re-asks,
counting towards `usage.n_calls`. Provider-side logprob failures are *not* corrected, because another turn
cannot make a provider report logprobs it does not have, and neither are `IncompleteAnswerError` or
`ModelRefusalError`: a cut-off or declined generation is reported as the provider's failure before any
readout is attempted, with the attempt history the other provider errors carry.

Transient failures are retried per call: HTTP `408`, `429`, `500`, `502`, `503`, `504`, `529`, plus
connection, timeout and `httpx` transport errors, backing off at `min(base_delay · 3ⁿ, max_delay)`. A
rate-limited provider's `Retry-After` or `retry-after-ms` header — delta-seconds or an HTTP date, in any
case — replaces that backoff and is waited out in full, since coming back sooner is another request the
server will refuse; `RetryPolicy(respect_retry_after=False)` keeps the curve alone.
`ProviderError` propagates only after every question has settled, in question insertion order.

## Inspecting a failed call

`response.debug` is always populated; `ProviderError.attempts` carries the same records when the call never
produced a response.

| Key | What it tells you |
| --- | --- |
| `method` | the effective method: what `auto` resolved to, or what you pinned |
| `methods` | `{question_id: method}` — present only under `auto`, since it chooses per question |
| `api` | the surface actually used: `chat_completions`, `responses` or `messages` |
| `server_limits` | the fields the surface the **answer** came from refused: `{"structured": "schema"|"object"|"none", "reasoning": bool, "include": bool, "cache_key": bool, "output_config": bool, "thinking": bool}` — the call was re-asked without them. A refusal on a surface jevper later left is in `retry_reasons`, not here |
| `reasoning_mode` | `off`, `native` or `two_step` |
| `llm_attempts` | one record per provider call: `question_id`, `surface`, `request` (the exact kwargs sent, or about to be sent), `response`, `error` (`"Type: message"`), `readout` |
| `readout.source` | which readout produced the answer: `logprobs`, `structured` or `discrete` |
| `readout.probabilities` / `missing_labels` | the parsed distribution, and labels the provider reported no logprob for (those get probability `0.0`) |
| `readout.observed_text` | the exact text the readout read — the first place to look when a label came back wrong |
| `retry_reasons` | why corrective retries happened, in order — including `… retrying the label readout on api='chat_completions'` after a surface move, `… answering on api='chat_completions'` after a missing route, and `… answering with method='structured'` after a fallback |
| `probability_errors` / `original_probabilities` | `{question_id: abs(sum − 1)}` for `structured` answers that missed 1 by more than `1e-6`, and the model's raw numbers for those questions |

`method`, `api` and `server_limits` describe the final shared context, so with questions running
concurrently they are the last word rather than the whole story: each `llm_attempts[i]` carries its own
`surface`, `request` and `readout`.

`usage.n_calls` counts the provider calls that returned a response, so it is larger than your question count
whenever something else happened: a two-step analysis pass (+1 per question), a corrective retry (+1), an
`auto` discovery call that came back unreadable (+1), or a re-ask on another surface that answered (+1). A
request the provider *rejected* with an exception is not counted — use `len(response.debug["llm_attempts"])`
for every attempt, including rejected ones. `n_retries` counts transient-failure retries only. A token count
is `None` when any constituent call omitted it — a reported `0` is preserved.

## Symptom → fix

| Symptom | Fix |
| --- | --- |
| `400 … logprobs are not supported with reasoning models.` after switching models | drop `method="logprobs"`; `auto` (the default) answers with `structured` on that model and keeps logprobs where they work |
| `400 Unknown name "logprobs"` (Gemini's OpenAI-compatibility endpoint) | same — the endpoint never had logprobs; `auto` falls back |
| `LabelReadoutError: no logprobs returned …` | the provider ignored the fields; check the model id and endpoint, then let `auto` fall back |
| `LabelReadoutError: … no alternatives …` | `top_logprobs` is `0`, or the provider reports only the sampled token: raise it, or accept `structured` |
| `400` naming `response_format`, `json_schema`, `text.format`, `reasoning_effort`, `include`, `prompt_cache_key`, `output_config` or `thinking` | absorbed automatically — the field is dropped, the call re-asked, and `debug["server_limits"]` records what the answering surface refused. Nothing to fix; on the Messages route the schema stays in the system prompt, so a server that drops `output_config` still answers |
| `400 budget_tokens: must be at least 1024`, or another complaint about the *number* in a field the server knows | not absorbed, on purpose: the provider's own error travels back rather than the call being re-asked with your reasoning silently switched off. Send a value that server accepts (Anthropic's floor is 1024, and the budget must stay below `max_tokens`) |
| `JevperError: prompt_cache_key must be …` | the key is checked before any request: pass a non-blank string of at most 256 characters |
| `UnsupportedMethodError` on an Anthropic-compatible client | that API has no logprobs: use `structured`/`discrete`, or an OpenAI-compatible client for a label readout. `auto` already answers in JSON there |
| `ClientCapabilityError: client has no messages.create …` | you passed `api="messages"` with an OpenAI client: pass an `anthropic.Anthropic` (or another client exposing `messages.create`), or use one of the OpenAI surfaces |
| `ProviderError` with `status_code=404` and a message about a missing route | the server has no such route and the client has no other surface to try, so the 404 is the answer on every call: check the `base_url` and port, or use a client that speaks the other surface |
| `LabelReadoutError: the provider rejected the logprob request …` under a pinned `method="logprobs"` | the surface that refused has no alternative to move to, and the readout is never swapped for `structured`: drop the pin, or name a surface that carries the distribution (`api="chat_completions"`). `auto` does both by itself |
| `MalformedAnswerError` mentioning reasoning only | a reasoning parser put the whole generation in a thinking block and returned no answer text: turn thinking off ([providers.md](providers.md)) — on SGLang also drop `--reasoning-parser` when serving a model that never emits the closing marker |
| `MalformedAnswerError` on a provider without strict schema support | `structured_outputs=False` sends `{"type": "json_object"}` instead and the schema travels in the system prompt, so the answer's shape is only as good as instruction-following — keep descriptions unambiguous, set `temperature=0.0`, and raise `n_retry_malformed` |
| `IncompleteAnswerError` naming the output budget | a reasoning model spent the cap thinking: turn thinking off ([providers.md](providers.md)) and raise `extra_body={"max_tokens": ...}`. The error is terminal for that call — no corrective retry is spent, because the same request with the same budget is cut short the same way |
| `IncompleteAnswerError` naming the context window | the request is longer than the model can answer, so raising `max_tokens` makes it worse: shorten the state or the examples, or use a model with a larger context |
| `ModelRefusalError` | the model declined and the provider said so — not a parsing bug, and not worth a retry: each surface puts the refusal somewhere of its own (Chat a `refusal` beside a null `content`, Responses a `refusal` content part, Messages `stop_reason: "refusal"`), and the model's own words travel in the message where the surface has them. Change the request or the model |
| `ProviderError: the provider reported status='failed' …` | a Responses generation the provider did not finish and did not complete: it is raised before any readout, so a failed body that still carried text is never reported as an answer. Read `.attempts`; retrying the same request is the only thing that can help |
| A label readout reports the first token of the reasoning | thinking is on: turn it off. The tail anchor only saves you when the server separates the trace and the token stream ends exactly with the answer |
| Probabilities all sit on one option with `confidence: 1.0` | you are on `discrete`, or a `structured` answer was one-hot — check `debug["readout"]["source"]`/`debug["methods"]` |
| `confidence` looks too high for `structured` | that is the model's self-report; set `temperature=0.0`, add calibration examples with explicit `probabilities`, or move to `logprobs` |
| A probability above `1.0`, or numbers that do not add up to 1 | `normalize_probabilities=False` hands the model's own numbers back untouched — the gap is still recorded in `debug["probability_errors"]`, and `confidence` is computed from those numbers. That is the contract, not a bug: normalize them yourself, or leave normalization on. A *negative* or non-finite probability is a `MalformedAnswerError` on either setting |
| A `Score` above `N-1`, or probabilities that do not add up to 1 | `score` is an expected value read off the rescaled distribution, so it stays on the 0..N-1 line whatever `normalize_probabilities` is; the reported probabilities are the ones that may not add up. Normalize them yourself, or leave normalization on |
| A one-option `Choice` under `logprobs`/`grammar` | that is answerable now, and the answer is the single label at probability 1.0 — there is no rival to read. Use a `Noul` if the question is really yes/no |
| Costs doubled unexpectedly | `reasoning` is on with `mode="two_step"` (two calls per question) — `usage.n_calls` shows it |
| A second call about the same rubric reports `cached_tokens` `0` or `None` | `None` means the server does not report it (vLLM and SGLang need a flag); `0` means the shared prefix was not reused — the state must stay last, and a hosted API only caches a prefix of 1024 tokens or more. A chat-list state whose last turn is the assistant's moves the question last by design (no server answers a conversation ending on an assistant turn), which costs that prefix: end the state on a user turn to keep it |
| Wide `Choice` (>26 options) raises `InvalidQuestionError` | use `auto`, `structured` or `discrete`; `logprobs` and `grammar` read one label token |
