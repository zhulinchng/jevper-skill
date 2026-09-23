# jevper troubleshooting

- [Does my provider do logprobs?](#does-my-provider-do-logprobs) — the support matrix, and the one-line probe
- [Error triage](#error-triage) — every error class, its cause, the first move
- [Inspecting a failed call](#inspecting-a-failed-call) — `debug` keys and what they mean
- [Symptom → fix](#symptom--fix) — the recurring ones

## Does my provider do logprobs?

| Provider | `logprobs` | Evidence |
| --- | --- | --- |
| OpenAI `gpt-4o`, `gpt-4.1`, older chat models | yes | |
| OpenAI reasoning models (`o`-series, `gpt-5` family) | no | `400 logprobs are not supported with reasoning models.` |
| OpenAI Responses surface | partial | `include=["message.output_text.logprobs"]` alone returns the sampled token and **no alternatives**; some models fail outright on `top_logprobs >= 2` |
| Anthropic Claude | no | no logprob API at all |
| Gemini via the OpenAI-compatibility endpoint | no | `400 Unknown name "logprobs": Cannot find field.` |
| Gemini native API | yes | not reachable through an OpenAI-compatible client |
| DeepSeek | yes | `top_logprobs` up to 20 |
| Together | yes | `logprobs: 1` alone returns the sampled token; send `top_logprobs` for alternatives |
| Ollama, llama.cpp, vLLM | yes | |
| everything else | unknown | reasoning models and thin compatibility layers are the ones that say no |

`auto` reads this situation for you: it asks for logprobs, and on a 4xx that names the logprob fields, a
response with no logprobs, or an answer token with no alternatives, it re-asks with `structured` and
remembers the verdict for that `(model, surface)` for the rest of the client's life.

Probe it before writing code, or after a suspicious switch of model:

```sh
python scripts/offline_stub.py --live --model <model-id>       # prints the resolved method and readout source
```

## Error triage

| Error | Cause | First move |
| --- | --- | --- |
| `InvalidQuestionError` | question or example is locally invalid: unknown fields, `Choice` outside 2–255 options, `Score` outside 2–10 levels, `Noul` criteria keys other than `true`/`false`, more than 26 options under `logprobs`/`grammar`, an example answer that matches no option | fix the question — zero requests were sent |
| `UnsupportedMethodError` | `method="grammar"` with a Responses surface | pass `api="chat_completions"`, or drop `grammar` |
| `ClientCapabilityError` | the client object lacks `responses.create` / `chat.completions.create`, or the provider returned no choices | pass the surface the client actually has, or fix the client |
| `LabelReadoutError` | the first answer token was not a label, or the provider returned no logprobs / no alternatives / no logprob for the answer token | if the message blames the provider, stop pinning `logprobs` — `auto` already answers with `structured`; otherwise the model misbehaved and one corrective retry was already spent |
| `MalformedAnswerError` | the JSON answer had missing/extra keys, a non-finite or out-of-range number, an unknown label, or a non-integer score | usually transient model behaviour; check the schema is in the prompt (`structured_outputs=False` on a provider without strict schemas) |
| `ProviderError` | the provider call failed after transient retries; `.attempts` holds the history | read `exc.attempts[-1]["error"]`; a 401/403 is credentials, a 404 is the model id or the endpoint path |
| `JevperError` | base class; also constructor misuse, a bad `state` message, non-JSON-serializable or non-finite content | fix the input |

Recoverable errors (`LabelReadoutError`, `MalformedAnswerError`) get one corrective retry by default — the
client appends the reason as a correction turn (`Your previous reply was invalid: {reason}. …`) and re-asks,
counting towards `usage.n_calls`. Provider-side logprob failures are *not* corrected, because another turn
cannot make a provider report logprobs it does not have.

Transient failures are retried per call: HTTP `429`, `500`, `502`, `503`, `504`, `529`, plus connection,
timeout and `httpx` transport errors. `ProviderError` propagates only after every question has settled, in
question insertion order.

## Inspecting a failed call

`response.debug` is always populated; `ProviderError.attempts` carries the same records when the call never
produced a response.

| Key | What it tells you |
| --- | --- |
| `method` | the effective method: what `auto` resolved to, or what you pinned |
| `methods` | `{question_id: method}` — present only under `auto`, since it chooses per question |
| `api` | the surface actually used |
| `reasoning_mode` | `off`, `native` or `two_step` |
| `llm_attempts` | one record per provider call: `question_id`, `surface`, `request` (the exact kwargs sent, or about to be sent), `response`, `error` (`"Type: message"`), `readout` |
| `readout.source` | which readout produced the answer: `logprobs`, `structured` or `discrete` |
| `readout.probabilities` / `missing_labels` | the parsed distribution, and labels the provider reported no logprob for (those get probability `0.0`) |
| `retry_reasons` | why corrective retries happened, in order — including `… answering with method='structured'` after an `auto` fallback |
| `probability_errors` / `original_probabilities` | `{question_id: abs(sum − 1)}` for `structured` answers that missed 1 by more than `1e-6`, and the model's raw numbers for those questions |

`usage.n_calls` counts the provider calls that returned a response, so it is larger than your question count
whenever something else happened: a two-step analysis pass (+1 per question), a corrective retry (+1), or an
`auto` discovery call that came back unreadable (+1). A call the provider *rejected* with an exception is not
counted — use `len(response.debug["llm_attempts"])` for every attempt, including rejected ones. `n_retries`
counts transient-failure retries only. A token count is `None` when any constituent call omitted it — a
reported `0` is preserved.

## Symptom → fix

| Symptom | Fix |
| --- | --- |
| `400 … logprobs are not supported with reasoning models.` after switching models | drop `method="logprobs"`; `auto` (the default) answers with `structured` on that model and keeps logprobs where they work |
| `400 Unknown name "logprobs"` (Gemini's OpenAI-compatibility endpoint) | same — the endpoint never had logprobs; `auto` falls back |
| `LabelReadoutError: no logprobs returned …` | the provider ignored the fields; check the model id and endpoint, then let `auto` fall back |
| `LabelReadoutError: … no alternatives …` | `top_logprobs` is `0`, or the provider reports only the sampled token: raise it, or accept `structured` |
| Probabilities all sit on one option with `confidence: 1.0` | you are on `discrete`, or a `structured` answer was one-hot — check `debug["readout"]["source"]`/`debug["methods"]` |
| `confidence` looks too high for `structured` | that is the model's self-report; set `temperature=0.0`, add calibration examples with explicit `probabilities`, or move to `logprobs` |
| `MalformedAnswerError` on a provider without strict schema support | `structured_outputs=False`, and/or raise `n_retry_malformed` |
| Costs doubled unexpectedly | `reasoning` is on with `mode="two_step"` (two calls per question) — `usage.n_calls` shows it |
| Wide `Choice` (>26 options) raises `InvalidQuestionError` | use `auto`, `structured` or `discrete`; `logprobs` and `grammar` read one label token |
