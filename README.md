# jevper

Agent skill for writing and debugging Python code that calls [`jevper`](https://github.com/zhulinchng/jevper) — the Jev (System One) interface that turns a state plus `Noul`/`Choice`/`Score` questions into typed answers with probabilities and confidence, over any OpenAI-compatible model.

## Install

```bash
npx skills add zhulinchng/jevper-skill
```

Single-skill repo: root `SKILL.md` with valid `name` + `description` frontmatter, so the installer picks it up directly (`--list` shows it without installing; `-a '*'` installs to every detected agent).

Written and verified against **jevper 0.7.4** (Python 3.10+, `pydantic>=2.7`). Every behaviour it describes — methods, fallbacks, the server-limits ladder, prompt caching, the three surfaces, error messages — was checked against that release, on five local servers; if you are on a newer one, its `docs/` is the authority.

## What it covers

- **Questions**: `Noul` (one probability of true), `Choice` (1–255 labelled options, a one-option question answered by every method), `Score` (2–10 ordered levels, read as an expected value) — criteria rendered as prompts, answers carrying `choice`/`probabilities`/`confidence`, `score`/`legend`, `noul`, and how each `confidence` is computed
- **Methods**: leave `method` unset and `auto` answers with `logprobs` where the provider returns them, `structured` where it does not, remembered per `(model, surface)`; when to pin `grammar`, `discrete`, `structured` or `logprobs` instead — and the 26-option ceiling on the two methods that read a label *token*
- **Surfaces**: Chat Completions and Responses from an OpenAI-compatible client, plus the Anthropic **Messages** API (`api="messages"`, an `anthropic.Anthropic` client) — where no logprobs exist at all, the JSON Schema rides in Anthropic's `output_config` field *and* in the system prompt, `max_tokens` is required (jevper sends `1024`, or `1024` plus the thinking budget) and thinking is a `budget_tokens` field
- **The Responses route, both dialects**: `/v1/responses` is OpenAI's Responses API *and* the [OpenResponses](https://www.openresponses.org) spec (LM Studio is a listed implementer, vLLM says its route aligns; llama.cpp, SGLang answer the measured shapes, ollama accepts them too). jevper sends the portable typed-item form and reads either back — text and reasoning part names, per-item `status`, `phase`-labelled messages, logprob `bytes` — with the two shapes that are failures rather than answers called out
- **Fallbacks**: the surface move when a route is missing (404) or answers without a distribution — which a pinned `method="logprobs"` also takes, and which a `grammar` request never does — a 404 whose `code` says the *model* is absent rather than the route, the capability-versus-bad-value distinction that decides what gets remembered *and* what may be dropped, the finite ladder that drops a refused schema (`json_schema` → `json_object` → nothing), reasoning parameters, the Responses `include` list, `prompt_cache_key` and the Messages `output_config` and `thinking` fields, each remembered per *(model, surface)* rather than per client
- **Provider support**: who returns logprobs (OpenAI chat models, DeepSeek, Together, llama.cpp, vLLM, SGLang, Ollama, LM Studio) and who rejects the fields (OpenAI reasoning models, Claude, Gemini's OpenAI-compatibility endpoint, OpenRouter's routing), with the exact error strings, per-surface request fields, what to pass to each local server, how to turn thinking off (ollama's Responses route wants `reasoning: {"effort": "none"}`, not `reasoning_effort`), and a one-line probe
- **Prompt caching**: the state-last message order (and the one state shape that cannot use it), the per-question derived `prompt_cache_key` — model, method, examples and question block — sent in the request *body* so an old SDK still carries it, how to override it, `usage.cached_tokens` and which server flag makes it appear, cache isolation via `cache_salt`, and the measured reuse numbers
- **Answers that never arrived**: `IncompleteAnswerError` and `ModelRefusalError` — a spent output budget, a context window too small, a model's refusal or a safety filter — raised as `ProviderError` subclasses before any readout, so a cut-off, declined or withheld generation is never read as a decision and no corrective retry is spent on one. The remedy names the cap *that surface* uses (`max_output_tokens` on Responses, `max_completion_tokens` on OpenAI Chat — `max_tokens` beside it for local servers — and `max_tokens` on Messages); an answer body must also carry *exactly* the field it was asked for
- **Reasoning**: `ReasoningConfig` — native provider reasoning vs a two-step analysis-then-answer pass, `reasoning_text()`, and the doubled call count that comes with it
- **Tracing**: MLflow SDK autolog traces calls through real OpenAI/Anthropic SDK clients, including rejected and fallback attempts; `@mlflow.trace` groups the per-SDK-call spans, and `pyfunc` or LangChain adapters can host jevper as a model
- **Few-shot examples**: `Example`, the three attachment levels and their precedence, answers rendered in the active method's format, calibration through explicit `probabilities`
- **Async**: `AsyncSystemOneClient`, same signatures, semaphore instead of thread pool
- **Client knobs**: `temperature`, `top_logprobs`, `structured_outputs`, `prompt_cache_key`, `normalize_probabilities`, `max_concurrency`, `n_retry_malformed`, `retry` (`408`/`409`/`429`/any `5xx`, `x-should-retry` honoured, `Retry-After` waited out up to a day), `api`
- **Failure triage**: which errors are local (zero requests sent — an unencodable string, a header value the HTTP layer could not carry, or an `extra_body` that refers to itself, all included), which are readout failures worth one corrective retry, which are `ProviderError` — including one carried in the body of a `200` (`ProviderError.embedded`) and one carried in an event *stream* (`event: error`, a typed `response.failed`, a bare `{"error": …}`), whose status then decides the retry; answers that ended early, were refused or filtered, or carried reasoning only — and say why; the `debug` keys that show what actually happened, per surface when a call used more than one, with credential-looking header values recorded as `<redacted>`
- **Offline testing**: a duck-typed stub client that drives the real readout path from canned bodies — no HTTP, no key, no tokens — plus a live probe that reads what a model advertises before it spends a call

## Layout

- `SKILL.md` — entry point: quick start, question/method/error tables, prompt caching, checklist
- `references/features.md` — reasoning, few-shot examples, async, surfaces, prompt caching, MLflow tracing, client knobs
- `references/providers.md` — the observed logprob matrix, per-surface request fields, local servers, the Anthropic Messages route, cache reporting
- `references/troubleshooting.md` — error triage, `debug` recipes, symptom → fix
- `scripts/offline_stub.py` — duck-typed stub client, OpenAI- or Anthropic-shaped (29 scenarios, from `logprobs` and `structured` through every refusal, route, lifecycle and retry shape above) plus its own self-test and live probe

## Verify

```bash
pip install jevper                                 # Python 3.10+, pydantic>=2.7; 0.7.4 or newer
python scripts/offline_stub.py --check             # offline self-test: no network, no API key
python scripts/offline_stub.py --live --model <id> # needs openai + OPENAI_API_KEY; OPENAI_BASE_URL for self-hosted
                                         # reports the advertised parameters first (free, no quota), then makes one system_one call
python scripts/offline_stub.py --live --model <id> --api messages   # needs anthropic; ANTHROPIC_BASE_URL for a local server
python scripts/offline_stub.py --live --model <id> --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'  # request fields, e.g. thinking off
```

`--live` prints a JSON report: what the model advertises, the resolved method per question, the surface used,
the readout source, `n_calls`, `cached_tokens`, `server_limits` and `retry_reasons`. A quota, credit or key
refusal is reported as the account answer it is — status, the provider's own words, and the next step — so a
`429` from an exhausted free-model quota never reads as a jevper failure; an answer that came back but could
not be read is reported as that, with what to do about it.

The self-test drives `StubClient` through the real jevper readout path: the logprobs softmax, the structured JSON readout, `auto`'s per-`(model, surface)` memory, the surface move when a provider refuses logprobs or a route is missing — under `auto` and under a pinned `method="logprobs"`, with a single-surface client kept on the surface it has, the server-limits ladder (`json_schema` → `json_object` → nothing, the Messages `output_config` and `thinking` fields, and the rung a refused format field skips), a refused `prompt_cache_key`, a refused *value* that travels back as the provider's error, the schema that rides in `output_config` and stays in the prompt, a caller's `extra_body` reaching a Messages request with no temperature set, the quoted untrusted state, the event-stream reader on all three surfaces (a failure frame retried by its own status, a JSON body split across `data:` lines, a stream with no failure), a `404` whose code says the model rather than the route, header validation, a self-referential body, an answer carrying an extra key, a level index that is a decimal fraction, ASCII-only label folding, a redacted credential in the record — and the provider shapes that used to read as an answer: an error in a `200`, a safety filter, an item left `in_progress`, alternatives that are not a distribution, a sampled token that contradicts its own text, two JSON objects, and the retry rules.

## Reference

The library is at [zhulinchng/jevper](https://github.com/zhulinchng/jevper) (Apache-2.0), where `docs/` holds the full reference — [`api.md`](https://github.com/zhulinchng/jevper/blob/main/docs/api.md), [`methods.md`](https://github.com/zhulinchng/jevper/blob/main/docs/methods.md), [`reasoning.md`](https://github.com/zhulinchng/jevper/blob/main/docs/reasoning.md), [`few-shot.md`](https://github.com/zhulinchng/jevper/blob/main/docs/few-shot.md), [`local-servers.md`](https://github.com/zhulinchng/jevper/blob/main/docs/local-servers.md), [`architecture.md`](https://github.com/zhulinchng/jevper/blob/main/docs/architecture.md), [`internals.md`](https://github.com/zhulinchng/jevper/blob/main/docs/internals.md), [`troubleshooting.md`](https://github.com/zhulinchng/jevper/blob/main/docs/troubleshooting.md), [`complete-example.md`](https://github.com/zhulinchng/jevper/blob/main/docs/complete-example.md), [`glossary.md`](https://github.com/zhulinchng/jevper/blob/main/docs/glossary.md), [`mlflow.md`](https://github.com/zhulinchng/jevper/blob/main/docs/mlflow.md).

`jevper` is an independent implementation of the documented System One wire format, not affiliated with, endorsed by, or supported by [TypeSafe AI](https://docs.typesafe.ai); questions about the hosted API itself belong in their docs. The reference jevper 0.7.4 matches behaviour-for-behaviour is TypeSafe's own LLM-backed drop-in, [`system-one-adapter`](https://pypi.org/project/system-one-adapter/)
