# jevper

Agent skill for writing and debugging Python code that calls [`jevper`](https://github.com/zhulinchng/jevper) — the Jev (System One) interface that turns a state plus `Noul`/`Choice`/`Score` questions into typed answers with probabilities and confidence, over any OpenAI-compatible model.

## Install

```bash
npx skills add zhulinchng/jevper-skill
```

Single-skill repo: root `SKILL.md` with valid `name` + `description` frontmatter, so the installer picks it up directly (`--list` shows it without installing; `-a '*'` installs to every detected agent).

Written and verified against **jevper 0.5.2** (Python 3.10+, `pydantic>=2.7`). Every behaviour it describes — methods, fallbacks, the server-limits ladder, prompt caching, the three surfaces, error messages — was checked against that release; if you are on a newer one, its `docs/` is the authority.

## What it covers

- **Questions**: `Noul` (one probability of true), `Choice` (2–255 labelled options), `Score` (2–10 ordered levels) — criteria rendered as prompts, answers carrying `choice`/`probabilities`/`confidence`, `score`/`legend`, `noul`, and how each `confidence` is computed
- **Methods**: leave `method` unset and `auto` answers with `logprobs` where the provider returns them, `structured` where it does not, remembered per `(model, surface)`; when to pin `grammar`, `discrete`, `structured` or `logprobs` instead — and the 26-option ceiling on the two methods that read a label *token*
- **Surfaces**: Chat Completions and Responses from an OpenAI-compatible client, plus the Anthropic **Messages** API (`api="messages"`, an `anthropic.Anthropic` client) — where no logprobs exist at all, the JSON Schema travels in the system prompt, `max_tokens` is required (jevper sends `1024`, or `1024` plus the thinking budget) and thinking is a `budget_tokens` field
- **Fallbacks**: the surface move when a route is missing (404) or answers without a distribution — which a pinned `method="logprobs"` also takes, and which a `grammar` request never does — the capability-versus-bad-value distinction that decides what gets remembered *and* what may be dropped, the finite ladder that drops a refused schema (`json_schema` → `json_object` → nothing), reasoning parameters, the Responses `include` list, `prompt_cache_key` and the Messages `thinking` field
- **Provider support**: who returns logprobs (OpenAI chat models, DeepSeek, Together, llama.cpp, vLLM, SGLang, Ollama, LM Studio) and who rejects the fields (OpenAI reasoning models, Claude, Gemini's OpenAI-compatibility endpoint, OpenRouter's routing), with the exact error strings, per-surface request fields, what to pass to each local server, how to turn thinking off, and a one-line probe
- **Prompt caching**: the state-last message order (and the one state shape that cannot use it), the per-question derived `prompt_cache_key` — model, method, examples and question block — and how to override it, `usage.cached_tokens` and which server flag makes it appear, cache isolation via `cache_salt`, and the measured reuse numbers
- **Answers that never arrived**: the output budget each surface names (`length`, `max_output_tokens`, `max_tokens`), a model's refusal and its own words, and a reasoning parser that returned no answer text — all said in the error rather than left as "malformed JSON"
- **Reasoning**: `ReasoningConfig` — native provider reasoning vs a two-step analysis-then-answer pass, `reasoning_text()`, and the doubled call count that comes with it
- **Tracing**: MLflow's own autolog covers jevper's calls — every request, including the ones it settles away from — plus hosting jevper as an MLflow model
- **Few-shot examples**: `Example`, the three attachment levels and their precedence, answers rendered in the active method's format, calibration through explicit `probabilities`
- **Async**: `AsyncSystemOneClient`, same signatures, semaphore instead of thread pool
- **Client knobs**: `temperature`, `top_logprobs`, `structured_outputs`, `prompt_cache_key`, `normalize_probabilities`, `max_concurrency`, `n_retry_malformed`, `retry`, `api`
- **Failure triage**: which errors are local (zero requests sent), which are readout failures worth one corrective retry, which are `ProviderError`; transient retry policy; answers that ended early — or carried reasoning only — and say why; the `debug` keys that show what actually happened
- **Offline testing**: a duck-typed stub client that drives the real readout path from canned bodies — no HTTP, no key, no tokens — plus a live probe that reads what a model advertises before it spends a call

## Layout

- `SKILL.md` — entry point: quick start, question/method/error tables, prompt caching, checklist
- `references/features.md` — reasoning, few-shot examples, async, surfaces, prompt caching, MLflow tracing, client knobs
- `references/providers.md` — the observed logprob matrix, per-surface request fields, local servers, the Anthropic Messages route, cache reporting
- `references/troubleshooting.md` — error triage, `debug` recipes, symptom → fix
- `scripts/offline_stub.py` — duck-typed stub client, OpenAI- or Anthropic-shaped (scenarios: `logprobs`, `structured`, `reject_logprobs`, `reject_include`, `no_alternatives`, `no_responses_route`, `no_messages_route`, `reject_schema`, `reject_format`, `reject_cache_key`, `reject_thinking`, `reject_budget_value`, `truncated`, `refusal`, `reasoning`, `reasoning_only`) plus its own self-test and live probe

## Verify

```bash
pip install jevper                                 # Python 3.10+, pydantic>=2.7; 0.5.2 or newer
python scripts/offline_stub.py --check             # offline self-test: no network, no API key
python scripts/offline_stub.py --live --model <id> # needs openai + OPENAI_API_KEY; OPENAI_BASE_URL for self-hosted
                                         # reads the model's advertised parameters first (free, no quota), then spends one call
python scripts/offline_stub.py --live --model <id> --api messages   # needs anthropic; ANTHROPIC_BASE_URL for a local server
```

`--live` prints a JSON report: what the model advertises, the resolved method per question, the surface used,
the readout source, `n_calls`, `cached_tokens`, `server_limits` and `retry_reasons`. A quota, credit or key
refusal is reported as the account answer it is — status, the provider's own words, and the next step — so a
`429` from an exhausted free-model quota never reads as a jevper failure.

The self-test drives `StubClient` through the real jevper readout path: the logprobs softmax, the structured JSON readout, `auto`'s per-`(model, surface)` memory, the surface move when a provider refuses logprobs or a route is missing — under `auto` and under a pinned `method="logprobs"`, with a single-surface client kept on the surface it has, the server-limits ladder (`json_schema` → `json_object` → nothing, the Messages `thinking` field, and the rung a refused format field skips), a refused `prompt_cache_key`, a refused *value* that travels back as the provider's error, the schema that travels in the prompt when the request cannot carry one, a caller's `extra_body` winning over jevper's own default, the derived cache key (method included) and `cached_tokens` on all three usage paths, the message order — including a state ending on the assistant's turn, the `LabelReadoutError` a pinned `logprobs` raises against a provider that cannot do them, the message a truncated answer carries on every surface, the refusal a pinned label readout gets on the Messages surface, the error a reasoning-only answer carries, two-step reasoning on the chat surface, and the live probe's own capability read and failure reporting.

## Reference

The library is at [zhulinchng/jevper](https://github.com/zhulinchng/jevper) (Apache-2.0), where `docs/` holds the full reference — [`api.md`](https://github.com/zhulinchng/jevper/blob/main/docs/api.md), [`methods.md`](https://github.com/zhulinchng/jevper/blob/main/docs/methods.md), [`reasoning.md`](https://github.com/zhulinchng/jevper/blob/main/docs/reasoning.md), [`few-shot.md`](https://github.com/zhulinchng/jevper/blob/main/docs/few-shot.md), [`local-servers.md`](https://github.com/zhulinchng/jevper/blob/main/docs/local-servers.md), [`internals.md`](https://github.com/zhulinchng/jevper/blob/main/docs/internals.md), [`mlflow.md`](https://github.com/zhulinchng/jevper/blob/main/docs/mlflow.md).

`jevper` is an independent implementation of the documented System One wire format, not affiliated with, endorsed by, or supported by [TypeSafe AI](https://docs.typesafe.ai); questions about the hosted API itself belong in their docs.
