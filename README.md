# jevper

Agent skill for writing and debugging Python code that calls [`jevper`](https://github.com/zhulinchng/jevper) — the Jev (System One) interface that turns a state plus `Noul`/`Choice`/`Score` questions into typed answers with probabilities and confidence, over any OpenAI-compatible model.

## Install

```bash
npx skills add zhulinchng/jevper-skill
```

Single-skill repo: root `SKILL.md` with valid `name` + `description` frontmatter, so the installer picks it up directly (`--list` shows it without installing; `-a '*'` installs to every detected agent).

## What it covers

- **Questions**: `Noul` (one probability of true), `Choice` (2–255 labelled options), `Score` (2–10 ordered levels) — criteria rendered as prompts, answers carrying `choice`/`probabilities`/`confidence`, `score`/`legend`, `noul`, and how each `confidence` is computed
- **Methods**: leave `method` unset and `auto` answers with `logprobs` where the provider returns them, `structured` where it does not, remembered per `(model, surface)`; when to pin `grammar`, `discrete`, `structured` or `logprobs` instead — and the 26-option ceiling on the two methods that read a label *token*
- **Provider support**: who returns logprobs (OpenAI chat models, DeepSeek, Together, Ollama, llama.cpp, vLLM) and who rejects the fields (OpenAI reasoning models, Claude, Gemini's OpenAI-compatibility endpoint), with the exact error strings and a one-line probe
- **Reasoning**: `ReasoningConfig` — native provider reasoning vs a two-step analysis-then-answer pass, `reasoning_text()`, and the doubled call count that comes with it
- **Few-shot examples**: `Example`, the three attachment levels and their precedence, answers rendered in the active method's format, calibration through explicit `probabilities`
- **Async**: `AsyncSystemOneClient`, same signatures, semaphore instead of thread pool
- **Surfaces**: Chat Completions vs Responses — logprobs via `include`, `response_format` vs `text.format`, grammar only on chat, `store=false`
- **Client knobs**: `temperature`, `top_logprobs`, `structured_outputs`, `normalize_probabilities`, `max_concurrency`, `n_retry_malformed`, `retry`
- **Failure triage**: which errors are local (zero requests sent), which are readout failures worth one corrective retry, which are `ProviderError`; transient retry policy; the `debug` keys that show what actually happened
- **Offline testing**: a duck-typed stub client that drives the real readout path from canned bodies — no HTTP, no key, no tokens

## Layout

- `SKILL.md` — entry point: quick start, question/method/error tables, provider matrix, checklist
- `references/features.md` — reasoning, few-shot examples, async, surfaces, client knobs
- `references/troubleshooting.md` — provider logprob matrix, error triage, `debug` recipes, symptom → fix
- `scripts/offline_stub.py` — duck-typed stub client (`logprobs`, `structured`, `reject_logprobs`, `no_alternatives`, `reasoning`) plus its own self-test and live probe

## Verify

```bash
pip install jevper                                 # Python 3.10+, pydantic>=2.7
python scripts/offline_stub.py --check             # offline self-test: no network, no API key
python scripts/offline_stub.py --live --model <id> # needs openai + OPENAI_API_KEY; OPENAI_BASE_URL for self-hosted
```

The self-test drives `StubClient` through the real jevper readout path: the logprobs softmax, the structured JSON readout, `auto`'s fallback and its per-`(model, surface)` memory, the `LabelReadoutError` a pinned `logprobs` raises against a provider that cannot do them, and two-step reasoning on the chat surface. The live probe prints which method a real provider resolves to, the readout source, and the answer — run it before writing an integration against a new model.

## Reference

The library is at [zhulinchng/jevper](https://github.com/zhulinchng/jevper) (Apache-2.0), where `docs/` holds the full reference — [`api.md`](https://github.com/zhulinchng/jevper/blob/main/docs/api.md), [`methods.md`](https://github.com/zhulinchng/jevper/blob/main/docs/methods.md), [`reasoning.md`](https://github.com/zhulinchng/jevper/blob/main/docs/reasoning.md), [`few-shot.md`](https://github.com/zhulinchng/jevper/blob/main/docs/few-shot.md), [`internals.md`](https://github.com/zhulinchng/jevper/blob/main/docs/internals.md).

`jevper` is an independent implementation of the documented System One wire format, not affiliated with, endorsed by, or supported by [TypeSafe AI](https://docs.typesafe.ai); questions about the hosted API itself belong in their docs.
