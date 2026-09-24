# jevper features

- [Reasoning](#reasoning) — `ReasoningConfig`, native vs two-step, reading the trace
- [Few-shot examples](#few-shot-examples) — `Example`, the three levels, calibration
- [Async](#async) — `AsyncSystemOneClient`
- [Surfaces](#surfaces) — Chat Completions, Responses and Messages, and the fallbacks between them
- [Prompt caching](#prompt-caching) — message order, `prompt_cache_key`, `usage.cached_tokens`
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
| `auto` (default) | native provider reasoning | two-step | two-step |
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
  default applies. Anthropic requires at least `1024` and strictly less than `max_tokens`; jevper only
  refuses a value below `1` and lets the provider answer for itself.
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

A bare sequence applies to every question in the call; a mapping is keyed by question id. An answer that
matches no option raises `InvalidQuestionError` before any request. `probabilities` is only read by
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
| JSON schema | `response_format={"type": "json_schema", ...}` | `text={"format": {"type": "json_schema", ...}}` | none — the schema goes in the system prompt |
| grammar | `extra_body={"grammar": "..."}` | not available | not available |
| output cap | server default | server default | `max_tokens` required: jevper sends `1024` |

Neither OpenAI builder sends `max_tokens`/`max_output_tokens`: reasoning tokens count against those caps, and
a small cap silently truncates a reasoning model. The Messages protocol has no server-side default, so jevper
always sends `max_tokens` there (`1024`, or `extra_body={"max_tokens": n}`). Any other provider field goes
through `extra_body`.

Wherever the request cannot state the answer's shape, the JSON Schema travels in the prompt instead: always
on the Messages surface, and on the OpenAI surfaces when `structured_outputs=False` or the server refused the
strict schema. It joins the leading system message, so the question block keeps its place — and the answer's
shape is then only as good as the model's instruction-following.

`auto` also falls back between the surfaces, and both verdicts are remembered for the client's life:

- A **404 that does not name the model** is a missing route: the call is re-asked on
  `chat_completions`. A 404 that quotes the model (`ollama` and `vLLM` answer a bad model id that way) is
  about the model and is reported as it stands. An explicit `api="responses"` never falls back.
- A **surface that answers without a distribution** is marked and left behind for that model — ollama's
  Responses route returns an empty logprob list, llama.cpp's refuses the fields — so later calls start where
  the distribution is. A distribution arriving later on a marked surface clears the mark.
- `reasoning="native"` stops the move: native reasoning exists only on Responses, so switching would
  silently turn it into a two-step pass.
- The Messages surface is never asked for a label readout at all: `method="logprobs"`/`"grammar"` raise
  `UnsupportedMethodError` before any request, and `auto` starts in JSON there.

The state goes last (`system`, examples, question block, state) so a rubric's calls share the cacheable
prefix; a chat-list `state` carrying its own `system`/`developer` turn has that content folded into jevper's
system prompt, because a system turn after a user turn is a `400` on vLLM and SGLang and a 500 on
llama.cpp's Qwen template. Keep a `state` list to `system`/`user`/`assistant` turns.

## Prompt caching

`assemble()` builds the prompt in cache order: the system prompt, the few-shot example turns, the question
block, then the state. A cached prefix is reused up to the first token that differs, so putting the state
last is what lets two calls about the same rubric share anything — and on a hosted API whose minimum
cacheable prefix is 1024 tokens, it is what gets a prompt carrying a few examples over the line at all.

Every request carries `prompt_cache_key`, on both surfaces:

- the caller's, from `SystemOneClient(..., prompt_cache_key=...)` or `system_one(..., prompt_cache_key=...)`
  (the per-call key wins over the client's). It must be a non-blank string of at most 256 characters,
  checked before any request is sent — anything else raises `JevperError`.
- otherwise one derived per question from the model, the example turns and the question block: `"jevper-"`
  plus 32 hex characters. The state and the system prompt are excluded on purpose, since states are what
  vary between calls about one rubric; both passes of a two-step call pass the same key, so they land on the
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

## Client knobs

| Option | Default | Change it when |
| --- | --- | --- |
| `method` | `"auto"` | you have a reason (see SKILL.md) |
| `api` | `"auto"` (prefers Responses, then Chat Completions, then Messages) | a client exposes several surfaces and you want a specific one |
| `temperature` | `None` (not sent) | `0.0` for `structured`/`discrete`; leave unset for `logprobs` |
| `top_logprobs` | `20` | lower it only if the provider rejects the field; a pinned `logprobs`/`grammar` needs at least 2, since one logprob is not a distribution |
| `structured_outputs` | `True` | `False` sends `{"type": "json_object"}` instead of a strict schema, and the schema then travels in the system prompt — for providers that reject strict schemas |
| `prompt_cache_key` | `None` (derived per question) | route one rubric's calls to a shared cache, or keep tenants apart; non-blank, at most 256 characters |
| `normalize_probabilities` | `True` | `False` returns the model's `structured` numbers verbatim (the error is still recorded in `debug`) |
| `max_concurrency` | `8` | your provider rate-limits per key |
| `n_retry_malformed` | `1` | a model that keeps answering in prose |
| `retry` | `RetryPolicy()` | transient-failure retries: `n_retries=2`, `base_delay=0.5`, `max_delay=8.0` |
| `extra_body`, `extra_headers` | `None` | provider-specific fields — including `max_tokens` on the Messages surface, where jevper's `1024` default may be too small |

Constructor misuse (unknown `method`/`api`, `top_logprobs` outside `[0, 20]`, a pinned label method below
2, `max_concurrency < 1`, a negative retry field, a blank or over-long `prompt_cache_key`) raises
`JevperError` immediately, so a typo never reaches a provider. `ReasoningConfig` is a pydantic model, so its
own validation (`effort`, `budget_tokens`, `mode`) raises `pydantic.ValidationError` at the point you build
it, not from the client.
