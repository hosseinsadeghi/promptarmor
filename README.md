# PromptArmor

An LLM-prompted **prompt-injection content guard** — a stateless, content-level
filter that screens untrusted text (tool outputs, RAG chunks, email/doc bodies,
web pages) before it reaches a downstream agent. This is an independent
reimplementation of the method in **arXiv:2507.15219** (Shi et al.).

Given an untrusted sample it:

1. prompts a **guardrail LLM** to detect injected instructions and extract the
   injected span(s);
2. **removes / blocks / flags** them per policy;
3. returns the sanitized text (or a block signal) for the downstream pipeline.

The guardrail LLM runs locally on a **vLLM OpenAI-compatible endpoint** — no
external APIs. Default model `Qwen/Qwen3-32B`.

> **Namesake collision:** the GitHub project `prompt-armor/prompt-armor` is an
> *unrelated* regex/meta-classifier with no LLM. This package implements the
> arXiv:2507.15219 method instead.
>
> **Affiliation:** this is an independent reimplementation and is **not
> affiliated with the paper authors**. The method is from the paper; the code is
> original and MIT-licensed.

## Install

```bash
pip install -e .            # runtime (openai)
pip install -e ".[dev]"     # + pytest
```

Requires Python 3.10+.

## Usage

```python
from promptarmor import PromptArmor

pa = PromptArmor()                         # uses local vLLM at :8000 by default
res = pa(tool_output, user_task=current_task)
if res["clean"] is None:                   # blocked (on_detect="block")
    abort_step()
else:
    feed_to_agent(res["clean"])            # sanitized text
# res = {"clean": <str|None>, "flagged": <bool>, "spans": [<injected text>, ...]}
```

Wrap **every untrusted-text ingress point** in your agent/pipeline.

### Policies (`PAConfig.on_detect`)

- `"remove"` (default) — fuzzy-delete the injected span(s); maximizes utility.
- `"block"` — return `clean=None` to halt the step; maximizes safety.
- `"flag"` — keep the text but mark it flagged.

> For high-stakes flows prefer `"block"` over `"remove"`. Partial removal can
> leave a steering fragment that the agent then acts on.

Other knobs: `model`, `base_url`, `enable_thinking` (recommended on for ~8B
models), `include_definition` (only for weak/small <8B models),
`removal_length_guard`, `max_concurrency`, `use_guided_json`.

## Serving the guardrail model (vLLM)

```bash
vllm serve Qwen/Qwen3-32B \
  --enable-auto-tool-choice --reasoning-parser qwen3 \
  --port 8000
```

`--reasoning-parser` separates Qwen3 thinking output into `reasoning_content`
(the client also strips stray `<think>...</think>` blocks defensively). Valid
JSON output is enforced via vLLM guided decoding (`guided_json`) when
`use_guided_json=True`.

## Evaluation

Scope: **detection quality only** — FPR (clean flagged as injected) and FNR
(injected missed). End-to-end ASR/UA require an agent loop (AgentDojo) and are
out of scope here; run this guard as a preprocessor inside AgentDojo for those.

```python
from promptarmor import PromptArmor
from promptarmor.eval import evaluate, hand_built_smoke_set

pa = PromptArmor()
print(evaluate(pa, hand_built_smoke_set()))
# -> {"FPR": ..., "FNR": ..., "TP": ..., "FP": ..., "TN": ..., "FN": ...}
```

**Datasets are referenced, not vendored.** Loaders point at corpora you supply:

- `load_agentdojo_injections(...)` — needs `pip install agentdojo`; builds
  positives by injecting its attack strings into clean tool outputs.
- `load_open_prompt_injection(path=...)` — point at a local JSONL
  (`{"text": ..., "label": bool}`) from
  [Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection).
- `hand_built_smoke_set()` — tiny public/synthetic set for a live-endpoint smoke
  test.

Targets (per the spec): on AgentDojo-style data, FPR/FNR < 1% for 32B. Expect a
few % on out-of-distribution sets — that's the real-world ceiling.

### Model sweep

Run Qwen3-32B (thinking off), Qwen3-8B (thinking on vs off — FNR should drop
with thinking on), and any other served models. This doubles as routing data:
thinking mode earns its cost mainly in the mid-size band.

## Download eval data & models

Weights and dataset corpora are **downloaded locally and never committed** —
`data/` and `models/` are git-ignored. The repo ships only the code that
fetches them. The downloaders live behind optional extras to keep the runtime
lean (`openai` only):

```bash
pip install -e ".[data,model]"     # datasets + huggingface_hub
```

**Eval data** — `scripts/fetch_data.py` pulls a labeled HuggingFace dataset and
writes the `{"text": str, "label": bool}` JSONL that the eval loader reads
(`label=True` = contains an injection). It auto-detects text/label columns:

```bash
python scripts/fetch_data.py --repo-id deepset/prompt-injections --split test --out data/eval/deepset.jsonl
python scripts/fetch_data.py --repo-id xTRam1/safe-guard-prompt-injection --split test --out data/eval/safeguard.jsonl
```

Then evaluate it against any OpenAI-compatible guardrail endpoint — one command,
no code to write:

```bash
python scripts/run_eval.py --data data/eval/deepset.jsonl \
    --base-url http://localhost:8000/v1 --model Qwen/Qwen3-32B
# point --base-url / --model / --api-key at any OpenAI-compatible endpoint;
# add --limit N for a quick smoke run, --enable-thinking for ~8B models.
```

Or from Python:

```python
from promptarmor import PromptArmor, PAConfig
from promptarmor.eval import load_labeled_jsonl, evaluate_batch
pa = PromptArmor(PAConfig(base_url="http://localhost:8000/v1", model="Qwen/Qwen3-32B"))
print(evaluate_batch(pa, load_labeled_jsonl("data/eval/deepset.jsonl")))
```

> **Direct vs. indirect caveat:** `deepset/prompt-injections` and
> `xTRam1/safe-guard-prompt-injection` skew toward **direct** (chatbot-prompt)
> injection — fine for a detector smoke test, but PromptArmor's real target is
> **indirect** content injection. For on-target evaluation prefer indirect
> sources: repo/MCP-content sets (e.g. `prodnull/prompt-injection-repo-dataset`),
> or AgentDojo attack strings embedded into clean tool outputs (via the
> `load_agentdojo_injections` loader). The script handles any `text`+`label`
> set; point it at whichever.

**Models** — `scripts/fetch_model.py` pre-stages weights to a local dir for
offline/air-gapped/compliance use (vLLM otherwise auto-pulls from HF on first
serve):

```bash
python scripts/fetch_model.py --repo-id Qwen/Qwen3-32B --local-dir models/Qwen3-32B
python scripts/fetch_model.py --repo-id Qwen/Qwen3-8B  --local-dir models/Qwen3-8B   # for the thinking-mode sweep
```

`HF_TOKEN` is needed only for gated repos (e.g. Llama); Qwen3 is ungated.

**Serve** — `scripts/serve_vllm.sh` accepts either a local staged dir (offline)
or an HF id (auto-download):

```bash
./scripts/serve_vllm.sh models/Qwen3-32B 8000     # or: ./scripts/serve_vllm.sh Qwen/Qwen3-32B 8000
```

Once weights are staged locally, uncomment `export HF_HUB_OFFLINE=1` in the
script so nothing hits the network at inference time.

## Tests

```bash
pytest                      # offline: parser + fuzzy-removal unit tests
```

The unit tests do not require a running endpoint.

## License

MIT — see [LICENSE](LICENSE).
