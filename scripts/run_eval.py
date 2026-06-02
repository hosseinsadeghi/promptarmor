"""Run the detection-quality eval against a labeled JSONL file.

Ties together: a dataset you downloaded with `scripts/fetch_data.py`
(`{"text", "label"}` JSONL) + a guardrail served at any OpenAI-compatible
`--base-url` -> FPR/FNR.

The guardrail can be local vLLM, or any OpenAI-compatible endpoint — just point
`--base-url`/`--model`/`--api-key` at it. No data download happens here; run
`fetch_data.py` first.

Examples:
    python scripts/run_eval.py --data data/eval/deepset.jsonl
    python scripts/run_eval.py --data data/eval/deepset.jsonl \\
        --base-url http://localhost:8000/v1 --model Qwen/Qwen3-32B --enable-thinking
    python scripts/run_eval.py --data data/eval/deepset.jsonl \\
        --base-url https://my-endpoint/v1 --model some-model --api-key "$MY_KEY" --limit 100
"""

import argparse
import json

from promptarmor import PromptArmor, PAConfig
from promptarmor.eval import load_labeled_jsonl, evaluate_batch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to labeled {text,label} JSONL")
    ap.add_argument("--base-url", default="http://localhost:8000/v1",
                    help="OpenAI-compatible endpoint (default: local vLLM)")
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--api-key", default="EMPTY", help='API key (default "EMPTY" for local vLLM)')
    ap.add_argument("--enable-thinking", action="store_true",
                    help="enable reasoning mode (recommended for ~8B models)")
    ap.add_argument("--include-definition", action="store_true",
                    help="add the injection definition to the prompt (for weak/small <8B models)")
    ap.add_argument("--no-guided-json", action="store_true",
                    help="disable vLLM guided decoding (for non-vLLM endpoints)")
    ap.add_argument("--max-concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N samples (smoke test)")
    args = ap.parse_args()

    samples = load_labeled_jsonl(args.data)
    if args.limit:
        samples = samples[: args.limit]
    pos = sum(1 for _, l in samples if l)
    print(f"loaded {len(samples)} samples ({pos} injection / {len(samples) - pos} benign) from {args.data}")

    cfg = PAConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        enable_thinking=args.enable_thinking,
        include_definition=args.include_definition,
        use_guided_json=not args.no_guided_json,
        max_concurrency=args.max_concurrency,
    )
    pa = PromptArmor(cfg)

    print(f"evaluating against {args.model} @ {args.base_url} ...")
    metrics = evaluate_batch(pa, samples, max_concurrency=args.max_concurrency)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
