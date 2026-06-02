"""Pre-stage guardrail model weights to a controlled local path.

vLLM auto-pulls from HuggingFace if you pass a repo id, so this is for the
offline / air-gapped / compliance case: download once into a git-ignored
`models/` dir, then serve with HF_HUB_OFFLINE=1 so nothing hits the network at
inference time.

Requires the `model` extra:  pip install -e ".[model]"
"""

import argparse
import os

from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="Qwen/Qwen3-32B")
    ap.add_argument("--local-dir", default="models/Qwen3-32B")
    ap.add_argument("--revision", default=None)
    ap.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="override allow-patterns; default fetches what vLLM needs",
    )
    args = ap.parse_args()

    allow = args.include or ["*.safetensors", "*.json", "*.txt", "tokenizer*", "*.model", "*.py"]
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        revision=args.revision,
        allow_patterns=allow,
        token=os.environ.get("HF_TOKEN"),   # required for gated repos (e.g. Llama); Qwen3 is ungated
    )
    print(f"downloaded {args.repo_id} -> {path}")


if __name__ == "__main__":
    main()
