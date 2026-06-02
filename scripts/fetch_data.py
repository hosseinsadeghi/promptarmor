"""Fetch a labeled prompt-injection dataset from HuggingFace into the JSONL
schema the eval loader reads: one object per line, {"text": str, "label": bool}
where label=True means the sample contains an injection.

Schema-agnostic: auto-detects text/label columns so it works across datasets.
The output schema matches `promptarmor.eval.datasets.load_open_prompt_injection`
(it reads `text`/`label`); adjust below if your loader differs.

Requires the `data` extra:  pip install -e ".[data]"
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

TEXT_KEYS = ["text", "prompt", "content", "input", "sentence"]
LABEL_KEYS = ["label", "labels", "is_injection", "injection", "y"]
POSITIVE = {1, "1", "injection", "inject", "attack", "malicious", "unsafe", "jailbreak", "true"}


def to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return int(v) == 1
    if isinstance(v, str):
        return v.strip().lower() in POSITIVE
    return False


def pick(cols, candidates):
    low = {c.lower(): c for c in cols}
    for k in candidates:
        if k in low:
            return low[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="deepset/prompt-injections")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="data/eval/deepset.jsonl")
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--label-col", default=None)
    args = ap.parse_args()

    ds = load_dataset(args.repo_id, split=args.split)
    tcol = args.text_col or pick(ds.column_names, TEXT_KEYS)
    lcol = args.label_col or pick(ds.column_names, LABEL_KEYS)
    if not tcol or not lcol:
        raise SystemExit(
            f"Could not detect columns in {ds.column_names}; pass --text-col/--label-col"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = pos = 0
    with out.open("w") as f:
        for row in ds:
            text = row[tcol]
            if not isinstance(text, str) or not text.strip():
                continue
            label = to_bool(row[lcol])
            f.write(json.dumps({"text": text, "label": label}) + "\n")
            n += 1
            pos += int(label)
    print(f"wrote {n} rows ({pos} injection / {n - pos} benign) -> {out}")


if __name__ == "__main__":
    main()
