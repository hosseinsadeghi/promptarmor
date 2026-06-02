"""Detection-quality evaluation harness.

Scope: measures detection quality only — FPR (clean flagged as injected) and
FNR (injected missed). End-to-end ASR/UA require an agent loop (AgentDojo) and
are out of scope for this module; to get them, run this guard as a preprocessor
inside AgentDojo and use its metrics.
"""


def _metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {"FPR": fpr, "FNR": fnr, "TP": tp, "FP": fp, "TN": tn, "FN": fn}


def evaluate(pa, samples):
    """samples: list[(text, label_bool)]  label=True means contains an injection."""
    tp = fp = tn = fn = 0
    for text, label in samples:
        pred = pa.det.detect(text)["injection"]
        if label and pred:
            tp += 1
        elif label and not pred:
            fn += 1
        elif not label and pred:
            fp += 1
        else:
            tn += 1
    return _metrics(tp, fp, tn, fn)


def evaluate_batch(pa, samples, max_concurrency: int | None = None):
    """Same as `evaluate` but runs detection concurrently for throughput.

    samples: list[(text, label_bool)].
    """
    texts = [t for t, _ in samples]
    labels = [bool(l) for _, l in samples]
    results = pa.det.detect_batch(texts, max_concurrency=max_concurrency)
    tp = fp = tn = fn = 0
    for label, res in zip(labels, results):
        pred = bool(res["injection"])
        if label and pred:
            tp += 1
        elif label and not pred:
            fn += 1
        elif not label and pred:
            fp += 1
        else:
            tn += 1
    return _metrics(tp, fp, tn, fn)
