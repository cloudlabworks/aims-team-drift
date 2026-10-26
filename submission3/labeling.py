"""Team Drift label construction: acquisition rule + the explicit response -> label rules (stdlib only, deterministic).

1. acquisition_function(input, *, labeled, context): reveal-or-skip for a streamed candidate. It is the kit's default
   random policy written out explicitly: reveal iff uniform(sha256(context, input)) < min(1, labels_remaining / items_remaining).
   `prediction` is deliberately NOT in the signature, so the evaluator makes no extra predict() calls for acquisition.
   Pure function of (input, context): the same stream always reveals the same items; nested budgets follow from the platform
   running one acquisition trajectory.
2. binarize_response / aggregate_responses: how recorded responses become 0/1 labels for training and for the local harness.
3. construct_label: when one subject-item pair has several recorded responses, reveal ONE of them chosen by a hash of an
   explicit key (not by an RNG state), with P(label=1) = k / n.
"""
import hashlib
import json
import math


def _uniform(key: str) -> float:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2 ** 64


def reveal_uniform(input, context) -> float:
    """The kit's reproducible uniform draw for one (context, input) candidate."""
    key = json.dumps([context, input], sort_keys=True, separators=(",", ":"))
    return _uniform(key)


def acquisition_function(input, *, labeled=None, context=None) -> bool:
    if context is None:
        raise ValueError("labeling.acquisition_function requires the streaming context")
    remaining, items = context["labels_remaining"], context["items_remaining"]
    if remaining <= 0 or items <= 0:
        return False
    return bool(reveal_uniform(input, context) < min(1.0, remaining / items))


def binarize_response(x):
    """Recorded response -> 0 / 1, or None when it is missing or not exactly 0 or 1 (graded values are never rounded)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    if v == 0.0:
        return 0
    if v == 1.0:
        return 1
    return None


def aggregate_responses(responses):
    """Iterable of recorded responses for one subject-item pair -> (n, k) over the binary ones; non-binary ones are dropped."""
    n = k = 0
    for x in responses:
        b = binarize_response(x)
        if b is not None:
            n += 1
            k += b
    return n, k


def construct_label(k: int, n: int, key: str) -> int:
    """Reveal one of the n recorded responses (k of them correct): 1 iff hash-uniform(key) < k / n. Deterministic in key."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n and n >= 1")
    return int(_uniform(key) < k / n)
