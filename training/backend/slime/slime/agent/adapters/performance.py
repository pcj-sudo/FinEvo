"""Opt-in numeric generation telemetry for controlled throughput experiments.

Never write bearer/session credentials, messages, tokens, upstream response
text, or arbitrary metadata. Prefix preservation and GPU cache reuse are
different measurements; missing cached_tokens stays null.
"""
import hashlib
import json
import math
import os
import time


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def generate_metrics(session_id, request_id, prompt_tokens, metadata, elapsed):
    return {
        "time": time.time(),
        "session_hash": hashlib.sha256((session_id or "default").encode()).hexdigest()[:20],
        "request_id": request_id,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": _number(metadata.get("completion_tokens")),
        "cached_tokens": _number(metadata.get("cached_tokens")),
        "elapsed_seconds": elapsed,
        "server_e2e_latency": _number(metadata.get("e2e_latency")),
        "server_queue_time": _number(metadata.get("queue_time")),
        "server_prefill_time": _number(metadata.get("prefill_time")),
        "server_decode_time": _number(metadata.get("decode_time")),
    }


def record_generate(session_id, request_id, prompt_tokens, metadata, elapsed):
    path = os.environ.get("RSI_GENERATE_METRICS")
    if not path:
        return
    event = generate_metrics(session_id, request_id, prompt_tokens, metadata, elapsed)
    # Require an existing parent. One bounded append; do not silently substitute
    # invented cache hit rates or lose a failed instrumentation run.
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")
