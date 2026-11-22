"""Atomic repeat checkpoints and CSV exports for the Day 2b-2 baselines."""

import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from dataclasses import fields
from io import StringIO
from pathlib import Path

from specserve.decode.greedy import DecodeIteration


JOIN = ("config_id", "method", "prompt_id", "repeat")
ITERATION_FIELDS = tuple(field.name for field in fields(DecodeIteration))
TIMING = JOIN + (
    "category", "batch_size", "max_new_tokens", "eos_token_id", "stopped_on_eos",
    "stop_reason", "tokens_generated", "prefill_s", "decode_s", "total_s",
    "decode_tok_per_s", "throughput_undefined_reason", "dtype", "seed", "git_commit",
    "model_revision",
)
ITERATIONS = JOIN + ("iteration",) + ITERATION_FIELDS
CORRECTNESS = ("stage", "method", "prompt_id", "repeat", "status", "details")
ITERATION_SUMMARY = ("scope",) + JOIN + (
    "excluded_final_rows", "retained_rows", "accepted", "trials", "tokens_emitted",
    "target_passes", "alpha_hat", "alpha_undefined_reason", "tokens_per_target_pass",
    "tokens_per_pass_undefined_reason",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    # The destination only becomes visible once the whole file is durable.
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as tmp:
        try:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_json(path, value):
    atomic_write(path, json_bytes(value))


def write_csv(path, columns, rows):
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, output.getvalue().encode())


def write_correctness(path, rows):
    write_csv(path, CORRECTNESS, (
        {**row, "details": json_bytes(row["details"]).decode().strip()} for row in rows
    ))


def write_summaries(directory, units):
    summary = summarize(units)
    write_csv(directory / "iteration_summary.csv", ITERATION_SUMMARY, (
        {"scope": scope, **row}
        for scope, key in (("repeat", "iterations_by_repeat"), ("method", "iterations_pooled_by_method"))
        for row in summary[key]
    ))
    write_json(directory / "summary.json", {"timing_by_prompt": summary["timing_by_prompt"]})


def throughput(tokens_generated, decode_s):
    if tokens_generated <= 1:
        return None, "no_tokens_after_prefill"
    if decode_s is None:
        return None, "missing_decode_duration"
    duration = float(decode_s)
    if not math.isfinite(duration) or duration <= 0:
        return None, "nonfinite_or_nonpositive_decode_duration"
    return (tokens_generated - 1) / duration, None


def stop_reason(token_ids, eos, budget):
    if token_ids and token_ids[-1] == eos:
        return "eos"
    if len(token_ids) == budget:
        return "budget"
    return "unexpected_stop"


def repeat_path(directory, key):
    return directory / key["method"] / key["prompt_id"] / f'{key["repeat"]}.json'


def save_repeat(path, run_id, unit):
    # Timing, iterations and generated IDs/stop information are a single commit.
    payload = {"run_id": run_id, "unit": unit}
    write_json(path, {"payload": payload, "sha256": digest(json_bytes(payload))})


def complete_unit(unit, key, expected):
    if set(unit) != {"timing", "iterations", "token_ids"}:
        return False
    timing = unit["timing"]
    if set(timing) != set(TIMING) or any(
        timing[field] != value for field, value in (key | expected).items()
    ):
        return False
    tokens = unit["token_ids"]
    rows = unit["iterations"]
    if not isinstance(tokens, list) or not isinstance(rows, list):
        return False
    if not all(type(token) is int and token >= 0 for token in tokens):
        return False
    eos = timing["eos_token_id"]
    budget = timing["max_new_tokens"]
    reason = stop_reason(tokens, eos, budget)
    rate, undefined = throughput(len(tokens), timing["decode_s"])
    if (
        not 1 <= len(tokens) <= budget or eos in tokens[:-1]
        or len(tokens) != timing["tokens_generated"] or reason != timing["stop_reason"]
        or timing["stopped_on_eos"] != (reason == "eos")
        or timing["decode_tok_per_s"] != rate or timing["throughput_undefined_reason"] != undefined
    ):
        return False
    for index, row in enumerate(rows):
        if set(row) != set(ITERATIONS) or row["iteration"] != index:
            return False
        if any(row[field] != value for field, value in key.items()):
            return False
        if row["stopped"] != (index == len(rows) - 1):
            return False
        for field in ("proposed", "accepted", "tokens_emitted", "target_passes"):
            if type(row[field]) is not int:
                return False
        if not (
            0 <= row["accepted"] <= row["proposed"] and 1 <= row["tokens_emitted"]
            and row["target_passes"] >= 1 and type(row["mismatch"]) is bool
            and row["accepted"] + row["mismatch"] <= row["proposed"]
        ):
            return False
    return sum(row["tokens_emitted"] for row in rows) == len(tokens) - 1


def load_repeat(path, run_id, key, expected):
    if not path.exists():
        return None
    try:
        envelope = json.loads(path.read_bytes())
        payload = envelope["payload"]
        if envelope["sha256"] != digest(json_bytes(payload)):
            return None
        if payload["run_id"] != run_id:
            raise RuntimeError(f"Repeat belongs to a different run: {path}")
        unit = payload["unit"]
        if complete_unit(unit, key, expected):
            return unit
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
        pass
    # The caller redoes this repeat and atomically replaces the incomplete file.
    return None


def open_run(directory, identity, config_bytes, new_run_id):
    manifest = directory / "baseline/run.json"
    config_path = directory / "baseline/bench.yaml"
    env_path = directory / "env.txt"
    environment = json_bytes(identity)
    if manifest.exists():
        saved = json.loads(manifest.read_bytes())
        for field, expected in identity.items():
            if saved["identity"].get(field) != expected:
                raise RuntimeError(f"Resume is incompatible: {field}. Use a fresh output root.")
        if not config_path.is_file() or config_path.read_bytes() != config_bytes:
            raise RuntimeError("Saved run YAML is missing or differs from the exact checkout YAML")
        if not env_path.is_file() or env_path.read_bytes() != environment:
            raise RuntimeError("Saved environment metadata is missing or inconsistent")
        return saved
    baseline = manifest.parent
    if baseline.exists() and any(
        path != config_path and not path.name.startswith(".pending-") for path in baseline.iterdir()
    ):
        raise RuntimeError("Run identity is missing. Use a fresh output root.")
    # With no manifest, an interrupted initialization may have left YAML/env,
    # but no repeat can have started yet.
    atomic_write(config_path, config_bytes)
    atomic_write(env_path, environment)
    saved = {"run_id": new_run_id, "identity": identity, "sessions": []}
    write_json(manifest, saved)
    return saved


def iteration_metrics(rows):
    retained = [row for row in rows if not row["stopped"]]
    accepted = sum(row["accepted"] for row in retained)
    trials = sum(row["accepted"] + int(row["mismatch"]) for row in retained)
    tokens = sum(row["tokens_emitted"] for row in retained)
    passes = sum(row["target_passes"] for row in retained)
    return {
        "excluded_final_rows": len(rows) - len(retained),
        "retained_rows": len(retained),
        "accepted": accepted,
        "trials": trials,
        "tokens_emitted": tokens,
        "target_passes": passes,
        "alpha_hat": accepted / trials if trials else None,
        "alpha_undefined_reason": None if trials else "zero_trials",
        "tokens_per_target_pass": tokens / passes if passes else None,
        "tokens_per_pass_undefined_reason": None if passes else "zero_retained_passes",
    }


def _distribution(values):
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {
        "median": statistics.median(valid) if valid else None,
        "min": min(valid) if valid else None,
        "max": max(valid) if valid else None,
        "spread_max_minus_min": max(valid) - min(valid) if valid else None,
        "valid_repeats": len(valid),
        "undefined_repeats": len(values) - len(valid),
    }


def summarize(units):
    by_prompt = {}
    by_method = {}
    per_repeat = []
    for unit in units:
        timing = unit["timing"]
        method_key = (timing["config_id"], timing["method"])
        prompt_key = (*method_key, timing["prompt_id"])
        by_prompt.setdefault(prompt_key, []).append(timing)
        by_method.setdefault(method_key, []).extend(unit["iterations"])
        per_repeat.append({
            **{field: timing[field] for field in JOIN},
            **iteration_metrics(unit["iterations"]),
        })
    timing_summary = []
    for (config_id, method, prompt_id), rows in by_prompt.items():
        timing_summary.append({
            "config_id": config_id, "method": method, "prompt_id": prompt_id,
            **{metric: _distribution([row[metric] for row in rows])
               for metric in ("prefill_s", "decode_s", "total_s", "decode_tok_per_s")},
        })
    greedy_medians = {
        (row["config_id"], row["prompt_id"]): row["decode_tok_per_s"]["median"]
        for row in timing_summary if row["method"] == "greedy"
    }
    for row in timing_summary:
        greedy = greedy_medians.get((row["config_id"], row["prompt_id"]))
        measured = row["decode_tok_per_s"]["median"]
        row["decode_speed_ratio_vs_greedy"] = measured / greedy if greedy and measured else None
    return {
        "timing_by_prompt": timing_summary,
        "iterations_by_repeat": per_repeat,
        "iterations_pooled_by_method": [
            {"config_id": config_id, "method": method, **iteration_metrics(rows)}
            for (config_id, method), rows in by_method.items()
        ],
    }
