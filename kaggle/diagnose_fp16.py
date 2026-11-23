"""Untimed T4 diagnostic for the original chat_qa-25 fp16 divergence."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import random
import re
import sys
import tarfile
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from unittest.mock import patch

AUDITED_COMMIT = "76232827e3de37ae03768ce98104a934fd227f39"
PROMPT_ID = "chat_qa-25"
POSITION = 90
CACHE_LENGTH = 123
TOKEN_IDS = (367, 2125, 13100)
WINNERS = (367, 13100)
REQUIRED = ("env.txt", "baseline/run.json", "baseline/bench.yaml", "baseline/correctness.csv")
ARCHIVED_AUDIT = re.compile(r"baseline/audits/[0-9a-f]{32}\.csv")
MAX_EVIDENCE_BYTES = 64 * 1024 * 1024


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--git-sha", choices=[AUDITED_COMMIT], required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-experiments", type=Path)
    source.add_argument("--source-archive", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Fresh diagnostic evidence directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["GIT_OPTIONAL_LOCKS"] = "0"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.checkout.resolve()))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from kaggle import run_baselines as baseline
from kaggle.baseline_artifacts import CORRECTNESS, atomic_write, digest, json_bytes, write_json
from kaggle.baseline_audit import ReplayTarget, _locate_forward, first_difference


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_archive(path):
    """Read selected regular files without extracting any member to disk."""
    payload = {}
    seen = set()
    with path.open("rb") as raw:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: raw.read(1024 * 1024), b""):
            checksum.update(chunk)
        raw.seek(0)
        with gzip.GzipFile(fileobj=raw) as compressed:
            data = compressed.read(MAX_EVIDENCE_BYTES + 1)
            require(len(data) <= MAX_EVIDENCE_BYTES, "Archive exceeds audit size limit")
            # A bounded read also checks the gzip checksum and detects a truncated stream.
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
                for member in archive:
                    name = PurePosixPath(member.name)
                    require(
                        name.parts and name.parts[0] == "experiments" and ".." not in name.parts
                        and "\\" not in member.name and name not in seen,
                        f"Invalid or duplicate archive path: {member.name}",
                    )
                    seen.add(name)
                    require(member.isdir() or member.isreg(), f"Non-regular archive member: {name}")
                    relative = name.relative_to("experiments").as_posix()
                    if relative in REQUIRED or ARCHIVED_AUDIT.fullmatch(relative):
                        require(member.isreg(), f"Required evidence is a directory: {name}")
                        with archive.extractfile(member) as source:
                            payload[relative] = source.read()
                        require(len(payload[relative]) == member.size, f"Truncated archive member: {name}")
                footer = data[archive.offset:]
                require(len(footer) >= 1024 and not any(footer), "Invalid or truncated tar end markers")
    return payload, checksum.hexdigest()


def load_source(directory, archive):
    def from_directory(name):
        path = directory / name
        require(path.is_file() and path.resolve().is_relative_to(directory.resolve()),
                f"Missing or invalid original evidence: {name}")
        return path.read_bytes()

    if archive is not None:
        files, archive_hash = read_archive(archive)
    else:
        files = {name: from_directory(name) for name in REQUIRED}
        archive_hash = None
    for name in REQUIRED:
        require(name in files and files[name], f"Missing or empty original evidence: {name}")
    manifest = json.loads(files["baseline/run.json"])
    require(isinstance(manifest, dict) and isinstance(manifest.get("sessions"), list)
            and manifest["sessions"], "Missing original run sessions")
    session = manifest["sessions"][-1]
    session_id = session.get("id") if isinstance(session, dict) else None
    require(isinstance(session_id, str) and re.fullmatch(r"[0-9a-f]{32}", session_id), "Invalid session ID")
    archived = f"baseline/audits/{session_id}.csv"
    if directory is not None:
        files[archived] = from_directory(archived)
    require(files.get(archived) == files["baseline/correctness.csv"], "Missing or inconsistent archived audit")
    return {name: files[name] for name in (*REQUIRED, archived)}, manifest, archive_hash


def validate_audit(files, manifest, identity, config, records):
    require(manifest.get("identity") == identity, "Original run identity differs from the pinned inputs or environment")
    require(json.loads(files["env.txt"]) == identity, "Original environment and run identity disagree")
    require(files["baseline/bench.yaml"] == config, "Original YAML differs from the audited checkout")
    session = manifest["sessions"][-1]
    require(
        isinstance(manifest.get("run_id"), str) and bool(manifest["run_id"])
        and manifest.get("requires_dev_review") is True and manifest.get("speed_results_accepted") is False
        and session.get("status") == "blocked" and session.get("failed_stage") == "fp16_audit",
        "Original manifest does not record a blocked untimed fp16 audit",
    )
    for dtype in ("torch.float16", "torch.float32"):
        runtime = manifest.get("target_runtime", {}).get(dtype)
        require(
            isinstance(runtime, dict) and runtime.get("attention_implementation") in ("sdpa", "eager")
            and runtime == session.get(dtype),
            f"Missing or inconsistent supported attention implementation for {dtype}",
        )

    reader = csv.DictReader(io.StringIO(files["baseline/correctness.csv"].decode()), strict=True)
    require(tuple(reader.fieldnames or ()) == CORRECTNESS, "Invalid correctness CSV header")
    rows = list(reader)
    require(all(list(row) == list(CORRECTNESS) and None not in row.values() for row in rows),
            "Invalid correctness CSV row")
    details = [json.loads(row["details"]) for row in rows]
    require(all(isinstance(item, dict) for item in details), "Invalid correctness CSV evidence")
    prompts = [record["id"] for record in records]
    subset = [prompt_id for prompt_id in prompts if int(prompt_id.rsplit("-", 1)[1]) <= 5]
    expected = [("integrity", "", ""), ("tokenizer_and_generation", "", "")]
    expected += [
        (stage, method, prompt_id)
        for stage, selected in (("fp32_first_20", subset), ("fp32_full", prompts))
        for method in baseline.METHODS
        for prompt_id in selected
    ]
    expected += [("fp16_untimed", "prompt_lookup", prompt_id) for prompt_id in prompts]
    require(
        [(row["stage"], row["method"], row["prompt_id"]) for row in rows] == expected
        and all(not row["repeat"] for row in rows),
        "Original audit has missing, duplicate, or out-of-order rows",
    )
    pre_fp16 = len(expected) - len(prompts)
    require(all(row["status"] == "pass" for row in rows[:pre_fp16]), "Original FP32 or integrity gate failed")
    require(details[0].get("benchmark_sha256") == identity["benchmark_sha256"], "Audit benchmark hash differs")
    require(
        details[1].get("revision") == identity["settings"]["model_revision"]
        and details[1].get("token_counts_checked") == len(prompts)
        and isinstance(details[1].get("generation_config"), dict),
        "Invalid original tokenizer or generation configuration evidence",
    )
    for row, item in zip(rows[2:pre_fp16], details[2:pre_fp16]):
        count = item.get("reference_length")
        require(type(count) is int and 0 < count <= identity["settings"]["max_new_tokens"]
                and count == item.get("output_length"), f"Invalid original FP32 row: {row['prompt_id']}")

    divergences = {}
    for row, item in zip(rows[pre_fp16:], details[pre_fp16:]):
        status = row["status"]
        if status == "pass":
            count = item.get("output_length")
            require(type(count) is int and 0 < count <= identity["settings"]["max_new_tokens"]
                    and "case_id" not in item, "Invalid passing FP16 row")
            continue
        require(status in ("fail", "reviewed"), "Invalid FP16 audit status")
        evidence = {key: item[key] for key in (
            "position", "reference_token_id", "output_token_id", "reference_stop", "output_stop",
            "reference_replay", "output_replay", "outputs_sha256",
        )}
        case_id = digest(json_bytes([identity, row["prompt_id"], row["method"], evidence]))
        require(item.get("case_id") == case_id, f"Inconsistent case ID: {row['prompt_id']}")
        require(isinstance(item["outputs_sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", item["outputs_sha256"]),
                "Invalid original output digest")
        divergences[row["prompt_id"]] = (status, item)
    require(
        session.get("fp16_divergences") == {"fp16_untimed": len(divergences), "fp16_measured": 0},
        "FP16 divergence count differs between CSV and manifest",
    )
    status, case = divergences[PROMPT_ID]
    require(
        status == "fail" and case.get("near_tie_reviewable") is False and case["position"] == POSITION
        and (case["reference_token_id"], case["output_token_id"]) == WINNERS,
        "Original chat_qa-25 outcome differs from the reported case",
    )
    prompt_length = next(item["prompt_tokens"] for item in records if item["id"] == PROMPT_ID)
    for name, pair, logits in (
        ("reference_replay", [367, 2125], [17.28125, 17.28125]),
        ("output_replay", [13100, 367], [17.296875, 17.28125]),
    ):
        replay = case[name]
        prediction = replay["prediction"]
        widths = prediction["query_lengths_through_forward"]
        require(
            replay.get("reproduced") is True and prediction["source"] == "original_shape_replay"
            and prediction["prefix_matches"] is True and prediction["forward_shape"] == [1, 1]
            and prediction["cache_length_before_forward"] == CACHE_LENGTH and prediction["logit_row"] == 0
            and prediction["top_two"] == {"ids": pair, "logits": logits, "gap": logits[0] - logits[1]}
            and prediction["compared_logits"] == {"367": 17.28125, "13100": logits[0]}
            and isinstance(widths, list) and bool(widths) and widths[0] == prompt_length and widths[-1] == 1
            and len(widths) == prediction["forward_index"] + 1
            and all(type(width) is int and 0 < width <= identity["settings"]["prompt_lookup_num_tokens"] + 1
                    for width in widths[1:])
            and (name != "reference_replay" or widths[1:] == [1] * POSITION),
            f"Invalid original chat_qa-25 evidence: {name}",
        )
    return case, details[1]["generation_config"]


def persist_run(output, script, work):
    """Keep completed artifacts and the active phase if the run raises."""
    output.mkdir(parents=True, exist_ok=False)
    state = {"script_sha256": digest(script.read_bytes()), "git_commit": AUDITED_COMMIT, "status": "running"}

    def phase(name):
        state["phase"] = name
        write_json(output / "diagnostic.json", state)

    phase("preflight")
    try:
        state["result"] = work(phase)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(output / "diagnostic.json", state)


def save_tensors(path, tensors):
    buffer = io.BytesIO()
    torch.save(tensors, buffer)
    atomic_write(path, buffer.getvalue())


def snapshot(cache):
    if cache is None:
        return None
    return [(layer.keys.to("cpu", copy=True), layer.values.to("cpu", copy=True)) for layer in cache.layers]


def clone_cache(target, tensors, device):
    if tensors is None:
        return None
    # DynamicCache appends in place. Each arm starts from its own copy.
    layers = [(key.to(device, copy=True), value.to(device, copy=True)) for key, value in tensors]
    return DynamicCache(ddp_cache_data=layers, config=target.config)


def mask_evidence(mask, causal, q_length, k_length, past_length):
    # For cached verify row i, the last legal key is at past_length + i.
    expected = torch.arange(k_length) <= (past_length + torch.arange(q_length)[:, None])
    if mask is None:
        allowed = torch.ones(q_length, k_length, dtype=torch.bool)
        if causal:
            allowed = allowed.tril()
        valid_values = True
    else:
        mask = mask.detach().cpu()
        allowed = mask if mask.dtype == torch.bool else mask == 0
        valid_values = mask.dtype == torch.bool or bool(
            (allowed | torch.isneginf(mask) | (mask == torch.finfo(mask.dtype).min)).all()
        )
    mask_shape_ok = mask is None or tuple(mask.shape[-2:]) == (q_length, k_length)
    return {
        "query_length": q_length, "key_length": k_length, "is_causal": causal,
        "mask_shape": None if mask is None else list(mask.shape),
        "mask_dtype": None if mask is None else str(mask.dtype),
        "expected_causal_pattern": (
            k_length == past_length + q_length and mask_shape_ok and valid_values
            and bool((allowed == expected).all())
        ),
    }


def observed_forward(target, ids, cache, kwargs, past_length, *, save_masks=False):
    positions, attention, masks = [], [], []
    backend = target.config._attn_implementation

    def rotary(_module, _args, call_kwargs):
        positions.extend(call_kwargs["position_ids"].cpu().tolist())

    def record(mask, causal, q_length, k_length):
        attention.append(mask_evidence(mask, causal, q_length, k_length, past_length))
        if save_masks:
            masks.append(None if mask is None else mask.to("cpu", copy=True))

    def eager(_module, _args, call_kwargs):
        record(call_kwargs["attention_mask"], False, ids.shape[1], past_length + ids.shape[1])

    sdpa = F.scaled_dot_product_attention

    def traced_sdpa(query, key, value, *args, **kwargs):
        record(kwargs["attn_mask"], kwargs["is_causal"], query.shape[-2], key.shape[-2])
        return sdpa(query, key, value, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(target.model.rotary_emb.register_forward_pre_hook(rotary, with_kwargs=True))
        if backend == "sdpa":
            stack.enter_context(patch.object(F, "scaled_dot_product_attention", traced_sdpa))
        else:
            require(backend == "eager", f"Unsupported attention observer: {backend}")
            for layer in target.model.layers:
                stack.enter_context(layer.self_attn.register_forward_pre_hook(eager, with_kwargs=True))
        result = target(input_ids=ids, past_key_values=cache, **kwargs)
    return result, {
        "positions": positions,
        "positions_match": positions == [list(range(past_length, past_length + ids.shape[1]))],
        "attention": attention,
        "masks_match": len(attention) == len(target.model.layers)
        and all(item["expected_causal_pattern"] for item in attention),
    }, masks


def logit_summary(logits):
    require(bool(logits.isfinite().all()), "Nonfinite logits in diagnostic")
    return {
        "winner": int(logits.argmax()),
        "tied_max_ids": (logits == logits.max()).nonzero().flatten().tolist(),
        "compared_logits": {str(token): float(logits[token]) for token in TOKEN_IDS if token < logits.numel()},
    }


def native_trace(target, decode, method, prompt, result, audited, output):
    call_index, logit_row = _locate_forward(result, POSITION)
    calls = []
    capture = {}

    def observe(*, input_ids, past_key_values, **kwargs):
        call = len(calls)
        if call > call_index:
            return target(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
        length = 0 if past_key_values is None else past_key_values.get_seq_length()
        focused = call == call_index
        if focused:
            capture["kv"] = snapshot(past_key_values)
            save_tensors(output / f"{method}-retained-kv.pt", capture["kv"])
        prediction, geometry, masks = observed_forward(
            target, input_ids, past_key_values, kwargs, length, save_masks=focused,
        )
        calls.append({
            "input_ids": input_ids[0].tolist(), "cache_before": length,
            "cache_after_forward": prediction.past_key_values.get_seq_length(),
            "logits_to_keep": kwargs.get("logits_to_keep", 0), **geometry,
        })
        if focused:
            capture["logits"] = prediction.logits[0, logit_row].cpu()
            save_tensors(output / f"{method}-prediction.pt", {"logits": capture["logits"], "masks": masks})
        return prediction

    observer = ReplayTarget(observe, result, POSITION, prompt.token_ids, WINNERS)
    try:
        replayed = decode(method, prompt, target=observer)
    except BaseException:
        write_json(output / f"{method}-partial-trace.json", {"calls": calls, "prediction": observer.evidence})
        raise
    require(len(calls) == call_index + 1, f"{method} missed the original prediction")
    for index, entry in enumerate(calls):
        iteration = asdict(result.iterations[index - 1]) if index else None
        entry["iteration"] = iteration
        entry["crop"] = iteration["proposed"] - iteration["accepted"] if iteration else 0
        entry["cache_after_crop"] = (
            calls[index + 1]["cache_before"] if index < call_index else entry["cache_after_forward"] - entry["crop"]
        )
    write_json(output / f"{method}-trace.json", {
        "calls": calls, "prediction": observer.evidence, "logits": logit_summary(capture["logits"]),
        "reproduced": replayed == result,
    })
    require(replayed == result and observer.evidence == audited["prediction"],
            f"{method} instrumented replay differs from the original audit")
    return calls, capture


def history_matches(calls, prefix):
    retained = []
    for call in calls:
        if call["cache_before"] != len(retained):
            return False
        retained.extend(call["input_ids"])
        if call["cache_after_forward"] != len(retained):
            return False
        if call["crop"]:
            del retained[-call["crop"]:]
        if call["cache_after_crop"] != len(retained) or retained != prefix[:len(retained)]:
            return False
    return retained == prefix


def cache_matches(tensors, length):
    return bool(tensors) and all(
        key.shape[-2] == value.shape[-2] == length for key, value in tensors
    )


def tensor_delta(left, right, *, kv=False):
    require(left.shape == right.shape, "Compared tensor shapes differ")
    require(bool(left.isfinite().all() and right.isfinite().all()), "Nonfinite diagnostic tensor")
    unequal = left != right
    absolute = (left.float() - right.float()).abs()
    summary = {
        "different": int(unequal.count_nonzero()),
        "max_abs": float(absolute.max()) if absolute.numel() else 0.0,
    }
    if kv:
        changed = unequal.any(dim=(0, 1, 3)).nonzero().flatten()
        summary["first_position"] = int(changed[0]) if changed.numel() else None
    return summary


def cache_delta(left, right):
    require(len(left) == len(right), "Compared cache layer counts differ")
    return [
        {"key": tensor_delta(key_a, key_b, kv=True), "value": tensor_delta(value_a, value_b, kv=True)}
        for (key_a, value_a), (key_b, value_b) in zip(left, right)
    ]


def crop_and_check(cache, discard, retained):
    expected = [
        (layer.keys[..., :retained, :].clone(), layer.values[..., :retained, :].clone())
        for layer in cache.layers
    ]
    cache.crop(-discard)
    return len(expected) == len(cache.layers) and all(
        torch.equal(key, layer.keys) and torch.equal(value, layer.values)
        for (key, value), layer in zip(expected, cache.layers)
    )


def fixed_history(target, calls, device, *, inspect_call=None):
    """Feed recorded IDs and crop lengths without using any replay argmax."""
    cache = None
    incoming = None
    checks = []
    for index, call in enumerate(calls):
        ids = torch.tensor([call["input_ids"]], device=device)
        if index == inspect_call:
            incoming = snapshot(cache)
        before = 0 if cache is None else cache.get_seq_length()
        kwargs = {"use_cache": True, "logits_to_keep": call["logits_to_keep"]}
        result, geometry, _ = observed_forward(target, ids, cache, kwargs, call["cache_before"])
        cache = result.past_key_values
        after_forward = cache.get_seq_length()
        slice_matches = True
        if call["crop"]:
            slice_matches = crop_and_check(cache, call["crop"], call["cache_after_crop"])
        checks.append(
            before == call["cache_before"] and after_forward == call["cache_after_forward"]
            and all(layer.keys.shape[-2] == layer.values.shape[-2] == call["cache_after_crop"]
                    for layer in cache.layers)
            and slice_matches and geometry["positions_match"] and geometry["masks_match"]
        )
    return result.logits[0, -1].cpu(), incoming, checks


def identical_cache_control(target, tensors, pending, device):
    ids = torch.tensor([[pending]], device=device)
    return [
        target(input_ids=ids, past_key_values=clone_cache(target, tensors, device), use_cache=True)
        .logits[0, -1].cpu()
        for _ in range(2)
    ]


def suffix_control(target, incoming, call, device):
    keep = len(call["input_ids"]) - call["crop"]
    require(0 < keep < len(call["input_ids"]), "Expected a rejected suffix")
    original = call["input_ids"]
    changed = original[:keep] + [(token + 1) % target.config.vocab_size for token in original[keep:]]
    arms = []
    for input_ids in (original, original, changed):
        cache = clone_cache(target, incoming, device)
        output, geometry, masks = observed_forward(
            target, torch.tensor([input_ids], device=device), cache, {"use_cache": True},
            call["cache_before"], save_masks=True,
        )
        cache = output.past_key_values
        retained = call["cache_before"] + keep
        slice_matches = crop_and_check(cache, call["crop"], retained)
        after = snapshot(cache)
        checked = geometry["positions_match"] and geometry["masks_match"] and slice_matches
        # The last retained query predicts the first rejected token.
        arms.append((output.logits[0, :keep].cpu(), after, geometry, masks, checked))

    original_arm, repeated, perturbed = arms

    def difference(other):
        return {
            "logits": tensor_delta(original_arm[0], other[0]),
            "kv": cache_delta(original_arm[1], other[1]),
        }

    repeat_delta, changed_delta = difference(repeated), difference(perturbed)
    comparisons = [(repeat_delta["logits"], changed_delta["logits"])] + [
        (repeat[name], changed[name])
        for repeat, changed in zip(repeat_delta["kv"], changed_delta["kv"])
        for name in ("key", "value")
    ]
    return {
        "original_input_ids": original, "perturbed_input_ids": changed,
        "retained_query_count": keep, "identical_input_delta": repeat_delta, "perturbation_delta": changed_delta,
        "repeat_exact": all(repeat["different"] == 0 for repeat, _ in comparisons),
        "suffix_influence": any(changed["max_abs"] > repeat["max_abs"] for repeat, changed in comparisons),
        "geometry": [arm[2] for arm in arms], "structural_checks": all(arm[4] for arm in arms),
    }, [arm[:2] + (arm[3],) for arm in arms]


def probe_index(calls, first_difference_position):
    rejections = [i for i, call in enumerate(calls[:-1]) if call["crop"]]
    if not rejections:
        return None
    if first_difference_position is None:
        return rejections[-1]
    at_or_before = [i for i in rejections if calls[i]["cache_before"] <= first_difference_position]
    return at_or_before[-1] if at_or_before else rejections[0]


@torch.inference_mode()
def diagnose(output, phase, prompt, settings, eos, case, make_target):
    phase("original_fp16_reproduction")
    target = make_target(torch.float16)
    decode = baseline.make_decoder(target, eos, settings)
    results = [decode(method, prompt) for method in baseline.METHODS]
    generated = [result.token_ids for result in results]
    position = first_difference(*generated)
    write_json(output / "original-outputs.json", {"position": position, "outputs": generated})
    require(position == POSITION and tuple(tokens[POSITION] for tokens in generated) == WINNERS,
            "Original fp16 divergence did not reproduce")
    require(digest(json_bytes(generated)) == case["outputs_sha256"], "Original output digest did not reproduce")

    phase("retained_cache_and_attention")
    traces, captures = zip(*[
        native_trace(target, decode, method, prompt, result, case[name], output)
        for method, result, name in zip(baseline.METHODS, results, ("reference_replay", "output_replay"))
    ])
    prefix = prompt.token_ids + generated[0][:POSITION]
    pending = prefix[-1]
    structure = all(history_matches(trace, prefix) for trace in traces) and all(
        len(capture["kv"]) == len(target.model.layers) and cache_matches(capture["kv"], CACHE_LENGTH)
        for capture in captures
    ) and all(call["positions_match"] and call["masks_match"] for trace in traces for call in trace)
    write_json(output / "native-invariants.json", {"prefix": prefix, "structural_checks": structure})
    if not structure:
        return "structural_invariant_violation"
    kv = cache_delta(captures[0]["kv"], captures[1]["kv"])
    first_position = min(
        (item["first_position"] for layer in kv for item in layer.values() if item["first_position"] is not None),
        default=None,
    )
    selected = probe_index(traces[1], first_position)
    first_verification = next((call["cache_before"] for call in traces[1][1:]
                               if len(call["input_ids"]) > 1), None)
    write_json(output / "retained-kv-comparison.json", {
        "layers": kv, "first_differing_position": first_position, "lookup_rejection_call": selected,
        "first_lookup_verification_position": first_verification,
    })

    phase("identical_cache_fp16")
    repeats = [identical_cache_control(target, capture["kv"], pending, prompt.tensor.device) for capture in captures]
    repeat_exact = all(torch.equal(*arm) for arm in repeats)
    follows_cache = all(torch.equal(arm[0], capture["logits"]) for arm, capture in zip(repeats, captures))
    save_tensors(output / "identical-cache-logits.pt", repeats)
    write_json(output / "identical-cache.json", {
        "repeat_exact": repeat_exact, "follows_cache": follows_cache,
        "logits": [[logit_summary(row) for row in arm] for arm in repeats],
        "repeat_deltas": [tensor_delta(*arm) for arm in repeats],
        "cross_cache_delta": tensor_delta(repeats[0][0], repeats[1][0]),
    })

    controls = {}
    for dtype, label in ((torch.float16, "fp16"), (torch.float32, "fp32")):
        if dtype == torch.float32:
            del decode, target
            phase("fixed_history_fp32")
            target = make_target(dtype)
        else:
            phase("fixed_history_fp16")
        logits, _, greedy_checks = fixed_history(target, traces[0], prompt.tensor.device)
        lookup, incoming, lookup_checks = fixed_history(
            target, traces[1], prompt.tensor.device, inspect_call=selected,
        )
        rows = [logits, lookup]
        if dtype == torch.float32:
            rows.append(target(
                input_ids=torch.tensor([prefix], device=prompt.tensor.device), use_cache=False, logits_to_keep=1,
            ).logits[0, -1].cpu())
        checks = {
            "structure": all(greedy_checks + lookup_checks),
            "greedy_checks": greedy_checks, "lookup_checks": lookup_checks,
            "logits": [logit_summary(row) for row in rows],
            "logit_deltas_from_greedy": [tensor_delta(rows[0], row) for row in rows[1:]],
        }
        if dtype == torch.float16:
            checks["matches_native"] = [torch.equal(row, capture["logits"]) for row, capture in zip(rows, captures)]
        else:
            # Match torch.testing.assert_close's float32 defaults, also used by
            # the CPU fixed-history test. The absolute term covers near-zero
            # logits and the relative term scales with the reference logit.
            rtol, atol = 1.3e-6, 1e-5
            pairs = {
                "lookup_vs_greedy": (rows[1], rows[0]),
                "greedy_vs_uncached": (rows[0], rows[2]),
                "lookup_vs_uncached": (rows[1], rows[2]),
            }
            checks["full_logit_comparison"] = {
                "rtol": rtol, "atol": atol,
                "pairwise_close": {
                    name: torch.allclose(actual, reference, rtol=rtol, atol=atol)
                    for name, (actual, reference) in pairs.items()
                },
                "lookup_delta_from_uncached": tensor_delta(rows[1], rows[2]),
            }
        save_tensors(output / f"{label}-fixed-history-logits.pt", rows)
        write_json(output / f"{label}-fixed-history.json", checks)
        if selected is not None:
            phase(f"rejected_suffix_{label}")
            checks["suffix"], tensors = suffix_control(target, incoming, traces[1][selected], prompt.tensor.device)
            save_tensors(output / f"{label}-suffix-tensors.pt", tensors)
            write_json(output / f"{label}-suffix.json", checks["suffix"])
        controls[label] = checks

    fp32 = controls["fp32"]
    if not all(fp32["full_logit_comparison"]["pairwise_close"].values()):
        return "unresolved"
    if any(
        not control["structure"]
        or ("suffix" in control and (
            not control["suffix"]["structural_checks"]
            or (control["suffix"]["repeat_exact"] and control["suffix"]["suffix_influence"])
        ))
        for control in controls.values()
    ):
        return "structural_invariant_violation"
    fp32_winners_agree = len({row["winner"] for row in fp32["logits"]}) == 1
    fp16_reproduces = all(controls["fp16"]["matches_native"])
    if (first_position is not None and first_verification is not None and first_position >= first_verification
            and repeat_exact and follows_cache and fp32_winners_agree and fp16_reproduces
            and all(control.get("suffix", {}).get("repeat_exact") for control in controls.values())):
        return "numerical_drift_supported_dev_review"
    return "unresolved"


def execute(args, script, output, phase):
    checkout = args.checkout.resolve()
    baseline.verify_checkout(checkout, args.git_sha)
    require(baseline.ROOT == checkout, "Runner import did not come from the audited checkout")
    config = (checkout / "configs/bench.yaml").read_bytes()
    settings = baseline.read_settings(config)
    records = baseline.read_frozen(checkout / "bench/prompts.jsonl", settings)
    phase("original_evidence")
    files, manifest, archive_hash = load_source(args.source_experiments, args.source_archive)
    environment = baseline.validate_environment()
    identity = {
        "git_commit": args.git_sha, "benchmark_sha256": baseline.PROMPTS_SHA256, "config_sha256": digest(config),
        "settings": settings, "environment": environment,
    }
    case, original_generation_config = validate_audit(files, manifest, identity, config, records)
    write_json(output / "provenance.json", {
        "identity": identity, "original_run_id": manifest["run_id"], "case_id": case["case_id"],
        "target_runtime": manifest["target_runtime"], "script_sha256": digest(script.read_bytes()),
        "source_archive_sha256": archive_hash, "source_file_sha256": {name: digest(data) for name, data in files.items()},
    })
    for name, data in files.items():
        atomic_write(output / "original" / "experiments" / name, data)

    phase("model_preflight")
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    tokenizer, generation_config = baseline.preflight_tokenizer_and_generation()
    require(generation_config.to_diff_dict() == original_generation_config, "Original generation config differs")
    record = next(record for record in records if record["id"] == PROMPT_ID)
    prompt, = baseline.tokenize_prompts([record], tokenizer, "cuda:0")

    def make_target(dtype):
        target = LlamaForCausalLM.from_pretrained(
            baseline.TARGET_MODEL_ID, revision=baseline.TARGET_REVISION, dtype=dtype, use_safetensors=True,
        ).to("cuda:0").eval()
        require(target.config._commit_hash == baseline.TARGET_REVISION and target.dtype == dtype,
                "Loaded target revision or dtype differs from the pin")
        baseline.validate_generation_config(target.generation_config, tokenizer)
        require(
            target.config._attn_implementation == manifest["target_runtime"][str(dtype)]["attention_implementation"],
            "Loaded attention implementation differs from original run",
        )
        return target

    return diagnose(output, phase, prompt, settings, tokenizer.eos_token_id, case, make_target)


def main(args):
    script = Path(__file__).resolve()
    checkout = args.checkout.resolve()
    output = args.output.resolve()
    require(not script.is_relative_to(checkout), "Stage the diagnostic script outside the audited checkout")
    require(not output.is_relative_to(checkout), "Diagnostic output must be outside the audited checkout")
    if args.source_experiments is not None:
        require(not output.is_relative_to(args.source_experiments.resolve()),
                "Diagnostic output must be outside the original experiments directory")
    persist_run(output, script, lambda phase: execute(args, script, output, phase))


if __name__ == "__main__":
    main(args)
