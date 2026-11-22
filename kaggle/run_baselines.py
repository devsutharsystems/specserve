"""Run the approved Day 2b-2 baselines on one Kaggle T4. See kaggle/README.md."""

import argparse
import fcntl
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from uuid import uuid4

# This entry point is run in a new process, before any CUDA initialization.
ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from transformers import GenerationConfig, LlamaForCausalLM

from kaggle.baseline_artifacts import (
    ITERATION_FIELDS, ITERATIONS, TIMING, complete_unit, digest, load_repeat, open_run,
    repeat_path, save_repeat,
    stop_reason, throughput, write_correctness, write_csv, write_json, write_summaries,
)
from kaggle.baseline_audit import (
    check_row, first_five_per_category, fp16_comparison, fp32_gate,
)
from specserve.decode.greedy import DecodeIteration, DecodeResult, greedy_decode
from specserve.decode.prompt_lookup import prompt_lookup_decode
from specserve.decode.prompting import TARGET_MODEL_ID, TARGET_REVISION, load_tokenizer, prompt_token_ids


PROMPTS_SHA256 = "f8d753325a1cc4e52d809e311e47f8182474c6c1504f4c29cd977d1f91289c36"
CATEGORIES = ("chat_qa", "summarization_extraction", "open_ended_writing", "code")
METHODS = ("greedy", "prompt_lookup")
LOCKED_SETTINGS = {
    "target_model_id": TARGET_MODEL_ID, "model_revision": TARGET_REVISION,
    "seed": 20261118, "batch_size": 1, "max_new_tokens": 128, "warmups": 1,
    "warmup_prompt_id": "chat_qa-01", "repeats": 3, "max_matching_ngram_size": 3,
    "prompt_lookup_num_tokens": 8,
}
KAGGLE_WORKING = Path("/kaggle/working")


@dataclass
class Prompt:
    id: str
    category: str
    budget: int
    token_ids: list[int]
    tensor: torch.Tensor


def read_settings(config_bytes):
    settings = yaml.safe_load(config_bytes)
    if not isinstance(settings, dict) or set(settings) != set(LOCKED_SETTINGS):
        raise ValueError("Run YAML must contain exactly the approved settings")
    for key, expected in LOCKED_SETTINGS.items():
        if type(settings[key]) is not type(expected) or settings[key] != expected:
            raise ValueError(f"Run YAML requires {key}={expected!r}")
    return settings


def verify_checkout(root, expected_sha):
    if not re.fullmatch(r"[0-9a-f]{40}", expected_sha):
        raise ValueError("--git-sha requires the full lowercase 40-character pushed commit SHA")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    if git("rev-parse", "HEAD") != expected_sha:
        raise RuntimeError("HEAD does not match the requested Git SHA")
    if git("rev-parse", "--abbrev-ref", "HEAD") != "HEAD":
        raise RuntimeError("Check out the supplied SHA in detached HEAD")
    if git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("The Git checkout must be clean")
    origin_refs = git(
        "for-each-ref", "--contains", expected_sha, "--format=%(refname)", "refs/remotes/origin/"
    ).splitlines()
    if not any(ref != "refs/remotes/origin/HEAD" for ref in origin_refs):
        raise RuntimeError("The SHA must be reachable from a fetched origin branch")
    return expected_sha


def read_frozen(path, settings):
    data = path.read_bytes()
    if digest(data) != PROMPTS_SHA256:
        raise ValueError("Frozen benchmark SHA-256 mismatch")
    records = [json.loads(line) for line in data.splitlines()]
    categories = Counter(record["category"] for record in records)
    if categories != dict.fromkeys(CATEGORIES, 25):
        raise ValueError("Frozen benchmark must have 25 records per category")
    seen = Counter()
    for record in records:
        category = record["category"]
        seen[category] += 1
        if record["id"] != f"{category}-{seen[category]:02d}":
            raise ValueError("Unexpected frozen ID or category ordering")
        if record["max_new_tokens"] != settings["max_new_tokens"]:
            raise ValueError("Unexpected frozen generation budget")
    return records


def tokenize_prompts(records, tokenizer, device):
    prompts = []
    for record in records:
        tokens = prompt_token_ids(tokenizer, record["prompt"])
        if (
            len(tokens) != record["prompt_tokens"] or len(tokens) > 896
            or len(tokens) + record["max_new_tokens"] > 1024
        ):
            raise ValueError(f"Frozen token count mismatch: {record['id']}")
        prompts.append(Prompt(
            record["id"], record["category"], record["max_new_tokens"], tokens,
            torch.tensor([tokens], device=device, dtype=torch.long),
        ))
    return prompts


def validate_generation_config(config, tokenizer):
    # All behavioral settings besides the pinned special IDs and default
    # max_length must be neutral. Each prompt supplies max_new_tokens to HF.
    defaults = GenerationConfig()
    pinned = {
        "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0, "max_length": 2048,
    }
    if (tokenizer.bos_token_id, tokenizer.eos_token_id) != (1, 2):
        raise ValueError("Unexpected pinned tokenizer BOS/EOS")
    for key in config.to_dict().keys() | defaults.to_dict().keys() | pinned.keys():
        if key.startswith("_") or key == "transformers_version":
            continue
        expected = pinned[key] if key in pinned else getattr(defaults, key, None)
        if getattr(config, key, None) != expected:
            raise ValueError(f"Unapproved generation setting: {key}")


def preflight_tokenizer_and_generation():
    tokenizer = load_tokenizer()
    config_file = hf_hub_download(TARGET_MODEL_ID, "tokenizer_config.json", revision=TARGET_REVISION)
    template = json.loads(Path(config_file).read_text())["chat_template"]
    example = "What is 2 + 2?"
    if (
        tokenizer.chat_template != template
        or tokenizer.apply_chat_template(
            [{"role": "user", "content": example}], tokenize=False, add_generation_prompt=True
        ) != "<|user|>\nWhat is 2 + 2?</s>\n<|assistant|>\n"
        or prompt_token_ids(tokenizer, example)[:5] != [1, 529, 29989, 1792, 29989]
    ):
        raise ValueError("Pinned tokenizer template/BOS verification failed")
    config = GenerationConfig.from_pretrained(TARGET_MODEL_ID, revision=TARGET_REVISION)
    validate_generation_config(config, tokenizer)
    return tokenizer, config


def validate_environment():
    torch_version = str(torch.__version__)
    versions = {package: version(package) for package in (
        "transformers", "tokenizers", "huggingface-hub", "numpy", "PyYAML", "datasets", "pytest",
    )}
    if (
        torch_version != "2.11.0+cu128" or torch.version.cuda != "12.8"
        or versions["transformers"] != "5.16.1" or versions["tokenizers"] != "0.23.3"
    ):
        raise RuntimeError(
            f"Expected torch 2.11.0+cu128, CUDA 12.8, Transformers 5.16.1 and tokenizers 0.23.3. "
            f"Found {torch_version}, {torch.version.cuda}, "
            f'{versions["transformers"]}, {versions["tokenizers"]}.'
        )
    visible = torch.cuda.device_count()
    if not torch.cuda.is_available() or visible != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Exactly one masked CUDA GPU must be visible")
    props = torch.cuda.get_device_properties(0)
    if props.name not in {"Tesla T4", "NVIDIA T4"} or (props.major, props.minor) != (7, 5):
        raise RuntimeError(f"Expected a T4, found {props.name}")
    drivers = set(subprocess.check_output([
        "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader",
    ], text=True).split())
    if len(drivers) != 1:
        raise RuntimeError("Could not identify a single NVIDIA driver version")
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch_version, "cuda": torch.version.cuda, "driver": drivers.pop(),
        "packages": versions, "cuda_visible_devices": "0", "visible_device_count": visible,
        "device_index": 0, "device_name": props.name, "total_memory_bytes": props.total_memory,
        "compute_capability": [props.major, props.minor],
        "multiprocessors": props.multi_processor_count, "cpu": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "sdpa_backends": {
            "math": torch.backends.cuda.math_sdp_enabled(),
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        },
    }


def make_decoder(target, eos, settings):
    def decode(method, prompt, *, target=target, on_prefill=None):
        kwargs = {
            "max_new_tokens": prompt.budget, "eos_token_id": eos, "on_prefill": on_prefill,
        }
        if method == "greedy":
            return greedy_decode(target, prompt.tensor, **kwargs)
        return prompt_lookup_decode(
            target, prompt.tensor, **kwargs,
            max_matching_ngram_size=settings["max_matching_ngram_size"],
            prompt_lookup_num_tokens=settings["prompt_lookup_num_tokens"],
        )
    return decode


def measure(decode, method, prompt, synchronize, clock=perf_counter):
    boundary = None

    def on_prefill():
        nonlocal boundary
        synchronize()
        boundary = clock()

    synchronize()
    start = clock()
    result = decode(method, prompt, on_prefill=on_prefill)
    synchronize()
    end = clock()
    if boundary is None:
        raise RuntimeError("Positive generation budget requires a prefill boundary")
    return result, {
        "prefill_s": boundary - start,
        "decode_s": end - boundary if result.iterations else 0.0,
        "total_s": end - start,
    }


def _measured_unit(result, durations, key, expected):
    durations = {field: value if value is None or math.isfinite(value) else str(value)
                 for field, value in durations.items()}
    tokens = result.token_ids
    rate, undefined = throughput(len(tokens), durations["decode_s"])
    return {
        "timing": {
            **key, **expected, **durations, "tokens_generated": len(tokens),
            "stopped_on_eos": result.stopped_on_eos,
            "stop_reason": stop_reason(tokens, expected["eos_token_id"], expected["max_new_tokens"]),
            "decode_tok_per_s": rate, "throughput_undefined_reason": undefined,
        },
        "iterations": [
            {**key, "iteration": index, **asdict(row)} for index, row in enumerate(result.iterations)
        ],
        "token_ids": tokens,
    }


def run_measurements(prompts, settings, manifest, directory, session, decode, eos,
                     synchronize, persist, audit_repeat, clock=perf_counter):
    units = []
    identity = manifest["identity"]
    warmup = next(prompt for prompt in prompts if prompt.id == settings["warmup_prompt_id"])
    for method in METHODS:
        pending = []
        for prompt in prompts:
            expected = {
                "category": prompt.category, "batch_size": settings["batch_size"],
                "max_new_tokens": prompt.budget, "eos_token_id": eos, "dtype": "fp16",
                "seed": settings["seed"], "git_commit": identity["git_commit"],
                "model_revision": settings["model_revision"],
            }
            for repeat in range(1, settings["repeats"] + 1):
                key = {"config_id": identity["config_sha256"], "method": method,
                       "prompt_id": prompt.id, "repeat": repeat}
                path = repeat_path(directory, key)
                unit = load_repeat(path, manifest["run_id"], key, expected)
                pending.append((prompt, key, expected, path, unit))
        if any(unit is None for _, _, _, _, unit in pending):
            for _ in range(settings["warmups"]):
                decode(method, warmup)
                synchronize()
                session["warmups"][method] = session["warmups"].get(method, 0) + 1
                persist()
        for prompt, key, expected, path, unit in pending:
            if unit is None:
                result, durations = measure(decode, method, prompt, synchronize, clock)
                unit = _measured_unit(result, durations, key, expected)
                if not complete_unit(unit, key, expected):
                    audit_repeat(method, prompt, key["repeat"], result)
                    raise RuntimeError(f"Incomplete measured repeat: {key}")
                save_repeat(path, manifest["run_id"], unit)
            else:
                result = DecodeResult(
                    unit["token_ids"],
                    [DecodeIteration(**{field: row[field] for field in ITERATION_FIELDS})
                     for row in unit["iterations"]],
                    unit["timing"]["stopped_on_eos"],
                )
                session["reused_repeats"] += 1
            audit_repeat(method, prompt, key["repeat"], result)
            units.append(unit)
    return units


def benchmark(prompts, settings, manifest, directory, session, make_target, generation_config,
              eos, reviews, audit, persist, synchronize, clock=perf_counter):
    session["status"] = "fp32_gates"
    target = make_target(torch.float32)
    decode = make_decoder(target, eos, settings)

    @torch.inference_mode()
    def hf(prompt):
        return target.generate(
            prompt.tensor, generation_config=generation_config, do_sample=False,
            max_new_tokens=prompt.budget,
        )[0, prompt.tensor.shape[1]:].tolist()

    fp32_gate(
        (("fp32_first_20", first_five_per_category(prompts)), ("fp32_full", prompts)),
        hf, decode, audit, persist,
    )
    del target, decode, hf

    target = make_target(torch.float16)
    decode = make_decoder(target, eos, settings)
    references = {}
    session["status"] = "fp16_audit"
    for prompt in prompts:
        reference = decode("greedy", prompt)
        references[prompt.id] = reference
        audit.append(fp16_comparison(
            "fp16_untimed", "prompt_lookup", prompt, reference, decode("prompt_lookup", prompt),
            target=target, decode=decode, eos=eos, settings=settings,
            reviews=reviews, identity=manifest["identity"],
        ))
    persist()
    if any(row["status"] == "fail" for row in audit):
        raise RuntimeError("Untimed fp16 audit has unresolved divergences. See correctness.csv.")

    def audit_repeat(method, prompt, repeat, result):
        audit.append(fp16_comparison(
            "fp16_measured", method, prompt, references[prompt.id], result, target=target,
            decode=decode, eos=eos, settings=settings, reviews=reviews,
            identity=manifest["identity"], repeat=repeat,
        ))

    session["status"] = "measuring"
    persist()
    return run_measurements(
        prompts, settings, manifest, directory, session, decode, eos, synchronize,
        persist, audit_repeat, clock,
    )


def read_reviews(path):
    if path is None:
        return {}
    reviews = json.loads(path.read_bytes())
    if not isinstance(reviews, dict):
        raise ValueError("Reviews must map case IDs to verdicts and written diagnoses")
    for case_id, review in reviews.items():
        if (
            not re.fullmatch(r"[0-9a-f]{64}", case_id) or not isinstance(review, dict)
            or set(review) != {"verdict", "diagnosis"}
            or review["verdict"] not in {"near_tie", "unexplained"}
            or not isinstance(review["diagnosis"], str) or not review["diagnosis"].strip()
        ):
            raise ValueError("Invalid review entry")
    return reviews


def execute(output, git_sha, reviews_path):
    git_sha = verify_checkout(ROOT, git_sha)
    config_bytes = (ROOT / "configs/bench.yaml").read_bytes()
    settings = read_settings(config_bytes)
    records = read_frozen(ROOT / "bench/prompts.jsonl", settings)
    environment = validate_environment()
    identity = {
        "git_commit": git_sha, "benchmark_sha256": PROMPTS_SHA256,
        "config_sha256": digest(config_bytes), "settings": settings, "environment": environment,
    }
    directory = output / "experiments"
    baseline = directory / "baseline"
    manifest = open_run(directory, identity, config_bytes, uuid4().hex)
    reviews = read_reviews(reviews_path)
    audit = [check_row("integrity", None, None, status="pass", benchmark_sha256=PROMPTS_SHA256)]
    session = {
        "id": uuid4().hex, "status": "preflight", "warmups": {}, "reused_repeats": 0,
        "gpu_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "")),
    }
    audits = baseline / "audits"
    prior_fp16_divergence = any(audits.glob("*.csv")) or any(
        any(previous.get("fp16_divergences", {}).values()) for previous in manifest["sessions"]
    )
    manifest["sessions"].append(session)
    manifest["speed_results_accepted"] = False

    def fp16_divergences():
        return {
            stage: sum(row["stage"] == stage and "case_id" in row["details"] for row in audit)
            for stage in ("fp16_untimed", "fp16_measured")
        }

    def persist():
        session["fp16_divergences"] = fp16_divergences()
        has_divergence = any(session["fp16_divergences"].values())
        manifest["requires_dev_review"] = prior_fp16_divergence or has_divergence
        if has_divergence:
            write_correctness(audits / f'{session["id"]}.csv', audit)
        write_correctness(baseline / "correctness.csv", audit)
        write_json(baseline / "run.json", manifest)

    persist()
    (baseline / "summary.json").unlink(missing_ok=True)
    (baseline / "iteration_summary.csv").unlink(missing_ok=True)
    try:
        random.seed(settings["seed"])
        np.random.seed(settings["seed"])
        torch.manual_seed(settings["seed"])
        torch.cuda.manual_seed_all(settings["seed"])
        tokenizer, generation_config = preflight_tokenizer_and_generation()
        prompts = tokenize_prompts(records, tokenizer, "cuda:0")
        eos = tokenizer.eos_token_id
        audit.append(check_row(
            "tokenizer_and_generation", None, None, status="pass",
            token_counts_checked=len(prompts), revision=TARGET_REVISION,
            generation_config=generation_config.to_diff_dict(),
        ))

        def make_target(dtype):
            target = LlamaForCausalLM.from_pretrained(
                TARGET_MODEL_ID, revision=TARGET_REVISION, dtype=dtype, use_safetensors=True,
            ).to("cuda:0").eval()
            if target.config._commit_hash != TARGET_REVISION or target.dtype != dtype:
                raise RuntimeError("Loaded model revision or dtype differs from the pin")
            validate_generation_config(target.generation_config, tokenizer)
            runtime = {"attention_implementation": target.config._attn_implementation}
            saved_runtime = manifest.setdefault("target_runtime", {}).setdefault(str(dtype), runtime)
            if saved_runtime != runtime:
                raise RuntimeError(f"Resume is incompatible: {dtype} runtime settings")
            session[str(dtype)] = runtime
            return target

        units = benchmark(
            prompts, settings, manifest, baseline / "repeats", session, make_target,
            generation_config, eos, reviews, audit, persist, torch.cuda.synchronize,
        )
        if len(units) != len(METHODS) * len(prompts) * settings["repeats"]:
            raise RuntimeError("Measured repeat count is incomplete")
        write_csv(baseline / "timing.csv", TIMING, (unit["timing"] for unit in units))
        write_csv(baseline / "iterations.csv", ITERATIONS, (
            row for unit in units for row in unit["iterations"]
        ))
        session["complete_repeats"] = len(units)
        if any(row["status"] == "fail" for row in audit):
            raise RuntimeError("Measured fp16 audit has unresolved divergences. See correctness.csv.")
        write_summaries(baseline, units)
        if prior_fp16_divergence or any(fp16_divergences().values()):
            session["status"] = "requires_dev_review"
        else:
            session["status"] = "accepted"
            manifest["speed_results_accepted"] = True
    except BaseException as error:
        stage = session["status"]
        session["status"] = "blocked" if isinstance(error, RuntimeError) else "interrupted_or_error"
        session["failed_stage"] = stage
        if not any(row["status"] == "fail" for row in audit):
            audit.append(check_row(stage, None, None, status="fail", error_type=type(error).__name__))
        raise
    finally:
        persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-sha", required=True, help="Full pushed, detached commit SHA")
    parser.add_argument("--output-root", type=Path, default=KAGGLE_WORKING / "specserve-day2b2")
    parser.add_argument("--fp16-reviews", type=Path, help="Optional evidence-bound review JSON")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if not output.is_relative_to(KAGGLE_WORKING) or output == KAGGLE_WORKING or output.is_relative_to(ROOT):
        parser.error("--output-root must be under /kaggle/working, outside the checkout")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(output, args.git_sha, args.fp16_reviews)
    print(f"Export the complete directory {output / 'experiments'}")


if __name__ == "__main__":
    main()
