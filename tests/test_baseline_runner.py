import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import GenerationConfig

from kaggle import run_baselines as runner
from kaggle.baseline_artifacts import (
    ITERATIONS, JOIN, TIMING, digest, iteration_metrics, load_repeat, open_run, repeat_path,
    save_repeat, summarize, throughput, write_csv,
)
from kaggle.baseline_audit import first_difference, first_five_per_category, fp16_comparison
from specserve.decode.greedy import DecodeIteration, DecodeResult
from specserve.decode.prompting import load_tokenizer


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config_bytes():
    return (ROOT / "configs/bench.yaml").read_bytes()


@pytest.fixture
def settings(config_bytes):
    return runner.read_settings(config_bytes)


def test_frozen_integrity_and_token_counts(settings, tmp_path):
    original = ROOT / "bench/prompts.jsonl"
    records = runner.read_frozen(original, settings)
    tokenizer = load_tokenizer()
    prompts = runner.tokenize_prompts(records, tokenizer, "cpu")
    assert [prompt.id for prompt in prompts] == [record["id"] for record in records]
    assert [prompt.id for prompt in first_five_per_category(prompts)] == [
        f"{category}-{index:02d}" for category in runner.CATEGORIES for index in range(1, 6)
    ]
    bad = tmp_path / "prompts.jsonl"
    bad.write_bytes(original.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        runner.read_frozen(bad, settings)
    bad_record = {**records[0], "prompt_tokens": 1}
    with pytest.raises(ValueError, match=records[0]["id"]):
        runner.tokenize_prompts([bad_record], tokenizer, "cpu")


@pytest.mark.parametrize(("old", "new"), [
    (b"batch_size: 1", b"batch_size: true"),
    (b"warmups: 1", b"warmups: 0"),
    (b"repeats: 3", b"repeats: 2"),
    (b"prompt_lookup_num_tokens: 8", b"prompt_lookup_num_tokens: 7"),
])
def test_locked_settings(config_bytes, old, new):
    with pytest.raises(ValueError, match="Run YAML"):
        runner.read_settings(config_bytes.replace(old, new))


@pytest.fixture
def generation_config():
    return GenerationConfig(bos_token_id=1, eos_token_id=2, pad_token_id=0, max_length=2048)


@pytest.mark.parametrize("option", [
    "forced_eos_token_id", "min_new_tokens", "repetition_penalty", "cache_implementation",
])
def test_generation_config(generation_config, option):
    tokenizer = SimpleNamespace(bos_token_id=1, eos_token_id=2)
    runner.validate_generation_config(generation_config, tokenizer)
    setattr(generation_config, option, -1)
    with pytest.raises(ValueError, match=option):
        runner.validate_generation_config(generation_config, tokenizer)


def test_checkout_verification(monkeypatch, tmp_path):
    sha = "1" * 40
    outputs = {
        "rev-parse HEAD": sha,
        "rev-parse --abbrev-ref HEAD": "HEAD",
        "status --porcelain --untracked-files=all": "",
        f"for-each-ref --contains {sha} --format=%(refname) refs/remotes/origin/":
            "refs/remotes/origin/main",
    }
    monkeypatch.setattr(runner.subprocess, "check_output",
                        lambda args, **kwargs: outputs[" ".join(args[1:])])
    assert runner.verify_checkout(tmp_path, sha) == sha
    with pytest.raises(ValueError, match="full lowercase"):
        runner.verify_checkout(tmp_path, sha[:7])
    with pytest.raises(RuntimeError, match="does not match"):
        runner.verify_checkout(tmp_path, "2" * 40)
    outputs["rev-parse --abbrev-ref HEAD"] = "main"
    with pytest.raises(RuntimeError, match="detached"):
        runner.verify_checkout(tmp_path, sha)
    outputs["rev-parse --abbrev-ref HEAD"] = "HEAD"
    outputs["status --porcelain --untracked-files=all"] = "?? untracked"
    with pytest.raises(RuntimeError, match="clean"):
        runner.verify_checkout(tmp_path, sha)
    outputs["status --porcelain --untracked-files=all"] = ""
    outputs[f"for-each-ref --contains {sha} --format=%(refname) refs/remotes/origin/"] = ""
    with pytest.raises(RuntimeError, match="fetched origin branch"):
        runner.verify_checkout(tmp_path, sha)


def prompt(prompt_id="chat_qa-01", budget=3, tokens=None):
    tokens = [1, 3] if tokens is None else tokens
    return runner.Prompt(prompt_id, "chat_qa", budget, tokens, torch.tensor([tokens]))


def normal_result():
    return DecodeResult([3, 4, 2], [
        DecodeIteration(0, 0, False, False, 1, 1),
        DecodeIteration(0, 0, False, True, 1, 1),
    ], True)


@pytest.fixture
def run_identity(config_bytes, settings):
    return {
        "git_commit": "1" * 40, "benchmark_sha256": runner.PROMPTS_SHA256,
        "config_sha256": digest(config_bytes), "settings": settings,
        "environment": {"torch": "cpu-fixture", "device": "simulated"},
    }


@pytest.mark.parametrize(("fail_dtype", "fail_method", "expected_stage"), [
    (torch.float32, "greedy", "fp32_first_20"),
    (torch.float32, "prompt_lookup", "fp32_first_20"),
    (torch.float16, "prompt_lookup", "fp16_untimed"),
    (None, None, None),
])
def test_ordered_gates_and_timing_block(
    settings, tmp_path, run_identity, generation_config, monkeypatch, fail_dtype,
    fail_method, expected_stage,
):
    calls = []
    audit = []
    prompts = [prompt(f"chat_qa-{index:02d}") for index in range(1, 7)]

    def target_for(dtype):
        calls.append(dtype)
        return SimpleNamespace(
            dtype=dtype, generate=lambda inputs, **kwargs: torch.cat(
                (inputs, torch.tensor([[3, 4, 2]])), dim=1
            ),
        )

    def make_decoder(target, eos, settings):
        def decode(method, prompt, **kwargs):
            if target.dtype == fail_dtype and method == fail_method:
                return DecodeResult([3, 4], [], False)
            return normal_result()
        return decode

    def timed(*_args):
        calls.append("timing")
        return []

    monkeypatch.setattr(runner, "make_decoder", make_decoder)
    monkeypatch.setattr(runner, "run_measurements", timed)
    args = (
        prompts, settings, {"identity": run_identity}, tmp_path, {}, target_for,
        generation_config, 2, {}, audit, lambda: None, lambda: None,
    )
    if expected_stage:
        with pytest.raises(RuntimeError):
            runner.benchmark(*args)
        assert [row["stage"] for row in audit if row["status"] == "fail"][0] == expected_stage
        assert "timing" not in calls
        if fail_dtype == torch.float32:
            assert torch.float16 not in calls
            assert all(row["stage"] == "fp32_first_20" for row in audit)
    else:
        runner.benchmark(*args)
        assert calls == [torch.float32, torch.float16, "timing"]
        assert [(row["stage"], row["method"], row["prompt_id"]) for row in audit] == [
            (stage, method, p.id)
            for stage, selected, methods in (
                ("fp32_first_20", prompts[:5], runner.METHODS),
                ("fp32_full", prompts, runner.METHODS),
                ("fp16_untimed", prompts, ("prompt_lookup",)),
            )
            for method in methods for p in selected
        ]


def test_measure_includes_first_token_and_sync():
    events = []
    times = iter([1.0, 3.0, 7.0])

    def decode(method, prompt, *, on_prefill):
        events.append("first_token")
        on_prefill()
        events.append("decode")
        return normal_result()

    result, durations = runner.measure(
        decode, "greedy", prompt(), lambda: events.append("sync"), lambda: next(times),
    )
    assert events == ["sync", "first_token", "sync", "decode", "sync"]
    assert durations == {"prefill_s": 2.0, "decode_s": 4.0, "total_s": 6.0}
    assert throughput(len(result.token_ids), durations["decode_s"]) == (0.5, None)

    def eos_on_prefill(method, prompt, *, on_prefill):
        on_prefill()
        return DecodeResult([2], [], True)

    result, durations = runner.measure(
        eos_on_prefill, "greedy", prompt(), lambda: None, iter([1.0, 3.0, 7.0]).__next__,
    )
    assert durations == {"prefill_s": 2.0, "decode_s": 0.0, "total_s": 6.0}
    assert throughput(len(result.token_ids), durations["decode_s"]) == (None, "no_tokens_after_prefill")


@pytest.mark.parametrize("duration", [None, 0, -1, float("inf"), float("nan")])
def test_invalid_decode_duration(duration):
    rate, reason = throughput(3, duration)
    assert rate is None and reason


def test_pooling_counts_and_excludes_only_final_rows():
    rows = [
        {"accepted": 2, "mismatch": True, "stopped": False, "tokens_emitted": 3, "target_passes": 1},
        {"accepted": 0, "mismatch": True, "stopped": False, "tokens_emitted": 1, "target_passes": 2},
        {"accepted": 4, "mismatch": False, "stopped": True, "tokens_emitted": 4, "target_passes": 1},
    ]
    assert iteration_metrics(rows) == {
        "accepted": 2, "trials": 4, "tokens_emitted": 4, "target_passes": 3,
        "retained_rows": 2, "excluded_final_rows": 1, "alpha_hat": 0.5,
        "alpha_undefined_reason": None, "tokens_per_target_pass": 4 / 3,
        "tokens_per_pass_undefined_reason": None,
    }
    assert iteration_metrics(rows[-1:])["excluded_final_rows"] == 1
    empty = iteration_metrics([])
    assert empty["excluded_final_rows"] == 0
    assert empty["alpha_hat"] is None and empty["tokens_per_target_pass"] is None


def test_resume_incompatible_hashes_yaml_env(tmp_path, config_bytes, run_identity):
    initial = open_run(tmp_path, run_identity, config_bytes, "run")
    assert json.loads((tmp_path / "env.txt").read_bytes()) == run_identity
    assert open_run(tmp_path, run_identity, config_bytes, "new") == initial
    for field in ("git_commit", "benchmark_sha256", "config_sha256", "settings", "environment"):
        with pytest.raises(RuntimeError, match=field):
            open_run(tmp_path, {**run_identity, field: None}, config_bytes, "new")
    with pytest.raises(RuntimeError, match="exact checkout YAML"):
        open_run(tmp_path, run_identity, config_bytes + b"\n", "new")
    (tmp_path / "env.txt").unlink()
    with pytest.raises(RuntimeError, match="environment metadata"):
        open_run(tmp_path, run_identity, config_bytes, "new")


def test_three_repeat_joins_warmups_resume_and_atomic_recovery(tmp_path, settings, run_identity):
    manifest = {"identity": run_identity, "run_id": "run"}
    first = prompt()
    second = prompt("chat_qa-02")
    calls = []
    audits = []

    def decode(method, prompt, *, on_prefill=None):
        calls.append((method, prompt.id, on_prefill is None))
        if on_prefill:
            on_prefill()
        return normal_result()

    def run():
        session = {"warmups": {}, "reused_repeats": 0}
        ticks = iter(range(200))
        units = runner.run_measurements(
            [first, second], settings, manifest, tmp_path, session, decode, 2,
            lambda: None, lambda: None, lambda *args: audits.append(args),
            clock=lambda: next(ticks),
        )
        return units, session

    units, session = run()
    assert len(units) == 2 * 2 * 3
    assert len(calls) == len(units) + 2
    assert session["warmups"] == {"greedy": 1, "prompt_lookup": 1}
    assert [call for call in calls if call[-1]] == [
        ("greedy", first.id, True), ("prompt_lookup", first.id, True),
    ]
    assert [unit["timing"]["repeat"] for unit in units] == [1, 2, 3] * 4
    assert len(audits) == len(units)

    path = repeat_path(tmp_path, units[0]["timing"])
    path.write_text('{"payload":')
    (path.parent / ".pending-interrupted").write_text("partial temporary record")
    other_path = repeat_path(tmp_path, units[1]["timing"])
    save_repeat(other_path, "run", {**units[1], "iterations": []})
    calls.clear()
    audits.clear()
    resumed, session = run()
    assert session["reused_repeats"] == len(units) - 2
    assert session["warmups"] == {"greedy": 1}
    assert calls == [("greedy", first.id, True)] + [("greedy", first.id, False)] * 2
    assert len(audits) == len(units)
    assert resumed == units

    save_repeat(other_path, "different-run", units[1])
    key = {field: units[1]["timing"][field] for field in JOIN}
    with pytest.raises(RuntimeError, match="different run"):
        load_repeat(other_path, "run", key, {})

    write_csv(tmp_path / "timing.csv", TIMING, (unit["timing"] for unit in resumed))
    write_csv(tmp_path / "iterations.csv", ITERATIONS, (
        row for unit in resumed for row in unit["iterations"]
    ))
    with (tmp_path / "timing.csv").open(newline="") as timing, (
        tmp_path / "iterations.csv"
    ).open(newline="") as iterations:
        timing_rows = list(csv.DictReader(timing))
        iteration_rows = list(csv.DictReader(iterations))
    timing_keys = {tuple(row[field] for field in JOIN) for row in timing_rows}
    iteration_keys = {tuple(row[field] for field in JOIN + ("iteration",)) for row in iteration_rows}
    assert len(timing_keys) == len(timing_rows) == len(units)
    assert len(iteration_keys) == len(iteration_rows) == len(units) * 2
    assert {key[:-1] for key in iteration_keys} == timing_keys
    summary = summarize(resumed)
    assert len(summary["iterations_by_repeat"]) == len(units)
    assert summary["iterations_pooled_by_method"][0]["excluded_final_rows"] == 6
    assert summary["timing_by_prompt"][0]["decode_tok_per_s"] == {
        "median": 2.0, "min": 2.0, "max": 2.0, "spread_max_minus_min": 0.0,
        "valid_repeats": 3, "undefined_repeats": 0,
    }
    assert first_difference([3], [3, 4]) == 1


class ReplayCache:
    def __init__(self):
        self.tokens = []

    def get_seq_length(self):
        return len(self.tokens)

    def crop(self, count):
        del self.tokens[count:]


class ShapeSensitiveTarget:
    def __init__(self, gap, *, flip=True, lookup_runner_up=4):
        self.gap = gap
        self.flip = flip
        self.lookup_runner_up = lookup_runner_up
        self.calls = []

    def __call__(self, *, input_ids, past_key_values=None, use_cache, logits_to_keep=None):
        tokens = input_ids[0].tolist()
        cache = ReplayCache() if past_key_values is None else past_key_values
        start = cache.get_seq_length()
        self.calls.append((tokens, cache.tokens.copy()))
        cache.tokens.extend(tokens)
        logits = torch.full((1, len(tokens), 16), -10.0, dtype=torch.float16)
        for row in range(len(tokens)):
            position = start + row
            if position < 11:
                logits[0, row, 3] = 1.0
            elif position == 11:
                winner, runner_up = (
                    (5, self.lookup_runner_up) if self.flip and len(tokens) > 1 else (4, 5)
                )
                logits[0, row, winner] = 1.0
                logits[0, row, runner_up] = 1.0 - self.gap
            else:
                logits[0, row, 2] = 1.0
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits, past_key_values=cache)


@pytest.mark.parametrize(("gap", "lookup_runner_up", "verdict", "status"), [
    (2**-10, 4, "near_tie", "reviewed"),
    (4.0, 4, "unexplained", "fail"),
    (2**-10, 6, "near_tie", "fail"),
], ids=["near-tie", "large-gap", "unrelated-pair"])
def test_fp16_divergence_replays_cache_and_shapes(
    settings, run_identity, gap, lookup_runner_up, verdict, status,
):
    target = ShapeSensitiveTarget(gap, lookup_runner_up=lookup_runner_up)
    item = prompt(budget=4, tokens=[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1])
    decode = runner.make_decoder(target, 2, settings)
    reference = decode("greedy", item)
    candidate = decode("prompt_lookup", item)
    assert reference.token_ids == [3, 4, 2]
    assert candidate.token_ids == [3, 5, 2]
    starting_calls = target.calls.copy()
    kwargs = {
        "target": target, "decode": decode, "eos": 2,
        "settings": settings, "identity": run_identity,
    }
    audit = fp16_comparison(
        "fp16_untimed", "prompt_lookup", item, reference, candidate, reviews={}, **kwargs,
    )
    assert audit["status"] == "fail"
    assert target.calls == starting_calls * 2
    details = audit["details"]
    assert details["position"] == 1
    assert details["reference_stop"]["reason"] == details["output_stop"]["reason"] == "eos"
    assert details["near_tie_reviewable"] == (lookup_runner_up == 4)
    for name, shape, top_two in (
        ("reference_replay", [1, 1], [4, 5]), ("output_replay", [1, 4], [5, lookup_runner_up]),
    ):
        prediction = details[name]["prediction"]
        assert details[name]["reproduced"]
        assert prediction["forward_shape"] == shape and prediction["prefix_matches"]
        assert prediction["cache_length_before_forward"] == 11
        assert prediction["top_two"] == {"ids": top_two, "logits": [1.0, 1.0 - gap], "gap": gap}

    review = {"verdict": verdict, "diagnosis": "Synthetic test judgment."}
    measured = fp16_comparison(
        "fp16_measured", "prompt_lookup", item, reference, candidate, repeat=2,
        reviews={details["case_id"]: review}, **kwargs,
    )
    assert measured["repeat"] == 2
    assert measured["details"]["case_id"] == details["case_id"]
    assert measured["details"]["review"] == review
    assert measured["status"] == status


def test_fp16_length_only_recovers_unexpected_stop_logits(settings, run_identity):
    target = ShapeSensitiveTarget(2**-10)
    item = prompt(budget=4, tokens=[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1])
    decode_regular = runner.make_decoder(target, 2, settings)

    def decode(method, prompt, *, target=target):
        if method == "greedy":
            return decode_regular(method, prompt, target=target)
        return runner.greedy_decode(target, prompt.tensor, max_new_tokens=1, eos_token_id=2)

    audit = fp16_comparison(
        "fp16_measured", "prompt_lookup", item, decode("greedy", item), decode("prompt_lookup", item),
        target=target, decode=decode, eos=2, settings=settings, identity=run_identity,
        reviews={}, repeat=1,
    )
    assert audit["status"] == "fail"
    details = audit["details"]
    assert details["position"] == 1 and details["output_token_id"] is None
    assert details["output_stop"]["reason"] == "unexpected_stop"
    prediction = details["output_replay"]["prediction"]
    assert prediction["source"] == "early_stop_continuation_probe"
    assert prediction["prefix_matches"]
    assert prediction["top_two"]["ids"] == [5, 4]


@pytest.mark.parametrize(("stage", "gap", "verdict", "outcome"), [
    (None, 2**-10, None, "accepted"),
    ("fp16_measured", 2**-10, None, "blocked"),
    ("fp16_measured", 2**-10, "near_tie", "requires_dev_review"),
    ("fp16_measured", 4.0, "near_tie", "requires_dev_review"),
    ("fp16_untimed", 2**-10, "near_tie", "requires_dev_review"),
], ids=["zero-divergence", "unresolved", "measured-review", "large-gap", "untimed-review-resume"])
def test_export_speed_acceptance(
    tmp_path, config_bytes, generation_config, monkeypatch, run_identity, settings,
    stage, gap, verdict, outcome,
):
    root = tmp_path / "checkout"
    (root / "configs").mkdir(parents=True)
    (root / "configs/bench.yaml").write_bytes(config_bytes)
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "verify_checkout", lambda root, sha: sha)
    monkeypatch.setattr(runner, "read_frozen", lambda *_args: [])
    monkeypatch.setattr(runner, "validate_environment", lambda: run_identity["environment"])
    monkeypatch.setattr(runner.torch.cuda, "get_device_properties", lambda _index: object())
    monkeypatch.setattr(runner.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        runner, "preflight_tokenizer_and_generation",
        lambda: (SimpleNamespace(eos_token_id=2), generation_config),
    )
    item = prompt(budget=4, tokens=[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1])
    monkeypatch.setattr(runner, "tokenize_prompts", lambda *_args: [item])

    fp32 = ShapeSensitiveTarget(gap, flip=False)
    fp32.generate = lambda inputs, **kwargs: torch.cat((inputs, torch.tensor([[3, 4, 2]])), dim=1)
    fp16 = ShapeSensitiveTarget(gap)
    reviews_path = None
    if verdict:
        decode = runner.make_decoder(fp16, 2, settings)
        case = fp16_comparison(
            stage, "prompt_lookup", item, decode("greedy", item), decode("prompt_lookup", item),
            target=fp16, decode=decode, eos=2, settings=settings, reviews={}, identity=run_identity,
        )
        review = {"verdict": verdict, "diagnosis": "Synthetic test judgment."}
        reviews_path = tmp_path / "reviews.json"
        reviews_path.write_text(json.dumps({case["details"]["case_id"]: review}))

    run_benchmark = runner.benchmark
    run_measurements = runner.run_measurements

    def benchmark(prompts, settings, manifest, directory, session, _make_target, *args):
        fp16.flip = stage == "fp16_untimed"
        targets = {torch.float32: fp32, torch.float16: fp16}
        return run_benchmark(
            prompts, settings, manifest, directory, session, targets.__getitem__, *args,
            clock=iter(range(100)).__next__,
        )

    def measure_repeats(*args, **kwargs):
        fp16.flip = stage == "fp16_measured"
        return run_measurements(*args, **kwargs)

    monkeypatch.setattr(runner, "benchmark", benchmark)
    monkeypatch.setattr(runner, "run_measurements", measure_repeats)
    output = tmp_path / "output"
    if outcome == "blocked":
        with pytest.raises(RuntimeError, match="Measured fp16 audit"):
            runner.execute(output, "1" * 40, None)
    else:
        runner.execute(output, "1" * 40, reviews_path)
    baseline = output / "experiments/baseline"
    metadata = json.loads((baseline / "run.json").read_bytes())
    assert metadata["speed_results_accepted"] == (outcome == "accepted")
    assert metadata["requires_dev_review"] == (stage is not None)
    session = metadata["sessions"][-1]
    assert session["status"] == outcome
    assert (baseline / "summary.json").exists() == (outcome != "blocked")
    summary = baseline / "iteration_summary.csv"
    assert summary.exists() == (outcome != "blocked")
    if outcome == "accepted":
        with summary.open(newline="") as source:
            rows = list(csv.DictReader(source))
        assert {row["scope"] for row in rows} == {"repeat", "method"}
        assert all(row["alpha_hat"] == "" and row["alpha_undefined_reason"] == "zero_trials"
                   for row in rows if row["method"] == "greedy")
    assert (baseline / "bench.yaml").read_bytes() == config_bytes
    assert (baseline / "timing.csv").is_file()
    assert (baseline / "iterations.csv").is_file()
    correctness = baseline / "correctness.csv"
    with correctness.open(newline="") as source:
        divergences = [json.loads(row["details"]) for row in csv.DictReader(source)
                       if row["stage"] == stage]
    divergences = [details for details in divergences if "case_id" in details]
    if stage is not None:
        assert len(divergences) == session["fp16_divergences"][stage]
        assert divergences
        for details in divergences:
            assert details["review"] == (review if verdict else None)
            for path in ("reference_replay", "output_replay"):
                assert details[path]["prediction"]["top_two"]["gap"] == gap
        archived = baseline / "audits" / f'{session["id"]}.csv'
        original = correctness.read_bytes()
        assert archived.read_bytes() == original

    if stage == "fp16_untimed":
        stage = None
        runner.execute(output, "1" * 40, None)
        metadata = json.loads((baseline / "run.json").read_bytes())
        resumed = metadata["sessions"][-1]
        assert not any(resumed["fp16_divergences"].values())
        assert resumed["reused_repeats"] == session["complete_repeats"]
        assert resumed["status"] == "requires_dev_review"
        assert metadata["requires_dev_review"] and not metadata["speed_results_accepted"]
        assert archived.read_bytes() == original


def test_runner_with_tiny_llama_and_resume(settings, run_identity, generation_config, tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    seen_dtypes = []

    def make_target(dtype):
        seen_dtypes.append(dtype)
        config = LlamaConfig(
            vocab_size=24, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(settings["seed"])
            return LlamaForCausalLM(config).to(dtype=dtype).eval()

    item = prompt(budget=4, tokens=[1, 5, 7, 9])
    manifest = {"identity": run_identity, "run_id": "tiny-llama"}
    calls = []

    def run():
        audit = []
        session = {"warmups": {}, "reused_repeats": 0}
        ticks = iter(range(100))
        units = runner.benchmark(
            [item], settings, manifest, tmp_path, session, make_target,
            generation_config, 2, {}, audit, lambda: None, lambda: calls.append("sync"),
            clock=lambda: next(ticks),
        )
        assert {row["stage"] for row in audit} == {
            "fp32_first_20", "fp32_full", "fp16_untimed", "fp16_measured",
        }
        assert all(row["status"] == "pass" for row in audit)
        return units, session

    initial, first_session = run()
    assert first_session["warmups"] == dict.fromkeys(runner.METHODS, 1)
    assert len(initial) == len(runner.METHODS) * settings["repeats"]
    assert len(calls) == len(initial) * 3 + len(runner.METHODS)
    calls.clear()
    resumed, resume_session = run()
    assert seen_dtypes == [torch.float32, torch.float16] * 2
    assert resumed == initial
    assert resume_session["warmups"] == {}
    assert resume_session["reused_repeats"] == len(initial)
    assert calls == []


def test_environment_requires_pinned_t4(monkeypatch):
    monkeypatch.setattr(runner, "version", lambda package: {
        "transformers": "5.16.1", "tokenizers": "0.23.3",
    }.get(package, "fixture"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(runner.torch, "__version__", "2.11.0+cu128")
    monkeypatch.setattr(runner.torch.version, "cuda", "12.8")
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 1)
    properties = SimpleNamespace(
        name="Tesla T4", major=7, minor=5, total_memory=15660040192, multi_processor_count=40,
    )
    monkeypatch.setattr(runner.torch.cuda, "get_device_properties", lambda _index: properties)
    check_output = runner.subprocess.check_output

    def nvidia_smi(args, **kwargs):
        if args[0] == "nvidia-smi":
            return "580.178.04\n"
        return check_output(args, **kwargs)

    monkeypatch.setattr(runner.subprocess, "check_output", nvidia_smi)
    environment = runner.validate_environment()
    assert environment["driver"] == "580.178.04"
    assert environment["visible_device_count"] == 1 and environment["device_name"] == "Tesla T4"
    properties.name = "NVIDIA A100"
    with pytest.raises(RuntimeError, match="Expected a T4"):
        runner.validate_environment()
    monkeypatch.setattr(runner.torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError, match="Exactly one"):
        runner.validate_environment()
    monkeypatch.setattr(runner.torch, "__version__", "2.11.0")
    with pytest.raises(RuntimeError, match="Expected torch"):
        runner.validate_environment()
