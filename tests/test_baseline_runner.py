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
        self.greedy_winner = 4
        self.lookup_winner = 5
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
                    (self.lookup_winner, self.lookup_runner_up)
                    if self.flip and len(tokens) > 1 else (self.greedy_winner, 5)
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
    assert (baseline / "timing.csv").is_file() == (outcome != "blocked")
    assert (baseline / "iterations.csv").is_file() == (outcome != "blocked")
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


@pytest.fixture
def fp16_diagnostic_evidence(tmp_path, config_bytes, settings, run_identity, generation_config):
    from kaggle import diagnose_fp16 as diagnostic
    from kaggle.baseline_artifacts import json_bytes, write_correctness

    identity = {**run_identity, "git_commit": diagnostic.AUDITED_COMMIT}
    records = runner.read_frozen(ROOT / "bench/prompts.jsonl", settings)
    source = tmp_path / "source/experiments"
    (source / "baseline/audits").mkdir(parents=True)
    runtime = {"attention_implementation": "sdpa"}
    session = {
        "id": "2" * 32, "status": "blocked", "failed_stage": "fp16_audit",
        "fp16_divergences": {"fp16_untimed": 1, "fp16_measured": 0},
        "torch.float16": runtime, "torch.float32": runtime,
    }
    manifest = {
        "run_id": "3" * 32, "identity": identity, "sessions": [session],
        "speed_results_accepted": False, "requires_dev_review": True,
        "target_runtime": {"torch.float16": runtime, "torch.float32": runtime},
    }
    (source / "env.txt").write_bytes(json_bytes(identity))
    (source / "baseline/run.json").write_bytes(json_bytes(manifest))
    (source / "baseline/bench.yaml").write_bytes(config_bytes)
    rows = [
        runner.check_row("integrity", None, None, status="pass", benchmark_sha256=runner.PROMPTS_SHA256),
        runner.check_row(
            "tokenizer_and_generation", None, None, status="pass",
            revision=runner.TARGET_REVISION, token_counts_checked=len(records),
            generation_config=generation_config.to_diff_dict(),
        ),
    ]
    subset = first_five_per_category([SimpleNamespace(**record) for record in records])
    for stage, selected in (("fp32_first_20", [row.id for row in subset]),
                            ("fp32_full", [row["id"] for row in records])):
        for method in runner.METHODS:
            rows.extend(runner.check_row(
                stage, method, prompt_id, status="pass", reference_length=128, output_length=128,
            ) for prompt_id in selected)

    def replay(pair, logits):
        widths = [34] + ([1] * 90 if pair[0] == 367 else [9] + [1] * 89)
        return {
            "reproduced": True,
            "prediction": {
                "source": "original_shape_replay", "prefix_matches": True, "forward_shape": [1, 1],
                "cache_length_before_forward": 123, "logit_row": 0,
                "forward_index": len(widths) - 1, "query_lengths_through_forward": widths,
                "top_two": {"ids": pair, "logits": logits, "gap": logits[0] - logits[1]},
                "compared_logits": {"367": 17.28125, "13100": logits[0]},
            },
        }

    evidence = {
        "position": 90, "reference_token_id": 367, "output_token_id": 13100,
        "reference_stop": {"reason": "budget"}, "output_stop": {"reason": "budget"},
        "reference_replay": replay([367, 2125], [17.28125, 17.28125]),
        "output_replay": replay([13100, 367], [17.296875, 17.28125]),
        "outputs_sha256": "4" * 64,
    }
    case_id = digest(json_bytes([identity, "chat_qa-25", "prompt_lookup", evidence]))
    for record in records:
        if record["id"] == "chat_qa-25":
            rows.append(runner.check_row(
                "fp16_untimed", "prompt_lookup", record["id"], status="fail",
                **evidence, near_tie_reviewable=False, case_id=case_id,
            ))
        else:
            rows.append(runner.check_row(
                "fp16_untimed", "prompt_lookup", record["id"], status="pass", output_length=128,
            ))
    correctness = source / "baseline/correctness.csv"
    write_correctness(correctness, rows)
    (source / f'baseline/audits/{session["id"]}.csv').write_bytes(correctness.read_bytes())
    return source, identity, records, case_id


@pytest.mark.parametrize("source_kind", ["directory", "archive"])
def test_fp16_diagnostic_validates_and_copies_original_evidence(
    tmp_path, monkeypatch, fp16_diagnostic_evidence, generation_config, source_kind,
):
    import tarfile

    from kaggle import diagnose_fp16 as diagnostic

    source, identity, _, case_id = fp16_diagnostic_evidence
    archive = tmp_path / "specserve-day2b2-audit.tar.gz"
    if source_kind == "archive":
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(source, arcname="experiments")
        source.rename(tmp_path / "saved-original")

    args = SimpleNamespace(
        checkout=ROOT, git_sha=diagnostic.AUDITED_COMMIT,
        source_experiments=source if source_kind == "directory" else None,
        source_archive=archive if source_kind == "archive" else None,
    )
    script = tmp_path / "diagnose_fp16.py"
    script.write_bytes(Path(diagnostic.__file__).read_bytes())
    output = tmp_path / "diagnostic"
    monkeypatch.setattr(runner, "verify_checkout", lambda *_args: diagnostic.AUDITED_COMMIT)
    monkeypatch.setattr(runner, "validate_environment", lambda: identity["environment"])
    monkeypatch.setattr(runner, "preflight_tokenizer_and_generation",
                        lambda: (SimpleNamespace(eos_token_id=2), generation_config))
    monkeypatch.setattr(runner, "tokenize_prompts", lambda *_args: [prompt("chat_qa-25")])
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _seed: None)

    def run_probe(_output, _phase, item, _settings, _eos, case, _make_target):
        assert item.id == "chat_qa-25" and case["case_id"] == case_id
        return "cpu_fixture"

    monkeypatch.setattr(diagnostic, "diagnose", run_probe)
    diagnostic.persist_run(output, script, lambda phase: diagnostic.execute(args, script, output, phase))
    provenance = json.loads((output / "provenance.json").read_bytes())
    assert provenance["case_id"] == case_id
    assert provenance["script_sha256"] == digest(script.read_bytes())
    assert provenance["source_archive_sha256"] == (
        digest(archive.read_bytes()) if source_kind == "archive" else None
    )
    original = source if source_kind == "directory" else tmp_path / "saved-original"
    for name, checksum in provenance["source_file_sha256"].items():
        data = (output / "original/experiments" / name).read_bytes()
        assert data == (original / name).read_bytes()
        assert digest(data) == checksum


@pytest.mark.parametrize(("defect", "match"), [
    ("missing_yaml", "Missing or empty original evidence"),
    ("stale_audit", "inconsistent archived audit"),
    ("bad_env", "environment and run identity disagree"),
    ("bad_yaml", "Original YAML differs"),
    ("changed_case", "Inconsistent case ID"),
    ("traversal", "Invalid or duplicate archive path"),
    ("symlink", "Non-regular archive member"),
    ("duplicate", "Invalid or duplicate archive path"),
    ("truncated", "end-of-stream marker"),
])
def test_fp16_diagnostic_rejects_invalid_archive(
    tmp_path, fp16_diagnostic_evidence, config_bytes, defect, match,
):
    import tarfile

    from kaggle import diagnose_fp16 as diagnostic

    source, identity, records, _ = fp16_diagnostic_evidence
    audit = next((source / "baseline/audits").iterdir())
    if defect == "missing_yaml":
        (source / "baseline/bench.yaml").unlink()
    if defect == "stale_audit":
        audit.write_bytes(b"outdated")
    if defect == "bad_env":
        (source / "env.txt").write_text("{}")
    if defect == "bad_yaml":
        (source / "baseline/bench.yaml").write_bytes(config_bytes + b"\n")
    if defect == "changed_case":
        corrupted = audit.read_bytes().replace(b"17.296875", b"17.312500")
        (source / "baseline/correctness.csv").write_bytes(corrupted)
        audit.write_bytes(corrupted)

    archive = tmp_path / "specserve-day2b2-audit.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="experiments")
        if defect == "traversal":
            tar.addfile(tarfile.TarInfo("experiments/../env.txt"))
        if defect == "duplicate":
            tar.addfile(tarfile.TarInfo("experiments/env.txt"))
        if defect == "symlink":
            member = tarfile.TarInfo("experiments/extra")
            member.type = tarfile.SYMTYPE
            member.linkname = "env.txt"
            tar.addfile(member)
    if defect == "truncated":
        archive.write_bytes(archive.read_bytes()[:-8])

    with pytest.raises((RuntimeError, EOFError), match=match):
        files, manifest, _ = diagnostic.load_source(None, archive)
        diagnostic.validate_audit(files, manifest, identity, config_bytes, records)


@pytest.mark.parametrize("backend", ["sdpa", "eager"])
@torch.inference_mode()
def test_fp16_diagnostic_fixed_prefix_and_suffix_controls(backend):
    from transformers import LlamaConfig, LlamaForCausalLM

    from kaggle import diagnose_fp16 as diagnostic

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        target = LlamaForCausalLM(LlamaConfig(
            vocab_size=24, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, attn_implementation=backend,
        )).eval()

    def call(tokens, before, crop=0):
        return {
            "input_ids": tokens, "cache_before": before, "cache_after_forward": before + len(tokens),
            "crop": crop, "cache_after_crop": before + len(tokens) - crop,
            "logits_to_keep": 1 if before == 0 else 0,
        }

    prefix = [1, 3, 6, 7, 8]
    greedy = [call([1, 3], 0)] + [call([token], i) for i, token in enumerate(prefix[2:], start=2)]
    lookup = [call([1, 3], 0), call([6, 7, 15, 16], 2, 2), call([8], 4)]
    assert diagnostic.history_matches(greedy, prefix)
    assert diagnostic.history_matches(lookup, prefix)
    assert not diagnostic.history_matches(lookup, [1, 3, 6, 15, 8])
    greedy_row, _, greedy_checks = diagnostic.fixed_history(target, greedy, "cpu")
    lookup_row, incoming, lookup_checks = diagnostic.fixed_history(target, lookup, "cpu", inspect_call=1)
    reference = target(input_ids=torch.tensor([prefix]), use_cache=False).logits[0, -1]
    assert all(greedy_checks + lookup_checks)
    torch.testing.assert_close(greedy_row, reference)
    torch.testing.assert_close(lookup_row, reference)

    left = diagnostic.clone_cache(target, incoming, "cpu")
    right = diagnostic.clone_cache(target, incoming, "cpu")
    left.layers[0].keys.zero_()
    assert torch.equal(right.layers[0].keys, incoming[0][0])
    assert not torch.equal(left.layers[0].keys, right.layers[0].keys)
    assert torch.equal(*diagnostic.identical_cache_control(target, incoming, 6, "cpu"))

    suffix, tensors = diagnostic.suffix_control(target, incoming, lookup[1], "cpu")
    assert suffix["structural_checks"] and suffix["repeat_exact"]
    assert not suffix["suffix_influence"]
    assert suffix["perturbed_input_ids"][:2] == [6, 7]
    assert suffix["perturbed_input_ids"][2:] != [15, 16]
    assert suffix["geometry"][0]["positions"] == [[2, 3, 4, 5]]
    mask = tensors[0][2][0]
    allowed = mask if mask.dtype == torch.bool else mask == 0
    assert torch.equal(allowed[0, 0], torch.arange(6) <= torch.arange(2, 6)[:, None])
    assert not diagnostic.mask_evidence(None, True, 4, 6, 2)["expected_causal_pattern"]


def test_fp16_diagnostic_persists_partial_evidence_on_failure(tmp_path):
    from kaggle import diagnose_fp16 as diagnostic

    output = tmp_path / "output"
    script = tmp_path / "diagnose_fp16.py"
    script.write_bytes(b"diagnostic fixture")
    evidence = torch.tensor([[1, 2]])

    def fail(phase):
        phase("fixed_history_fp16")
        diagnostic.save_tensors(output / "completed.pt", evidence)
        phase("fixed_history_fp32")
        raise RuntimeError("controlled failure")

    with pytest.raises(RuntimeError, match="controlled failure"):
        diagnostic.persist_run(output, script, fail)
    assert torch.equal(torch.load(output / "completed.pt", weights_only=True), evidence)
    state = json.loads((output / "diagnostic.json").read_bytes())
    assert state["phase"] == "fixed_history_fp32" and state["status"] == "failed"
    assert state["error_type"] == "RuntimeError" and state["error"] == "controlled failure"
    assert state["script_sha256"] == digest(script.read_bytes())
    assert not list(output.glob(".pending-*"))


@torch.inference_mode()
def test_fp16_diagnostic_observes_native_replay(tmp_path, settings, monkeypatch):
    from transformers import LlamaConfig, LlamaForCausalLM

    from kaggle import diagnose_fp16 as diagnostic
    from kaggle.baseline_audit import ReplayTarget

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        target = LlamaForCausalLM(LlamaConfig(
            vocab_size=24, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, attn_implementation="sdpa",
        )).eval()
    item = prompt(budget=4, tokens=[1, 4, 5])
    decode = runner.make_decoder(target, None, settings)
    monkeypatch.setattr(diagnostic, "POSITION", 2)
    monkeypatch.setattr(diagnostic, "WINNERS", (5, 7))
    for method in runner.METHODS:
        original = decode(method, item)
        audit_observer = ReplayTarget(target, original, 2, item.token_ids, (5, 7))
        assert decode(method, item, target=audit_observer) == original

        calls, captured = diagnostic.native_trace(
            target, decode, method, item, original, {"prediction": audit_observer.evidence}, tmp_path,
        )
        assert diagnostic.history_matches(calls, item.token_ids + original.token_ids[:2])
        assert calls[-1]["cache_before"] == 4
        assert calls[-1]["positions"] == [[4]]
        assert calls[-1]["masks_match"]
        assert captured["logits"].argmax().item() == original.token_ids[2]
        on_disk = torch.load(tmp_path / f"{method}-retained-kv.pt", weights_only=True)
        assert diagnostic.cache_matches(on_disk, 4)
        prediction = torch.load(tmp_path / f"{method}-prediction.pt", weights_only=True)
        assert torch.equal(prediction["logits"], captured["logits"])
        assert prediction["masks"] == [None]


@pytest.mark.parametrize(("lookup_tail", "outcome"), [
    (1e-6, "numerical_drift_supported_dev_review"),
    (-0.25, "unresolved"),
], ids=["rounding-only", "same-winner-material-difference"])
def test_fp16_diagnostic_requires_full_fp32_logit_agreement(
    tmp_path, settings, monkeypatch, lookup_tail, outcome,
):
    from kaggle import diagnose_fp16 as diagnostic
    from kaggle.baseline_artifacts import json_bytes

    item = prompt(tokens=[1, 3])
    generated = [[4, 5, 0], [4, 5, 1]]
    case = {
        "outputs_sha256": digest(json_bytes(generated)),
        "reference_replay": {}, "output_replay": {},
    }
    methods = list(runner.METHODS)
    monkeypatch.setattr(diagnostic, "POSITION", 2)
    monkeypatch.setattr(diagnostic, "WINNERS", (0, 1))
    monkeypatch.setattr(diagnostic, "CACHE_LENGTH", 3)
    monkeypatch.setattr(runner, "make_decoder", lambda *_args: (
        lambda method, _prompt: DecodeResult(generated[methods.index(method)], [], False)
    ))

    def call(tokens, before, crop=0):
        end = before + len(tokens)
        return {
            "input_ids": tokens, "cache_before": before, "cache_after_forward": end,
            "cache_after_crop": end - crop, "crop": crop, "positions_match": True, "masks_match": True,
        }

    traces = [
        [call([1, 3], 0), call([4], 2), call([5], 3)],
        [call([1, 3], 0), call([4, 6], 2, crop=1), call([5], 3)],
    ]
    fp16_rows = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float16)
    fp32_rows = torch.tensor([[2.0, 0.0], [2.0, lookup_tail], [2.0, 0.0]], dtype=torch.float32)

    def native_trace(_target, _decode, method, *_args):
        index = methods.index(method)
        keys = torch.tensor([[[[0.0], [0.0], [index]]]], dtype=torch.float16)
        return traces[index], {"kv": [(keys, keys)], "logits": fp16_rows[index]}

    class Target:
        model = SimpleNamespace(layers=[object()])

        def __init__(self, dtype):
            self.dtype = dtype

        def __call__(self, **_kwargs):
            return SimpleNamespace(logits=fp32_rows[2].reshape(1, 1, -1))

    def fixed_history(target, calls, _device, *, inspect_call=None):
        rows = fp16_rows if target.dtype == torch.float16 else fp32_rows
        return rows[int(bool(calls[1]["crop"]))], None, [True]

    monkeypatch.setattr(diagnostic, "native_trace", native_trace)
    monkeypatch.setattr(diagnostic, "fixed_history", fixed_history)
    monkeypatch.setattr(diagnostic, "identical_cache_control", lambda _target, kv, *_args: [
        fp16_rows[int(kv[0][0].flatten()[-1])],
    ] * 2)
    monkeypatch.setattr(diagnostic, "suffix_control", lambda *_args: (
        {"repeat_exact": True, "structural_checks": True, "suffix_influence": False}, [],
    ))

    assert diagnostic.diagnose(tmp_path, lambda _phase: None, item, settings, None, case, Target) == outcome
    saved = torch.load(tmp_path / "fp32-fixed-history-logits.pt", weights_only=True)
    assert torch.equal(torch.stack(saved), fp32_rows)
    evidence = json.loads((tmp_path / "fp32-fixed-history.json").read_bytes())
    assert [row["winner"] for row in evidence["logits"]] == [0, 0, 0]
    assert evidence["logit_deltas_from_greedy"][0]["max_abs"] == abs(fp32_rows[1, 1].item())
    comparison = evidence["full_logit_comparison"]
    assert comparison["rtol"] == 1.3e-6 and comparison["atol"] == 1e-5
    close = outcome == "numerical_drift_supported_dev_review"
    assert comparison["pairwise_close"] == {
        "lookup_vs_greedy": close, "greedy_vs_uncached": True, "lookup_vs_uncached": close,
    }
    assert comparison["lookup_delta_from_uncached"] == evidence["logit_deltas_from_greedy"][0]


@pytest.fixture
def fp16_drift_review(tmp_path, settings, run_identity, config_bytes):
    from kaggle.baseline_artifacts import json_bytes, write_correctness, write_json
    from kaggle.baseline_audit import DRIFT_VERDICT, FP16_EVIDENCE_FIELDS

    item = prompt(budget=4, tokens=[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1])
    target = ShapeSensitiveTarget(2**-10, lookup_runner_up=6)
    decode = runner.make_decoder(target, 2, settings)
    reference, output = [decode(method, item) for method in runner.METHODS]

    def compare(identity, reviews=None, **kwargs):
        return fp16_comparison(
            "fp16_untimed", "prompt_lookup", item, reference, output,
            target=target, decode=decode, eos=2, settings=settings,
            reviews=reviews or {}, identity=identity, **kwargs,
        )

    original_identity = {**run_identity, "git_commit": "2" * 40}
    source_row = compare(original_identity)
    source_case = source_row["details"]
    assert source_case["near_tie_reviewable"] is False and source_row["status"] == "fail"
    evidence = {name: source_case[name] for name in FP16_EVIDENCE_FIELDS}
    case_id = compare(run_identity)["details"]["case_id"]
    assert case_id != source_case["case_id"]
    assert case_id == digest(json_bytes([run_identity, item.id, "prompt_lookup", evidence]))
    runtime = dict.fromkeys(("torch.float32", "torch.float16"), {"attention_implementation": "sdpa"})
    session_id = "3" * 32
    source_run_id = "4" * 32
    root = tmp_path / "evidence"
    original = root / "original/experiments"
    archived_name = f"baseline/audits/{session_id}.csv"
    write_json(original / "env.txt", original_identity)
    write_json(original / "baseline/run.json", {
        "identity": original_identity, "target_runtime": runtime,
        "run_id": source_run_id, "sessions": [{"id": session_id}],
    })
    (original / "baseline/bench.yaml").write_bytes(config_bytes)
    for name in ("baseline/correctness.csv", archived_name):
        write_correctness(original / name, [source_row])
    script_hash = "5" * 64
    write_json(root / "diagnostic.json", {
        "status": "complete", "result": "numerical_drift_supported_dev_review",
        "git_commit": original_identity["git_commit"], "script_sha256": script_hash,
    })
    write_json(root / "provenance.json", {
        "case_id": source_case["case_id"], "identity": original_identity, "target_runtime": runtime,
        "original_run_id": source_run_id, "script_sha256": script_hash,
        "source_file_sha256": {
            path.relative_to(original).as_posix(): digest(path.read_bytes())
            for path in original.rglob("*") if path.is_file()
        },
    })
    reports = {
        "original-outputs": {"outputs": [reference.token_ids, output.token_ids], "position": 1},
        "native-invariants": {"structural_checks": True},
        "retained-kv-comparison": {
            "first_differing_position": 11, "first_lookup_verification_position": 11, "lookup_rejection_call": 1,
        },
        "identical-cache": {
            "repeat_exact": True, "follows_cache": True, "logits": [[{"winner": 4}] * 2, [{"winner": 5}] * 2],
        },
        "fp16-fixed-history": {
            "structure": True, "matches_native": [True, True], "logits": [{"winner": 4}, {"winner": 5}],
        },
        "fp32-fixed-history": {
            "structure": True, "logits": [{"winner": 4}] * 3,
            "full_logit_comparison": {
                "rtol": 1.3e-6, "atol": 1e-5,
                "pairwise_close": {
                    "lookup_vs_greedy": True, "greedy_vs_uncached": True, "lookup_vs_uncached": True,
                },
            },
        },
        **{
            f"{dtype}-suffix": {"structural_checks": True, "repeat_exact": True, "suffix_influence": False}
            for dtype in ("fp16", "fp32")
        },
        **{
            f"{method}-trace": source_case[f"{role}_replay"]
            for method, role in (("greedy", "reference"), ("prompt_lookup", "output"))
        },
    }
    for name, report in reports.items():
        write_json(root / f"{name}.json", report)
    for name in (
        "greedy-retained-kv.pt", "greedy-prediction.pt", "prompt_lookup-retained-kv.pt",
        "prompt_lookup-prediction.pt", "identical-cache-logits.pt", "fp16-fixed-history-logits.pt",
        "fp32-fixed-history-logits.pt", "fp16-suffix-tensors.pt", "fp32-suffix-tensors.pt",
    ):
        torch.save(torch.tensor([1.0, 2.0]), root / name)

    def tree_hash():
        return digest(json_bytes({
            path.relative_to(root).as_posix(): digest(path.read_bytes())
            for path in root.rglob("*") if path.is_file()
        }))

    review = {
        "verdict": DRIFT_VERDICT, "diagnosis": "Synthetic Dev review of diagnosed cache drift.",
        "run_identity_sha256": digest(json_bytes(run_identity)), "outputs_sha256": evidence["outputs_sha256"],
        "diagnostic_dir": "evidence", "diagnostic_sha256": tree_hash(),
    }
    review_path = tmp_path / "reviews.json"

    def write_review(key=case_id):
        write_json(review_path, {key: review})
        return review_path

    write_review()
    return SimpleNamespace(
        item=item, target=target, compare=compare, evidence=evidence, case_id=case_id, runtime=runtime,
        root=root, review=review, review_path=review_path, tree_hash=tree_hash, write_review=write_review,
        original_identity=original_identity, source_case_id=source_case["case_id"],
    )


@pytest.mark.parametrize("mismatch", [
    "none", "no-approval", "case-id", "identity-hash", "source-identity", "runtime", "changed-replay",
])
def test_fp16_drift_review_binds_live_case(fp16_drift_review, run_identity, mismatch):
    from kaggle.baseline_artifacts import json_bytes

    fixture = fp16_drift_review
    identity = run_identity
    runtime = fixture.runtime
    path = fixture.review_path
    if mismatch == "no-approval":
        path = None
    elif mismatch == "case-id":
        path = fixture.write_review(fixture.source_case_id)
    elif mismatch == "identity-hash":
        fixture.review["run_identity_sha256"] = digest(json_bytes(fixture.original_identity))
        path = fixture.write_review()
    elif mismatch == "source-identity":
        identity = {**run_identity, "environment": {"device": "changed"}}
        fixture.review["run_identity_sha256"] = digest(json_bytes(identity))
        path = fixture.write_review(
            digest(json_bytes([identity, fixture.item.id, "prompt_lookup", fixture.evidence])),
        )
    elif mismatch == "runtime":
        runtime = {**runtime, "torch.float16": {"attention_implementation": "eager"}}
    elif mismatch == "changed-replay":
        fixture.target.gap *= 2
        changed = fixture.compare(identity)["details"]
        assert changed["outputs_sha256"] == fixture.review["outputs_sha256"]
        assert changed["case_id"] != fixture.case_id
        path = fixture.write_review(changed["case_id"])

    row = fixture.compare(identity, runner.read_reviews(path), target_runtime=runtime)
    assert row["status"] == ("reviewed" if mismatch == "none" else "fail")
    assert row["details"]["near_tie_reviewable"] is False
    if mismatch != "changed-replay":
        assert all(row["details"][name] == value for name, value in fixture.evidence.items())
    if mismatch == "none":
        assert row["details"]["review"]["diagnostic_sha256"] == fixture.review["diagnostic_sha256"]
        assert row["details"]["review"]["diagnostic"]["case_id"] == fixture.source_case_id
        assert row["details"]["diagnosis"] == fixture.review["diagnosis"]


@pytest.mark.parametrize(("defect", "error"), [
    ("missing-review-field", "Invalid numerical-drift review entry"),
    ("wrong-output-hash", "output hash does not match"),
    ("changed-file", "evidence checksum mismatch"),
    ("missing-file", "Missing diagnostic tensor evidence"),
    ("missing-dir", "Missing or invalid diagnostic directory"),
    ("symlink", "regular files and directories"),
    ("failed-outcome", "original run provenance is inconsistent"),
    ("changed-source-case", "Invalid diagnostic evidence"),
    ("changed-original-audit", "original run provenance is inconsistent"),
    ("failed-repeat", "cache or fixed-history controls did not pass"),
    ("inconsistent-repeat", "cache or fixed-history controls did not pass"),
    ("failed-fp32-winner", "cache or fixed-history controls did not pass"),
    ("failed-fp32-vector", "cache or fixed-history controls did not pass"),
    ("failed-suffix", "rejected-suffix control did not pass"),
])
def test_fp16_drift_review_rejects_invalid_diagnostic(fp16_drift_review, defect, error):
    fixture = fp16_drift_review
    root = fixture.root
    tensor = root / "fp32-fixed-history-logits.pt"

    def change_report(name, key, value):
        path = root / name
        report = json.loads(path.read_bytes())
        report[key] = value
        path.write_text(json.dumps(report))

    if defect == "missing-review-field":
        del fixture.review["diagnostic_sha256"]
    elif defect == "wrong-output-hash":
        fixture.review["outputs_sha256"] = "9" * 64
    elif defect == "changed-file":
        tensor.write_bytes(b"changed tensor")
    elif defect == "missing-file":
        tensor.unlink()
    elif defect == "missing-dir":
        fixture.review["diagnostic_dir"] = "missing"
    elif defect == "symlink":
        tensor.rename(root / "replacement.pt")
        tensor.symlink_to("replacement.pt")
    elif defect == "failed-outcome":
        change_report("diagnostic.json", "result", "unresolved")
    elif defect == "changed-source-case":
        change_report("provenance.json", "case_id", "0" * 64)
    elif defect == "changed-original-audit":
        (root / "original/experiments/baseline/correctness.csv").write_bytes(b"changed audit")
    elif defect == "failed-repeat":
        change_report("identical-cache.json", "repeat_exact", False)
    elif defect == "inconsistent-repeat":
        change_report("identical-cache.json", "logits", [[{"winner": 4}] * 2, [{"winner": 7}] * 2])
    elif defect == "failed-fp32-winner":
        change_report("fp32-fixed-history.json", "logits", [{"winner": 4}, {"winner": 5}, {"winner": 4}])
    elif defect == "failed-fp32-vector":
        change_report("fp32-fixed-history.json", "full_logit_comparison", {
            "rtol": 1.3e-6, "atol": 1e-5, "pairwise_close": {
                "lookup_vs_greedy": False, "greedy_vs_uncached": True, "lookup_vs_uncached": False,
            },
        })
    elif defect == "failed-suffix":
        change_report("fp16-suffix.json", "suffix_influence", True)
    if defect not in ("missing-review-field", "changed-file", "symlink"):
        fixture.review["diagnostic_sha256"] = fixture.tree_hash()

    with pytest.raises(ValueError, match=error):
        runner.read_reviews(fixture.write_review())


@pytest.mark.parametrize(("scenario", "blocked_stage"), [
    ("approved", None),
    ("first-fp32", "fp32_first_20"),
    ("full-fp32", "fp32_full"),
    ("unexplained", "fp16_audit"),
    ("first-measured-repeat", "measuring"),
    ("changed-repeat", "measuring"),
])
def test_fp16_drift_review_preserves_runner_gates(
    tmp_path, settings, run_identity, config_bytes, generation_config, fp16_drift_review,
    monkeypatch, scenario, blocked_stage,
):
    fixture = fp16_drift_review
    root = tmp_path / "checkout"
    (root / "configs").mkdir(parents=True)
    (root / "configs/bench.yaml").write_bytes(config_bytes)
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "verify_checkout", lambda _root, sha: sha)
    monkeypatch.setattr(runner, "read_frozen", lambda *_args: [])
    monkeypatch.setattr(runner, "validate_environment", lambda: run_identity["environment"])
    monkeypatch.setattr(runner.torch.cuda, "get_device_properties", lambda _index: object())
    monkeypatch.setattr(runner.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(runner, "preflight_tokenizer_and_generation",
                        lambda: (SimpleNamespace(eos_token_id=2), generation_config))
    prompts = [fixture.item] + [
        prompt(f"chat_qa-{index:02d}", budget=fixture.item.budget, tokens=fixture.item.token_ids)
        for index in range(2, 7)
    ]
    monkeypatch.setattr(runner, "tokenize_prompts", lambda *_args: prompts)
    if scenario == "unexplained":
        fixture.review.clear()
        fixture.review.update(verdict="unexplained", diagnosis="Further evidence required.")
        fixture.write_review()

    fp32 = ShapeSensitiveTarget(2**-10, flip=False)

    def hf_generate(inputs, **kwargs):
        failed = scenario == "first-fp32" or (scenario == "full-fp32" and inputs is prompts[-1].tensor)
        continuation = [3, 2] if failed else [3, 4, 2]
        return torch.cat((inputs, torch.tensor([continuation])), dim=1)

    fp32.generate = hf_generate
    fp16 = fixture.target
    original_make_decoder = runner.make_decoder

    def make_decoder(target, eos, settings):
        wrapped = original_make_decoder(target, eos, settings)

        def decode(method, item, **kwargs):
            if target is fp16:
                # Only the reviewed prompt diverges. Change its measured
                # output in one scenario while preserving a complete repeat.
                fp16.flip = item.id == fixture.item.id
            return wrapped(method, item, **kwargs)

        return decode

    monkeypatch.setattr(runner, "make_decoder", make_decoder)
    original_benchmark = runner.benchmark
    original_measurements = runner.run_measurements
    original_measure = runner.measure
    measuring = []
    timed_calls = []

    def benchmark(items, settings, manifest, directory, session, _make_target, *args):
        manifest["target_runtime"] = fixture.runtime
        fp16.greedy_winner = 4
        targets = {torch.float32: fp32, torch.float16: fp16}
        return original_benchmark(
            items, settings, manifest, directory, session, targets.__getitem__, *args,
            clock=iter(range(500)).__next__,
        )

    def run_measurements(*args):
        measuring.append(True)
        if scenario == "first-measured-repeat":
            fp16.greedy_winner = 7
        elif scenario == "changed-repeat":
            fp16.lookup_winner = 7
        return original_measurements(*args)

    def measure(decode, method, item, synchronize, clock):
        timed_calls.append((method, item.id))
        return original_measure(decode, method, item, synchronize, clock)

    monkeypatch.setattr(runner, "benchmark", benchmark)
    monkeypatch.setattr(runner, "run_measurements", run_measurements)
    monkeypatch.setattr(runner, "measure", measure)
    output = tmp_path / "run"
    if blocked_stage:
        with pytest.raises(RuntimeError, match="exact-match failed|fp16 audit"):
            runner.execute(output, run_identity["git_commit"], fixture.review_path)
    else:
        runner.execute(output, run_identity["git_commit"], fixture.review_path)

    baseline = output / "experiments/baseline"
    metadata = json.loads((baseline / "run.json").read_bytes())
    assert metadata["speed_results_accepted"] is False
    assert bool(measuring) == (scenario in ("approved", "first-measured-repeat", "changed-repeat"))
    session = metadata["sessions"][-1]
    with (baseline / "correctness.csv").open(newline="") as stream:
        audit_rows = list(csv.DictReader(stream))
    if blocked_stage:
        assert session["status"] == "blocked"
        if blocked_stage.startswith("fp32"):
            assert session["failed_stage"] == "fp32_gates"
            assert [row["stage"] for row in audit_rows if row["status"] == "fail"] == [blocked_stage]
        else:
            assert session["failed_stage"] == blocked_stage
    else:
        assert session["status"] == "requires_dev_review" and metadata["requires_dev_review"] is True
        assert len([row for row in audit_rows if row["status"] == "reviewed"]) == 1 + settings["repeats"]
        assert (baseline / "summary.json").is_file()
    if scenario in ("first-measured-repeat", "changed-repeat"):
        failed_method = "greedy" if scenario == "first-measured-repeat" else "prompt_lookup"
        expected_calls = [] if failed_method == "greedy" else [
            ("greedy", item.id) for item in prompts for _ in range(settings["repeats"])
        ]
        assert timed_calls == [*expected_calls, (failed_method, fixture.item.id)]
        failed, = [row for row in audit_rows if row["stage"] == "fp16_measured" and row["status"] == "fail"]
        assert (failed["method"], failed["prompt_id"], failed["repeat"]) == (failed_method, fixture.item.id, "1")
        details = json.loads(failed["details"])
        assert details["case_id"] != fixture.case_id and details["review"] is None
        assert details["outputs_sha256"] != fixture.review["outputs_sha256"]
        assert session["fp16_divergences"]["fp16_measured"] == 1
        archived = baseline / "audits" / f'{session["id"]}.csv'
        original_audit = (baseline / "correctness.csv").read_bytes()
        assert archived.read_bytes() == original_audit
        repeats = baseline / "repeats"
        checkpoints = list(repeats.rglob("*.json"))
        assert len(checkpoints) == len(timed_calls)
        checkpoint = repeats / failed_method / fixture.item.id / "1.json"
        original_checkpoint = checkpoint.read_bytes()
        assert json.loads(original_checkpoint)["payload"]["unit"]["token_ids"] == [3, 7, 2]
        assert not (checkpoint.parent / "2.json").exists()
        assert not (baseline / "summary.json").exists()

        if scenario == "first-measured-repeat":
            timed_calls.clear()
            with pytest.raises(RuntimeError, match=r"Measured fp16 audit.*greedy.*chat_qa-01.*repeat 1"):
                runner.execute(output, run_identity["git_commit"], fixture.review_path)
            resumed = json.loads((baseline / "run.json").read_bytes())["sessions"][-1]
            assert resumed["status"] == "blocked" and resumed["reused_repeats"] == 1
            assert not timed_calls
            assert checkpoint.read_bytes() == original_checkpoint
            assert archived.read_bytes() == original_audit
