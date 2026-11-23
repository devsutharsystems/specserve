"""Day 5 training validation on a fixed synthetic token corpus."""

import hashlib
import json
import math
import os
import platform
import random
import subprocess
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

from specserve.model.config import DraftConfig
from specserve.model.draft import DraftModel


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = ROOT / "configs/day05_overfit.yaml"
EVIDENCE_DIR = ROOT / "experiments/draft_a"


@contextmanager
def seeded_cpu(settings):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        with torch.random.fork_rng(devices=[]), torch.autocast("cpu", enabled=False):
            random.seed(settings["seed"])
            np.random.seed(settings["seed"])
            torch.random.default_generator.manual_seed(settings["seed"])
            torch.set_num_threads(settings["cpu_threads"])
            torch.use_deterministic_algorithms(settings["deterministic_algorithms"])
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def make_corpus(dataset, vocab_size):
    start = dataset["token_start"]
    width = dataset["cycle_length"]
    count = dataset["sequences"]
    length = dataset["tokens_per_sequence"]
    if not (0 <= start < start + width <= vocab_size and 1 <= count <= width and length >= 2):
        raise ValueError("Token cycle, sequence count, or sequence length is out of range")

    rng = torch.Generator(device="cpu").manual_seed(dataset["seed"])
    cycle = start + torch.randperm(width, generator=rng, device="cpu")
    offsets = torch.randperm(width, generator=rng, device="cpu")[:count]
    positions = offsets[:, None] + torch.arange(length, device="cpu")[None, :]
    return cycle[positions % width], rng


def next_token_pairs(tokens):
    return tokens[:, :-1], tokens[:, 1:]


@torch.no_grad()
def evaluate_corpus(model, corpus):
    loss_sum = 0.0
    target_count = 0
    for row in corpus:
        inputs, labels = next_token_pairs(row.unsqueeze(0))
        logits = model(inputs)
        loss_sum += F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum").item()
        target_count += labels.numel()

    mean_loss = loss_sum / target_count
    if not math.isfinite(mean_loss):
        raise ValueError("Nonfinite full-training-set mean loss")
    return mean_loss, target_count


def atomic_write(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


class OverfitRun:
    def __init__(self, architecture, settings):
        supported = {
            "batch_size": 1, "optimizer": "AdamW", "device": "cpu", "dtype": "float32",
            "scheduler": None, "grad_scaler": None,
        }
        if any(settings[name] != value for name, value in supported.items()):
            raise ValueError("Day 5 requires batch-one CPU FP32 AdamW with constant learning rate")
        self.settings = settings
        self.config = {"draft_model": asdict(architecture), "overfit": settings}
        self.corpus, self.data_rng = make_corpus(settings["dataset"], architecture.vocab_size)
        self.identity = {
            "config_sha256": hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest(),
            "corpus_sha256": hashlib.sha256(self.corpus.numpy().astype("<i8").tobytes()).hexdigest(),
        }
        self.model = DraftModel(architecture).to(device="cpu", dtype=torch.float32)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **settings["optimizer_kwargs"])
        self.step_losses = []
        self.evaluations = []

    @property
    def step(self):
        return len(self.step_losses)

    @property
    def next_sequence(self):
        return self.step % len(self.corpus)

    def train_step(self, tokens):
        inputs, labels = next_token_pairs(tokens.unsqueeze(0))
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        value = loss.item()
        if not math.isfinite(value):
            raise ValueError("Nonfinite training step loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.settings["grad_clip_norm"], error_if_nonfinite=True,
        )
        self.optimizer.step()
        return value

    def evaluate(self):
        self.model.eval()
        try:
            mean_loss, count = evaluate_corpus(self.model, self.corpus)
        finally:
            self.model.train()
        self.evaluations.append({"step": self.step, "target_count": count, "mean_loss": mean_loss})

    def train_until(self, step_limit, publish):
        if not self.evaluations:
            self.evaluate()
            publish()
        initial_loss = self.evaluations[0]["mean_loss"]
        if abs(initial_loss - math.log(self.model.config.vocab_size)) > self.settings["initial_loss_tolerance"]:
            raise ValueError(f"Unexpected initial mean loss: {initial_loss}")

        while self.step < step_limit and self.evaluations[-1]["mean_loss"] >= self.settings["loss_threshold"]:
            self.step_losses.append(self.train_step(self.corpus[self.next_sequence]))
            epoch_end = self.next_sequence == 0
            if epoch_end:
                self.evaluate()
            if epoch_end or self.step % self.settings["checkpoint_interval_steps"] == 0 or self.step == step_limit:
                publish()

    def save(self, path):
        # Primitive NumPy state keeps the checkpoint compatible with weights-only loading.
        algorithm, keys, index, has_gaussian, cached_gaussian = np.random.get_state()
        state = {
            **self.identity,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None,
            "grad_scaler": None,
            "python_rng": random.getstate(),
            "numpy_rng": (algorithm, keys.tolist(), index, has_gaussian, cached_gaussian),
            "torch_rng": torch.get_rng_state(),
            "data_rng": self.data_rng.get_state(),
            "step": self.step,
            "next_sequence": self.next_sequence,
            "step_losses": self.step_losses,
            "evaluations": self.evaluations,
        }
        atomic_write(path, lambda stream: torch.save(state, stream))

    def load(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if any(state.get(name) != value for name, value in self.identity.items()):
            raise ValueError("Checkpoint config or corpus fingerprint differs from this run")

        step = state["step"]
        steps_per_epoch = len(self.corpus)
        if (not 0 <= step <= self.settings["max_epochs"] * steps_per_epoch
                or len(state["step_losses"]) != step
                or state["next_sequence"] != step % steps_per_epoch):
            raise ValueError("Checkpoint data cursor is inconsistent with its update history")
        evaluated_steps = [result["step"] for result in state["evaluations"]]
        if evaluated_steps != list(range(0, step + 1, steps_per_epoch)):
            raise ValueError("Checkpoint epochs are inconsistent with its evaluation history")
        expected_targets = self.corpus.numel() - steps_per_epoch
        if any(result["target_count"] != expected_targets or not math.isfinite(result["mean_loss"])
               for result in state["evaluations"]):
            raise ValueError("Checkpoint evaluation must cover the full training corpus with finite loss")

        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step_losses = state["step_losses"]
        self.evaluations = state["evaluations"]
        algorithm, keys, index, has_gaussian, cached_gaussian = state["numpy_rng"]
        random.setstate(state["python_rng"])
        np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32), index, has_gaussian, cached_gaussian))
        torch.set_rng_state(state["torch_rng"])
        self.data_rng.set_state(state["data_rng"])


@pytest.fixture
def settings():
    return yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def tiny_setup(settings):
    full = DraftConfig.from_yaml(ROOT / settings["model_config"])
    architecture = replace(full, d_model=16, n_layer=2, n_head=4, hidden_dim=32, context_length=16, vocab_size=32)
    settings["dataset"].update(token_start=4, cycle_length=8, sequences=4, tokens_per_sequence=12)
    settings["checkpoint_interval_steps"] = 3
    return architecture, settings


def test_overfit_corpus_and_label_shift(settings):
    corpus, _ = make_corpus(settings["dataset"], vocab_size=32000)
    repeated, _ = make_corpus(settings["dataset"], vocab_size=32000)
    assert corpus.dtype == torch.long and corpus.shape == (100, 256)
    assert torch.equal(corpus, repeated)
    assert torch.equal(corpus.unique(), torch.arange(256, 384))
    assert corpus.unique(dim=0).shape == corpus.shape
    assert torch.equal(corpus[:, :128], corpus[:, 128:])
    inputs, targets = next_token_pairs(corpus)
    assert inputs.shape == targets.shape == (100, 255)
    assert targets.numel() == 25_500
    assert torch.equal(inputs, corpus[:, :255])
    assert torch.equal(targets, corpus[:, 1:256])
    assert (inputs != targets).all()


def test_overfit_training_step_uses_gradients_and_updates_parameters(tiny_setup):
    architecture, settings = tiny_setup
    with seeded_cpu(settings):
        run = OverfitRun(architecture, settings)
        before = run.model.token_embedding.weight.detach().clone()
        loss = run.train_step(run.corpus[0])
        assert math.isfinite(loss)
        assert not torch.equal(before, run.model.token_embedding.weight)
        assert run.model.parameter_counts()["total"] == sum(p.numel() for p in run.model.parameters())
        for parameter in run.model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_overfit_evaluation_scores_every_shifted_target(tiny_setup):
    architecture, settings = tiny_setup
    with seeded_cpu(settings), torch.no_grad():
        run = OverfitRun(architecture, settings)
        run.evaluate()
        log_prob = run.model(run.corpus[:, :-1]).log_softmax(dim=-1)
        nll = -log_prob.gather(-1, run.corpus[:, 1:].unsqueeze(-1))
        assert run.evaluations[0]["mean_loss"] == pytest.approx(nll.double().mean().item(), abs=1e-6)
        assert run.evaluations[0]["target_count"] == nll.numel()


def rng_draw(run):
    return (random.random(), np.random.random(), torch.rand(()).item(),
            torch.rand((), generator=run.data_rng).item())


def test_overfit_cpu_resume_is_exact(tiny_setup, tmp_path):
    architecture, settings = tiny_setup
    checkpoint = tmp_path / "checkpoint.pt"
    with seeded_cpu(settings):
        reference = OverfitRun(architecture, settings)
        reference.train_until(3, lambda: None)
        next_tokens = reference.corpus[reference.next_sequence].clone()
        expected_draw = rng_draw(reference)
        reference.train_until(8, lambda: None)

    with seeded_cpu(settings):
        interrupted = OverfitRun(architecture, settings)
        interrupted.train_until(3, lambda: interrupted.save(checkpoint))
        rng_draw(interrupted)
        resumed = OverfitRun(architecture, settings)
        resumed.load(checkpoint)
        assert resumed.step == resumed.next_sequence == 3
        assert torch.equal(resumed.corpus[resumed.next_sequence], next_tokens)
        assert rng_draw(resumed) == expected_draw
        resumed.train_until(8, lambda: resumed.save(checkpoint))

    assert reference.step == resumed.step == 8
    assert resumed.step_losses == reference.step_losses
    assert resumed.evaluations == reference.evaluations
    torch.testing.assert_close(resumed.model.state_dict(), reference.model.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(resumed.optimizer.state_dict(), reference.optimizer.state_dict(), rtol=0, atol=0)


def test_overfit_rejects_mismatched_checkpoint(tiny_setup, tmp_path):
    architecture, settings = tiny_setup
    checkpoint = tmp_path / "checkpoint.pt"
    with seeded_cpu(settings):
        run = OverfitRun(architecture, settings)
        run.train_until(1, lambda: run.save(checkpoint))
        other = OverfitRun(architecture, {**settings, "dataset": {**settings["dataset"], "seed": 7}})
        with pytest.raises(ValueError, match="fingerprint differs"):
            other.load(checkpoint)
        assert other.step == 0


def test_failed_checkpoint_write_preserves_previous_state(tiny_setup, tmp_path, monkeypatch):
    architecture, settings = tiny_setup
    checkpoint = tmp_path / "checkpoint.pt"

    def fail_after_write(state, stream):
        stream.write(b"partial checkpoint")
        raise OSError("injected save failure")

    with seeded_cpu(settings):
        run = OverfitRun(architecture, settings)
        run.train_until(1, lambda: run.save(checkpoint))
        before = checkpoint.read_bytes()
        monkeypatch.setattr(torch, "save", fail_after_write)
        with pytest.raises(OSError, match="injected save failure"):
            run.save(checkpoint)
    assert checkpoint.read_bytes() == before
    assert list(tmp_path.iterdir()) == [checkpoint]


@pytest.mark.skipif(os.environ.get("SPECSERVE_RUN_OVERFIT") != "1", reason="full Draft-A gate is opt-in")
def test_draft_a_overfit_gate(settings):
    checkpoint = Path(os.environ["SPECSERVE_OVERFIT_CHECKPOINT"]).expanduser().resolve()
    if checkpoint.is_relative_to(ROOT):
        raise ValueError("Keep Day 5 checkpoints outside the repository")
    resume = os.environ.get("SPECSERVE_OVERFIT_RESUME") == "1"
    if resume != checkpoint.is_file():
        raise ValueError("Resume requires an existing checkpoint and a fresh run requires a new path")

    architecture = DraftConfig.from_yaml(ROOT / settings["model_config"])
    with seeded_cpu(settings):
        run = OverfitRun(architecture, settings)
        assert run.model.parameter_counts()["total"] == 35_658_240
        assert run.corpus.shape == (100, 256)
        assert settings["max_epochs"] == 3 and settings["loss_threshold"] == 0.1
        if resume:
            run.load(checkpoint)

        metadata = {
            "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "training_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            **run.identity,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "pytest": pytest.__version__,
            "pyyaml": yaml.__version__,
            "trainable_parameters": run.model.parameter_counts()["total"],
        }
        snapshot = yaml.safe_dump(run.config, sort_keys=False)
        atomic_write(EVIDENCE_DIR / "day05_overfit_config.yaml", lambda stream: stream.write(snapshot.encode()))

        def evidence(verdict, error=None):
            log = yaml.safe_dump({
                **metadata,
                "verdict": verdict,
                "error": error,
                "optimizer_steps_completed": run.step,
                "full_corpus_evaluations": run.evaluations,
                "training_step_losses": run.step_losses,
            }, sort_keys=False)
            atomic_write(EVIDENCE_DIR / "day05_overfit.txt", lambda stream: stream.write(log.encode()))

        def publish():
            run.save(checkpoint)
            evidence("INCOMPLETE")
            last = run.evaluations[-1]
            print(f"saved step {run.step}; full mean at step {last['step']}: {last['mean_loss']:.17g}", flush=True)

        try:
            evidence("INCOMPLETE")
            run.train_until(settings["max_epochs"] * len(run.corpus), publish)
            final = run.evaluations[-1]
            assert final["target_count"] == 25_500 and final["step"] == run.step
        except BaseException as error:
            evidence("INCOMPLETE", f"{type(error).__name__}: {error}")
            raise

        passed = final["mean_loss"] < 0.1
        evidence("PASSED" if passed else "FAILED")
        print(f"full_training_set_mean_loss={final['mean_loss']:.17g}, targets={final['target_count']}")
        assert passed, "Full training corpus must have mean cross-entropy below 0.1"
