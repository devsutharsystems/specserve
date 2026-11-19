"""Count Day 2a UltraChat candidates using the approved fixed filters."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from tokenizers import __version__ as tokenizers_version
from transformers import AutoTokenizer, __version__ as transformers_version

SEED = 20261118
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
DATASET_SPLIT = "test_sft"
SOURCE = f"{DATASET_ID}:{DATASET_SPLIT}"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
MAX_PROMPT_TOKENS = 896
MAX_NEW_TOKENS = 128
CATEGORY_SLUGS = {
    "chat Q&A": "chat_qa",
    "summarization/extraction": "summarization_extraction",
    "open-ended writing": "open_ended_writing",
    "code": "code",
}
CATEGORY_ORDER = (
    "chat Q&A",
    "summarization/extraction",
    "open-ended writing",
    "code",
)

EXPECTED_RENDERED = "<|user|>\nWhat is 2 + 2?</s>\n<|assistant|>\n"
EXAMPLE_MESSAGES = [{"role": "user", "content": "What is 2 + 2?"}]

CODE_VERBS = r"\b(?:write|implement|debug|fix|refactor|optimize|convert)\b"
CODE_TERMS = (
    r"(?:\bpython\b|\btypescript\b|\bjavascript\b|\bjava\b|"
    r"(?<!\w)c\+\+(?!\w)|(?<!\w)c#(?!\w)|\bphp\b|\bkotlin\b|\bmatlab\b|"
    r"\bperl\b|\bsql\b|\bhtml\b|\bcss\b|\bbash\b|\bregex\b|\bsource\s+code\b)"
)
SUMMARY_MARKER = (
    r"(?:\bpassage\s*:|\btext\s*:|\barticle\s*:|\bdocument\s*:|\bcontent\s*:|"
    r"\banswer\s+according\s+to\b|\bgiven\s+the\s+text\b|"
    r"\bbased\s+on\s+the\s+text\b|\bbased\s+on\s+the\s+passage\b|"
    r"\bread\s+the\s+passage\b|\baccording\s+to\s+the\s+text\b)"
)
SUMMARY_TASK = r"(?:\bsummarize\b|\bsummary\b|\bextract\b|\bkey\s+points\b|\bmain\s+ideas\b)"
WRITING_VERB = r"(?:\bwrite\b|\bcompose\b|\bdraft\b|\bcreate\b|\bgenerate\b|\btell\b)"
WRITING_OBJECT = (
    r"(?:\bstory\b|\bessay\b|\bpoem\b|\bletter\b|\bblog\b|\bdialogue\b|"
    r"\bspeech\b|\bdescription\b|\barticle\b|\bcreative\b)"
)
QUESTION_START = (
    r"^\s*(?:\bwho\b|\bwhat\b|\bwhen\b|\bwhere\b|\bwhy\b|\bhow\b|\bcan\b|"
    r"\bcould\b|\bwould\b|\bis\b|\bare\b|\bdo\b|\bdoes\b)"
)
FENCED_BLOCK = r"```[\s\S]*?```"

def validate_revision_format(name: str, revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise AssertionError(
            f"{name} revision must be 40 lowercase hexadecimal characters: {revision}"
        )


def resolve_and_validate_revisions() -> None:
    validate_revision_format("dataset", DATASET_REVISION)
    validate_revision_format("model", MODEL_REVISION)
    api = HfApi()
    dataset_info = api.repo_info(
        DATASET_ID, revision=DATASET_REVISION, repo_type="dataset"
    )
    model_info = api.repo_info(
        MODEL_ID, revision=MODEL_REVISION, repo_type="model"
    )
    resolved_dataset = dataset_info.sha
    resolved_model = model_info.sha
    print(f"resolved_dataset_sha={resolved_dataset}")
    print(f"resolved_model_sha={resolved_model}")
    if resolved_dataset != DATASET_REVISION:
        raise AssertionError(
            f"Dataset revision mismatch: constant={DATASET_REVISION}, "
            f"resolved={resolved_dataset}"
        )
    if resolved_model != MODEL_REVISION:
        raise AssertionError(
            f"Model revision mismatch: constant={MODEL_REVISION}, "
            f"resolved={resolved_model}"
        )


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def is_english(text: str) -> bool:
    non_whitespace = [char for char in text if not char.isspace()]
    return bool(non_whitespace) and (
        sum(char.isascii() for char in non_whitespace) / len(non_whitespace) >= 0.95
    )


def category_for(prompt: str) -> str | None:
    matches = category_matches(prompt)
    for category in (
        "code",
        "summarization/extraction",
        "open-ended writing",
        "chat Q&A",
    ):
        if category in matches:
            return category
    return None


def writing_object_within_8_words(text: str) -> bool:
    words = re.findall(r"\b[\w+#.-]+\b", text)
    verb_match = re.compile(r"write|compose|draft|create|generate|tell")
    object_match = re.compile(
        r"story|essay|poem|letter|blog|dialogue|speech|description|article|creative"
    )
    for index, word in enumerate(words):
        if verb_match.fullmatch(word):
            for following_index in range(index + 1, min(index + 9, len(words))):
                if object_match.fullmatch(words[following_index]):
                    return True
    return False


def category_matches(prompt: str) -> set[str]:
    text = normalize(prompt)
    matches: set[str] = set()
    if re.match(r"^\d+\s*[.)]\s", text):
        return matches
    if re.search(CODE_VERBS, text) and (
        re.search(CODE_TERMS, text) or re.search(FENCED_BLOCK, text)
    ):
        matches.add("code")
    pasted_source = bool(re.search(SUMMARY_MARKER, text)) or len(
        re.sub(r"\s+", "", text)
    ) >= 1000
    task_cue = bool(re.search(SUMMARY_TASK, text)) or "?" in text
    if pasted_source and task_cue:
        matches.add("summarization/extraction")
    if re.search(WRITING_VERB, text) and writing_object_within_8_words(text):
        matches.add("open-ended writing")
    non_whitespace_chars = len(re.sub(r"\s+", "", text))
    starts_with_question_word = re.match(
        r"^(?:who|what|when|where|why|how|which|can|could|would|is|are|do|does)\b",
        text,
    )
    if non_whitespace_chars <= 500 and starts_with_question_word and text.endswith("?"):
        matches.add("chat Q&A")
    return matches


def first_user_text(prompt_field: object) -> str:
    if isinstance(prompt_field, str):
        return prompt_field
    if not isinstance(prompt_field, Sequence) or isinstance(prompt_field, (bytes, bytearray)):
        raise TypeError(f"Unexpected prompt field type: {type(prompt_field)!r}")
    for message in prompt_field:
        if isinstance(message, Mapping) and message.get("role") == "user":
            content = message.get("content")
            if not isinstance(content, str):
                raise TypeError("The first user message content is not a string")
            return content
    raise ValueError("Prompt has no user turn")


def input_ids_from_template(tokenizer: AutoTokenizer, messages: list[dict[str, str]]) -> list[int]:
    result = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    if isinstance(result, Mapping):
        input_ids = result["input_ids"]
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
    else:
        input_ids = result
    if not isinstance(input_ids, list) or (
        input_ids and isinstance(input_ids[0], list)
    ):
        raise TypeError(f"Expected one list of input IDs, got {type(input_ids)!r}")
    if tokenizer.bos_token_id is not None and tokenizer.bos_token_id not in input_ids:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        with_special_tokens = tokenizer(
            rendered, add_special_tokens=True
        )["input_ids"]
        if not isinstance(with_special_tokens, list):
            raise TypeError("Tokenizer did not return a list of input IDs")
        input_ids = with_special_tokens
    return input_ids


def run_template_checks(tokenizer: AutoTokenizer) -> None:
    rendered = tokenizer.apply_chat_template(
        EXAMPLE_MESSAGES, tokenize=False, add_generation_prompt=True
    )
    print(f"template_expected={EXPECTED_RENDERED!r}")
    print(f"template_actual={rendered!r}")
    print(f"template_expected_match={rendered == EXPECTED_RENDERED}")
    print(f"template_has_default_system_text={'<|system|>' in rendered}")
    print(f"template_user_turn_only={rendered.startswith('<|user|>')}")
    rendered_bos_text = tokenizer.bos_token in rendered if tokenizer.bos_token else False
    print(f"rendered_contains_bos_text={rendered_bos_text}")
    template_ids = tokenizer.apply_chat_template(
        EXAMPLE_MESSAGES, tokenize=True, add_generation_prompt=True
    )
    if isinstance(template_ids, Mapping):
        template_ids = template_ids["input_ids"]
        if hasattr(template_ids, "tolist"):
            template_ids = template_ids.tolist()
    print(f"template_tokenize_true_first5={template_ids[:5]}")
    special_true_ids = tokenizer(rendered, add_special_tokens=True)["input_ids"]
    special_false_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    print(f"tokenizer_special_true_first5={special_true_ids[:5]}")
    print(f"tokenizer_special_false_first5={special_false_ids[:5]}")
    bos_id = tokenizer.bos_token_id
    print(f"bos_token={tokenizer.bos_token!r}")
    print(f"bos_token_id={bos_id}")
    ids = special_true_ids
    print(f"bos_count={ids.count(bos_id) if bos_id is not None else 'unavailable'}")
    if bos_id is not None and not rendered_bos_text and special_true_ids[:1] == [bos_id]:
        print("bos_origin=tokenizer special-token addition")
    elif bos_id is not None and rendered_bos_text and bos_id in template_ids:
        print("bos_origin=chat template")
    else:
        print("bos_origin=unresolved")
    if rendered != EXPECTED_RENDERED:
        raise AssertionError("The hand-written chat-template expectation did not match")
    if "<|system|>" in rendered:
        raise AssertionError("The tokenizer added default system text")
    if bos_id is not None and ids.count(bos_id) != 1:
        raise AssertionError("Expected exactly one BOS token")


def git_is_tracked(path: Path) -> bool:
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def train_sft_first_user_prompts(dataset_id: str, revision: str) -> set[str]:
    dataset = load_dataset(
        dataset_id,
        split="train_sft",
        revision=revision,
    )
    return {
        first_user_text(row["prompt"])
        for row in dataset
        if row.get("prompt") is not None
    }


def near_duplicate_anchors(prompt: str) -> list[str]:
    normalized = normalize(prompt)
    if len(normalized) < 240:
        return [normalized[:80]] if len(normalized) >= 80 else [normalized]
    return [
        normalized[int((len(normalized) * position) // 100) : int((len(normalized) * position) // 100) + 80]
        for position in (25, 50, 75)
    ]


def validate_selected(
    records: list[dict[str, object]],
    train_prompts: set[str],
    train_normalized: set[str],
) -> bool:
    all_ok = True

    def check(name: str, passed: bool) -> None:
        nonlocal all_ok
        print(f"VALIDATION {'PASS' if passed else 'FAIL'} {name}")
        all_ok = all_ok and passed

    counts = Counter(str(record["category"]) for record in records)
    check("25_per_category_and_100_total", len(records) == 100 and all(counts[slug] == 25 for slug in (
        "chat_qa", "summarization_extraction", "open_ended_writing", "code"
    )))
    field_names = [
        "id", "category", "source", "source_id", "prompt", "max_new_tokens", "prompt_tokens"
    ]
    check("exact_field_set_and_order", all(list(record.keys()) == field_names for record in records))
    check("max_new_tokens_128", all(record["max_new_tokens"] == 128 for record in records))
    check("source_value", all(record["source"] == SOURCE for record in records))
    check("prompt_tokens_bounds", all(
        0 < int(record["prompt_tokens"]) <= 896
        and int(record["prompt_tokens"]) + 128 <= 1024
        for record in records
    ))
    ids = [str(record["id"]) for record in records]
    source_ids = [str(record["source_id"]) for record in records]
    prompts = [str(record["prompt"]) for record in records]
    check("unique_ids", len(ids) == len(set(ids)))
    check("unique_source_ids", len(source_ids) == len(set(source_ids)))
    check("unique_exact_prompt_texts", len(prompts) == len(set(prompts)))


    no_exact = all(str(record["prompt"]) not in train_prompts for record in records)
    no_normalized = all(normalize(str(record["prompt"])) not in train_normalized for record in records)
    check("no_exact_train_sft_leak", no_exact)
    check("no_normalized_train_sft_leak", no_normalized)
    return all_ok


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print(f"seed={SEED}")
    resolve_and_validate_revisions()
    print(f"dataset={DATASET_ID}")
    print(f"dataset_revision={DATASET_REVISION}")
    print(f"dataset_split={DATASET_SPLIT}")
    print(f"source={SOURCE}")
    print(f"model={MODEL_ID}")
    print(f"model_revision={MODEL_REVISION}")
    print(f"transformers_version={transformers_version}")
    print(f"tokenizers_version={tokenizers_version}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, use_fast=True, cache_dir=args.cache_dir
    )
    print("tokenizer_version=TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    run_template_checks(tokenizer)

    dataset = load_dataset(
        DATASET_ID,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )

    regression_checks = [
        ("00783392b9b329dcb0562e6a27a546bf0d1188fdb8b16f571e9d9624340f8487", None, "code"),
        ("000a8504dc7ffee325a9c76a553bd8b7e8ca4bc5b3652433f7ee6a7e18ce351f", "summarization/extraction", None),
        ("00256379f6e5fd0eb2ecf9d25b93ddd43c045d1ed53658f41a95e909f6a7372f", "summarization/extraction", None),
        ("0011149885ffa18178b2eb380f87c400396b25e612791d66d71ad1cfa2f263b0", "summarization/extraction", None),
        ("006710b359b1b23416131ea36e7eaa30e811daf6e87682cf757203917194a993", "summarization/extraction", None),
        ("007d069bc2955dadf58e0e7b1ef4cb878017f4c92c7dd7b0ed2bbafe2900293f", None, "open-ended writing"),
        ("002b2423583af01340a9966f4f2d83598f1b5fb8cf0aa77eb9d3e6f9b3a56f30", "chat Q&A", None),
        ("0172f9c1b5ad3f4f38479239c4b23fd5d1abe1cd58d68f3b7db101d407f7d14c", "code", None),
        ("2da27db4fa1c5810cba9155519249034985c6f6452e59e180edf89b53d95daf9", "excluded", None),
        ("c237edabca60bf1959c8c033df41876863ffbb37e2e788748737b71a15bedf8a", None, "chat Q&A"),
        ("f764fa870e6f7217dd6fe6b0d380792adc487efaaf782a9d8c925f11a59839bf", None, "chat Q&A"),
    ]
    rows_by_prompt_id = {
        str(row["prompt_id"]): row for row in dataset
    }
    for prompt_id, expected, forbidden in regression_checks:
        row = rows_by_prompt_id.get(prompt_id)
        if row is None:
            print(f"REGRESSION FAIL id={prompt_id} reason=prompt id not found")
            continue
        prompt = first_user_text(row["prompt"])
        actual = category_for(prompt) if is_english(prompt) else None
        passed = (
            (
                actual is None
                if expected == "excluded"
                else expected is None or actual == expected
            )
            and (forbidden is None or actual != forbidden)
        )
        print(f"REGRESSION {'PASS' if passed else 'FAIL'} id={prompt_id}")
        if not passed:
            print(f"regression_prompt_id={prompt_id}")
            print(f"regression_prompt={prompt[:300]!r}")
            print(f"regression_actual_category={actual}")
            print(f"regression_rules_fired={sorted(category_matches(prompt))}")

    candidates: dict[str, list[dict[str, object]]] = Counter()
    cap_excluded: dict[str, int] = Counter()
    dropped_english = 0
    overlap_pairs: dict[tuple[str, str], int] = Counter()
    category_names = (
        "code",
        "summarization/extraction",
        "open-ended writing",
        "chat Q&A",
    )
    for row in dataset:
        prompt = first_user_text(row["prompt"])
        if not is_english(prompt):
            dropped_english += 1
            continue
        matches = category_matches(prompt)
        for first_index, first_category in enumerate(category_names):
            for second_category in category_names[first_index + 1 :]:
                if first_category in matches and second_category in matches:
                    overlap_pairs[(first_category, second_category)] += 1
        category = category_for(prompt)
        if category is None:
            continue
        prompt_id = str(row["prompt_id"])
        token_ids = input_ids_from_template(
            tokenizer, [{"role": "user", "content": prompt}]
        )
        prompt_tokens = len(token_ids)
        if prompt_tokens > MAX_PROMPT_TOKENS:
            cap_excluded[category] += 1
            continue
        candidates.setdefault(category, []).append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "prompt_tokens": prompt_tokens,
            }
        )

    print(f"english_filter_dropped={dropped_english}")
    for category in (
        "chat Q&A",
        "summarization/extraction",
        "open-ended writing",
        "code",
    ):
        rows = sorted(candidates.get(category, []), key=lambda row: row["prompt_id"])
        print(f"candidate_count[{category}]={len(rows)}")
        print(f"prompt_896_cap_excluded[{category}]={cap_excluded.get(category, 0)}")
        random_rows = list(rows)
        random.Random(SEED).shuffle(random_rows)
        for index, row in enumerate(random_rows[:10], start=1):
            preview = str(row["prompt"]).replace("\n", " ")[:300]
            print(
                f"seeded_random_sample[{category}][{index}] "
                f"prompt_id={row['prompt_id']} prompt_tokens={row['prompt_tokens']} "
                f"text={preview!r}"
            )
            if category == "summarization/extraction":
                tail = str(row["prompt"])[-200:].replace("\n", " ")
                print(f"summarization_tail[{index}]={tail!r}")
        shortest = sorted(rows, key=lambda row: (int(row["prompt_tokens"]), str(row["prompt_id"])))[:5]
        longest = sorted(
            rows,
            key=lambda row: (-int(row["prompt_tokens"]), str(row["prompt_id"])),
        )[:5]
        for index, row in enumerate(shortest, start=1):
            preview = str(row["prompt"]).replace("\n", " ")[:300]
            print(
                f"shortest[{category}][{index}] "
                f"prompt_id={row['prompt_id']} prompt_tokens={row['prompt_tokens']} "
                f"text={preview!r}"
            )
        for index, row in enumerate(longest, start=1):
            preview = str(row["prompt"]).replace("\n", " ")[:300]
            print(
                f"longest[{category}][{index}] "
                f"prompt_id={row['prompt_id']} prompt_tokens={row['prompt_tokens']} "
                f"text={preview!r}"
            )
    for pair in (
        ("code", "summarization/extraction"),
        ("code", "open-ended writing"),
        ("code", "chat Q&A"),
        ("summarization/extraction", "open-ended writing"),
        ("summarization/extraction", "chat Q&A"),
        ("open-ended writing", "chat Q&A"),
    ):
        print(
            f"overlap[{pair[0]}][{pair[1]}]={overlap_pairs.get(pair, 0)}"
        )

    if not args.select:
        return

    output_path = Path("bench/prompts.jsonl")
    if git_is_tracked(output_path):
        raise RuntimeError("Refusing to write: bench/prompts.jsonl is already tracked")
    if output_path.exists() and not args.overwrite:
        raise RuntimeError("Refusing to write: bench/prompts.jsonl exists; pass --overwrite")

    train_prompts = train_sft_first_user_prompts(DATASET_ID, DATASET_REVISION)
    train_normalized = {normalize(prompt) for prompt in train_prompts}
    train_joined = "\n".join(sorted(normalize(prompt) for prompt in train_prompts))

    def select_once(
        candidates_override: dict[str, list[dict[str, object]]] | None = None,
    ) -> tuple[list[dict[str, object]], list[tuple[str, str]]]:
        selected: list[dict[str, object]] = []
        skipped: list[tuple[str, str]] = []
        accepted_exact: set[str] = set()
        accepted_normalized: set[str] = set()
        accepted_source_ids: set[str] = set()

        for category in CATEGORY_ORDER:
            slug = CATEGORY_SLUGS[category]
            source_candidates = (
                candidates_override.get(category, [])
                if candidates_override is not None
                else candidates.get(category, [])
            )
            shuffled = sorted(
                source_candidates,
                key=lambda row: str(row["prompt_id"]),
            )
            shuffled = list(shuffled)
            random.Random(SEED).shuffle(shuffled)
            accepted_in_category = 0
            for row in shuffled:
                prompt_id = str(row["prompt_id"])
                prompt = str(row["prompt"])
                prompt_normalized = normalize(prompt)
                if prompt_id in accepted_source_ids:
                    skipped.append((prompt_id, "duplicate prompt_id"))
                    continue
                if prompt in accepted_exact:
                    skipped.append((prompt_id, "duplicate exact text"))
                    continue
                if prompt_normalized in accepted_normalized:
                    skipped.append((prompt_id, "duplicate normalized text"))
                    continue
                if prompt in train_prompts:
                    skipped.append((prompt_id, "exact train_sft leak"))
                    continue
                if prompt_normalized in train_normalized:
                    skipped.append((prompt_id, "normalized train_sft leak"))
                    continue
                index = accepted_in_category + 1
                record = {
                    "id": f"{slug}-{index:02d}",
                    "category": slug,
                    "source": SOURCE,
                    "source_id": prompt_id,
                    "prompt": prompt,
                    "max_new_tokens": 128,
                    "prompt_tokens": int(row["prompt_tokens"]),
                }
                selected.append(record)
                accepted_exact.add(prompt)
                accepted_normalized.add(prompt_normalized)
                accepted_source_ids.add(prompt_id)
                accepted_in_category += 1
                if accepted_in_category == 25:
                    break
            if accepted_in_category < 25:
                raise RuntimeError(
                    f"Only {accepted_in_category} prompts available for {category}; "
                    "fewer than 25"
                )
        return selected, skipped

    selected_first, skipped = select_once()
    reversed_candidates = {
        category: list(reversed(
            sorted(
                candidates.get(category, []),
                key=lambda row: str(row["prompt_id"]),
            )
        ))
        for category in CATEGORY_ORDER
    }
    selected_second, skipped_second = select_once(reversed_candidates)
    independent_ok = (
        selected_first == selected_second
        and skipped == skipped_second
    )
    print(
        "VALIDATION "
        + ("PASS" if independent_ok else "FAIL")
        + " selection_independent_of_candidate_order"
    )
    if not independent_ok:
        raise RuntimeError(
            "Selection changed when candidate input order was reversed"
        )

    if not validate_selected(selected_first, train_prompts, train_normalized):
        raise RuntimeError("Selected benchmark validation failed")

    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in selected_first:
            output_file.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )

    print(f"sha256={sha256_file(output_path)}")
    for category in CATEGORY_ORDER:
        slug = CATEGORY_SLUGS[category]
        selected_ids = [
            str(record["source_id"])
            for record in selected_first
            if record["category"] == slug
        ]
        print(f"selected[{category}]={','.join(selected_ids)}")
    print(f"report_skipped_candidate_count={len(skipped)}")
    for prompt_id, reason in skipped:
        print(f"skipped prompt_id={prompt_id} reason={reason}")

    for category in CATEGORY_ORDER:
        slug = CATEGORY_SLUGS[category]
        selected_category = [
            record for record in selected_first if record["category"] == slug
        ]
        prompt_lengths = sorted(
            int(record["prompt_tokens"]) for record in selected_category
        )
        median = prompt_lengths[len(prompt_lengths) // 2]
        replacement_count = sum(
            "\ufffd" in str(record["prompt"]) for record in selected_category
        )
        non_ascii_count = sum(
            any(ord(char) > 127 for char in str(record["prompt"]))
            for record in selected_category
        )
        print(
            f"report[{slug}] min_prompt_tokens={prompt_lengths[0]} "
            f"median_prompt_tokens={median} max_prompt_tokens={prompt_lengths[-1]} "
            f"replacement_character_count={replacement_count} "
            f"non_ascii_character_count={non_ascii_count}"
        )

    near_duplicate_hits: list[tuple[str, str, list[str]]] = []
    for record in selected_first:
        hits = []
        for anchor in near_duplicate_anchors(str(record["prompt"])):
            if anchor and train_joined.find(anchor) != -1:
                hits.append(anchor)
        if hits:
            near_duplicate_hits.append(
                (str(record["source_id"]), str(record["id"]), hits)
            )
    print(f"near_duplicate_hit_count={len(near_duplicate_hits)}")
    for prompt_id, record_id, hits in near_duplicate_hits:
        print(
            f"near_duplicate_hit record_id={record_id} prompt_id={prompt_id} "
            f"anchor_count={len(hits)}"
        )

    print("=== REVIEW ===")
    for record in selected_first:
        print(f"[{record['id']}] (prompt_tokens={record['prompt_tokens']})")
        prompt = str(record["prompt"])
        if (
            record["category"] == "summarization_extraction"
            and len(prompt) > 800
        ):
            print(prompt[:300] + " [...] " + prompt[-300:])
        else:
            print(prompt)


if __name__ == "__main__":
    main()
