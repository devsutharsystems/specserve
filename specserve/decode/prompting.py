"""Shared TinyLlama prompt formatting for the frozen benchmark."""

from transformers import AutoTokenizer

TARGET_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TARGET_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"


def load_tokenizer(*, local_files_only: bool = False):
    return AutoTokenizer.from_pretrained(
        TARGET_MODEL_ID, revision=TARGET_REVISION, local_files_only=local_files_only
    )


def prompt_token_ids(tokenizer, prompt: str) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # The tokenizer adds BOS. The pinned chat template supplies the role markers.
    ids = tokenizer(rendered, add_special_tokens=True)["input_ids"]
    bos = tokenizer.bos_token_id
    if not ids or ids[0] != bos or ids.count(bos) != 1:
        raise ValueError("Expected exactly one BOS at the start of the prompt")
    return ids
