import argparse
import json
from pathlib import Path

import sentencepiece as spm

from src.text_encoder import CTCTextEncoder
from src.utils.io_utils import ROOT_PATH


def train_bpe(
    index_path: Path,
    model_prefix: Path,
    vocab_size: int,
) -> None:
    with index_path.open(encoding="utf-8") as file:
        index = json.load(file)

    data = [CTCTextEncoder.normalize_text(item["text"]).strip() for item in index]
    data = [text for text in data if text]
    if not data:
        raise ValueError("The training index has no nonempty texts")

    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(f"{model_prefix}.txt")
    corpus_path.write_text("\n".join(data) + "\n", encoding="utf-8")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=1.0,
        pad_id=0,
        pad_piece="<blank>",
        unk_id=1,
        bos_id=-1,
        eos_id=-1,
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,
        shuffle_input_sentence=False,
        num_threads=2,
        minloglevel=1,
    )

    tokenizer = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    assert tokenizer.pad_id() == 0
    ids = tokenizer.encode(data[0], out_type=int)
    assert 0 not in ids and tokenizer.decode(ids) == data[0]

    print(f"Training texts: {len(data)}")
    print(f"Vocabulary size: {tokenizer.vocab_size()}")
    print(f"Model: {model_prefix}.model")
    print(f"Vocabulary: {model_prefix}.vocab")
    print(f"Text: {data[0]}")
    print(f"Pieces: {tokenizer.encode(data[0], out_type=str)}")
    print(f"IDs: {ids}")
    print(f"Decoded: {tokenizer.decode(ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-path",
        type=Path,
        default=ROOT_PATH / "data/datasets/librispeech/train-clean-100_index.json",
    )
    parser.add_argument(
        "--model-prefix",
        type=Path,
        default=ROOT_PATH / "data/tokenizers/bpe_256",
    )
    parser.add_argument("--vocab-size", type=int, default=256)
    args = parser.parse_args()
    train_bpe(args.index_path, args.model_prefix, args.vocab_size)
