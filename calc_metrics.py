import argparse
import json
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder.ctc_text_encoder import CTCTextEncoder


def calculate(ground_truth: Path, predictions: Path) -> dict:
    targets = {p.stem: p for p in ground_truth.glob("*.txt")}
    outputs = {p.stem: p for p in predictions.glob("*.txt")}

    if not targets:
        raise ValueError(f"No reference .txt files in {ground_truth}")

    if targets.keys() != outputs.keys():
        missing = sorted(targets.keys() - outputs.keys())
        extra = sorted(outputs.keys() - targets.keys())
        raise ValueError(
            f"Prediction IDs differ: missing={missing[:5]}, extra={extra[:5]}"
        )

    char_errors = word_errors = chars = words = 0
    total_cer = total_wer = 0.0

    for utterance_id, path in sorted(targets.items()):
        target = CTCTextEncoder.normalize_text(path.read_text(encoding="utf-8")).strip()
        prediction = CTCTextEncoder.normalize_text(
            outputs[utterance_id].read_text(encoding="utf-8")
        ).strip()
        cer, wer = calc_cer(target, prediction), calc_wer(target, prediction)

        total_cer += cer
        total_wer += wer

        char_errors += round(cer * max(1, len(target)))
        word_errors += round(wer * max(1, len(target.split())))

        chars += len(target)
        words += len(target.split())

    return {
        "recordings": len(targets),
        "CER": total_cer / len(targets),
        "WER": total_wer / len(targets),
        "corpus_CER": char_errors / max(1, chars),
        "corpus_WER": word_errors / max(1, words),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score matching UTF-8 transcription files. Rates are fractions, not percentages."
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)

    args = parser.parse_args()
    result = calculate(args.ground_truth, args.predictions)
    text = json.dumps(result, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
