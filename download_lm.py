import argparse
import gzip
import shutil
import subprocess
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent


def download_lm(output_dir: Path, order: int = 3) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    name = "3-gram.pruned.1e-7" if order == 3 else "4-gram"
    arpa = output_dir / f"{name}.lower.arpa"
    vocab = output_dir / ("unigrams.txt" if order == 3 else "4-gram.unigrams.txt")
    if arpa.exists() and vocab.exists():
        return

    archive = output_dir / f"{name}.arpa.gz"
    if not archive.exists():
        partial = archive.with_suffix(".gz.part")
        url = f"https://www.openslr.org/resources/11/{archive.name}"
        if shutil.which("aria2c"):
            subprocess.run(
                [
                    "aria2c",
                    "-c",
                    "-x8",
                    "-s8",
                    "-k1M",
                    "--file-allocation=none",
                    "--auto-file-renaming=false",
                    "-d",
                    str(output_dir),
                    "-o",
                    partial.name,
                    url,
                ],
                check=True,
            )
        else:
            urlretrieve(url, partial)
        partial.replace(archive)

    partial = arpa.with_suffix(".arpa.part")
    with gzip.open(archive, "rt", encoding="utf-8") as source, partial.open(
        "w", encoding="utf-8"
    ) as target, vocab.open("w", encoding="utf-8") as words:
        in_unigrams = False
        for line in source:
            line = line.lower()
            target.write(line)
            if line.startswith("\\"):
                in_unigrams = line.strip() == "\\1-grams:"
            elif in_unigrams and line.strip():
                word = line.split()[1]
                if not word.startswith("<"):
                    words.write(word + "\n")
    partial.replace(arpa)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/lm/librispeech")
    parser.add_argument("--order", type=int, choices=[3, 4], default=3)
    args = parser.parse_args()
    download_lm(args.output_dir, args.order)
