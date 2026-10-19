import argparse
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from download_lm import download_lm

ROOT = Path(__file__).resolve().parent


def download_resources(output_dir: Path = ROOT, lm_order: int = 3) -> None:
    archive = hf_hub_download(
        repo_id="YaItco/DLA-1-Conformer-CTC",
        filename="final_ctc.zip",
        revision="c72fe53b22df3c0a2e589c2448d2656491076693",
    )
    with zipfile.ZipFile(archive) as source:
        source.extractall(output_dir)
    download_lm(output_dir / "data/lm/librispeech", order=lm_order)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--lm-order", type=int, choices=[3, 4], default=3)
    args = parser.parse_args()
    download_resources(args.output_dir, args.lm_order)
