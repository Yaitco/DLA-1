import argparse
import shutil
import tarfile
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import gdown
from gdown.parse_url import parse_url


def download_file(url: str, path: Path) -> None:
    with urlopen(url, timeout=60) as response:
        if response.headers.get_content_type() == "text/html":
            raise RuntimeError(f"Expected a file, received HTML: {url}")
        with path.open("wb") as target:
            shutil.copyfileobj(response, target)


def drive_url(file_id: str) -> str:
    return (
        "https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&confirm=t"
    )


def download_dataset(
    url: str, output_dir: Path = Path("data/demo/datasets"), workers: int = 4
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(dir=output_dir))
    content = folder / "content"
    is_drive = urlparse(url).hostname == "drive.google.com"

    if is_drive and "/folders/" in url:
        files = gdown.download_folder(
            url=url, output=str(content), quiet=True, skip_download=True
        )

        def download_recording(file):
            path = Path(file.local_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            download_file(drive_url(file.id), path)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(download_recording, files))
    else:
        archive = folder / "dataset.archive"
        if is_drive:
            file_id, _ = parse_url(url)
            url = drive_url(file_id)
        download_file(url, archive)
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as source:
                source.extractall(content)
        else:
            with tarfile.open(archive) as source:
                source.extractall(content, filter="data")

    (audio_dir,) = [
        path
        for path in content.rglob("audio")
        if path.is_dir() and "__MACOSX" not in path.parts
    ]
    return audio_dir.parent.resolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/demo/datasets"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(download_dataset(args.url, args.output_dir, args.workers))
