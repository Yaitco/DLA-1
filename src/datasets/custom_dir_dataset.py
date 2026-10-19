from pathlib import Path

import soundfile as sf

from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    def __init__(self, data_dir: str | Path, **kwargs) -> None:
        self.data_dir = Path(data_dir)
        index = self._create_index(self.data_dir)
        super().__init__(index, **kwargs)

    @staticmethod
    def _create_index(data_dir: Path) -> list[dict]:
        audio_dir = data_dir / "audio"
        if not audio_dir.is_dir():
            raise FileNotFoundError(f"Missing audio directory: {audio_dir}")
        index, ids = [], set()

        allowed_suffix = [".wav", ".flac", ".mp3"]
        for path in sorted(audio_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in allowed_suffix:
                continue

            if path.stem in ids:
                raise ValueError(f"Duplicate audio ID: {path.stem}")
            ids.add(path.stem)

            text_path = data_dir / "transcriptions" / f"{path.stem}.txt"
            text = (
                text_path.read_text(encoding="utf-8").strip()
                if text_path.is_file()
                else ""
            )

            info = sf.info(path)
            index.append(
                {
                    "path": str(path.resolve()),
                    "text": text,
                    "audio_len": info.frames / info.samplerate,
                }
            )
        return index
