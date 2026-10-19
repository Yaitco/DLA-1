# Conformer ASR

Conformer-CTC (16 × 144), BPE-256 and KenLM for English speech recognition.
[Weights, tokenizer and training logs](https://huggingface.co/YaItco/DLA-1-Conformer-CTC).

## Demo

Open [demo.ipynb in Colab](https://colab.research.google.com/github/Yaitco/DLA-1/blob/main/demo.ipynb), enter a public dataset URL and run all cells. A GPU is recommended.
For the private repository, add `GITGUB_TOKEN` to Colab Secrets and allow notebook access. For a public repository, set `PRIVATE = False`.

## Installation

Python 3.11.

```bash
git clone https://github.com/Yaitco/DLA-1.git
cd DLA-1
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Inference

The input directory must contain `audio/{id}.wav|flac|mp3` and optional UTF-8 `transcriptions/{id}.txt`. Each text file contains only the transcription.
The demo accepts ZIP/TAR URLs and Google Drive files or folders in this format.

```bash
python download_resources.py --lm-order 4
python inference.py -cn inference_final_4gram \
  datasets.test.data_dir=/path/to/dataset \
  inferencer.save_path=demo/custom
python calc_metrics.py \
  --ground-truth /path/to/dataset/transcriptions \
  --predictions data/saved/demo/custom/test
```

Predictions keep the original IDs. Without references, skip `calc_metrics.py`.
To run the bundled example, use `datasets.test.data_dir=resources/final/sample_dataset`.

## Training

Place LibriSpeech splits in `data/datasets/librispeech/`; missing splits are downloaded automatically. The first dataset load creates transcript indexes for `train_bpe.py`.

```bash
python train_bpe.py
python train.py -cn CONFIG_NAME
```



Based on the [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template). Code license: [MIT](LICENSE).
