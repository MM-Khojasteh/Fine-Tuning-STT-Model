# Wav2Vec2 Persian STT — Fine-Tuning Pipeline

PyTorch pipeline for fine-tuning the Persian Wav2Vec2 checkpoint
(`jonatasgrosman/wav2vec2-large-xlsr-53-persian`) on custom audio-transcript
pairs, with training, evaluation, and streaming inference.

## Features

- **Training** — GPU + mixed precision, gradient accumulation & clipping, LR warmup + ReduceLROnPlateau, early stopping, best/periodic checkpointing
- **Evaluation** — WER, CER, BLEU, ROUGE-1/2/L, Word F1, exact match, plus error-type and length-bucketed breakdowns
- **Inference** — chunked, memory-efficient streaming for arbitrarily long audio
- **Configuration** — every operational value overridable via environment variables; no secrets or data in git

## Project Structure

```text
Fine-Tuning-STT-Model/
├── src/                     # config, data loading, dataset, trainer, evaluator, utils, logging
├── config/config.yaml       # configuration schema (safe defaults)
├── train.py                 # fine-tune
├── evaluate.py              # test-set evaluation
├── test_predictions.py      # per-sample prediction analysis & reporting
├── inference.py             # streaming transcription CLI
├── install_pytorch_cuda.ps1 # CUDA 12.1 PyTorch installer
├── requirements.txt
└── .env.example             # documented environment variable template
```

`dataset/` (audio + transcripts) and `outputs/` (checkpoints, logs, results)
are required at runtime but intentionally excluded from the repository.

## Setup

```bash
pip install -r requirements.txt

# GPU (CUDA 12.1) — Windows alternative: .\install_pytorch_cuda.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

cp .env.example .env   # add HUGGINGFACE_TOKEN (private/gated models) and optional overrides
```

## Data Layout

```text
dataset/
├── audio_files/     # .wav / .mp3 / .flac training audio
├── info/            # transcript JSONs (fields: recording_file | audio_file_path, transcript)
└── test_audio/      # held-out audio + optional test.json (audio_file_path, transcript)
```

Audio files are matched to transcripts by filename (case-insensitive).

## Configuration

Resolution order: **shell env → `.env` → `config/config.yaml` (safe defaults)**.
All overrides use `WAV2VEC2_*` variables — dataset paths, learning rate, batch
size, early stopping, chunk duration, device toggles, and more. The full
documented matrix is in [`.env.example`](.env.example). Invalid values fail
fast with a clear error naming the variable.

## Usage

```bash
python train.py                                                 # train → outputs/best_model.pt
python evaluate.py --checkpoint outputs/best_model.pt           # test-set metrics
python test_predictions.py --checkpoint outputs/best_model.pt   # per-sample analysis + confidence
python inference.py audio.wav --checkpoint outputs/best_model.pt  # timestamped transcription
```

Each script supports `--help`. Training artifacts land in `outputs/`:
`best_model.pt`, `checkpoints/`, `training_history.json`, `training.log`.

## Requirements

- Python 3.9+
- CUDA GPU recommended (≥ 8 GB VRAM); CPU fallback is automatic

## Security

Secrets live in `.env` (gitignored) and are never hardcoded. Datasets, audio,
weights, and logs are never tracked — `.gitignore` uses a strict allowlist so
nothing sensitive can be committed accidentally.

## License & Acknowledgments

Usage is governed by the base model's
[license](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-persian).
Built with [Hugging Face Transformers](https://huggingface.co/transformers/).
