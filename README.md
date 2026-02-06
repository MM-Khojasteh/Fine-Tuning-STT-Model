# Wav2Vec2 Fine-tuning Project

A complete PyTorch GPU-based fine-tuning pipeline for the Persian Wav2Vec2 model (`jonatasgrosman/wav2vec2-large-xlsr-53-persian`) using custom audio-transcript pairs.

## Features

- **GPU-accelerated training** with automatic device detection
- **Comprehensive evaluation metrics**: WER (Word Error Rate), CER (Character Error Rate), and BLEU score
- **Robust data handling**: Automatically matches audio files with transcripts from JSON files
- **Train/Validation/Test splits**: 80/20 split for train/validation, separate test set
- **Model checkpointing**: Saves best model and periodic checkpoints
- **Chunked inference**: Handles long audio files by splitting into chunks

## Project Structure

```
Fine-Tuning-STT-Model/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Load and match audio files with transcripts
│   ├── dataset.py               # PyTorch Dataset class
│   ├── trainer.py               # Training loop with GPU support
│   ├── evaluator.py             # Evaluation metrics (WER, CER, BLEU)
│   └── utils.py                 # Utility functions
├── config/
│   └── config.yaml              # Training hyperparameters
├── dataset/
│   ├── audio_files/             # Training audio files
│   ├── info/                    # JSON files with transcripts
│   └── test_audio/              # Test audio files
├── train.py                     # Main training script
├── evaluate.py                  # Evaluation script
├── inference.py                 # Inference script
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Installation

1. Clone or navigate to the project directory:
```bash
cd Fine-Tuning-STT-Model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional, only needed for private models or gated repositories):
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Hugging Face token
# Get your token from: https://huggingface.co/settings/tokens
# HUGGINGFACE_TOKEN=your_huggingface_token_here
```

4. **Important**: This project requires PyTorch 2.6.0 or higher due to security requirements. If you have an older version, upgrade with:
```bash
pip install --upgrade torch>=2.6.0 torchaudio>=2.6.0
```

5. For GPU support, ensure you have CUDA-enabled PyTorch installed. Install from [PyTorch website](https://pytorch.org/) if needed.

## Dataset Format

The project expects the following structure:

- **Audio files**: Located in `dataset/audio_files/` (supports .wav, .mp3, .flac)
- **Transcripts**: JSON files in `dataset/info/`:
  - `postgres_laravel_calls.json`: Contains `recording_file` and `transcript` fields
  - `postgres_public_call_transcripts.json`: Contains `audio_file_path` and `transcript` fields
- **Test audio**: Located in `dataset/test_audio/`

The data loader automatically matches audio files with their corresponding transcripts based on filename.

## Configuration

Edit `config/config.yaml` to customize training parameters:

```yaml
model:
  name: "jonatasgrosman/wav2vec2-large-xlsr-53-persian"

training:
  batch_size: 4
  num_epochs: 5
  learning_rate: 3e-4
  # ... more parameters
```

## Usage

### Training

Train the model on your dataset:

```bash
python train.py
```

The training script will:
1. Load audio-transcript pairs from JSON files
2. Split into 80% train / 20% validation
3. Fine-tune the Wav2Vec2 model
4. Save checkpoints and best model to `outputs/`

### Evaluation

Evaluate the fine-tuned model on the test set:

```bash
python evaluate.py --checkpoint outputs/best_model.pt
```

Options:
- `--checkpoint`: Path to model checkpoint (default: `outputs/best_model.pt`)
- `--test_dir`: Path to test audio directory (overrides config)
- `--output`: Path to save evaluation results (default: `outputs/evaluation_results.json`)

### Inference

Transcribe an audio file using the fine-tuned model:

```bash
python inference.py path/to/audio.wav --checkpoint outputs/best_model.pt
```

Options:
- `audio_file`: Path to audio file to transcribe (required)
- `--checkpoint`: Path to fine-tuned model checkpoint (optional, uses pretrained if not specified)
- `--model`: Hugging Face model name (default: Persian Wav2Vec2)
- `--chunk-duration`: Chunk duration in seconds for long files (default: 30)
- `--output`: Path to save transcription (optional, prints to stdout if not specified)

Example:
```bash
python inference.py dataset/test_audio/test_file_1.wav --checkpoint outputs/best_model.pt --output transcription.txt
```

## Training Output

Training produces:
- **Checkpoints**: Saved in `outputs/checkpoints/` (one per epoch)
- **Best model**: `outputs/best_model.pt` (model with lowest validation WER)
- **Training history**: `outputs/training_history.json` (metrics for each epoch)

## Evaluation Metrics

The evaluation script calculates:
- **WER (Word Error Rate)**: Percentage of word-level errors (lower is better)
- **CER (Character Error Rate)**: Percentage of character-level errors (lower is better)
- **BLEU Score**: N-gram overlap metric (higher is better, 0-100 scale)

## GPU Requirements

- CUDA-compatible GPU recommended for training
- Minimum 8GB VRAM for batch size 4
- Adjust `batch_size` in config if you encounter out-of-memory errors

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `batch_size` in `config/config.yaml`
- Reduce `max_audio_length` to process shorter audio segments
- Enable gradient accumulation: set `gradient_accumulation_steps > 1`

### No Transcripts Found
- Verify JSON files are in `dataset/info/`
- Check that audio filenames match the `recording_file` or `audio_file_path` fields in JSON
- Ensure JSON files are valid and contain `transcript` fields

### Slow Training
- Ensure GPU is being used (check device output in training logs)
- Reduce `num_workers` if data loading is bottleneck
- Consider freezing feature extractor for faster training (uncomment in `train.py`)

### PyTorch Version Error
If you encounter an error about `torch.load` requiring PyTorch 2.6.0+, upgrade PyTorch:
```bash
pip install --upgrade torch>=2.6.0 torchaudio>=2.6.0
```
This is required due to security vulnerabilities in older versions of PyTorch.

## License

This project uses the Wav2Vec2 model from Hugging Face. Please refer to the model's license for usage terms.

## Acknowledgments

- Model: [jonatasgrosman/wav2vec2-large-xlsr-53-persian](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-persian)
- Framework: [Hugging Face Transformers](https://huggingface.co/transformers/)

