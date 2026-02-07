# Architecture Review: Wav2Vec2 Persian STT Fine-Tuning Pipeline

## Project Summary

A PyTorch-based pipeline for fine-tuning `jonatasgrosman/wav2vec2-large-xlsr-53-persian` on custom Persian call-center audio-transcript pairs using CTC loss, with comprehensive evaluation and inference tooling.

---

## Architecture Overview

```
train.py ─────────┐
evaluate.py ──────┤
inference.py ─────┤──> src/
test_predictions.py┘     ├── data_loader.py   (JSON/audio matching)
                         ├── dataset.py       (PyTorch Dataset + collate)
                         ├── trainer.py       (Training loop)
                         ├── evaluator.py     (WER/CER/BLEU/ROUGE metrics)
                         └── utils.py         (CTC decode, GPU helpers)
```

**Data flow:** JSON files + audio directory --> `data_loader` matches pairs --> `AudioTranscriptDataset` preprocesses audio + tokenizes text --> `DataLoader` with custom `collate_fn` --> `Trainer` runs CTC training --> `evaluator` computes metrics.

---

## Strengths

### 1. Clean Separation of Concerns

Each module has a single responsibility. Data loading, dataset construction, training, evaluation, and utilities are cleanly separated into their own files. The entry-point scripts (`train.py`, `evaluate.py`, `inference.py`, `test_predictions.py`) are thin orchestrators that compose these modules. This makes the codebase easy to navigate and modify individually.

### 2. CTC Implementation Details Done Right

- **Sequence length calculation** (`trainer.py:202`): Uses `logits.size(1)` instead of raw audio length for CTC input lengths. This is a common mistake in CTC implementations, and it is correctly handled here.
- **Blank token resolution** (`trainer.py:117`): Properly reads `pad_token_id` from the processor rather than hardcoding.
- **`zero_infinity=True`** on CTCLoss prevents training from blowing up on pathological alignments.
- **Custom CTC decoding** (`utils.py:21-73`): Correctly implements blank removal + consecutive-duplicate collapse, which is the CTC decoding contract.

### 3. GPU Memory Management (Within Your Constraints)

Given limited GPU resources, the pipeline uses the right set of techniques:
- `max_audio_length=15s` truncation prevents OOM from long recordings.
- Gradient accumulation (`4 steps`) gives an effective batch size of 16 from physical batch size of 4.
- Mixed precision (FP16) with `GradScaler` reduces memory footprint.
- `empty_cache_after_val` reclaims fragmented VRAM between train/val phases.
- These are the correct levers to pull before resorting to model sharding or multi-GPU.

### 4. Robust Batch Handling

The trainer gracefully handles invalid CTC batches (empty labels, length mismatches) by skipping them and reporting statistics (`trainer.py:208-215`). The >50% skip-rate warning (`trainer.py:300-301`) is a useful canary for data quality issues. This is better than crashing mid-epoch.

### 5. Comprehensive Evaluation Suite

The evaluator (`evaluator.py`) computes 10+ metrics including WER, CER, BLEU, ROUGE-1/2/L, Word F1, exact match, error type analysis, and performance-by-length binning. The per-sample breakdown (`evaluate_samples`) and confidence analysis (`test_predictions.py`) give actionable insight into where the model struggles. This level of evaluation is more thorough than most fine-tuning projects.

### 6. Reproducibility

- Seeded random split (`torch.Generator().manual_seed(42)`) ensures consistent train/val partitions.
- Config-driven hyperparameters (YAML) separate concerns from code.
- Training history is saved as JSON after every epoch for crash recovery.

### 7. Persian-Specific Text Cleaning

The `_clean_text` method (`dataset.py:180-197`) correctly strips speaker labels (e.g., `اپراتور (رسمی):`), special markers (`[زنگ تلفن]`, `[پایان مکالمه]`), and normalizes whitespace. This is critical for call-center data where raw transcripts contain metadata that would confuse the model.

---

## Weaknesses

### 1. Audio Loading Happens at Training Time (No Caching)

**Impact: High (training speed)**

Every call to `__getitem__` (`dataset.py:97-178`) reads the audio file from disk, resamples it, normalizes it, and runs it through the processor. On repeated epochs, the same files are loaded and processed identically. With `num_workers=4`, this creates significant I/O pressure and redundant CPU work.

**What could help:** Pre-process all audio to 16kHz mono `.pt` tensors once and save them. The dataset `__getitem__` would then just load a tensor and tokenize the transcript. This would substantially reduce per-epoch overhead without touching GPU configuration.

### 2. No Data Augmentation

**Impact: High (model quality)**

The pipeline applies zero augmentation to the audio. For a fine-tuning task with presumably limited data (call-center recordings), the model is likely to overfit. Standard ASR augmentations include:
- **SpecAugment** (masking time/frequency bands) -- effectively free in compute cost
- **Speed perturbation** (0.9x-1.1x) -- cheap
- **Noise injection** from background audio
- **Volume perturbation**

SpecAugment in particular is nearly standard practice for Wav2Vec2 fine-tuning and can be enabled with just a few lines via the processor or a simple transform.

### 3. Text Cleaning Regex May Be Too Aggressive

**Impact: Medium (data quality)**

The regex `re.sub(r'[^:]+\([^)]+\):', '', text)` in `dataset.py:192` is meant to strip speaker labels like `اپراتور (رسمی):`, but it will also strip any text segment that happens to match the pattern `word(word):`. For Persian text that may contain parenthetical clarifications followed by colons, this could silently remove real transcript content. A more targeted regex anchored to known speaker labels would be safer.

### 4. `load_model` Function is Duplicated

**Impact: Medium (maintainability)**

The `load_model` function appears in three places with minor variations:
- `evaluate.py:29-70`
- `test_predictions.py:34-75`
- `inference.py:18-69`

Each has slightly different error handling and parameter handling. This creates a maintenance burden -- a bug fix or behavior change needs to happen in three places. This should live in a single shared module.

### 5. `collate_fn` Pads to Batch-Maximum Length

**Impact: Medium (GPU efficiency)**

The custom `collate_fn` (`dataset.py:12-68`) pads all sequences to the longest in the batch. If one sample in a batch of 4 is 15 seconds and the others are 3 seconds, all four get padded to 15 seconds worth of features. This wastes compute on padding tokens.

**What could help:** A `BucketingSampler` that groups audio of similar length into the same batch would reduce wasted padding. This is a purely CPU-side change that improves GPU utilization without requiring more VRAM.

### 6. No Checkpoint Resumption

**Impact: Medium (training resilience)**

The trainer saves checkpoints every epoch (`trainer.py:441-478`) including optimizer and scheduler states, but `train.py` has no logic to resume from a checkpoint. If training is interrupted at epoch 3 of 5, you must restart from epoch 1. The checkpoint already contains everything needed for resumption (model state, optimizer state, scheduler state, epoch number) -- it just needs a loading path in `train.py`.

### 7. Feature Extractor is Not Frozen

**Impact: Medium (training stability & GPU memory)**

The comment at `train.py:87` shows `model.freeze_feature_extractor()` is intentionally left commented out. For most Wav2Vec2 fine-tuning scenarios -- especially with limited data -- freezing the feature extractor (the CNN layers) and only training the transformer + CTC head is the recommended practice. Training the feature extractor:
- Increases memory usage (more parameters with gradients).
- Risks destabilizing pre-trained audio representations.
- Requires more data to justify.

This is a deliberate tradeoff you may have reasons for, but if you are working with a smaller dataset (hundreds to low thousands of samples), freezing the feature extractor would likely improve results and reduce memory usage.

### 8. Learning Rate 3e-4 is High for Fine-Tuning

**Impact: Medium (training stability)**

A learning rate of `3e-4` (`config.yaml:18`) is standard for training from scratch, but aggressive for fine-tuning a pre-trained model. The typical range for Wav2Vec2 fine-tuning is `1e-5` to `3e-5`. Combined with an unfrozen feature extractor, this risks catastrophic forgetting of the pre-trained representations. The `ReduceLROnPlateau` scheduler helps, but the initial rate is still high.

### 9. Validation Split is Random, Not Stratified

**Impact: Low-Medium (evaluation reliability)**

The train/val split (`train.py:103-107`) uses `random_split`, which does not account for the distribution of audio lengths, speakers, or transcript complexity. For call-center data with varying call types, this could produce a validation set that is not representative. A stratified split (e.g., by audio duration bins or speaker) would give more reliable validation metrics.

### 10. No Logging Framework

**Impact: Low (operational maturity)**

All logging is done via `print()` statements. This works but makes it harder to:
- Filter by severity (debug vs warning vs error).
- Redirect output to files while keeping console clean.
- Integrate with monitoring systems (TensorBoard, W&B).

Python's `logging` module or integration with TensorBoard/Weights & Biases would provide better observability, especially for longer training runs.

### 11. Inference Chunking Has No Overlap

**Impact: Low (inference quality)**

The inference script (`inference.py:121-133`) splits long audio into fixed 30-second chunks with no overlap. Words at chunk boundaries get cut, producing artifacts in the transcription. A sliding window with 2-5 seconds of overlap and deduplication of the overlapping region would produce cleaner results.

### 12. `save_every_n_epochs` Config is Unused

**Impact: Low (config drift)**

The config defines `save_every_n_epochs: 1` (`config.yaml:36`), but the trainer unconditionally saves a checkpoint every epoch (`trainer.py:517`). The config value is never read. This is dead configuration that could mislead someone adjusting settings.

---

## Summary Table

| Category | Strengths | Weaknesses |
|---|---|---|
| **Architecture** | Clean module separation, thin entry points | `load_model` duplicated 3x, dead config values |
| **CTC** | Correct sequence lengths, blank handling, zero_infinity | -- |
| **GPU Efficiency** | Grad accumulation, FP16, max_audio_length, cache clearing | No length-bucketing sampler, unfrozen feature extractor uses more VRAM |
| **Data Pipeline** | Multi-format JSON support, Persian text cleaning, fallback audio loading | No caching of preprocessed audio, no data augmentation, aggressive regex |
| **Training** | Early stopping, LR scheduling, batch skip handling | No resume from checkpoint, high initial LR for fine-tuning, random (not stratified) split |
| **Evaluation** | 10+ metrics, per-length analysis, confidence scoring | -- |
| **Inference** | Chunking support, timestamped output | No chunk overlap for boundary words |
| **Observability** | Training history JSON, GPU memory logging | No structured logging framework |

---

## Prioritized Recommendations (No GPU Changes Required)

If you were to iterate on this project, these are the changes ordered by impact-to-effort ratio:

1. **Add SpecAugment** -- A few lines of code for meaningful regularization.
2. **Lower learning rate to ~1e-5 to 3e-5** -- One config change.
3. **Freeze the feature extractor** -- Uncomment one line, saves memory and improves stability.
4. **Pre-process audio to cached tensors** -- Significant speedup across epochs, purely CPU-side.
5. **Add checkpoint resumption** -- The data is already in the checkpoint; just needs loading logic.
6. **Extract `load_model` to a shared module** -- Reduces maintenance burden.
7. **Add bucketing sampler** -- Reduces wasted padding, improves GPU utilization.
8. **Tighten the text-cleaning regex** -- Prevents accidental transcript corruption.

All of these are achievable without any changes to GPU allocation or hardware.
