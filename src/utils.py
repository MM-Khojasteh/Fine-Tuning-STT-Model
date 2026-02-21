"""Utility functions for the fine-tuning project."""

import torch
import torchaudio
import soundfile as sf
from transformers import Wav2Vec2Processor
from typing import List, Optional, Tuple, Generator
import os


def format_timestamp(seconds):
    """Format seconds into MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string in MM:SS format
    """
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02}:{secs:02}"


def decode_ctc_predictions(
    processor: Wav2Vec2Processor,
    predicted_ids: torch.Tensor,
    blank_token_id: int = None
) -> List[str]:
    """Decode CTC predictions with blank token filtering.
    
    CTC models can predict blank tokens (typically 0 or pad_token_id) which
    should be filtered out before decoding. This function also collapses
    consecutive duplicate tokens (CTC collapse rule).
    
    Args:
        processor: Wav2Vec2Processor for decoding
        predicted_ids: Tensor of shape (batch_size, sequence_length) with predicted token IDs
        blank_token_id: Token ID for blank/padding (defaults to processor's pad_token_id or 0)
        
    Returns:
        List of decoded text strings
    """
    if blank_token_id is None:
        # Use pad_token_id if available, otherwise default to 0
        blank_token_id = processor.tokenizer.pad_token_id
        if blank_token_id is None:
            blank_token_id = 0
    
    decoded_texts = []
    
    for pred_seq in predicted_ids:
        # Convert to list if tensor
        if isinstance(pred_seq, torch.Tensor):
            pred_seq = pred_seq.cpu().tolist()
        
        # Remove blank tokens and collapse consecutive duplicates (CTC collapse)
        filtered = []
        prev_token = None
        for token in pred_seq:
            # Skip blank tokens and consecutive duplicates
            if token != blank_token_id and token != prev_token:
                filtered.append(token)
            prev_token = token
        
        # Decode filtered sequence
        if filtered:
            try:
                text = processor.decode(filtered)
                decoded_texts.append(text.strip())
            except Exception as e:
                # If decoding fails, return empty string
                decoded_texts.append("")
        else:
            decoded_texts.append("")
    
    return decoded_texts


def get_gpu_memory_info(device: Optional[torch.device] = None) -> dict:
    """Get GPU memory usage information.
    
    Args:
        device: PyTorch device (defaults to current CUDA device)
        
    Returns:
        Dictionary with memory information in MB
    """
    if not torch.cuda.is_available():
        return {
            'allocated_mb': 0,
            'reserved_mb': 0,
            'free_mb': 0,
            'total_mb': 0
        }
    
    if device is None:
        device = torch.cuda.current_device()
    
    allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
    reserved = torch.cuda.memory_reserved(device) / 1024**2  # MB
    total = torch.cuda.get_device_properties(device).total_memory / 1024**2  # MB
    free = total - reserved
    
    return {
        'allocated_mb': allocated,
        'reserved_mb': reserved,
        'free_mb': free,
        'total_mb': total
    }


def log_gpu_memory(stage: str = "", device: Optional[torch.device] = None):
    """Log GPU memory usage.
    
    Args:
        stage: Description of current stage (e.g., "After validation")
        device: PyTorch device
    """
    if not torch.cuda.is_available():
        return
    
    mem_info = get_gpu_memory_info(device)
    stage_str = f" [{stage}]" if stage else ""
    print(f"GPU Memory{stage_str}: "
          f"Allocated: {mem_info['allocated_mb']:.1f} MB, "
          f"Reserved: {mem_info['reserved_mb']:.1f} MB, "
          f"Free: {mem_info['free_mb']:.1f} MB / {mem_info['total_mb']:.1f} MB")


def clear_gpu_cache(device: Optional[torch.device] = None):
    """Clear GPU cache to free up memory.
    
    Args:
        device: PyTorch device
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if device is not None:
            torch.cuda.synchronize(device)


def inspect_predicted_tokens(
    predicted_ids: torch.Tensor,
    processor: Wav2Vec2Processor,
    blank_token_id: int = None,
    max_samples: int = 3
) -> dict:
    """Inspect predicted token IDs to diagnose decoding issues.
    
    Args:
        predicted_ids: Tensor of predicted token IDs
        processor: Wav2Vec2Processor
        blank_token_id: Blank token ID (defaults to processor's pad_token_id or 0)
        max_samples: Maximum number of samples to inspect
        
    Returns:
        Dictionary with diagnostic information
    """
    if blank_token_id is None:
        blank_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
    
    if isinstance(predicted_ids, torch.Tensor):
        predicted_ids = predicted_ids.cpu()
    
    batch_size = predicted_ids.shape[0] if len(predicted_ids.shape) > 1 else 1
    if len(predicted_ids.shape) == 1:
        predicted_ids = predicted_ids.unsqueeze(0)
    
    diagnostics = {
        'total_samples': batch_size,
        'samples_inspected': min(max_samples, batch_size),
        'blank_token_id': blank_token_id,
        'sample_details': []
    }
    
    for i in range(min(max_samples, batch_size)):
        seq = predicted_ids[i].tolist()
        unique_tokens = list(set(seq))
        blank_count = sum(1 for t in seq if t == blank_token_id)
        non_blank_tokens = [t for t in seq if t != blank_token_id]
        unique_non_blank = list(set(non_blank_tokens))
        
        # Try to decode non-blank tokens
        decoded_text = ""
        if unique_non_blank:
            try:
                decoded_text = processor.decode(unique_non_blank[:10])  # Decode first 10 unique tokens
            except:
                decoded_text = "<decode_error>"
        
        diagnostics['sample_details'].append({
            'sample_idx': i,
            'sequence_length': len(seq),
            'blank_count': blank_count,
            'blank_ratio': blank_count / len(seq) if seq else 0.0,
            'unique_tokens': len(unique_tokens),
            'unique_non_blank_tokens': len(unique_non_blank),
            'non_blank_token_ids': unique_non_blank[:20],  # First 20 unique non-blank tokens
            'decoded_preview': decoded_text[:50]  # First 50 chars of decoded text
        })
    
    # Overall statistics
    all_tokens = predicted_ids.flatten().tolist()
    total_tokens = len(all_tokens)
    total_blanks = sum(1 for t in all_tokens if t == blank_token_id)
    
    diagnostics['overall'] = {
        'total_tokens': total_tokens,
        'total_blanks': total_blanks,
        'blank_ratio': total_blanks / total_tokens if total_tokens > 0 else 0.0,
        'unique_token_count': len(set(all_tokens))
    }
    
    return diagnostics


def check_blank_token_prediction(
    predicted_ids: torch.Tensor,
    blank_token_id: int = 0,
    threshold: float = 0.95
) -> dict:
    """Check if model is predicting mostly blank tokens (diagnostic for empty predictions).
    
    Args:
        predicted_ids: Tensor of predicted token IDs
        blank_token_id: Blank token ID to check for
        threshold: Threshold ratio above which to consider "mostly blanks"
        
    Returns:
        Dictionary with check results
    """
    if isinstance(predicted_ids, torch.Tensor):
        predicted_ids = predicted_ids.cpu()
    
    if len(predicted_ids.shape) == 1:
        predicted_ids = predicted_ids.unsqueeze(0)
    
    all_tokens = predicted_ids.flatten().tolist()
    total_tokens = len(all_tokens)
    blank_count = sum(1 for t in all_tokens if t == blank_token_id)
    blank_ratio = blank_count / total_tokens if total_tokens > 0 else 0.0
    
    is_mostly_blanks = blank_ratio >= threshold
    
    return {
        'is_mostly_blanks': is_mostly_blanks,
        'blank_ratio': blank_ratio,
        'blank_count': blank_count,
        'total_tokens': total_tokens,
        'threshold': threshold,
        'warning': f"Model is predicting {blank_ratio*100:.1f}% blank tokens!" if is_mostly_blanks else None
    }


def load_audio_file_streaming(
    audio_path: str,
    target_sampling_rate: int = 16000,
    chunk_duration: float = 30.0,
    mono: bool = True,
    normalize: bool = True
):
    """Stream audio file in chunks to minimize memory usage for long files.

    Uses soundfile.blocks() for memory-efficient loading of large audio files.
    Yields one chunk at a time, keeping memory usage constant regardless of file size.

    Args:
        audio_path: Path to the audio file
        target_sampling_rate: Target sampling rate for resampling
        chunk_duration: Duration of each chunk in seconds
        mono: Whether to convert to mono
        normalize: Whether to normalize audio

    Yields:
        Tuple of (waveform_chunk: torch.Tensor, start_time: float, end_time: float)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Open file to get sample rate and info
    with sf.SoundFile(audio_path) as f:
        original_sr = f.samplerate
        total_samples = len(f)
        duration = total_samples / original_sr

        # Calculate chunk size in samples (at original sample rate)
        chunk_samples = int(chunk_duration * original_sr)

        # Calculate resampling ratio
        resample_ratio = target_sampling_rate / original_sr

        block_index = 0
        for block in sf.blocks(
            audio_path,
            blocksize=chunk_samples,
            dtype='float32',
            always_2d=True
        ):
            # Convert to torch tensor: (samples, channels) -> (channels, samples)
            waveform = torch.from_numpy(block.T.copy())

            # Convert to mono if needed
            if mono and waveform.shape[0] > 1:
                waveform = waveform.mean(0, keepdim=True)

            # Resample if needed
            if original_sr != target_sampling_rate:
                waveform = torchaudio.functional.resample(
                    waveform, original_sr, target_sampling_rate
                )

            # Normalize
            if normalize and waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()

            # Calculate timestamps
            start_sample = block_index * chunk_samples
            end_sample = min(start_sample + chunk_samples, total_samples)
            start_time = start_sample / original_sr
            end_time = min(end_sample / original_sr, duration)

            block_index += 1

            # Skip very short chunks (< 0.1 seconds)
            if waveform.shape[1] < target_sampling_rate // 10:
                continue

            yield waveform, start_time, end_time


def load_audio_file(
    audio_path: str,
    target_sampling_rate: int = 16000,
    mono: bool = True,
    normalize: bool = True
) -> Tuple[torch.Tensor, int]:
    """Load and preprocess audio file with consistent processing.

    This is a shared utility function used by both the dataset and inference
    to ensure consistent audio loading behavior across the codebase.

    Args:
        audio_path: Path to the audio file
        target_sampling_rate: Target sampling rate for resampling (default: 16000)
        mono: Whether to convert to mono (default: True)
        normalize: Whether to normalize audio to [-1, 1] range (default: True)

    Returns:
        Tuple of (waveform_tensor, sample_rate)
        - waveform_tensor: Shape (channels, samples) or (1, samples) if mono=True
        - sample_rate: Original sample rate before resampling

    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If audio loading fails
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Use soundfile directly to avoid torchcodec dependency in newer torchaudio
    try:
        data, sr = sf.read(audio_path, dtype='float32')
        # Convert to torch tensor and ensure shape matches torchaudio format: (channels, samples)
        if data.ndim == 1:
            # Mono audio: (samples,) -> (1, samples)
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            # Multi-channel audio: (samples, channels) -> (channels, samples)
            waveform = torch.from_numpy(data).transpose(0, 1)
    except Exception as e:
        # Fallback to torchaudio.load if soundfile fails
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e2:
            raise RuntimeError(f"Failed to load audio file {audio_path}: "
                               f"soundfile error: {e}, torchaudio error: {e2}") from e

    # Convert to mono if stereo and mono=True
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)

    # Resample to target sampling rate if needed
    if sr != target_sampling_rate:
        waveform = torchaudio.functional.resample(waveform, sr, target_sampling_rate)

    # Normalize audio to [-1, 1] range
    if normalize and waveform.abs().max() > 0:
        waveform = waveform / waveform.abs().max()

    return waveform, sr


def preprocess_audio_for_inference(
    audio_path: str,
    processor: Wav2Vec2Processor,
    target_sampling_rate: int = 16000,
    chunk_duration: Optional[float] = None,
    device: Optional[torch.device] = None
) -> List[dict]:
    """Load and preprocess audio file for inference with optional chunking.

    Args:
        audio_path: Path to the audio file
        processor: Wav2Vec2Processor for feature extraction
        target_sampling_rate: Target sampling rate (default: 16000)
        chunk_duration: Duration of each chunk in seconds (None for no chunking)
        device: PyTorch device (for moving tensors to GPU)

    Returns:
        List of dictionaries containing:
        - 'input_values': Preprocessed input tensor
        - 'start_sample': Start sample index
        - 'end_sample': End sample index
        - 'start_time': Start time in seconds
        - 'end_time': End time in seconds
    """
    waveform, original_sr = load_audio_file(
        audio_path,
        target_sampling_rate=target_sampling_rate,
        mono=True,
        normalize=True
    )

    # Handle device
    if device is not None:
        waveform = waveform.to(device)

    # If no chunking requested, process entire file
    if chunk_duration is None:
        input_values = processor(
            waveform.squeeze().cpu().numpy(),
            sampling_rate=target_sampling_rate,
            return_tensors="pt",
            padding=True
        ).input_values

        if device is not None:
            input_values = input_values.to(device)

        return [{
            'input_values': input_values,
            'start_sample': 0,
            'end_sample': waveform.shape[1],
            'start_time': 0.0,
            'end_time': waveform.shape[1] / target_sampling_rate
        }]

    # Chunk the audio
    chunk_samples = int(chunk_duration * target_sampling_rate)
    total_samples = waveform.shape[1]
    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples

    chunks = []
    for i in range(num_chunks):
        start_sample = i * chunk_samples
        end_sample = min((i + 1) * chunk_samples, total_samples)

        # Skip very short chunks (less than 0.1 seconds)
        if end_sample - start_sample < target_sampling_rate // 10:
            continue

        chunk_waveform = waveform[:, start_sample:end_sample]

        # Process chunk
        input_values = processor(
            chunk_waveform.squeeze().cpu().numpy(),
            sampling_rate=target_sampling_rate,
            return_tensors="pt",
            padding=True
        ).input_values

        if device is not None:
            input_values = input_values.to(device)

        chunks.append({
            'input_values': input_values,
            'start_sample': start_sample,
            'end_sample': end_sample,
            'start_time': start_sample / target_sampling_rate,
            'end_time': end_sample / target_sampling_rate
        })

    return chunks
