"""Utility functions for the fine-tuning project."""

import torch
from transformers import Wav2Vec2Processor
from typing import List, Optional


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

