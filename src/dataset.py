"""PyTorch Dataset class for audio-transcript pairs."""

import torch
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor
from typing import List, Tuple, Optional, Dict
import re

from src.utils import load_audio_file


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function to pad sequences in a batch.
    
    Args:
        batch: List of dictionaries from dataset
        
    Returns:
        Dictionary with padded tensors
    """
    # Get max lengths
    max_input_length = max(item['input_values'].shape[0] for item in batch)
    max_label_length = max(item['labels'].shape[0] for item in batch)
    
    # Get processor pad token id
    pad_token_id = batch[0].get('pad_token_id', 0)
    
    # Initialize lists
    input_values_list = []
    attention_mask_list = []
    labels_list = []
    
    for item in batch:
        input_values = item['input_values']
        labels = item['labels']
        attention_mask = item.get('attention_mask')
        
        # Pad input_values
        input_padding = max_input_length - input_values.shape[0]
        if input_padding > 0:
            input_values = torch.nn.functional.pad(input_values, (0, input_padding), value=0.0)
        
        # Pad attention_mask
        if attention_mask is not None:
            attn_padding = max_input_length - attention_mask.shape[0]
            if attn_padding > 0:
                attention_mask = torch.nn.functional.pad(attention_mask, (0, attn_padding), value=0)
        else:
            # Create attention mask if not provided
            attention_mask = torch.ones(max_input_length, dtype=torch.long)
            if input_padding > 0:
                attention_mask[-input_padding:] = 0
        
        # Pad labels (use -100 for padding as it's ignored in loss calculation)
        label_padding = max_label_length - labels.shape[0]
        if label_padding > 0:
            labels = torch.nn.functional.pad(labels, (0, label_padding), value=-100)
        
        input_values_list.append(input_values)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)
    
    # Stack tensors
    return {
        'input_values': torch.stack(input_values_list),
        'attention_mask': torch.stack(attention_mask_list),
        'labels': torch.stack(labels_list)
    }


class AudioTranscriptDataset(Dataset):
    """Dataset class for audio-transcript pairs for Wav2Vec2 fine-tuning."""
    
    def __init__(
        self,
        audio_transcript_pairs: List[Tuple[str, str]],
        processor: Wav2Vec2Processor,
        target_sampling_rate: int = 16000,
        max_audio_length: Optional[float] = None
    ):
        """Initialize dataset.
        
        Args:
            audio_transcript_pairs: List of (audio_path, transcript) tuples
            processor: Wav2Vec2Processor for feature extraction and tokenization
            target_sampling_rate: Target sampling rate (default: 16000 for Wav2Vec2)
            max_audio_length: Maximum audio length in seconds (None for no limit)
        """
        self.audio_transcript_pairs = audio_transcript_pairs
        self.processor = processor
        self.target_sampling_rate = target_sampling_rate
        self.max_audio_length = max_audio_length
        
    def __len__(self) -> int:
        return len(self.audio_transcript_pairs)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single audio-transcript pair.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with 'input_values', 'attention_mask', and 'labels'
        """
        audio_path, transcript = self.audio_transcript_pairs[idx]

        # Load and preprocess audio using shared utility
        waveform, original_sr = load_audio_file(
            audio_path,
            target_sampling_rate=self.target_sampling_rate,
            mono=True,
            normalize=True
        )

        # Calculate durations for logging
        original_duration = waveform.shape[1] / self.target_sampling_rate

        # Truncate if max_audio_length is specified
        was_truncated = False
        if self.max_audio_length is not None:
            max_samples = int(self.max_audio_length * self.target_sampling_rate)
            if waveform.shape[1] > max_samples:
                was_truncated = True
                waveform = waveform[:, :max_samples]

        final_duration = waveform.shape[1] / self.target_sampling_rate

        # Squeeze channel dimension for processor (mono audio -> 1D tensor)
        waveform = waveform.squeeze(0)
        
        # Log audio processing info (only for first few samples or when truncation occurs)
        # This helps verify audio processing is working correctly
        if idx < 3 or was_truncated:
            truncation_info = f" (truncated from {original_duration:.2f}s)" if was_truncated else ""
            print(f"Audio [{idx}]: {final_duration:.2f}s{truncation_info} - {audio_path}")
        
        # Process audio with processor
        processed = self.processor(
            waveform.numpy(),
            sampling_rate=self.target_sampling_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Clean and normalize Persian text
        cleaned_transcript = self._clean_text(transcript)
        
        # Tokenize transcript using the new API
        # Use text parameter directly instead of deprecated as_target_processor
        labels = self.processor(text=cleaned_transcript, return_tensors="pt").input_ids
        # Replace padding token id with -100 (ignored in loss calculation)
        labels = labels.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "input_values": processed.input_values.squeeze(0),
            "attention_mask": processed.attention_mask.squeeze(0) if processed.attention_mask is not None else None,
            "labels": labels
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize Persian text.
        
        Args:
            text: Raw transcript text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove speaker labels and timestamps if present (e.g., "اپراتور (رسمی):")
        text = re.sub(r'[^:]+\([^)]+\):', '', text)
        # Remove special markers like [زنگ تلفن], [پایان مکالمه]
        text = re.sub(r'\[.*?\]', '', text)
        # Strip whitespace
        text = text.strip()
        return text

