"""PyTorch Dataset class for audio-transcript pairs."""

import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor
from typing import List, Tuple, Optional
import re


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
        
        # Load and preprocess audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)
        
        # Resample to target sampling rate
        if sr != self.target_sampling_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.target_sampling_rate
            )
        
        # Normalize audio
        waveform = waveform.squeeze(0)
        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()
        
        # Truncate if max_audio_length is specified
        if self.max_audio_length is not None:
            max_samples = int(self.max_audio_length * self.target_sampling_rate)
            if waveform.shape[0] > max_samples:
                waveform = waveform[:max_samples]
        
        # Process audio with processor
        processed = self.processor(
            waveform.numpy(),
            sampling_rate=self.target_sampling_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Clean and normalize Persian text
        cleaned_transcript = self._clean_text(transcript)
        
        # Tokenize transcript
        with self.processor.as_target_processor():
            labels = self.processor(cleaned_transcript, return_tensors="pt").input_ids
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

