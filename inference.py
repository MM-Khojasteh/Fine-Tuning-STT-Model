"""Inference script for fine-tuned Wav2Vec2 model with chunking support."""

import os
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

from src.utils import format_timestamp

# Load environment variables from .env file
load_dotenv()


def load_model(checkpoint_path: str = None, model_name: str = None, device: torch.device = None):
    """Load model from checkpoint or pretrained.
    
    Args:
        checkpoint_path: Path to fine-tuned checkpoint (optional)
        model_name: Hugging Face model name (default: Persian Wav2Vec2)
        device: PyTorch device (auto-detect if None)
        
    Returns:
        model, processor, device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name is None:
        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-persian"
    
    print(f"Loading model: {model_name}")
    # Get Hugging Face token from environment variable (optional)
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    processor = Wav2Vec2Processor.from_pretrained(
        model_name,
        token=hf_token if hf_token else None
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        token=hf_token if hf_token else None
    )
    
    # Load fine-tuned checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading fine-tuned checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Fine-tuned model loaded successfully!")
    else:
        print("Using pretrained model (no fine-tuning)")
    
    model.to(device)
    model.eval()
    
    return model, processor, device


def transcribe_audio_file(
    file_path: str,
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    device: torch.device,
    chunk_duration: int = 30,
    target_sr: int = 16000
):
    """Transcribe audio file with chunking support for long files.
    
    Args:
        file_path: Path to audio file
        model: Wav2Vec2ForCTC model
        processor: Wav2Vec2Processor
        device: PyTorch device
        chunk_duration: Duration of each chunk in seconds
        target_sr: Target sampling rate
        
    Returns:
        List of dictionaries with timestamped transcriptions
    """
    # Load audio
    waveform, sr = torchaudio.load(file_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    
    # Resample to target sampling rate
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    
    # Normalize
    if waveform.abs().max() > 0:
        waveform = waveform / waveform.abs().max()
    
    # Chunk audio
    chunk_samples = chunk_duration * target_sr
    total_samples = waveform.shape[1]
    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
    
    print(f"Splitting into {num_chunks} chunks...")
    
    transcript_data = []
    
    for i in tqdm(range(num_chunks), desc="Transcribing"):
        start_sample = i * chunk_samples
        end_sample = min((i + 1) * chunk_samples, total_samples)
        chunk = waveform[:, start_sample:end_sample]
        
        # Skip very short chunks
        if chunk.shape[1] < 1000:
            continue
        
        # Process chunk
        input_values = processor(
            chunk.squeeze(),
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True
        ).input_values.to(device)
        
        # Transcribe
        with torch.no_grad():
            logits = model(input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0].strip()
        
        # Filter out empty or non-speech transcriptions
        if transcription and transcription.lower() not in ["music", "no speech", ""]:
            start_time = format_timestamp(i * chunk_duration)
            end_time = format_timestamp(min((i + 1) * chunk_duration, total_samples // target_sr))
            transcript_data.append({
                "chunk": i + 1,
                "start": start_time,
                "end": end_time,
                "text": transcription
            })
    
    return transcript_data


def main():
    """Main inference function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Transcribe audio file using fine-tuned Wav2Vec2 model')
    parser.add_argument(
        'audio_file',
        type=str,
        help='Path to audio file to transcribe'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to fine-tuned model checkpoint (optional)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Hugging Face model name (default: Persian Wav2Vec2)'
    )
    parser.add_argument(
        '--chunk-duration',
        type=int,
        default=30,
        help='Chunk duration in seconds (default: 30)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save transcription (optional, prints to stdout if not specified)'
    )
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        raise FileNotFoundError(f"Audio file not found: {args.audio_file}")
    
    # Load model
    model, processor, device = load_model(
        checkpoint_path=args.checkpoint,
        model_name=args.model
    )
    
    print(f"Using device: {device}")
    print(f"Transcribing: {args.audio_file}")
    
    # Transcribe
    transcript_data = transcribe_audio_file(
        file_path=args.audio_file,
        model=model,
        processor=processor,
        device=device,
        chunk_duration=args.chunk_duration
    )
    
    # Output results
    output_text = "\nTimestamped Transcription:\n\n"
    for entry in transcript_data:
        output_text += f"[{entry['start']} - {entry['end']}] {entry['text']}\n"
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"\nTranscription saved to: {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()

