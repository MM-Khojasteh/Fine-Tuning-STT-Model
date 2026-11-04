"""Inference script for fine-tuned Wav2Vec2 model with chunking support."""

import os
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from tqdm import tqdm
from pathlib import Path

from src.config import get_hf_token, env_str, env_int, DEFAULT_MODEL_NAME
from src.utils import format_timestamp, decode_ctc_predictions, load_audio_file, load_audio_file_streaming
from src.logging_config import get_logger

logger = get_logger(__name__)


def load_model(checkpoint_path: str = None, model_name: str = None, device: torch.device = None):
    """Load model from checkpoint or pretrained.

    Args:
        checkpoint_path: Path to fine-tuned checkpoint (optional)
        model_name: Hugging Face model name (default: WAV2VEC2_MODEL_NAME env
            variable, falling back to the public Persian Wav2Vec2 checkpoint)
        device: PyTorch device (auto-detect if None)

    Returns:
        model, processor, device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name is None:
        model_name = env_str("WAV2VEC2_MODEL_NAME", DEFAULT_MODEL_NAME)

    logger.info(f"Loading model: {model_name}")
    # Get Hugging Face token from environment variable (optional)
    hf_token = get_hf_token()
    processor = Wav2Vec2Processor.from_pretrained(
        model_name,
        token=hf_token
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        token=hf_token,
        use_safetensors=True
    )
    
    # Load fine-tuned checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading fine-tuned checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Fine-tuned model loaded successfully!")
        except (ValueError, RuntimeError) as e:
            if "torch.load" in str(e) or "CVE-2025-32434" in str(e):
                raise RuntimeError(
                    f"Checkpoint loading failed due to PyTorch security restrictions. "
                    f"Please upgrade PyTorch to 2.6.0+ or use safetensors format for checkpoints. "
                    f"Error: {e}"
                ) from e
            raise
    else:
        logger.info("Using pretrained model (no fine-tuning)")
    
    model.to(device)
    model.eval()
    
    return model, processor, device


def transcribe_audio_file(
    file_path: str,
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    device: torch.device,
    chunk_duration: int = 30,
    target_sr: int = 16000,
    streaming: bool = True,
    max_memory_mb: float = 500
):
    """Transcribe audio file with chunking support for long files.

    Uses memory-efficient streaming for long audio files to avoid loading
    entire files into memory. Falls back to full loading for small files.

    Args:
        file_path: Path to audio file
        model: Wav2Vec2ForCTC model
        processor: Wav2Vec2Processor
        device: PyTorch device
        chunk_duration: Duration of each chunk in seconds
        target_sr: Target sampling rate
        streaming: If True, use memory-efficient streaming (recommended for files > 5 min)
        max_memory_mb: Threshold for automatic streaming (files larger than this use streaming)

    Returns:
        List of dictionaries with timestamped transcriptions
    """
    # Check file size to decide between streaming and full loading
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    use_streaming = streaming or file_size_mb > max_memory_mb

    if use_streaming:
        logger.info(f"Using memory-efficient streaming for {file_path} ({file_size_mb:.1f} MB)")
        return _transcribe_audio_streaming(
            file_path, model, processor, device, chunk_duration, target_sr
        )
    else:
        logger.info(f"Loading full audio file {file_path} ({file_size_mb:.1f} MB)")
        return _transcribe_audio_full(
            file_path, model, processor, device, chunk_duration, target_sr
        )


def _transcribe_audio_streaming(
    file_path: str,
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    device: torch.device,
    chunk_duration: int = 30,
    target_sr: int = 16000
):
    """Transcribe using memory-efficient streaming.

    Processes audio in streaming fashion, keeping memory usage constant
    regardless of file length.
    """
    transcript_data = []
    chunk_index = 0

    for waveform, start_time, end_time in load_audio_file_streaming(
        file_path,
        target_sampling_rate=target_sr,
        chunk_duration=chunk_duration,
        mono=True,
        normalize=True
    ):
        chunk_index += 1

        # Process chunk
        input_values = processor(
            waveform.squeeze().numpy(),
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True
        ).input_values.to(device)

        # Transcribe
        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcriptions = decode_ctc_predictions(processor, predicted_ids)
        transcription = transcriptions[0] if transcriptions else ""

        # Filter out empty or non-speech transcriptions
        if transcription and transcription.lower() not in ["music", "no speech", ""]:
            transcript_data.append({
                "chunk": chunk_index,
                "start": format_timestamp(int(start_time)),
                "end": format_timestamp(int(end_time)),
                "text": transcription
            })

        # Explicitly delete tensors to free GPU memory immediately
        del input_values, logits, predicted_ids
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return transcript_data


def _transcribe_audio_full(
    file_path: str,
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    device: torch.device,
    chunk_duration: int = 30,
    target_sr: int = 16000
):
    """Transcribe by loading full audio file (original behavior).

    Suitable for smaller audio files where memory is not a concern.
    """
    # Load audio using shared utility
    waveform, _ = load_audio_file(
        file_path,
        target_sampling_rate=target_sr,
        mono=True,
        normalize=True
    )

    # Chunk audio
    chunk_samples = chunk_duration * target_sr
    total_samples = waveform.shape[1]
    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples

    logger.info(f"Splitting into {num_chunks} chunks...")

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
        transcriptions = decode_ctc_predictions(processor, predicted_ids)
        transcription = transcriptions[0] if transcriptions else ""

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
        help='Hugging Face model name (default: WAV2VEC2_MODEL_NAME env variable, '
             'then the public Persian Wav2Vec2 checkpoint)'
    )
    parser.add_argument(
        '--chunk-duration',
        type=int,
        default=env_int('WAV2VEC2_CHUNK_DURATION', 30),
        help='Chunk duration in seconds (default: WAV2VEC2_CHUNK_DURATION env variable, else 30)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save transcription (optional, prints to stdout if not specified)'
    )
    parser.add_argument(
        '--streaming',
        action='store_true',
        default=True,
        help='Use memory-efficient streaming for large audio files (default: True)'
    )
    parser.add_argument(
        '--no-streaming',
        action='store_false',
        dest='streaming',
        help='Disable streaming and load full audio file into memory'
    )
    parser.add_argument(
        '--max-memory-mb',
        type=float,
        default=500,
        help='Threshold in MB for automatic streaming (default: 500)'
    )

    args = parser.parse_args()

    # Setup logging
    from src.logging_config import setup_logging
    setup_logging(log_level="INFO")

    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        raise FileNotFoundError(f"Audio file not found: {args.audio_file}")

    # Load model
    model, processor, device = load_model(
        checkpoint_path=args.checkpoint,
        model_name=args.model
    )

    logger.info(f"Using device: {device}")
    logger.info(f"Transcribing: {args.audio_file}")

    # Transcribe
    transcript_data = transcribe_audio_file(
        file_path=args.audio_file,
        model=model,
        processor=processor,
        device=device,
        chunk_duration=args.chunk_duration,
        streaming=args.streaming,
        max_memory_mb=args.max_memory_mb
    )

    # Output results
    output_text = "\nTimestamped Transcription:\n\n"
    for entry in transcript_data:
        output_text += f"[{entry['start']} - {entry['end']}] {entry['text']}\n"

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_text)
        logger.info(f"Transcription saved to: {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()

