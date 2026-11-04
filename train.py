"""Main training script for fine-tuning Wav2Vec2 model."""

import torch
from torch.utils.data import DataLoader, random_split
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.config import load_config, get_hf_token
from src.data_loader import load_dataset
from src.dataset import AudioTranscriptDataset
from src.trainer import Trainer
from src.utils import log_gpu_memory
from src.logging_config import setup_logging, get_logger

# Setup logging
logger = get_logger(__name__)


def setup_device(config: dict) -> torch.device:
    """Setup PyTorch device.

    Args:
        config: Configuration dictionary

    Returns:
        PyTorch device
    """
    use_cuda = config.get('device', {}).get('use_cuda', True) and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    logger.info(f"Using device: {device}")
    if use_cuda:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        log_gpu_memory("Initial", device)
    return device


def main():
    """Main training function."""
    # Load configuration
    config = load_config()

    # Setup logging based on config (with defaults)
    log_file = config.get('output', {}).get('log_file')
    log_level = config.get('output', {}).get('log_level', 'INFO')
    setup_logging(log_file=log_file, log_level=log_level)

    logger.info("=" * 50)
    logger.info("Starting Wav2Vec2 Fine-tuning")
    logger.info("=" * 50)

    # Setup device
    device = setup_device(config)

    # Load dataset
    logger.info("Loading dataset...")
    audio_transcript_pairs = load_dataset(
        audio_files_dir=config['data']['audio_files_dir'],
        json_info_dir=config['data']['json_info_dir']
    )

    if len(audio_transcript_pairs) == 0:
        raise ValueError("No audio-transcript pairs found! Check your data paths.")

    logger.info(f"Loaded {len(audio_transcript_pairs)} audio-transcript pairs")

    # Load model and processor
    logger.info(f"Loading model: {config['model']['name']}...")
    # Get Hugging Face token from environment variable (optional)
    hf_token = get_hf_token()
    processor = Wav2Vec2Processor.from_pretrained(
        config['model']['name'],
        token=hf_token
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        config['model']['name'],
        token=hf_token,
        use_safetensors=True
    )

    # Freeze feature extractor (optional - can be enabled for faster training)
    # model.freeze_feature_extractor()

    # Create dataset
    dataset = AudioTranscriptDataset(
        audio_transcript_pairs=audio_transcript_pairs,
        processor=processor,
        target_sampling_rate=16000,
        max_audio_length=config['training'].get('max_audio_length')
    )

    # Split into train and validation (80/20 split)
    # Note: Test set is separate and loaded from test_audio folder during evaluation
    train_val_split = config['data']['train_val_split']
    train_size = int(train_val_split * len(dataset))
    val_size = len(dataset) - train_size

    # Use configurable random seed for reproducibility
    random_seed = config['training'].get('random_seed', 42)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    logger.info(f"Dataset Split:")
    logger.info(f"  Train samples: {len(train_dataset)} ({len(train_dataset)/len(dataset)*100:.1f}%)")
    logger.info(f"  Validation samples: {len(val_dataset)} ({len(val_dataset)/len(dataset)*100:.1f}%)")
    logger.info(f"  Test samples: Will be loaded from '{config['data']['test_audio_dir']}' during evaluation")

    # Create DataLoaders with custom collate function for padding
    from src.dataset import collate_fn

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn
    )

    # Create trainer
    # Note: 'device' must be forwarded so mixed_precision and
    # empty_cache_after_val settings actually take effect in Trainer.
    trainer_config = {
        **config['training'],
        'output_dir': config['output']['output_dir'],
        'device': config.get('device', {})
    }

    trainer = Trainer(
        model=model,
        processor=processor,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=trainer_config
    )

    # Start training
    trainer.train()

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()

