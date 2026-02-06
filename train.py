"""Main training script for fine-tuning Wav2Vec2 model."""

import os
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src.data_loader import load_dataset
from src.dataset import AudioTranscriptDataset
from src.trainer import Trainer


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config: dict) -> torch.device:
    """Setup PyTorch device.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        PyTorch device
    """
    use_cuda = config.get('device', {}).get('use_cuda', True) and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"Using device: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    return device


def main():
    """Main training function."""
    # Load configuration
    config = load_config()
    
    # Setup device
    device = setup_device(config)
    
    # Load dataset
    print("Loading dataset...")
    audio_transcript_pairs = load_dataset(
        audio_files_dir=config['data']['audio_files_dir'],
        json_info_dir=config['data']['json_info_dir']
    )
    
    if len(audio_transcript_pairs) == 0:
        raise ValueError("No audio-transcript pairs found! Check your data paths.")
    
    print(f"Loaded {len(audio_transcript_pairs)} audio-transcript pairs")
    
    # Load model and processor
    print(f"Loading model: {config['model']['name']}...")
    processor = Wav2Vec2Processor.from_pretrained(config['model']['name'])
    model = Wav2Vec2ForCTC.from_pretrained(config['model']['name'])
    
    # Freeze feature extractor (optional - can be enabled for faster training)
    # model.freeze_feature_extractor()
    
    # Create dataset
    dataset = AudioTranscriptDataset(
        audio_transcript_pairs=audio_transcript_pairs,
        processor=processor,
        target_sampling_rate=16000,
        max_audio_length=config['training'].get('max_audio_length')
    )
    
    # Split into train and validation
    train_val_split = config['data']['train_val_split']
    train_size = int(train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create trainer
    trainer_config = {
        **config['training'],
        'output_dir': config['output']['output_dir']
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
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()

