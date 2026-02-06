"""Evaluation script for fine-tuned Wav2Vec2 model."""

import os
import yaml
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from tqdm import tqdm

from src.data_loader import load_dataset, match_audio_with_transcripts
from src.dataset import AudioTranscriptDataset
from src.evaluator import evaluate_batch, evaluate_samples


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: PyTorch device
        
    Returns:
        model, processor, config
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    model_name = config.get('model', {}).get('name', 'jonatasgrosman/wav2vec2-large-xlsr-53-persian')
    
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, processor, config


def evaluate_test_set(
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    test_loader: DataLoader,
    device: torch.device
) -> dict:
    """Evaluate model on test set.
    
    Args:
        model: Fine-tuned model
        processor: Wav2Vec2Processor
        test_loader: Test DataLoader
        device: PyTorch device
        
    Returns:
        Dictionary with evaluation metrics and per-sample results
    """
    model.eval()
    all_predictions = []
    all_references = []
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_values = batch['input_values'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_values=input_values, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Decode predictions
            predicted_ids = torch.argmax(logits, dim=-1)
            predictions = processor.batch_decode(predicted_ids)
            
            # Get references
            references = []
            for label_seq in labels:
                label_seq_clean = label_seq[label_seq != -100]
                if len(label_seq_clean) > 0:
                    ref = processor.decode(label_seq_clean.cpu().tolist())
                    references.append(ref)
                else:
                    references.append("")
            
            all_predictions.extend(predictions)
            all_references.extend(references)
    
    # Calculate aggregate metrics
    aggregate_metrics = evaluate_batch(all_references, all_predictions)
    
    # Calculate per-sample metrics
    per_sample_metrics = evaluate_samples(all_references, all_predictions)
    
    return {
        'aggregate': aggregate_metrics,
        'per_sample': {
            'references': all_references,
            'predictions': all_predictions,
            'wer': per_sample_metrics['wer'],
            'cer': per_sample_metrics['cer'],
            'bleu': per_sample_metrics['bleu']
        }
    }


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned Wav2Vec2 model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/best_model.pt',
        help='Path to model checkpoint (default: outputs/best_model.pt)'
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        default=None,
        help='Path to test audio directory (overrides config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/evaluation_results.json',
        help='Path to save evaluation results'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    print(f"Loading model from: {args.checkpoint}")
    model, processor, model_config = load_model(args.checkpoint, device)
    
    # Load test dataset
    test_dir = args.test_dir or config['data']['test_audio_dir']
    
    # Try to match test audio files with transcripts
    # First, try to find transcripts in the JSON files
    test_audio_transcript_pairs = []
    
    if os.path.exists(test_dir):
        # Try to match with transcripts from JSON files
        json_files = [
            (os.path.join(config['data']['json_info_dir'], "postgres_laravel_calls.json"), "recording_file"),
            (os.path.join(config['data']['json_info_dir'], "postgres_public_call_transcripts.json"), "audio_file_path")
        ]
        
        test_audio_transcript_pairs = match_audio_with_transcripts(
            test_dir,
            json_files,
            config['data']['audio_files_dir']
        )
        
        # If no transcripts found, create dataset with empty transcripts (for inference-only evaluation)
        if len(test_audio_transcript_pairs) == 0:
            print(f"Warning: No transcripts found for test audio files in {test_dir}")
            print("Creating test dataset without transcripts (inference-only mode)...")
            test_audio_files = list(Path(test_dir).glob("*.wav")) + list(Path(test_dir).glob("*.mp3")) + list(Path(test_dir).glob("*.flac"))
            test_audio_transcript_pairs = [(str(f), "") for f in test_audio_files]
    else:
        print(f"Warning: Test directory not found: {test_dir}")
        print("Skipping test evaluation.")
        return
    
    if len(test_audio_transcript_pairs) == 0:
        print("No test samples found!")
        return
    
    print(f"Found {len(test_audio_transcript_pairs)} test samples")
    
    # Create test dataset
    test_dataset = AudioTranscriptDataset(
        audio_transcript_pairs=test_audio_transcript_pairs,
        processor=processor,
        target_sampling_rate=16000,
        max_audio_length=config['training'].get('max_audio_length')
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Evaluate
    results = evaluate_test_set(model, processor, test_loader, device)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Test Samples: {len(test_audio_transcript_pairs)}")
    print(f"\nAggregate Metrics:")
    print(f"  WER: {results['aggregate']['wer']:.4f}")
    print(f"  CER: {results['aggregate']['cer']:.4f}")
    print(f"  BLEU: {results['aggregate']['bleu']:.2f}")
    print("="*50)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

