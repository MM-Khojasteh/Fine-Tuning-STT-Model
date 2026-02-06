"""Evaluation script for fine-tuned Wav2Vec2 model."""

import os
import yaml
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from tqdm import tqdm
from dotenv import load_dotenv

from src.data_loader import load_dataset, match_audio_with_transcripts, load_test_dataset
from src.dataset import AudioTranscriptDataset
from src.evaluator import evaluate_batch, evaluate_samples

# Load environment variables from .env file
load_dotenv()


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
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except (ValueError, RuntimeError) as e:
        if "torch.load" in str(e) or "CVE-2025-32434" in str(e):
            raise RuntimeError(
                f"Checkpoint loading failed due to PyTorch security restrictions. "
                f"Please upgrade PyTorch to 2.6.0+ or use safetensors format for checkpoints. "
                f"Error: {e}"
            ) from e
        raise
    
    config = checkpoint.get('config', {})
    
    model_name = config.get('model', {}).get('name', 'jonatasgrosman/wav2vec2-large-xlsr-53-persian')
    
    # Get Hugging Face token from environment variable (optional)
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    processor = Wav2Vec2Processor.from_pretrained(
        model_name,
        token=hf_token if hf_token else None
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        token=hf_token if hf_token else None,
        use_safetensors=True
    )
    
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
    
    # Calculate performance by length
    from src.evaluator import calculate_performance_by_length
    performance_by_length = calculate_performance_by_length(all_references, all_predictions)
    
    return {
        'aggregate': aggregate_metrics,
        'per_sample': {
            'references': all_references,
            'predictions': all_predictions,
            'wer': per_sample_metrics['wer'],
            'cer': per_sample_metrics['cer'],
            'bleu': per_sample_metrics['bleu'],
            'word_accuracy': per_sample_metrics['word_accuracy'],
            'character_accuracy': per_sample_metrics['character_accuracy'],
            'exact_match': per_sample_metrics['exact_match']
        },
        'performance_by_length': performance_by_length
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
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        
        # Check if outputs directory exists and list available checkpoints
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            checkpoints_dir = outputs_dir / "checkpoints"
            if checkpoints_dir.exists():
                checkpoint_files = list(checkpoints_dir.glob("*.pt"))
                if checkpoint_files:
                    print(f"\nAvailable checkpoints in {checkpoints_dir}:")
                    for cp in sorted(checkpoint_files):
                        print(f"  - {cp}")
                    print(f"\nYou can use one of these checkpoints with:")
                    print(f"  python evaluate.py --checkpoint {checkpoints_dir / checkpoint_files[-1].name}")
                else:
                    print(f"\nNo checkpoint files found in {checkpoints_dir}")
            else:
                print(f"\nCheckpoints directory not found: {checkpoints_dir}")
        
        print(f"\nTo train a model first, run:")
        print(f"  python train.py")
        print(f"\nAfter training, the best model will be saved to: outputs/best_model.pt")
        return
    
    print(f"Loading model from: {args.checkpoint}")
    model, processor, model_config = load_model(args.checkpoint, device)
    
    # Load test dataset from test_audio folder
    # Note: Test set is separate from train/validation split
    test_dir = args.test_dir or config['data']['test_audio_dir']
    
    print(f"Loading test set from: {test_dir}")
    
    # Try to load from test.json first (preferred method for test_audio folder)
    test_audio_transcript_pairs = []
    
    if os.path.exists(test_dir):
        # First, try loading from test.json in the test_audio directory
        test_audio_transcript_pairs = load_test_dataset(test_dir)
        
        if len(test_audio_transcript_pairs) > 0:
            print(f"Loaded {len(test_audio_transcript_pairs)} test samples from test.json")
        else:
            # Fall back to matching with main JSON files (for backward compatibility)
            print("test.json not found or empty, trying to match with main JSON files...")
            json_files = [
                (os.path.join(config['data']['json_info_dir'], "postgres_laravel_calls.json"), "recording_file"),
                (os.path.join(config['data']['json_info_dir'], "postgres_public_call_transcripts.json"), "audio_file_path")
            ]
            
            test_audio_transcript_pairs = match_audio_with_transcripts(
                test_dir,
                json_files,
                config['data']['audio_files_dir']
            )
            
            if len(test_audio_transcript_pairs) > 0:
                print(f"Loaded {len(test_audio_transcript_pairs)} test samples from main JSON files")
        
        # Test evaluation requires transcripts
        if len(test_audio_transcript_pairs) == 0:
            print(f"Error: No transcripts found for test audio files in {test_dir}")
            print("Note: Test evaluation requires transcripts.")
            print("Please ensure either:")
            print("  1. test.json exists in the test_audio folder with 'audio_file_path' and 'transcript' fields, or")
            print("  2. Test audio files have matching transcripts in the main JSON files.")
            return
    else:
        print(f"Error: Test directory not found: {test_dir}")
        print("Please ensure test audio files are in the test_audio folder.")
        return
    
    print(f"Found {len(test_audio_transcript_pairs)} test samples")
    
    # Create test dataset
    test_dataset = AudioTranscriptDataset(
        audio_transcript_pairs=test_audio_transcript_pairs,
        processor=processor,
        target_sampling_rate=16000,
        max_audio_length=config['training'].get('max_audio_length')
    )
    
    from src.dataset import collate_fn
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn
    )
    
    # Evaluate
    results = evaluate_test_set(model, processor, test_loader, device)
    
    # Print results
    print("\n" + "="*70)
    print("TEST SET EVALUATION RESULTS")
    print("="*70)
    print(f"Test Samples: {len(test_audio_transcript_pairs)}")
    print(f"\n{'='*70}")
    print("AGGREGATE METRICS")
    print(f"{'='*70}")
    
    metrics = results['aggregate']
    
    # Error Rates (lower is better)
    print(f"\nError Rates:")
    print(f"  Word Error Rate (WER):        {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
    print(f"  Character Error Rate (CER):    {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")
    
    # Accuracy Metrics (higher is better)
    print(f"\nAccuracy Metrics:")
    print(f"  Word Accuracy:                  {metrics['word_accuracy']:.4f} ({metrics['word_accuracy']*100:.2f}%)")
    print(f"  Character Accuracy:             {metrics['character_accuracy']:.4f} ({metrics['character_accuracy']*100:.2f}%)")
    print(f"  Exact Match Accuracy:           {metrics['exact_match_accuracy']:.4f} ({metrics['exact_match_accuracy']*100:.2f}%)")
    
    # Quality Scores
    print(f"\nQuality Scores:")
    print(f"  BLEU Score:                     {metrics['bleu']:.2f}")
    if 'word_f1_score' in metrics:
        print(f"  Word F1 Score:                  {metrics['word_f1_score']:.4f}")
    if 'rouge1_f' in metrics and metrics['rouge1_f'] > 0:
        print(f"  ROUGE-1 F-Score:                {metrics['rouge1_f']:.4f}")
        print(f"  ROUGE-2 F-Score:                {metrics['rouge2_f']:.4f}")
        print(f"  ROUGE-L F-Score:                {metrics['rougeL_f']:.4f}")
    
    # Error Type Analysis
    if 'empty_predictions' in metrics:
        print(f"\nError Type Analysis:")
        print(f"  Empty Predictions:              {metrics.get('empty_predictions', 0)}")
        print(f"  Over-predictions (longer):       {metrics.get('over_predictions', 0)} ({metrics.get('over_prediction_rate', 0.0)*100:.1f}%)")
        print(f"  Under-predictions (shorter):     {metrics.get('under_predictions', 0)} ({metrics.get('under_prediction_rate', 0.0)*100:.1f}%)")
        print(f"  Exact Length Predictions:        {metrics.get('exact_length_predictions', 0)}")
        print(f"  Average Length Difference:       {metrics.get('avg_length_difference', 0.0):.2f} words")
    
    # Detailed WER Breakdown
    print(f"\nWord Error Breakdown:")
    print(f"  Total Words:                    {metrics['total_words']:.0f}")
    print(f"  Correct Words (Hits):           {metrics['word_hits']:.0f}")
    print(f"  Substitutions:                  {metrics['word_substitutions']:.0f}")
    print(f"  Insertions:                     {metrics['word_insertions']:.0f}")
    print(f"  Deletions:                      {metrics['word_deletions']:.0f}")
    
    # Length Statistics
    print(f"\nLength Statistics:")
    print(f"  Avg Reference Length:          {metrics['avg_reference_length']:.2f} words")
    print(f"  Avg Prediction Length:          {metrics['avg_prediction_length']:.2f} words")
    print(f"  Avg Length Ratio:               {metrics['avg_length_ratio']:.3f}")
    print(f"  Reference Length Range:         {metrics['min_reference_length']:.0f} - {metrics['max_reference_length']:.0f} words")
    print(f"  Prediction Length Range:       {metrics['min_prediction_length']:.0f} - {metrics['max_prediction_length']:.0f} words")
    
    # Performance by Length Categories
    if 'performance_by_length' in results and results['performance_by_length']:
        print(f"\n{'='*70}")
        print("PERFORMANCE BY TRANSCRIPT LENGTH")
        print(f"{'='*70}")
        for length_range, perf in sorted(results['performance_by_length'].items()):
            print(f"\n  Length Range: {length_range} words ({perf['count']} samples)")
            print(f"    WER: {perf['wer']:.4f} | CER: {perf['cer']:.4f}")
            print(f"    Word Accuracy: {perf['word_accuracy']:.4f} | Character Accuracy: {perf['character_accuracy']:.4f}")
            print(f"    Exact Match: {perf['exact_match']:.4f}")
    
    # Sample Predictions (Best and Worst)
    if 'per_sample' in results:
        per_sample = results['per_sample']
        wer_scores = per_sample['wer']
        best_indices = sorted(range(len(wer_scores)), key=lambda i: wer_scores[i])[:3]
        worst_indices = sorted(range(len(wer_scores)), key=lambda i: wer_scores[i], reverse=True)[:3]
        
        print(f"\n{'='*70}")
        print("SAMPLE PREDICTIONS")
        print(f"{'='*70}")
        
        print(f"\nBest Predictions (Lowest WER):")
        for i, idx in enumerate(best_indices, 1):
            print(f"\n  Sample {i} (WER: {wer_scores[idx]:.4f}):")
            print(f"    Reference:  {per_sample['references'][idx]}")
            print(f"    Prediction: {per_sample['predictions'][idx]}")
        
        print(f"\nWorst Predictions (Highest WER):")
        for i, idx in enumerate(worst_indices, 1):
            print(f"\n  Sample {i} (WER: {wer_scores[idx]:.4f}):")
            print(f"    Reference:  {per_sample['references'][idx]}")
            print(f"    Prediction: {per_sample['predictions'][idx]}")
    
    print(f"\n{'='*70}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

