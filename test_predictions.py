"""Comprehensive prediction test script with detailed analysis and sample outputs."""

import os
import yaml
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from tqdm import tqdm
from dotenv import load_dotenv
from collections import defaultdict

from src.data_loader import load_dataset, match_audio_with_transcripts, load_test_dataset
from src.dataset import AudioTranscriptDataset
from src.evaluator import (
    evaluate_batch, evaluate_samples, 
    calculate_performance_by_length,
    normalize_text
)

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


def get_predictions_with_details(
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    test_loader: DataLoader,
    device: torch.device
) -> list:
    """Get predictions with detailed information.
    
    Args:
        model: Fine-tuned model
        processor: Wav2Vec2Processor
        test_loader: Test DataLoader
        device: PyTorch device
        
    Returns:
        List of dictionaries with predictions and details
    """
    model.eval()
    all_results = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Predicting")):
            input_values = batch['input_values'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_values=input_values, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get prediction probabilities for confidence analysis
            probs = torch.softmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1)[0]
            avg_confidence = max_probs.mean(dim=1).cpu().tolist()
            
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
            
            # Store results with details
            for i, (ref, pred, conf) in enumerate(zip(references, predictions, avg_confidence)):
                all_results.append({
                    'reference': ref,
                    'prediction': pred,
                    'confidence': conf,
                    'batch_idx': batch_idx,
                    'sample_idx': len(all_results)
                })
    
    return all_results


def analyze_predictions(results: list) -> dict:
    """Analyze predictions and generate detailed statistics.
    
    Args:
        results: List of prediction results
        
    Returns:
        Dictionary with analysis results
    """
    references = [r['reference'] for r in results]
    predictions = [r['prediction'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    # Calculate metrics
    aggregate_metrics = evaluate_batch(references, predictions)
    per_sample_metrics = evaluate_samples(references, predictions)
    
    # Performance by length
    performance_by_length = calculate_performance_by_length(references, predictions)
    
    # Find best and worst predictions
    wer_scores = per_sample_metrics['wer']
    best_indices = sorted(range(len(wer_scores)), key=lambda i: wer_scores[i])[:10]
    worst_indices = sorted(range(len(wer_scores)), key=lambda i: wer_scores[i], reverse=True)[:10]
    
    best_predictions = [results[i] for i in best_indices]
    worst_predictions = [results[i] for i in worst_indices]
    
    # Confidence analysis
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    high_confidence_indices = [i for i, c in enumerate(confidences) if c > avg_confidence]
    low_confidence_indices = [i for i, c in enumerate(confidences) if c <= avg_confidence]
    
    high_conf_wer = [wer_scores[i] for i in high_confidence_indices]
    low_conf_wer = [wer_scores[i] for i in low_confidence_indices]
    
    avg_high_conf_wer = sum(high_conf_wer) / len(high_conf_wer) if high_conf_wer else 0.0
    avg_low_conf_wer = sum(low_conf_wer) / len(low_conf_wer) if low_conf_wer else 0.0
    
    return {
        'aggregate_metrics': aggregate_metrics,
        'per_sample_metrics': per_sample_metrics,
        'performance_by_length': performance_by_length,
        'best_predictions': best_predictions,
        'worst_predictions': worst_predictions,
        'confidence_analysis': {
            'avg_confidence': avg_confidence,
            'avg_high_conf_wer': avg_high_conf_wer,
            'avg_low_conf_wer': avg_low_conf_wer,
            'high_conf_count': len(high_confidence_indices),
            'low_conf_count': len(low_confidence_indices)
        }
    }


def print_detailed_report(results: list, analysis: dict):
    """Print comprehensive evaluation report.
    
    Args:
        results: List of prediction results
        analysis: Analysis dictionary
    """
    metrics = analysis['aggregate_metrics']
    conf_analysis = analysis['confidence_analysis']
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL PERFORMANCE REPORT")
    print("="*80)
    print(f"Total Test Samples: {len(results)}")
    
    # Main Metrics
    print("\n" + "-"*80)
    print("MAIN PERFORMANCE METRICS")
    print("-"*80)
    print(f"\nError Rates (Lower is Better):")
    print(f"  Word Error Rate (WER):           {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
    print(f"  Character Error Rate (CER):      {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")
    
    print(f"\nAccuracy Metrics (Higher is Better):")
    print(f"  Word Accuracy:                   {metrics['word_accuracy']:.4f} ({metrics['word_accuracy']*100:.2f}%)")
    print(f"  Character Accuracy:               {metrics['character_accuracy']:.4f} ({metrics['character_accuracy']*100:.2f}%)")
    print(f"  Exact Match Accuracy:             {metrics['exact_match_accuracy']:.4f} ({metrics['exact_match_accuracy']*100:.2f}%)")
    print(f"  Word F1 Score:                    {metrics.get('word_f1_score', 0.0):.4f}")
    
    print(f"\nQuality Scores (Higher is Better):")
    print(f"  BLEU Score:                      {metrics['bleu']:.2f}")
    if 'rouge1_f' in metrics:
        print(f"  ROUGE-1 F-Score:                 {metrics['rouge1_f']:.4f}")
        print(f"  ROUGE-2 F-Score:                 {metrics['rouge2_f']:.4f}")
        print(f"  ROUGE-L F-Score:                 {metrics['rougeL_f']:.4f}")
    
    # Detailed WER Breakdown
    print(f"\n" + "-"*80)
    print("DETAILED WORD ERROR BREAKDOWN")
    print("-"*80)
    print(f"  Total Words:                     {metrics['total_words']:.0f}")
    print(f"  Correct Words (Hits):            {metrics['word_hits']:.0f}")
    print(f"  Substitutions:                   {metrics['word_substitutions']:.0f}")
    print(f"  Insertions:                      {metrics['word_insertions']:.0f}")
    print(f"  Deletions:                       {metrics['word_deletions']:.0f}")
    
    # Error Type Analysis
    print(f"\n" + "-"*80)
    print("ERROR TYPE ANALYSIS")
    print("-"*80)
    print(f"  Empty Predictions:               {metrics.get('empty_predictions', 0)}")
    print(f"  Over-predictions (longer):        {metrics.get('over_predictions', 0)} ({metrics.get('over_prediction_rate', 0.0)*100:.1f}%)")
    print(f"  Under-predictions (shorter):      {metrics.get('under_predictions', 0)} ({metrics.get('under_prediction_rate', 0.0)*100:.1f}%)")
    print(f"  Exact Length Predictions:         {metrics.get('exact_length_predictions', 0)}")
    print(f"  Average Length Difference:        {metrics.get('avg_length_difference', 0.0):.2f} words")
    
    # Confidence Analysis
    print(f"\n" + "-"*80)
    print("CONFIDENCE ANALYSIS")
    print("-"*80)
    print(f"  Average Confidence:              {conf_analysis['avg_confidence']:.4f}")
    print(f"  High Confidence Samples:         {conf_analysis['high_conf_count']} (WER: {conf_analysis['avg_high_conf_wer']:.4f})")
    print(f"  Low Confidence Samples:           {conf_analysis['low_conf_count']} (WER: {conf_analysis['avg_low_conf_wer']:.4f})")
    
    # Performance by Length
    if analysis['performance_by_length']:
        print(f"\n" + "-"*80)
        print("PERFORMANCE BY TRANSCRIPT LENGTH")
        print("-"*80)
        for length_range, perf in sorted(analysis['performance_by_length'].items()):
            print(f"\n  Length Range: {length_range} words ({perf['count']} samples)")
            print(f"    WER: {perf['wer']:.4f} | CER: {perf['cer']:.4f}")
            print(f"    Word Accuracy: {perf['word_accuracy']:.4f} | Exact Match: {perf['exact_match']:.4f}")
    
    # Best Predictions
    print(f"\n" + "-"*80)
    print("BEST PREDICTIONS (Lowest WER)")
    print("-"*80)
    for i, result in enumerate(analysis['best_predictions'][:5], 1):
        wer_score = analysis['per_sample_metrics']['wer'][result['sample_idx']]
        print(f"\n  Sample {i} (WER: {wer_score:.4f}, Confidence: {result['confidence']:.4f}):")
        print(f"    Reference:  {result['reference']}")
        print(f"    Prediction: {result['prediction']}")
    
    # Worst Predictions
    print(f"\n" + "-"*80)
    print("WORST PREDICTIONS (Highest WER)")
    print("-"*80)
    for i, result in enumerate(analysis['worst_predictions'][:5], 1):
        wer_score = analysis['per_sample_metrics']['wer'][result['sample_idx']]
        print(f"\n  Sample {i} (WER: {wer_score:.4f}, Confidence: {result['confidence']:.4f}):")
        print(f"    Reference:  {result['reference']}")
        print(f"    Prediction: {result['prediction']}")
    
    print("\n" + "="*80)


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive prediction test for fine-tuned Wav2Vec2 model')
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
        default='outputs/prediction_test_results.json',
        help='Path to save detailed results'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of sample predictions to show (default: all)'
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
                    print(f"  python test_predictions.py --checkpoint {checkpoints_dir / checkpoint_files[-1].name}")
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
    
    # Load test dataset
    test_dir = args.test_dir or config['data']['test_audio_dir']
    print(f"Loading test set from: {test_dir}")
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found: {test_dir}")
        return
    
    # Try to load from test.json first (preferred method for test_audio folder)
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
    
    if len(test_audio_transcript_pairs) == 0:
        print(f"Error: No transcripts found for test audio files in {test_dir}")
        print("Note: Test evaluation requires transcripts.")
        print("Please ensure either:")
        print("  1. test.json exists in the test_audio folder with 'audio_file_path' and 'transcript' fields, or")
        print("  2. Test audio files have matching transcripts in the main JSON files.")
        return
    
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
    
    # Get predictions
    results = get_predictions_with_details(model, processor, test_loader, device)
    
    # Analyze predictions
    analysis = analyze_predictions(results)
    
    # Print report
    print_detailed_report(results, analysis)
    
    # Save detailed results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare output data
    output_data = {
        'config': {
            'checkpoint': args.checkpoint,
            'test_samples': len(results),
            'device': str(device)
        },
        'aggregate_metrics': analysis['aggregate_metrics'],
        'performance_by_length': analysis['performance_by_length'],
        'confidence_analysis': analysis['confidence_analysis'],
        'best_predictions': [
            {
                'sample_idx': r['sample_idx'],
                'reference': r['reference'],
                'prediction': r['prediction'],
                'confidence': r['confidence'],
                'wer': analysis['per_sample_metrics']['wer'][r['sample_idx']],
                'cer': analysis['per_sample_metrics']['cer'][r['sample_idx']]
            }
            for r in analysis['best_predictions']
        ],
        'worst_predictions': [
            {
                'sample_idx': r['sample_idx'],
                'reference': r['reference'],
                'prediction': r['prediction'],
                'confidence': r['confidence'],
                'wer': analysis['per_sample_metrics']['wer'][r['sample_idx']],
                'cer': analysis['per_sample_metrics']['cer'][r['sample_idx']]
            }
            for r in analysis['worst_predictions']
        ],
        'all_predictions': [
            {
                'sample_idx': r['sample_idx'],
                'reference': r['reference'],
                'prediction': r['prediction'],
                'confidence': r['confidence'],
                'wer': analysis['per_sample_metrics']['wer'][r['sample_idx']],
                'cer': analysis['per_sample_metrics']['cer'][r['sample_idx']],
                'word_accuracy': analysis['per_sample_metrics']['word_accuracy'][r['sample_idx']],
                'exact_match': analysis['per_sample_metrics']['exact_match'][r['sample_idx']]
            }
            for r in results
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()

