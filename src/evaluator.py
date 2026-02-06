"""Evaluation metrics for ASR: WER, CER, BLEU, and additional metrics."""

import re
from typing import List, Dict, Tuple
from jiwer import wer, cer, compute_measures
try:
    from sacrebleu import BLEU
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        NLTK_AVAILABLE = True
    except ImportError:
        NLTK_AVAILABLE = False


def normalize_text(text: str) -> str:
    """Normalize text for evaluation (remove extra spaces, lowercase for English).
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def calculate_wer(references: List[str], predictions: List[str]) -> float:
    """Calculate Word Error Rate (WER).
    
    Args:
        references: List of reference (ground truth) texts
        predictions: List of predicted texts
        
    Returns:
        WER score (lower is better)
    """
    # Normalize texts
    refs_normalized = [normalize_text(ref) for ref in references]
    preds_normalized = [normalize_text(pred) for pred in predictions]
    
    return wer(refs_normalized, preds_normalized)


def calculate_cer(references: List[str], predictions: List[str]) -> float:
    """Calculate Character Error Rate (CER).
    
    Args:
        references: List of reference (ground truth) texts
        predictions: List of predicted texts
        
    Returns:
        CER score (lower is better)
    """
    # Normalize texts
    refs_normalized = [normalize_text(ref) for ref in references]
    preds_normalized = [normalize_text(pred) for pred in predictions]
    
    return cer(refs_normalized, preds_normalized)


def calculate_bleu(references: List[str], predictions: List[str]) -> float:
    """Calculate BLEU score.
    
    Args:
        references: List of reference (ground truth) texts
        predictions: List of predicted texts
        
    Returns:
        BLEU score (higher is better, 0-100 scale)
    """
    if SACREBLEU_AVAILABLE:
        # Use sacrebleu (recommended)
        bleu = BLEU()
        # sacrebleu expects references as list of lists
        refs = [[normalize_text(ref)] for ref in references]
        preds = [normalize_text(pred) for pred in predictions]
        score = bleu.corpus_score(preds, refs)
        return score.score  # Returns score on 0-100 scale
    elif NLTK_AVAILABLE:
        # Use NLTK as fallback
        smoothing = SmoothingFunction().method1
        scores = []
        for ref, pred in zip(references, predictions):
            ref_tokens = normalize_text(ref).split()
            pred_tokens = normalize_text(pred).split()
            if len(ref_tokens) == 0:
                continue
            score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
            scores.append(score)
        return sum(scores) / len(scores) * 100 if scores else 0.0
    else:
        print("Warning: Neither sacrebleu nor nltk available. BLEU score will be 0.")
        return 0.0


def calculate_word_accuracy(references: List[str], predictions: List[str]) -> float:
    """Calculate Word Accuracy (1 - WER).
    
    Args:
        references: List of reference (ground truth) texts
        predictions: List of predicted texts
        
    Returns:
        Word Accuracy score (higher is better, 0-1 scale)
    """
    wer_score = calculate_wer(references, predictions)
    return 1.0 - wer_score


def calculate_character_accuracy(references: List[str], predictions: List[str]) -> float:
    """Calculate Character Accuracy (1 - CER).
    
    Args:
        references: List of reference (ground truth) texts
        predictions: List of predicted texts
        
    Returns:
        Character Accuracy score (higher is better, 0-1 scale)
    """
    cer_score = calculate_cer(references, predictions)
    return 1.0 - cer_score


def calculate_detailed_wer_metrics(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """Calculate detailed WER metrics including substitutions, insertions, deletions.
    
    Args:
        references: List of reference (ground truth) texts
        predictions: List of predicted texts
        
    Returns:
        Dictionary with detailed WER metrics
    """
    refs_normalized = [normalize_text(ref) for ref in references]
    preds_normalized = [normalize_text(pred) for pred in predictions]
    
    # Use jiwer's compute_measures for detailed metrics
    measures = compute_measures(refs_normalized, preds_normalized)
    
    return {
        'wer': measures['wer'],
        'word_substitutions': measures['substitutions'],
        'word_insertions': measures['insertions'],
        'word_deletions': measures['deletions'],
        'word_hits': measures['hits'],
        'total_words': measures['hits'] + measures['substitutions'] + measures['deletions']
    }


def calculate_exact_match_accuracy(references: List[str], predictions: List[str]) -> float:
    """Calculate Exact Match Accuracy (percentage of predictions that exactly match references).
    
    Args:
        references: List of reference (ground truth) texts
        predictions: List of predicted texts
        
    Returns:
        Exact match accuracy (0-1 scale)
    """
    if len(references) != len(predictions):
        raise ValueError(f"References and predictions must have same length. "
                        f"Got {len(references)} and {len(predictions)}")
    
    exact_matches = 0
    for ref, pred in zip(references, predictions):
        if normalize_text(ref) == normalize_text(pred):
            exact_matches += 1
    
    return exact_matches / len(references) if len(references) > 0 else 0.0


def calculate_length_statistics(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """Calculate length-based statistics.
    
    Args:
        references: List of reference (ground truth) texts
        predictions: List of predicted texts
        
    Returns:
        Dictionary with length statistics
    """
    ref_lengths = [len(normalize_text(ref).split()) for ref in references]
    pred_lengths = [len(normalize_text(pred).split()) for pred in predictions]
    
    avg_ref_length = sum(ref_lengths) / len(ref_lengths) if ref_lengths else 0.0
    avg_pred_length = sum(pred_lengths) / len(pred_lengths) if pred_lengths else 0.0
    
    # Calculate length ratio
    length_ratios = []
    for ref_len, pred_len in zip(ref_lengths, pred_lengths):
        if ref_len > 0:
            length_ratios.append(pred_len / ref_len)
        elif pred_len == 0:
            length_ratios.append(1.0)
        else:
            length_ratios.append(float('inf'))
    
    avg_length_ratio = sum([r for r in length_ratios if r != float('inf')]) / len([r for r in length_ratios if r != float('inf')]) if length_ratios else 1.0
    
    return {
        'avg_reference_length': avg_ref_length,
        'avg_prediction_length': avg_pred_length,
        'avg_length_ratio': avg_length_ratio,
        'min_reference_length': min(ref_lengths) if ref_lengths else 0,
        'max_reference_length': max(ref_lengths) if ref_lengths else 0,
        'min_prediction_length': min(pred_lengths) if pred_lengths else 0,
        'max_prediction_length': max(pred_lengths) if pred_lengths else 0
    }


def evaluate_batch(
    references: List[str],
    predictions: List[str]
) -> Dict[str, float]:
    """Calculate all evaluation metrics for a batch.
    
    Args:
        references: List of reference (ground truth) texts
        predictions: List of predicted texts
        
    Returns:
        Dictionary with comprehensive evaluation metrics
    """
    if len(references) != len(predictions):
        raise ValueError(f"References and predictions must have same length. "
                        f"Got {len(references)} and {len(predictions)}")
    
    # Basic metrics
    wer_score = calculate_wer(references, predictions)
    cer_score = calculate_cer(references, predictions)
    bleu_score = calculate_bleu(references, predictions)
    
    # Accuracy metrics (derived from error rates)
    word_accuracy = calculate_word_accuracy(references, predictions)
    character_accuracy = calculate_character_accuracy(references, predictions)
    
    # Detailed WER metrics
    detailed_wer = calculate_detailed_wer_metrics(references, predictions)
    
    # Exact match accuracy
    exact_match = calculate_exact_match_accuracy(references, predictions)
    
    # Length statistics
    length_stats = calculate_length_statistics(references, predictions)
    
    # Combine all metrics
    metrics = {
        # Error rates (lower is better)
        'wer': wer_score,
        'cer': cer_score,
        
        # Accuracy metrics (higher is better)
        'word_accuracy': word_accuracy,
        'character_accuracy': character_accuracy,
        'exact_match_accuracy': exact_match,
        
        # Quality scores (higher is better)
        'bleu': bleu_score,
        
        # Detailed WER breakdown
        'word_substitutions': detailed_wer['word_substitutions'],
        'word_insertions': detailed_wer['word_insertions'],
        'word_deletions': detailed_wer['word_deletions'],
        'word_hits': detailed_wer['word_hits'],
        'total_words': detailed_wer['total_words'],
        
        # Length statistics
        **length_stats
    }
    
    return metrics


def evaluate_samples(
    references: List[str],
    predictions: List[str]
) -> Dict[str, List[float]]:
    """Calculate per-sample metrics.
    
    Args:
        references: List of reference (ground truth) texts
        predictions: List of predicted texts
        
    Returns:
        Dictionary with lists of per-sample metrics
    """
    if len(references) != len(predictions):
        raise ValueError(f"References and predictions must have same length. "
                        f"Got {len(references)} and {len(predictions)}")
    
    per_sample_wer = []
    per_sample_cer = []
    per_sample_bleu = []
    per_sample_word_accuracy = []
    per_sample_character_accuracy = []
    per_sample_exact_match = []
    
    for ref, pred in zip(references, predictions):
        ref_norm = normalize_text(ref)
        pred_norm = normalize_text(pred)
        
        # WER and CER
        sample_wer = wer([ref_norm], [pred_norm])
        sample_cer = cer([ref_norm], [pred_norm])
        per_sample_wer.append(sample_wer)
        per_sample_cer.append(sample_cer)
        
        # Accuracies (derived from error rates)
        per_sample_word_accuracy.append(1.0 - sample_wer)
        per_sample_character_accuracy.append(1.0 - sample_cer)
        
        # Exact match
        per_sample_exact_match.append(1.0 if ref_norm == pred_norm else 0.0)
        
        # BLEU
        if SACREBLEU_AVAILABLE:
            bleu = BLEU()
            score = bleu.corpus_score([pred_norm], [[ref_norm]])
            per_sample_bleu.append(score.score)
        elif NLTK_AVAILABLE:
            ref_tokens = ref_norm.split()
            pred_tokens = pred_norm.split()
            if len(ref_tokens) > 0:
                smoothing = SmoothingFunction().method1
                score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
                per_sample_bleu.append(score * 100)
            else:
                per_sample_bleu.append(0.0)
        else:
            per_sample_bleu.append(0.0)
    
    return {
        'wer': per_sample_wer,
        'cer': per_sample_cer,
        'bleu': per_sample_bleu,
        'word_accuracy': per_sample_word_accuracy,
        'character_accuracy': per_sample_character_accuracy,
        'exact_match': per_sample_exact_match
    }

