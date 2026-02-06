"""Evaluation metrics for ASR: WER, CER, BLEU, and additional metrics."""

import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
from jiwer import wer, cer, process_words
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

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    rouge_scorer = None


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
    
    # Calculate WER
    wer_score = wer(refs_normalized, preds_normalized)
    
    # Process words to get detailed alignment information
    total_substitutions = 0
    total_insertions = 0
    total_deletions = 0
    total_hits = 0
    
    for ref, pred in zip(refs_normalized, preds_normalized):
        # Process words to get detailed metrics
        output = process_words(ref, pred)
        
        # Accumulate metrics from WordOutput
        total_substitutions += output.substitutions
        total_insertions += output.insertions
        total_deletions += output.deletions
        total_hits += output.hits
    
    total_words = total_hits + total_substitutions + total_deletions
    
    return {
        'wer': wer_score,
        'word_substitutions': total_substitutions,
        'word_insertions': total_insertions,
        'word_deletions': total_deletions,
        'word_hits': total_hits,
        'total_words': total_words
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


def calculate_rouge_scores(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    
    Args:
        references: List of reference (ground truth) texts
        predictions: List of predicted texts
        
    Returns:
        Dictionary with ROUGE scores
    """
    if not ROUGE_AVAILABLE:
        return {
            'rouge1_f': 0.0,
            'rouge1_p': 0.0,
            'rouge1_r': 0.0,
            'rouge2_f': 0.0,
            'rouge2_p': 0.0,
            'rouge2_r': 0.0,
            'rougeL_f': 0.0,
            'rougeL_p': 0.0,
            'rougeL_r': 0.0
        }
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = {'f': [], 'p': [], 'r': []}
    rouge2_scores = {'f': [], 'p': [], 'r': []}
    rougeL_scores = {'f': [], 'p': [], 'r': []}
    
    for ref, pred in zip(references, predictions):
        ref_norm = normalize_text(ref)
        pred_norm = normalize_text(pred)
        
        scores = scorer.score(ref_norm, pred_norm)
        
        for rouge_type, score_dict in [('rouge1', rouge1_scores), ('rouge2', rouge2_scores), ('rougeL', rougeL_scores)]:
            score = scores[rouge_type]
            score_dict['f'].append(score.fmeasure)
            score_dict['p'].append(score.precision)
            score_dict['r'].append(score.recall)
    
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0
    
    return {
        'rouge1_f': avg(rouge1_scores['f']),
        'rouge1_p': avg(rouge1_scores['p']),
        'rouge1_r': avg(rouge1_scores['r']),
        'rouge2_f': avg(rouge2_scores['f']),
        'rouge2_p': avg(rouge2_scores['p']),
        'rouge2_r': avg(rouge2_scores['r']),
        'rougeL_f': avg(rougeL_scores['f']),
        'rougeL_p': avg(rougeL_scores['p']),
        'rougeL_r': avg(rougeL_scores['r'])
    }


def calculate_word_f1_score(references: List[str], predictions: List[str]) -> float:
    """Calculate F1 score for word-level matching.
    
    Args:
        references: List of reference (ground truth) texts
        predictions: List of predicted texts
        
    Returns:
        F1 score (0-1 scale)
    """
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    
    for ref, pred in zip(references, predictions):
        ref_words = set(normalize_text(ref).split())
        pred_words = set(normalize_text(pred).split())
        
        true_positives = len(ref_words & pred_words)
        false_positives = len(pred_words - ref_words)
        false_negatives = len(ref_words - pred_words)
        
        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives
    
    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0.0
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1


def calculate_error_type_analysis(references: List[str], predictions: List[str]) -> Dict[str, any]:
    """Analyze types of errors made by the model.
    
    Args:
        references: List of reference (ground truth) texts
        predictions: List of predicted texts
        
    Returns:
        Dictionary with error type analysis
    """
    refs_normalized = [normalize_text(ref) for ref in references]
    preds_normalized = [normalize_text(pred) for pred in predictions]
    
    empty_predictions = sum(1 for pred in preds_normalized if not pred or pred.strip() == "")
    empty_references = sum(1 for ref in refs_normalized if not ref or ref.strip() == "")
    
    # Calculate length differences
    length_diffs = []
    for ref, pred in zip(refs_normalized, preds_normalized):
        ref_len = len(ref.split())
        pred_len = len(pred.split())
        length_diffs.append(pred_len - ref_len)
    
    avg_length_diff = sum(length_diffs) / len(length_diffs) if length_diffs else 0.0
    over_predictions = sum(1 for diff in length_diffs if diff > 0)
    under_predictions = sum(1 for diff in length_diffs if diff < 0)
    exact_length = sum(1 for diff in length_diffs if diff == 0)
    
    return {
        'empty_predictions': empty_predictions,
        'empty_references': empty_references,
        'avg_length_difference': avg_length_diff,
        'over_predictions': over_predictions,
        'under_predictions': under_predictions,
        'exact_length_predictions': exact_length,
        'over_prediction_rate': over_predictions / len(predictions) if predictions else 0.0,
        'under_prediction_rate': under_predictions / len(predictions) if predictions else 0.0
    }


def calculate_performance_by_length(
    references: List[str],
    predictions: List[str],
    length_bins: Optional[List[int]] = None
) -> Dict[str, Dict[str, float]]:
    """Calculate performance metrics grouped by reference length.
    
    Args:
        references: List of reference (ground truth) texts
        predictions: List of predicted texts
        length_bins: List of length thresholds (e.g., [5, 10, 20] for bins: <5, 5-10, 10-20, >20)
        
    Returns:
        Dictionary with metrics for each length category
    """
    if length_bins is None:
        length_bins = [5, 10, 20, 30]
    
    # Create bins
    bins = {}
    bin_labels = []
    
    # Add first bin
    bin_labels.append(f"<{length_bins[0]}")
    bins[bin_labels[-1]] = []
    
    # Add middle bins
    for i in range(len(length_bins) - 1):
        label = f"{length_bins[i]}-{length_bins[i+1]}"
        bin_labels.append(label)
        bins[label] = []
    
    # Add last bin
    bin_labels.append(f">{length_bins[-1]}")
    bins[bin_labels[-1]] = []
    
    # Categorize samples
    for ref, pred in zip(references, predictions):
        ref_len = len(normalize_text(ref).split())
        
        if ref_len < length_bins[0]:
            bins[bin_labels[0]].append((ref, pred))
        elif ref_len >= length_bins[-1]:
            bins[bin_labels[-1]].append((ref, pred))
        else:
            for i in range(len(length_bins) - 1):
                if length_bins[i] <= ref_len < length_bins[i+1]:
                    bins[bin_labels[i+1]].append((ref, pred))
                    break
    
    # Calculate metrics for each bin
    bin_metrics = {}
    for label, samples in bins.items():
        if len(samples) == 0:
            continue
        
        refs_bin = [ref for ref, _ in samples]
        preds_bin = [pred for _, pred in samples]
        
        bin_metrics[label] = {
            'count': len(samples),
            'wer': calculate_wer(refs_bin, preds_bin),
            'cer': calculate_cer(refs_bin, preds_bin),
            'word_accuracy': calculate_word_accuracy(refs_bin, preds_bin),
            'character_accuracy': calculate_character_accuracy(refs_bin, preds_bin),
            'exact_match': calculate_exact_match_accuracy(refs_bin, preds_bin)
        }
    
    return bin_metrics


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
    
    # Advanced metrics
    rouge_scores = calculate_rouge_scores(references, predictions)
    word_f1 = calculate_word_f1_score(references, predictions)
    error_analysis = calculate_error_type_analysis(references, predictions)
    
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
        'word_f1_score': word_f1,
        
        # ROUGE scores (higher is better)
        **rouge_scores,
        
        # Detailed WER breakdown
        'word_substitutions': detailed_wer['word_substitutions'],
        'word_insertions': detailed_wer['word_insertions'],
        'word_deletions': detailed_wer['word_deletions'],
        'word_hits': detailed_wer['word_hits'],
        'total_words': detailed_wer['total_words'],
        
        # Error type analysis
        **error_analysis,
        
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

