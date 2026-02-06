"""Evaluation metrics for ASR: WER, CER, and BLEU."""

import re
from typing import List, Dict
from jiwer import wer, cer
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


def evaluate_batch(
    references: List[str],
    predictions: List[str]
) -> Dict[str, float]:
    """Calculate all evaluation metrics for a batch.
    
    Args:
        references: List of reference (ground truth) texts
        predictions: List of predicted texts
        
    Returns:
        Dictionary with 'wer', 'cer', and 'bleu' scores
    """
    if len(references) != len(predictions):
        raise ValueError(f"References and predictions must have same length. "
                        f"Got {len(references)} and {len(predictions)}")
    
    metrics = {
        'wer': calculate_wer(references, predictions),
        'cer': calculate_cer(references, predictions),
        'bleu': calculate_bleu(references, predictions)
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
        Dictionary with lists of per-sample 'wer', 'cer', and 'bleu' scores
    """
    if len(references) != len(predictions):
        raise ValueError(f"References and predictions must have same length. "
                        f"Got {len(references)} and {len(predictions)}")
    
    per_sample_wer = []
    per_sample_cer = []
    per_sample_bleu = []
    
    for ref, pred in zip(references, predictions):
        per_sample_wer.append(wer([normalize_text(ref)], [normalize_text(pred)]))
        per_sample_cer.append(cer([normalize_text(ref)], [normalize_text(pred)]))
        
        if SACREBLEU_AVAILABLE:
            bleu = BLEU()
            score = bleu.corpus_score([normalize_text(pred)], [[normalize_text(ref)]])
            per_sample_bleu.append(score.score)
        elif NLTK_AVAILABLE:
            ref_tokens = normalize_text(ref).split()
            pred_tokens = normalize_text(pred).split()
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
        'bleu': per_sample_bleu
    }

