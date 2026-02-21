"""Training utilities for Wav2Vec2 fine-tuning."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from tqdm import tqdm
import json
from typing import Dict, Optional
from pathlib import Path

from src.evaluator import evaluate_batch
from src.utils import (
    decode_ctc_predictions, clear_gpu_cache, log_gpu_memory,
    inspect_predicted_tokens, check_blank_token_prediction
)
from src.logging_config import get_logger

logger = get_logger(__name__)


class Trainer:
    """Trainer class for fine-tuning Wav2Vec2 model."""
    
    def __init__(
        self,
        model: Wav2Vec2ForCTC,
        processor: Wav2Vec2Processor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Dict
    ):
        """Initialize trainer.
        
        Args:
            model: Wav2Vec2ForCTC model
            processor: Wav2Vec2Processor
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            device: PyTorch device (cuda or cpu)
            config: Configuration dictionary
        """
        self.model = model.to(device)
        self.processor = processor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Setup optimizer
        learning_rate = config.get('learning_rate', 3e-4)
        weight_decay = config.get('weight_decay', 0.01)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Setup learning rate scheduler
        num_epochs = config.get('num_epochs', 3)
        num_training_steps = len(train_loader) * num_epochs

        warmup_steps = config.get('warmup_steps', 500)

        # Warmup scheduler (for initial warmup phase)
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        ) if warmup_steps > 0 else None

        # ReduceLROnPlateau scheduler (for reducing LR when validation plateaus)
        use_reduce_lr = config.get('use_reduce_lr_on_plateau', False)
        if use_reduce_lr:
            reduce_lr_factor = config.get('reduce_lr_factor', 0.5)
            reduce_lr_patience = config.get('reduce_lr_patience', 2)
            reduce_lr_min = config.get('reduce_lr_min', 1e-6)

            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',  # Minimize validation WER
                factor=reduce_lr_factor,
                patience=reduce_lr_patience,
                min_lr=reduce_lr_min,
                verbose=True
            )
        else:
            self.lr_scheduler = None
        
        # Track warmup steps
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
        # CTC Loss configuration
        # Wav2Vec2 typically uses 0 as the blank token, but we check the processor's pad_token_id
        # For CTC, the blank token should match what the model expects
        blank_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
        self.blank_token_id = blank_token_id
        
        # CTC Loss with proper configuration:
        # - blank: token ID for blank/padding (must match processor)
        # - reduction='mean': average loss over batch
        # - zero_infinity=True: replace infinite losses with zeros (handles invalid alignments)
        self.criterion = nn.CTCLoss(blank=blank_token_id, reduction='mean', zero_infinity=True)
        
        # Verify CTC configuration
        logger.info("CTC Loss Configuration:")
        logger.info(f"  Blank token ID: {blank_token_id}")
        logger.info(f"  Processor pad_token_id: {processor.tokenizer.pad_token_id}")
        logger.info(f"  Vocab size: {processor.tokenizer.vocab_size}")
        logger.info("  CTC Loss reduction: mean")
        logger.info("  Zero infinity: True")
        
        # Training state
        self.best_val_wer = float('inf')
        self.training_history = []
        self.epochs_without_improvement = 0
        
        # Mixed precision training
        self.use_mixed_precision = config.get('device', {}).get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_mixed_precision and device.type == 'cuda' else None
        
        # Early stopping
        self.early_stopping_patience = config.get('early_stopping_patience', None)
        self.early_stopping_min_delta = config.get('early_stopping_min_delta', 0.001)

        # Gradient clipping
        self.gradient_clip_norm = config.get('gradient_clip_norm', 1.0)
        
        # Create output directory
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        skipped_batches = 0
        skipped_samples = 0
        total_samples = 0
        
        # Gradient accumulation
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        self.optimizer.zero_grad()
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            batch_size = batch['input_values'].size(0)
            total_samples += batch_size
            # Move batch to device
            input_values = batch['input_values'].to(self.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass (with mixed precision if enabled)
            if self.use_mixed_precision and self.scaler is not None:
                with autocast():
                    outputs = self.model(input_values=input_values, attention_mask=attention_mask)
                    logits = outputs.logits
            else:
                outputs = self.model(input_values=input_values, attention_mask=attention_mask)
                logits = outputs.logits
            
            # Calculate CTC loss
            # CTC expects: log_probs (time, batch, vocab), targets (1D), input_lengths, target_lengths
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            
            # Get input and target lengths
            # CRITICAL FIX: Use logits.shape[1] (sequence length after feature extraction) 
            # instead of attention_mask length (raw audio length)
            # logits shape is (batch_size, sequence_length, vocab_size)
            sequence_length = logits.size(1)  # This is the correct sequence length for CTC
            batch_size = logits.size(0)
            input_lengths = torch.full((batch_size,), sequence_length, dtype=torch.long, device=self.device)
            
            target_lengths = (labels != -100).sum(dim=1).long()
            
            # Filter out samples with invalid lengths
            valid_mask = (target_lengths > 0) & (input_lengths > 0)
            valid_count = valid_mask.sum().item()
            skipped_samples += (batch_size - valid_count)
            
            if not valid_mask.any():
                skipped_batches += 1
                continue
                
            log_probs = log_probs[valid_mask]
            labels_valid = labels[valid_mask]
            input_lengths = input_lengths[valid_mask]
            target_lengths = target_lengths[valid_mask]
            
            # Prepare labels for CTC (remove -100 padding and flatten)
            labels_list = []
            for label_seq in labels_valid:
                clean_seq = label_seq[label_seq != -100]
                if len(clean_seq) > 0:
                    labels_list.append(clean_seq)
            
            if len(labels_list) == 0:
                skipped_batches += 1
                continue
            
            # Concatenate all labels into 1D tensor
            labels_flat = torch.cat(labels_list)
            
            # Calculate loss
            try:
                loss = self.criterion(
                    log_probs.transpose(0, 1),  # (time, batch, vocab)
                    labels_flat,  # (total_target_length,)
                    input_lengths,  # (batch,)
                    target_lengths  # (batch,)
                )
            except RuntimeError as e:
                # Skip batch if CTC loss calculation fails
                skipped_batches += 1
                logger.warning(f"Skipping batch {batch_idx} due to CTC error: {e}")
                continue
            
            # Backward pass (with gradient accumulation and mixed precision)
            if self.use_mixed_precision and self.scaler is not None:
                # Scale loss for mixed precision
                scaled_loss = self.scaler.scale(loss / gradient_accumulation_steps)
                scaled_loss.backward()
            else:
                # Normal backward pass with gradient accumulation
                (loss / gradient_accumulation_steps).backward()
            
            # Update weights only after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_mixed_precision and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad()

                # Update learning rate (warmup scheduler)
                self.current_step += 1
                if self.warmup_scheduler and self.current_step <= self.warmup_steps:
                    self.warmup_scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Report batch processing statistics
        processed_batches = len(self.train_loader) - skipped_batches
        skip_rate = (skipped_batches / len(self.train_loader) * 100) if len(self.train_loader) > 0 else 0.0

        if skipped_batches > 0:
            logger.info(f"Training Statistics - Processed: {processed_batches}/{len(self.train_loader)} batches "
                       f"({skip_rate:.1f}% skipped), Samples: {total_samples - skipped_samples}/{total_samples} "
                       f"({skipped_samples} filtered)")

        if skip_rate > 50.0:
            logger.warning(f"More than 50% of batches were skipped! This may indicate data issues.")
        
        return {
            'loss': avg_loss,
            'processed_batches': processed_batches,
            'skipped_batches': skipped_batches,
            'skip_rate': skip_rate,
            'total_samples': total_samples,
            'skipped_samples': skipped_samples
        }
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        skipped_batches = 0
        skipped_samples = 0
        total_samples = 0
        all_predictions = []
        all_references = []
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        with torch.no_grad():
            for batch in progress_bar:
                batch_size = batch['input_values'].size(0)
                total_samples += batch_size
                # Move batch to device
                input_values = batch['input_values'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_values=input_values, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Decode predictions with CTC blank token filtering
                predicted_ids = torch.argmax(logits, dim=-1)
                
                # Diagnostic check for blank token prediction (only on first batch of first epoch)
                if epoch == 0 and len(all_predictions) == 0:
                    blank_check = check_blank_token_prediction(predicted_ids, blank_token_id=self.processor.tokenizer.pad_token_id or 0)
                    if blank_check['is_mostly_blanks']:
                        logger.warning(f"Model is predicting {blank_check['blank_ratio']*100:.1f}% blank tokens!")
                        logger.warning("This may indicate a training issue. Inspecting token predictions...")
                        token_diagnostics = inspect_predicted_tokens(predicted_ids, self.processor, max_samples=2)
                        logger.warning(f"Token diagnostics: {token_diagnostics['overall']}")
                
                predictions = decode_ctc_predictions(self.processor, predicted_ids)
                
                # Get references (decode labels)
                references = []
                for label_seq in labels:
                    # Remove -100 padding
                    label_seq_clean = label_seq[label_seq != -100]
                    if len(label_seq_clean) > 0:
                        ref = self.processor.decode(label_seq_clean.cpu().tolist())
                        references.append(ref)
                    else:
                        references.append("")
                
                all_predictions.extend(predictions)
                all_references.extend(references)
                
                # Calculate loss (similar to training)
                log_probs = nn.functional.log_softmax(logits, dim=-1)
                
                # Get input and target lengths for CTC loss (same as training)
                sequence_length = logits.size(1)  # Sequence length after Wav2Vec2 feature extraction
                batch_size = logits.size(0)
                input_lengths = torch.full((batch_size,), sequence_length, dtype=torch.long, device=self.device)
                
                target_lengths = (labels != -100).sum(dim=1).long()
                
                valid_mask = (target_lengths > 0) & (input_lengths > 0)
                valid_count = valid_mask.sum().item()
                skipped_samples += (batch_size - valid_count)
                
                if valid_mask.any():
                    log_probs = log_probs[valid_mask]
                    labels_valid = labels[valid_mask]
                    input_lengths = input_lengths[valid_mask]
                    target_lengths = target_lengths[valid_mask]
                    
                    labels_list = []
                    for label_seq in labels_valid:
                        clean_seq = label_seq[label_seq != -100]
                        if len(clean_seq) > 0:
                            labels_list.append(clean_seq)
                    
                    if len(labels_list) > 0:
                        try:
                            labels_flat = torch.cat(labels_list)
                            loss = self.criterion(
                                log_probs.transpose(0, 1),
                                labels_flat,
                                input_lengths,
                                target_lengths
                            )
                            total_loss += loss.item()
                            num_batches += 1
                        except RuntimeError as e:
                            skipped_batches += 1
                            # Silently skip validation batches with errors
                            pass
                else:
                    skipped_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Calculate evaluation metrics
        metrics = evaluate_batch(all_references, all_predictions)
        metrics['loss'] = avg_loss
        
        # Add diagnostic information
        processed_batches = len(self.val_loader) - skipped_batches
        skip_rate = (skipped_batches / len(self.val_loader) * 100) if len(self.val_loader) > 0 else 0.0
        metrics['processed_batches'] = processed_batches
        metrics['skipped_batches'] = skipped_batches
        metrics['skip_rate'] = skip_rate
        metrics['total_samples'] = total_samples
        metrics['skipped_samples'] = skipped_samples
        
        # Clear GPU cache after validation if configured
        if self.config.get('device', {}).get('empty_cache_after_val', False):
            if self.device.type == 'cuda':
                clear_gpu_cache(self.device)
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save scheduler state dicts if they exist
        if self.warmup_scheduler:
            checkpoint['warmup_scheduler_state_dict'] = self.warmup_scheduler.state_dict()
        if self.lr_scheduler:
            # ReduceLROnPlateau doesn't have state_dict, but we can save its state
            checkpoint['lr_scheduler_state'] = {
                'best': self.lr_scheduler.best,
                'num_bad_epochs': self.lr_scheduler.num_bad_epochs,
                'last_epoch': self.lr_scheduler.last_epoch
            }
        
        # Save regular checkpoint
        # Note: Checkpoints are saved using torch.save() which may have security restrictions
        # with PyTorch < 2.6.0. When loading, ensure PyTorch >= 2.6.0 or handle the error appropriately.
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with WER: {metrics['wer']:.4f}")

    def train(self):
        """Run full training loop."""
        num_epochs = self.config.get('num_epochs', 3)

        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update learning rate scheduler (ReduceLROnPlateau)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(val_metrics['wer'])
            
            # Check if best model
            wer_improvement = self.best_val_wer - val_metrics['wer']
            is_best = val_metrics['wer'] < self.best_val_wer
            
            # Always update best_val_wer if this is the best model so far
            if is_best:
                self.best_val_wer = val_metrics['wer']
                # Only reset early stopping counter if improvement is significant
                if wer_improvement >= self.early_stopping_min_delta:
                    self.epochs_without_improvement = 0
                else:
                    # Still count as no improvement for early stopping (improvement too small)
                    self.epochs_without_improvement += 1
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            # Log metrics
            epoch_log = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'validation': val_metrics
            }
            self.training_history.append(epoch_log)
            
            # Save training history after each epoch (in case of interruption)
            try:
                history_path = self.output_dir / 'training_history.json'
                with open(history_path, 'w', encoding='utf-8') as f:
                    json.dump(self.training_history, f, indent=2, ensure_ascii=False)
            except Exception as e:
                # Don't fail training if history save fails
                pass
            
            # Log summary
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            if 'skip_rate' in train_metrics and train_metrics['skip_rate'] > 0:
                logger.info(f"  (Processed {train_metrics['processed_batches']}/{len(self.train_loader)} batches, "
                           f"{train_metrics['skip_rate']:.1f}% skipped)")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            if 'skip_rate' in val_metrics and val_metrics['skip_rate'] > 0:
                logger.info(f"  (Processed {val_metrics['processed_batches']}/{len(self.val_loader)} batches, "
                           f"{val_metrics['skip_rate']:.1f}% skipped)")
            logger.info("Validation Metrics:")
            logger.info(f"  WER: {val_metrics['wer']:.4f} | CER: {val_metrics['cer']:.4f}")
            logger.info(f"  Word Accuracy: {val_metrics['word_accuracy']:.4f} | Character Accuracy: {val_metrics['character_accuracy']:.4f}")
            logger.info(f"  Exact Match: {val_metrics['exact_match_accuracy']:.4f} | BLEU: {val_metrics['bleu']:.2f}")
            logger.info(f"  Word Errors - Substitutions: {val_metrics['word_substitutions']:.1f}, "
                       f"Insertions: {val_metrics['word_insertions']:.1f}, "
                       f"Deletions: {val_metrics['word_deletions']:.1f}")
            logger.info(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            if self.epochs_without_improvement > 0:
                logger.info(f"  Epochs without improvement: {self.epochs_without_improvement}")
            logger.info("-" * 50)

            # Early stopping check
            if self.early_stopping_patience is not None:
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                    logger.info(f"No improvement for {self.epochs_without_improvement} epochs.")
                    logger.info(f"Best validation WER: {self.best_val_wer:.4f}")
                    break

        # Save training history (always save, even if training was interrupted)
        try:
            history_path = self.output_dir / 'training_history.json'
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=2, ensure_ascii=False)
            logger.info(f"Training history saved to: {history_path}")
        except Exception as e:
            logger.warning(f"Failed to save training history: {e}")

        logger.info(f"Training completed! Best validation WER: {self.best_val_wer:.4f}")
        logger.info(f"Model saved to: {self.output_dir}")

