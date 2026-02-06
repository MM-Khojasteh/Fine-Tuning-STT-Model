"""Training utilities for Wav2Vec2 fine-tuning."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from tqdm import tqdm
import json
from typing import Dict, Optional
from pathlib import Path

from src.evaluator import evaluate_batch


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
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 3e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Setup learning rate scheduler
        num_training_steps = len(train_loader) * config.get('num_epochs', 3)
        warmup_steps = config.get('warmup_steps', 500)
        
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        ) if warmup_steps > 0 else None
        
        # CTC Loss - blank token is typically 0 for Wav2Vec2
        # Wav2Vec2 uses 0 as the blank/padding token in CTC
        blank_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
        self.criterion = nn.CTCLoss(blank=blank_token_id, reduction='mean', zero_infinity=True)
        
        # Training state
        self.best_val_wer = float('inf')
        self.training_history = []
        
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
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_values = batch['input_values'].to(self.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_values=input_values, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Calculate CTC loss
            # CTC expects: log_probs (time, batch, vocab), targets (1D), input_lengths, target_lengths
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            
            # Get input and target lengths
            if attention_mask is not None:
                input_lengths = attention_mask.sum(dim=1).long()
            else:
                input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=self.device)
            
            target_lengths = (labels != -100).sum(dim=1).long()
            
            # Filter out samples with invalid lengths
            valid_mask = (target_lengths > 0) & (input_lengths > 0)
            if not valid_mask.any():
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
                print(f"Warning: Skipping batch due to CTC error: {e}")
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
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
        return {'loss': avg_loss}
    
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
        all_predictions = []
        all_references = []
        
        progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                input_values = batch['input_values'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_values=input_values, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Decode predictions
                predicted_ids = torch.argmax(logits, dim=-1)
                predictions = self.processor.batch_decode(predicted_ids)
                
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
                
                if attention_mask is not None:
                    input_lengths = attention_mask.sum(dim=1).long()
                else:
                    input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=self.device)
                
                target_lengths = (labels != -100).sum(dim=1).long()
                
                valid_mask = (target_lengths > 0) & (input_lengths > 0)
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
                        except RuntimeError:
                            pass
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Calculate evaluation metrics
        metrics = evaluate_batch(all_references, all_predictions)
        metrics['loss'] = avg_loss
        
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
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with WER: {metrics['wer']:.4f}")
    
    def train(self):
        """Run full training loop."""
        num_epochs = self.config.get('num_epochs', 3)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Check if best model
            is_best = val_metrics['wer'] < self.best_val_wer
            if is_best:
                self.best_val_wer = val_metrics['wer']
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            # Log metrics
            epoch_log = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'validation': val_metrics
            }
            self.training_history.append(epoch_log)
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val WER: {val_metrics['wer']:.4f}")
            print(f"Val CER: {val_metrics['cer']:.4f}")
            print(f"Val BLEU: {val_metrics['bleu']:.2f}")
            print("-" * 50)
        
        # Save training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        
        print(f"\nTraining completed! Best validation WER: {self.best_val_wer:.4f}")
        print(f"Model saved to: {self.output_dir}")

