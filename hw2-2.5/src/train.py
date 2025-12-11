"""
DETR Training Script with TensorBoard logging and profiling
"""
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
    AutoModelForObjectDetection
)
from tqdm import tqdm
import numpy as np

from dataset import CocoSubsetDataset, collate_fn, prepare_coco_subset


class DETRTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        output_dir,
        writer,
        num_classes,
        profiler_steps=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.writer = writer
        self.num_classes = num_classes
        self.profiler_steps = profiler_steps
        
        self.global_step = 0
        self.best_loss = float('inf')
    
    def train_epoch(self, epoch, use_profiler=False):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_class_loss = 0
        total_bbox_loss = 0
        total_giou_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        if use_profiler and self.profiler_steps:
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.output_dir / 'profiler')),
                record_shapes=True,
                with_stack=True
            )
            prof.start()
        
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch['pixel_values'].to(self.device)
            labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch['labels']]
            
            # Forward pass
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            
            loss = outputs.loss
            loss_dict = outputs.loss_dict
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_class_loss += loss_dict.get('loss_ce', 0)
            total_bbox_loss += loss_dict.get('loss_bbox', 0)
            total_giou_loss += loss_dict.get('loss_giou', 0)
            
            # Log to TensorBoard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/loss_ce', loss_dict.get('loss_ce', 0), self.global_step)
                self.writer.add_scalar('train/loss_bbox', loss_dict.get('loss_bbox', 0), self.global_step)
                self.writer.add_scalar('train/loss_giou', loss_dict.get('loss_giou', 0), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ce': f"{loss_dict.get('loss_ce', 0):.4f}",
                'bbox': f"{loss_dict.get('loss_bbox', 0):.4f}"
            })
            
            self.global_step += 1
            
            if use_profiler and self.profiler_steps:
                prof.step()
                if batch_idx >= self.profiler_steps:
                    break
        
        if use_profiler and self.profiler_steps:
            prof.stop()
        
        avg_loss = total_loss / len(self.train_loader)
        avg_class_loss = total_class_loss / len(self.train_loader)
        avg_bbox_loss = total_bbox_loss / len(self.train_loader)
        avg_giou_loss = total_giou_loss / len(self.train_loader)
        
        return {
            'loss': avg_loss,
            'loss_ce': avg_class_loss,
            'loss_bbox': avg_bbox_loss,
            'loss_giou': avg_giou_loss
        }
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_class_loss = 0
        total_bbox_loss = 0
        total_giou_loss = 0
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(self.device)
            labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch['labels']]
            
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            
            loss = outputs.loss
            loss_dict = outputs.loss_dict
            
            total_loss += loss.item()
            total_class_loss += loss_dict.get('loss_ce', 0)
            total_bbox_loss += loss_dict.get('loss_bbox', 0)
            total_giou_loss += loss_dict.get('loss_giou', 0)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(self.val_loader)
        avg_class_loss = total_class_loss / len(self.val_loader)
        avg_bbox_loss = total_bbox_loss / len(self.val_loader)
        avg_giou_loss = total_giou_loss / len(self.val_loader)
        
        # Log to TensorBoard
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/loss_ce', avg_class_loss, epoch)
        self.writer.add_scalar('val/loss_bbox', avg_bbox_loss, epoch)
        self.writer.add_scalar('val/loss_giou', avg_giou_loss, epoch)
        
        return {
            'loss': avg_loss,
            'loss_ce': avg_class_loss,
            'loss_bbox': avg_bbox_loss,
            'loss_giou': avg_giou_loss
        }
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'global_step': self.global_step
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / 'checkpoints' / f'checkpoint_epoch_{epoch}.pt'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'checkpoints' / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model with loss: {metrics['loss']:.4f}")
    
    def train(self, num_epochs, profile_epoch=None):
        """Full training loop"""
        print(f"\n{'='*50}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*50}\n")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            use_profiler = (profile_epoch is not None and epoch == profile_epoch)
            train_metrics = self.train_epoch(epoch, use_profiler=use_profiler)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
            print(f"Train CE: {train_metrics['loss_ce']:.4f} | Val CE: {val_metrics['loss_ce']:.4f}")
            print(f"Train BBox: {train_metrics['loss_bbox']:.4f} | Val BBox: {val_metrics['loss_bbox']:.4f}")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_loss
            if is_best:
                self.best_loss = val_metrics['loss']
            
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
        
        print(f"\n{'='*50}")
        print(f"Training completed!")
        print(f"Best validation loss: {self.best_loss:.4f}")
        print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description='Train DETR on COCO subset')
    parser.add_argument('--data_dir', type=str, default='/home/salex139s/test/data/coco',
                        help='Path to COCO dataset')
    parser.add_argument('--output_dir', type=str, default='/home/salex139s/test',
                        help='Output directory')
    parser.add_argument('--model_name', type=str, default='facebook/detr-resnet-50',
                        help='Pretrained model name')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--profile_epoch', type=int, default=2,
                        help='Epoch to run profiler (None to disable)')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Maximum training samples (for testing)')
    parser.add_argument('--max_val_samples', type=int, default=None,
                        help='Maximum validation samples (for testing)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    
    # Setup TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=output_dir / 'logs' / f'run_{timestamp}')
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset_info = prepare_coco_subset(args.data_dir)
    
    # Initialize processor
    processor = DetrImageProcessor.from_pretrained(args.model_name)
    
    # Create datasets
    train_dataset = CocoSubsetDataset(
        img_folder=dataset_info['train_img_folder'],
        ann_file=dataset_info['train_ann_file'],
        processor=processor,
        selected_classes=dataset_info['selected_classes'],
        max_samples=args.max_train_samples
    )
    
    val_dataset = CocoSubsetDataset(
        img_folder=dataset_info['val_img_folder'],
        ann_file=dataset_info['val_ann_file'],
        processor=processor,
        selected_classes=dataset_info['selected_classes'],
        max_samples=args.max_val_samples
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Load model
    print(f"Loading model: {args.model_name}")
    num_classes = len(dataset_info['selected_classes'])
    
    model = DetrForObjectDetection.from_pretrained(
        args.model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Save config
    config = {
        'model_name': args.model_name,
        'num_classes': num_classes,
        'selected_classes': dataset_info['selected_classes'],
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.lr,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset)
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create trainer
    trainer = DETRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        writer=writer,
        num_classes=num_classes,
        profiler_steps=100 if args.profile_epoch else None
    )
    
    # Train
    trainer.train(num_epochs=args.num_epochs, profile_epoch=args.profile_epoch)
    
    # Close writer
    writer.close()
    
    print("Training complete!")


if __name__ == '__main__':
    main()

