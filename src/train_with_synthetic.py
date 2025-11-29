#!/usr/bin/env python3
"""
Train DETR model with synthetic data
"""
import sys
sys.path.insert(0, '.')

import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from torch.utils.data import DataLoader
from pathlib import Path
import json
import argparse

from src.dataset_with_synthetic import create_combined_dataset
from src.dataset import collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/coco')
    parser.add_argument('--synthetic_dir', default='./data/synthetic')
    parser.add_argument('--output_dir', default='./outputs/with_synthetic')
    parser.add_argument('--model_name', default='facebook/detr-resnet-50')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--max_train', type=int, default=5000)
    parser.add_argument('--max_val', type=int, default=500)
    
    args = parser.parse_args()
    
    SELECTED_CLASSES = [
        'person', 'car', 'dog', 'cat', 'chair',
        'bottle', 'bicycle', 'airplane', 'bus', 'train'
    ]
    
    print("Loading processor and model...")
    processor = DetrImageProcessor.from_pretrained(args.model_name)
    model = DetrForObjectDetection.from_pretrained(
        args.model_name,
        num_labels=len(SELECTED_CLASSES),
        ignore_mismatched_sizes=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"\n{'='*60}")
    print("Creating datasets...")
    print(f"{'='*60}\n")
    
    # Train dataset WITH synthetic
    train_dataset = create_combined_dataset(
        coco_dir=args.data_dir,
        synthetic_dir=args.synthetic_dir,
        split='train',
        processor=processor,
        selected_classes=SELECTED_CLASSES,
        max_coco_samples=args.max_train,
        use_synthetic=True
    )
    
    # Val dataset (no synthetic)
    val_dataset = create_combined_dataset(
        coco_dir=args.data_dir,
        synthetic_dir=None,
        split='val',
        processor=processor,
        selected_classes=SELECTED_CLASSES,
        max_coco_samples=args.max_val,
        use_synthetic=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4)
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}\n")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    
    # Training
    output_dir = Path(args.output_dir)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = {
        'model': args.model_name,
        'classes': SELECTED_CLASSES,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'synthetic_used': True,
        'batch_size': args.batch_size,
        'epochs': args.num_epochs
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    best_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        train_loss = 0
        
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 60)
        
        for batch_idx, (pixel_values, labels) in enumerate(train_loader):
            pixel_values = pixel_values.to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
            
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for pixel_values, labels in val_loader:
                pixel_values = pixel_values.to(device)
                labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
                
                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        checkpoint_path = output_dir / 'checkpoints' / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            best_path = output_dir / 'checkpoints' / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best model: {output_dir / 'checkpoints' / 'best_model.pt'}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

