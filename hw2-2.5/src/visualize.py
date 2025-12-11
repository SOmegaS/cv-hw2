"""
Visualization utilities
"""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from tqdm import tqdm

from dataset import CocoSubsetDataset, collate_fn, prepare_coco_subset


def visualize_predictions(
    model,
    processor,
    dataset,
    class_names,
    device,
    num_images=10,
    threshold=0.5,
    output_dir=None
):
    """Visualize model predictions"""
    model.eval()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select random images
    indices = np.random.choice(len(dataset), min(num_images, len(dataset)), replace=False)
    
    for idx, img_idx in enumerate(tqdm(indices, desc="Visualizing")):
        pixel_values, label = dataset[img_idx]
        
        # Run inference
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values.unsqueeze(0).to(device))
        
        # Get predictions
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold
        
        pred_boxes = outputs.pred_boxes[0, keep].cpu()
        pred_scores = probas[keep].max(-1).values.cpu()
        pred_labels = probas[keep].argmax(-1).cpu()
        
        # Get ground truth
        gt_boxes = label['boxes']
        gt_labels = label['class_labels']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Denormalize image
        image = pixel_values.permute(1, 2, 0).numpy()
        image = (image - image.min()) / (image.max() - image.min())
        
        h, w = image.shape[:2]
        
        # Plot ground truth
        ax1.imshow(image)
        ax1.set_title('Ground Truth', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        for box, label_idx in zip(gt_boxes, gt_labels):
            cx, cy, bw, bh = box.numpy()
            x = (cx - bw/2) * w
            y = (cy - bh/2) * h
            width = bw * w
            height = bh * h
            
            rect = patches.Rectangle(
                (x, y), width, height,
                linewidth=2, edgecolor='green', facecolor='none'
            )
            ax1.add_patch(rect)
            
            label_name = class_names[label_idx.item()]
            ax1.text(
                x, y - 5,
                label_name,
                color='white',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.8)
            )
        
        # Plot predictions
        ax2.imshow(image)
        ax2.set_title('Predictions', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        for box, score, label_idx in zip(pred_boxes, pred_scores, pred_labels):
            cx, cy, bw, bh = box.numpy()
            x = (cx - bw/2) * w
            y = (cy - bh/2) * h
            width = bw * w
            height = bh * h
            
            rect = patches.Rectangle(
                (x, y), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax2.add_patch(rect)
            
            label_name = class_names[label_idx.item()]
            ax2.text(
                x, y - 5,
                f'{label_name} {score:.2f}',
                color='white',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.8)
            )
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(output_dir / f'prediction_{idx:03d}.png', dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def plot_training_curves(log_dir, output_path):
    """Plot training curves from TensorBoard logs"""
    from tensorboard.backend.event_processing import event_accumulator
    
    # Load TensorBoard logs
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()
    
    # Get available tags
    tags = ea.Tags()['scalars']
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot total loss
    if 'train/loss' in tags and 'val/loss' in tags:
        ax = axes[0, 0]
        
        train_loss = [(s.step, s.value) for s in ea.Scalars('train/loss')]
        val_loss = [(s.step, s.value) for s in ea.Scalars('val/loss')]
        
        ax.plot([s[0] for s in train_loss], [s[1] for s in train_loss], label='Train', alpha=0.7)
        ax.plot([s[0] for s in val_loss], [s[1] for s in val_loss], label='Val', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot classification loss
    if 'train/loss_ce' in tags and 'val/loss_ce' in tags:
        ax = axes[0, 1]
        
        train_ce = [(s.step, s.value) for s in ea.Scalars('train/loss_ce')]
        val_ce = [(s.step, s.value) for s in ea.Scalars('val/loss_ce')]
        
        ax.plot([s[0] for s in train_ce], [s[1] for s in train_ce], label='Train', alpha=0.7)
        ax.plot([s[0] for s in val_ce], [s[1] for s in val_ce], label='Val', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Classification Loss (CE)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot bbox loss
    if 'train/loss_bbox' in tags and 'val/loss_bbox' in tags:
        ax = axes[1, 0]
        
        train_bbox = [(s.step, s.value) for s in ea.Scalars('train/loss_bbox')]
        val_bbox = [(s.step, s.value) for s in ea.Scalars('val/loss_bbox')]
        
        ax.plot([s[0] for s in train_bbox], [s[1] for s in train_bbox], label='Train', alpha=0.7)
        ax.plot([s[0] for s in val_bbox], [s[1] for s in val_bbox], label='Val', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('BBox Regression Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot GIoU loss
    if 'train/loss_giou' in tags and 'val/loss_giou' in tags:
        ax = axes[1, 1]
        
        train_giou = [(s.step, s.value) for s in ea.Scalars('train/loss_giou')]
        val_giou = [(s.step, s.value) for s in ea.Scalars('val/loss_giou')]
        
        ax.plot([s[0] for s in train_giou], [s[1] for s in train_giou], label='Train', alpha=0.7)
        ax.plot([s[0] for s in val_giou], [s[1] for s in val_giou], label='Val', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('GIoU Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize DETR predictions')
    parser.add_argument('--data_dir', type=str, default='/home/salex139s/test/data/coco')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='/home/salex139s/test/visualizations')
    parser.add_argument('--num_images', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--plot_curves', action='store_true',
                        help='Plot training curves from TensorBoard logs')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='TensorBoard log directory for plotting curves')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.plot_curves and args.log_dir:
        print("Plotting training curves...")
        plot_training_curves(
            log_dir=args.log_dir,
            output_path=output_dir / 'training_curves.png'
        )
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset_info = prepare_coco_subset(args.data_dir)
    dataset_info['selected_classes'] = config['selected_classes']
    
    processor = DetrImageProcessor.from_pretrained(config['model_name'])
    
    val_dataset = CocoSubsetDataset(
        img_folder=dataset_info['val_img_folder'],
        ann_file=dataset_info['val_ann_file'],
        processor=processor,
        selected_classes=dataset_info['selected_classes']
    )
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = DetrForObjectDetection.from_pretrained(
        config['model_name'],
        num_labels=config['num_classes'],
        ignore_mismatched_sizes=True
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Visualize predictions
    print(f"Visualizing {args.num_images} predictions...")
    visualize_predictions(
        model=model,
        processor=processor,
        dataset=val_dataset,
        class_names=config['selected_classes'],
        device=device,
        num_images=args.num_images,
        threshold=args.threshold,
        output_dir=output_dir / 'predictions'
    )
    
    print(f"\nVisualizations saved to {output_dir}")


if __name__ == '__main__':
    main()

