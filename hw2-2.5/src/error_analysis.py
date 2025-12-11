"""
Error Analysis: Classification and Localization Errors
"""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from tqdm import tqdm

from dataset import CocoSubsetDataset, collate_fn, prepare_coco_subset


def box_iou(boxes1, boxes2):
    """Calculate IoU between two sets of boxes"""
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    
    # Convert to corner format
    lt1 = boxes1[:, :2] - boxes1[:, 2:] / 2
    rb1 = boxes1[:, :2] + boxes1[:, 2:] / 2
    lt2 = boxes2[:, :2] - boxes2[:, 2:] / 2
    rb2 = boxes2[:, :2] + boxes2[:, 2:] / 2
    
    # Intersection
    lt = torch.max(lt1[:, None, :], lt2[None, :, :])
    rb = torch.min(rb1[:, None, :], rb2[None, :, :])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    # Union
    union = area1[:, None] + area2[None, :] - inter
    
    iou = inter / union
    return iou


class ErrorAnalyzer:
    def __init__(self, model, processor, class_names, device):
        self.model = model
        self.processor = processor
        self.class_names = class_names
        self.device = device
        
        # Error statistics
        self.classification_errors = defaultdict(int)
        self.localization_errors = defaultdict(int)
        self.false_positives = 0
        self.false_negatives = 0
        self.confusion_matrix = np.zeros((len(class_names), len(class_names)))
        
        # Examples for visualization
        self.error_examples = {
            'classification': [],
            'localization': [],
            'false_positive': [],
            'false_negative': []
        }
    
    @torch.no_grad()
    def analyze_batch(self, pixel_values, labels, images, threshold=0.5, iou_threshold=0.5):
        """Analyze errors in a batch"""
        outputs = self.model(pixel_values=pixel_values.to(self.device))
        
        for i, (output, image) in enumerate(zip(outputs.logits, images)):
            # Get predictions
            probas = output.softmax(-1)[:, :-1]
            keep = probas.max(-1).values > threshold
            
            pred_boxes = outputs.pred_boxes[i][keep]
            pred_scores = probas[keep].max(-1).values
            pred_labels = probas[keep].argmax(-1)
            
            # Get ground truth from labels[i]
            label = labels[i]
            gt_boxes = label['boxes'].to(self.device) if isinstance(label['boxes'], torch.Tensor) else torch.tensor(label['boxes']).to(self.device)
            gt_labels = label['class_labels'].to(self.device) if isinstance(label['class_labels'], torch.Tensor) else torch.tensor(label['class_labels']).to(self.device)
            
            if len(gt_boxes) == 0:
                continue
            
            # Match predictions to ground truth
            if len(pred_boxes) > 0:
                iou_matrix = box_iou(pred_boxes, gt_boxes)
                
                # For each GT, find best matching prediction
                matched_preds = set()
                for gt_idx in range(len(gt_boxes)):
                    ious = iou_matrix[:, gt_idx]
                    if len(ious) > 0 and ious.max() > iou_threshold:
                        pred_idx = ious.argmax().item()
                        matched_preds.add(pred_idx)
                        
                        # Check if classification is correct
                        if pred_labels[pred_idx] != gt_labels[gt_idx]:
                            # Classification error
                            self.classification_errors[
                                f"{self.class_names[gt_labels[gt_idx].item()]}->"
                                f"{self.class_names[pred_labels[pred_idx].item()]}"
                            ] += 1
                            
                            self.confusion_matrix[
                                gt_labels[gt_idx].item(),
                                pred_labels[pred_idx].item()
                            ] += 1
                            
                            if len(self.error_examples['classification']) < 10:
                                self.error_examples['classification'].append({
                                    'image': image,
                                    'gt_box': gt_boxes[gt_idx].cpu(),
                                    'pred_box': pred_boxes[pred_idx].cpu(),
                                    'gt_label': self.class_names[gt_labels[gt_idx].item()],
                                    'pred_label': self.class_names[pred_labels[pred_idx].item()],
                                    'score': pred_scores[pred_idx].item()
                                })
                        elif ious[pred_idx] < 0.75:
                            # Localization error
                            self.localization_errors[self.class_names[gt_labels[gt_idx].item()]] += 1
                            
                            if len(self.error_examples['localization']) < 10:
                                self.error_examples['localization'].append({
                                    'image': image,
                                    'gt_box': gt_boxes[gt_idx].cpu(),
                                    'pred_box': pred_boxes[pred_idx].cpu(),
                                    'label': self.class_names[gt_labels[gt_idx].item()],
                                    'iou': ious[pred_idx].item(),
                                    'score': pred_scores[pred_idx].item()
                                })
                    else:
                        # False negative (missed detection)
                        self.false_negatives += 1
                
                # Unmatched predictions are false positives
                for pred_idx in range(len(pred_boxes)):
                    if pred_idx not in matched_preds:
                        self.false_positives += 1
                        
                        if len(self.error_examples['false_positive']) < 10:
                            self.error_examples['false_positive'].append({
                                'image': image,
                                'pred_box': pred_boxes[pred_idx].cpu(),
                                'pred_label': self.class_names[pred_labels[pred_idx].item()],
                                'score': pred_scores[pred_idx].item()
                            })
            else:
                # No predictions means all GT are false negatives
                self.false_negatives += len(gt_boxes)
    
    def plot_confusion_matrix(self, output_path):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            self.confusion_matrix,
            annot=True,
            fmt='.0f',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cmap='Blues'
        )
        plt.title('Classification Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Confusion matrix saved to {output_path}")
    
    def plot_error_distribution(self, output_path):
        """Plot error distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Classification errors
        if self.classification_errors:
            ax = axes[0, 0]
            errors = sorted(self.classification_errors.items(), key=lambda x: x[1], reverse=True)[:10]
            ax.barh([e[0] for e in errors], [e[1] for e in errors])
            ax.set_xlabel('Count')
            ax.set_title('Top 10 Classification Errors')
            ax.invert_yaxis()
        
        # Localization errors
        if self.localization_errors:
            ax = axes[0, 1]
            errors = sorted(self.localization_errors.items(), key=lambda x: x[1], reverse=True)
            ax.bar(range(len(errors)), [e[1] for e in errors])
            ax.set_xticks(range(len(errors)))
            ax.set_xticklabels([e[0] for e in errors], rotation=45, ha='right')
            ax.set_ylabel('Count')
            ax.set_title('Localization Errors by Class')
        
        # False positives/negatives
        ax = axes[1, 0]
        ax.bar(['False Positives', 'False Negatives'], 
               [self.false_positives, self.false_negatives])
        ax.set_ylabel('Count')
        ax.set_title('Detection Errors')
        
        # Summary
        ax = axes[1, 1]
        ax.axis('off')
        total_errors = sum(self.classification_errors.values())
        total_loc_errors = sum(self.localization_errors.values())
        summary_text = f"""
        Error Summary:
        
        Classification Errors: {total_errors}
        Localization Errors: {total_loc_errors}
        False Positives: {self.false_positives}
        False Negatives: {self.false_negatives}
        """
        ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Error distribution saved to {output_path}")
    
    def visualize_errors(self, output_dir):
        """Visualize error examples"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for error_type, examples in self.error_examples.items():
            if not examples:
                continue
            
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.flatten()
            
            for idx, example in enumerate(examples[:10]):
                if idx >= len(axes):
                    break
                
                ax = axes[idx]
                
                # Convert tensor to PIL Image
                img = example['image'].cpu().permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                img_pil = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pil)
                
                h, w = img.shape[:2]
                
                if 'gt_box' in example:
                    # Draw ground truth (green)
                    box = example['gt_box'].numpy()
                    x1 = int((box[0] - box[2]/2) * w)
                    y1 = int((box[1] - box[3]/2) * h)
                    x2 = int((box[0] + box[2]/2) * w)
                    y2 = int((box[1] + box[3]/2) * h)
                    draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
                
                if 'pred_box' in example:
                    # Draw prediction (red)
                    box = example['pred_box'].numpy()
                    x1 = int((box[0] - box[2]/2) * w)
                    y1 = int((box[1] - box[3]/2) * h)
                    x2 = int((box[0] + box[2]/2) * w)
                    y2 = int((box[1] + box[3]/2) * h)
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                
                ax.imshow(img_pil)
                ax.axis('off')
                
                # Title
                if error_type == 'classification':
                    title = f"GT: {example['gt_label']}\nPred: {example['pred_label']}"
                elif error_type == 'localization':
                    title = f"{example['label']}\nIoU: {example['iou']:.2f}"
                elif error_type == 'false_positive':
                    title = f"FP: {example['pred_label']}\nScore: {example['score']:.2f}"
                else:
                    title = "False Negative"
                
                ax.set_title(title, fontsize=10)
            
            # Hide unused subplots
            for idx in range(len(examples), len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle(f'{error_type.replace("_", " ").title()} Examples', fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / f'{error_type}_examples.png', dpi=150)
            plt.close()
            
            print(f"Saved {error_type} examples")
    
    def save_report(self, output_path):
        """Save error analysis report"""
        report = {
            'classification_errors': dict(self.classification_errors),
            'localization_errors': dict(self.localization_errors),
            'false_positives': int(self.false_positives),
            'false_negatives': int(self.false_negatives),
            'confusion_matrix': self.confusion_matrix.tolist()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Error report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Error analysis for DETR')
    parser.add_argument('--data_dir', type=str, default='/home/salex139s/test/data/coco')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='/home/salex139s/test/visualizations')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Prepare dataset
    dataset_info = prepare_coco_subset(args.data_dir)
    dataset_info['selected_classes'] = config['selected_classes']
    
    processor = DetrImageProcessor.from_pretrained(config['model_name'])
    
    val_dataset = CocoSubsetDataset(
        img_folder=dataset_info['val_img_folder'],
        ann_file=dataset_info['val_ann_file'],
        processor=processor,
        selected_classes=dataset_info['selected_classes']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Load model
    model = DetrForObjectDetection.from_pretrained(
        config['model_name'],
        num_labels=config['num_classes'],
        ignore_mismatched_sizes=True
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create analyzer
    analyzer = ErrorAnalyzer(
        model=model,
        processor=processor,
        class_names=config['selected_classes'],
        device=device
    )
    
    # Analyze
    print("Running error analysis...")
    for batch in tqdm(val_loader):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        
        # Get original images (denormalize)
        images = pixel_values.clone()
        
        analyzer.analyze_batch(pixel_values, labels, images, 
                              threshold=args.threshold,
                              iou_threshold=args.iou_threshold)
    
    # Generate reports
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer.plot_confusion_matrix(output_dir / 'confusion_matrix.png')
    analyzer.plot_error_distribution(output_dir / 'error_distribution.png')
    analyzer.visualize_errors(output_dir / 'error_examples')
    analyzer.save_report(output_dir / 'error_report.json')
    
    print("\nError analysis complete!")


if __name__ == '__main__':
    main()

