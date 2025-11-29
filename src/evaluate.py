"""
Evaluation script with mAP calculation
"""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from tqdm import tqdm

from dataset import CocoSubsetDataset, collate_fn, prepare_coco_subset


class CocoEvaluator:
    def __init__(self, coco_gt, iou_types=["bbox"]):
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_eval = {}
        self.results = []
    
    def update(self, predictions):
        """Add predictions"""
        self.results.extend(predictions)
    
    def synchronize_between_processes(self):
        """Placeholder for distributed training"""
        pass
    
    def prepare(self):
        """Prepare for evaluation"""
        if len(self.results) == 0:
            print("No predictions to evaluate!")
            return
        
        # Create COCO results format
        coco_dt = self.coco_gt.loadRes(self.results)
        
        # Initialize COCOeval objects after loading results
        for iou_type in self.iou_types:
            coco_eval = COCOeval(self.coco_gt, coco_dt, iouType=iou_type)
            coco_eval.params.imgIds = list(set([r['image_id'] for r in self.results]))
            coco_eval.evaluate()
            self.coco_eval[iou_type] = coco_eval
    
    def accumulate(self):
        """Accumulate evaluation"""
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()
    
    def summarize(self):
        """Print summary"""
        print("\nEvaluation Results:")
        print("=" * 50)
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"\nIoU metric: {iou_type}")
            coco_eval.summarize()
        print("=" * 50)


def convert_to_coco_api(model_outputs, image_ids, orig_sizes, threshold=0.5):
    """Convert model outputs to COCO API format"""
    results = []
    
    for outputs, image_id, orig_size in zip(model_outputs, image_ids, orig_sizes):
        # Get predictions above threshold
        probas = outputs['logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold
        
        boxes = outputs['pred_boxes'][0, keep]
        scores = probas[keep].max(-1).values
        labels = probas[keep].argmax(-1)
        
        # Convert boxes to COCO format [x, y, width, height]
        orig_h, orig_w = orig_size
        
        # Denormalize boxes
        boxes_denorm = boxes.clone()
        boxes_denorm[:, 0] = boxes[:, 0] * orig_w  # cx
        boxes_denorm[:, 1] = boxes[:, 1] * orig_h  # cy
        boxes_denorm[:, 2] = boxes[:, 2] * orig_w  # w
        boxes_denorm[:, 3] = boxes[:, 3] * orig_h  # h
        
        # Convert from center format to corner format
        boxes_coco = boxes_denorm.clone()
        boxes_coco[:, 0] = boxes_denorm[:, 0] - boxes_denorm[:, 2] / 2  # x
        boxes_coco[:, 1] = boxes_denorm[:, 1] - boxes_denorm[:, 3] / 2  # y
        
        for box, score, label in zip(boxes_coco.tolist(), scores.tolist(), labels.tolist()):
            results.append({
                'image_id': int(image_id),
                'category_id': int(label) + 1,  # COCO categories start at 1
                'bbox': box,
                'score': float(score)
            })
    
    return results


@torch.no_grad()
def evaluate(model, data_loader, device, coco_evaluator, threshold=0.5):
    """Run evaluation"""
    model.eval()
    
    for batch in tqdm(data_loader, desc="Evaluating"):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels']
        
        # Get model outputs
        outputs = model(pixel_values=pixel_values)
        
        # Extract image info
        image_ids = [label['image_id'].item() for label in labels]
        orig_sizes = [label['orig_size'].tolist() for label in labels]
        
        # Convert predictions
        predictions = []
        for i in range(len(image_ids)):
            single_output = {
                'logits': outputs.logits[i:i+1],
                'pred_boxes': outputs.pred_boxes[i:i+1]
            }
            pred = convert_to_coco_api(
                [single_output],
                [image_ids[i]],
                [orig_sizes[i]],
                threshold=threshold
            )
            predictions.extend(pred)
        
        coco_evaluator.update(predictions)
    
    # Finalize evaluation
    coco_evaluator.prepare()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    
    # Extract metrics
    metrics = {}
    for iou_type, coco_eval in coco_evaluator.coco_eval.items():
        metrics[f'{iou_type}_mAP'] = coco_eval.stats[0]
        metrics[f'{iou_type}_mAP50'] = coco_eval.stats[1]
        metrics[f'{iou_type}_mAP75'] = coco_eval.stats[2]
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate DETR on COCO subset')
    parser.add_argument('--data_dir', type=str, default='/home/salex139s/test/data/coco',
                        help='Path to COCO dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--output', type=str, default='metrics.json',
                        help='Output file for metrics')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset_info = prepare_coco_subset(args.data_dir)
    dataset_info['selected_classes'] = config['selected_classes']
    
    # Initialize processor
    processor = DetrImageProcessor.from_pretrained(config['model_name'])
    
    # Create validation dataset
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
    print(f"Loading model from {args.checkpoint}")
    model = DetrForObjectDetection.from_pretrained(
        config['model_name'],
        num_labels=config['num_classes'],
        ignore_mismatched_sizes=True
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Setup COCO evaluator
    coco_gt = COCO(dataset_info['val_ann_file'])
    coco_evaluator = CocoEvaluator(coco_gt, iou_types=["bbox"])
    
    # Evaluate
    metrics = evaluate(model, val_loader, device, coco_evaluator, threshold=args.threshold)
    
    # Save metrics
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to {output_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()

