"""
Extended Dataset with Synthetic Data Support
"""
import os
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import numpy as np
from src.dataset import CocoSubsetDataset


class SyntheticDataset(Dataset):
    """Dataset for synthetic images generated via augmentation"""
    
    def __init__(self, synthetic_dir, processor, selected_classes):
        self.synthetic_dir = Path(synthetic_dir)
        self.processor = processor
        self.selected_classes = selected_classes
        
        # Create class name to index mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
        
        # Load all synthetic images
        self.samples = []
        for class_name in selected_classes:
            class_dir = self.synthetic_dir / class_name
            if not class_dir.exists():
                continue
            
            # Find all PNG files
            for img_path in class_dir.glob("*.png"):
                metadata_path = img_path.with_suffix('.json')
                self.samples.append({
                    'image_path': img_path,
                    'class': class_name,
                    'class_idx': self.class_to_idx[class_name],
                    'metadata_path': metadata_path if metadata_path.exists() else None
                })
        
        print(f"Loaded {len(self.samples)} synthetic images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # For synthetic images, create a simple full-image bounding box
        # (since we don't have precise annotations)
        w, h = image.size
        
        # Create COCO-format annotation
        # Using the full image as bbox (approximation)
        target = {
            'image_id': idx,
            'annotations': [{
                'bbox': [0, 0, w, h],  # x, y, width, height
                'category_id': sample['class_idx'],
                'area': w * h,
                'iscrowd': 0
            }]
        }
        
        # Process through DETR processor
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        
        # Remove batch dimension
        pixel_values = encoding["pixel_values"].squeeze(0)
        
        # encoding["labels"] is a list with one element
        labels = encoding["labels"][0] if isinstance(encoding["labels"], list) else encoding["labels"]
        
        # Return only 2 elements to match CocoSubsetDataset
        return pixel_values, labels


def create_combined_dataset(
    coco_dir,
    synthetic_dir,
    split,
    processor,
    selected_classes,
    max_coco_samples=None,
    use_synthetic=True
):
    """Create dataset combining COCO and synthetic data"""
    
    # COCO dataset
    if split == 'train':
        img_folder = os.path.join(coco_dir, 'train2017')
        ann_file = os.path.join(coco_dir, 'annotations', 'instances_train2017.json')
    else:
        img_folder = os.path.join(coco_dir, 'val2017')
        ann_file = os.path.join(coco_dir, 'annotations', 'instances_val2017.json')
    
    coco_dataset = CocoSubsetDataset(
        img_folder=img_folder,
        ann_file=ann_file,
        processor=processor,
        selected_classes=selected_classes,
        max_samples=max_coco_samples
    )
    
    print(f"COCO {split}: {len(coco_dataset)} samples")
    
    # Add synthetic data only for training
    if use_synthetic and split == 'train' and synthetic_dir and os.path.exists(synthetic_dir):
        try:
            synthetic_dataset = SyntheticDataset(
                synthetic_dir=synthetic_dir,
                processor=processor,
                selected_classes=selected_classes
            )
            
            if len(synthetic_dataset) > 0:
                combined_dataset = ConcatDataset([coco_dataset, synthetic_dataset])
                print(f"Combined dataset: {len(coco_dataset)} COCO + {len(synthetic_dataset)} synthetic = {len(combined_dataset)} total")
                return combined_dataset
            else:
                print("No synthetic samples found, using COCO only")
                return coco_dataset
        except Exception as e:
            print(f"Warning: Could not load synthetic data: {e}")
            print("Using COCO dataset only")
            return coco_dataset
    
    return coco_dataset

