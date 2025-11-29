"""
Dataset preparation for COCO subset
"""
import os
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
from transformers import DetrImageProcessor
from PIL import Image
import numpy as np


class CocoSubsetDataset(Dataset):
    """COCO Subset Dataset for DETR training"""
    
    def __init__(
        self, 
        img_folder, 
        ann_file, 
        processor, 
        selected_classes=None,
        max_samples=None
    ):
        self.img_folder = img_folder
        self.processor = processor
        
        # Load COCO annotations
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Filter by selected classes if specified
        if selected_classes is not None:
            # Build mapping: COCO category_id -> 0-based index
            selected_cat_names = set(selected_classes)
            category_list = []
            for cat in self.coco_data['categories']:
                if cat['name'] in selected_cat_names:
                    category_list.append(cat)
            
            # Sort by name for consistent ordering
            category_list.sort(key=lambda x: selected_classes.index(x['name']))
            
            # Create mapping
            self.category_mapping = {cat['id']: idx for idx, cat in enumerate(category_list)}
            self.selected_cat_ids = set(self.category_mapping.keys())
            
            # Store as regular dict for pickling
            self.cat_id_to_idx = dict(self.category_mapping)
            
            # Filter images that contain selected categories
            valid_img_ids = set()
            for ann in self.coco_data['annotations']:
                if ann['category_id'] in self.selected_cat_ids:
                    valid_img_ids.add(ann['image_id'])
            
            self.images = [
                img for img in self.coco_data['images'] 
                if img['id'] in valid_img_ids
            ]
        else:
            self.images = self.coco_data['images']
            self.category_mapping = {
                cat['id']: idx for idx, cat in enumerate(self.coco_data['categories'])
            }
            self.cat_id_to_idx = dict(self.category_mapping)
        
        # Limit samples if specified
        if max_samples is not None:
            self.images = self.images[:max_samples]
        
        # Create annotation lookup
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        
        # Load image
        img_path = os.path.join(self.img_folder, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations
        anns = self.img_to_anns.get(img_id, [])
        
        img_w, img_h = image.size
        
        # Prepare annotations in COCO format for processor
        coco_annotations = []
        for ann in anns:
            if ann['category_id'] not in self.category_mapping:
                continue
            
            # COCO format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            
            # Keep original category_id for processor
            coco_annotations.append({
                'bbox': [x, y, w, h],
                'category_id': ann['category_id'],  # Use original COCO category_id
                'area': w * h,
                'iscrowd': 0
            })
        
        # Prepare target in COCO format
        target = {
            'image_id': img_id,
            'annotations': coco_annotations
        }
        
        # Apply processor (it will normalize boxes and convert format)
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        
        pixel_values = encoding["pixel_values"].squeeze()
        labels = encoding["labels"][0]
        
        # Remap category ids to 0-based indices
        if len(labels['class_labels']) > 0:
            remapped_labels = torch.tensor([
                self.cat_id_to_idx.get(cid.item(), 0)  # Use .get() for safety
                for cid in labels['class_labels']
            ], dtype=torch.int64)
            labels['class_labels'] = remapped_labels
        
        return pixel_values, labels


def prepare_coco_subset(
    data_dir,
    selected_classes=None,
    train_split='train2017',
    val_split='val2017'
):
    """
    Prepare COCO subset dataset
    
    Args:
        data_dir: Path to COCO dataset
        selected_classes: List of class names to include (None for all)
        train_split: Training split name
        val_split: Validation split name
    
    Returns:
        Dictionary with paths and metadata
    """
    if selected_classes is None:
        # Default: 10 diverse classes
        selected_classes = [
            'person', 'car', 'dog', 'cat', 'chair',
            'bottle', 'bicycle', 'airplane', 'bus', 'train'
        ]
    
    data_dir = Path(data_dir)
    
    return {
        'train_img_folder': data_dir / train_split,
        'train_ann_file': data_dir / f'annotations/instances_{train_split}.json',
        'val_img_folder': data_dir / val_split,
        'val_ann_file': data_dir / f'annotations/instances_{val_split}.json',
        'selected_classes': selected_classes
    }


def collate_fn(batch):
    """Custom collate function for DETR"""
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Find max dimensions
    max_h = max([img.shape[1] for img in pixel_values])
    max_w = max([img.shape[2] for img in pixel_values])
    
    # Pad images to same size
    padded_images = []
    for img in pixel_values:
        c, h, w = img.shape
        padded = torch.zeros((c, max_h, max_w), dtype=img.dtype)
        padded[:, :h, :w] = img
        padded_images.append(padded)
    
    return {
        'pixel_values': torch.stack(padded_images),
        'labels': labels
    }

