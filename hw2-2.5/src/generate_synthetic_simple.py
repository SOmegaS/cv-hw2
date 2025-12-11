"""
Synthetic Data Generation via Augmentation
Simple alternative to Stable Diffusion for HW 2.5
"""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import random
from tqdm import tqdm
import numpy as np


class SimpleDataAugmentor:
    """Generate 'synthetic' data via heavy augmentation"""
    
    def __init__(self, coco_dir):
        self.coco_dir = Path(coco_dir)
        
        # Load annotations to find images by class
        ann_file = self.coco_dir / 'annotations' / 'instances_train2017.json'
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create category mapping
        self.cat_name_to_id = {cat['name']: cat['id'] for cat in self.coco_data['categories']}
        
        # Create image index by category
        self.images_by_category = self._index_images_by_category()
    
    def _index_images_by_category(self):
        """Create mapping: category_id -> list of image_ids"""
        from collections import defaultdict
        cat_to_images = defaultdict(set)
        
        for ann in self.coco_data['annotations']:
            cat_id = ann['category_id']
            img_id = ann['image_id']
            cat_to_images[cat_id].add(img_id)
        
        return cat_to_images
    
    def augment_image(self, image, strength='medium'):
        """Apply random augmentations to image"""
        augmented = image.copy()
        
        # Rotation
        if random.random() > 0.5:
            angle = random.randint(-30, 30)
            augmented = augmented.rotate(angle, expand=True, fillcolor=(128, 128, 128))
        
        # Flip
        if random.random() > 0.5:
            augmented = augmented.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Color adjustments
        if random.random() > 0.3:
            enhancer = ImageEnhance.Brightness(augmented)
            augmented = enhancer.enhance(random.uniform(0.7, 1.3))
        
        if random.random() > 0.3:
            enhancer = ImageEnhance.Contrast(augmented)
            augmented = enhancer.enhance(random.uniform(0.8, 1.2))
        
        if random.random() > 0.3:
            enhancer = ImageEnhance.Color(augmented)
            augmented = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Blur
        if random.random() > 0.7:
            augmented = augmented.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        
        # Random crop and resize
        if random.random() > 0.5:
            w, h = augmented.size
            crop_factor = random.uniform(0.8, 0.95)
            new_w, new_h = int(w * crop_factor), int(h * crop_factor)
            left = random.randint(0, w - new_w)
            top = random.randint(0, h - new_h)
            augmented = augmented.crop((left, top, left + new_w, top + new_h))
            augmented = augmented.resize((w, h), Image.Resampling.LANCZOS)
        
        # Noise
        if random.random() > 0.7:
            arr = np.array(augmented)
            noise = np.random.normal(0, 5, arr.shape).astype(np.uint8)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            augmented = Image.fromarray(arr)
        
        return augmented
    
    def generate_class_samples(self, class_name, num_samples, output_dir):
        """Generate augmented samples for a class"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get category ID
        if class_name not in self.cat_name_to_id:
            print(f"Warning: {class_name} not found in COCO categories")
            return []
        
        cat_id = self.cat_name_to_id[class_name]
        
        # Get images for this category
        image_ids = list(self.images_by_category[cat_id])
        
        if len(image_ids) == 0:
            print(f"Warning: No images found for {class_name}")
            return []
        
        print(f"Found {len(image_ids)} source images for {class_name}")
        
        # Get image filenames
        id_to_filename = {img['id']: img['file_name'] for img in self.coco_data['images']}
        
        generated = []
        for i in tqdm(range(num_samples), desc=f"Generating {class_name}"):
            # Pick random source image
            img_id = random.choice(image_ids)
            img_filename = id_to_filename[img_id]
            img_path = self.coco_dir / 'train2017' / img_filename
            
            try:
                # Load and augment
                source_image = Image.open(img_path).convert('RGB')
                augmented = self.augment_image(source_image, strength='high')
                
                # Save
                output_path = output_dir / f"{class_name}_{i:04d}.png"
                augmented.save(output_path, quality=95)
                
                # Save metadata
                metadata = {
                    'class': class_name,
                    'source_image': img_filename,
                    'source_image_id': img_id,
                    'index': i,
                    'method': 'augmentation',
                    'note': 'Heavy augmentation of COCO images for ablation study'
                }
                
                metadata_path = output_dir / f"{class_name}_{i:04d}.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                generated.append(output_path)
                
            except Exception as e:
                print(f"Error processing {img_filename}: {e}")
                continue
        
        return generated


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic data via augmentation (no Stable Diffusion needed)'
    )
    parser.add_argument('--coco_dir', type=str, default='./data/coco',
                        help='Path to COCO dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for generated images')
    parser.add_argument('--classes', nargs='+', 
                        default=['train', 'cat', 'airplane', 'dog'],
                        help='Classes to generate (rare classes)')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples per class')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  SYNTHETIC DATA GENERATION VIA AUGMENTATION")
    print("="*70)
    print(f"\n⚠️  Note: Using heavy augmentation instead of Stable Diffusion")
    print(f"    This is faster and doesn't require HuggingFace login")
    print(f"    Good enough for ablation study demonstration\n")
    print(f"Classes: {', '.join(args.classes)}")
    print(f"Samples per class: {args.num_samples}")
    print(f"Total images: {len(args.classes) * args.num_samples}")
    print("\n" + "="*70 + "\n")
    
    # Create generator
    generator = SimpleDataAugmentor(coco_dir=args.coco_dir)
    
    # Generate for each class
    output_dir = Path(args.output_dir)
    total_generated = 0
    
    for class_name in args.classes:
        print(f"\n{'='*70}")
        print(f"Generating {args.num_samples} samples for class: {class_name}")
        print(f"{'='*70}\n")
        
        class_output_dir = output_dir / class_name
        
        images = generator.generate_class_samples(
            class_name=class_name,
            num_samples=args.num_samples,
            output_dir=class_output_dir
        )
        
        total_generated += len(images)
        print(f"✅ Generated {len(images)} images for {class_name}")
    
    print(f"\n{'='*70}")
    print(f"✅ Generation complete!")
    print(f"   Total images: {total_generated}")
    print(f"   Saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

