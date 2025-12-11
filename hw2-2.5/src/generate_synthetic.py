"""
Synthetic Data Generation using Stable Diffusion
Simplified version - text-to-image without ControlNet for faster generation
"""
import argparse
import json
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import numpy as np
from tqdm import tqdm
import random


class SyntheticDataGenerator:
    def __init__(
        self,
        model_id="stabilityai/stable-diffusion-2-1-base",
        device="cuda"
    ):
        self.device = device
        
        print(f"Loading Stable Diffusion from {model_id}...")
        print("⚠️  Первая загрузка займет время (~5GB модель)")
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(device)
        
        # Faster scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        print("✅ Model loaded successfully!")
    
    def generate_from_pose(
        self,
        pose_image,
        prompt,
        negative_prompt="",
        num_inference_steps=20,
        guidance_scale=7.5,
        seed=None
    ):
        """Generate image from pose"""
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        image = self.pipe(
            prompt=prompt,
            image=pose_image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        return image
    
    def generate_variations(
        self,
        source_image,
        prompt_template,
        num_variations=10,
        output_dir=None
    ):
        """Generate variations of a source image"""
        # Extract pose
        pose_image = self.pose_detector(source_image)
        
        images = []
        for i in range(num_variations):
            # Add variation to prompt
            variations = [
                "different lighting",
                "different angle",
                "different background",
                "different time of day",
                "different weather",
                "indoor scene",
                "outdoor scene",
                "studio lighting",
                "natural lighting",
                "artistic style"
            ]
            
            variation = random.choice(variations)
            prompt = f"{prompt_template}, {variation}"
            
            print(f"Generating variation {i+1}/{num_variations}: {prompt}")
            
            image = self.generate_from_pose(
                pose_image=pose_image,
                prompt=prompt,
                negative_prompt="low quality, blurry, distorted",
                seed=random.randint(0, 1000000)
            )
            
            images.append(image)
            
            if output_dir:
                output_path = Path(output_dir) / f"variation_{i:03d}.png"
                image.save(output_path)
        
        return images
    
    def generate_class_samples(
        self,
        class_name,
        num_samples=50,
        output_dir=None,
        use_controlnet=True
    ):
        """Generate samples for a specific class"""
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Class-specific prompts
        prompt_templates = {
            'person': [
                "a person walking",
                "a person standing",
                "a person sitting",
                "a person running",
                "professional photo of a person"
            ],
            'car': [
                "a car on the street",
                "a parked car",
                "a moving car",
                "luxury car",
                "sports car"
            ],
            'dog': [
                "a dog playing",
                "a dog sitting",
                "a dog running",
                "cute dog",
                "dog portrait"
            ],
            'cat': [
                "a cat sleeping",
                "a cat playing",
                "a cat sitting",
                "cute cat",
                "cat portrait"
            ],
            'chair': [
                "a modern chair",
                "an office chair",
                "a dining chair",
                "a comfortable chair",
                "designer chair"
            ],
            'bottle': [
                "a water bottle",
                "a glass bottle",
                "a plastic bottle",
                "a bottle on table",
                "product photo of bottle"
            ],
            'bicycle': [
                "a bicycle on street",
                "a parked bicycle",
                "a mountain bike",
                "a road bike",
                "vintage bicycle"
            ],
            'airplane': [
                "an airplane in sky",
                "an airplane landing",
                "a commercial airplane",
                "a private jet",
                "airplane at airport"
            ],
            'bus': [
                "a bus on street",
                "a city bus",
                "a school bus",
                "a tour bus",
                "double decker bus"
            ],
            'train': [
                "a train at station",
                "a moving train",
                "a passenger train",
                "a freight train",
                "high speed train"
            ]
        }
        
        templates = prompt_templates.get(class_name, [f"a {class_name}"])
        
        images = []
        for i in tqdm(range(num_samples), desc=f"Generating {class_name}"):
            prompt = random.choice(templates)
            
            # Add quality modifiers
            quality_modifiers = [
                "high quality, detailed",
                "professional photo",
                "8k resolution",
                "photorealistic",
                "studio quality"
            ]
            prompt = f"{prompt}, {random.choice(quality_modifiers)}"
            
            if use_controlnet:
                # For controlnet, we would need a pose/edge map
                # For simplicity, using text-to-image
                pass
            
            # Generate using standard text-to-image
            seed = random.randint(0, 1000000)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            image = self.pipe(
                prompt=prompt,
                negative_prompt="low quality, blurry, distorted, deformed, text, watermark",
                num_inference_steps=25,  # Faster
                guidance_scale=7.5,
                width=512,
                height=512,
                generator=generator
            ).images[0]
            
            images.append(image)
            
            if output_dir:
                output_path = output_dir / f"{class_name}_{i:04d}.png"
                image.save(output_path)
                
                # Save metadata
                metadata = {
                    'prompt': prompt,
                    'class': class_name,
                    'index': i
                }
                
                metadata_path = output_dir / f"{class_name}_{i:04d}.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        return images


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data using Stable Diffusion')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for generated images')
    parser.add_argument('--classes', nargs='+', 
                        default=['dog', 'cat', 'train', 'airplane'],
                        help='Classes to generate (rare classes)')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples per class')
    parser.add_argument('--model_id', type=str, 
                        default='stabilityai/stable-diffusion-2-1-base',
                        help='Stable Diffusion model ID')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for generation (keep at 1 for stability)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  SYNTHETIC DATA GENERATION")
    print("="*60)
    print(f"\nClasses: {', '.join(args.classes)}")
    print(f"Samples per class: {args.num_samples}")
    print(f"Total images: {len(args.classes) * args.num_samples}")
    print(f"Model: {args.model_id}")
    print("\n" + "="*60 + "\n")
    
    # Create generator
    generator = SyntheticDataGenerator(
        model_id=args.model_id,
        device=args.device
    )
    
    # Generate for each class
    output_dir = Path(args.output_dir)
    
    for class_name in args.classes:
        print(f"\n{'='*50}")
        print(f"Generating {args.num_samples} samples for class: {class_name}")
        print(f"{'='*50}\n")
        
        class_output_dir = output_dir / class_name
        
        generator.generate_class_samples(
            class_name=class_name,
            num_samples=args.num_samples,
            output_dir=class_output_dir
        )
    
    print(f"\n{'='*50}")
    print(f"Generation complete! Images saved to {output_dir}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    main()

