#!/usr/bin/env python3
"""
Tumour Region Filter for HER2 Pipeline
Implements Otsu thresholding and ConvNeXt-tiny classifier for tumour region detection
"""

import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import timm

class ConvNeXtTinyClassifier(nn.Module):
    """ConvNeXt-tiny based tumour region classifier"""
    
    def __init__(self, num_classes=2):  # tumour vs non-tumour
        super(ConvNeXtTinyClassifier, self).__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class TumourFilter:
    """Tumour region filter using Otsu + ConvNeXt-tiny"""
    
    def __init__(self, weights_path=None, use_gpu=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.weights_path = weights_path or './weights/convnext_tiny.ckpt'
        
        # Initialize model
        self.model = ConvNeXtTinyClassifier()
        if os.path.exists(self.weights_path):
            try:
                self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
                print(f"Loaded ConvNeXt-tiny weights from {self.weights_path}")
            except Exception as e:
                print(f"Warning: Could not load weights ({e}), using pretrained model")
        else:
            print(f"Warning: Weights file not found at {self.weights_path}, using pretrained model")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def apply_otsu_mask(self, image):
        """Apply Otsu thresholding to create tissue mask"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Apply Otsu thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def classify_tumour_region(self, image_patch):
        """Classify image patch as tumour/non-tumour using ConvNeXt"""
        if isinstance(image_patch, np.ndarray):
            image_patch = Image.fromarray(image_patch)
        
        # Preprocess image
        input_tensor = self.transform(image_patch).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            tumour_prob = probabilities[0][1].item()  # probability of being tumour
            
        return tumour_prob
    
    def filter_image(self, image_path, save_preview=False):
        """Apply complete tumour filtering pipeline"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Step 1: Apply Otsu mask
        otsu_mask = self.apply_otsu_mask(image_rgb)
        
        # Step 2: Classify tumour regions
        tumour_prob = self.classify_tumour_region(image_rgb)
        
        # Step 3: Combine masks (Otsu AND tumour classification)
        tumour_threshold = 0.5
        is_tumour = tumour_prob > tumour_threshold
        
        if is_tumour:
            final_mask = otsu_mask
        else:
            final_mask = np.zeros_like(otsu_mask)
        
        # Apply mask to original image
        filtered_image = image_rgb.copy()
        filtered_image[final_mask == 0] = [255, 255, 255]  # White background for non-tumour
        
        if save_preview:
            self.save_preview(image_rgb, otsu_mask, final_mask, filtered_image, 
                            tumour_prob, image_path)
        
        return filtered_image, final_mask, tumour_prob
    
    def save_preview(self, original, otsu_mask, final_mask, filtered, tumour_prob, image_path):
        """Save preview of filtering results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Otsu mask
        axes[0, 1].imshow(otsu_mask, cmap='gray')
        axes[0, 1].set_title('Otsu Tissue Mask')
        axes[0, 1].axis('off')
        
        # Final tumour mask
        axes[1, 0].imshow(final_mask, cmap='gray')
        axes[1, 0].set_title(f'Tumour Mask (Prob: {tumour_prob:.3f})')
        axes[1, 0].axis('off')
        
        # Filtered result
        axes[1, 1].imshow(filtered)
        axes[1, 1].set_title('Filtered Result')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save preview
        preview_dir = Path('./previews')
        preview_dir.mkdir(exist_ok=True)
        preview_path = preview_dir / f"{Path(image_path).stem}_tumour_filter.png"
        plt.savefig(preview_path, dpi=150, bbox_inches='tight')
        print(f"Saved preview to: {preview_path}")
        plt.show()
    
    def process_dataset(self, data_path, output_path=None):
        """Process entire dataset with tumour filtering"""
        data_path = Path(data_path)
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(exist_ok=True)
        
        class_dirs = ['her2_neg', 'her2_low', 'her2_high']
        results = {}
        
        for class_name in class_dirs:
            class_path = data_path / class_name
            if not class_path.exists():
                continue
                
            print(f"\nProcessing {class_name}...")
            class_results = []
            
            image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
            
            for i, img_file in enumerate(image_files):
                try:
                    _, mask, tumour_prob = self.filter_image(img_file, save_preview=(i < 3))
                    class_results.append({
                        'filename': img_file.name,
                        'tumour_probability': tumour_prob,
                        'is_tumour': tumour_prob > 0.5,
                        'mask_area': np.sum(mask > 0)
                    })
                    
                    if (i + 1) % 20 == 0:
                        print(f"Processed {i + 1}/{len(image_files)} images")
                        
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
            
            results[class_name] = class_results
            print(f"Completed {class_name}: {len(class_results)} images processed")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='HER2 Tumour Region Filter')
    parser.add_argument('--data-path', default='./MyLightningProject/data',
                       help='Path to HER2 dataset')
    parser.add_argument('--weights-path', default='./weights/convnext_tiny.ckpt',
                       help='Path to ConvNeXt-tiny weights')
    parser.add_argument('--preview', action='store_true',
                       help='Generate preview images')
    parser.add_argument('--use-gpu', action='store_true', default=True,
                       help='Use GPU for processing')
    
    args = parser.parse_args()
    
    print("🔍 HER2 Tumour Filter")
    print("="*30)
    
    # Initialize filter
    print("Initializing tumour filter...")
    tumour_filter = TumourFilter(weights_path=args.weights_path, use_gpu=args.use_gpu)
    
    if args.preview:
        print("Generating preview images...")
        # Process a few sample images for preview
        data_path = Path(args.data_path)
        sample_images = []
        
        for class_dir in ['her2_neg', 'her2_low', 'her2_high']:
            class_path = data_path / class_dir
            if class_path.exists():
                images = list(class_path.glob('*.jpg'))[:2]  # Take first 2 images
                sample_images.extend(images)
        
        for img_path in sample_images:
            print(f"Processing preview for: {img_path.name}")
            try:
                tumour_filter.filter_image(img_path, save_preview=True)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    else:
        print("Processing full dataset...")
        results = tumour_filter.process_dataset(args.data_path)
        
        # Print summary
        print("\n" + "="*40)
        print("TUMOUR FILTERING RESULTS")
        print("="*40)
        
        for class_name, class_results in results.items():
            if class_results:
                tumour_count = sum(1 for r in class_results if r['is_tumour'])
                avg_prob = np.mean([r['tumour_probability'] for r in class_results])
                print(f"{class_name}: {tumour_count}/{len(class_results)} tumour regions "
                      f"(avg prob: {avg_prob:.3f})")
    
    print("\n✅ Tumour filtering complete!")

if __name__ == "__main__":
    main()