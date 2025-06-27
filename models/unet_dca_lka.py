#!/usr/bin/env python3
"""
U-Net with Dual Channel Attention (DCA) and Large Kernel Attention (LKA)
for 3-class HER2 segmentation (negative, low, high)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from sklearn.model_selection import StratifiedKFold
import argparse
import os
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class LargeKernelAttention(nn.Module):
    """Large Kernel Attention Module"""
    
    def __init__(self, channels, kernel_size=21):
        super(LargeKernelAttention, self).__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # Depthwise convolution with large kernel
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, 
                                padding=padding, groups=channels)
        
        # Pointwise convolution
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Apply large kernel attention
        attn = self.dw_conv(x)
        attn = self.pw_conv(attn)
        attn = self.sigmoid(attn)
        
        return x * attn

class DualChannelAttention(nn.Module):
    """Dual Channel Attention Module"""
    
    def __init__(self, channels, reduction=16):
        super(DualChannelAttention, self).__init__()
        self.channels = channels
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel attention
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        x = x * spatial_att
        
        return x

class ConvBlock(nn.Module):
    """Convolutional block with batch norm and ReLU"""
    
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class UNetDCALKA(pl.LightningModule):
    """U-Net with DCA and LKA for 3-class HER2 segmentation"""
    
    def __init__(self, in_channels=3, num_classes=3, base_channels=64, 
                 learning_rate=1e-3, class_weights=None):
        super(UNetDCALKA, self).__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        # Class weights for loss function
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None
        
        # Encoder
        self.encoder1 = ConvBlock(in_channels, base_channels)
        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.encoder4 = ConvBlock(base_channels * 4, base_channels * 8)
        
        # Bottleneck with attention
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)
        self.bottleneck_lka = LargeKernelAttention(base_channels * 16)
        self.bottleneck_dca = DualChannelAttention(base_channels * 16)
        
        # Decoder
        self.decoder4 = ConvBlock(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.decoder3 = ConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.decoder2 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.decoder1 = ConvBlock(base_channels * 2 + base_channels, base_channels)
        
        # Attention modules for decoder
        self.decoder4_dca = DualChannelAttention(base_channels * 8)
        self.decoder3_dca = DualChannelAttention(base_channels * 4)
        self.decoder2_dca = DualChannelAttention(base_channels * 2)
        self.decoder1_dca = DualChannelAttention(base_channels)
        
        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, num_classes, 1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Metrics tracking
        self.train_iou_scores = []
        self.val_iou_scores = []
        
    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # Bottleneck with attention
        bottleneck = self.bottleneck(self.pool(enc4))
        bottleneck = self.bottleneck_lka(bottleneck)
        bottleneck = self.bottleneck_dca(bottleneck)
        
        # Decoder path
        dec4 = self.upsample(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        dec4 = self.decoder4_dca(dec4)
        
        dec3 = self.upsample(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        dec3 = self.decoder3_dca(dec3)
        
        dec2 = self.upsample(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        dec2 = self.decoder2_dca(dec2)
        
        dec1 = self.upsample(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.decoder1_dca(dec1)
        
        # Final output
        output = self.final_conv(dec1)
        return output
    
    def calculate_iou(self, pred, target, num_classes=3):
        """Calculate IoU for each class"""
        pred = torch.argmax(pred, dim=1)
        ious = []
        
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)
            
            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()
            
            if union == 0:
                iou = 1.0  # Perfect score when both are empty
            else:
                iou = intersection / union
            
            ious.append(iou.item())
        
        return ious
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        
        # Calculate loss
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        loss = criterion(outputs, masks.long())
        
        # Calculate IoU
        ious = self.calculate_iou(outputs, masks)
        
        # Log metrics
        self.log('train_loss', loss)
        self.log('train_iou_neg', ious[0])
        self.log('train_iou_low', ious[1])
        self.log('train_iou_high', ious[2])
        self.log('train_iou_mean', np.mean(ious))
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        
        # Calculate loss
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        loss = criterion(outputs, masks.long())
        
        # Calculate IoU
        ious = self.calculate_iou(outputs, masks)
        
        # Log metrics
        self.log('val_loss', loss)
        self.log('val_iou_neg', ious[0])
        self.log('val_iou_low', ious[1])
        self.log('val_iou_high', ious[2])
        self.log('val_iou_mean', np.mean(ious))
        
        return {'val_loss': loss, 'val_ious': ious}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

class HER2Dataset(Dataset):
    """HER2 dataset for segmentation"""
    
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask (assuming same name but different extension)
        mask_path = self.mask_paths[idx] if idx < len(self.mask_paths) else None
        if mask_path and mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Create dummy mask based on class from folder name
            class_name = self.image_paths[idx].parent.name
            if 'neg' in class_name:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
            elif 'low' in class_name:
                mask = np.ones(image.shape[:2], dtype=np.uint8)
            else:  # high
                mask = np.full(image.shape[:2], 2, dtype=np.uint8)
        
        if self.transform:
            # Apply transforms
            image = self.transform(image)
            mask = torch.from_numpy(mask).long()
        
        return image, mask

def load_class_weights(weights_path):
    """Load class weights from CSV file"""
    if not os.path.exists(weights_path):
        return None
    
    df = pd.read_csv(weights_path)
    weights = df['Weight'].tolist()
    return weights

def create_data_splits(data_path, n_splits=5):
    """Create stratified k-fold splits based on patient IDs"""
    data_path = Path(data_path)
    
    # Collect all image paths and labels
    image_paths = []
    labels = []
    patient_ids = []
    
    class_mapping = {'her2_neg': 0, 'her2_low': 1, 'her2_high': 2}
    
    for class_name, class_label in class_mapping.items():
        class_path = data_path / class_name
        if class_path.exists():
            for img_path in class_path.glob('*.jpg'):
                # Extract patient ID
                patient_id = img_path.stem.split('-')[0] + '-' + img_path.stem.split('-')[1]
                
                image_paths.append(img_path)
                labels.append(class_label)
                patient_ids.append(patient_id)
    
    # Create patient-level splits
    unique_patients = list(set(patient_ids))
    patient_labels = []
    
    for patient in unique_patients:
        # Get majority class for this patient
        patient_class_votes = [labels[i] for i, pid in enumerate(patient_ids) if pid == patient]
        majority_class = max(set(patient_class_votes), key=patient_class_votes.count)
        patient_labels.append(majority_class)
    
    # Create stratified splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = []
    
    for fold, (train_patient_idx, val_patient_idx) in enumerate(skf.split(unique_patients, patient_labels)):
        train_patients = [unique_patients[i] for i in train_patient_idx]
        val_patients = [unique_patients[i] for i in val_patient_idx]
        
        # Convert patient splits to image splits
        train_indices = [i for i, pid in enumerate(patient_ids) if pid in train_patients]
        val_indices = [i for i, pid in enumerate(patient_ids) if pid in val_patients]
        
        splits.append((train_indices, val_indices))
    
    return splits, image_paths, labels

def train_model(args):
    """Train the U-Net DCA LKA model with cross-validation"""
    print("🚂 Training HER2 3-class U-Net DCA LKA")
    print("="*40)
    
    # Load class weights if available
    class_weights = None
    if args.class_weights == 'auto':
        weights_path = './weights/class_weights.csv'
        class_weights = load_class_weights(weights_path)
        if class_weights:
            print(f"Loaded class weights: {class_weights}")
    
    # Create data splits
    print("Creating cross-validation splits...")
    splits, image_paths, labels = create_data_splits(args.data_path, args.cv_folds)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Train each fold
    fold_results = []
    
    for fold, (train_indices, val_indices) in enumerate(splits):
        print(f"\n--- Fold {fold + 1}/{args.cv_folds} ---")
        
        # Create datasets
        train_paths = [image_paths[i] for i in train_indices]
        val_paths = [image_paths[i] for i in val_indices]
        
        train_dataset = HER2Dataset(train_paths, [], train_transform)
        val_dataset = HER2Dataset(val_paths, [], val_transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                 shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=4)
        
        # Initialize model
        model = UNetDCALKA(
            in_channels=3,
            num_classes=3,
            base_channels=64,
            learning_rate=args.learning_rate,
            class_weights=class_weights
        )
        
        # Train model
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            gpus=1 if torch.cuda.is_available() else 0,
            precision=16 if torch.cuda.is_available() else 32,
            enable_checkpointing=True,
            logger=False
        )
        
        trainer.fit(model, train_loader, val_loader)
        
        # Validate model
        val_results = trainer.validate(model, val_loader)
        fold_results.append(val_results[0])
        
        # Print fold results
        val_loss = val_results[0]['val_loss']
        val_iou_neg = val_results[0]['val_iou_neg']
        val_iou_low = val_results[0]['val_iou_low']
        val_iou_high = val_results[0]['val_iou_high']
        
        print(f"Fold {fold + 1} Results:")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  IoU - Neg: {val_iou_neg:.3f} | Low: {val_iou_low:.3f} | High: {val_iou_high:.3f}")
        
        # Save model
        model_path = f'./models/unet_dca_lka_fold_{fold + 1}.ckpt'
        trainer.save_checkpoint(model_path)
        print(f"Saved model: {model_path}")
    
    # Print overall results
    print("\n" + "="*50)
    print("CROSS-VALIDATION RESULTS")
    print("="*50)
    
    avg_loss = np.mean([r['val_loss'] for r in fold_results])
    avg_iou_neg = np.mean([r['val_iou_neg'] for r in fold_results])
    avg_iou_low = np.mean([r['val_iou_low'] for r in fold_results])
    avg_iou_high = np.mean([r['val_iou_high'] for r in fold_results])
    
    print(f"Average Validation Loss: {avg_loss:.4f}")
    print(f"Average IoU - Neg: {avg_iou_neg:.3f} | Low: {avg_iou_low:.3f} | High: {avg_iou_high:.3f}")
    print(f"Mean IoU: {np.mean([avg_iou_neg, avg_iou_low, avg_iou_high]):.3f}")
    
    return fold_results

def main():
    parser = argparse.ArgumentParser(description='Train U-Net DCA LKA for HER2 segmentation')
    parser.add_argument('--data-path', default='./MyLightningProject/data',
                       help='Path to HER2 dataset')
    parser.add_argument('--class-weights', default=None,
                       help='Class weights mode (auto or None)')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Create models directory
    os.makedirs('./models', exist_ok=True)
    
    # Train model
    results = train_model(args)
    
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()