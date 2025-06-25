#!/usr/bin/env python3
"""
Multiple Instance Learning (MIL) Classifier for HER2 Weakly-Supervised Learning
Implements attention-based MIL for slide-level HER2 classification
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import os
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionMIL(nn.Module):
    """Attention-based Multiple Instance Learning Network"""
    
    def __init__(self, feature_dim=512, hidden_dim=256, num_classes=3, dropout=0.25):
        super(AttentionMIL, self).__init__()
        
        # Feature extraction backbone (ResNet-like)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Residual blocks
            self._make_residual_block(64, 128, 2),
            self._make_residual_block(128, 256, 2),
            self._make_residual_block(256, 512, 2),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def _make_residual_block(self, in_channels, out_channels, stride=1):
        """Create a residual block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # x shape: (batch_size, num_instances, channels, height, width)
        batch_size, num_instances = x.shape[:2]
        
        # Reshape to process all instances
        x = x.view(-1, *x.shape[2:])  # (batch_size * num_instances, channels, height, width)
        
        # Extract features
        features = self.feature_extractor(x)  # (batch_size * num_instances, feature_dim)
        features = features.view(batch_size, num_instances, -1)  # (batch_size, num_instances, feature_dim)
        
        # Compute attention weights
        attention_weights = self.attention(features)  # (batch_size, num_instances, 1)
        attention_weights = F.softmax(attention_weights, dim=1)  # Normalize across instances
        
        # Weighted aggregation
        bag_representation = torch.sum(attention_weights * features, dim=1)  # (batch_size, feature_dim)
        
        # Classification
        logits = self.classifier(bag_representation)
        
        return logits, attention_weights.squeeze(-1)

class HER2MILDataset(Dataset):
    """Dataset for HER2 MIL training"""
    
    def __init__(self, bag_paths, labels, max_instances=50, transform=None):
        self.bag_paths = bag_paths
        self.labels = labels
        self.max_instances = max_instances
        self.transform = transform
        
    def __len__(self):
        return len(self.bag_paths)
    
    def __getitem__(self, idx):
        bag_path = self.bag_paths[idx]
        label = self.labels[idx]
        
        # Load all images in the bag (patient folder)
        if isinstance(bag_path, str):
            bag_path = Path(bag_path)
        
        image_files = list(bag_path.glob('*.jpg'))[:self.max_instances]
        
        instances = []
        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if self.transform:
                    image = self.transform(image)
                instances.append(image)
        
        # Pad or truncate to max_instances
        while len(instances) < self.max_instances:
            if instances:
                instances.append(instances[-1])  # Repeat last instance
            else:
                # Create dummy instance if no valid images
                dummy = torch.zeros(3, 224, 224)
                instances.append(dummy)
        
        instances = instances[:self.max_instances]
        bag_tensor = torch.stack(instances)
        
        return bag_tensor, torch.tensor(label, dtype=torch.long)

class HER2MILClassifier(pl.LightningModule):
    """Lightning module for HER2 MIL classification"""
    
    def __init__(self, feature_dim=512, hidden_dim=256, num_classes=3, 
                 learning_rate=1e-4, class_weights=None):
        super(HER2MILClassifier, self).__init__()
        self.save_hyperparameters()
        
        self.model = AttentionMIL(feature_dim, hidden_dim, num_classes)
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        # Class weights for imbalanced datasets
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None
        
        # Metrics tracking
        self.train_accuracies = []
        self.val_accuracies = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        bags, labels = batch
        logits, attention_weights = self(bags)
        
        # Calculate loss
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        loss = criterion(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        bags, labels = batch
        logits, attention_weights = self(bags)
        
        # Calculate loss
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        loss = criterion(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        
        return {
            'val_loss': loss,
            'val_acc': acc,
            'preds': preds,
            'labels': labels,
            'attention_weights': attention_weights
        }
    
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

def create_patient_bags(data_path):
    """Create patient-level bags for MIL training"""
    data_path = Path(data_path)
    
    # Group images by patient ID
    patient_bags = {}
    patient_labels = {}
    
    class_mapping = {'her2_neg': 0, 'her2_low': 1, 'her2_high': 2}
    
    for class_name, class_label in class_mapping.items():
        class_path = data_path / class_name
        if class_path.exists():
            for img_path in class_path.glob('*.jpg'):
                # Extract patient ID from filename
                patient_id = img_path.stem.split('-')[0] + '-' + img_path.stem.split('-')[1]
                
                if patient_id not in patient_bags:
                    patient_bags[patient_id] = []
                    patient_labels[patient_id] = class_label
                
                patient_bags[patient_id].append(img_path)
    
    # Convert to bag paths and labels
    bag_paths = []
    labels = []
    
    for patient_id, images in patient_bags.items():
        # Create a temporary directory structure for each patient bag
        bag_dir = Path(f'./temp_bags/{patient_id}')
        bag_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy/link images to bag directory (or just use the image list directly)
        bag_paths.append(images)  # List of image paths for this patient
        labels.append(patient_labels[patient_id])
    
    return bag_paths, labels

class PatientBagDataset(Dataset):
    """Dataset that handles patient bags directly from image lists"""
    
    def __init__(self, patient_image_lists, labels, max_instances=50, transform=None):
        self.patient_image_lists = patient_image_lists
        self.labels = labels
        self.max_instances = max_instances
        self.transform = transform
        
    def __len__(self):
        return len(self.patient_image_lists)
    
    def __getitem__(self, idx):
        image_list = self.patient_image_lists[idx]
        label = self.labels[idx]
        
        # Sample or use all images from this patient
        selected_images = image_list[:self.max_instances]
        
        instances = []
        for img_path in selected_images:
            try:
                image = cv2.imread(str(img_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if self.transform:
                        image = self.transform(image)
                    instances.append(image)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
        
        # Pad instances if needed
        while len(instances) < self.max_instances:
            if instances:
                instances.append(instances[-1])  # Repeat last valid instance
            else:
                # Create dummy instance
                dummy = torch.zeros(3, 224, 224)
                instances.append(dummy)
        
        instances = instances[:self.max_instances]
        bag_tensor = torch.stack(instances)
        
        return bag_tensor, torch.tensor(label, dtype=torch.long)

def train_mil_model(args):
    """Train MIL model with cross-validation"""
    print("🧠 Training HER2 MIL Classifier")
    print("="*40)
    
    # Create patient bags
    print("Creating patient bags...")
    bag_paths, labels = create_patient_bags(args.data_path)
    print(f"Created {len(bag_paths)} patient bags")
    
    # Create cross-validation splits
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    fold_results = []
    
    # Train each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(bag_paths, labels)):
        print(f"\n--- Fold {fold + 1}/{args.cv_folds} ---")
        
        # Create fold datasets
        train_bags = [bag_paths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_bags = [bag_paths[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        train_dataset = PatientBagDataset(train_bags, train_labels, 
                                        max_instances=args.max_instances, 
                                        transform=train_transform)
        val_dataset = PatientBagDataset(val_bags, val_labels, 
                                      max_instances=args.max_instances, 
                                      transform=val_transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                              shuffle=False, num_workers=2)
        
        # Initialize model
        model = HER2MILClassifier(
            feature_dim=512,
            hidden_dim=256,
            num_classes=3,
            learning_rate=args.learning_rate
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
        
        # Validate
        val_results = trainer.validate(model, val_loader)
        fold_results.append(val_results[0])
        
        # Save model
        model_path = f'./models/mil_classifier_fold_{fold + 1}.ckpt'
        trainer.save_checkpoint(model_path)
        print(f"Saved MIL model: {model_path}")
        
        # Print fold results
        val_acc = val_results[0]['val_acc']
        val_loss = val_results[0]['val_loss']
        print(f"Fold {fold + 1} - Val Accuracy: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
    
    # Print overall results
    print("\n" + "="*50)
    print("MIL CROSS-VALIDATION RESULTS")
    print("="*50)
    
    avg_acc = np.mean([r['val_acc'] for r in fold_results])
    avg_loss = np.mean([r['val_loss'] for r in fold_results])
    
    print(f"Average Validation Accuracy: {avg_acc:.4f}")
    print(f"Average Validation Loss: {avg_loss:.4f}")
    
    return fold_results

def main():
    parser = argparse.ArgumentParser(description='Train HER2 MIL Classifier')
    parser.add_argument('--data-path', default='./MyLightningProject/data',
                       help='Path to HER2 dataset')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size (number of bags)')
    parser.add_argument('--max-instances', type=int, default=50,
                       help='Maximum instances per bag')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Create models directory
    os.makedirs('./models', exist_ok=True)
    
    # Train MIL model
    results = train_mil_model(args)
    
    print("\n✅ MIL training complete!")

if __name__ == "__main__":
    main()