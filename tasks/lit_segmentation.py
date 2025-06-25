#!/usr/bin/env python3
"""
Lightning Segmentation Task for HER2 Pipeline
Integrates with the main U-Net DCA LKA model
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import argparse
import os

# Import our main model
from models.unet_dca_lka import UNetDCALKA, HER2Dataset, create_data_splits, load_class_weights

class HER2SegmentationTask(pl.LightningModule):
    """Lightning task wrapper for HER2 segmentation"""
    
    def __init__(self, model_config=None, data_config=None):
        super(HER2SegmentationTask, self).__init__()
        
        # Default configurations
        self.model_config = model_config or {
            'in_channels': 3,
            'num_classes': 3,
            'base_channels': 64,
            'learning_rate': 1e-3
        }
        
        self.data_config = data_config or {
            'data_path': './MyLightningProject/data',
            'batch_size': 8,
            'num_workers': 4,
            'cv_folds': 5
        }
        
        # Initialize model
        self.model = UNetDCALKA(**self.model_config)
        
        # Setup data transforms
        self.setup_transforms()
        
    def setup_transforms(self):
        """Setup data transforms"""
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def prepare_data(self):
        """Prepare data splits"""
        self.splits, self.image_paths, self.labels = create_data_splits(
            self.data_config['data_path'], 
            self.data_config['cv_folds']
        )
        print(f"Prepared {len(self.splits)} cross-validation splits")
        print(f"Total images: {len(self.image_paths)}")
    
    def setup_fold(self, fold_idx):
        """Setup specific fold for training"""
        if not hasattr(self, 'splits'):
            self.prepare_data()
        
        train_indices, val_indices = self.splits[fold_idx]
        
        # Create datasets for this fold
        train_paths = [self.image_paths[i] for i in train_indices]
        val_paths = [self.image_paths[i] for i in val_indices]
        
        self.train_dataset = HER2Dataset(train_paths, [], self.train_transform)
        self.val_dataset = HER2Dataset(val_paths, [], self.val_transform)
        
        print(f"Fold {fold_idx + 1}: Train={len(train_paths)}, Val={len(val_paths)}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=True,
            num_workers=self.data_config['num_workers'],
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=False,
            num_workers=self.data_config['num_workers'],
            pin_memory=True
        )
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.model.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        return self.model.configure_optimizers()
    
    def run_cross_validation(self, max_epochs=50):
        """Run complete cross-validation training"""
        print("🚂 Starting Cross-Validation Training")
        print("="*50)
        
        fold_results = []
        
        for fold in range(self.data_config['cv_folds']):
            print(f"\n--- Training Fold {fold + 1}/{self.data_config['cv_folds']} ---")
            
            # Setup fold data
            self.setup_fold(fold)
            
            # Create trainer
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                gpus=1 if torch.cuda.is_available() else 0,
                precision=16 if torch.cuda.is_available() else 32,
                enable_checkpointing=True,
                logger=pl.loggers.TensorBoardLogger('./logs', name=f'fold_{fold + 1}'),
                callbacks=[
                    pl.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=True),
                    pl.callbacks.ModelCheckpoint(
                        dirpath='./models',
                        filename=f'segmentation_fold_{fold + 1}_{{val_loss:.4f}}',
                        monitor='val_loss',
                        save_top_k=1
                    )
                ]
            )
            
            # Train
            trainer.fit(self)
            
            # Validate
            val_results = trainer.validate(self)
            fold_results.append(val_results[0])
            
            # Print fold summary
            val_loss = val_results[0]['val_loss']
            val_iou_mean = val_results[0]['val_iou_mean']
            print(f"Fold {fold + 1} completed - Val Loss: {val_loss:.4f}, Mean IoU: {val_iou_mean:.4f}")
        
        # Print overall results
        print("\n" + "="*60)
        print("CROSS-VALIDATION SUMMARY")
        print("="*60)
        
        avg_loss = torch.mean(torch.tensor([r['val_loss'] for r in fold_results]))
        avg_iou = torch.mean(torch.tensor([r['val_iou_mean'] for r in fold_results]))
        
        print(f"Average Validation Loss: {avg_loss:.4f}")
        print(f"Average Mean IoU: {avg_iou:.4f}")
        
        # Per-class averages
        for class_name in ['neg', 'low', 'high']:
            avg_class_iou = torch.mean(torch.tensor([r[f'val_iou_{class_name}'] for r in fold_results]))
            print(f"Average IoU {class_name}: {avg_class_iou:.4f}")
        
        return fold_results

class HER2TaskRunner:
    """Runner for different HER2 pipeline tasks"""
    
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.load_config()
    
    def load_config(self):
        """Load configuration from file or use defaults"""
        self.config = {
            'model': {
                'in_channels': 3,
                'num_classes': 3,
                'base_channels': 64,
                'learning_rate': 1e-3
            },
            'data': {
                'data_path': './MyLightningProject/data',
                'batch_size': 8,
                'num_workers': 4,
                'cv_folds': 5
            },
            'training': {
                'max_epochs': 50,
                'use_class_weights': True
            }
        }
    
    def run_segmentation_task(self):
        """Run segmentation task"""
        print("🎯 Running HER2 Segmentation Task")
        
        # Load class weights if available
        if self.config['training']['use_class_weights']:
            class_weights = load_class_weights('./weights/class_weights.csv')
            if class_weights:
                self.config['model']['class_weights'] = class_weights
                print(f"Using class weights: {class_weights}")
        
        # Create task
        task = HER2SegmentationTask(
            model_config=self.config['model'],
            data_config=self.config['data']
        )
        
        # Run cross-validation
        results = task.run_cross_validation(
            max_epochs=self.config['training']['max_epochs']
        )
        
        return results
    
    def run_mil_task(self):
        """Run MIL task"""
        print("🧠 Running HER2 MIL Task")
        
        # Import and run MIL training
        from lit_mil_classifier import train_mil_model
        
        # Create args object
        class Args:
            def __init__(self):
                self.data_path = self.config['data']['data_path']
                self.cv_folds = self.config['data']['cv_folds']
                self.epochs = 30
                self.batch_size = 4
                self.max_instances = 50
                self.learning_rate = 1e-4
        
        args = Args()
        results = train_mil_model(args)
        
        return results

def main():
    parser = argparse.ArgumentParser(description='HER2 Segmentation Task Runner')
    parser.add_argument('--task', choices=['segmentation', 'mil'], default='segmentation',
                       help='Task to run')
    parser.add_argument('--config', default=None,
                       help='Path to configuration file')
    parser.add_argument('--data-path', default='./MyLightningProject/data',
                       help='Path to data')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    # Create and run task
    runner = HER2TaskRunner(args.config)
    
    if args.task == 'segmentation':
        runner.config['data']['data_path'] = args.data_path
        runner.config['training']['max_epochs'] = args.epochs
        runner.config['data']['batch_size'] = args.batch_size
        
        results = runner.run_segmentation_task()
    elif args.task == 'mil':
        results = runner.run_mil_task()
    
    print(f"\n✅ {args.task.capitalize()} task completed!")
    return results

if __name__ == "__main__":
    main()