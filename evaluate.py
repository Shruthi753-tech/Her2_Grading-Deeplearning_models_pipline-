#!/usr/bin/env python3
"""
HER2 Pipeline Evaluation Script
Comprehensive evaluation with per-class IoU, Dice, Confusion Matrix, ROC curves
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import argparse
import os
from datetime import datetime
import json

# Import our model
from models.unet_dca_lka import UNetDCALKA, HER2Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class HER2Evaluator:
    """Comprehensive evaluator for HER2 segmentation models"""
    
    def __init__(self, model_paths, data_path, device='cuda'):
        self.model_paths = model_paths if isinstance(model_paths, list) else [model_paths]
        self.data_path = Path(data_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.class_names = ['HER2-', 'HER2+', 'HER2++']
        self.num_classes = 3
        
        # Create timestamp for reports
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = Path(f'./reports/{self.timestamp}')
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📊 HER2 Evaluation Report: {self.timestamp}")
        print(f"Report directory: {self.report_dir}")
    
    def load_models(self):
        """Load all trained models"""
        models = []
        for model_path in self.model_paths:
            if os.path.exists(model_path):
                model = UNetDCALKA.load_from_checkpoint(model_path)
                model.to(self.device)
                model.eval()
                models.append(model)
                print(f"Loaded model: {model_path}")
            else:
                print(f"Warning: Model not found: {model_path}")
        
        return models
    
    def prepare_test_data(self):
        """Prepare test dataset"""
        # Collect test images from all classes
        image_paths = []
        labels = []
        
        class_mapping = {'her2_neg': 0, 'her2_low': 1, 'her2_high': 2}
        
        for class_name, class_label in class_mapping.items():
            class_path = self.data_path / class_name
            if class_path.exists():
                class_images = list(class_path.glob('*.jpg'))[:50]  # Limit for evaluation
                image_paths.extend(class_images)
                labels.extend([class_label] * len(class_images))
        
        # Create test dataset and dataloader
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = HER2Dataset(image_paths, [], test_transform)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
        
        return test_loader, labels
    
    def calculate_metrics(self, predictions, targets):
        """Calculate comprehensive metrics"""
        # Convert to numpy
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        # Flatten arrays
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        metrics = {}
        
        # Per-class IoU and Dice
        for cls in range(self.num_classes):
            pred_cls = (pred_flat == cls)
            target_cls = (target_flat == cls)
            
            # IoU
            intersection = np.sum(pred_cls & target_cls)
            union = np.sum(pred_cls | target_cls)
            iou = intersection / union if union > 0 else 1.0
            
            # Dice coefficient
            dice = (2 * intersection) / (np.sum(pred_cls) + np.sum(target_cls)) if (np.sum(pred_cls) + np.sum(target_cls)) > 0 else 1.0
            
            metrics[f'IoU_{self.class_names[cls]}'] = iou
            metrics[f'Dice_{self.class_names[cls]}'] = dice
        
        # Mean IoU and Dice
        metrics['mIoU'] = np.mean([metrics[f'IoU_{name}'] for name in self.class_names])
        metrics['mDice'] = np.mean([metrics[f'Dice_{name}'] for name in self.class_names])
        
        # Overall accuracy
        metrics['Accuracy'] = np.sum(pred_flat == target_flat) / len(pred_flat)
        
        return metrics
    
    def evaluate_models(self, models, test_loader):
        """Evaluate all models and ensemble"""
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Ensemble predictions from all models
                ensemble_logits = torch.zeros((images.size(0), self.num_classes, 
                                             images.size(2), images.size(3))).to(self.device)
                
                for model in models:
                    logits = model(images)
                    ensemble_logits += F.softmax(logits, dim=1)
                
                ensemble_logits /= len(models)  # Average ensemble
                
                # Get predictions
                predictions = torch.argmax(ensemble_logits, dim=1)
                
                all_predictions.append(predictions)
                all_targets.append(targets)
                all_probabilities.append(ensemble_logits)
                
                if batch_idx % 10 == 0:
                    print(f"Evaluated batch {batch_idx + 1}/{len(test_loader)}")
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_probabilities = torch.cat(all_probabilities, dim=0)
        
        return all_predictions, all_targets, all_probabilities
    
    def plot_confusion_matrix(self, predictions, targets):
        """Generate confusion matrix plot"""
        pred_flat = predictions.cpu().numpy().flatten()
        target_flat = targets.cpu().numpy().flatten()
        
        cm = confusion_matrix(target_flat, pred_flat)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('HER2 Classification Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        
        # Add accuracy annotations
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if cm[i, j] > 0:
                    accuracy = cm[i, j] / np.sum(cm[i, :]) * 100
                    plt.text(j + 0.5, i + 0.7, f'{accuracy:.1f}%', 
                            ha='center', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        
        # Save plot
        confusion_path = self.report_dir / 'confusion_matrix.png'
        plt.savefig(confusion_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix: {confusion_path}")
        plt.show()
        
        return cm
    
    def plot_roc_curves(self, probabilities, targets):
        """Generate ROC curves for each class"""
        # Convert targets to one-hot encoding
        targets_np = targets.cpu().numpy().flatten()
        probs_np = probabilities.cpu().numpy()
        
        # Reshape probabilities to (n_samples, n_classes)
        probs_reshaped = probs_np.reshape(-1, self.num_classes)
        
        # Binarize targets for multiclass ROC
        targets_binary = label_binarize(targets_np, classes=range(self.num_classes))
        
        plt.figure(figsize=(12, 8))
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(self.class_names):
            if targets_binary.shape[1] > 1:  # Multiclass case
                fpr, tpr, _ = roc_curve(targets_binary[:, i], probs_reshaped[:, i])
            else:  # Binary case
                fpr, tpr, _ = roc_curve(targets_binary, probs_reshaped[:, 1])
            
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal reference line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('HER2 Classification ROC Curves', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        roc_path = self.report_dir / 'roc_curves.png'
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curves: {roc_path}")
        plt.show()
    
    def plot_class_metrics(self, metrics):
        """Plot per-class IoU and Dice metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # IoU plot
        iou_scores = [metrics[f'IoU_{name}'] for name in self.class_names]
        colors = ['#ff6b6b', '#feca57', '#48ca7a']
        
        bars1 = ax1.bar(self.class_names, iou_scores, color=colors, alpha=0.8)
        ax1.set_title('Per-Class IoU Scores', fontsize=14, fontweight='bold')
        ax1.set_ylabel('IoU Score', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars1, iou_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Dice plot
        dice_scores = [metrics[f'Dice_{name}'] for name in self.class_names]
        
        bars2 = ax2.bar(self.class_names, dice_scores, color=colors, alpha=0.8)
        ax2.set_title('Per-Class Dice Scores', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Dice Score', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars2, dice_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        metrics_path = self.report_dir / 'class_metrics.png'
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        print(f"Saved class metrics: {metrics_path}")
        plt.show()
    
    def save_detailed_report(self, metrics, confusion_matrix, predictions, targets):
        """Save comprehensive evaluation report"""
        # Create detailed report
        report = {
            'timestamp': self.timestamp,
            'evaluation_summary': {
                'mean_iou': metrics['mIoU'],
                'mean_dice': metrics['mDice'],
                'accuracy': metrics['Accuracy']
            },
            'per_class_metrics': {
                class_name: {
                    'iou': metrics[f'IoU_{class_name}'],
                    'dice': metrics[f'Dice_{class_name}']
                }
                for class_name in self.class_names
            },
            'confusion_matrix': confusion_matrix.tolist(),
            'model_paths': self.model_paths
        }
        
        # Save JSON report
        json_path = self.report_dir / 'evaluation_report.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save detailed CSV
        pred_flat = predictions.cpu().numpy().flatten()
        target_flat = targets.cpu().numpy().flatten()
        
        detailed_df = pd.DataFrame({
            'True_Class': [self.class_names[t] for t in target_flat],
            'Predicted_Class': [self.class_names[p] for p in pred_flat],
            'Correct': pred_flat == target_flat
        })
        
        csv_path = self.report_dir / 'detailed_predictions.csv'
        detailed_df.to_csv(csv_path, index=False)
        
        # Print summary report
        print("\n" + "="*60)
        print("EVALUATION REPORT SUMMARY")
        print("="*60)
        print(f"Mean IoU: {metrics['mIoU']:.4f}")
        print(f"Mean Dice: {metrics['mDice']:.4f}")
        print(f"Overall Accuracy: {metrics['Accuracy']:.4f}")
        print("\nPer-Class Results:")
        for class_name in self.class_names:
            iou = metrics[f'IoU_{class_name}']
            dice = metrics[f'Dice_{class_name}']
            print(f"  {class_name}: IoU={iou:.4f}, Dice={dice:.4f}")
        
        print(f"\nDetailed reports saved to: {self.report_dir}")
        print("="*60)
        
        return report
    
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        print("Loading models...")
        models = self.load_models()
        
        if not models:
            print("❌ No models loaded. Exiting evaluation.")
            return
        
        print("Preparing test data...")
        test_loader, labels = self.prepare_test_data()
        
        print("Running evaluation...")
        predictions, targets, probabilities = self.evaluate_models(models, test_loader)
        
        print("Calculating metrics...")
        metrics = self.calculate_metrics(predictions, targets)
        
        print("Generating visualizations...")
        confusion_mat = self.plot_confusion_matrix(predictions, targets)
        self.plot_roc_curves(probabilities, targets)
        self.plot_class_metrics(metrics)
        
        print("Saving detailed report...")
        report = self.save_detailed_report(metrics, confusion_mat, predictions, targets)
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Evaluate HER2 segmentation models')
    parser.add_argument('--model-paths', nargs='+', 
                       default=['./models/unet_dca_lka_fold_1.ckpt',
                               './models/unet_dca_lka_fold_2.ckpt',
                               './models/unet_dca_lka_fold_3.ckpt',
                               './models/unet_dca_lka_fold_4.ckpt',
                               './models/unet_dca_lka_fold_5.ckpt'],
                       help='Paths to trained model checkpoints')
    parser.add_argument('--data-path', default='./MyLightningProject/data',
                       help='Path to test dataset')
    parser.add_argument('--device', default='cuda',
                       help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = HER2Evaluator(args.model_paths, args.data_path, args.device)
    report = evaluator.run_evaluation()
    
    print("\n✅ Evaluation complete!")
    return report

if __name__ == "__main__":
    main()