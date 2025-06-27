#!/usr/bin/env python3
"""
WSI Multiple Instance Learning Training Script
Compatible with HER2 Pipeline cross-validation splits
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from MyLightningProject.tasks.lit_mil_classifier import train_mil_model
import argparse

def main():
    """Main training function that matches CV splits from segmentation"""
    parser = argparse.ArgumentParser(description='WSI-MIL Training for HER2 Pipeline')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds (matches segmentation)')
    parser.add_argument('--data-path', default='../MyLightningProject/data',
                       help='Path to HER2 dataset')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size (number of patient bags)')
    parser.add_argument('--max-instances', type=int, default=50,
                       help='Maximum tiles per patient bag')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    print("🧠 WSI-MIL Training for HER2 Pipeline")
    print("="*45)
    print(f"Cross-validation folds: {args.cv_folds}")
    print(f"Data path: {args.data_path}")
    print(f"Max instances per bag: {args.max_instances}")
    
    # Run MIL training with same CV strategy as segmentation
    results = train_mil_model(args)
    
    print("\n✅ WSI-MIL training completed!")
    print("Models saved to: ../models/mil_classifier_fold_*.ckpt")
    
    return results

if __name__ == "__main__":
    main()