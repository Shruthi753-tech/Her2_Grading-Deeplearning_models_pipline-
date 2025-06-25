#!/usr/bin/env python3
"""
HER2 Dataset Statistics Generator
Analyzes slide & tile counts per HER2 class and generates visualizations
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import numpy as np

def count_files_by_class(data_path: str):
    """Count files in each HER2 class directory"""
    stats = {}
    class_dirs = ['her2_neg', 'her2_low', 'her2_high']
    
    for class_name in class_dirs:
        class_path = Path(data_path) / class_name
        if class_path.exists():
            # Count image files
            image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png')) + list(class_path.glob('*.tiff'))
            stats[class_name] = len(image_files)
            print(f"Found {len(image_files)} images in {class_name}")
        else:
            stats[class_name] = 0
            print(f"Directory {class_name} not found")
    
    return stats

def extract_patient_ids(data_path: str):
    """Extract patient IDs from filenames for patient-level statistics"""
    patient_stats = {}
    class_dirs = ['her2_neg', 'her2_low', 'her2_high']
    
    for class_name in class_dirs:
        class_path = Path(data_path) / class_name
        if class_path.exists():
            patient_ids = set()
            for img_file in class_path.glob('*.jpg'):
                # Extract patient ID from filename (assumes format like AC000-000-0000492.jpg)
                patient_id = img_file.stem.split('-')[0] + '-' + img_file.stem.split('-')[1]
                patient_ids.add(patient_id)
            patient_stats[class_name] = len(patient_ids)
    
    return patient_stats

def generate_barplot(tile_stats, patient_stats, output_dir: str):
    """Generate barplot visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Tile count plot
    classes = list(tile_stats.keys())
    tile_counts = list(tile_stats.values())
    colors = ['#ff6b6b', '#feca57', '#48ca7a']  # red, yellow, green for neg, low, high
    
    bars1 = ax1.bar(classes, tile_counts, color=colors, alpha=0.8)
    ax1.set_title('Tile Count per HER2 Class', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Tiles', fontsize=12)
    ax1.set_xlabel('HER2 Class', fontsize=12)
    
    # Add value labels on bars
    for bar, count in zip(bars1, tile_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(tile_counts)*0.01,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Patient count plot
    patient_counts = list(patient_stats.values())
    bars2 = ax2.bar(classes, patient_counts, color=colors, alpha=0.8)
    ax2.set_title('Patient Count per HER2 Class', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Patients', fontsize=12)
    ax2.set_xlabel('HER2 Class', fontsize=12)
    
    # Add value labels on bars
    for bar, count in zip(bars2, patient_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(patient_counts)*0.01,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'stats.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved statistics plot to: {output_path}")
    plt.show()

def save_csv_summary(tile_stats, patient_stats, output_dir: str):
    """Save CSV summary of statistics"""
    # Create summary dataframe
    summary_data = []
    for class_name in tile_stats.keys():
        summary_data.append({
            'HER2_Class': class_name,
            'Tile_Count': tile_stats[class_name],
            'Patient_Count': patient_stats[class_name],
            'Tiles_per_Patient': tile_stats[class_name] / max(1, patient_stats[class_name])
        })
    
    df = pd.DataFrame(summary_data)
    
    # Add totals row
    total_row = {
        'HER2_Class': 'TOTAL',
        'Tile_Count': sum(tile_stats.values()),
        'Patient_Count': sum(patient_stats.values()),
        'Tiles_per_Patient': sum(tile_stats.values()) / max(1, sum(patient_stats.values()))
    }
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    
    # Save CSV
    output_path = Path(output_dir) / 'dataset_summary.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved CSV summary to: {output_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)
    
    return df

def calculate_class_weights(tile_stats):
    """Calculate class weights for training"""
    total_samples = sum(tile_stats.values())
    num_classes = len(tile_stats)
    
    class_weights = {}
    for class_name, count in tile_stats.items():
        if count > 0:
            weight = total_samples / (num_classes * count)
            class_weights[class_name] = weight
        else:
            class_weights[class_name] = 1.0
    
    print(f"\nCalculated class weights: {class_weights}")
    
    # Save class weights
    weights_df = pd.DataFrame(list(class_weights.items()), columns=['Class', 'Weight'])
    weights_path = Path('./weights') / 'class_weights.csv'
    weights_path.parent.mkdir(exist_ok=True)
    weights_df.to_csv(weights_path, index=False)
    print(f"Saved class weights to: {weights_path}")
    
    return class_weights

def main():
    parser = argparse.ArgumentParser(description='Generate HER2 dataset statistics')
    parser.add_argument('--data-path', default='./MyLightningProject/data', 
                       help='Path to HER2 dataset')
    parser.add_argument('--output-dir', default='./', 
                       help='Output directory for plots and CSV')
    
    args = parser.parse_args()
    
    print("🩺 HER2 Dataset Statistics Generator")
    print("="*40)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Count tiles and patients
    print("Analyzing dataset...")
    tile_stats = count_files_by_class(args.data_path)
    patient_stats = extract_patient_ids(args.data_path)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_barplot(tile_stats, patient_stats, args.output_dir)
    
    # Save CSV summary
    print("\nSaving summary...")
    df_summary = save_csv_summary(tile_stats, patient_stats, args.output_dir)
    
    # Calculate class weights
    print("\nCalculating class weights...")
    class_weights = calculate_class_weights(tile_stats)
    
    print("\n✅ Dataset analysis complete!")

if __name__ == "__main__":
    main()