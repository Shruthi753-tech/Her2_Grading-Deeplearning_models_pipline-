#!/usr/bin/env python3
"""
MIL Bag Creator and Preview Generator
Creates patient-level bags and generates preview data for VS Code extension
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MILBagCreator:
    """Creates and manages MIL bags for HER2 classification"""
    
    def __init__(self, max_instances=1000):
        self.max_instances = max_instances
        self.her2_mapping = {
            '0': 'Negative',
            '1+': 'Low',
            '2+': 'Low', 
            '3+': 'High'
        }
        self.clinical_colors = {
            'Negative': '#2E8B57',  # Sea Green
            'Low': '#FF8C00',       # Dark Orange  
            'High': '#DC143C'       # Crimson
        }
        
    def create_bags_from_tiles(self, data_path: Path, output_dir: Path) -> Path:
        """Create patient bags from tile directory structure"""
        logger.info("🎒 Creating patient bags from tiles...")
        
        # Find all tile files
        tile_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            tile_files.extend(data_path.rglob(ext))
        
        if not tile_files:
            logger.error(f"No tile files found in {data_path}")
            return None
        
        # Group tiles by patient
        patient_groups = self._group_tiles_by_patient(tile_files)
        
        # Create bags directory
        bags_dir = output_dir / 'bags'
        bags_dir.mkdir(exist_ok=True)
        
        # Create manifest directory
        manifests_dir = output_dir / 'manifests'
        manifests_dir.mkdir(exist_ok=True)
        
        bag_manifest = []
        
        for patient_id, tile_info in patient_groups.items():
            # Sample tiles if too many
            tiles = tile_info['tiles'][:self.max_instances]
            
            # Create bag metadata
            bag_data = {
                'patient_id': patient_id,
                'tile_paths': [str(t.relative_to(output_dir)) for t in tiles],
                'tile_count': len(tiles),
                'her2_score': tile_info.get('her2_score', 'Unknown'),
                'clinical_category': tile_info.get('clinical_category', 'Unknown'),
                'label_numeric': self._get_numeric_label(tile_info.get('clinical_category', 'Unknown'))
            }
            
            # Save bag file
            bag_file = bags_dir / f"{patient_id}_bag.json"
            with open(bag_file, 'w') as f:
                json.dump(bag_data, f, indent=2)
            
            # Add to manifest
            bag_manifest.append({
                'patient_id': patient_id,
                'bag_path': str(bag_file.relative_to(output_dir)),
                'tile_count': len(tiles),
                'her2_score': bag_data['her2_score'],
                'clinical_category': bag_data['clinical_category'],
                'label_numeric': bag_data['label_numeric']
            })
        
        # Save bag manifest
        manifest_df = pd.DataFrame(bag_manifest)
        manifest_path = manifests_dir / 'bag_manifest.csv'
        manifest_df.to_csv(manifest_path, index=False)
        
        logger.info(f"Created {len(patient_groups)} patient bags")
        logger.info(f"Bag manifest saved to: {manifest_path}")
        
        return manifest_path
    
    def _group_tiles_by_patient(self, tile_files: List[Path]) -> Dict:
        """Group tile files by patient ID"""
        patient_groups = {}
        
        for tile_file in tile_files:
            # Extract patient ID from filename or path
            patient_id = self._extract_patient_id(tile_file)
            
            # Infer HER2 category from path structure
            her2_category = self._infer_her2_category(tile_file)
            
            if patient_id not in patient_groups:
                patient_groups[patient_id] = {
                    'tiles': [],
                    'her2_score': 'Unknown',
                    'clinical_category': her2_category
                }
            
            patient_groups[patient_id]['tiles'].append(tile_file)
        
        return patient_groups
    
    def _extract_patient_id(self, tile_file: Path) -> str:
        """Extract patient ID from tile filename"""
        filename = tile_file.stem
        
        # Pattern: AC000-000-0000111 -> AC000-000
        parts = filename.split('-')
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
        
        # Fallback to first 10 characters
        return filename[:10]
    
    def _infer_her2_category(self, tile_file: Path) -> str:
        """Infer HER2 category from file path"""
        path_str = str(tile_file).lower()
        
        if 'her2_neg' in path_str or 'negative' in path_str:
            return 'Negative'
        elif 'her2_low' in path_str or 'low' in path_str:
            return 'Low'
        elif 'her2_high' in path_str or 'high' in path_str:
            return 'High'
        
        return 'Unknown'
    
    def _get_numeric_label(self, clinical_category: str) -> int:
        """Convert clinical category to numeric label"""
        mapping = {'Negative': 0, 'Low': 1, 'High': 2, 'Unknown': -1}
        return mapping.get(clinical_category, -1)
    
    def generate_preview_data(self, manifest_path: Path, num_preview_tiles=25) -> List[Dict]:
        """Generate preview data for VS Code webview"""
        logger.info("🎞 Generating bag preview data...")
        
        if not manifest_path.exists():
            logger.error(f"Manifest file not found: {manifest_path}")
            return []
        
        manifest_df = pd.read_csv(manifest_path)
        bags_dir = manifest_path.parent.parent / 'bags'
        
        preview_data = []
        
        for _, row in manifest_df.iterrows():
            patient_id = row['patient_id']
            bag_file = bags_dir / f"{patient_id}_bag.json"
            
            if not bag_file.exists():
                continue
            
            with open(bag_file, 'r') as f:
                bag_data = json.load(f)
            
            # Generate tile previews
            tile_previews = self._generate_tile_previews(
                bag_data['tile_paths'][:num_preview_tiles],
                manifest_path.parent.parent
            )
            
            bag_preview = {
                'patient_id': patient_id,
                'tile_count': bag_data['tile_count'],
                'her2_score': bag_data['her2_score'],
                'clinical_category': bag_data['clinical_category'],
                'label_numeric': bag_data['label_numeric'],
                'color': self.clinical_colors.get(bag_data['clinical_category'], '#808080'),
                'tile_previews': tile_previews,
                'attention_weights': self._simulate_attention_weights(len(tile_previews))
            }
            
            preview_data.append(bag_preview)
        
        logger.info(f"Generated preview data for {len(preview_data)} bags")
        return preview_data
    
    def _generate_tile_previews(self, tile_paths: List[str], base_path: Path) -> List[Dict]:
        """Generate thumbnail previews for tiles"""
        previews = []
        
        for i, tile_path in enumerate(tile_paths):
            full_path = base_path / tile_path
            
            if not full_path.exists():
                # Create placeholder preview
                preview = {
                    'index': i,
                    'path': tile_path,
                    'thumbnail': None,
                    'exists': False,
                    'size': (0, 0)
                }
            else:
                try:
                    # Load image and get basic info
                    image = cv2.imread(str(full_path))
                    if image is not None:
                        height, width = image.shape[:2]
                        preview = {
                            'index': i,
                            'path': tile_path,
                            'thumbnail': None,  # Could add base64 encoding here
                            'exists': True,
                            'size': (width, height),
                            'original_size': (width, height)
                        }
                    else:
                        preview = {
                            'index': i,
                            'path': tile_path,
                            'thumbnail': None,
                            'exists': False,
                            'size': (0, 0)
                        }
                except Exception as e:
                    logger.warning(f"Failed to process tile {full_path}: {e}")
                    preview = {
                        'index': i,
                        'path': tile_path,
                        'thumbnail': None,
                        'exists': False,
                        'size': (0, 0)
                    }
            
            previews.append(preview)
        
        return previews
    
    def _simulate_attention_weights(self, num_tiles: int) -> List[float]:
        """Simulate attention weights for visualization"""
        # Generate realistic attention distribution
        # Higher attention for some tiles, lower for others
        weights = np.random.beta(2, 5, num_tiles)  # Beta distribution for realistic attention
        weights = weights / weights.sum()  # Normalize to sum to 1
        return weights.tolist()

def main():
    parser = argparse.ArgumentParser(description='MIL Bag Creator and Preview Generator')
    parser.add_argument('--data-path', help='Path to tile data directory')
    parser.add_argument('--manifest-path', help='Path to existing bag manifest')
    parser.add_argument('--output-dir', default='./output', help='Output directory')
    parser.add_argument('--max-instances', type=int, default=1000, 
                       help='Maximum instances per bag')
    parser.add_argument('--preview', action='store_true', 
                       help='Generate preview data for VS Code')
    parser.add_argument('--output-json', action='store_true',
                       help='Output preview data as JSON')
    
    args = parser.parse_args()
    
    # Initialize bag creator
    creator = MILBagCreator(max_instances=args.max_instances)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create bags if data path provided
    manifest_path = None
    if args.data_path:
        manifest_path = creator.create_bags_from_tiles(Path(args.data_path), output_dir)
    elif args.manifest_path:
        manifest_path = Path(args.manifest_path)
    
    # Generate preview if requested
    if args.preview and manifest_path:
        preview_data = creator.generate_preview_data(manifest_path)
        
        if args.output_json:
            print(json.dumps(preview_data, indent=2))
        else:
            # Save preview data
            preview_file = output_dir / 'bag_preview_data.json'
            with open(preview_file, 'w') as f:
                json.dump(preview_data, f, indent=2)
            logger.info(f"Preview data saved to: {preview_file}")
    
    logger.info("✅ Bag creation and preview generation completed!")

if __name__ == "__main__":
    main()