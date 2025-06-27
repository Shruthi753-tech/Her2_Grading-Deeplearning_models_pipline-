#!/usr/bin/env python3
"""
HER2 Preprocessing Pipeline with Slideflow Integration
Supports both segmentation (256px) and MIL (224px) workflows
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json

try:
    import slideflow as sf
    import cv2
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("Install with: pip install slideflow opencv-python tqdm")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HER2Preprocessor:
    """HER2-specific preprocessing with clinical awareness"""
    
    def __init__(self, tile_size=224, overlap=0, magnification=0.5):
        self.tile_size = tile_size
        self.overlap = overlap
        self.magnification = magnification
        self.her2_mapping = {
            '0': 'Negative',
            '1+': 'Low', 
            '2+': 'Low',  # Requires ISH confirmation per ASCO 2018
            '3+': 'High'
        }
        
    def extract_tiles_slideflow(self, slide_path: Path, output_dir: Path, 
                               patient_id: str = None) -> List[Path]:
        """Extract tiles using Slideflow with clinical optimizations"""
        try:
            # Initialize slideflow project
            project_dir = output_dir.parent / 'slideflow_project'
            project_dir.mkdir(exist_ok=True)
            
            # Create or load slideflow project
            try:
                P = sf.Project(project_dir)
            except:
                P = sf.create_project(
                    root=str(project_dir),
                    name='her2_pipeline',
                    annotations=None
                )
            
            # Configure tile extraction parameters
            tile_px = self.tile_size
            tile_um = 256  # Microns - clinical standard
            
            logger.info(f"Extracting tiles from {slide_path.name}")
            logger.info(f"Parameters: {tile_px}px, {tile_um}μm, {self.magnification}mpp")
            
            # Extract tiles with Otsu tissue masking
            dataset = P.dataset(
                sources=[str(slide_path.parent)],
                filters={'slide': [slide_path.name]}
            )
            
            # Configure extraction
            extraction_kwargs = {
                'tile_px': tile_px,
                'tile_um': tile_um,
                'enable_downsample': True,
                'quality_filter': 'otsu',  # ASCO 2018 tissue detection
                'filter_threshold': 0.6,   # Remove background tiles
                'img_format': 'jpg'
            }
            
            # Extract to output directory
            tfrecords_dir = output_dir / 'tfrecords'
            tiles_dir = output_dir / 'tiles'
            tiles_dir.mkdir(exist_ok=True)
            
            # Extract tiles
            dataset.extract_tiles(
                tile_px=tile_px,
                tile_um=tile_um,
                save_tfrecords=True,
                save_tiles=True,
                tfrecord_dir=str(tfrecords_dir),
                tile_dir=str(tiles_dir),
                **extraction_kwargs
            )
            
            # Get extracted tile paths
            tile_paths = list(tiles_dir.glob(f"{slide_path.stem}*.jpg"))
            logger.info(f"Extracted {len(tile_paths)} tiles")
            
            return tile_paths
            
        except Exception as e:
            logger.error(f"Slideflow extraction failed for {slide_path}: {e}")
            # Fallback to basic extraction
            return self._basic_tile_extraction(slide_path, output_dir)
    
    def _basic_tile_extraction(self, slide_path: Path, output_dir: Path) -> List[Path]:
        """Fallback tile extraction without slideflow"""
        logger.warning("Using basic tile extraction fallback")
        
        try:
            import openslide
        except ImportError:
            logger.error("OpenSlide not available for fallback extraction")
            return []
        
        try:
            # Open slide
            slide = openslide.OpenSlide(str(slide_path))
            
            # Get dimensions at desired magnification
            level = 0  # Use highest resolution level
            level_dims = slide.level_dimensions[level]
            
            tiles = []
            tile_dir = output_dir / 'tiles'
            tile_dir.mkdir(exist_ok=True)
            
            # Calculate step size
            step = self.tile_size - self.overlap
            
            x_coords = range(0, level_dims[0] - self.tile_size, step)
            y_coords = range(0, level_dims[1] - self.tile_size, step)
            
            total_tiles = len(x_coords) * len(y_coords)
            logger.info(f"Extracting {total_tiles} tiles using basic method")
            
            tile_count = 0
            for i, x in enumerate(x_coords):
                for j, y in enumerate(y_coords):
                    # Extract tile
                    tile = slide.read_region(
                        (x, y), level, (self.tile_size, self.tile_size)
                    ).convert('RGB')
                    
                    # Apply basic tissue detection (Otsu threshold)
                    tile_array = np.array(tile)
                    gray = cv2.cvtColor(tile_array, cv2.COLOR_RGB2GRAY)
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Check tissue content
                    tissue_ratio = np.sum(binary == 0) / binary.size
                    if tissue_ratio > 0.1:  # At least 10% tissue
                        tile_filename = f"{slide_path.stem}_{tile_count:06d}.jpg"
                        tile_path = tile_dir / tile_filename
                        tile.save(tile_path, 'JPEG', quality=95)
                        tiles.append(tile_path)
                        tile_count += 1
            
            slide.close()
            logger.info(f"Extracted {len(tiles)} tiles with tissue content")
            return tiles
            
        except Exception as e:
            logger.error(f"Basic tile extraction failed: {e}")
            return []
    
    def create_mil_manifest(self, slide_paths: List[Path], output_dir: Path, 
                           clinical_labels: Dict[str, str] = None) -> Path:
        """Create data manifest for MIL training"""
        manifest_data = []
        
        for slide_path in slide_paths:
            # Extract patient and slide info
            slide_name = slide_path.stem
            patient_id = self._extract_patient_id(slide_name)
            
            # Get clinical label if available
            her2_label = 'Unknown'
            clinical_category = 'Unknown'
            
            if clinical_labels and slide_name in clinical_labels:
                her2_score = clinical_labels[slide_name]
                clinical_category = self.her2_mapping.get(her2_score, 'Unknown')
                her2_label = her2_score
            
            # Get tile paths for this slide
            tile_dir = output_dir / 'tiles'
            slide_tiles = list(tile_dir.glob(f"{slide_name}*.jpg"))
            
            for tile_path in slide_tiles:
                manifest_data.append({
                    'slide_id': slide_name,
                    'patient_id': patient_id,
                    'tile_path': str(tile_path.relative_to(output_dir)),
                    'her2_score': her2_label,
                    'clinical_category': clinical_category,
                    'coordinates': self._extract_coordinates(tile_path.name),
                    'magnification': self.magnification,
                    'tile_size': self.tile_size
                })
        
        # Save manifest
        manifest_df = pd.DataFrame(manifest_data)
        manifest_path = output_dir / 'manifests' / 'data_manifest.csv'
        manifest_path.parent.mkdir(exist_ok=True)
        manifest_df.to_csv(manifest_path, index=False)
        
        logger.info(f"Created MIL manifest: {manifest_path}")
        logger.info(f"Total tiles: {len(manifest_data)}")
        logger.info(f"Unique patients: {manifest_df['patient_id'].nunique()}")
        logger.info(f"Label distribution:\n{manifest_df['clinical_category'].value_counts()}")
        
        return manifest_path
    
    def _extract_patient_id(self, slide_name: str) -> str:
        """Extract patient ID from slide filename"""
        # Common patterns: AC000-000-0000111 -> AC000-000
        parts = slide_name.split('-')
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
        return slide_name[:10]  # Fallback
    
    def _extract_coordinates(self, tile_filename: str) -> Tuple[int, int]:
        """Extract tile coordinates from filename"""
        # Pattern: slidename_123456.jpg where 123456 encodes coordinates
        try:
            base = tile_filename.split('_')[-1].split('.')[0]
            coord_id = int(base)
            # Simple encoding: coord_id = y * 10000 + x (assuming max 10000 tiles per row)
            y = coord_id // 10000
            x = coord_id % 10000
            return (x, y)
        except:
            return (0, 0)
    
    def create_patient_bags(self, manifest_path: Path, output_dir: Path, 
                           max_instances: int = 1000) -> Path:
        """Create patient-level bags for MIL training"""
        manifest_df = pd.read_csv(manifest_path)
        
        # Group by patient
        patient_bags = {}
        for _, row in manifest_df.iterrows():
            patient_id = row['patient_id']
            if patient_id not in patient_bags:
                patient_bags[patient_id] = {
                    'tiles': [],
                    'her2_score': row['her2_score'],
                    'clinical_category': row['clinical_category']
                }
            patient_bags[patient_id]['tiles'].append(row['tile_path'])
        
        # Create bag files
        bags_dir = output_dir / 'bags'
        bags_dir.mkdir(exist_ok=True)
        
        bag_manifest = []
        
        for patient_id, bag_data in patient_bags.items():
            # Sample tiles if too many
            tiles = bag_data['tiles']
            if len(tiles) > max_instances:
                # Random sampling for now - could use attention-based sampling later
                tiles = np.random.choice(tiles, max_instances, replace=False).tolist()
            
            # Create bag metadata
            bag_info = {
                'patient_id': patient_id,
                'tile_paths': tiles,
                'tile_count': len(tiles),
                'her2_score': bag_data['her2_score'],
                'clinical_category': bag_data['clinical_category'],
                'label_numeric': self._get_numeric_label(bag_data['clinical_category'])
            }
            
            # Save bag file
            bag_path = bags_dir / f"{patient_id}_bag.json"
            with open(bag_path, 'w') as f:
                json.dump(bag_info, f, indent=2)
            
            bag_manifest.append({
                'patient_id': patient_id,
                'bag_path': str(bag_path.relative_to(output_dir)),
                'tile_count': len(tiles),
                'her2_score': bag_data['her2_score'],
                'clinical_category': bag_data['clinical_category'],
                'label_numeric': bag_info['label_numeric']
            })
        
        # Save bag manifest
        bag_manifest_df = pd.DataFrame(bag_manifest)
        bag_manifest_path = output_dir / 'manifests' / 'bag_manifest.csv'
        bag_manifest_df.to_csv(bag_manifest_path, index=False)
        
        logger.info(f"Created {len(patient_bags)} patient bags")
        logger.info(f"Bag manifest: {bag_manifest_path}")
        
        return bag_manifest_path
    
    def _get_numeric_label(self, clinical_category: str) -> int:
        """Convert clinical category to numeric label"""
        mapping = {'Negative': 0, 'Low': 1, 'High': 2, 'Unknown': -1}
        return mapping.get(clinical_category, -1)

def load_clinical_labels(labels_path: Path) -> Dict[str, str]:
    """Load clinical HER2 labels from CSV"""
    if not labels_path.exists():
        logger.warning(f"Clinical labels file not found: {labels_path}")
        return {}
    
    try:
        df = pd.read_csv(labels_path)
        # Expected columns: slide_id, her2_score
        if 'slide_id' in df.columns and 'her2_score' in df.columns:
            labels = dict(zip(df['slide_id'], df['her2_score']))
            logger.info(f"Loaded {len(labels)} clinical labels")
            return labels
        else:
            logger.warning("Clinical labels CSV must have 'slide_id' and 'her2_score' columns")
            return {}
    except Exception as e:
        logger.error(f"Failed to load clinical labels: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description='HER2 Preprocessing Pipeline')
    parser.add_argument('--input-path', required=True, help='Path to input slides directory')
    parser.add_argument('--tile-size', type=int, default=224, help='Tile size (224 for MIL, 256 for segmentation)')
    parser.add_argument('--overlap', type=int, default=0, help='Tile overlap in pixels')
    parser.add_argument('--magnification', type=float, default=0.5, help='Magnification (mpp)')
    parser.add_argument('--output-tiles', default='./tiles', help='Output directory for tiles')
    parser.add_argument('--output-bags', default='./bags', help='Output directory for MIL bags')
    parser.add_argument('--output-manifests', default='./manifests', help='Output directory for manifests')
    parser.add_argument('--clinical-labels', help='Path to clinical labels CSV')
    parser.add_argument('--max-instances', type=int, default=1000, help='Max instances per patient bag')
    
    args = parser.parse_args()
    
    logger.info("🩺 Starting HER2 Preprocessing Pipeline")
    logger.info("=" * 50)
    logger.info(f"Input path: {args.input_path}")
    logger.info(f"Tile size: {args.tile_size}px")
    logger.info(f"Magnification: {args.magnification}mpp")
    
    # Initialize preprocessor
    preprocessor = HER2Preprocessor(
        tile_size=args.tile_size,
        overlap=args.overlap,
        magnification=args.magnification
    )
    
    # Setup paths
    input_path = Path(args.input_path)
    output_dir = Path(args.output_tiles).parent
    output_dir.mkdir(exist_ok=True)
    
    # Find slide files
    slide_extensions = ['.svs', '.ndpi', '.tiff', '.tif', '.mrxs']
    slide_paths = []
    
    for ext in slide_extensions:
        slide_paths.extend(input_path.glob(f"*{ext}"))
        slide_paths.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not slide_paths:
        logger.error(f"No slide files found in {input_path}")
        logger.error(f"Supported formats: {slide_extensions}")
        return
    
    logger.info(f"Found {len(slide_paths)} slides to process")
    
    # Load clinical labels if provided
    clinical_labels = {}
    if args.clinical_labels:
        clinical_labels = load_clinical_labels(Path(args.clinical_labels))
    
    # Process each slide
    all_tile_paths = []
    for i, slide_path in enumerate(slide_paths, 1):
        logger.info(f"Processing slide {i}/{len(slide_paths)}: {slide_path.name}")
        
        tile_paths = preprocessor.extract_tiles_slideflow(
            slide_path, Path(args.output_tiles), 
            preprocessor._extract_patient_id(slide_path.stem)
        )
        all_tile_paths.extend(tile_paths)
        
        logger.info(f"Slide {i} complete: {len(tile_paths)} tiles extracted")
    
    logger.info(f"Total tiles extracted: {len(all_tile_paths)}")
    
    # Create MIL manifest
    logger.info("Creating MIL data manifest...")
    manifest_path = preprocessor.create_mil_manifest(
        slide_paths, output_dir, clinical_labels
    )
    
    # Create patient bags
    logger.info("Creating patient-level bags...")
    bag_manifest_path = preprocessor.create_patient_bags(
        manifest_path, output_dir, args.max_instances
    )
    
    logger.info("✅ HER2 preprocessing pipeline completed!")
    logger.info(f"📁 Tiles: {args.output_tiles}")
    logger.info(f"📊 Bags: {args.output_bags}")
    logger.info(f"📋 Manifests: {args.output_manifests}")

if __name__ == "__main__":
    main()