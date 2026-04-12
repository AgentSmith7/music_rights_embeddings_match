#!/usr/bin/env python3
"""
Complete training pipeline for music rights document classification.

This script:
1. Extracts the training archive (if not already extracted)
2. Analyzes the directory structure and generates EDA
3. Parses documents and generates embeddings
4. Indexes all documents to Qdrant
"""

import os
import sys
import tarfile
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from loguru import logger
import pandas as pd

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

# Constants
ARCHIVE_PATH = "/workspace/training_data/Copy of trainData.tgz"
EXTRACT_PATH = "/workspace/training_data/extracted"
QDRANT_PATH = "/workspace/qdrant_data"
RESULTS_PATH = "/workspace/results"

# Directory structure: Fast/TrainData/RYLTY/Organizer/Statement/{CLASS_LABEL}/{filename}
CLASS_LEVEL = 5  # 0-indexed position of class label in path


@dataclass
class ClassStats:
    """Statistics for a single class."""
    name: str
    file_count: int = 0
    total_size: int = 0
    extensions: Dict[str, int] = None
    
    def __post_init__(self):
        if self.extensions is None:
            self.extensions = defaultdict(int)


def check_extraction_status() -> bool:
    """Check if archive is already extracted."""
    if not os.path.exists(EXTRACT_PATH):
        return False
    
    # Check if extraction looks complete
    statement_path = os.path.join(EXTRACT_PATH, "Fast/TrainData/RYLTY/Organizer/Statement")
    if os.path.exists(statement_path):
        classes = os.listdir(statement_path)
        if len(classes) > 10:  # Should have many class folders
            logger.info(f"Found {len(classes)} class folders - extraction appears complete")
            return True
    return False


def extract_archive():
    """Extract the training archive."""
    if check_extraction_status():
        logger.info("Archive already extracted, skipping...")
        return
    
    logger.info(f"Extracting archive: {ARCHIVE_PATH}")
    logger.info(f"Destination: {EXTRACT_PATH}")
    
    os.makedirs(EXTRACT_PATH, exist_ok=True)
    
    start = time.time()
    with tarfile.open(ARCHIVE_PATH, 'r:gz') as tar:
        # Count members for progress bar
        members = tar.getmembers()
        logger.info(f"Archive contains {len(members)} entries")
        
        for member in tqdm(members, desc="Extracting"):
            tar.extract(member, EXTRACT_PATH)
    
    elapsed = time.time() - start
    logger.info(f"Extraction complete in {elapsed/60:.1f} minutes")


def analyze_structure() -> Tuple[Dict[str, ClassStats], Dict]:
    """Analyze the extracted directory structure."""
    logger.info("Analyzing directory structure...")
    
    statement_path = os.path.join(EXTRACT_PATH, "Fast/TrainData/RYLTY/Organizer/Statement")
    
    if not os.path.exists(statement_path):
        raise ValueError(f"Statement path not found: {statement_path}")
    
    class_stats: Dict[str, ClassStats] = {}
    global_stats = {
        'total_files': 0,
        'total_size': 0,
        'extensions': defaultdict(int),
    }
    
    class_folders = os.listdir(statement_path)
    logger.info(f"Found {len(class_folders)} class folders")
    
    for class_name in tqdm(class_folders, desc="Analyzing classes"):
        class_path = os.path.join(statement_path, class_name)
        if not os.path.isdir(class_path):
            continue
        
        stats = ClassStats(name=class_name)
        
        for filename in os.listdir(class_path):
            filepath = os.path.join(class_path, filename)
            if os.path.isfile(filepath):
                stats.file_count += 1
                stats.total_size += os.path.getsize(filepath)
                ext = os.path.splitext(filename)[1].lower()
                stats.extensions[ext] += 1
                
                global_stats['total_files'] += 1
                global_stats['total_size'] += os.path.getsize(filepath)
                global_stats['extensions'][ext] += 1
        
        class_stats[class_name] = stats
    
    return class_stats, global_stats


def print_eda_report(class_stats: Dict[str, ClassStats], global_stats: Dict):
    """Print EDA report."""
    print("\n" + "="*80)
    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("="*80)
    
    print(f"\n📊 GLOBAL STATISTICS")
    print(f"   Total classes: {len(class_stats)}")
    print(f"   Total files: {global_stats['total_files']:,}")
    print(f"   Total size: {global_stats['total_size'] / (1024**3):.2f} GB")
    
    print(f"\n📁 FILE EXTENSIONS")
    for ext, count in sorted(global_stats['extensions'].items(), key=lambda x: -x[1]):
        pct = count / global_stats['total_files'] * 100
        print(f"   {ext or '(no ext)'}: {count:,} files ({pct:.1f}%)")
    
    print(f"\n🏆 TOP 20 CLASSES BY FILE COUNT")
    sorted_classes = sorted(class_stats.values(), key=lambda x: -x.file_count)[:20]
    for i, stats in enumerate(sorted_classes, 1):
        exts = ', '.join(f"{k}:{v}" for k, v in sorted(stats.extensions.items(), key=lambda x: -x[1])[:3])
        print(f"   {i:2}. {stats.name}: {stats.file_count:,} files, {stats.total_size/(1024**2):.1f} MB")
        print(f"       Extensions: {exts}")
    
    print(f"\n📉 BOTTOM 10 CLASSES BY FILE COUNT")
    sorted_classes = sorted(class_stats.values(), key=lambda x: x.file_count)[:10]
    for i, stats in enumerate(sorted_classes, 1):
        print(f"   {i:2}. {stats.name}: {stats.file_count:,} files")
    
    # Class size distribution
    file_counts = [s.file_count for s in class_stats.values()]
    print(f"\n📈 CLASS SIZE DISTRIBUTION")
    print(f"   Min files per class: {min(file_counts)}")
    print(f"   Max files per class: {max(file_counts)}")
    print(f"   Median files per class: {sorted(file_counts)[len(file_counts)//2]}")
    print(f"   Mean files per class: {sum(file_counts)/len(file_counts):.1f}")


def save_eda_results(class_stats: Dict[str, ClassStats], global_stats: Dict):
    """Save EDA results to JSON."""
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    results = {
        'global': {
            'total_classes': len(class_stats),
            'total_files': global_stats['total_files'],
            'total_size_bytes': global_stats['total_size'],
            'extensions': dict(global_stats['extensions']),
        },
        'classes': {
            name: {
                'file_count': stats.file_count,
                'total_size': stats.total_size,
                'extensions': dict(stats.extensions),
            }
            for name, stats in class_stats.items()
        }
    }
    
    output_path = os.path.join(RESULTS_PATH, 'eda_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"EDA results saved to {output_path}")


def sample_documents(class_stats: Dict[str, ClassStats], n_per_class: int = 3) -> List[Tuple[str, str, str]]:
    """Sample documents from each class for content analysis."""
    samples = []
    statement_path = os.path.join(EXTRACT_PATH, "Fast/TrainData/RYLTY/Organizer/Statement")
    
    for class_name, stats in class_stats.items():
        class_path = os.path.join(statement_path, class_name)
        files = os.listdir(class_path)[:n_per_class]
        for filename in files:
            filepath = os.path.join(class_path, filename)
            samples.append((class_name, filename, filepath))
    
    return samples


def analyze_sample_content(samples: List[Tuple[str, str, str]]):
    """Analyze content of sample documents."""
    print("\n" + "="*80)
    print("SAMPLE DOCUMENT CONTENT ANALYSIS")
    print("="*80)
    
    csv_samples = [(c, f, p) for c, f, p in samples if f.lower().endswith('.csv')][:10]
    
    for class_name, filename, filepath in csv_samples:
        print(f"\n📄 {class_name} / {filename}")
        try:
            df = pd.read_csv(filepath, nrows=5, encoding='utf-8', on_bad_lines='skip')
            print(f"   Columns ({len(df.columns)}): {', '.join(df.columns[:10])}")
            if len(df.columns) > 10:
                print(f"   ... and {len(df.columns) - 10} more columns")
        except Exception as e:
            print(f"   Error reading: {e}")


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("MUSIC RIGHTS DOCUMENT CLASSIFICATION - TRAINING PIPELINE")
    logger.info("="*60)
    
    # Check if archive exists
    if not os.path.exists(ARCHIVE_PATH):
        logger.error(f"Archive not found: {ARCHIVE_PATH}")
        logger.info("Please ensure the archive is downloaded first")
        return
    
    # Step 1: Extract archive
    logger.info("\n[STEP 1/4] Extracting archive...")
    extract_archive()
    
    # Step 2: Analyze structure
    logger.info("\n[STEP 2/4] Analyzing directory structure...")
    class_stats, global_stats = analyze_structure()
    
    # Step 3: Print and save EDA
    logger.info("\n[STEP 3/4] Generating EDA report...")
    print_eda_report(class_stats, global_stats)
    save_eda_results(class_stats, global_stats)
    
    # Step 4: Sample content analysis
    logger.info("\n[STEP 4/4] Analyzing sample document content...")
    samples = sample_documents(class_stats, n_per_class=2)
    analyze_sample_content(samples)
    
    logger.info("\n" + "="*60)
    logger.info("EDA COMPLETE - Ready for embedding generation")
    logger.info("="*60)


if __name__ == "__main__":
    main()
