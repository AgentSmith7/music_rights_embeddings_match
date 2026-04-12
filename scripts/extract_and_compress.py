#!/usr/bin/env python3
"""
Extract training data and optionally recompress with zstd for better compression.
Also provides utilities for managing extracted data efficiently.
"""

import os
import sys
import tarfile
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_tgz(archive_path: str, output_dir: str):
    """Extract .tgz archive to output directory."""
    log(f"Extracting {archive_path} to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with tarfile.open(archive_path, 'r:gz') as tar:
        total_members = len(tar.getmembers())
        log(f"Total members in archive: {total_members}")
        
        for i, member in enumerate(tar.getmembers()):
            tar.extract(member, output_dir)
            if (i + 1) % 10000 == 0:
                log(f"Extracted {i+1}/{total_members} files...")
    
    log(f"Extraction complete!")


def analyze_extracted_data(data_dir: str):
    """Analyze the extracted data structure."""
    log(f"Analyzing {data_dir}...")
    
    stats = {
        "total_files": 0,
        "total_size_bytes": 0,
        "extensions": defaultdict(int),
        "classes": defaultdict(int),
        "class_sizes": defaultdict(int)
    }
    
    statement_path = os.path.join(data_dir, "Fast/TrainData/RYLTY/Organizer/Statement")
    
    if not os.path.exists(statement_path):
        log(f"Statement path not found: {statement_path}")
        # Try to find it
        for root, dirs, files in os.walk(data_dir):
            if "Statement" in dirs:
                statement_path = os.path.join(root, "Statement")
                log(f"Found Statement at: {statement_path}")
                break
    
    if os.path.exists(statement_path):
        for class_name in os.listdir(statement_path):
            class_path = os.path.join(statement_path, class_name)
            if not os.path.isdir(class_path):
                continue
            
            for filename in os.listdir(class_path):
                filepath = os.path.join(class_path, filename)
                if not os.path.isfile(filepath):
                    continue
                
                stats["total_files"] += 1
                size = os.path.getsize(filepath)
                stats["total_size_bytes"] += size
                
                ext = os.path.splitext(filename)[1].lower()
                stats["extensions"][ext] += 1
                stats["classes"][class_name] += 1
                stats["class_sizes"][class_name] += size
    
    log(f"\n{'='*60}")
    log(f"DATA ANALYSIS")
    log(f"{'='*60}")
    log(f"Total files: {stats['total_files']:,}")
    log(f"Total size: {stats['total_size_bytes'] / (1024**3):.2f} GB")
    log(f"Number of classes: {len(stats['classes'])}")
    
    log(f"\nFile extensions:")
    for ext, count in sorted(stats["extensions"].items(), key=lambda x: -x[1])[:10]:
        log(f"  {ext or '(no ext)'}: {count:,}")
    
    log(f"\nTop 10 classes by file count:")
    for cls, count in sorted(stats["classes"].items(), key=lambda x: -x[1])[:10]:
        size_mb = stats["class_sizes"][cls] / (1024**2)
        log(f"  {cls}: {count:,} files ({size_mb:.1f} MB)")
    
    log(f"\nSmallest 10 classes:")
    for cls, count in sorted(stats["classes"].items(), key=lambda x: x[1])[:10]:
        log(f"  {cls}: {count:,} files")
    
    return stats


def compress_with_zstd(input_dir: str, output_file: str, compression_level: int = 3):
    """Compress directory with zstd (much faster than gzip, better compression)."""
    log(f"Compressing {input_dir} to {output_file} with zstd (level {compression_level})")
    
    # Check if zstd is available
    try:
        subprocess.run(["zstd", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        log("zstd not found, installing...")
        subprocess.run(["apt-get", "update"], capture_output=True)
        subprocess.run(["apt-get", "install", "-y", "zstd"], capture_output=True)
    
    # Create tar.zst archive
    tar_file = output_file.replace('.zst', '')
    if not tar_file.endswith('.tar'):
        tar_file = output_file + '.tar'
    
    log(f"Creating tar archive...")
    subprocess.run([
        "tar", "-cf", tar_file, "-C", os.path.dirname(input_dir), os.path.basename(input_dir)
    ], check=True)
    
    log(f"Compressing with zstd...")
    subprocess.run([
        "zstd", f"-{compression_level}", "--rm", tar_file, "-o", output_file
    ], check=True)
    
    final_size = os.path.getsize(output_file) / (1024**3)
    log(f"Compressed archive: {output_file} ({final_size:.2f} GB)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract and manage training data")
    parser.add_argument('--extract', type=str, help='Path to .tgz archive to extract')
    parser.add_argument('--output', type=str, default='/workspace/training_data/extracted',
                        help='Output directory for extraction')
    parser.add_argument('--analyze', type=str, help='Analyze extracted data directory')
    parser.add_argument('--compress', type=str, help='Directory to compress with zstd')
    parser.add_argument('--compress-output', type=str, help='Output path for compressed archive')
    parser.add_argument('--compress-level', type=int, default=3, help='Zstd compression level (1-19)')
    
    args = parser.parse_args()
    
    if args.extract:
        extract_tgz(args.extract, args.output)
        analyze_extracted_data(args.output)
    
    if args.analyze:
        analyze_extracted_data(args.analyze)
    
    if args.compress:
        if not args.compress_output:
            args.compress_output = args.compress + '.tar.zst'
        compress_with_zstd(args.compress, args.compress_output, args.compress_level)


if __name__ == "__main__":
    main()
