#!/usr/bin/env python3
"""
Efficient single-pass EDA for training data.
O(n) traversal, O(1) per-file operations.
"""

import os
import json
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

def single_pass_eda(base_path: str) -> dict:
    """
    Single traversal collecting all stats.
    Returns structured results for documentation.
    """
    # Data structures for O(1) aggregation
    class_file_counts = Counter()
    class_sizes = defaultdict(int)
    ext_counts = Counter()
    ext_sizes = defaultdict(int)
    
    total_files = 0
    total_size = 0
    sample_files = []  # First 10 files per extension for inspection
    samples_per_ext = defaultdict(list)
    
    base = Path(base_path)
    
    # Single os.walk traversal - most efficient for large directories
    for root, dirs, files in os.walk(base):
        root_path = Path(root)
        # Extract class label from path structure
        # /workspace/training_data/extracted/Fast/TrainData/RYLTY/Organizer/Statement/{CLASS}/...
        parts = root_path.parts
        
        # Find class label (folder after "Statement")
        try:
            stmt_idx = parts.index('Statement')
            if len(parts) > stmt_idx + 1:
                class_label = parts[stmt_idx + 1]
            else:
                class_label = "UNKNOWN"
        except ValueError:
            class_label = "UNKNOWN"
        
        for fname in files:
            fpath = root_path / fname
            ext = fpath.suffix.lower()
            
            try:
                size = fpath.stat().st_size
            except (OSError, PermissionError):
                size = 0
            
            # Aggregate
            class_file_counts[class_label] += 1
            class_sizes[class_label] += size
            ext_counts[ext] += 1
            ext_sizes[ext] += size
            total_files += 1
            total_size += size
            
            # Collect samples (max 5 per extension)
            if len(samples_per_ext[ext]) < 5:
                samples_per_ext[ext].append(str(fpath))
    
    # Compute statistics
    counts = list(class_file_counts.values())
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "base_path": base_path,
        "summary": {
            "total_classes": len(class_file_counts),
            "total_files": total_files,
            "total_size_gb": round(total_size / 1e9, 2),
        },
        "class_stats": {
            "min_files": min(counts) if counts else 0,
            "max_files": max(counts) if counts else 0,
            "mean_files": round(sum(counts) / len(counts), 1) if counts else 0,
            "median_files": sorted(counts)[len(counts)//2] if counts else 0,
            "classes_lt_5_files": len([c for c in counts if c < 5]),
            "classes_lt_10_files": len([c for c in counts if c < 10]),
            "classes_gt_100_files": len([c for c in counts if c > 100]),
            "classes_gt_500_files": len([c for c in counts if c > 500]),
        },
        "class_distribution": {
            "top_30": dict(class_file_counts.most_common(30)),
            "bottom_10": dict(class_file_counts.most_common()[:-11:-1]),
        },
        "extension_distribution": {
            "counts": dict(ext_counts.most_common()),
            "sizes_gb": {k: round(v/1e9, 2) for k, v in sorted(ext_sizes.items(), key=lambda x: -x[1])},
        },
        "samples": dict(samples_per_ext),
    }
    
    return results


def print_results(results: dict):
    """Pretty print EDA results."""
    print("=" * 60)
    print("MUSIC RIGHTS TRAINING DATA - EDA RESULTS")
    print("=" * 60)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Base path: {results['base_path']}")
    
    s = results['summary']
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Total classes:     {s['total_classes']}")
    print(f"  Total files:       {s['total_files']:,}")
    print(f"  Total size:        {s['total_size_gb']} GB")
    
    cs = results['class_stats']
    print(f"\n{'='*60}")
    print("CLASS STATISTICS")
    print(f"{'='*60}")
    print(f"  Min files/class:   {cs['min_files']}")
    print(f"  Max files/class:   {cs['max_files']}")
    print(f"  Mean files/class:  {cs['mean_files']}")
    print(f"  Median files/class:{cs['median_files']}")
    print(f"  Classes < 5 files: {cs['classes_lt_5_files']}")
    print(f"  Classes < 10 files:{cs['classes_lt_10_files']}")
    print(f"  Classes > 100 files:{cs['classes_gt_100_files']}")
    print(f"  Classes > 500 files:{cs['classes_gt_500_files']}")
    
    print(f"\n{'='*60}")
    print("TOP 30 CLASSES BY FILE COUNT")
    print(f"{'='*60}")
    for name, count in results['class_distribution']['top_30'].items():
        print(f"  {count:5d}  {name}")
    
    print(f"\n{'='*60}")
    print("BOTTOM 10 CLASSES BY FILE COUNT")
    print(f"{'='*60}")
    for name, count in results['class_distribution']['bottom_10'].items():
        print(f"  {count:5d}  {name}")
    
    print(f"\n{'='*60}")
    print("FILE EXTENSIONS")
    print(f"{'='*60}")
    for ext, count in results['extension_distribution']['counts'].items():
        size = results['extension_distribution']['sizes_gb'].get(ext, 0)
        print(f"  {count:6d}  {ext:10s}  ({size} GB)")


if __name__ == '__main__':
    import sys
    
    base = sys.argv[1] if len(sys.argv) > 1 else '/workspace/training_data/extracted/Fast/TrainData/RYLTY/Organizer/Statement'
    
    print(f"Running single-pass EDA on: {base}")
    print("This should complete in one traversal...")
    
    results = single_pass_eda(base)
    print_results(results)
    
    # Save JSON for documentation
    output_path = '/workspace/eda_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
