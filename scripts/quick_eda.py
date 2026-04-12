#!/usr/bin/env python3
"""Quick EDA script for analyzing training data structure."""

import tarfile
from collections import defaultdict
import os
import sys

def analyze_archive(archive_path: str, max_entries: int = 500000):
    """Analyze archive structure quickly."""
    class_stats = defaultdict(lambda: {'count': 0, 'size': 0, 'extensions': set()})
    total_files = 0
    total_size = 0
    
    print(f'Analyzing archive: {archive_path}')
    print(f'Max entries to process: {max_entries}')
    print('-' * 60)
    
    with tarfile.open(archive_path, 'r:gz') as tar:
        for i, member in enumerate(tar):
            if member.isfile():
                parts = member.name.split('/')
                if len(parts) >= 6:
                    class_label = parts[5]
                    ext = os.path.splitext(member.name)[1].lower()
                    class_stats[class_label]['count'] += 1
                    class_stats[class_label]['size'] += member.size
                    class_stats[class_label]['extensions'].add(ext)
                    total_files += 1
                    total_size += member.size
            
            if i % 50000 == 0 and i > 0:
                print(f'Processed {i} entries, {len(class_stats)} classes, {total_files} files')
            
            if max_entries and i > max_entries:
                print(f'Reached {max_entries} entries limit')
                break
    
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'Total files analyzed: {total_files:,}')
    print(f'Total size: {total_size / (1024**3):.2f} GB')
    print(f'Number of classes: {len(class_stats)}')
    
    print('\n' + '=' * 60)
    print('TOP 20 CLASSES BY FILE COUNT')
    print('=' * 60)
    sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:20]
    for rank, (cls, stats) in enumerate(sorted_classes, 1):
        exts = ', '.join(sorted(stats['extensions']))
        print(f'{rank:2}. {cls}: {stats["count"]:,} files, {stats["size"]/(1024**2):.1f} MB')
        print(f'    Extensions: {exts}')
    
    print('\n' + '=' * 60)
    print('ALL CLASSES')
    print('=' * 60)
    for cls, stats in sorted(class_stats.items()):
        print(f'{cls}: {stats["count"]:,} files')
    
    print('\n' + '=' * 60)
    print('FILE EXTENSION DISTRIBUTION')
    print('=' * 60)
    ext_counts = defaultdict(int)
    for stats in class_stats.values():
        for ext in stats['extensions']:
            ext_counts[ext] += 1
    for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
        print(f'{ext or "(no extension)"}: appears in {count} classes')
    
    return class_stats

if __name__ == '__main__':
    archive_path = '/workspace/training_data/Copy of trainData.tgz'
    max_entries = int(sys.argv[1]) if len(sys.argv) > 1 else 500000
    analyze_archive(archive_path, max_entries)
