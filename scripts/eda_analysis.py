#!/usr/bin/env python3
"""
Comprehensive EDA for Music Rights Training Data
Generates visualizations and statistics about the training dataset.

Directory Structure:
    Fast/TrainData/RYLTY/Organizer/<DocType>/<ClassLabel>/files
    
Where ClassLabel (the parent folder) is the class we want to predict.
"""
import os
import sys
import json
import tarfile
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import io

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


@dataclass
class FileInfo:
    path: str
    name: str
    extension: str
    size: int
    class_label: str  # Parent folder = class label
    doc_type: str     # Document type (Statement, Contract, etc.)
    full_hierarchy: str


@dataclass
class ClassStats:
    name: str
    file_count: int = 0
    total_size: int = 0
    extensions: Dict[str, int] = field(default_factory=dict)
    sample_files: List[str] = field(default_factory=list)


class TrainingDataEDA:
    """EDA analysis for training data."""
    
    def __init__(self, tar_path: str, output_dir: str = "/workspace/eda_output"):
        self.tar_path = tar_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.files: List[FileInfo] = []
        self.class_stats: Dict[str, ClassStats] = {}
        self.doc_type_stats: Dict[str, int] = defaultdict(int)
        self.total_files = 0
        self.total_size = 0
        self.hierarchy_levels: Dict[str, set] = defaultdict(set)
        
    def analyze_tar_structure(self, max_files: Optional[int] = None):
        """Analyze the tar archive structure."""
        logger.info(f"Analyzing tar archive: {self.tar_path}")
        
        with tarfile.open(self.tar_path, 'r:*') as tar:
            for i, member in enumerate(tar):
                if max_files and i >= max_files:
                    break
                
                if not member.isfile():
                    continue
                
                path = member.name
                parts = Path(path).parts
                
                if len(parts) < 2:
                    continue
                
                file_name = parts[-1]
                ext = Path(file_name).suffix.lower()
                
                # Skip non-document files
                if ext not in ['.pdf', '.csv', '.xlsx', '.xls', '.txt', '.doc', '.docx']:
                    continue
                
                # Extract class label - it's the PARENT FOLDER of the file
                # Structure: Fast/TrainData/RYLTY/Organizer/Statement/<CLASS_LABEL>/file.csv
                class_label = parts[-2] if len(parts) >= 2 else "Unknown"
                
                # Extract document type (e.g., Statement, Contract)
                doc_type = parts[-3] if len(parts) >= 3 else "Unknown"
                
                # Track hierarchy levels
                for level, part in enumerate(parts[:-1]):  # Exclude filename
                    self.hierarchy_levels[f"level_{level}"].add(part)
                
                file_info = FileInfo(
                    path=path,
                    name=file_name,
                    extension=ext,
                    size=member.size,
                    class_label=class_label,
                    doc_type=doc_type,
                    full_hierarchy="/".join(parts[:-1])
                )
                
                self.files.append(file_info)
                self.total_files += 1
                self.total_size += member.size
                self.doc_type_stats[doc_type] += 1
                
                # Update class stats
                if class_label not in self.class_stats:
                    self.class_stats[class_label] = ClassStats(name=class_label)
                
                stats = self.class_stats[class_label]
                stats.file_count += 1
                stats.total_size += member.size
                stats.extensions[ext] = stats.extensions.get(ext, 0) + 1
                
                if len(stats.sample_files) < 5:
                    stats.sample_files.append(path)
                
                if i % 10000 == 0 and i > 0:
                    logger.info(f"Processed {i} entries, found {len(self.class_stats)} classes...")
        
        logger.info(f"Analysis complete: {self.total_files} files, {len(self.class_stats)} classes")
    
    def generate_statistics(self) -> Dict:
        """Generate comprehensive statistics."""
        stats = {
            "total_files": self.total_files,
            "total_size_bytes": self.total_size,
            "total_size_gb": round(self.total_size / (1024**3), 2),
            "num_classes": len(self.class_stats),
            "doc_types": dict(self.doc_type_stats),
            "hierarchy_levels": {k: list(v)[:20] for k, v in self.hierarchy_levels.items()},
            "classes": {},
            "extension_distribution": {},
            "size_distribution": {
                "min_bytes": 0,
                "max_bytes": 0,
                "mean_bytes": 0,
                "median_bytes": 0
            }
        }
        
        # Extension distribution
        ext_counts = Counter()
        sizes = []
        
        for f in self.files:
            ext_counts[f.extension] += 1
            sizes.append(f.size)
        
        stats["extension_distribution"] = dict(ext_counts.most_common())
        
        if sizes:
            stats["size_distribution"] = {
                "min_bytes": min(sizes),
                "max_bytes": max(sizes),
                "mean_bytes": int(np.mean(sizes)),
                "median_bytes": int(np.median(sizes))
            }
        
        # Class statistics
        for class_name, class_stat in sorted(self.class_stats.items(), 
                                              key=lambda x: x[1].file_count, 
                                              reverse=True):
            stats["classes"][class_name] = {
                "file_count": class_stat.file_count,
                "total_size_mb": round(class_stat.total_size / (1024**2), 2),
                "extensions": class_stat.extensions,
                "sample_files": class_stat.sample_files[:3]
            }
        
        return stats
    
    def plot_class_distribution(self):
        """Plot class distribution."""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # 1. File count by class (top 30)
        ax1 = axes[0, 0]
        classes = sorted(self.class_stats.items(), key=lambda x: x[1].file_count, reverse=True)[:30]
        names = [c[0][:35] for c in classes]
        counts = [c[1].file_count for c in classes]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        bars = ax1.barh(names, counts, color=colors)
        ax1.set_xlabel('Number of Files', fontsize=12)
        ax1.set_title('Top 30 Classes by File Count', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{count:,}', va='center', fontsize=8)
        
        # 2. Size by class (top 30)
        ax2 = axes[0, 1]
        sizes_mb = [c[1].total_size / (1024**2) for c in classes]
        
        bars = ax2.barh(names, sizes_mb, color=colors)
        ax2.set_xlabel('Total Size (MB)', fontsize=12)
        ax2.set_title('Top 30 Classes by Total Size', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        
        # 3. Extension distribution
        ax3 = axes[1, 0]
        ext_counts = Counter()
        for f in self.files:
            ext_counts[f.extension] += 1
        
        exts = ext_counts.most_common(10)
        ext_names = [e[0] for e in exts]
        ext_values = [e[1] for e in exts]
        
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(ext_names)))
        wedges, texts, autotexts = ax3.pie(ext_values, labels=ext_names, autopct='%1.1f%%', 
                                           startangle=90, colors=colors_pie)
        ax3.set_title('File Type Distribution', fontsize=14, fontweight='bold')
        
        # 4. Class count distribution (histogram)
        ax4 = axes[1, 1]
        class_file_counts = [c.file_count for c in self.class_stats.values()]
        
        ax4.hist(class_file_counts, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Files per Class', fontsize=12)
        ax4.set_ylabel('Number of Classes', fontsize=12)
        ax4.set_title('Distribution of Class Sizes', fontsize=14, fontweight='bold')
        ax4.axvline(np.median(class_file_counts), color='red', linestyle='--', 
                   label=f'Median: {np.median(class_file_counts):.0f}')
        ax4.axvline(np.mean(class_file_counts), color='green', linestyle='--', 
                   label=f'Mean: {np.mean(class_file_counts):.0f}')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved class distribution plot to {self.output_dir / 'class_distribution.png'}")
    
    def plot_detailed_analysis(self):
        """Generate additional detailed plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Document type distribution
        ax1 = axes[0, 0]
        doc_types = sorted(self.doc_type_stats.items(), key=lambda x: x[1], reverse=True)[:15]
        dt_names = [d[0][:30] for d in doc_types]
        dt_counts = [d[1] for d in doc_types]
        
        ax1.barh(dt_names, dt_counts, color='coral')
        ax1.set_xlabel('Number of Files')
        ax1.set_title('Document Types Distribution')
        ax1.invert_yaxis()
        
        # 2. File size distribution (log scale)
        ax2 = axes[0, 1]
        sizes_kb = [f.size / 1024 for f in self.files if f.size > 0]
        
        ax2.hist(sizes_kb, bins=100, color='green', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('File Size (KB)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('File Size Distribution')
        ax2.set_xlim(0, np.percentile(sizes_kb, 95))
        
        # 3. Classes with most diverse file types
        ax3 = axes[1, 0]
        class_diversity = [(name, len(stats.extensions)) 
                          for name, stats in self.class_stats.items()]
        class_diversity.sort(key=lambda x: x[1], reverse=True)
        class_diversity = class_diversity[:15]
        
        cd_names = [c[0][:30] for c in class_diversity]
        cd_values = [c[1] for c in class_diversity]
        
        ax3.barh(cd_names, cd_values, color='purple')
        ax3.set_xlabel('Number of File Types')
        ax3.set_title('Classes with Most File Type Diversity')
        ax3.invert_yaxis()
        
        # 4. Cumulative distribution of classes
        ax4 = axes[1, 1]
        sorted_counts = sorted([c.file_count for c in self.class_stats.values()], reverse=True)
        cumsum = np.cumsum(sorted_counts) / sum(sorted_counts) * 100
        
        ax4.plot(range(1, len(cumsum)+1), cumsum, 'b-', linewidth=2)
        ax4.axhline(80, color='red', linestyle='--', label='80% of files')
        ax4.axhline(90, color='orange', linestyle='--', label='90% of files')
        
        # Find how many classes cover 80% and 90%
        classes_80 = np.searchsorted(cumsum, 80) + 1
        classes_90 = np.searchsorted(cumsum, 90) + 1
        
        ax4.axvline(classes_80, color='red', linestyle=':', alpha=0.5)
        ax4.axvline(classes_90, color='orange', linestyle=':', alpha=0.5)
        
        ax4.set_xlabel('Number of Classes')
        ax4.set_ylabel('Cumulative % of Files')
        ax4.set_title(f'Cumulative Distribution\n({classes_80} classes = 80%, {classes_90} classes = 90%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'detailed_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved detailed analysis to {self.output_dir / 'detailed_analysis.png'}")
    
    def sample_and_analyze_content(self, num_samples: int = 20):
        """Sample files and analyze their content."""
        logger.info(f"Sampling {num_samples} files for content analysis...")
        
        samples = []
        sampled_classes = set()
        
        # Prioritize sampling from different classes
        target_classes = list(self.class_stats.keys())[:num_samples]
        
        with tarfile.open(self.tar_path, 'r:*') as tar:
            for file_info in self.files:
                if len(samples) >= num_samples:
                    break
                
                # Try to sample from different classes
                if file_info.class_label in sampled_classes:
                    continue
                
                if file_info.class_label not in target_classes:
                    continue
                
                try:
                    member = tar.getmember(file_info.path)
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    
                    content = f.read()
                    
                    sample_info = {
                        "path": file_info.path,
                        "class_label": file_info.class_label,
                        "doc_type": file_info.doc_type,
                        "extension": file_info.extension,
                        "size_bytes": file_info.size,
                        "content_preview": None,
                        "columns": None,
                        "num_rows": None,
                        "analysis": {}
                    }
                    
                    if file_info.extension == '.csv':
                        sample_info["analysis"] = self._analyze_csv(content, file_info.name)
                    elif file_info.extension == '.pdf':
                        sample_info["analysis"] = self._analyze_pdf(content, file_info.name)
                    elif file_info.extension in ['.xlsx', '.xls']:
                        sample_info["analysis"] = self._analyze_excel(content, file_info.name)
                    
                    samples.append(sample_info)
                    sampled_classes.add(file_info.class_label)
                    logger.info(f"Sampled: {file_info.class_label} - {file_info.name[:50]}")
                    
                except Exception as e:
                    logger.warning(f"Error sampling {file_info.path}: {e}")
        
        # Save samples
        with open(self.output_dir / 'content_samples.json', 'w') as f:
            json.dump(samples, f, indent=2, default=str)
        
        logger.info(f"Saved {len(samples)} content samples to content_samples.json")
        return samples
    
    def _analyze_csv(self, content: bytes, name: str) -> Dict:
        """Analyze CSV content."""
        if pd is None:
            return {"error": "pandas not available"}
        
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(io.BytesIO(content), nrows=100, encoding=encoding)
                    break
                except:
                    continue
            else:
                return {"error": "Could not decode CSV"}
            
            return {
                "columns": list(df.columns)[:20],
                "num_columns": len(df.columns),
                "num_rows": len(df),
                "dtypes": {col: str(dtype) for col, dtype in list(df.dtypes.items())[:10]},
                "content_preview": df.head(3).to_string()[:1000],
                "column_sample_values": {col: str(df[col].dropna().iloc[0])[:100] 
                                         for col in list(df.columns)[:5] 
                                         if len(df[col].dropna()) > 0}
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_pdf(self, content: bytes, name: str) -> Dict:
        """Analyze PDF content."""
        if fitz is None:
            return {"error": "PyMuPDF not available"}
        
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            num_pages = len(doc)
            
            text = ""
            for page in doc:
                text += page.get_text()
                if len(text) > 3000:
                    break
            doc.close()
            
            # Extract some key terms
            words = text.split()
            word_freq = Counter(w.lower() for w in words if len(w) > 3 and w.isalpha())
            
            return {
                "num_pages": num_pages,
                "content_preview": text[:1500],
                "has_text": len(text.strip()) > 0,
                "text_length": len(text),
                "top_words": dict(word_freq.most_common(20))
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_excel(self, content: bytes, name: str) -> Dict:
        """Analyze Excel content."""
        if pd is None:
            return {"error": "pandas not available"}
        
        try:
            df = pd.read_excel(io.BytesIO(content), nrows=100)
            return {
                "columns": list(df.columns)[:20],
                "num_columns": len(df.columns),
                "num_rows": len(df),
                "content_preview": df.head(3).to_string()[:1000]
            }
        except Exception as e:
            return {"error": str(e)}
    
    def generate_report(self):
        """Generate comprehensive EDA report."""
        stats = self.generate_statistics()
        
        # Save statistics
        with open(self.output_dir / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Generate plots
        self.plot_class_distribution()
        self.plot_detailed_analysis()
        
        # Generate markdown report
        report = f"""# Training Data EDA Report

## Overview
- **Total Files:** {stats['total_files']:,}
- **Total Size:** {stats['total_size_gb']} GB
- **Number of Classes (Distributors/Sources):** {stats['num_classes']}

## Key Finding: Class Labels
The class labels are the **parent folder names** of each file.
These represent different distributors, publishers, or data sources.

## File Size Statistics
- **Min:** {stats['size_distribution']['min_bytes']:,} bytes
- **Max:** {stats['size_distribution']['max_bytes']:,} bytes  
- **Mean:** {stats['size_distribution']['mean_bytes']:,} bytes
- **Median:** {stats['size_distribution']['median_bytes']:,} bytes

## Document Types
| Document Type | Count |
|--------------|-------|
"""
        for doc_type, count in sorted(stats['doc_types'].items(), key=lambda x: x[1], reverse=True)[:10]:
            report += f"| {doc_type} | {count:,} |\n"
        
        report += f"""
## Extension Distribution
| Extension | Count | Percentage |
|-----------|-------|------------|
"""
        total = sum(stats['extension_distribution'].values())
        for ext, count in stats['extension_distribution'].items():
            pct = (count / total) * 100
            report += f"| {ext} | {count:,} | {pct:.1f}% |\n"
        
        report += f"""
## Class Distribution (Top 30)
| Class (Distributor/Source) | Files | Size (MB) |
|---------------------------|-------|-----------|
"""
        for i, (class_name, class_info) in enumerate(list(stats['classes'].items())[:30]):
            report += f"| {class_name[:50]} | {class_info['file_count']:,} | {class_info['total_size_mb']:.1f} |\n"
        
        # Class imbalance analysis
        file_counts = [c['file_count'] for c in stats['classes'].values()]
        report += f"""
## Class Imbalance Analysis
- **Total Classes:** {len(file_counts)}
- **Largest Class:** {max(file_counts):,} files
- **Smallest Class:** {min(file_counts):,} files
- **Median Class Size:** {int(np.median(file_counts)):,} files
- **Mean Class Size:** {int(np.mean(file_counts)):,} files
- **Std Dev:** {int(np.std(file_counts)):,} files

## Visualizations
- ![Class Distribution](class_distribution.png)
- ![Detailed Analysis](detailed_analysis.png)

## Sample Files
See `content_samples.json` for detailed content analysis of sample files from each class.

## Recommendations for Classification
1. **Class labels = Parent folder names** (distributors/sources)
2. Consider grouping small classes or using hierarchical classification
3. Heavy class imbalance - may need stratified sampling or class weights
4. Mix of CSV and PDF files - need robust parsers for both
"""
        
        with open(self.output_dir / 'EDA_REPORT.md', 'w') as f:
            f.write(report)
        
        logger.info(f"Generated EDA report at {self.output_dir / 'EDA_REPORT.md'}")
        return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="EDA for training data")
    parser.add_argument("--tar-path", type=str, 
                       default="/workspace/training_data/Copy of trainData.tgz",
                       help="Path to training tar archive")
    parser.add_argument("--output-dir", type=str,
                       default="/workspace/eda_output",
                       help="Output directory for EDA results")
    parser.add_argument("--max-files", type=int, default=None,
                       help="Maximum files to analyze (for testing)")
    parser.add_argument("--sample-content", action="store_true",
                       help="Sample and analyze file contents")
    
    args = parser.parse_args()
    
    eda = TrainingDataEDA(args.tar_path, args.output_dir)
    
    print("=" * 60)
    print("Training Data EDA Analysis")
    print("=" * 60)
    
    # Analyze structure
    eda.analyze_tar_structure(max_files=args.max_files)
    
    # Generate report
    stats = eda.generate_report()
    
    # Sample content if requested
    if args.sample_content:
        eda.sample_and_analyze_content(num_samples=30)
    
    print("\n" + "=" * 60)
    print("EDA Summary")
    print("=" * 60)
    print(f"Total Files: {stats['total_files']:,}")
    print(f"Total Size: {stats['total_size_gb']} GB")
    print(f"Number of Classes: {stats['num_classes']}")
    
    print(f"\nTop 10 Classes (by file count):")
    for i, (name, info) in enumerate(list(stats['classes'].items())[:10]):
        print(f"  {i+1}. {name}: {info['file_count']:,} files ({info['total_size_mb']:.1f} MB)")
    
    print(f"\nFile Types:")
    for ext, count in stats['extension_distribution'].items():
        print(f"  {ext}: {count:,}")
    
    print(f"\nOutput saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
