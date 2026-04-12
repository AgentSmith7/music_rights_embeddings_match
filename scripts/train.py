#!/usr/bin/env python3
"""Training script for music rights document classifier."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.pipelines.training_pipeline import TrainingPipeline
from src.config.settings import get_config


def main():
    parser = argparse.ArgumentParser(description="Train the document classifier")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to training data (directory or tar file)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training_stats.json",
        help="Output path for training statistics"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting training pipeline")
    
    config = get_config()
    pipeline = TrainingPipeline(config)
    pipeline.initialize()
    
    source_path = Path(args.source)
    
    if source_path.is_dir():
        stats = pipeline.run_from_directory(str(source_path), max_files=args.max_files)
    elif source_path.suffix in ['.tgz', '.tar', '.gz']:
        stats = pipeline.run_from_tar(str(source_path), max_files=args.max_files)
    else:
        logger.error(f"Unsupported source type: {source_path}")
        sys.exit(1)
    
    final_stats = pipeline.get_stats()
    stats["final_stats"] = final_stats
    
    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Training complete. Stats saved to {args.output}")
    
    print("\n=== Training Summary ===")
    print(f"Total files processed: {stats['total_files']}")
    print(f"Total vectors created: {stats['total_vectors']}")
    print(f"Classes found: {len(stats['files_by_class'])}")
    print(f"Errors: {len(stats['errors'])}")
    
    print("\nFiles per class:")
    for cls, count in sorted(stats['files_by_class'].items()):
        print(f"  {cls}: {count}")


if __name__ == "__main__":
    main()
