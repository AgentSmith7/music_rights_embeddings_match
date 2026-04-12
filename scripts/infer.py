#!/usr/bin/env python3
"""Inference script for classifying documents."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.pipelines.inference_pipeline import InferencePipeline
from src.config.settings import get_config


def main():
    parser = argparse.ArgumentParser(description="Classify documents using trained model")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input (ZIP file or directory)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="classification_results.json",
        help="Output path for classification results"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting inference pipeline")
    
    config = get_config()
    pipeline = InferencePipeline(config)
    pipeline.initialize()
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        results = pipeline.run_on_directory(str(input_path), output_path=args.output)
    elif input_path.suffix == '.zip':
        results = pipeline.run_on_zip(str(input_path), output_path=args.output)
    else:
        logger.error(f"Unsupported input type: {input_path}")
        sys.exit(1)
    
    summary = pipeline.get_summary(results)
    
    print("\n=== Classification Summary ===")
    print(f"Total files: {summary['total_files']}")
    print(f"Needs review: {summary['needs_review_count']}")
    print(f"Errors: {summary['error_count']}")
    print(f"Average confidence: {summary['average_confidence']:.3f}")
    
    print("\nClass distribution:")
    for cls, count in sorted(summary['class_distribution'].items()):
        print(f"  {cls}: {count}")


if __name__ == "__main__":
    main()
