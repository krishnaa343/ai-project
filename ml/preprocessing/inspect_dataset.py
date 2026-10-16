"""
Dataset Inspection & Quality Verification Script
Project: AI-Based Intelligent Crop Disease Detection Framework for Agriculture

This script programmatically inspects a local crop disease dataset:
- Scans directory tree and identifies class folders
- Counts total images and per-class distributions
- Validates file extensions and identifies non-image/unsupported files
- Detects corrupted or unreadable images using PIL
- Extracts image dimensions and color channel statistics
- Detects exact binary duplicates via cryptographic MD5 hashing
- Computes class balance statistics
- Outputs structured JSON metrics and formatted text summary
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict
from PIL import Image

# Supported standard image extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def compute_file_hash(filepath, chunk_size=65536):
    """Computes MD5 hash of a file for exact duplicate detection."""
    hasher = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None

def inspect_dataset(data_dir):
    """
    Performs comprehensive diagnostic inspection on dataset directory.
    
    Args:
        data_dir (str or Path): Path to raw dataset root containing class subdirectories.
    
    Returns:
        dict: Complete metrics and statistics dictionary.
    """
    root_path = Path(data_dir).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root_path}")

    # Discover candidate class directories
    class_dirs = [d for d in root_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    class_dirs.sort(key=lambda p: p.name)

    stats = {
        "dataset_path": str(root_path),
        "total_classes": len(class_dirs),
        "total_files_scanned": 0,
        "total_valid_images": 0,
        "total_corrupted_images": 0,
        "total_unsupported_files": 0,
        "total_duplicate_images": 0,
        "classes": {},
        "dimensions": defaultdict(int),
        "channel_modes": defaultdict(int),
        "file_extensions": defaultdict(int),
        "corrupted_files": [],
        "unsupported_files": [],
        "duplicate_groups": {}
    }

    seen_hashes = {} # hash -> original_filepath

    for class_dir in class_dirs:
        class_name = class_dir.name
        class_files = [f for f in class_dir.iterdir() if f.is_file()]
        
        valid_count = 0
        corrupted_count = 0

        for file_path in class_files:
            stats["total_files_scanned"] += 1
            ext = file_path.suffix.lower()
            stats["file_extensions"][ext] += 1

            # Check extension support
            if ext not in VALID_EXTENSIONS:
                stats["total_unsupported_files"] += 1
                stats["unsupported_files"].append(str(file_path))
                continue

            # Verify readability and integrity
            try:
                with Image.open(file_path) as img:
                    img.verify() # Verify file header / container integrity
                
                # Re-open for dimension and mode extraction (verify closes/invalidates the image)
                with Image.open(file_path) as img:
                    w, h = img.size
                    mode = img.mode
                    stats["dimensions"][f"{w}x{h}"] += 1
                    stats["channel_modes"][mode] += 1
                
                valid_count += 1
                stats["total_valid_images"] += 1

                # Check for exact duplicate via MD5
                file_hash = compute_file_hash(file_path)
                if file_hash:
                    if file_hash in seen_hashes:
                        stats["total_duplicate_images"] += 1
                        if file_hash not in stats["duplicate_groups"]:
                            stats["duplicate_groups"][file_hash] = [seen_hashes[file_hash]]
                        stats["duplicate_groups"][file_hash].append(str(file_path))
                    else:
                        seen_hashes[file_hash] = str(file_path)

            except Exception as e:
                corrupted_count += 1
                stats["total_corrupted_images"] += 1
                stats["corrupted_files"].append({"file": str(file_path), "error": str(e)})

        stats["classes"][class_name] = {
            "valid_images": valid_count,
            "corrupted_images": corrupted_count,
            "total_files": len(class_files)
        }

    # Convert defaultdicts to regular dicts for JSON serialization
    stats["dimensions"] = dict(stats["dimensions"])
    stats["channel_modes"] = dict(stats["channel_modes"])
    stats["file_extensions"] = dict(stats["file_extensions"])

    # Calculate class balance metrics
    counts = [data["valid_images"] for data in stats["classes"].values()]
    if counts:
        stats["min_class_count"] = min(counts)
        stats["max_class_count"] = max(counts)
        stats["mean_class_count"] = round(sum(counts) / len(counts), 2)
        stats["imbalance_ratio"] = round(max(counts) / max(1, min(counts)), 2)

    return stats

def print_summary_report(stats):
    """Prints a formatted human-readable terminal report."""
    print("=" * 80)
    print("DATASET INSPECTION REPORT")
    print("=" * 80)
    print(f"Dataset Path:           {stats['dataset_path']}")
    print(f"Total Classes Found:    {stats['total_classes']}")
    print(f"Total Files Scanned:    {stats['total_files_scanned']}")
    print(f"Total Valid Images:     {stats['total_valid_images']}")
    print(f"Corrupted Images:       {stats['total_corrupted_images']}")
    print(f"Unsupported Files:      {stats['total_unsupported_files']}")
    print(f"Duplicate Files:        {stats['total_duplicate_images']}")
    if "imbalance_ratio" in stats:
        print(f"Class Size Range:       Min: {stats['min_class_count']} | Max: {stats['max_class_count']} | Mean: {stats['mean_class_count']}")
        print(f"Imbalance Ratio:        {stats['imbalance_ratio']}:1")
    print("-" * 80)
    print("IMAGE FORMATS & MODES:")
    for ext, cnt in stats["file_extensions"].items():
        print(f"  Extension {ext or '[none]'}: {cnt}")
    for mode, cnt in stats["channel_modes"].items():
        print(f"  Color Mode {mode}: {cnt}")
    print("TOP IMAGE DIMENSIONS:")
    sorted_dims = sorted(stats["dimensions"].items(), key=lambda x: x[1], reverse=True)[:5]
    for dim, cnt in sorted_dims:
        print(f"  {dim}: {cnt} images")
    print("-" * 80)
    print(f"{'Class Name':<45} | {'Count':<8}")
    print("-" * 80)
    for cls_name, info in stats["classes"].items():
        print(f"{cls_name:<45} | {info['valid_images']:<8}")
    print("=" * 80)

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "ml/data/raw"
    try:
        report = inspect_dataset(target_dir)
        print_summary_report(report)
        
        # Save output JSON report
        out_json = Path(target_dir).parent / "dataset_inspection_report.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report written to: {out_json}")
    except Exception as exc:
        print(f"Inspection Error: {exc}", file=sys.stderr)
        sys.exit(1)
