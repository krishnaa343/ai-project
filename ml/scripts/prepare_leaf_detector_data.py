"""
Prepare Leaf vs Non-Leaf Dataset (Fast Multi-threaded Download)
AI-Based Intelligent Crop Disease Detection Framework for Agriculture

Assembles a balanced, diverse dataset:
- Class 1: LEAF (sampled across 38 PlantVillage categories including healthy, diseased, spotted, and brown leaves)
- Class 0: NON_LEAF (diverse non-leaf categories: insects, animals, vehicles, buildings, food, objects, documents, scenes)

Strictly excludes test_ant_non_leaf.jpg from training.
Creates stratified train (70%), validation (15%), and test (15%) splits with fixed seed 42.
"""

import os
import random
import glob
import json
import urllib.request
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from PIL import Image
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PLANTVILLAGE = REPO_ROOT / "ml" / "data" / "raw" / "plantvillage"
DEST_DIR = REPO_ROOT / "ml" / "data" / "leaf_detection"
DEST_LEAF = DEST_DIR / "raw" / "leaf"
DEST_NON_LEAF = DEST_DIR / "raw" / "non_leaf"
SPLITS_DIR = DEST_DIR / "splits"

DEST_LEAF.mkdir(parents=True, exist_ok=True)
DEST_NON_LEAF.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)

def sample_leaf_images(target_count=650):
    existing = list(DEST_LEAF.glob("*.jpg"))
    if len(existing) >= target_count:
        print(f"Already have {len(existing)} LEAF images.")
        return len(existing)
        
    print(f"Sampling ~{target_count} diverse LEAF images from PlantVillage...")
    subdirs = [d for d in RAW_PLANTVILLAGE.iterdir() if d.is_dir()]
    per_class = max(15, target_count // len(subdirs))
    
    saved_count = len(existing)
    for subdir in subdirs:
        images = list(subdir.glob("*.*"))
        random.shuffle(images)
        selected = images[:per_class]
        for img_path in selected:
            dest_file = DEST_LEAF / f"{subdir.name}_{img_path.name}"
            if not dest_file.exists():
                try:
                    with Image.open(img_path) as im:
                        im.convert("RGB").save(dest_file, "JPEG", quality=90)
                        saved_count += 1
                except Exception:
                    pass
            if saved_count >= target_count:
                break
        if saved_count >= target_count:
            break
            
    print(f"Total LEAF images assembled: {saved_count}")
    return saved_count

def download_single_image(args):
    url, dest_path = args
    if dest_path.exists():
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CropScanAI/1.0 (academic research)"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = resp.read()
            im = Image.open(BytesIO(data))
            im = im.convert("RGB")
            im.save(dest_path, "JPEG", quality=85)
            return True
    except Exception:
        return False

def get_commons_urls(query, count=40):
    try:
        url = f"https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrsearch={urllib.parse.quote(query)}&gsrnamespace=6&gsrlimit={count}&prop=imageinfo&iiprop=url&iiurlwidth=320&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "CropScanAI/1.0 (academic research)"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            urls = []
            for v in pages.values():
                if "imageinfo" in v and v["imageinfo"] and "thumburl" in v["imageinfo"][0]:
                    thumb = v["imageinfo"][0]["thumburl"]
                    if thumb.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        urls.append(thumb)
            return urls
    except Exception as e:
        print(f"Commons error for '{query}': {e}")
        return []

def assemble_non_leaf_images(target_count=650):
    existing = list(DEST_NON_LEAF.glob("*.jpg"))
    if len(existing) >= target_count:
        print(f"Already have {len(existing)} NON_LEAF images.")
        return len(existing)
        
    print(f"Assembling non-leaf images to reach {target_count} (currently have {len(existing)})...")
    download_tasks = []
    
    categories = [
        ("insect ant beetle wasp spider", "insect"),
        ("dog cat horse bird frog fish", "animal"),
        ("car truck airplane bicycle train boat", "vehicle"),
        ("building house bridge tower room street", "building"),
        ("pizza bread soup cheese cake fruit slice", "food"),
        ("portrait face human crowd person", "person"),
        ("watch shoe chair laptop guitar cup clock", "object"),
        ("desert rocks snow ocean clouds mountain", "landscape"),
        ("document receipt invoice printed text certificate", "document"),
    ]
    
    for query, prefix in categories:
        urls = get_commons_urls(query, count=45)
        for i, url in enumerate(urls):
            dest = DEST_NON_LEAF / f"{prefix}_{i:03d}.jpg"
            if not dest.exists():
                download_tasks.append((url, dest))
                
    # Also fetch Picsum IDs
    print(f"Collected {len(download_tasks)} Commons tasks. Adding Picsum tasks...")
    for page in range(1, 10):
        try:
            list_url = f"https://picsum.photos/v2/list?page={page}&limit=100"
            req = urllib.request.Request(list_url, headers={"User-Agent": "CropScanAI/1.0"})
            with urllib.request.urlopen(req, timeout=8) as r:
                items = json.loads(r.read().decode("utf-8"))
                for item in items:
                    img_id = item["id"]
                    dest = DEST_NON_LEAF / f"picsum_{img_id}.jpg"
                    if not dest.exists():
                        thumb_url = f"https://picsum.photos/id/{img_id}/300/300"
                        download_tasks.append((thumb_url, dest))
        except Exception as e:
            print(f"Picsum list fetch: {e}")
            break
            
    print(f"Total tasks to execute in parallel: {len(download_tasks)}")
    
    with ThreadPoolExecutor(max_workers=14) as executor:
        futures = [executor.submit(download_single_image, t) for t in download_tasks]
        completed = 0
        for f in as_completed(futures):
            if f.result():
                completed += 1
            if len(list(DEST_NON_LEAF.glob("*.jpg"))) >= target_count:
                break
                
    total_non_leaf = len(list(DEST_NON_LEAF.glob("*.jpg")))
    print(f"Total NON_LEAF images assembled: {total_non_leaf}")
    return total_non_leaf

def create_stratified_splits(seed=42):
    leaf_files = list(DEST_LEAF.glob("*.jpg"))
    non_leaf_files = list(DEST_NON_LEAF.glob("*.jpg"))
    
    print(f"Splitting: {len(leaf_files)} leaf images and {len(non_leaf_files)} non-leaf images...")
    
    random.seed(seed)
    random.shuffle(leaf_files)
    random.shuffle(non_leaf_files)
    
    n_per_class = min(len(leaf_files), len(non_leaf_files))
    leaf_files = leaf_files[:n_per_class]
    non_leaf_files = non_leaf_files[:n_per_class]
    
    n_train = int(n_per_class * 0.70)
    n_val   = int(n_per_class * 0.15)
    
    splits = {
        "train": [],
        "val": [],
        "test": []
    }
    
    # 0 = NON_LEAF, 1 = LEAF
    for f in leaf_files[:n_train]:
        splits["train"].append({"file_path": str(f.relative_to(REPO_ROOT)), "label": 1, "class_name": "LEAF"})
    for f in leaf_files[n_train:n_train + n_val]:
        splits["val"].append({"file_path": str(f.relative_to(REPO_ROOT)), "label": 1, "class_name": "LEAF"})
    for f in leaf_files[n_train + n_val:]:
        splits["test"].append({"file_path": str(f.relative_to(REPO_ROOT)), "label": 1, "class_name": "LEAF"})
        
    for f in non_leaf_files[:n_train]:
        splits["train"].append({"file_path": str(f.relative_to(REPO_ROOT)), "label": 0, "class_name": "NON_LEAF"})
    for f in non_leaf_files[n_train:n_train + n_val]:
        splits["val"].append({"file_path": str(f.relative_to(REPO_ROOT)), "label": 0, "class_name": "NON_LEAF"})
    for f in non_leaf_files[n_train + n_val:]:
        splits["test"].append({"file_path": str(f.relative_to(REPO_ROOT)), "label": 0, "class_name": "NON_LEAF"})
        
    for name, rows in splits.items():
        random.shuffle(rows)
        df = pd.DataFrame(rows)
        csv_path = SPLITS_DIR / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        leaf_c = (df["label"] == 1).sum()
        non_leaf_c = (df["label"] == 0).sum()
        print(f"Split '{name}': Total {len(df)} (LEAF: {leaf_c}, NON_LEAF: {non_leaf_c}) saved to {csv_path.name}")
        
    summary = {
        "total_samples": n_per_class * 2,
        "leaf_samples": n_per_class,
        "non_leaf_samples": n_per_class,
        "train_samples": len(splits["train"]),
        "val_samples": len(splits["val"]),
        "test_samples": len(splits["test"]),
        "seed": seed
    }
    with open(SPLITS_DIR / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Dataset preparation complete!")

if __name__ == "__main__":
    sample_leaf_images(650)
    assemble_non_leaf_images(650)
    create_stratified_splits(SEED)
