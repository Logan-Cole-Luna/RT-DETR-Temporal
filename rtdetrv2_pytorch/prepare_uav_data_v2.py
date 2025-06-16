#!/usr/bin/env python3
"""
Prepare Anti-UAV dataset (v2)

Dataset layout:
  train/
    <sequence_folder>/
      *.jpg (or *.png)
      IR_label.json
  test/
    <sequence_folder>/
      *.jpg (or *.png)
      IR_label.json

Produces:
  output_root/images/{train,val,test}/
  output_root/labels/{train,val,test}/
  output_root/sequences/{train,val,test}.txt

Splits train sequences into train/val by VAL_RATIO.
"""
import os
import json
import random
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import cv2

# --- Configuration ---
INPUT_TRAIN_DIR = r"C:\Users\Logan\Downloads\train_uav\train"
INPUT_TEST_DIR  = None
OUTPUT_ROOT     = r"C:\data\processed_anti_uav_v2"

SEQ_LEN   = 5       # number of frames per sequence
VAL_RATIO = 0.2     # fraction of training sequences to use for validation
SPLIT_MODE = 'train_val_test'      # 'train_val' or 'train_val_test'
TEST_RATIO = 0.1              # fraction of training sequences to reserve for test when SPLIT_MODE='train_val_test'
IMG_SIZE  = (640, 512)
WORKERS   = os.cpu_count()

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def process_sequence_folder(args):
    seq_dir, split, output_root, seq_len, img_size = args
    seq_name = seq_dir.name
    out_img_dir = Path(output_root) / "images" / split
    out_lbl_dir = Path(output_root) / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # load annotations
    label_file = seq_dir / "IR_label.json"
    if not label_file.exists():
        print(f"[WARN] Missing label file: {label_file}")
        raw = []
    else:
        with open(label_file, 'r') as f:
            raw = json.load(f).get("gt_rect", [])

    # list and sort image files
    img_files = sorted([
        p for p in seq_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    ])

    frames = []
    proc = 0
    # process each image
    for idx, img_path in enumerate(img_files):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h0, w0 = frame.shape[:2]
        # resize
        im = cv2.resize(frame, img_size)
        fname = f"{seq_name}_{proc:05d}.jpg"
        cv2.imwrite(str(out_img_dir / fname), im)

        # annotation
        if proc < len(raw) and isinstance(raw[proc], (list, tuple)) and len(raw[proc]) == 4:
            x, y, w, h = raw[proc]
            sx = img_size[0] / w0
            sy = img_size[1] / h0
            x *= sx; w *= sx
            y *= sy; h *= sy
        else:
            x = y = w = h = 0

        lbl_path = out_lbl_dir / f"{seq_name}_{proc:05d}.txt"
        if w > 0 and h > 0:
            cx = (x + w/2) / img_size[0]
            cy = (y + h/2) / img_size[1]
            wn = w / img_size[0]
            hn = h / img_size[1]
            cx = min(max(cx,0),1)
            cy = min(max(cy,0),1)
            wn = min(max(wn,0),1)
            hn = min(max(hn,0),1)
            lbl_path.write_text(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}\n")
        else:
            lbl_path.touch()

        frames.append(f"images/{split}/{fname}")
        proc += 1
    
    # generate sequence lines
    seq_lines = []
    for i in range(max(0, len(frames) - seq_len + 1)):
        seq_lines.append(" ".join(frames[i:i+seq_len]))
    return seq_lines


def main():
    random.seed(42)
    output_root = Path(OUTPUT_ROOT)
    # Prepare split directories mapping
    splits = {}
    # TRAIN splitting
    if INPUT_TRAIN_DIR:
        # unzip if needed
        if INPUT_TRAIN_DIR.lower().endswith('.zip'):
            import zipfile
            tmp = output_root / 'tmp_train'
            if tmp.exists(): shutil.rmtree(tmp)
            with zipfile.ZipFile(INPUT_TRAIN_DIR, 'r') as z: z.extractall(tmp)
            root = tmp
        else:
            root = Path(INPUT_TRAIN_DIR)
        seq_dirs = [d for d in root.iterdir() if d.is_dir()]
        random.shuffle(seq_dirs)
        n_total = len(seq_dirs)
        if SPLIT_MODE == 'train_val_test':
            # split into test, val, train from training data
            n_test = int(n_total * TEST_RATIO)
            n_rem  = n_total - n_test
            n_val  = int(n_rem * VAL_RATIO)
            splits['test']  = seq_dirs[:n_test]
            splits['val']   = seq_dirs[n_test:n_test + n_val]
            splits['train'] = seq_dirs[n_test + n_val:]
        else:
            # only train/val split
            split_idx = int(n_total * (1 - VAL_RATIO))
            splits['train'] = seq_dirs[:split_idx]
            splits['val']   = seq_dirs[split_idx:]
    # external TEST folder (only when using train_val mode)
    if SPLIT_MODE == 'train_val' and INPUT_TEST_DIR:
        splits['test'] = [d for d in Path(INPUT_TEST_DIR).iterdir() if d.is_dir()]
    
    # Skip if no splits defined
    if not splits:
        print("[WARN] No input splits provided. Nothing to do.")
        return

    # Consolidate processing
    tasks = []
    for split_name, dirs in splits.items():
        print(f"[INFO] Preparing {split_name}: {len(dirs)} sequences")
        for d in dirs:
            tasks.append((d, split_name, output_root, SEQ_LEN, IMG_SIZE))

    # Process all sequences in a single pool
    results = {k: [] for k in splits}
    with ProcessPoolExecutor(max_workers=WORKERS) as exe:
        for args, seq_lines in zip(tasks, exe.map(process_sequence_folder, tasks)):
            _, split_name, *_ = args
            results[split_name].extend(seq_lines)

    # Write sequence files
    for split_name, lines in results.items():
        seq_file = output_root / 'sequences' / f"{split_name}.txt"
        seq_file.parent.mkdir(parents=True, exist_ok=True)
        seq_file.write_text("\n".join(lines))
        print(f"  → Wrote {len(lines)} sequences to {seq_file}")

    print("[INFO] All splits processed!")


if __name__ == "__main__":
    main()
