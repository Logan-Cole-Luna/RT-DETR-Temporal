#!/usr/bin/env python3
"""
Prepare Anti-UAV dataset (v7) with PyTorch-accelerated paper-style temporal high-pass motion filter, sample dump, and subset by percentage option.

For each sequence folder, this:
  • slides a window of up to SEQ_LEN frames (handles <SEQ_LEN sequences),
  • reads frames in grayscale (fast I/O),
  • computes a 1-channel motion map via PyTorch,
  • stacks motion and last-frame gray into a 3-channel PNG (dummy R=0),
  • writes the corresponding YOLO-style bbox for the last frame,
  • shows a tqdm progress bar (green for train, blue for val),
  • dumps a sample of the first valid window (last frame, motion map, combined), and
  • records the sample paths in sequences/{split}.txt.

Configure SUBSET_PCT to limit percentage of sequences processed for quick testing (e.g. 10 for 10%).
"""
import os
import json
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm

# --- Configuration ---
INPUT_TRAIN_DIR = r"C:\Users\Logan\Downloads\train_uav\train"
OUTPUT_ROOT     = r"C:\data\processed_anti_uav_v1"

SEQ_LEN      = 5               # frames per window
IMG_SIZE     = (640, 512)      # (width, height)
VAL_RATIO    = 0.2             # fraction for validation split
SPLIT_MODE   = 'train_val_test'# 'train_val' or 'train_val_test'
TEST_RATIO   = 0.1             # fraction for test split in train_val_test mode
IMAGE_EXTS   = {".jpg", ".jpeg", ".png"}
SUBSET_PCT   = 5            # int percentage of sequences to process for quick testing

# PyTorch device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Flag to dump sample only once
SAMPLE_DUMPED = False


def compute_motion_map(frames_gray: list[np.ndarray]) -> np.ndarray:
    """
    Bio-inspired temporal high-pass filter implemented in PyTorch:
      - Stack grayscale frames [F(t-k),...,F(t)]
      - Compute frame-to-frame abs differences
      - Average to obtain motion strength
    """
    tensors = [torch.from_numpy(f).to(DEVICE, dtype=torch.float32) for f in frames_gray]
    diffs = [torch.abs(tensors[i] - tensors[i-1]) for i in range(1, len(tensors))]
    motion = torch.stack(diffs, dim=0).mean(dim=0)
    motion = motion.clamp(0, 255).to(torch.uint8)
    return motion.cpu().numpy()


def process_sequence_folder(seq_dir: Path, split: str) -> list[str]:
    global SAMPLE_DUMPED
    seq_name = seq_dir.name
    img_out_dir = Path(OUTPUT_ROOT) / "images" / split
    lbl_out_dir = Path(OUTPUT_ROOT) / "labels" / split
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    label_file = seq_dir / "IR_label.json"
    raw = json.load(open(label_file)).get('gt_rect', []) if label_file.exists() else []

    # Gather and read grayscale frames once
    img_files = sorted([p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    grays = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in img_files]
    total = len(grays)

    out_lines = []
    # Determine window size (<= SEQ_LEN)
    window_size = SEQ_LEN if total >= SEQ_LEN else total
    starts = list(range(total - window_size + 1)) if total > 0 else []

    for idx in starts:
        window = grays[idx:idx+window_size]
        if any(f is None for f in window):
            continue

        motion = compute_motion_map(window)
        last_frame = window[-1]

        last_r = cv2.resize(last_frame, IMG_SIZE)
        mot_r  = cv2.resize(motion, IMG_SIZE)
        zero = np.zeros_like(last_r, dtype=np.uint8)
        combined = np.stack([last_r, mot_r, zero], axis=-1)

        # Dump sample on first full window
        if not SAMPLE_DUMPED and window_size >= SEQ_LEN:
            sample_dir = Path(OUTPUT_ROOT) / 'sample'
            sample_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(sample_dir / 'sample_last.png'), last_r)
            cv2.imwrite(str(sample_dir / 'sample_motion.png'), mot_r)
            cv2.imwrite(str(sample_dir / 'sample_combined.png'), combined)
            SAMPLE_DUMPED = True

        name = f"{seq_name}_{idx:05d}.png"
        cv2.imwrite(str(img_out_dir / name), combined)

        # YOLO label for last frame
        lbl_idx = idx + window_size - 1
        cx = cy = wn = hn = 0.0
        if lbl_idx < len(raw) and isinstance(raw[lbl_idx], (list, tuple)) and len(raw[lbl_idx]) == 4:
            x, y, w, h = raw[lbl_idx]
            h0, w0 = window[-1].shape[:2]
            sx, sy = IMG_SIZE[0] / w0, IMG_SIZE[1] / h0
            x, w = x * sx, w * sx
            y, h = y * sy, h * sy
            cx = (x + w/2) / IMG_SIZE[0]
            cy = (y + h/2) / IMG_SIZE[1]
            wn = w / IMG_SIZE[0]
            hn = h / IMG_SIZE[1]
            cx, cy, wn, hn = np.clip([cx, cy, wn, hn], 0, 1)

        lbl_path = lbl_out_dir / f"{seq_name}_{idx:05d}.txt"
        lbl_path.write_text(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}\n")
        out_lines.append(f"images/{split}/{name}")
    return out_lines


def main():
    random.seed(42)
    # List only directories under INPUT_TRAIN_DIR
    seq_dirs = [d for d in Path(INPUT_TRAIN_DIR).iterdir() if d.is_dir()]
    if SUBSET_PCT is not None:
        count = max(1, int(len(seq_dirs) * SUBSET_PCT / 100))
        seq_dirs = seq_dirs[:count]
    random.shuffle(seq_dirs)
    n = len(seq_dirs)

    if SPLIT_MODE == 'train_val_test':
        n_test = int(n * TEST_RATIO)
        rem = n - n_test
        n_val = int(rem * VAL_RATIO)
        splits = {
            'train': seq_dirs[n_test + n_val:],
            'val':   seq_dirs[n_test:n_test + n_val],
            'test':  seq_dirs[:n_test]
        }
    else:
        idx = int(n * (1 - VAL_RATIO))
        splits = {'train': seq_dirs[:idx], 'val': seq_dirs[idx:]}

    results = {}
    for split, dirs in splits.items():
        colour = 'green' if split == 'train' else 'blue'
        print(f"[INFO] {split.upper()} ({len(dirs)} seqs) DEVICE={DEVICE}")
        out = []
        for d in tqdm(dirs, desc=split, colour=colour, unit='seq'):
            out += process_sequence_folder(d, split)
        results[split] = out

    seq_root = Path(OUTPUT_ROOT) / 'sequences'
    seq_root.mkdir(parents=True, exist_ok=True)
    for split, lines in results.items():
        path = seq_root / f"{split}.txt"
        path.write_text("\n".join(lines))
        print(f"[INFO] Wrote {len(lines)} entries to {path}")
    print("[INFO] All done.")

if __name__ == '__main__':
    main()
