import concurrent.futures
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from . import paths
from . import forensics

ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_train_csv(csv_path: Path, base_dir: Path, max_rows: int = None, seed: int = 42) -> list:
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        raw = list(csv.DictReader(f))
    if max_rows and len(raw) > max_rows:
        raw = random.Random(seed).sample(raw, max_rows)

    rows = []
    for row in raw:
        orgl_rel = (row.get('orgl_img_path') or '').strip()
        chng_rel = (row.get('chng_img_path') or '').strip()
        gt_rel = (row.get('gt_path') or '').strip()
        rows.append({
            'chng': paths.resolve_data_path(chng_rel, base_dir),
            'gt': paths.resolve_data_path(gt_rel, base_dir),
            'orgl': paths.resolve_data_path(orgl_rel, base_dir) if orgl_rel else None,
            'is_synthetic_clean': False,
            'is_pseudo_label': False,
        })

    return rows


def read_test_csv(csv_path: Path, base_dir: Path) -> list:
    rows = []
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            img_path = row.get('img_path')
            if img_path is None:
                img_path = row.get('chng_img_path')
            if img_path is None:
                raise ValueError(f'test csv must contain "img_path" (or legacy "chng_img_path"), got: {row.keys()}')
            rel = img_path.strip()
            rows.append({'chng': paths.resolve_data_path(rel, base_dir), 'chng_rel': rel})

    return rows

def compute_has_mask_flags(rows: list, max_workers: int = 8) -> list:
    def _flag(row):
        if row.get('gt') is None:
            return False
        m = cv2.imread(str(row['gt']), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Could not read mask: {row['gt']}")
        
        return bool(m.max() > 128)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_flag, rows))


def stratified_split(rows: list, flags: list, val_ratio: float, seed: int) -> tuple:
    assert len(rows) == len(flags)
    pos_idx = [i for i, f in enumerate(flags) if f]
    neg_idx = [i for i, f in enumerate(flags) if not f]
    rng = random.Random(seed)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    n_val_pos = max(1, int(len(pos_idx) * val_ratio)) if pos_idx else 0
    n_val_neg = max(1, int(len(neg_idx) * val_ratio)) if neg_idx else 0
    val_idx = set(pos_idx[:n_val_pos]) | set(neg_idx[:n_val_neg])
    train_rows = [rows[i] for i in range(len(rows)) if i not in val_idx]
    train_flags = [flags[i] for i in range(len(rows)) if i not in val_idx]
    val_rows = [rows[i] for i in range(len(rows)) if i in val_idx]
    val_flags = [flags[i] for i in range(len(rows)) if i in val_idx]

    return train_rows, train_flags, val_rows, val_flags


def build_clean_example_rows(train_rows: list, train_flags: list, target_clean_fraction: float,
                              seed: int) -> list:
    assert len(train_rows) == len(train_flags)
    if target_clean_fraction <= 0 or target_clean_fraction >= 1:
        return []
    
    already_clean = sum(1 for f in train_flags if not f)
    already_manip = len(train_flags) - already_clean
    target_total_clean = target_clean_fraction * already_manip / (1 - target_clean_fraction)
    n_extra_needed = max(0, round(target_total_clean) - already_clean)
    candidates = [r for r in train_rows if r.get('orgl') is not None]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    chosen = candidates[:n_extra_needed]

    return [
        {'chng': r['orgl'], 'gt': None, 'orgl': None,
         'is_synthetic_clean': True, 'is_pseudo_label': False}
        for r in chosen
    ]

def orgl_clean_rows(rows: list) -> list:
    return [
        {'chng': r['orgl'], 'gt': None, 'orgl': None, 'is_synthetic_clean': True, 'is_pseudo_label': False} 
        for r in rows if r.get('orgl') is not None
    ]

def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert('RGB'))


def _load_real_mask(path: Path) -> np.ndarray:
    m = np.asarray(Image.open(path))
    if m.ndim == 3:
        m = m[:, :, 0]

    return (m > 128).astype(np.float32)

def stack_channels(img_rgb: np.ndarray, ela: np.ndarray, hp: np.ndarray) -> np.ndarray:
    rgb = img_rgb.astype(np.float32) / 255.0

    return np.concatenate([rgb, ela[:, :, None], hp[:, :, None]], axis=2)


def load_image5_ch(row: dict, img_size: int, use_forensics: bool = True, flip_h: bool = False, flip_v: bool = False) -> tuple:
    img = _load_rgb(row['chng'])
    if flip_h:
        img = np.ascontiguousarray(img[:, ::-1, :])
    if flip_v:
        img = np.ascontiguousarray(img[::-1, :, :])
    h, w = img.shape[:2]

    if use_forensics:
        ela = cv2.resize(forensics.compute_ela(img), (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        hp = cv2.resize(forensics.compute_highpass(img), (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    else:
        ela = np.zeros((img_size, img_size), dtype=np.float32)
        hp = np.zeros((img_size, img_size), dtype=np.float32)

    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    stacked = stack_channels(img, ela, hp)
    image = torch.from_numpy(stacked.transpose(2, 0, 1)).float()

    return image, h, w

def load_gt_mask_for_row(row: dict) -> np.ndarray:
    img = _load_rgb(row['chng'])
    if row.get('gt') is not None:
        mask = _load_real_mask(row['gt'])
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask
    
    return np.zeros(img.shape[:2], dtype=np.float32)


class SegDataset(Dataset):
    """Информация батча:
      image (5, S, S) float32 [0..1]
      mask (1, S, S) float32 {0, 1}
      has_manip float32 (0/1) — по нативной маске, до ресайза
      orig_h, orig_w — int, нативный размер
        train=True включает рандомные флипы
    """

    def __init__(self, rows: list, img_size: int, train: bool = False, use_forensics: bool = True):
        self.rows = rows 
        self.img_size = img_size
        self.train = train
        self.use_forensics = use_forensics

    def __len__(self) -> int:
        return len(self.rows)

    def _load_mask(self, row: dict, img_hw: tuple) -> np.ndarray:
        if row['gt'] is None:
            return np.zeros(img_hw, dtype=np.float32)

        mask = _load_real_mask(row['gt'])
        if mask.shape[:2] != img_hw:
            mask = cv2.resize(mask, (img_hw[1], img_hw[0]), interpolation=cv2.INTER_NEAREST)

        return mask

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        flip_h = self.train and (random.random() < 0.5)
        flip_v = self.train and (random.random() < 0.5)

        image, h, w = load_image5_ch(row, self.img_size, self.use_forensics, flip_h, flip_v)
        mask = self._load_mask(row, (h, w))

        if flip_h:
            mask = np.ascontiguousarray(mask[:, ::-1])

        if flip_v:
            mask = np.ascontiguousarray(mask[::-1, :])

        has_manip = float(mask.sum() > 0)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        return {
            'image': image,
            'mask': torch.from_numpy(mask).float().unsqueeze(0),
            'has_manip': torch.tensor(has_manip, dtype=torch.float32),
            'orig_h': h, 'orig_w': w,
        }

class SegCropDataset(Dataset):
    def __init__(self, rows: list, crop_size: int = 576, train: bool = True,
                 use_forensics: bool = True):
        self.rows = rows
        self.crop_size = crop_size
        self.train = train
        self.use_forensics = use_forensics

    def __len__(self) -> int:
        return len(self.rows)

    def _sample_window(self, h: int, w: int, mask: np.ndarray) -> tuple:
        cs = self.crop_size
        if h <= cs and w <= cs:
            return 0, 0, max(h, cs), max(w, cs)
        ys = np.random.randint(0, max(1, h - cs + 1)) if h > cs else 0
        xs = np.random.randint(0, max(1, w - cs + 1)) if w > cs else 0
        if mask is not None and mask.sum() > 0:
            for _ in range(20):
                my, mx = np.argwhere(mask > 0)[np.random.randint(len(np.argwhere(mask > 0)))]
                if my >= ys and my < ys + cs and mx >= xs and mx < xs + cs:
                    break
                ys = int(min(max(0, my - cs // 2), max(0, h - cs)))
                xs = int(min(max(0, mx - cs // 2), max(0, w - cs)))

        return ys, xs, h, w

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        img = _load_rgb(row['chng'])
        h, w = img.shape[:2]
        mask = self._load_mask(row, (h, w))
        has_manip = float(mask.sum() > 0)

        if self.train:
            ys, xs, hh, ww = self._sample_window(h, w, mask)
            pad_h, pad_w = max(0, self.crop_size - h), max(0, self.crop_size - w)
            img_np = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0))) if pad_h or pad_w else img
            mask_np = np.pad(mask, ((0, pad_h), (0, pad_w))) if pad_h or pad_w else mask
            img_c = img_np[ys:ys + self.crop_size, xs:xs + self.crop_size]
            mask_c = mask_np[ys:ys + self.crop_size, xs:xs + self.crop_size]
        else:
            img_c, mask_c = img, mask

        if self.use_forensics:
            ela = forensics.compute_ela(img_c)
            hp = forensics.compute_highpass(img_c)
        else:
            ela = np.zeros(img_c.shape[:2], dtype=np.float32)
            hp = np.zeros(img_c.shape[:2], dtype=np.float32)

        stacked = stack_channels(img_c, ela, hp)
        image = torch.from_numpy(stacked.transpose(2, 0, 1)).float()
        mask_r = cv2.resize(mask_c, (self.crop_size, self.crop_size),
                            interpolation=cv2.INTER_NEAREST) \
            if mask_c.shape[:2] != (self.crop_size, self.crop_size) else mask_c

        return {
            'image': image,
            'mask': torch.from_numpy(np.ascontiguousarray(mask_r)).float().unsqueeze(0),
            'has_manip': torch.tensor(has_manip, dtype=torch.float32),
            'orig_h': h, 'orig_w': w,
        }

    def _load_mask(self, row: dict, img_hw: tuple) -> np.ndarray:
        if row['gt'] is None:
            return np.zeros(img_hw, dtype=np.float32)
        
        m = _load_real_mask(row['gt'])
        if m.shape[:2] != img_hw:
            m = cv2.resize(m, (img_hw[1], img_hw[0]), interpolation=cv2.INTER_NEAREST)
            
        return m

class TestDataset(Dataset):
    def __init__(self, rows: list, img_size: int, use_forensics: bool = True):
        self.rows = rows
        self.img_size = img_size
        self.use_forensics = use_forensics

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        image, h, w = load_image5_ch(row, self.img_size, self.use_forensics)

        return {
            'image': image,
            'chng_rel': row['chng_rel'],
            'orig_h': h,
            'orig_w': w
        }