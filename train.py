import argparse
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src import data, loop

IN_CHANNELS = 5


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def _worker_init(worker_id: int):
    cv2.setNumThreads(0)

def build_model():
    return smp.Unet(
        encoder_name='timm-efficientnet-b3',
        encoder_weights='imagenet',
        in_channels=IN_CHANNELS,
        classes=1,
        activation=None,
        aux_params=dict(pooling='avg', dropout=0.2, classes=1),
    )


def lr_lambda(epoch, total_epochs, warmup_epochs):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description='Forensics segmentation: full training')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=Path('runs_full'))
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-epochs', type=int, default=2)
    parser.add_argument('--img-size', type=int, default=576)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--max-rows', type=int, default=None, help='сэмпл для смоука; None = весь датасет')
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--clean-fraction', type=float, default=0.30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-crops', action='store_true', help='обучение на нативных кропах 576 + гарантия захвата правок')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / 'best_model.pth'

    rows = data.read_train_csv(args.data_dir / 'train.csv', args.data_dir,
                               max_rows=args.max_rows, seed=args.seed)
    flags = data.compute_has_mask_flags(rows)
    tr_rows, tr_flags, val_rows, _ = data.stratified_split(rows, flags, args.val_ratio, args.seed)
    tr_rows = tr_rows + data.build_clean_example_rows(tr_rows, tr_flags, args.clean_fraction, args.seed)
    val_rows = val_rows + data.orgl_clean_rows(val_rows)
    n_val_clean = sum(1 for r in val_rows if r['gt'] is None)
    print(f'режим: {"кропы" if args.use_crops else "ресайз"} | train: {len(tr_rows)} | '
          f'val: {len(val_rows)} (clean: {n_val_clean})', flush=True)

    if args.use_crops:
        train_ds = data.SegCropDataset(tr_rows, crop_size=args.img_size,
                                       train=True, use_forensics=True)
    else:
        train_ds = data.SegDataset(tr_rows, args.img_size, train=True)
    val_ds = data.SegDataset(val_rows, args.img_size, train=False)

    worker_init = _worker_init if args.num_workers > 0 else None

    loader_kwargs = dict(num_workers=args.num_workers, pin_memory=True,
                         persistent_workers=args.num_workers > 0,
                         worker_init_fn=worker_init)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = build_model().to(device)
    loss_fn = loop.CombinedLoss(alpha=0.7, beta=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: lr_lambda(e, args.epochs, args.warmup_epochs))

    writer = SummaryWriter(str(out_dir / 'tb'))
    best_score = -float('inf')

    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs}, lr={optimizer.param_groups[0]["lr"]:.3e}', flush=True)
        train_logs = loop.train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch, writer)
        val_logs = loop.evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        score = 0.5 * val_logs['dice'] + 0.5 * val_logs['clean']
        writer.add_scalar('Val/dice', val_logs['dice'], epoch)
        writer.add_scalar('Val/clean', val_logs['clean'], epoch)
        writer.add_scalar('Val/loss', val_logs['loss'], epoch)
        writer.add_scalar('Metrics/Val_score', score, epoch)
        print(f"  train_loss={train_logs['loss']:.4f}  val_loss={val_logs['loss']:.4f}"
              f"dice={val_logs['dice']:.4f}  clean={val_logs['clean']:.4f}  score={score:.4f}", flush=True)

        if score > best_score:
            best_score = score
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch + 1, 'best_score': best_score}, ckpt_path)
            print(f'score = {best_score})', flush=True)

    writer.close()
    print(f'Best score: {best_score}', flush=True)


if __name__ == '__main__':
    main()