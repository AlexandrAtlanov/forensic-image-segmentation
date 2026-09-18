import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def use_amp(device) -> bool:
    return device.type == 'cuda' and torch.cuda.is_bf16_supported()


class CombinedLoss(torch.nn.Module):

    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, tversky_weight=0.5, bce_weight=0.5, aux_weight=1.0, eps=1e-6):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.tversky_weight, self.bce_weight, self.aux_weight = \
            tversky_weight, bce_weight, aux_weight
        self.eps = eps
        self.last = {}

    def forward(self, seg_logits, aux_logits, mask, has_manip):
        seg_logits = seg_logits.float()
        aux_logits = aux_logits.float()
        dims = (1, 2, 3)

        probs = torch.sigmoid(seg_logits)
        tp = (probs * mask).sum(dim=dims)
        fp = (probs * (1 - mask)).sum(dim=dims)
        fn = ((1 - probs) * mask).sum(dim=dims)
        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        ft = (1 - tversky) ** self.gamma
        ft = ft * (mask.sum(dim=dims) > 0).float()

        bce = F.binary_cross_entropy_with_logits(
            seg_logits, mask, reduction='none').mean(dim=dims)
        aux = F.binary_cross_entropy_with_logits(
            aux_logits.reshape(-1), has_manip.reshape(-1))

        total = (self.tversky_weight * ft + self.bce_weight * bce).mean() \
            + self.aux_weight * aux
        self.last = {
            'tversky': ft.mean().item(),
            'bce_seg': bce.mean().item(),
            'aux': aux.item(),
        }

        return total


def train_one_epoch(model, loader, loss_fn, optimizer, device, epoch, writer=None, log_every=50):
    model.train()
    total = 0.0
    n = 0

    for i, batch in enumerate(tqdm(loader, desc='train', leave=False)):
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        has_manip = batch['has_manip'].to(device, non_blocking=True)

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp(device)):
            seg_logits, aux_logits = model(images)
        loss = loss_fn(seg_logits, aux_logits, masks, has_manip)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        total += loss.item() * bs
        n += bs

        if writer is not None and (i % log_every == 0 or i == len(loader) - 1):
            step = epoch * len(loader) + i
            writer.add_scalar('Loss/step', loss.item(), step)
            for k, v in loss_fn.last.items():
                writer.add_scalar(f'Loss/{k}', v, step)

    return {'loss': total / n}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, n = 0.0, 0
    dices, cleans = [], []
    for batch in tqdm(loader, desc='valid', leave=False):
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        has_manip = batch['has_manip'].to(device, non_blocking=True)

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp(device)):
            seg_logits, aux_logits = model(images)
        loss = loss_fn(seg_logits, aux_logits, masks, has_manip)
        total += loss.item() * images.size(0)
        n += images.size(0)

        preds = (torch.sigmoid(seg_logits.float()) > 0.5).float()[:, 0]
        gt = masks[:, 0]
        pos = gt.sum(dim=(1, 2)) > 0

        if pos.any():
            p, g = preds[pos], gt[pos]
            tp = (p * g).sum(dim=(1, 2))
            fp = (p * (1 - g)).sum(dim=(1, 2))
            fn = ((1 - p) * g).sum(dim=(1, 2))
            dices.extend((2 * tp / (2 * tp + fp + fn + 1e-7)).tolist())
        if (~pos).any():
            cleans.extend((1.0 - preds[~pos].mean(dim=(1, 2))).tolist())

    return {
        'loss': total / n,
        'dice': float(sum(dices) / len(dices)) if dices else 0.0,
        'clean': float(sum(cleans) / len(cleans)) if cleans else 1.0,
    }