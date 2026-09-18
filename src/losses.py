"""Combined segmentation loss: Focal-Tversky + BCE on the mask, plus an
auxiliary BCE on the image-level "has-manipulation" classification head.

Tversky alpha multiplies false positives, beta multiplies false negatives.
alpha > beta makes the loss precision-leaning -- false positives on clean
images are penalized harder than missed pixels on edited ones, matching the
metric's "no false positives on clean images" half. (Careful: some public
Tversky-loss references use the opposite convention or swap which term is
"recall-leaning" -- this module always multiplies FP by alpha and FN by beta,
so don't copy alpha/beta values from elsewhere without checking that.)
"""
import torch
import torch.nn.functional as F

FOCAL_TVERSKY_ALPHA = 0.7   # FP weight -- precision-leaning
FOCAL_TVERSKY_BETA = 0.3    # FN weight
FOCAL_TVERSKY_GAMMA = 0.75  # gamma < 1 sharpens focus on hard/small examples
TVERSKY_EPS = 1e-6

TVERSKY_WEIGHT = 1.0
BCE_WEIGHT = 0.5
AUX_BCE_WEIGHT = 0.2


def focal_tversky_loss(logits: torch.Tensor, targets: torch.Tensor, pad_mask: torch.Tensor = None,
                        alpha: float = FOCAL_TVERSKY_ALPHA, beta: float = FOCAL_TVERSKY_BETA,
                        gamma: float = FOCAL_TVERSKY_GAMMA, eps: float = TVERSKY_EPS) -> torch.Tensor:
    """Per-image Focal-Tversky loss. logits/targets/pad_mask: (B, 1, H, W). Returns (B,).

    No 0/0 risk (eps > 0 in the denominator always), but NOT usable as-is for an
    all-zero-target (clean) image: FP = sum(probs) is a SUM over every real
    pixel, which stays >> eps at realistic resolutions even once the model has
    learned to predict near-zero everywhere (e.g. 262144 pixels x 1e-4 average
    probability = ~26, vs eps=1e-6) -- the tversky index then saturates near 0
    (loss near 1) with a vanishing gradient, regardless of prediction quality.
    Dice/Tversky-style overlap scores are fundamentally degenerate when there is
    no positive region to overlap with. Callers (see combined_loss) MUST zero
    out this term for images with no positive GT pixels and rely on BCE, which
    has no such degeneracy, to supervise those.
    """
    probs = torch.sigmoid(logits)
    if pad_mask is not None:
        probs = probs * pad_mask
        targets = targets * pad_mask
    dims = (1, 2, 3)
    tp = (probs * targets).sum(dim=dims)
    fp = (probs * (1 - targets)).sum(dim=dims)
    fn = ((1 - probs) * targets).sum(dim=dims)
    tversky_index = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1 - tversky_index) ** gamma


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
    """Per-image BCE, excluding letterbox-padded pixels from both the numerator
    and the denominator (not just zeroing their contribution) so the model
    isn't trained on the synthetic pad border at all. Returns (B,)."""
    per_pixel = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    if pad_mask is not None:
        per_pixel = per_pixel * pad_mask
        denom = pad_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        return per_pixel.sum(dim=(1, 2, 3)) / denom
    return per_pixel.mean(dim=(1, 2, 3))


def combined_loss(mask_logits: torch.Tensor, mask_targets: torch.Tensor, aux_logits: torch.Tensor,
                   aux_targets: torch.Tensor, sample_weight: torch.Tensor = None,
                   pad_mask: torch.Tensor = None) -> dict:
    """sample_weight (B,): per-image loss weight, default 1.0, reduced (e.g. 0.3)
    for nanobanana weak/pseudo-labeled rows -- this is the entire mechanism by
    which those rows are down-weighted, no other special-casing needed in the
    training loop. Only scales the mask losses; the aux "was this image edited"
    signal is left full-weight since it's reliable even from weak pairs.
    """
    ft = focal_tversky_loss(mask_logits, mask_targets, pad_mask=pad_mask)
    bce = masked_bce_loss(mask_logits, mask_targets, pad_mask=pad_mask)

    # Zero out the Tversky term for images with no positive GT pixels (clean
    # images) -- see focal_tversky_loss's docstring for why it's degenerate
    # there. Padded pixels are always 0 in mask_targets by construction, so
    # this sum already reflects only the real (non-padded) region.
    has_positive = (mask_targets.sum(dim=(1, 2, 3)) > 0).float()
    ft = ft * has_positive

    if sample_weight is not None:
        ft = ft * sample_weight
        bce = bce * sample_weight
    seg_loss = (TVERSKY_WEIGHT * ft + BCE_WEIGHT * bce).mean()
    aux_loss = F.binary_cross_entropy_with_logits(aux_logits.squeeze(1), aux_targets)
    total = seg_loss + AUX_BCE_WEIGHT * aux_loss
    return {
        'total': total,
        'tversky': ft.mean().detach(),
        'bce': bce.mean().detach(),
        'aux_bce': aux_loss.detach(),
    }
