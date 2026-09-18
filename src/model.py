"""UNet (segmentation_models_pytorch) + an auxiliary image-level
"has-manipulation" classification head, using smp's own built-in `aux_params`
support (global-average-pool + linear probe on the encoder's deepest feature
map) rather than re-implementing the encoder/decoder plumbing ourselves.

in_channels defaults to 5 (RGB + ELA + high-pass). segmentation_models_pytorch
adapts the pretrained first conv for in_channels in {1,2,3,4} (reused/rescaled
imagenet weights) but randomly initializes it for in_channels > 4 -- the rest
of the encoder (the vast majority of parameters) stays imagenet-pretrained
regardless. Accepted trade-off: keeps the two forensic channels disentangled
instead of collapsing them into one to stay under that threshold.
"""
import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(encoder: str = 'resnet34', in_channels: int = 5,
                 encoder_weights: str = 'imagenet', decoder: str = 'Unet') -> nn.Module:
    """model(x) returns (mask_logits, aux_logits) -- see smp's aux_params.

    `decoder` names any smp segmentation head ('Unet', 'FPN', 'MAnet',
    'DeepLabV3Plus', ...). It defaults to 'Unet' so existing callers are
    unaffected; screen_configs.py uses it to compare architectures. All of these
    take the same constructor keywords, so no per-decoder special-casing is
    needed -- but they differ enormously in cost (UnetPlusPlus measured 187
    GFLOPs at 576px against a 100 GFLOPs budget), so check FLOPs before training.
    """
    try:
        decoder_cls = getattr(smp, decoder)
    except AttributeError as err:
        raise ValueError(f'unknown smp decoder {decoder!r}') from err
    return decoder_cls(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=1,
        activation=None,
        aux_params=dict(classes=1, pooling='avg', dropout=0.2, activation=None),
    )
