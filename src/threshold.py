import cv2
import numpy as np
from itertools import product


def postprocess_mask(prob_map, prob_threshold=0.5, min_area=20, reject_threshold=0.0, aux_prob=None, gate_threshold=0.0):
    
    if gate_threshold > 0 and aux_prob is not None and aux_prob < gate_threshold:
        return np.zeros_like(binary := (prob_map > prob_threshold).astype(np.uint8))

    binary = (prob_map > prob_threshold).astype(np.uint8)

    if reject_threshold > 0 and binary.sum() > 0:
        flat_probs = prob_map.flatten()
        k = max(1, int(len(flat_probs) * 0.001))
        top_k_avg = np.mean(np.sort(flat_probs)[-k:])

        if top_k_avg < reject_threshold:
            return np.zeros_like(binary)

    if min_area > 0 and binary.sum() > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        cleaned = np.zeros_like(binary)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = 1
        binary = cleaned

    return binary


def grid_search_threshold(prob_maps, gt_maps, aux_probs=None):
    best_score = -1.0
    best_params = {
        'prob_threshold': 0.5,
        'min_area': 0,
        'reject_threshold': 0.0,
        'gate_threshold': 0.0
    }

    prob_thresholds = np.arange(0.2, 0.9, 0.05)
    min_areas = [0, 15, 30, 50]
    reject_thresholds = [0.0, 0.1, 0.2]
    gate_thresholds = [0.0, 0.2, 0.35, 0.5] if aux_probs is not None else [0.0]

    if aux_probs is not None:
        items = list(zip(prob_maps, gt_maps, aux_probs))
    else:
        items = [(pm, gt, None) for pm, gt in zip(prob_maps, gt_maps)]

    for p_thresh, min_a, rej_thresh, gate_th in product(
            prob_thresholds, min_areas, reject_thresholds, gate_thresholds):
        dice_scores = []
        clean_scores = []

        for probs, gt, aux_p in items:
            bin_mask = postprocess_mask(probs, p_thresh, min_a, rej_thresh, aux_p, gate_th)

            if gt.sum() > 0:
                tp = (bin_mask * gt).sum()
                fp = (bin_mask * (1 - gt)).sum()
                fn = ((1 - bin_mask) * gt).sum()
                dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
                dice_scores.append(dice)
            else:
                clean_scores.append(1.0 - bin_mask.mean())

        avg_dice = np.mean(dice_scores) if len(dice_scores) > 0 else 0.0
        avg_clean = np.mean(clean_scores) if len(clean_scores) > 0 else 1.0
        score = 0.5 * avg_dice + 0.5 * avg_clean

        if score > best_score:
            best_score = score
            best_params = {
                'prob_threshold': float(p_thresh),
                'min_area': int(min_a),
                'reject_threshold': float(rej_thresh),
                'gate_threshold': float(gate_th)
            }

    print(f"Best Score: {best_score:.4f}, Params: {best_params}")

    return best_params