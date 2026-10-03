#!/usr/bin/env python3
"""Apply the MTARSI automatic foreground gate to a MAR20 SAM2 batch."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def quality(mask: np.ndarray, score: float, gates: dict) -> dict:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    sizes = sorted((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)), reverse=True)
    total = sum(sizes)
    dominant = sizes[0] / total if total else 0.0
    secondary = sum(sizes[1:]) / total if total else 1.0
    area = float(mask.mean())
    ys, xs = np.where(mask)
    if len(xs):
        bbox_area = (int(ys.max()) - int(ys.min()) + 1) * (int(xs.max()) - int(xs.min()) + 1)
        bbox_fill = total / bbox_area if bbox_area else 0.0
    else:
        bbox_fill = 0.0
    reasons = []
    if score < gates['min_score']:
        reasons.append('sam_score_below_threshold')
    if dominant < gates['min_dominant']:
        reasons.append('dominant_component_below_threshold')
    if secondary > gates['max_secondary']:
        reasons.append('secondary_components_above_threshold')
    if area > gates['max_mask_area']:
        reasons.append('mask_area_above_threshold')
    if bbox_fill < gates['min_bbox_fill']:
        reasons.append('bbox_fill_below_threshold')
    return {
        'accepted': not reasons, 'reasons': reasons, 'sam_score': float(score),
        'component_count': len(sizes), 'dominant_component_ratio': dominant,
        'secondary_component_ratio': secondary, 'mask_area_ratio': area,
        'bbox_fill_ratio': bbox_fill,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', required=True, type=Path)
    ap.add_argument('--min-score', type=float, default=.50)
    ap.add_argument('--min-dominant', type=float, default=.90)
    ap.add_argument('--max-secondary', type=float, default=.07)
    ap.add_argument('--max-mask-area', type=float, default=.45)
    ap.add_argument('--min-bbox-fill', type=float, default=.10)
    args = ap.parse_args()
    manifest_path = args.batch / 'manifest.json'
    records = json.loads(manifest_path.read_text())
    gates = {'min_score': args.min_score, 'min_dominant': args.min_dominant,
             'max_secondary': args.max_secondary, 'max_mask_area': args.max_mask_area,
             'min_bbox_fill': args.min_bbox_fill}
    passed_root = args.batch / 'viewer_auto_accepted'
    train = passed_root / 'train'
    (train / 'images').mkdir(parents=True, exist_ok=True)
    (train / 'labels').mkdir(parents=True, exist_ok=True)
    reasons = Counter()
    class_total, class_passed = Counter(), Counter()
    decisions = []
    for record in records:
        class_total[record['class_name']] += 1
        mask = np.asarray(Image.open(record['mask_path']).convert('L')) > 0
        result = quality(mask, record['selected_score'], gates)
        record['quality_gate'] = result
        reasons.update(result['reasons'])
        decisions.append({'id': record['id'], 'class_id': record['class_id'],
                          'class_name': record['class_name'], **result})
        if not result['accepted']:
            continue
        class_passed[record['class_name']] += 1
        stem = record['id']
        shutil.copy2(record['viewer_image_path'], train / 'images' / f'{stem}.jpg')
        shutil.copy2(record['viewer_label_path'], train / 'labels' / f'{stem}.txt')
    shutil.copy2(args.batch / 'viewer' / 'data.yaml', passed_root / 'data.yaml')
    (args.batch / 'manifest_quality_gated.json').write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    qa = args.batch / 'qa'
    qa.mkdir(exist_ok=True)
    (qa / 'decisions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in decisions))
    passed = sum(class_passed.values())
    summary = {
        'pipeline': 'mar20-sam2-foreground-v1', 'source_split': 'train only',
        'candidate_count': len(records), 'passed': passed,
        'rejected': len(records) - passed,
        'pass_rate': passed / len(records) if records else 0.0,
        'classes_total': dict(class_total), 'classes_passed': dict(class_passed),
        'rejection_reasons_nonexclusive': dict(reasons), 'thresholds': gates,
        'filter_implementation': 'OpenCV 8-connected components; thresholds reused from MTARSI foreground gate',
        'human_or_ai_visual_review': False, 'training_ready': False,
    }
    (qa / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
