#!/usr/bin/env python3
"""Automatic no-bad-case gate for a foreground extraction batch.

The gate deliberately favours precision over yield.  It rejects an output when
SAM confidence is low or when the mask is fragmented into multiple meaningful
foreground islands--both are strong indicators of runway, neighbour aircraft,
or shadow leakage in MTARSI crops.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def component_sizes(mask: np.ndarray) -> list[int]:
    """Four-connected component sizes without an extra scipy/opencv dependency."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    for row, col in zip(*np.where(mask)):
        if seen[row, col]:
            continue
        stack = [(int(row), int(col))]
        seen[row, col] = True
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < height and 0 <= xx < width and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        sizes.append(size)
    return sorted(sizes, reverse=True)


def quality(mask: np.ndarray, sam_score: float, min_score: float, min_dominant: float,
            max_secondary: float, max_mask_area: float, min_bbox_fill: float) -> dict:
    sizes = component_sizes(mask)
    total = sum(sizes)
    dominant = sizes[0] / total if total else 0.0
    secondary = sum(sizes[1:]) / total if total else 1.0
    mask_area = float(mask.mean())
    rows, cols = np.where(mask)
    if len(rows):
        bbox_area = (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)
        bbox_fill = total / bbox_area
    else:
        bbox_fill = 0.0
    reasons = []
    if sam_score < min_score:
        reasons.append(f'sam_score<{min_score:.2f}')
    if dominant < min_dominant:
        reasons.append(f'dominant_component<{min_dominant:.2f}')
    if secondary > max_secondary:
        reasons.append(f'secondary_components>{max_secondary:.2f}')
    if mask_area > max_mask_area:
        reasons.append(f'mask_area>{max_mask_area:.2f}')
    if bbox_fill < min_bbox_fill:
        reasons.append(f'bbox_fill<{min_bbox_fill:.2f}')
    return {
        'accepted': not reasons,
        'reasons': reasons,
        'sam_score': float(sam_score),
        'component_count': len(sizes),
        'dominant_component_ratio': dominant,
        'secondary_component_ratio': secondary,
        'mask_area_ratio_after_postprocess': mask_area,
        'bbox_fill_ratio': bbox_fill,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True, type=Path)
    parser.add_argument('--min-score', type=float, default=.50)
    parser.add_argument('--min-dominant', type=float, default=.90)
    parser.add_argument('--max-secondary', type=float, default=.07)
    parser.add_argument('--max-mask-area', type=float, default=.45,
                        help='Reject masks that consume implausibly much of the crop.')
    parser.add_argument('--min-bbox-fill', type=float, default=.10,
                        help='Reject thin outlines that are not a filled aircraft silhouette.')
    parser.add_argument('--output-name', default='viewer_auto_accepted')
    parser.add_argument('--manifest-name', default='manifest_quality_gated.json')
    args = parser.parse_args()
    records = json.loads((args.batch / 'manifest.json').read_text())
    target = args.batch / args.output_name / 'train'
    (target / 'images').mkdir(parents=True, exist_ok=True)
    (target / 'labels').mkdir(parents=True, exist_ok=True)
    accepted = 0
    for record in records:
        mask = np.asarray(Image.open(record['mask_path']).convert('L')) > 0
        gate = quality(mask, record['selected_score'], args.min_score, args.min_dominant,
                       args.max_secondary, args.max_mask_area, args.min_bbox_fill)
        record['quality_gate'] = gate
        if not gate['accepted']:
            continue
        accepted += 1
        stem = record['id']
        shutil.copy2(args.batch / 'viewer' / 'images' / f'{stem}.jpg', target / 'images' / f'{stem}.jpg')
        shutil.copy2(args.batch / 'viewer' / 'labels' / f'{stem}.txt', target / 'labels' / f'{stem}.txt')
    (args.batch / args.manifest_name).write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    shutil.copy2(args.batch / 'viewer' / 'data.yaml', args.batch / args.output_name / 'data.yaml')
    print(json.dumps({'total': len(records), 'accepted': accepted, 'rejected': len(records) - accepted}, ensure_ascii=False))


if __name__ == '__main__':
    main()
