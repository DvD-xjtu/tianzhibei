#!/usr/bin/env python3
"""Build mask-aligned geometric/photometric variants from accepted MAR20 cutouts."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from augment_mtarsi_foregrounds_v1 import transform, sha256, write_png

SOURCE = Path('/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/foreground_mar20_500_v1')
CLASS_NAMES = [
    'A1 SU-35', 'A2 C-130', 'A3 C-17', 'A4 C-5', 'A5 F-16',
    'A6 TU-160', 'A7 E-3', 'A8 B-52', 'A9 P-3C', 'A10 B-1B',
    'A11 E-8', 'A12 TU-22', 'A13 F-15', 'A14 KC-135', 'A15 F-22',
    'A16 F/A-18', 'A17 TU-95', 'A18 KC-10', 'A19 SU-34', 'A20 SU-24',
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=SOURCE)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=20260927)
    ap.add_argument('--variants-per-source', type=int, default=4)
    args = ap.parse_args()
    if args.variants_per_source < 1 or (args.output.exists() and any(args.output.iterdir())):
        ap.error('variants must be positive and output must be new or empty')
    source_records = json.loads((args.source/'manifest_quality_gated.json').read_text())
    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    records, rejected = [], Counter()
    for src in source_records:
        if not src.get('quality_gate', {}).get('accepted'):
            continue
        cls = src['class_name']
        if cls not in CLASS_NAMES:
            rejected['unknown_class'] += 1
            continue
        context = np.asarray(Image.open(src['source_crop_path']).convert('RGB'))
        rgba = np.asarray(Image.open(src['cutout_path']).convert('RGBA'))
        if context.shape[:2] != rgba.shape[:2]:
            rejected['shape_mismatch'] += args.variants_per_source
            continue
        for vi in range(args.variants_per_source):
            angle = py_rng.uniform(0, 360)
            zoom = py_rng.uniform(.8, 1.2)
            brightness = py_rng.uniform(-3, 3)
            contrast = py_rng.uniform(.97, 1.03)
            noise_sigma = py_rng.uniform(0, 1.2)
            ctx, variant, params = transform(context, rgba, angle, zoom, brightness,
                                             contrast, noise_sigma, np_rng)
            alpha = variant[:, :, 3] > 127
            n, _, stats, _ = cv2.connectedComponentsWithStats(alpha.astype(np.uint8), 8)
            areas = sorted(stats[1:n, cv2.CC_STAT_AREA].tolist(), reverse=True)
            total = int(sum(areas))
            secondary = sum(areas[1:]) / max(total, 1)
            if total < 100 or not areas or areas[0]/total < .90 or secondary > .07 or alpha.mean() > .45:
                rejected['transformed_mask_gate'] += 1
                continue
            stem = f'{src["id"]}__aug{vi+1:02d}'
            cut_path = args.output/'cutouts'/f'{stem}.png'
            context_path = args.output/'source_context'/f'{stem}.png'
            write_png(cut_path, variant)
            write_png(context_path, ctx)
            records.append({
                'id': stem, 'class': cls, 'source': 'mar20_train',
                'source_path': str(context_path), 'cutout_path': str(cut_path),
                'mask_path': '', 'foreground_pool': 'MAR20_SAM2_accepted_cutout_aug_v2',
                'base_foreground_id': src['id'], 'augmentation': params,
                'paste_scale_factor': zoom, 'source_manifest': str(args.source/'manifest_quality_gated.json'),
                'source_cutout_sha256': sha256(Path(src['cutout_path'])),
                'variant_sha256': sha256(cut_path), 'mask_area_ratio': float(alpha.mean()),
                'component_count': int(n-1), 'secondary_component_ratio': float(secondary),
                'gate': 'passed',
            })
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    summary = {
        'pipeline_version': 'mar20-foreground-augmentation-v2',
        'base_sources': sum(bool(x.get('quality_gate', {}).get('accepted')) for x in source_records),
        'variants_per_source_requested': args.variants_per_source,
        'generated_variants': len(records), 'rejected_variants': dict(rejected),
        'classes': dict(Counter(x['class'] for x in records)),
        'base_sources_by_class': dict(Counter(x['class_name'] for x in source_records
                                              if x.get('quality_gate', {}).get('accepted'))), 'seed': args.seed,
        'rotation_deg': [0, 360], 'zoom_range': [.8, 1.2],
        'photometric_perturbation': {'brightness': [-3, 3], 'contrast': [.97, 1.03], 'noise_sigma': [0, 1.2]},
        'augmentation_gate': {'min_alpha_pixels': 100, 'min_largest_component_ratio': .90,
                              'max_secondary_component_ratio': .07, 'max_alpha_area_ratio': .45},
    }
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
