#!/usr/bin/env python3
"""Create deterministic, mask-aligned geometric/photometric MTARSI variants."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path('/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/foreground_all_15taxonomy_v6')
CLASSES = ['C-130', 'C-17', 'F-16', 'E-3', 'B-52', 'B-1B', 'F-15', 'F/A-18']


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, compress_level=3)


def warp_expand(arr: np.ndarray, matrix: np.ndarray, size: tuple[int, int], *, mask: bool = False) -> np.ndarray:
    flags = cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR
    border = cv2.BORDER_CONSTANT
    return cv2.warpAffine(arr, matrix, size, flags=flags, borderMode=border,
                          borderValue=0 if arr.ndim == 2 else (0, 0, 0))


def transform(rgb: np.ndarray, rgba: np.ndarray, angle: float, zoom: float,
              brightness: float, contrast: float, noise_sigma: float,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    alpha = rgba[:, :, 3].astype(np.float32) / 255
    fg = rgba[:, :, :3].astype(np.float32)
    premult = fg * alpha[:, :, None]
    h, w = alpha.shape
    cx, cy = w / 2, h / 2
    mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    out_w = int(math.ceil(h * abs(mat[0, 1]) + w * abs(mat[0, 0])))
    out_h = int(math.ceil(h * abs(mat[0, 0]) + w * abs(mat[0, 1])))
    mat[0, 2] += out_w / 2 - cx
    mat[1, 2] += out_h / 2 - cy
    p = warp_expand(premult, mat, (out_w, out_h))
    a = np.clip(warp_expand(alpha, mat, (out_w, out_h)), 0, 1)
    # Keep a matching transformed source crop so the paste filter can inspect
    # source aircraft pixels against the original crop's local background ring.
    source = warp_expand(rgb, mat, (out_w, out_h)).astype(np.float32)
    colors = np.clip(p / np.maximum(a[:, :, None], 1e-4), 0, 255)
    colors = (colors - 127.5) * contrast + 127.5 + brightness
    if noise_sigma:
        colors += rng.normal(0, noise_sigma, colors.shape).astype(np.float32) * (a[:, :, None] > .8)
    colors = np.clip(colors, 0, 255)
    out = np.dstack([colors, a * 255]).astype(np.uint8)
    # retain transformed RGB context only where alpha is zero; avoid zero-padding
    # creating an artificial high-contrast ring around rotated source crops.
    context = np.clip(source, 0, 255).astype(np.uint8)
    return context, out, {'rotation_deg': round(angle, 4), 'zoom': round(zoom, 5),
                          'brightness': round(brightness, 4), 'contrast': round(contrast, 5),
                          'noise_sigma': round(noise_sigma, 4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260927)
    parser.add_argument('--variants-per-source', type=int, default=4)
    parser.add_argument('--classes', default=','.join(CLASSES))
    args = parser.parse_args()
    classes = [x.strip() for x in args.classes.split(',')]
    if args.variants_per_source < 1 or len(set(classes)) != len(classes) or any(x not in CLASSES for x in classes):
        parser.error('variants must be positive and classes must be unique supported classes')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be new or empty')
    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    sources = []
    for manifest in sorted(ROOT.glob('shard_*/manifest_quality_gated_v2.json')):
        for row in json.loads(manifest.read_text()):
            if (row.get('quality_gate', {}).get('accepted') and row['class'] in classes
                    and row['source'] in {'fixed', 'innar_train'}):
                row['_manifest'] = str(manifest)
                sources.append(row)
    sources.sort(key=lambda x: x['id'])
    out_rgba = args.output / 'cutouts'
    out_rgb = args.output / 'source_context'
    records = []
    rejected = Counter()
    for source in sources:
        rgb = np.asarray(Image.open(source['source_path']).convert('RGB'))
        rgba = np.asarray(Image.open(source['cutout_path']).convert('RGBA'))
        mask = np.asarray(Image.open(source['mask_path']).convert('L')) > 0
        if rgb.shape[:2] != mask.shape or rgba.shape[:2] != mask.shape:
            rejected['shape_mismatch'] += args.variants_per_source
            continue
        for vi in range(args.variants_per_source):
            angle = py_rng.uniform(0, 360)
            zoom = py_rng.uniform(.8, 1.2)
            brightness = py_rng.uniform(-3, 3)
            contrast = py_rng.uniform(.97, 1.03)
            noise_sigma = py_rng.uniform(0, 1.2)
            context, variant, params = transform(rgb, rgba, angle, zoom, brightness,
                                                  contrast, noise_sigma, np_rng)
            alpha = variant[:, :, 3] > 127
            n, labels, stats, _ = cv2.connectedComponentsWithStats(alpha.astype(np.uint8), 8)
            areas = sorted(stats[1:n, cv2.CC_STAT_AREA].tolist(), reverse=True)
            total = int(sum(areas))
            secondary = sum(areas[1:]) / max(total, 1)
            if (total < 100 or not areas or areas[0] / total < .90 or secondary > .07
                    or alpha.mean() > .45):
                rejected['transformed_mask_gate'] += 1
                continue
            stem = f'{source["id"]}__aug{vi+1:02d}'
            cut_path = out_rgba / f'{stem}.png'
            context_path = out_rgb / f'{stem}.png'
            write_png(cut_path, variant)
            write_png(context_path, context)
            # scale factor is applied at paste time to preserve the requested
            # 0.8–1.2 zoom despite the class/scene size prior.
            record = {'id': stem, 'class': source['class'], 'source': source['source'],
                      'source_path': str(context_path), 'cutout_path': str(cut_path),
                      'mask_path': '', 'foreground_pool': 'MTARSI_accepted_cutout_aug_v1',
                      'base_foreground_id': source['id'], 'augmentation': params,
                      'paste_scale_factor': zoom, 'source_manifest': source['_manifest'],
                      'source_cutout_sha256': sha256(Path(source['cutout_path'])),
                      'variant_sha256': sha256(cut_path), 'mask_area_ratio': float(alpha.mean()),
                      'component_count': int(n - 1), 'secondary_component_ratio': float(secondary),
                      'gate': 'passed'}
            records.append(record)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'manifest.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in records))
    summary = {'source_count': len(sources), 'variants_per_source_requested': args.variants_per_source,
               'generated_variants': len(records), 'rejected_variants': dict(rejected),
               'classes': dict(Counter(x['class'] for x in records)),
               'base_sources_by_class': dict(Counter(x['class'] for x in sources)),
               'source_kinds': dict(Counter(x['source'] for x in records)), 'seed': args.seed,
               'angle_deg': [0, 360], 'zoom_range': [.8, 1.2],
               'photometric_perturbation': {'brightness': [-3, 3], 'contrast': [.97, 1.03], 'noise_sigma': [0, 1.2]},
               'augmentation_gate': {'min_alpha_pixels': 100, 'min_largest_component_ratio': .90,
                                     'max_secondary_component_ratio': .07, 'max_alpha_area_ratio': .45}}
    (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
