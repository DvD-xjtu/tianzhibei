#!/usr/bin/env python3
"""Purely programmatic, conservative quality gate for v6/v7 MAR20 paste batches.

No human decisions are read. The thresholds are intentionally conservative and
were selected using the earlier 40-case development batch; the new 200-case
batch is held out for the user to assess.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from synthesize_copypaste_v6 import atomic_bytes, digest
from verify_copypaste_v6 import audit

GATES = {
    'max_source_background_like_fraction': .30,
    'max_secondary_mask_fraction': .02,
    'max_placement_texture_std_lab': 8.0,
    'max_placement_vegetation_fraction': .003,
    'max_placement_edge_fraction': .075,
    'max_placement_color_distance_lab': 6.0,
    'min_paste_lightness_delta_abs': 18.0,
    'min_paste_lightness_delta': -85.0,
    'max_paste_lightness_delta': 20.0,
    'max_paste_saturation_delta_abs': 45.0,
    'grayscale_background_saturation_max': 7.0,
    'grayscale_paste_saturation_min': 18.0,
}


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def metrics(record: dict) -> dict:
    source = np.asarray(Image.open(record['foreground_source_path']).convert('RGB'))
    rgba = np.asarray(Image.open(record['foreground_path']).convert('RGBA'))
    mask_source = rgba[:, :, 3] > 127
    if source.shape[:2] != mask_source.shape or mask_source.sum() < 100:
        raise ValueError(f"bad source mask: {record['id']}")
    ring = cv2.dilate(mask_source.astype('uint8'), np.ones((11, 11), np.uint8)) > 0
    ring &= ~mask_source
    if ring.sum() < 50:
        raise ValueError(f"no source ring: {record['id']}")
    source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    reference = np.median(source_lab[ring], axis=0)
    like_fraction = float((np.linalg.norm(source_lab[mask_source]-reference, axis=1) < 15).mean())
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask_source.astype('uint8'), 8)
    areas = sorted(stats[1:n, cv2.CC_STAT_AREA].tolist(), reverse=True)
    secondary = float(sum(areas[1:]) / sum(areas)) if areas else 1.0

    background = np.asarray(Image.open(record['background_image_path']).convert('RGB'))
    composite = np.asarray(Image.open(record['image_path']).convert('RGB'))
    pasted_mask = cv2.imread(record['instance_mask_path'], cv2.IMREAD_GRAYSCALE) > 0
    if composite.shape != background.shape or pasted_mask.shape != background.shape[:2]:
        raise ValueError(f"misaligned composite: {record['id']}")
    bg_lab = cv2.cvtColor(background, cv2.COLOR_RGB2LAB).astype(np.float32)
    paste_lab = cv2.cvtColor(composite, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_l = float(np.median(bg_lab[:, :, 0][pasted_mask]))
    paste_l = float(np.median(paste_lab[:, :, 0][pasted_mask]))
    bg_pixels = background[pasted_mask].astype(np.float32)
    paste_pixels = composite[pasted_mask].astype(np.float32)
    bg_sat = float(np.median(bg_pixels.max(axis=1)-bg_pixels.min(axis=1)))
    paste_sat = float(np.median(paste_pixels.max(axis=1)-paste_pixels.min(axis=1)))
    return {
        'source_background_like_fraction': like_fraction,
        'secondary_mask_fraction': secondary,
        'placement_texture_std_lab': float(record['placement']['texture_std_lab']),
        'placement_vegetation_fraction': float(record['placement']['vegetation_fraction']),
        'placement_edge_fraction': float(record['placement']['edge_fraction']),
        'placement_color_distance_lab': float(record['placement']['color_distance_lab']),
        'paste_lightness_delta': paste_l-bg_l,
        'paste_saturation_delta': paste_sat-bg_sat,
        'background_saturation': bg_sat,
        'paste_saturation': paste_sat,
    }


def rejection_reasons(value: dict) -> list[str]:
    reasons = []
    if value['source_background_like_fraction'] > GATES['max_source_background_like_fraction']:
        reasons.append('source_background_leak')
    if value['secondary_mask_fraction'] > GATES['max_secondary_mask_fraction']:
        reasons.append('mask_fragments')
    if value['placement_texture_std_lab'] > GATES['max_placement_texture_std_lab']:
        reasons.append('placement_texture')
    if value['placement_vegetation_fraction'] > GATES['max_placement_vegetation_fraction']:
        reasons.append('placement_vegetation')
    if value['placement_edge_fraction'] > GATES['max_placement_edge_fraction']:
        reasons.append('placement_edges')
    if value['placement_color_distance_lab'] > GATES['max_placement_color_distance_lab']:
        reasons.append('placement_surface_color')
    delta = value['paste_lightness_delta']
    if abs(delta) < GATES['min_paste_lightness_delta_abs']:
        reasons.append('insufficient_aircraft_contrast')
    if not GATES['min_paste_lightness_delta'] <= delta <= GATES['max_paste_lightness_delta']:
        reasons.append('paste_lightness_mismatch')
    if abs(value['paste_saturation_delta']) > GATES['max_paste_saturation_delta_abs']:
        reasons.append('paste_saturation_mismatch')
    if (value['background_saturation'] <= GATES['grayscale_background_saturation_max']
            and value['paste_saturation'] >= GATES['grayscale_paste_saturation_min']):
        reasons.append('color_paste_on_grayscale_background')
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be new or empty')
    checked = audit(args.input)
    if checked['mode'] != 'mar20':
        parser.error('MAR20 OBB mode required for this training-format export')
    records = [json.loads(line) for line in (args.input / 'manifest.jsonl').read_text().splitlines() if line]
    accepted = []
    decisions = []
    reasons_count: Counter[str] = Counter()
    primary_reasons: Counter[str] = Counter()
    for record in records:
        value = metrics(record)
        reasons = rejection_reasons(value)
        for reason in reasons:
            reasons_count[reason] += 1
        if reasons:
            primary_reasons[reasons[0]] += 1
        passed = not reasons
        decisions.append({'id': record['id'], 'class_name': record['class_name'],
                          'accepted': passed, 'reasons': reasons,
                          'metrics': {k: round(v, 6) for k, v in value.items()}})
        if not passed:
            continue
        stem = record['id']
        image = args.output / 'train/images' / f'{stem}.jpg'
        label = args.output / 'train/labels' / f'{stem}.txt'
        mask = args.output / 'instance_masks' / f'{stem}.png'
        comparison = args.output / 'comparison/train/images' / f'{stem}.jpg'
        comparison_label = args.output / 'comparison/train/labels' / f'{stem}.txt'
        copy_file(Path(record['image_path']), image)
        copy_file(Path(record['label_path']), label)
        copy_file(Path(record['instance_mask_path']), mask)
        copy_file(args.input / 'comparison/train/images' / f'{stem}.jpg', comparison)
        copy_file(args.input / 'comparison/train/labels' / f'{stem}.txt', comparison_label)
        item = dict(record)
        item.update({'image_path': str(image), 'label_path': str(label),
                     'instance_mask_path': str(mask), 'review_status': 'engineering_pass_unreviewed',
                     'automatic_metrics': {k: round(v, 6) for k, v in value.items()},
                     'source_manifest_path': str(args.input / 'manifest.jsonl')})
        for field, sha in [('image_path', 'image_sha256'), ('label_path', 'label_sha256'),
                           ('instance_mask_path', 'instance_mask_sha256')]:
            if digest(Path(item[field])) != item[sha]:
                raise RuntimeError(f"copy checksum mismatch: {stem}: {field}")
        accepted.append(item)
    copy_file(args.input / 'final/data.yaml', args.output / 'data.yaml')
    copy_file(args.input / 'comparison/data.yaml', args.output / 'comparison/data.yaml')
    atomic_bytes(args.output / 'manifest.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in accepted).encode())
    atomic_bytes(args.output / 'qa/decisions.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in decisions).encode())
    summary = {'source_count': len(records), 'passed': len(accepted), 'rejected': len(records)-len(accepted),
               'engineering_pass_rate': len(accepted)/len(records),
               'passed_classes': dict(Counter(r['class_name'] for r in accepted)),
               'rejection_reasons_all': dict(reasons_count),
               'rejection_primary_reasons': dict(primary_reasons),
               'source_manifest_sha256': digest(args.input / 'manifest.jsonl'),
               'filter_version': 'engineering-v7-1', 'thresholds': GATES,
               'human_or_ai_visual_review_performed_on_this_batch': False,
               'label_format': 'MAR20 YOLO-OBB 20-class; target-15 mapping pending'}
    atomic_bytes(args.output / 'qa/summary.json', (json.dumps(summary, ensure_ascii=False, indent=2)+'\n').encode())
    for start in range(0, len(accepted), 10):
        chunk = accepted[start:start+10]
        sheet = Image.new('RGB', (1600, math.ceil(len(chunk)/2)*430), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, record in enumerate(chunk):
            panel = Image.open(args.output / 'comparison/train/images' / f'{record["id"]}.jpg')
            thumb = panel.resize((800, 400), Image.Resampling.LANCZOS)
            sx, sy = i % 2 * 800, i // 2 * 430
            draw.text((sx+6, sy+5), record['id'], fill='black')
            sheet.paste(thumb, (sx, sy+25))
        page = args.output / 'qa/contact_sheets' / f'page_{start//10+1:02d}.jpg'
        page.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(page, quality=90)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
