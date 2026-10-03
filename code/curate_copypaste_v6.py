#!/usr/bin/env python3
"""Fail-closed curation of v6 MAR20 composites after automatic and visual QA.

The automatic gate catches masks that mostly resemble their source background.
It cannot judge aircraft identity or scene realism; those require explicit review
decisions in a versioned JSON file. No input artifacts are modified.
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


def background_like_fraction(record: dict) -> float:
    source = np.asarray(Image.open(record['foreground_source_path']).convert('RGB'))
    alpha = np.asarray(Image.open(record['foreground_path']).convert('RGBA'))[:, :, 3]
    if source.shape[:2] != alpha.shape:
        raise ValueError(f"misaligned foreground: {record['id']}")
    mask = alpha > 127
    if mask.sum() < 100:
        raise ValueError(f"empty foreground: {record['id']}")
    ring = cv2.dilate(mask.astype('uint8'), np.ones((11, 11), np.uint8)) > 0
    ring &= ~mask
    if ring.sum() < 50:
        raise ValueError(f"missing source context: {record['id']}")
    lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    context = np.median(lab[ring], axis=0)
    distance = np.linalg.norm(lab[mask] - context, axis=1)
    return float((distance < 15).mean())


def copy_relative(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-background-like-fraction', type=float, default=.40)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be new or empty')
    if not 0 <= args.max_background_like_fraction <= 1:
        parser.error('invalid background-like threshold')
    checked = audit(args.input)
    if checked['mode'] != 'mar20':
        parser.error('only MAR20 OBB mode can be exported as a training-format set')
    records = [json.loads(line) for line in (args.input / 'manifest.jsonl').read_text().splitlines() if line]
    review = json.loads(args.review.read_text())
    if review.get('source_manifest_sha256') != digest(args.input / 'manifest.jsonl'):
        parser.error('review was not made for this exact source manifest')
    decisions = review['decisions']
    ids = {r['id'] for r in records}
    if set(decisions) != ids:
        missing, extra = ids - set(decisions), set(decisions) - ids
        parser.error(f'review must cover every record exactly; missing={sorted(missing)}, extra={sorted(extra)}')
    curated = []
    decisions_out = []
    reasons: Counter[str] = Counter()
    for record in records:
        decision = decisions[record['id']]
        if type(decision.get('accept')) is not bool or not isinstance(decision.get('reason'), str):
            parser.error(f"bad decision for {record['id']}")
        fraction = background_like_fraction(record)
        auto_ok = fraction <= args.max_background_like_fraction
        accepted = auto_ok and decision['accept']
        reason = 'accepted' if accepted else ('foreground_background_contamination' if not auto_ok else decision['reason'])
        if not accepted:
            reasons[reason] += 1
        row = {'id': record['id'], 'class_name': record['class_name'],
               'automatic_background_like_fraction': round(fraction, 5),
               'automatic_pass': auto_ok, 'visual_accept': decision['accept'],
               'accepted': accepted, 'reason': reason}
        decisions_out.append(row)
        if not accepted:
            continue
        stem = record['id']
        image = args.output / 'train/images' / f'{stem}.jpg'
        label = args.output / 'train/labels' / f'{stem}.txt'
        mask = args.output / 'instance_masks' / f'{stem}.png'
        comparison = args.output / 'comparison/train/images' / f'{stem}.jpg'
        comparison_label = args.output / 'comparison/train/labels' / f'{stem}.txt'
        copy_relative(Path(record['image_path']), image)
        copy_relative(Path(record['label_path']), label)
        copy_relative(Path(record['instance_mask_path']), mask)
        copy_relative(args.input / 'comparison/train/images' / f'{stem}.jpg', comparison)
        copy_relative(args.input / 'comparison/train/labels' / f'{stem}.txt', comparison_label)
        item = dict(record)
        item.update({'image_path': str(image), 'label_path': str(label),
                     'instance_mask_path': str(mask), 'review_status': 'visual_approved',
                     'review_reason': decision['reason'],
                     'automatic_background_like_fraction': round(fraction, 5),
                     'source_manifest_path': str(args.input / 'manifest.jsonl')})
        for path_field, hash_field in [('image_path', 'image_sha256'),
                                       ('label_path', 'label_sha256'),
                                       ('instance_mask_path', 'instance_mask_sha256')]:
            if digest(Path(item[path_field])) != item[hash_field]:
                raise RuntimeError(f"copy checksum mismatch: {stem}: {path_field}")
        curated.append(item)
    copy_relative(args.input / 'final/data.yaml', args.output / 'data.yaml')
    copy_relative(args.input / 'comparison/data.yaml', args.output / 'comparison/data.yaml')
    atomic_bytes(args.output / 'manifest.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in curated).encode())
    atomic_bytes(args.output / 'qa/decisions.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in decisions_out).encode())
    summary = {'source_count': len(records), 'accepted': len(curated), 'rejected': len(records)-len(curated),
               'acceptance_rate': len(curated)/len(records),
               'accepted_classes': dict(Counter(r['class_name'] for r in curated)),
               'rejection_reasons': dict(reasons),
               'source_manifest_sha256': digest(args.input / 'manifest.jsonl'),
               'review_sha256': digest(args.review), 'reviewer': review.get('reviewer'),
               'automatic_gate': {'max_background_like_fraction': args.max_background_like_fraction,
                                  'lab_distance_threshold': 15, 'source_context_ring_px': 5},
               'label_format': 'MAR20 YOLO-OBB 20-class; target-15 mapping still pending',
               'visual_qc_not_model_validation': True}
    atomic_bytes(args.output / 'qa/summary.json', (json.dumps(summary, ensure_ascii=False, indent=2)+'\n').encode())
    for start in range(0, len(curated), 10):
        chunk = curated[start:start+10]
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
