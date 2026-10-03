#!/usr/bin/env python3
"""Extract 500 balanced MAR20-train aircraft instances with prompted SAM2."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from extract_mtarsi_foregrounds import choose_conservative_mask

DATA_ROOT = Path('/data3/tianzhibei/datasets/MAR20/yolo_obb/train')
CLASS_NAMES = [
    'A1 SU-35', 'A2 C-130', 'A3 C-17', 'A4 C-5', 'A5 F-16',
    'A6 TU-160', 'A7 E-3', 'A8 B-52', 'A9 P-3C', 'A10 B-1B',
    'A11 E-8', 'A12 TU-22', 'A13 F-15', 'A14 KC-135', 'A15 F-22',
    'A16 F/A-18', 'A17 TU-95', 'A18 KC-10', 'A19 SU-34', 'A20 SU-24',
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_instances() -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for label_path in sorted((DATA_ROOT / 'labels').glob('*.txt')):
        image_path = DATA_ROOT / 'images' / f'{label_path.stem}.jpg'
        if not image_path.is_file():
            continue
        for object_index, line in enumerate(label_path.read_text().splitlines()):
            fields = line.split()
            if len(fields) != 9:
                continue
            try:
                class_id = int(fields[0])
                poly = [float(x) for x in fields[1:]]
            except ValueError:
                continue
            if not 0 <= class_id < len(CLASS_NAMES) or not np.isfinite(poly).all():
                continue
            groups[class_id].append({
                'image_path': image_path, 'label_path': label_path,
                'object_index': object_index, 'polygon_normalized': poly,
            })
    return groups


def choose_instances(groups: dict[int, list[dict]], count: int, seed: int) -> list[dict]:
    if count > 500:
        raise ValueError('this first batch is capped at 500 instances')
    rng = random.Random(seed)
    classes = sorted(groups)
    for cls in classes:
        rng.shuffle(groups[cls])
    chosen = []
    while len(chosen) < count:
        progressed = False
        for cls in classes:
            if groups[cls] and len(chosen) < count:
                rec = dict(groups[cls].pop())
                rec['class_id'] = cls
                chosen.append(rec)
                progressed = True
        if not progressed:
            break
    if len(chosen) != count:
        raise ValueError(f'only {len(chosen)} valid train instances available for requested {count}')
    return chosen


def crop_and_prompt(image: Image.Image, polygon: list[float], context: float) -> tuple[Image.Image, list[float], list[int]]:
    width, height = image.size
    xs = np.asarray(polygon[0::2], dtype=np.float32) * width
    ys = np.asarray(polygon[1::2], dtype=np.float32) * height
    x1, x2 = float(np.clip(xs.min(), 0, width)), float(np.clip(xs.max(), 0, width))
    y1, y2 = float(np.clip(ys.min(), 0, height)), float(np.clip(ys.max(), 0, height))
    bw, bh = max(2.0, x2 - x1), max(2.0, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(bw, bh) * context
    left = max(0, int(round(cx - side / 2)))
    top = max(0, int(round(cy - side / 2)))
    right = min(width, int(round(cx + side / 2)))
    bottom = min(height, int(round(cy + side / 2)))
    # If an object lies near an image boundary, shift the crop inward before clipping.
    desired = max(2, int(round(side)))
    if right - left < desired:
        if left == 0:
            right = min(width, desired)
        elif right == width:
            left = max(0, width - desired)
    if bottom - top < desired:
        if top == 0:
            bottom = min(height, desired)
        elif bottom == height:
            top = max(0, height - desired)
    crop = image.crop((left, top, right, bottom)).convert('RGB')
    margin_x, margin_y = bw * .05, bh * .08
    prompt = [
        max(0.0, x1 - left - margin_x), max(0.0, y1 - top - margin_y),
        min(float(crop.width), x2 - left + margin_x),
        min(float(crop.height), y2 - top + margin_y),
    ]
    return crop, prompt, [left, top, right, bottom]


def save_preview(image: np.ndarray, mask: np.ndarray, path: Path, class_name: str) -> list[float]:
    rgba = np.dstack((image, (mask.astype(np.uint8) * 255)))
    fg = Image.fromarray(rgba, 'RGBA')
    original_w, original_h = fg.size
    fg.thumbnail((460, 460), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (512, 512), (232, 232, 232))
    ox, oy = (512 - fg.width) // 2, (512 - fg.height) // 2
    canvas.paste(fg, (ox, oy), fg)
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), class_name, fill=(20, 20, 20), stroke_width=2, stroke_fill=(240, 240, 240))
    canvas.save(path, quality=94)
    rows, cols = np.where(mask)
    x1 = ox + cols.min() * fg.width / original_w
    x2 = ox + (cols.max() + 1) * fg.width / original_w
    y1 = oy + rows.min() * fg.height / original_h
    y2 = oy + (rows.max() + 1) * fg.height / original_h
    return [(x1+x2)/1024, (y1+y2)/1024, (x2-x1)/512, (y2-y1)/512]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, default=Path('/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/checkpoints/sam2.1_hiera_large.pt'))
    ap.add_argument('--config', default='configs/sam2.1/sam2.1_hiera_l.yaml')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--count', type=int, default=500)
    ap.add_argument('--seed', type=int, default=20260926)
    ap.add_argument('--context', type=float, default=1.8,
                    help='square crop side as a multiple of the OBB enclosing-box long side')
    args = ap.parse_args()
    if args.count < 1 or args.count > 500:
        raise ValueError('--count must be between 1 and 500')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    for rel in ('masks', 'cutouts', 'source_crops', 'overlays', 'viewer/images', 'viewer/labels'):
        (args.output / rel).mkdir(parents=True, exist_ok=True)

    groups = read_instances()
    chosen = choose_instances(groups, args.count, args.seed)
    predictor = SAM2ImagePredictor(build_sam2(args.config, str(args.checkpoint), device='cuda'))
    records = []
    for index, item in enumerate(chosen, 1):
        original = Image.open(item['image_path']).convert('RGB')
        crop, prompt, crop_box = crop_and_prompt(original, item['polygon_normalized'], args.context)
        image = np.array(crop, dtype=np.uint8, copy=True)
        predictor.set_image(image)
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            masks, scores, _ = predictor.predict(box=np.asarray(prompt, dtype=np.float32), multimask_output=True)
        torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000
        mask, score, selection = choose_conservative_mask(masks, scores)
        mask = mask.astype(bool)
        class_id = item['class_id']
        cls_name = CLASS_NAMES[class_id]
        stem = f'{index:04d}__A{class_id + 1:02d}__{item["image_path"].stem}__obj{item["object_index"]:02d}'
        mask_path = args.output / 'masks' / f'{stem}.png'
        cutout_path = args.output / 'cutouts' / f'{stem}.png'
        source_crop_path = args.output / 'source_crops' / f'{stem}.jpg'
        overlay_path = args.output / 'overlays' / f'{stem}.jpg'
        preview_path = args.output / 'viewer/images' / f'{stem}.jpg'
        label_out = args.output / 'viewer/labels' / f'{stem}.txt'
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        Image.fromarray(np.dstack((image, mask.astype(np.uint8) * 255)), 'RGBA').save(cutout_path)
        crop.save(source_crop_path, quality=95)
        overlay = image.copy()
        overlay[mask] = (.45 * overlay[mask] + .55 * np.array([255, 30, 30])).astype(np.uint8)
        Image.fromarray(overlay).save(overlay_path, quality=94)
        preview_xywh = save_preview(image, mask, preview_path, cls_name)
        rows, cols = np.where(mask)
        if not len(rows):
            raise RuntimeError(f'empty mask unexpectedly selected: {stem}')
        x1, y1, x2, y2 = int(cols.min()), int(rows.min()), int(cols.max() + 1), int(rows.max() + 1)
        label_out.write_text(f'{class_id} ' + ' '.join(f'{v:.6f}' for v in preview_xywh) + '\n')
        record = {
            'id': stem, 'class_id': class_id, 'class_name': cls_name,
            'source_dataset': 'MAR20', 'source_split': 'train',
            'source_image_path': str(item['image_path']),
            'source_label_path': str(item['label_path']),
            'source_image_sha256': sha256(item['image_path']),
            'object_index_in_label': item['object_index'],
            'source_polygon_normalized': item['polygon_normalized'],
            'crop_xyxy_in_source': crop_box, 'prompt_box_xyxy_in_crop': prompt,
            'mask_path': str(mask_path), 'cutout_path': str(cutout_path),
            'source_crop_path': str(source_crop_path), 'overlay_path': str(overlay_path),
            'viewer_image_path': str(preview_path), 'viewer_label_path': str(label_out),
            'selected_score': float(score), 'sam_scores': [float(s) for s in scores],
            'selection_detail': selection, 'mask_area_ratio': float(mask.mean()),
            'inference_ms': inference_ms, 'pipeline_status': 'automatic_unreviewed',
        }
        records.append(record)
        if index % 25 == 0:
            print(json.dumps({'completed': index, 'total': len(chosen), 'class': cls_name}), flush=True)
    (args.output / 'viewer/data.yaml').write_text('names:\n' + ''.join(f'  {i}: {name}\n' for i, name in enumerate(CLASS_NAMES)))
    (args.output / 'manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    (args.output / 'generation_config.json').write_text(json.dumps({
        'pipeline_version': 'mar20-sam2-foreground-v1', 'source_split': 'train only',
        'requested': args.count, 'generated': len(records), 'seed': args.seed,
        'sam_checkpoint': str(args.checkpoint), 'sam_config': args.config,
        'crop_context': args.context, 'selection': 'MTARSI conservative mask selector',
        'human_or_ai_visual_review': False,
    }, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'generated': len(records), 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
