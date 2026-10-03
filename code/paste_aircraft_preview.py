#!/usr/bin/env python3
"""Deterministic, context-guided MTARSI -> DIOR Copy-Paste smoke preview.

This produces review images and provenance, not a train-ready MAR20 dataset.
Only DIOR_train.json images and SAM2 quality-gate-v2 accepted cutouts are used.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path('/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1')
FG_ROOT = ROOT / 'foreground_all_15taxonomy_v6'
DIOR = Path('/data2/tianzhibei/LAE-1M/LAE-FOD/DIOR')
EXAMPLE_CLASSES = ['A-10', 'B-1B', 'B-52', 'C-130', 'C-17',
                   'E-3', 'F-15', 'F-16', 'F/A-18', 'C-17']
FIXED_CYCLE_POSITIONS = {1, 5}  # Show both MTARSI versions in every ten cases.
LABEL_NAMES = list(dict.fromkeys(EXAMPLE_CLASSES))


def close_to_box(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or
                a[3] + pad <= b[1] or b[3] + pad <= a[1])


def load_foregrounds() -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for shard in range(4):
        for item in json.loads((FG_ROOT / f'shard_{shard}' / 'manifest_quality_gated_v2.json').read_text()):
            if (item['quality_gate']['accepted'] and item['class'] in LABEL_NAMES
                    and item['source'] in {'fixed', 'innar_train'}):
                groups[item['class']].append(item)
    return groups


def load_backgrounds() -> list[dict]:
    data = json.loads((DIOR / 'DIOR_train.json').read_text())
    images = {item['id']: item for item in data['images']}
    airplane = next(item['id'] for item in data['categories'] if item['name'] == 'airplane')
    anns: dict[int, list[dict]] = defaultdict(list)
    for item in data['annotations']:
        anns[item['image_id']].append(item)
    records = []
    for image_id, obj in images.items():
        boxes = [x for x in anns[image_id] if x['category_id'] == airplane]
        if len(boxes) < 2:
            continue
        path = DIOR / 'JPEGImages-trainval' / obj['file_name']
        if path.exists():
            records.append({'image': obj, 'annotations': anns[image_id],
                            'airplanes': boxes, 'path': path})
    return records


def crop_foreground(record: dict) -> tuple[Image.Image, Image.Image, np.ndarray]:
    rgba = Image.open(record['cutout_path']).convert('RGBA')
    source = np.asarray(Image.open(record['source_path']).convert('RGB'), dtype=np.float32)
    alpha = np.asarray(rgba.getchannel('A')) > 128
    # A ring immediately outside the source mask estimates source illumination.
    grown = np.asarray(Image.fromarray((alpha * 255).astype('uint8')).filter(ImageFilter.MaxFilter(17))) > 0
    ring = grown & ~alpha
    if ring.sum() < 100:
        ring = ~alpha
    source_surround = np.median(source[ring], axis=0)
    bbox = rgba.getchannel('A').getbbox()
    if bbox is None:
        raise ValueError('empty foreground alpha')
    return rgba.crop(bbox).convert('RGB'), rgba.getchannel('A').crop(bbox), source_surround


def choose_placement(bg: Image.Image, annotations: list[dict], airplanes: list[dict],
                     fg_size: tuple[int, int], rng: random.Random) -> tuple[int, int, dict, dict] | None:
    width, height = bg.size
    fw, fh = fg_size
    existing = []
    for ann in annotations:
        x, y, w, h = ann['bbox']
        existing.append((int(x), int(y), int(x + w), int(y + h)))
    rgb = np.asarray(bg, dtype=np.float32)
    possibilities = []
    hosts = sorted(airplanes, key=lambda a: a['bbox'][2] * a['bbox'][3], reverse=True)[:8]
    for host in hosts:
        hx, hy, hw, hh = host['bbox']
        hcx, hcy = hx + hw / 2, hy + hh / 2
        side = max(hw, hh)
        # Sample a ring around an existing aircraft, which is a weak but useful
        # proxy for airport/parking context. Use fixed angles for reproducibility.
        for angle_deg in range(0, 360, 30):
            angle = math.radians(angle_deg)
            for radius_factor in (1.4, 1.9, 2.4, 3.0):
                radius = max(side, max(fw, fh)) * radius_factor
                cx = hcx + radius * math.cos(angle)
                cy = hcy + radius * math.sin(angle)
                left, top = round(cx - fw / 2), round(cy - fh / 2)
                candidate = (left, top, left + fw, top + fh)
                if left < 10 or top < 10 or candidate[2] > width - 10 or candidate[3] > height - 10:
                    continue
                if any(close_to_box(candidate, box, pad=8) for box in existing):
                    continue
                target = rgb[top:top + fh, left:left + fw]
                if target.size == 0:
                    continue
                # Similar local surface to the vicinity of the reference plane.
                margin = int(.5 * side)
                x1, y1 = max(0, int(hx) - margin), max(0, int(hy) - margin)
                x2, y2 = min(width, int(hx + hw) + margin), min(height, int(hy + hh) + margin)
                context = rgb[y1:y2, x1:x2]
                if context.size == 0:
                    continue
                target_luma = target.mean(axis=2)
                context_luma = context.mean(axis=2)
                luma_diff = abs(float(np.median(target_luma)) - float(np.median(context_luma)))
                # Avoid highly structured buildings/vegetation while preserving
                # textured apron and runway surfaces.
                texture = float(target_luma.std())
                if texture > 48 or luma_diff > 38:
                    continue
                score = luma_diff * .9 + texture * .25 + radius_factor * 2
                score += rng.random() * 2
                possibilities.append((score, left, top, host, {'luma_diff': luma_diff,
                                                                 'texture_std': texture,
                                                                 'radius_factor': radius_factor}))
    if not possibilities:
        return None
    _, x, y, host, metrics = min(possibilities, key=lambda v: v[0])
    return x, y, host, metrics


def color_correct(fg: Image.Image, source_surround: np.ndarray, destination: np.ndarray) -> Image.Image:
    target_surround = np.median(destination.reshape(-1, 3), axis=0)
    shift = np.clip((target_surround - source_surround) * .55, -25, 25)
    data = np.asarray(fg, dtype=np.float32)
    return Image.fromarray(np.clip(data + shift, 0, 255).astype('uint8'), 'RGB')


def draw_box(image: Image.Image, bbox: tuple[int, int, int, int], caption: str) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle(bbox, outline=(255, 87, 42), width=3)
    x1, y1, _, _ = bbox
    draw.rectangle((x1, max(0, y1 - 21), x1 + 210, y1), fill=(255, 87, 42))
    draw.text((x1 + 3, max(0, y1 - 18)), caption, fill='white')
    return result


def save_label(path: Path, class_id: int, bbox: tuple[int, int, int, int], size: tuple[int, int]) -> None:
    x1, y1, x2, y2 = bbox
    width, height = size
    path.write_text(f'{class_id} {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} '
                    f'{(x2-x1)/width:.6f} {(y2-y1)/height:.6f}\n')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--seed', type=int, default=20260924)
    ap.add_argument('--count', type=int, default=10, help='Number of preview cases; repeats the ten-class/source schedule.')
    args = ap.parse_args()
    if args.count < 1:
        ap.error('--count must be positive')
    rng = random.Random(args.seed)
    backgrounds = load_backgrounds()
    rng.shuffle(backgrounds)
    foregrounds = load_foregrounds()
    for part in ('comparison/train/images', 'comparison/train/labels',
                 'final/train/images', 'final/train/labels', 'foregrounds', 'backgrounds'):
        (args.output / part).mkdir(parents=True, exist_ok=True)
    records = []
    used_backgrounds = set()
    used_foregrounds = set()
    for index in range(1, args.count + 1):
        cycle_position = (index - 1) % len(EXAMPLE_CLASSES) + 1
        class_name = EXAMPLE_CLASSES[cycle_position - 1]
        preferred_source = 'fixed' if cycle_position in FIXED_CYCLE_POSITIONS else 'innar_train'
        candidates = [item for item in foregrounds[class_name]
                      if item['source'] == preferred_source]
        rng.shuffle(candidates)
        success = None
        for fg_record in candidates[:60]:
            if fg_record['id'] in used_foregrounds:
                continue
            fg, alpha, source_surround = crop_foreground(fg_record)
            # Exclude tiny or corrupted source silhouettes before resizing.
            if min(fg.size) < 25:
                continue
            for bg_record in backgrounds:
                bg_path = bg_record['path']
                if bg_path in used_backgrounds:
                    continue
                bg = Image.open(bg_path).convert('RGB')
                plane_lengths = [max(a['bbox'][2:]) for a in bg_record['airplanes']]
                typical = float(np.median(plane_lengths))
                # A nearby aircraft supplies the image-specific pixel scale.
                long_side = int(np.clip(typical * rng.uniform(.85, 1.10), 85, 150))
                factor = long_side / max(fg.size)
                size = (max(12, round(fg.width * factor)), max(12, round(fg.height * factor)))
                if min(size) < 16 or max(size) >= min(bg.size) / 3:
                    continue
                pf = fg.resize(size, Image.Resampling.LANCZOS)
                pm = alpha.resize(size, Image.Resampling.LANCZOS)
                location = choose_placement(bg, bg_record['annotations'], bg_record['airplanes'], size, rng)
                if location is None:
                    continue
                x, y, host, placement_metrics = location
                destination = np.asarray(bg.crop((x, y, x + size[0], y + size[1])), dtype=np.float32)
                pf = color_correct(pf, source_surround, destination)
                # The hard alpha is the instance geometry. Feather only softens
                # a 1 px boundary; HBB is computed from the pre-feather mask.
                binary = pm.point(lambda a: 255 if a >= 128 else 0)
                feathered = binary.filter(ImageFilter.GaussianBlur(radius=.8))
                final = bg.copy()
                final.paste(pf, (x, y), feathered)
                cutout_box = binary.getbbox()
                if cutout_box is None:
                    continue
                bbox = (x + cutout_box[0], y + cutout_box[1],
                        x + cutout_box[2], y + cutout_box[3])
                success = (fg_record, bg_record, bg, final, pf, binary, bbox,
                           host, placement_metrics, long_side)
                break
            if success is not None:
                break
        if success is None:
            raise RuntimeError(f'No feasible placement for {class_name}')
        fg_record, bg_record, bg, final, pf, binary, bbox, host, metrics, long_side = success
        used_backgrounds.add(bg_record['path'])
        used_foregrounds.add(fg_record['id'])
        stem = f'{index:02d}__{class_name.replace("/", "-")}__{bg_record["path"].stem}'
        bg.save(args.output / 'backgrounds' / f'{stem}.jpg', quality=95)
        final.save(args.output / 'final/train/images' / f'{stem}.jpg', quality=96)
        cutout = Image.new('RGBA', pf.size)
        cutout.paste(pf, (0, 0))
        cutout.putalpha(binary)
        cutout.save(args.output / 'foregrounds' / f'{stem}.png')
        before = draw_box(bg, bbox, 'planned location')
        after = draw_box(final, bbox, class_name)
        panel = Image.new('RGB', (bg.width * 2, bg.height), 'white')
        panel.paste(before, (0, 0)); panel.paste(after, (bg.width, 0))
        panel.save(args.output / 'comparison/train/images' / f'{stem}.jpg', quality=95)
        save_label(args.output / 'comparison/train/labels' / f'{stem}.txt',
                   LABEL_NAMES.index(class_name), tuple(v + (bg.width if j % 2 == 0 else 0)
                                    for j, v in enumerate(bbox)), panel.size)
        save_label(args.output / 'final/train/labels' / f'{stem}.txt', LABEL_NAMES.index(class_name), bbox, bg.size)
        records.append({
            'id': stem, 'class': class_name, 'foreground_id': fg_record['id'],
            'foreground_path': fg_record['cutout_path'],
            'foreground_source': fg_record['source'],
            'foreground_source_path': fg_record['source_path'],
            'background_dataset': 'DIOR', 'background_split': 'train',
            'background_image_path': str(bg_record['path']),
            'background_annotation_image_id': bg_record['image']['id'],
            'host_airplane_bbox_xywh': host['bbox'],
            'pasted_bbox_xyxy': bbox, 'target_long_side_px': long_side,
            'placement_metrics': metrics, 'blend': 'local_color_shift_0.55_clip25+alpha_feather_0.8px',
            'train_ready': False, 'reason': 'Original DIOR labels use another taxonomy; synthetic label is preview-only.',
        })
    for ds in ('comparison', 'final'):
        (args.output / ds / 'data.yaml').write_text('names:\n' + ''.join(f'  {i}: {n}\n' for i, n in enumerate(LABEL_NAMES)))
    (args.output / 'manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    # Ten cases per contact-sheet page, so a forty-case audit stays readable.
    sheet_dir = args.output / 'contact_sheets'
    sheet_dir.mkdir(exist_ok=True)
    for start in range(0, len(records), 10):
        batch = records[start:start + 10]
        rows = math.ceil(len(batch) / 2)
        sheet = Image.new('RGB', (1600, rows * 430), 'white')
        sheet_draw = ImageDraw.Draw(sheet)
        for n, item in enumerate(batch):
            panel = Image.open(args.output / 'comparison/train/images' / f'{item["id"]}.jpg').convert('RGB')
            thumb = panel.resize((800, 400), Image.Resampling.LANCZOS)
            sx, sy = (n % 2) * 800, (n // 2) * 430
            sheet_draw.text((sx + 6, sy + 5),
                            f'{start + n + 1:02d} {item["class"]}  {item["foreground_source"]}', fill='black')
            sheet.paste(thumb, (sx, sy + 25))
        sheet.save(sheet_dir / f'page_{start // 10 + 1:02d}.jpg', quality=90)
        if start == 0:
            sheet.save(args.output / 'contact_sheet.jpg', quality=90)
    print(json.dumps({'count': len(records), 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
