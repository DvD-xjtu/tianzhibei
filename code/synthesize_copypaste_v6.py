#!/usr/bin/env python3
"""Split-safe MTARSI copy-paste smoke pipeline with explicit label semantics.

DIOR mode: LAE-1M visual preview; original coarse labels stay separate.
MAR20 mode: original YOLO-OBB labels are retained and new same-taxonomy OBBs
are appended, producing a train-formatted smoke sample. No source files change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

FOREGROUND_ROOT = Path('/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/foreground_all_15taxonomy_v6')
C5_SMOKE_ROOT = Path('/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/foreground_auto_large_v4_100')
DIOR_ROOT = Path('/data2/tianzhibei/LAE-1M/LAE-FOD/DIOR')
MAR20_ROOT = Path('/data3/tianzhibei/datasets/MAR20/yolo_obb/train')
MAR20_NAMES = [f'A{i}' for i in range(1, 21)]
CLASS_ID = {'C-130': 1, 'C-17': 2, 'C-5': 3, 'F-16': 4, 'E-3': 6,
            'B-52': 7, 'B-1B': 9, 'F-15': 12, 'F/A-18': 15}
CLASS_CYCLE = list(CLASS_ID)
SCRIPT_VERSION = 'copy-paste-v6-smoke-1'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_image(path: Path, rgb: np.ndarray) -> None:
    ext = path.suffix.lower()
    ok, encoded = cv2.imencode(ext, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 95] if ext == '.jpg' else [])
    if not ok:
        raise RuntimeError(f'cannot encode {path}')
    atomic_bytes(path, encoded.tobytes())


def atomic_mask(path: Path, mask: np.ndarray) -> None:
    ok, encoded = cv2.imencode('.png', mask.astype('uint8'))
    if not ok:
        raise RuntimeError(f'cannot encode {path}')
    atomic_bytes(path, encoded.tobytes())


def box_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or
                a[3] + pad <= b[1] or b[3] + pad <= a[1])


def load_foregrounds() -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for shard in range(4):
        file = FOREGROUND_ROOT / f'shard_{shard}' / 'manifest_quality_gated_v2.json'
        for item in json.loads(file.read_text()):
            if (item['quality_gate']['accepted'] and item['class'] in CLASS_ID
                    and item['source'] in {'fixed', 'innar_train'}):
                item['foreground_pool'] = 'full_15taxonomy_v2_gate'
                groups[item['class']].append(item)
    # C-5 was outside the initial 15-class full run. Five v4 smoke cases have
    # passed the same structural gate, enough for this small v6 experiment.
    for item in json.loads((C5_SMOKE_ROOT / 'manifest_quality_gated.json').read_text()):
        if (item['class'] == 'C-5' and item['quality_gate']['accepted']
                and item['source'] in {'fixed', 'innar_train'}):
            item['foreground_pool'] = 'C5_v4_100_quality_gate'
            groups['C-5'].append(item)
    for group in groups.values():
        group.sort(key=lambda item: item['id'])
    return groups


def load_augmented_foregrounds(path: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get('gate') != 'passed' or item.get('class') not in CLASS_ID:
            continue
        item['foreground_pool'] = item.get('foreground_pool', 'augmented_foregrounds')
        groups[item['class']].append(item)
    for group in groups.values():
        group.sort(key=lambda item: item['id'])
    return groups


def parse_mar20_label(label: Path, width: int, height: int) -> tuple[list[str], list[dict]]:
    raw_lines = [line.strip() for line in label.read_text().splitlines() if line.strip()]
    lines = []
    annotations = []
    for line in raw_lines:
        values = line.split()
        if len(values) != 9:
            raise ValueError(f'bad MAR20 OBB: {label}: {line}')
        cls = int(values[0])
        if not 0 <= cls < 20:
            raise ValueError(f'bad MAR20 class: {label}: {cls}')
        points = np.asarray([float(v) for v in values[1:]], dtype=np.float32).reshape(4, 2)
        if not np.isfinite(points).all() or (points < -.05).any() or (points > 1.05).any():
            raise ValueError(f'bad MAR20 coordinates: {label}: {line}')
        # Some source OBB vertices exceed the image edge by a few thousandths.
        # Clip rather than writing invalid normalized output labels.
        points = np.clip(points, 0, 1)
        lines.append(str(cls) + ' ' + ' '.join(f'{v:.6f}' for v in points.reshape(-1)))
        points *= [width, height]
        x1, y1 = np.floor(points.min(axis=0)).astype(int)
        x2, y2 = np.ceil(points.max(axis=0)).astype(int)
        annotations.append({'bbox': (x1, y1, x2, y2), 'class_id': cls})
    return lines, annotations


def load_backgrounds(mode: str) -> list[dict]:
    if mode == 'dior':
        data = json.loads((DIOR_ROOT / 'DIOR_train.json').read_text())
        airplane_id = next(c['id'] for c in data['categories'] if c['name'] == 'airplane')
        annotations: dict[int, list[dict]] = defaultdict(list)
        for ann in data['annotations']:
            annotations[ann['image_id']].append(ann)
        result = []
        for image in sorted(data['images'], key=lambda item: item['id']):
            anns = annotations[image['id']]
            planes = [a for a in anns if a['category_id'] == airplane_id]
            if len(planes) < 2:
                continue
            path = DIOR_ROOT / 'JPEGImages-trainval' / image['file_name']
            if not path.is_file():
                continue
            boxes = [tuple(map(int, (a['bbox'][0], a['bbox'][1],
                                    a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3])))
                     for a in anns]
            plane_boxes = [tuple(map(int, (a['bbox'][0], a['bbox'][1],
                                          a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3])))
                           for a in planes]
            result.append({'id': f'dior_{image["id"]}', 'image': path, 'label': DIOR_ROOT / 'DIOR_train.json',
                           'boxes': boxes, 'plane_boxes': plane_boxes,
                           'original_annotations': anns, 'original_lines': None})
        return result

    result = []
    for image in sorted((MAR20_ROOT / 'images').glob('*.jpg')):
        label = MAR20_ROOT / 'labels' / f'{image.stem}.txt'
        if not label.is_file():
            continue
        with Image.open(image) as im:
            width, height = im.size
        lines, anns = parse_mar20_label(label, width, height)
        if len(anns) < 2:
            continue
        boxes = [a['bbox'] for a in anns]
        result.append({'id': f'mar20_{image.stem}', 'image': image, 'label': label,
                       'boxes': boxes, 'plane_boxes': boxes,
                       'original_annotations': anns, 'original_lines': lines})
    return result


def class_scale_prior() -> dict[int, tuple[float, float, float]]:
    """MAR20 train HBB area-ratio P10/P50/P90, never test statistics."""
    ratios: dict[int, list[float]] = defaultdict(list)
    for image in (MAR20_ROOT / 'images').glob('*.jpg'):
        label = MAR20_ROOT / 'labels' / f'{image.stem}.txt'
        if not label.is_file():
            continue
        with Image.open(image) as im:
            width, height = im.size
        _, anns = parse_mar20_label(label, width, height)
        for ann in anns:
            x1, y1, x2, y2 = ann['bbox']
            ratios[ann['class_id']].append((x2 - x1) * (y2 - y1) / (width * height))
    return {cls: tuple(float(v) for v in np.quantile(vals, [.10, .50, .90]))
            for cls, vals in ratios.items() if len(vals) >= 5}


def foreground_premult(item: dict, angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgba = np.asarray(Image.open(item['cutout_path']).convert('RGBA'), dtype=np.float32)
    source = np.asarray(Image.open(item['source_path']).convert('RGB'), dtype=np.uint8)
    alpha = rgba[:, :, 3] / 255.0
    hard = alpha > .5
    if hard.sum() < 100 or source.shape[:2] != alpha.shape:
        raise ValueError('empty or misaligned source mask')
    ring = cv2.dilate(hard.astype('uint8'), np.ones((13, 13), np.uint8)) > 0
    ring &= ~hard
    if ring.sum() < 50:
        ring = ~hard
    source_context = np.median(rgb_to_lab(source[ring].reshape(-1, 1, 3)).reshape(-1, 3), axis=0)
    points = cv2.findNonZero(hard.astype('uint8'))
    x, y, w, h = cv2.boundingRect(points)
    pad = 3
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(alpha.shape[1], x + w + pad), min(alpha.shape[0], y + h + pad)
    rgb = rgba[y1:y2, x1:x2, :3] / 255.0
    a = alpha[y1:y2, x1:x2]
    premult = rgb * a[:, :, None]
    center = ((x2 - x1) / 2, (y2 - y1) / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    abs_cos, abs_sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    out_w = math.ceil((y2 - y1) * abs_sin + (x2 - x1) * abs_cos)
    out_h = math.ceil((y2 - y1) * abs_cos + (x2 - x1) * abs_sin)
    matrix[0, 2] += out_w / 2 - center[0]
    matrix[1, 2] += out_h / 2 - center[1]
    premult = cv2.warpAffine(premult, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR)
    a = cv2.warpAffine(a, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR)
    return premult, a, source_context


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def resize_premult(premult: np.ndarray, alpha: np.ndarray,
                   size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    w, h = size
    p = cv2.resize(premult, (w, h), interpolation=cv2.INTER_AREA if w < premult.shape[1] else cv2.INTER_LINEAR)
    a = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA if w < alpha.shape[1] else cv2.INTER_LINEAR)
    rgb = np.clip(p / np.maximum(a[:, :, None], 1e-4), 0, 1)
    return (rgb * 255).astype('uint8'), np.clip(a, 0, 1)


def surface_reference(image: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    blocked = np.zeros((height, width), np.uint8)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(blocked, (max(0, x1 - 4), max(0, y1 - 4)),
                      (min(width - 1, x2 + 4), min(height - 1, y2 + 4)), 1, -1)
    samples = []
    for x1, y1, x2, y2 in boxes[:8]:
        side = max(x2 - x1, y2 - y1)
        pad = int(np.clip(side * .35, 12, 45))
        a, b = max(0, x1 - pad), max(0, y1 - pad)
        c, d = min(width, x2 + pad), min(height, y2 + pad)
        patch = image[b:d, a:c]
        valid = blocked[b:d, a:c] == 0
        if valid.sum() > 100:
            samples.append(patch[valid])
    pixels = np.concatenate(samples) if samples else image.reshape(-1, 3)
    lab = rgb_to_lab(pixels.reshape(-1, 1, 3)).reshape(-1, 3)
    return np.median(lab, axis=0), float(np.std(lab[:, 0]))


def choose_placement(image: np.ndarray, boxes: list[tuple[int, int, int, int]],
                     plane_boxes: list[tuple[int, int, int, int]], size: tuple[int, int],
                     rng: random.Random, cached_lab: np.ndarray | None = None,
                     cached_reference: np.ndarray | None = None) -> tuple[int, int, dict] | None:
    height, width = image.shape[:2]
    fw, fh = size
    if cached_lab is None or cached_reference is None:
        ref, _ = surface_reference(image, plane_boxes)
        lab = rgb_to_lab(image)
    else:
        ref, lab = cached_reference, cached_lab
    options = []
    for host in sorted(plane_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)[:8]:
        x1, y1, x2, y2 = host
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1, fw, fh)
        for radius_factor in (1.4, 1.9, 2.4):
            for degrees in range(0, 360, 30):
                angle = math.radians(degrees)
                x = round(cx + side * radius_factor * math.cos(angle) - fw / 2)
                y = round(cy + side * radius_factor * math.sin(angle) - fh / 2)
                candidate = (x, y, x + fw, y + fh)
                if x < 8 or y < 8 or candidate[2] > width - 8 or candidate[3] > height - 8:
                    continue
                if any(box_overlap(candidate, box, 8) for box in boxes):
                    continue
                patch = lab[y:y+fh, x:x+fw]
                rgb_patch = image[y:y+fh, x:x+fw].astype(np.float32)
                median = np.median(patch.reshape(-1, 3), axis=0)
                color_distance = float(np.linalg.norm(median - ref))
                texture = float(patch[:, :, 0].std())
                edge_fraction = float((cv2.Canny(image[y:y+fh, x:x+fw], 70, 150) > 0).mean())
                red, green, blue = rgb_patch[:, :, 0], rgb_patch[:, :, 1], rgb_patch[:, :, 2]
                vegetation_fraction = float(((green > red * 1.08) & (green > blue * 1.05)
                                             & ((green - red) > 5)).mean())
                if (color_distance > 10 or texture > 19 or edge_fraction > .11
                        or vegetation_fraction > .02):
                    continue
                score = color_distance + .25 * texture + 10 * edge_fraction + radius_factor
                score += rng.random() * 1.5
                options.append((score, x, y, {'color_distance_lab': color_distance,
                                              'texture_std_lab': texture,
                                              'edge_fraction': edge_fraction,
                                              'vegetation_fraction': vegetation_fraction,
                                              'radius_factor': radius_factor}))
    if not options:
        return None
    _, x, y, metrics = min(options, key=lambda row: row[0])
    return x, y, metrics


def harmonize(fg: np.ndarray, alpha: np.ndarray, source_lab: np.ndarray,
              target: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    target_lab = rgb_to_lab(target)
    target_context = np.median(target_lab.reshape(-1, 3), axis=0)
    delta = np.clip((target_context - source_lab) * .55, [-12, -6, -6], [12, 6, 6])
    fg_lab = rgb_to_lab(fg)
    fg_lab += delta
    fg_lab[:, :, 0] = np.clip(fg_lab[:, :, 0], 0, 100)
    fg_lab[:, :, 1:] = np.clip(fg_lab[:, :, 1:], -127, 127)
    corrected = np.clip(cv2.cvtColor(fg_lab, cv2.COLOR_LAB2RGB) * 255, 0, 255).astype('uint8')
    # Only modestly reduce sharpness when the inserted crop is much sharper than the target.
    source_lap = cv2.Laplacian(cv2.cvtColor(corrected, cv2.COLOR_RGB2GRAY), cv2.CV_32F)
    target_lap = cv2.Laplacian(cv2.cvtColor(target, cv2.COLOR_RGB2GRAY), cv2.CV_32F)
    sharp_fg = float(source_lap[alpha > .8].var()) if (alpha > .8).any() else 0.0
    sharp_bg = float(target_lap.var())
    sigma = .55 if sharp_fg > 2.5 * max(sharp_bg, 1) else 0.0
    if sigma:
        corrected = cv2.GaussianBlur(corrected, (0, 0), sigma)
    soft = cv2.GaussianBlur(alpha.astype('float32'), (0, 0), .65)
    soft = np.clip(soft, 0, 1)
    return corrected, soft, {'lab_shift': delta.tolist(), 'blur_sigma': sigma,
                             'premultiplied_resize': True, 'feather_sigma': .65}


def yolo_hbb_line(class_id: int, box: tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    return (f'{class_id} {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} '
            f'{(x2-x1)/width:.6f} {(y2-y1)/height:.6f}')


def yolo_obb_line(class_id: int, mask: np.ndarray, offset: tuple[int, int],
                  width: int, height: int) -> str:
    points = cv2.findNonZero((mask > .5).astype('uint8'))
    rect = cv2.minAreaRect(points)
    corners = cv2.boxPoints(rect)
    corners[:, 0] = np.clip(corners[:, 0] + offset[0], 0, width)
    corners[:, 1] = np.clip(corners[:, 1] + offset[1], 0, height)
    coords = (corners / [width, height]).reshape(-1)
    return str(class_id) + ' ' + ' '.join(f'{v:.6f}' for v in coords)


def draw_comparison(before: np.ndarray, after: np.ndarray, box: tuple[int, int, int, int],
                    class_name: str) -> np.ndarray:
    left, right = Image.fromarray(before).copy(), Image.fromarray(after).copy()
    for image, title in ((left, 'planned'), (right, class_name)):
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline=(255, 87, 42), width=3)
        draw.text((box[0] + 3, max(0, box[1] - 16)), title, fill=(255, 87, 42))
    return np.concatenate([np.asarray(left), np.asarray(right)], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['dior', 'mar20'], required=True)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20260924)
    parser.add_argument('--classes', type=str, default=','.join(CLASS_CYCLE),
                        help='comma-separated class names from the supported MAR20 mapping')
    parser.add_argument('--foreground-manifest', type=Path, default=None,
                        help='optional augmented foreground JSONL; otherwise use the standard accepted pool')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    classes = [name.strip() for name in args.classes.split(',')]
    if not classes or any(name not in CLASS_ID for name in classes) or len(classes) != len(set(classes)):
        parser.error('classes must be unique supported names')
    if args.count < 1 or args.output.exists() and any(args.output.iterdir()):
        parser.error('count must be positive and output must be new or empty')
    rng = random.Random(args.seed)
    foregrounds = load_augmented_foregrounds(args.foreground_manifest) if args.foreground_manifest else load_foregrounds()
    backgrounds = load_backgrounds(args.mode)
    rng.shuffle(backgrounds)
    # MAR20 train backgrounds are small enough to cache in RAM. The generator
    # examines many scenes per paste candidate; decoding JPEG and recomputing
    # full-image Lab/reference statistics inside that loop is needlessly slow.
    for bg in backgrounds:
        cached_image = np.asarray(Image.open(bg['image']).convert('RGB'))
        cached_lab = rgb_to_lab(cached_image)
        cached_reference, _ = surface_reference(cached_image, bg['plane_boxes'])
        bg['_image_array'] = cached_image
        bg['_lab_array'] = cached_lab
        bg['_surface_reference'] = cached_reference
    priors = class_scale_prior()
    used_fg: set[str] = set()
    used_bg: set[str] = set()
    records = []
    failures: Counter[str] = Counter()
    for index in range(1, args.count + 1):
        class_name = classes[(index - 1) % len(classes)]
        class_id = CLASS_ID[class_name]
        source = 'fixed' if index % 5 == 1 and any(x['source'] == 'fixed' for x in foregrounds[class_name]) else 'innar_train'
        candidates = [x for x in foregrounds[class_name] if x['source'] == source and x['id'] not in used_fg]
        rng.shuffle(candidates)
        success = None
        for item in candidates[:50]:
            angle = rng.uniform(-15, 15)
            try:
                premult, alpha, source_context = foreground_premult(item, angle)
            except (ValueError, OSError):
                failures['bad_foreground'] += 1
                continue
            hard = alpha > .5
            ys, xs = np.where(hard)
            if len(xs) == 0:
                failures['empty_mask_after_rotation'] += 1
                continue
            source_area = max(1, (xs.max()-xs.min()+1)*(ys.max()-ys.min()+1))
            for bg in backgrounds:
                if bg['id'] in used_bg:
                    continue
                image = bg['_image_array']
                height, width = image.shape[:2]
                plane_areas = [(b[2]-b[0])*(b[3]-b[1])/(width*height) for b in bg['plane_boxes']]
                local = float(np.median(plane_areas))
                p10, p50, p90 = priors[class_id]
                ratio = float(np.clip(math.sqrt(local*p50) * rng.uniform(.85, 1.15), p10, p90))
                scale = math.sqrt(ratio * width * height / source_area)
                paste_scale_factor = float(item.get('paste_scale_factor', 1.0))
                scale *= paste_scale_factor
                size = (round(premult.shape[1] * scale), round(premult.shape[0] * scale))
                if min(size) < 18 or max(size) > min(width, height) / 3:
                    failures['bad_scale'] += 1
                    continue
                fg, mask = resize_premult(premult, alpha, size)
                placement = choose_placement(image, bg['boxes'], bg['plane_boxes'], size, rng,
                                             bg['_lab_array'], bg['_surface_reference'])
                if placement is None:
                    failures['no_placement'] += 1
                    continue
                x, y, metrics = placement
                roi = image[y:y+size[1], x:x+size[0]]
                fg, soft, blend_info = harmonize(fg, mask, source_context, roi)
                composed = image.copy()
                composed[y:y+size[1], x:x+size[0]] = np.clip(
                    fg.astype('float32')*soft[:, :, None] + roi.astype('float32')*(1-soft[:, :, None]),
                    0, 255).astype('uint8')
                points = cv2.findNonZero((mask > .5).astype('uint8'))
                bx, by, bw, bh = cv2.boundingRect(points)
                box = (x+bx, y+by, x+bx+bw, y+by+bh)
                if any(box_overlap(box, existing, 8) for existing in bg['boxes']):
                    failures['mask_collision'] += 1
                    continue
                obb_line = yolo_obb_line(class_id, mask, (x, y), width, height)
                success = (item, bg, image, composed, fg, mask, box, (x, y), ratio,
                           paste_scale_factor,
                           angle, metrics, blend_info, obb_line)
                break
            if success:
                break
        if not success:
            raise RuntimeError(f'No feasible sample for {class_name}; generated {len(records)}/{args.count}')
        item, bg, before, after, fg, mask, box, offset, ratio, paste_scale_factor, angle, metrics, blend_info, obb_line = success
        used_fg.add(item['id']); used_bg.add(bg['id'])
        stem = f'{index:03d}__{class_name.replace("/", "-")}__{bg["id"]}'
        final_image = args.output / 'final/train/images' / f'{stem}.jpg'
        final_label = args.output / 'final/train/labels' / f'{stem}.txt'
        comparison_image = args.output / 'comparison/train/images' / f'{stem}.jpg'
        comparison_label = args.output / 'comparison/train/labels' / f'{stem}.txt'
        mask_path = args.output / 'instance_masks' / f'{stem}.png'
        atomic_image(final_image, after)
        atomic_image(comparison_image, draw_comparison(before, after, box, class_name))
        full_mask = np.zeros(before.shape[:2], np.uint8)
        full_mask[offset[1]:offset[1]+mask.shape[0], offset[0]:offset[0]+mask.shape[1]] = (mask > .5).astype('uint8') * 255
        atomic_mask(mask_path, full_mask)
        hbb = yolo_hbb_line(class_id, box, before.shape[1], before.shape[0])
        atomic_bytes(comparison_label, (yolo_hbb_line(class_id,
            (box[0]+before.shape[1], box[1], box[2]+before.shape[1], box[3]),
            before.shape[1]*2, before.shape[0])+'\n').encode())
        label_policy = 'MAR20_clipped_original_OBB_plus_synthetic_OBB' if args.mode == 'mar20' else 'synthetic_only;DIOR_original_separate'
        lines = bg['original_lines'] + [obb_line] if args.mode == 'mar20' else [hbb]
        atomic_bytes(final_label, ('\n'.join(lines)+'\n').encode())
        if args.mode == 'dior':
            original_file = args.output / 'original_dior_annotations' / f'{stem}.json'
            atomic_bytes(original_file, (json.dumps(bg['original_annotations'], ensure_ascii=False)+'\n').encode())
        record = {'schema_version': 1, 'pipeline_version': SCRIPT_VERSION,
                  'id': stem, 'mode': args.mode, 'class_name': class_name,
                  'mar20_class_id': class_id, 'foreground_id': item['id'],
                  'foreground_source': item['source'], 'foreground_path': item['cutout_path'],
                  'foreground_pool': item['foreground_pool'],
                  'foreground_base_id': item.get('base_foreground_id', item['id']),
                  'foreground_augmentation': item.get('augmentation'),
                  'paste_scale_factor': paste_scale_factor,
                  'foreground_sha256': digest(Path(item['cutout_path'])),
                  'foreground_source_path': item['source_path'], 'background_id': bg['id'],
                  'background_split': 'train', 'background_image_path': str(bg['image']),
                  'background_label_path': str(bg['label']), 'background_label_sha256': digest(bg['label']),
                  'image_path': str(final_image), 'image_sha256': digest(final_image),
                  'instance_mask_path': str(mask_path), 'instance_mask_sha256': digest(mask_path),
                  'label_path': str(final_label), 'label_sha256': digest(final_label),
                  'bbox_xyxy': box, 'rotation_deg': angle, 'target_hbb_area_ratio': ratio,
                  'actual_hbb_area_ratio': (box[2]-box[0])*(box[3]-box[1])/(before.shape[0]*before.shape[1]),
                  'placement': metrics, 'blend': blend_info, 'label_policy': label_policy,
                  'train_ready': args.mode == 'mar20', 'review_status': 'automatic_unreviewed'}
        records.append(record)
    names = MAR20_NAMES
    for sub in ('final', 'comparison'):
        atomic_bytes(args.output / sub / 'data.yaml',
                     ('names:\n'+''.join(f'  {i}: {name}\n' for i, name in enumerate(names))).encode())
    atomic_bytes(args.output / 'manifest.jsonl',
                 ''.join(json.dumps(record, ensure_ascii=False)+'\n' for record in records).encode())
    bg_manifest = [{'background_id': x['background_id'], 'split': 'train',
                    'image_path': x['background_image_path'], 'image_sha256': digest(Path(x['background_image_path'])),
                    'label_path': x['background_label_path'], 'label_sha256': x['background_label_sha256']}
                   for x in records]
    atomic_bytes(args.output / 'background_train.jsonl',
                 ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in bg_manifest).encode())
    summary = {'pipeline_version': SCRIPT_VERSION, 'mode': args.mode, 'requested': args.count,
               'generated': len(records), 'train_ready': args.mode == 'mar20',
               'generation_classes': classes,
               'foreground_manifest': str(args.foreground_manifest) if args.foreground_manifest else None,
               'foreground_sources': dict(Counter(x['foreground_source'] for x in records)),
               'classes': dict(Counter(x['class_name'] for x in records)), 'failures': dict(failures),
               'seed': args.seed, 'background_manifest_sha256': digest(args.output / 'background_train.jsonl')}
    atomic_bytes(args.output / 'qa/summary.json', (json.dumps(summary, ensure_ascii=False, indent=2)+'\n').encode())
    config = {'pipeline_version': SCRIPT_VERSION, 'mode': args.mode, 'count': args.count,
              'seed': args.seed, 'generation_classes': classes,
              'foreground_manifest': str(args.foreground_manifest) if args.foreground_manifest else None,
              'classes_mar20_zero_based': CLASS_ID,
              'sources': ['fixed', 'innar_train'], 'background_split': 'train',
              'rotation_deg': [-15, 15], 'placement': {'max_color_distance_lab': 10,
              'max_texture_std_lab': 19, 'max_edge_fraction': .11,
              'max_vegetation_fraction': .02, 'collision_pad_px': 8},
              'blend': {'premultiplied_resize': True, 'lab_shift_strength': .55,
                        'lab_shift_clip': [12, 6, 6], 'feather_sigma': .65,
                        'conditional_blur_sigma': .55}}
    atomic_bytes(args.output / 'generation_config.json',
                 (json.dumps(config, ensure_ascii=False, indent=2)+'\n').encode())
    for start in range(0, len(records), 10):
        chunk = records[start:start+10]
        sheet = Image.new('RGB', (1600, math.ceil(len(chunk)/2)*430), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, record in enumerate(chunk):
            panel = Image.open(args.output / 'comparison/train/images' / f'{record["id"]}.jpg')
            thumb = panel.resize((800, 400), Image.Resampling.LANCZOS)
            sx, sy = i % 2 * 800, i // 2 * 430
            draw.text((sx+6, sy+5), f'{record["id"]}  {record["foreground_source"]}', fill='black')
            sheet.paste(thumb, (sx, sy+25))
        page = args.output / 'qa/contact_sheets' / f'page_{start//10+1:02d}.jpg'
        page.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(page, quality=90)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
