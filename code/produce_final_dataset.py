#!/usr/bin/env python3
"""Stream accepted LAE composites into an independent, resumable final dataset."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
from PIL import Image

from augment_foregrounds_to_class_quota import CLASSES_13, BASE
from filter_copypaste_v7 import GATES, rejection_reasons, metrics
from synthesize_copypaste_v6 import (atomic_bytes, atomic_mask, box_overlap,
    choose_placement, foreground_premult, harmonize, resize_premult,
    rgb_to_lab, surface_reference, yolo_hbb_line, digest)
from synthesize_lae1m_v9 import LAE_ROOT, SPECS, load_lae_backgrounds

SUBSETS = list(SPECS)
FINAL_ROOT = Path('/data3/tianzhibei/derived/aircraft_copypaste_final_20260928')
VERSION = 'final-13class-v1-v9-mtarsi-v1-mar20-engineering-v7'


def class_quotas(cls, per_class=1000):
    if cls == 'E-2' and per_class == 1000:
        return {ds: 0 if ds == 'xView' else 200 for ds in SUBSETS}
    ci = CLASSES_13.index(cls)
    return {ds: per_class // 6 + int((i-ci) % 6 < per_class % 6)
            for i, ds in enumerate(SUBSETS)}


def taxonomy():
    original = set()
    for ds, spec in SPECS.items():
        p = LAE_ROOT / spec.get('root', ds if ds != 'xView' else 'xview') / spec['manifest']
        original.update(x['name'] for x in json.loads(p.read_text())['categories'])
    return CLASSES_13 + ['LAE/' + name for name in sorted(original)]


def foregrounds():
    groups = defaultdict(lambda: defaultdict(list))
    for src in ('mtarsi', 'mar20'):
        path = BASE / f'foreground_{src}_13class_aug_v1/manifest.jsonl'
        for line in path.read_text().splitlines():
            row = json.loads(line)
            groups[row['class']][src].append(row)
    return groups


@lru_cache(maxsize=2048)
def source_values(cutout, context):
    rgba = np.asarray(Image.open(cutout).convert('RGBA'))
    rgb = np.asarray(Image.open(context).convert('RGB'))
    mask = rgba[:, :, 3] > 127
    if rgb.shape[:2] != mask.shape or mask.sum() < 100:
        raise ValueError('bad source mask')
    ring = cv2.dilate(mask.astype('uint8'), np.ones((11, 11), np.uint8)) > 0
    ring &= ~mask
    if ring.sum() < 50:
        raise ValueError('bad source ring')
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref = np.median(lab[ring], axis=0)
    like = float((np.linalg.norm(lab[mask] - ref, axis=1) < 15).mean())
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'), 8)
    areas = sorted(stats[1:n, cv2.CC_STAT_AREA].tolist(), reverse=True)
    return {'source_background_like_fraction': like,
            'secondary_mask_fraction': float(sum(areas[1:]) / sum(areas))}


def json_write(path, obj):
    atomic_bytes(path, (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode())


def run(ds, root, per_class, seed, class_name=None):
    cv2.setNumThreads(1)
    rng = random.Random(seed + SUBSETS.index(ds))
    names = json.loads((root/'taxonomy.json').read_text()) if (root/'taxonomy.json').exists() else taxonomy()
    names_ids = {name: i for i, name in enumerate(names)}
    pools = foregrounds()
    for sources in pools.values():
        for pool in sources.values(): rng.shuffle(pool)
    backgrounds, ratios, _ = load_lae_backgrounds(ds)
    p10, p50, p90 = np.quantile(ratios, [.1, .5, .9])
    active_classes = [class_name] if class_name else CLASSES_13
    output = root / 'shards' / ds
    if class_name:
        output = output / class_name.replace('/', '-')
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'manifest.jsonl'
    existing = [json.loads(s) for s in manifest.read_text().splitlines()] if manifest.exists() else []
    counts = Counter(r['class_name'] for r in existing)
    quotas = {cls: class_quotas(cls, per_class)[ds] for cls in active_classes}
    stats_path = output / 'progress.json'
    previous = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    attempts = Counter(previous.get('attempts_by_class', {}))
    candidate_counts = Counter(previous.get('composites_checked_by_class', {}))
    rejects = Counter(previous.get('rejection_reasons', {}))
    rejects_by_class = defaultdict(Counter, {k: Counter(v) for k, v in previous.get('rejection_reasons_by_class', {}).items()})
    placement_fail = Counter(previous.get('generation_failures', {}))
    cache = OrderedDict()
    placement_cache = OrderedDict()
    used_bg = Counter(r['source_background_id'] for r in existing)
    start = time.monotonic()

    def background(bg):
        key = bg['id']
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        image = np.asarray(Image.open(bg['image']).convert('RGB'))
        value = (image, rgb_to_lab(image), surface_reference(image, bg['plane_boxes'])[0])
        cache[key] = value
        if len(cache) > 8: cache.popitem(last=False)
        return value

    def save_progress(done=False):
        json_write(stats_path, {'dataset': ds, 'version': VERSION, 'completed': done,
            'quota_by_class': quotas, 'passed_by_class': dict(counts),
            'attempts_by_class': dict(attempts), 'composites_checked_by_class': dict(candidate_counts),
            'rejection_reasons': dict(rejects),
            'rejection_reasons_by_class': {k: dict(v) for k, v in rejects_by_class.items()},
            'generation_failures': dict(placement_fail), 'unique_backgrounds': len(used_bg),
            'elapsed_this_run_seconds': round(time.monotonic() - start, 2),
            'quality_gate': GATES, 'rejected_images_saved': 0})

    with manifest.open('a', buffering=1) as handle:
        while any(counts[c] < quotas[c] for c in active_classes):
            for cls in active_classes:
                if counts[cls] >= quotas[cls]: continue
                attempts[cls] += 1
                if attempts[cls] > quotas[cls] * 500:
                    save_progress()
                    raise RuntimeError(f'{ds}/{cls}: attempt limit reached at {counts[cls]}/{quotas[cls]}')
                available = sorted(pools[cls])
                src = available[(attempts[cls] - 1) % len(available)]
                pool = pools[cls][src]
                item = pool[((attempts[cls] - 1) // len(available)) % len(pool)]
                try:
                    source = source_values(item['cutout_path'], item['source_path'])
                    if source['source_background_like_fraction'] > GATES['max_source_background_like_fraction']:
                        placement_fail[cls + ':source_background_leak'] += 1
                        continue
                    if source['secondary_mask_fraction'] > GATES['max_secondary_mask_fraction']:
                        placement_fail[cls + ':mask_fragments'] += 1
                        continue
                    angle = rng.uniform(-15, 15)
                    premult, alpha, context = foreground_premult(item, angle)
                except (ValueError, OSError):
                    placement_fail[cls + ':bad_foreground'] += 1
                    continue
                ys, xs = np.where(alpha > .5)
                if not len(xs):
                    placement_fail[cls + ':empty_mask'] += 1
                    continue
                source_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
                bg = rng.choice(backgrounds)
                before, lab, ref = background(bg)
                h, w = before.shape[:2]
                local = np.median([(b[2]-b[0])*(b[3]-b[1])/(w*h) for b in bg['plane_boxes']])
                ratio = float(np.clip(math.sqrt(local*p50) * rng.uniform(.85, 1.15), p10, p90))
                scale = math.sqrt(ratio*w*h/source_area) * float(item.get('paste_scale_factor', 1))
                # Four-pixel size bins retain scene scale while allowing placement caching.
                size = tuple(max(4, round(v*scale/4)*4) for v in (premult.shape[1], premult.shape[0]))
                if min(size) < 18 or max(size) > min(w, h)/3:
                    placement_fail[cls + ':bad_scale'] += 1
                    continue
                key = (bg['id'], size)
                if key in placement_cache:
                    position = placement_cache[key]
                    placement_cache.move_to_end(key)
                else:
                    position = choose_placement(before, bg['boxes'], bg['plane_boxes'], size, rng, lab, ref)
                    placement_cache[key] = position
                    if len(placement_cache) > 10000: placement_cache.popitem(last=False)
                if position is None:
                    placement_fail[cls + ':no_placement'] += 1
                    continue
                x, y, pm = position
                if (pm['texture_std_lab'] > GATES['max_placement_texture_std_lab']
                    or pm['vegetation_fraction'] > GATES['max_placement_vegetation_fraction']
                    or pm['edge_fraction'] > GATES['max_placement_edge_fraction']
                    or pm['color_distance_lab'] > GATES['max_placement_color_distance_lab']):
                    placement_fail[cls + ':strict_placement_gate'] += 1
                    continue
                fg, mask = resize_premult(premult, alpha, size)
                roi = before[y:y+size[1], x:x+size[0]]
                fg, soft, blend = harmonize(fg, mask, context, roi)
                feather = rng.uniform(.50, .80)
                soft = np.clip(cv2.GaussianBlur(mask.astype('float32'), (0, 0), feather), 0, 1)
                blend['feather_sigma'] = feather
                after = before.copy()
                after[y:y+size[1], x:x+size[0]] = np.clip(fg*soft[:, :, None] + roi*(1-soft[:, :, None]), 0, 255).astype('uint8')
                bx, by, bw, bh = cv2.boundingRect(cv2.findNonZero((mask > .5).astype('uint8')))
                box = (x+bx, y+by, x+bx+bw, y+by+bh)
                if any(box_overlap(box, b, 8) for b in bg['boxes']):
                    placement_fail[cls + ':mask_collision'] += 1
                    continue
                full_mask = np.zeros((h, w), 'uint8')
                full_mask[y:y+size[1], x:x+size[0]] = (mask > .5).astype('uint8')*255
                # Gate the actual JPEG pixels in memory; rejected bytes never touch disk.
                ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(after, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok: raise RuntimeError('JPEG encoding failed')
                decoded = np.asarray(Image.open(io.BytesIO(encoded.tobytes())).convert('RGB'))
                hard = full_mask > 0
                bg_pixels = before[hard]
                out_pixels = decoded[hard]
                bg_lab = cv2.cvtColor(bg_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB)
                out_lab = cv2.cvtColor(out_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB)
                bg_sat = float(np.median(bg_pixels.max(axis=1).astype(float)-bg_pixels.min(axis=1)))
                out_sat = float(np.median(out_pixels.max(axis=1).astype(float)-out_pixels.min(axis=1)))
                value = dict(source, placement_texture_std_lab=pm['texture_std_lab'],
                    placement_vegetation_fraction=pm['vegetation_fraction'], placement_edge_fraction=pm['edge_fraction'],
                    placement_color_distance_lab=pm['color_distance_lab'],
                    paste_lightness_delta=float(np.median(out_lab[:, :, 0]))-float(np.median(bg_lab[:, :, 0])),
                    paste_saturation_delta=out_sat-bg_sat, background_saturation=bg_sat, paste_saturation=out_sat)
                candidate_counts[cls] += 1
                reasons = rejection_reasons(value)
                if reasons:
                    rejects.update(reasons)
                    rejects_by_class[cls].update(reasons)
                    continue
                ordinal = counts[cls] + 1
                stem = f'{CLASSES_13.index(cls):02d}__{cls.replace("/", "-")}__{ds}__{ordinal:04d}'
                image_path = output / 'train/images' / (stem + '.jpg')
                label_path = output / 'train/labels' / (stem + '.txt')
                mask_path = output / 'instance_masks' / (stem + '.png')
                lines = [yolo_hbb_line(CLASSES_13.index(cls), box, w, h)]
                for ann in bg['annotations']:
                    xx, yy, ww, hh = ann['bbox_xywh']
                    b = (max(0, xx), max(0, yy), min(w, xx+ww), min(h, yy+hh))
                    if b[2] > b[0] and b[3] > b[1]:
                        lines.append(yolo_hbb_line(names_ids['LAE/'+ann['category_name']], b, w, h))
                label_bytes = ('\n'.join(lines)+'\n').encode()
                atomic_bytes(image_path, encoded.tobytes())
                atomic_bytes(label_path, label_bytes)
                atomic_mask(mask_path, full_mask)
                row = {'id': stem, 'pipeline_version': VERSION, 'class_name': cls,
                    'target_class_id': CLASSES_13.index(cls), 'subdataset': ds,
                    'foreground_source': src, 'foreground_base_id': item.get('base_foreground_id', item['id']),
                    'foreground_id': item['id'], 'foreground_path': item['cutout_path'],
                    'foreground_source_path': item['source_path'], 'foreground_augmentation': item.get('augmentation'),
                    'rotation_deg': angle, 'paste_scale_factor': item.get('paste_scale_factor'),
                    'source_background_id': bg['id'], 'background_image_path': str(bg['image']),
                    'background_split': 'train', 'background_label_path': str(bg['label']),
                    'image_path': str(image_path), 'label_path': str(label_path), 'instance_mask_path': str(mask_path),
                    'image_sha256': hashlib.sha256(encoded.tobytes()).hexdigest(),
                    'label_sha256': hashlib.sha256(label_bytes).hexdigest(), 'bbox_xyxy': list(box),
                    'width': w, 'height': h, 'original_annotations': bg['annotations'],
                    'placement': pm, 'blend': blend, 'automatic_metrics': value,
                    'quality_gate_version': 'engineering-v7-1', 'quality_gate_passed': True,
                    'train_ready': True, 'label_policy': '13_finegrained_plus_original_LAE_classes_HBB',
                    'visual_review_performed': False}
                # Verify equivalence with the unchanged file-based gate on the first pass per class.
                if counts[cls] == 0:
                    checked = metrics(row)
                    if rejection_reasons(checked) or any(abs(checked[k]-value[k]) > 1e-5 for k in value):
                        raise RuntimeError(f'gate equivalence failed: {stem}')
                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
                counts[cls] += 1
                used_bg[bg['id']] += 1
                if sum(counts.values()) % 25 == 0:
                    save_progress()
                    print(json.dumps({'dataset': ds, 'passed': sum(counts.values()),
                        'goal': sum(quotas.values()), 'counts': dict(counts),
                        'checked': sum(candidate_counts.values()), 'elapsed': round(time.monotonic()-start)}), flush=True)
        save_progress(True)
    print(json.dumps({'dataset': ds, 'done': True, 'passed': sum(counts.values())}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=SUBSETS, required=True)
    ap.add_argument('--root', type=Path, default=FINAL_ROOT)
    ap.add_argument('--per-class', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=20260928)
    ap.add_argument('--class-name', choices=CLASSES_13)
    args = ap.parse_args()
    run(args.dataset, args.root.resolve(), args.per_class, args.seed, args.class_name)
