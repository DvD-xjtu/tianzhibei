#!/usr/bin/env python3
"""Create one LAE-1M subdataset preview batch from augmented MTARSI cutouts.

Original LAE annotations are kept as sidecars. Synthetic MAR20 class labels are
preview labels only; this output is deliberately not declared train-ready.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from augment_mtarsi_foregrounds_v1 import CLASSES, sha256
from synthesize_copypaste_v6 import (
    CLASS_ID, MAR20_NAMES, atomic_bytes, atomic_image, atomic_mask, box_overlap,
    choose_placement, digest, draw_comparison, foreground_premult, harmonize,
    load_augmented_foregrounds, parse_mar20_label, resize_premult, rgb_to_lab,
    surface_reference, yolo_hbb_line,
)

LAE_ROOT = Path('/data2/tianzhibei/LAE-1M/LAE-FOD')
PIPELINE_VERSION = 'copy-paste-v9-lae-preview-1'
SPECS = {
    'DIOR': {'manifest': 'processed_LAE-1M_DIOR_train.json', 'images': 'JPEGImages-trainval',
             'aircraft': {'airplane'}},
    'DOTAv2': {'manifest': 'processed_LAE-1M_DOTAv2_train.json', 'images': 'images',
               'aircraft': {'plane'}},
    'FAIR1M': {'manifest': 'processed_LAE-1M_FAIR1M_train.json', 'images': 'images',
               'aircraft': {'A220', 'A321', 'A330', 'A350', 'ARJ21', 'Boeing737',
                            'Boeing747', 'Boeing777', 'Boeing787', 'C919', 'other airplane'}},
    'NWPU': {'root': 'NWPU VHR-10', 'manifest': 'processed_LAE-1M_NWPU-VHR-10_train.json', 'images': 'images',
             'aircraft': {'airplane'}},
    'RSOD': {'manifest': 'processed_LAE-1M_RSOD_train.json', 'images': 'images',
             'aircraft': {'aircraft'}},
    'xView': {'manifest': 'processed_LAE-1M_Xview_train_1024_05.json', 'images': 'train_images_1024_05',
              'aircraft': {'Small Aircraft', 'Cargo Plane'}},
}


def hbb_from_xywh(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = map(float, box)
    x1 = int(np.clip(math.floor(x), 0, width - 1))
    y1 = int(np.clip(math.floor(y), 0, height - 1))
    x2 = int(np.clip(math.ceil(x + w), x1 + 1, width))
    y2 = int(np.clip(math.ceil(y + h), y1 + 1, height))
    return x1, y1, x2, y2


def load_lae_backgrounds(dataset: str, split: str = 'train') -> tuple[list[dict], list[float], str]:
    if split not in ('train', 'valid'):
        raise ValueError(f'unsupported LAE background split: {split}')
    if split == 'valid' and dataset != 'FAIR1M':
        raise ValueError('only FAIR1M valid_ tiles have an audited independent background pool')
    spec = SPECS[dataset]
    root = LAE_ROOT / spec.get('root', dataset if dataset != 'xView' else 'xview')
    manifest_path = root / spec['manifest']
    data = json.loads(manifest_path.read_text())
    categories = {int(x['id']): str(x['name']) for x in data['categories']}
    aircraft_ids = {i for i, name in categories.items() if name in spec['aircraft']}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data.get('annotations', []):
        anns_by_image[int(ann['image_id'])].append(ann)
    backgrounds = []
    all_plane_ratios = []
    for item in data['images']:
        name = item['file_name']
        # FAIR1M's processed index contains separate train_ and valid_ tiles.
        if dataset == 'FAIR1M' and not name.startswith(split + '_'):
            continue
        path = root / spec['images'] / name
        if not path.is_file():
            continue
        width, height = int(item['width']), int(item['height'])
        anns = anns_by_image[int(item['id'])]
        boxes, plane_boxes, saved_anns = [], [], []
        for ann in anns:
            category_id = int(ann['category_id'])
            if category_id not in categories:
                continue
            bbox = hbb_from_xywh(ann['bbox'], width, height)
            boxes.append(bbox)
            if category_id in aircraft_ids:
                plane_boxes.append(bbox)
                all_plane_ratios.append((bbox[2]-bbox[0])*(bbox[3]-bbox[1])/(width*height))
            saved_anns.append({'category_id': category_id, 'category_name': categories[category_id],
                               'bbox_xywh': [float(v) for v in ann['bbox']]})
        if not plane_boxes:
            continue
        backgrounds.append({'id': f'{dataset.lower()}_{Path(name).stem}', 'dataset': dataset,
                            'split': split, 'image': path, 'label': manifest_path,
                            'width': width, 'height': height, 'boxes': boxes,
                            'plane_boxes': plane_boxes, 'annotations': saved_anns,
                            'label_format': 'processed_COCO_bbox_xywh'})
    if not backgrounds:
        raise RuntimeError(f'no eligible {split} backgrounds: {dataset}')
    return backgrounds, all_plane_ratios, str(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=sorted(SPECS), required=True)
    parser.add_argument('--count', type=int, default=600,
                        help='number of candidate composites for this LAE-1M subdataset')
    parser.add_argument('--seed', type=int, default=20260930)
    parser.add_argument('--index-offset', type=int, default=0,
                        help='ID index offset for supplemental batches')
    parser.add_argument('--exclude-foreground-ids', type=Path, default=None,
                        help='newline-delimited foreground IDs already used in an earlier batch')
    parser.add_argument('--foreground-manifest', type=Path, required=True)
    parser.add_argument('--foreground-source', choices=('mtarsi', 'mar20'), default='mtarsi')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1 or (args.output.exists() and any(args.output.iterdir())):
        parser.error('count must be positive and output must be new or empty')
    rng = random.Random(args.seed)
    foregrounds = load_augmented_foregrounds(args.foreground_manifest)
    backgrounds, ratios, label_path = load_lae_backgrounds(args.dataset)
    p10, p50, p90 = [float(x) for x in np.quantile(ratios, [.10, .50, .90])]
    rng.shuffle(backgrounds)
    background_pool = (rng.sample(backgrounds, args.count)
                       if len(backgrounds) > args.count else list(backgrounds))
    # Reuse scenes only when the subdataset has fewer airplane-labelled train
    # scenes than requested. Track repeats explicitly in the output IDs.
    bg_slots = [background_pool[i % len(background_pool)] for i in range(args.count)]
    rng.shuffle(bg_slots)
    cache = {}
    for bg in background_pool:
        image = np.asarray(Image.open(bg['image']).convert('RGB'))
        bg['_image_array'] = image
        bg['_lab_array'] = rgb_to_lab(image)
        bg['_surface_reference'], _ = surface_reference(image, bg['plane_boxes'])
        cache[bg['id']] = bg
    used_fg: set[str] = set(args.exclude_foreground_ids.read_text().splitlines()) if args.exclude_foreground_ids else set()
    used_base: Counter[str] = Counter()
    records = []
    failures: Counter[str] = Counter()
    classes = list(CLASSES)
    for index in range(1, args.count + 1):
        class_name = classes[(index - 1) % len(classes)]
        class_id = CLASS_ID[class_name]
        pool = foregrounds.get(class_name, [])
        if not pool:
            raise RuntimeError(f'no accepted augmented foregrounds for {class_name}')
        candidates = [x for x in pool if x['id'] not in used_fg]
        rng.shuffle(candidates)
        primary_bg = bg_slots[index - 1]
        success = None
        for item in candidates[:80]:
            angle = rng.uniform(-15, 15)
            try:
                premult, alpha, source_context = foreground_premult(item, angle)
            except (ValueError, OSError):
                failures['bad_foreground'] += 1
                continue
            hard = alpha > .5
            ys, xs = np.where(hard)
            if not len(xs):
                failures['empty_mask_after_rotation'] += 1
                continue
            source_area = max(1, (xs.max()-xs.min()+1)*(ys.max()-ys.min()+1))
            # Sample a bounded number of distinct scenes for each foreground.
            # If the assigned scene fails, the fallback is still same-subset train.
            trial_bgs = [primary_bg] + rng.sample(background_pool, min(23, len(background_pool)))
            seen_bg = set()
            for bg in trial_bgs:
                if bg['id'] in seen_bg:
                    continue
                seen_bg.add(bg['id'])
                image = bg['_image_array']
                height, width = image.shape[:2]
                plane_areas = [(b[2]-b[0])*(b[3]-b[1])/(width*height) for b in bg['plane_boxes']]
                local = float(np.median(plane_areas))
                ratio = float(np.clip(math.sqrt(local*p50)*rng.uniform(.85, 1.15), p10, p90))
                scale = math.sqrt(ratio*width*height/source_area) * float(item.get('paste_scale_factor', 1.0))
                size = (round(premult.shape[1]*scale), round(premult.shape[0]*scale))
                if min(size) < 18 or max(size) > min(width, height)/3:
                    failures['bad_scale'] += 1
                    continue
                fg, mask = resize_premult(premult, alpha, size)
                placement = choose_placement(image, bg['boxes'], bg['plane_boxes'], size, rng,
                                             bg['_lab_array'], bg['_surface_reference'])
                if placement is None:
                    failures['no_placement'] += 1
                    continue
                x, y, placement_metrics = placement
                roi = image[y:y+size[1], x:x+size[0]]
                fg, soft, blend = harmonize(fg, mask, source_context, roi)
                composite = image.copy()
                composite[y:y+size[1], x:x+size[0]] = np.clip(
                    fg.astype(np.float32)*soft[:, :, None] + roi.astype(np.float32)*(1-soft[:, :, None]),
                    0, 255).astype(np.uint8)
                points = cv2.findNonZero((mask > .5).astype(np.uint8))
                bx, by, bw, bh = cv2.boundingRect(points)
                box = (x+bx, y+by, x+bx+bw, y+by+bh)
                if any(box_overlap(box, existing, 8) for existing in bg['boxes']):
                    failures['mask_collision'] += 1
                    continue
                success = (item, bg, image, composite, mask, box, (x, y), ratio,
                           angle, placement_metrics, blend)
                break
            if success:
                break
        if not success:
            raise RuntimeError(f'{args.dataset}: no feasible sample for {class_name}, {len(records)}/{args.count}')
        item, bg, before, after, mask, box, offset, ratio, angle, placement_metrics, blend = success
        used_fg.add(item['id'])
        used_base[item.get('base_foreground_id', item['id'])] += 1
        output_index = index + args.index_offset
        stem = f'{output_index:04d}__{args.dataset}__{class_name.replace("/", "-")}__{Path(bg["image"]).stem}'
        final_image = args.output / 'final/train/images' / f'{stem}.jpg'
        final_label = args.output / 'final/train/labels' / f'{stem}.txt'
        cmp_image = args.output / 'comparison/train/images' / f'{stem}.jpg'
        cmp_label = args.output / 'comparison/train/labels' / f'{stem}.txt'
        mask_path = args.output / 'instance_masks' / f'{stem}.png'
        original_path = args.output / 'original_annotations' / f'{stem}.json'
        atomic_image(final_image, after)
        atomic_image(cmp_image, draw_comparison(before, after, box, class_name))
        full_mask = np.zeros(before.shape[:2], dtype=np.uint8)
        full_mask[offset[1]:offset[1]+mask.shape[0], offset[0]:offset[0]+mask.shape[1]] = (mask > .5).astype(np.uint8)*255
        atomic_mask(mask_path, full_mask)
        line = yolo_hbb_line(class_id, box, before.shape[1], before.shape[0])
        atomic_bytes(final_label, (line+'\n').encode())
        right_box = (box[0]+before.shape[1], box[1], box[2]+before.shape[1], box[3])
        cmp_line = yolo_hbb_line(class_id, right_box, before.shape[1]*2, before.shape[0])
        atomic_bytes(cmp_label, (cmp_line+'\n').encode())
        atomic_bytes(original_path, (json.dumps({'dataset': args.dataset, 'background_id': bg['id'],
            'split': 'train', 'label_format': bg['label_format'], 'annotations': bg['annotations']},
            ensure_ascii=False)+'\n').encode())
        repeat_index = sum(1 for r in records if r['source_background_id'] == bg['id']) + 1
        row = {'background_id': f'{bg["id"]}__use{repeat_index:02d}', 'source_background_id': bg['id'],
               'dataset': args.dataset, 'split': 'train', 'image_path': str(bg['image']),
               'label_path': str(bg['label']), 'image_sha256': digest(bg['image']),
               'label_sha256': digest(bg['label'])}
        record = {'schema_version': 1, 'pipeline_version': PIPELINE_VERSION,
                  'id': stem, 'mode': 'lae1m_preview', 'subdataset': args.dataset,
                  'class_name': class_name, 'target_class_id': class_id,
                  'foreground_id': item['id'], 'foreground_base_id': item.get('base_foreground_id', item['id']),
                  'foreground_source': item['source'], 'foreground_path': item['cutout_path'],
                  'foreground_pool': item.get('foreground_pool'), 'foreground_sha256': digest(Path(item['cutout_path'])),
                  'foreground_source_path': item['source_path'], 'foreground_augmentation': item.get('augmentation'),
                  'background_id': row['background_id'], 'source_background_id': bg['id'],
                  'background_dataset': args.dataset, 'background_split': 'train',
                  'background_image_path': str(bg['image']), 'background_image_sha256': row['image_sha256'],
                  'background_label_path': str(bg['label']), 'background_label_sha256': row['label_sha256'],
                  'original_annotations_path': str(original_path), 'original_annotations_sha256': digest(original_path),
                  'image_path': str(final_image), 'image_sha256': digest(final_image),
                  'instance_mask_path': str(mask_path), 'instance_mask_sha256': digest(mask_path),
                  'label_path': str(final_label), 'label_sha256': digest(final_label),
                  'bbox_xyxy': list(box), 'rotation_deg': angle, 'paste_scale_factor': item.get('paste_scale_factor', 1.0),
                  'target_hbb_area_ratio': ratio,
                  'actual_hbb_area_ratio': (box[2]-box[0])*(box[3]-box[1])/(before.shape[0]*before.shape[1]),
                  'placement': placement_metrics, 'blend': blend,
                  'label_policy': 'preview_only_synthetic_MAR20_HBB_original_LAE_annotations_sidecar',
                  'train_ready': False, 'review_status': 'automatic_unreviewed'}
        records.append(record)
    args.output.mkdir(parents=True, exist_ok=True)
    names_text = 'names:\n' + ''.join(f'  {i}: {name}\n' for i, name in enumerate(MAR20_NAMES))
    for name in ('final', 'comparison'):
        atomic_bytes(args.output / name / 'data.yaml', names_text.encode())
    atomic_bytes(args.output / 'manifest.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records).encode())
    atomic_bytes(args.output / 'background_train.jsonl', ''.join(json.dumps({
        'background_id': r['background_id'], 'source_background_id': r['source_background_id'],
        'dataset': r['background_dataset'], 'split': r['background_split'],
        'image_path': r['background_image_path'], 'label_path': r['background_label_path'],
        'image_sha256': r['background_image_sha256'], 'label_sha256': r['background_label_sha256']},
        ensure_ascii=False)+'\n' for r in records).encode())
    config = {'pipeline_version': PIPELINE_VERSION, 'dataset': args.dataset,
              'count': args.count, 'seed': args.seed, 'index_offset': args.index_offset,
              'foreground_source': args.foreground_source,
              'foreground_manifest': str(args.foreground_manifest),
              'background_label_source': label_path, 'background_split': 'train',
              'generation_classes': classes, 'original_annotations_sidecar': True,
              'train_ready': False, 'background_sampling_with_replacement': True,
              'target_area_ratio_p10_p50_p90': [p10, p50, p90]}
    atomic_bytes(args.output / 'generation_config.json',
                 (json.dumps(config, ensure_ascii=False, indent=2)+'\n').encode())
    summary = {'pipeline_version': PIPELINE_VERSION, 'mode': 'lae1m_preview',
               'subdataset': args.dataset, 'requested': args.count, 'generated': len(records),
               'index_offset': args.index_offset,
               'train_ready': False, 'label_policy': 'synthetic MAR20 HBB only; original annotations saved separately',
               'generation_classes': classes, 'classes': dict(Counter(r['class_name'] for r in records)),
               'foreground_sources': dict(Counter(r['foreground_source'] for r in records)),
               'unique_base_foregrounds': len(used_base), 'foreground_variants_used': len(used_fg),
               'available_unique_train_aircraft_backgrounds': len(backgrounds),
               'sampled_unique_train_background_pool': len(background_pool),
               'unique_train_backgrounds_used': len({r['source_background_id'] for r in records}),
               'dataset_aircraft_area_ratio_p10_p50_p90': [p10, p50, p90],
               'failures_during_search': dict(failures), 'seed': args.seed,
               'source_foreground_manifest': str(args.foreground_manifest),
               'background_label_source': label_path}
    atomic_bytes(args.output / 'qa/summary.json', (json.dumps(summary, ensure_ascii=False, indent=2)+'\n').encode())
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
