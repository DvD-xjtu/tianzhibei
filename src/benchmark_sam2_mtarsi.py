#!/usr/bin/env python3
"""Benchmark SAM2 automatic masks on a deterministic MTARSI sample.

This is an evaluation utility, not the production mask-selection pipeline.  It
uses automatic mask generation, then records one centre-biased candidate per
image so that it can be visually audited in a contact sheet.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2


FIXED = Path('/data3/tianzhibei/datasets/MTARSI-fixed')
INNAR = Path('/data3/tianzhibei/datasets/MTARSI-INNAR/extracted/MTARSI-INNAR')


def image_paths(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})


def select_mask(masks: list[dict], width: int, height: int) -> dict | None:
    """Pick a likely centred aircraft candidate for manual audit.

    This is deliberately conservative: it rejects masks too small/large and
    rewards masks whose centroid is close to crop centre.  The manual pass
    label remains the authoritative quality decision.
    """
    best, best_score = None, float('-inf')
    image_area = width * height
    for item in masks:
        area = float(item.get('area', 0)) / image_area
        if not 0.01 <= area <= 0.70:
            continue
        x, y, w, h = item.get('bbox', (0, 0, 0, 0))
        dx = (x + w / 2 - width / 2) / width
        dy = (y + h / 2 - height / 2) / height
        centrality = max(0.0, 1.0 - 2.0 * (dx * dx + dy * dy) ** 0.5)
        score = (float(item.get('predicted_iou', 0)) + float(item.get('stability_score', 0))) * centrality
        if score > best_score:
            best, best_score = item, score
    return best


def overlay(image: np.ndarray, mask: np.ndarray | None) -> Image.Image:
    out = Image.fromarray(image).convert('RGBA')
    if mask is not None:
        tint = np.zeros((*mask.shape, 4), dtype=np.uint8)
        tint[mask] = (255, 0, 0, 96)
        out = Image.alpha_composite(out, Image.fromarray(tint, 'RGBA'))
    return out.convert('RGB')


def make_sheet(rows: list[dict], output: Path) -> None:
    cell_w, cell_h = 256, 230
    sheet = Image.new('RGB', (cell_w * 5, cell_h * ((len(rows) + 4) // 5)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        im = row['preview'].copy()
        im.thumbnail((cell_w - 8, cell_h - 40))
        x, y = (i % 5) * cell_w, (i // 5) * cell_h
        sheet.paste(im, (x + (cell_w - im.width) // 2, y + 2))
        draw.text((x + 4, y + cell_h - 35), row['source'], fill='black')
        draw.text((x + 4, y + cell_h - 20), f"{row['ms']:.0f}ms area={row['area_ratio']:.2%}", fill='black')
    sheet.save(output, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', default='configs/sam2.1/sam2.1_hiera_t.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-source', type=int, default=25)
    parser.add_argument('--seed', type=int, default=20260923)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    items = []
    for name, root in [('fixed', FIXED), ('innar', INNAR)]:
        choices = image_paths(root)
        items += [(name, path) for path in rng.sample(choices, min(args.per_source, len(choices)))]

    model = build_sam2(args.config, str(args.checkpoint), device='cuda')
    generator = SAM2AutomaticMaskGenerator(
        model, points_per_side=32, pred_iou_thresh=0.80,
        stability_score_thresh=0.90, crop_n_layers=0,
    )
    records, preview_rows = [], []
    for idx, (source, path) in enumerate(items):
        image = np.asarray(Image.open(path).convert('RGB'))
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        masks = generator.generate(image)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        chosen = select_mask(masks, image.shape[1], image.shape[0])
        segmentation = chosen['segmentation'] if chosen is not None else None
        ratio = float(chosen['area']) / (image.shape[0] * image.shape[1]) if chosen is not None else 0.0
        record = {
            'id': idx, 'source': source, 'path': str(path), 'width': int(image.shape[1]),
            'height': int(image.shape[0]), 'candidate_masks': len(masks),
            'selected': chosen is not None, 'selected_area_ratio': ratio,
            'inference_ms': elapsed_ms,
            'peak_memory_mib': torch.cuda.max_memory_allocated() / 1024 / 1024,
        }
        records.append(record)
        preview_rows.append({**record, 'ms': elapsed_ms, 'area_ratio': ratio, 'preview': overlay(image, segmentation)})
        if segmentation is not None:
            Image.fromarray((segmentation * 255).astype(np.uint8)).save(args.output / f'{idx:03d}.png')
    (args.output / 'records.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
    make_sheet(preview_rows, args.output / 'contact_sheet.jpg')
    print(json.dumps({
        'images': len(records),
        'mean_ms': sum(x['inference_ms'] for x in records) / len(records),
        'peak_mib_max': max(x['peak_memory_mib'] for x in records),
        'selected': sum(x['selected'] for x in records),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
