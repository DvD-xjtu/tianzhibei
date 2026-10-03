#!/usr/bin/env python3
"""Conservative, prompted SAM2 extraction of a small MTARSI foreground batch.

The pipeline is entirely automatic.  It records why a mask was selected, but
does not use a human acceptance filter when writing its output batch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

FIXED = Path('/data3/tianzhibei/datasets/MTARSI-fixed')
INNAR_ROOT = Path('/data3/tianzhibei/datasets/MTARSI-INNAR/extracted/MTARSI-INNAR')
CLASS_MAP = {
    # 11 classes directly mappable to the user's 15-class aircraft taxonomy.
    'A-10': 'A-10', 'B-1': 'B-1B', 'B-52': 'B-52', 'C-130': 'C-130',
    'C-17': 'C-17', 'E-2': 'E-2',
    'A-10_Thunderbolt': 'A-10', 'B-1_Lancer': 'B-1B',
    'B-52_Stratofortress': 'B-52', 'C-130_Hercules': 'C-130',
    'C-17_Globemaster': 'C-17', 'E-2_Hawkeye': 'E-2', 'E-3_Sentry': 'E-3',
    'F-15_Eagle': 'F-15', 'F-16_Falcon': 'F-16', 'F-18_Hornit': 'F/A-18',
    'F-35_JSF': 'F-35',
    # C-5 remains enabled for the existing MAR20 first-round experiment, but
    # is outside the user's original 15-class list.
    'C-5': 'C-5', 'C-5_Galaxy': 'C-5',
}


def candidates() -> dict[str, list[tuple[str, Path]]]:
    out: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    sources = [
        ('fixed', FIXED),
        ('innar_train', INNAR_ROOT / 'Train'),
        ('innar_test', INNAR_ROOT / 'Test'),
        ('innar_valid', INNAR_ROOT / 'valid'),
    ]
    for source, root in sources:
        for folder, target in CLASS_MAP.items():
            d = root / folder
            if d.is_dir():
                out[target].extend((source, p) for p in d.glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    return out


def choose_batch(n: int, seed: int, only_classes: set[str] | None = None) -> list[tuple[str, str, Path]]:
    rng = random.Random(seed)
    groups = candidates()
    if only_classes is not None:
        groups = {name: paths for name, paths in groups.items() if name in only_classes}
    classes = sorted(groups)
    chosen: list[tuple[str, str, Path]] = []
    # Round-robin prevents C-130/F-16 from monopolising a 20-image smoke batch.
    while len(chosen) < n and any(groups.values()):
        progressed = False
        for cls in classes:
            if groups[cls] and len(chosen) < n:
                i = rng.randrange(len(groups[cls]))
                source, path = groups[cls].pop(i)
                chosen.append((cls, source, path)); progressed = True
        if not progressed: break
    return chosen


def geometry(mask: np.ndarray) -> tuple[float, float, float]:
    """Return area, border contact and centre distance for a binary mask."""
    area = float(mask.mean())
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return area, 1.0, 1.0
    h, w = mask.shape
    cx, cy = xs.mean() / w, ys.mean() / h
    dist = ((cx - .5) ** 2 + (cy - .5) ** 2) ** .5
    border = (mask[0].sum() + mask[-1].sum() + mask[:, 0].sum() + mask[:, -1].sum()) / max(mask.sum(), 1)
    return area, float(border), float(dist)


def choose_conservative_mask(masks: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float, dict]:
    """Select the highest-confidence geometrically plausible full silhouette.

    With SAM2.1-large, the highest-confidence loose-box candidate preserved
    aircraft extremities better than a forced union/larger-mask heuristic.
    The 3x3 closing only repairs isolated pixel gaps.
    """
    candidates = []
    for index, (mask, confidence) in enumerate(zip(masks, scores)):
        binary = mask.astype(bool)
        area, border, distance = geometry(binary)
        if .01 <= area <= .50 and border <= .30:
            candidates.append((index, binary, float(confidence), area, border, distance))
    if candidates:
        chosen = max(candidates, key=lambda item: item[2])
    else:
        index = int(np.argmax(scores)); binary = masks[index].astype(bool)
        area, border, distance = geometry(binary)
        chosen = (index, binary, float(scores[index]), area, border, distance)
    index, binary, confidence, area, border, distance = chosen
    closed = np.asarray(
        Image.fromarray((binary * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    ) > 0
    return closed, confidence, {
        'candidate_index': index, 'candidate_area_ratio': area,
        'candidate_border_ratio': border, 'candidate_centre_distance': distance,
        'selection': 'highest_sam_score_among_geometrically_valid_candidates',
        'postprocess': '3x3_binary_closing',
    }


def sheet(rows: list[dict], path: Path) -> None:
    cw, ch = 540, 190
    out = Image.new('RGB', (cw * 2, ch * ((len(rows) + 1) // 2)), 'white')
    d = ImageDraw.Draw(out)
    for i, r in enumerate(rows):
        im = r['panel']; im.thumbnail((cw - 8, ch - 28))
        x, y = (i % 2) * cw, (i // 2) * ch
        out.paste(im, (x + (cw - im.width) // 2, y + 2))
        d.text((x + 4, y + ch - 21), f"{r['id']}  {r['class']}  score={r['selected_score']:.3f}", fill='black')
    out.save(path, quality=92)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, type=Path)
    ap.add_argument('--config', default='configs/sam2.1/sam2.1_hiera_t.yaml')
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--seed', type=int, default=20260923)
    ap.add_argument('--only-classes', nargs='+', default=None,
                    help='Optional canonical output classes for a targeted QA batch, e.g. F-35')
    ap.add_argument('--shard-index', type=int, default=0,
                    help='Zero-based shard number; use with --num-shards for parallel full runs.')
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--id-prefix', default='',
                    help='Prefix output ids when separate shards will later be merged.')
    ap.add_argument('--box-margin', type=float, default=.05,
                    help='Loose box preserves wings near the crop boundary.')
    args = ap.parse_args()
    output = args.output
    for p in ('masks', 'cutouts', 'overlays', 'viewer/images', 'viewer/labels'):
        (output / p).mkdir(parents=True, exist_ok=True)

    predictor = SAM2ImagePredictor(build_sam2(args.config, str(args.checkpoint), device='cuda'))
    records, rows = [], []
    requested = set(args.only_classes) if args.only_classes else None
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError('--shard-index must be in [0, --num-shards)')
    all_items = choose_batch(args.n, args.seed, requested)
    shard_items = [item for index, item in enumerate(all_items) if index % args.num_shards == args.shard_index]
    for i, (cls, source, img_path) in enumerate(shard_items, 1):
        image = np.array(Image.open(img_path).convert('RGB'), copy=True)
        h, w = image.shape[:2]; m = args.box_margin
        box = np.array([m*w, m*h, (1-m)*w, (1-m)*h], dtype=np.float32)
        predictor.set_image(image)
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        torch.cuda.synchronize(); elapsed = (time.perf_counter()-start)*1000
        chosen, selected_score, selection_detail = choose_conservative_mask(masks, scores); ys, xs = np.where(chosen)
        prefix = f'{args.id_prefix}__' if args.id_prefix else ''
        stem = f'{prefix}{i:04d}__{cls.replace("/", "-")}__{img_path.stem}'
        Image.fromarray((chosen * 255).astype(np.uint8)).save(output/'masks'/f'{stem}.png')
        rgba = np.dstack((image, (chosen * 255).astype(np.uint8)))
        Image.fromarray(rgba, 'RGBA').save(output/'cutouts'/f'{stem}.png')
        red = image.copy(); red[chosen] = (.45*red[chosen] + .55*np.array([255, 30, 30])).astype(np.uint8)
        Image.fromarray(red).save(output/'overlays'/f'{stem}.jpg', quality=95)
        # Neutral JPEG is only for the existing viewer; the transparent PNG above is canonical.
        canvas = Image.new('RGB', (512,512), (232,232,232)); fg=Image.fromarray(image).convert('RGBA'); fg.putalpha(Image.fromarray((chosen*255).astype(np.uint8)))
        fg.thumbnail((460,460)); x,y=(512-fg.width)//2,(512-fg.height)//2; canvas.paste(fg,(x,y),fg)
        canvas.save(output/'viewer/images'/f'{stem}.jpg',quality=95)
        (output/'viewer/labels'/f'{stem}.txt').write_text(f'0 0.5 0.5 {fg.width/512:.6f} {fg.height/512:.6f}\n')
        bbox = None if len(xs)==0 else [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
        rec={'id':stem,'class':cls,'source':source,'source_path':str(img_path),'mask_path':str(output/'masks'/f'{stem}.png'),'cutout_path':str(output/'cutouts'/f'{stem}.png'),'prompt_box_xyxy':box.tolist(),'pipeline_status':'automatic_unreviewed','bbox_xyxy':bbox,'mask_area_ratio':float(chosen.mean()),'sam_scores':[float(x) for x in scores],'selected_score':selected_score,'selection_detail':selection_detail,'inference_ms':elapsed}
        records.append(rec); rows.append({**rec,'panel':Image.fromarray(red)})
    (output/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    (output/'viewer/data.yaml').write_text('names:\n  0: candidate_foreground\n')
    sheet(rows, output/'contact_sheet_overlay.jpg')
    print(json.dumps({'count':len(records),'output':str(output)},ensure_ascii=False))
if __name__=='__main__': main()
