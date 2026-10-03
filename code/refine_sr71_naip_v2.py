#!/usr/bin/env python3
"""Refine SR-71 NAIP masks without overwriting the SAM2 v1 extraction.

The site-specific method is deliberately conservative: Castle's blue-gray
airframe is separated from its red pad by RGB contrast, while sites with
neutral-colored supports use a luminance split inside the v1 SAM2 proposal.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from filter_mar20_foregrounds import quality

ROOT = Path('/data3/tianzhibei/derived/aircraft_copypaste_final_20260928/extra/SR71_NAIP_sources_20261003')
V1 = ROOT / 'extraction_v1'
OUT = ROOT / 'extraction_v2'
LUMINANCE_SITES = {'barksdale', 'edwards', 'eglin', 'lackland'}
GATES = dict(min_score=.50, min_dominant=.90, max_secondary=.04,
             max_mask_area=.60, min_bbox_fill=.10)


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == index


def refine(image: np.ndarray, old: np.ndarray, site: str) -> tuple[np.ndarray, dict]:
    if site == 'castle':
        # Blackbird paint has blue-channel excess; the plinth is red/brown.
        feature = image[:, :, 2].astype(np.int16) - image[:, :, 0].astype(np.int16)
        proposal = old & (feature >= 2)
        method = 'blue_minus_red_inside_sam2'
        threshold = 2
    elif site in LUMINANCE_SITES:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        threshold, _ = cv2.threshold(gray[old], 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = int(threshold) + 3
        proposal = old & (gray <= threshold)
        method = 'otsu_luminance_inside_sam2'
    else:
        proposal = old.copy()
        method = 'sam2_largest_component_only'
        threshold = None
    proposal = cv2.morphologyEx(proposal.astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((3, 3), dtype=np.uint8)) > 0
    new = largest_component(proposal)
    return new, dict(method=method, threshold=threshold,
                     v1_pixels=int(old.sum()), v2_pixels=int(new.sum()),
                     retained_fraction=round(float(new.sum() / max(old.sum(), 1)), 4))


def main():
    rows = json.loads((V1 / 'manifest.json').read_text())
    v1_by_id = {r['id']: r for r in rows}
    output = []
    for r in rows:
        image = np.asarray(Image.open(r['source_crop_path']).convert('RGB'))
        old = np.asarray(Image.open(r['mask_path']).convert('L')) > 0
        mask, refinement = refine(image, old, r['site'])
        result = quality(mask, r['selected_score'], GATES)
        if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            result['reasons'].append('touches_context_border')
        if refinement['retained_fraction'] < .22:
            result['reasons'].append('substantial_mask_loss')
        result['accepted'] = not result['reasons']
        for child in ('contexts', 'masks', 'cutouts'):
            (OUT / child).mkdir(parents=True, exist_ok=True)
        stem = r['id']
        context_path = OUT / 'contexts' / f'{stem}.png'
        mask_path = OUT / 'masks' / f'{stem}.png'
        cutout_path = OUT / 'cutouts' / f'{stem}.png'
        Image.fromarray(image).save(context_path)
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        Image.fromarray(np.dstack([image, mask.astype(np.uint8) * 255])).save(cutout_path)
        output.append(dict(r, source_crop_path=str(context_path), mask_path=str(mask_path),
                           cutout_path=str(cutout_path), refinement=refinement,
                           v1_quality=r['quality'], quality=result,
                           stage='refined_review', training_eligible=False))
    (OUT / 'manifest.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    summary = dict(total=len(output), auto_pass=sum(r['quality']['accepted'] for r in output),
                   auto_fail=sum(not r['quality']['accepted'] for r in output),
                   by_site={s: dict(total=sum(r['site']==s for r in output),
                                    auto_pass=sum(r['site']==s and r['quality']['accepted'] for r in output))
                            for s in sorted(set(r['site'] for r in output))},
                   methods=dict(Counter(r['refinement']['method'] for r in output)),
                   rejected_reasons=dict(Counter(e for r in output for e in r['quality']['reasons'])),
                   review_status='automatic_only_not_human_approved',
                   training_ready=False)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    cell_w, cell_h = 420, 245
    sheet = Image.new('RGB', (cell_w*4, cell_h*((len(output)+3)//4)), (216,216,216))
    draw = ImageDraw.Draw(sheet)
    for i,r in enumerate(output):
        old = Image.open(v1_by_id[r['id']]['cutout_path']).convert('RGBA')
        new = Image.open(r['cutout_path']).convert('RGBA')
        x, y = (i%4)*cell_w, (i//4)*cell_h
        for j,im in enumerate((old,new)):
            im.thumbnail((198,210))
            xx = x + j*210 + (210-im.width)//2
            yy = y + (210-im.height)//2
            sheet.paste(im, (xx,yy), im)
        draw.text((x+5,y+214),f"{r['site']} {r['acquisition_datetime'][:10]}  {r['refinement']['retained_fraction']:.2f}",fill='black')
    sheet.save(OUT/'v1_v2_contact_sheet.jpg',quality=92)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
