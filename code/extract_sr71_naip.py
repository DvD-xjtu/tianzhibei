#!/usr/bin/env python3
"""Segment verified SR-71 museum airframes from dated NAIP aerial imagery."""
from __future__ import annotations

import json
import hashlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from extract_mtarsi_foregrounds import choose_conservative_mask
from filter_mar20_foregrounds import quality


ROOT = Path('/data3/tianzhibei/derived/aircraft_copypaste_final_20260928/extra/SR71_NAIP_sources_20261003')
OUT = ROOT / 'extraction_v1'
# Prompt boxes in the fixed 1200x960 NAIP context. The site and target airframe
# have been checked separately; Palmdale's neighboring A-12 is outside its box.
BOXES = {
    'barksdale': (568, 439, 646, 566),
    'palmdale': (551, 429, 651, 544),
    'lackland': (515, 453, 626, 591),
    'eglin': (582, 454, 700, 610),
    'edwards': (535, 384, 672, 535),
    'castle': (528, 464, 689, 558),
    'beale': (486, 429, 673, 535),
}
GATES = dict(min_score=.50, min_dominant=.90, max_secondary=.04,
             max_mask_area=.60, min_bbox_fill=.10)


def jobs():
    rows = json.loads((ROOT / 'historical_manifest.json').read_text())
    seen = set()
    seen_pixels = {}
    duplicate_rows = []
    for r in rows:
        site = r['site']
        date = r['acquisition_datetime'][:10]
        if site not in BOXES or (site == 'eglin' and date.startswith('2019')):
            continue
        key = (site, date)
        if key in seen:
            continue
        seen.add(key)
        box = BOXES[site]
        image = Image.open(r['image_path']).convert('RGB')
        sample = image.crop((box[0]-35,box[1]-35,box[2]+35,box[3]+35))
        digest = hashlib.sha256(sample.tobytes()).hexdigest()
        if digest in seen_pixels:
            duplicate_rows.append(dict(id=r['id'],same_pixels_as=seen_pixels[digest]))
            continue
        seen_pixels[digest] = r['id']
        yield r
    (OUT/'exact_duplicate_sources.json').write_text(json.dumps(duplicate_rows,ensure_ascii=False,indent=2)+'\n')


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    model = build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',
                       '/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/checkpoints/sam2.1_hiera_large.pt',
                       device='cuda')
    predictor = SAM2ImagePredictor(model)
    results = []
    for n, r in enumerate(jobs(), 1):
        site = r['site']
        xyxy = BOXES[site]
        image = Image.open(r['image_path']).convert('RGB')
        if image.size != (1200, 960):
            raise ValueError((r['id'], image.size))
        pad = 35
        x0 = max(0, xyxy[0]-pad)
        y0 = max(0, xyxy[1]-pad)
        x1 = min(1200, xyxy[2]+pad)
        y1 = min(960, xyxy[3]+pad)
        context = np.asarray(image.crop((x0,y0,x1,y1)))
        box = np.array([xyxy[0]-x0,xyxy[1]-y0,xyxy[2]-x0,xyxy[3]-y0], dtype=np.float32)
        with torch.inference_mode():
            predictor.set_image(context)
            masks,scores,_ = predictor.predict(box=box, multimask_output=True)
        mask,score,sel = choose_conservative_mask(masks,scores)
        q = quality(mask, score, GATES)
        # Pixels on the crop border indicate that the plane was truncated.
        if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():
            q['reasons'].append('touches_context_border')
        q['accepted'] = not q['reasons']
        sub = OUT / ('accepted' if q['accepted'] else 'rejected')
        for child in ('cutouts','contexts','masks'):
            (sub/child).mkdir(parents=True,exist_ok=True)
        stem=r['id']
        cp=sub/'cutouts'/f'{stem}.png'
        sp=sub/'contexts'/f'{stem}.png'
        mp=sub/'masks'/f'{stem}.png'
        Image.fromarray(context).save(sp)
        Image.fromarray(np.dstack([context,mask.astype('uint8')*255])).save(cp)
        Image.fromarray(mask.astype('uint8')*255).save(mp)
        results.append(dict(r,cutout_path=str(cp),source_crop_path=str(sp),
                            mask_path=str(mp),prompt_box_xyxy=list(xyxy),
                            context_rect=[x0,y0,x1,y1],selected_score=float(score),
                            selection=sel,quality=q,stage='segmented_review',
                            training_eligible=False))
        print(n,site,r['acquisition_datetime'][:10],q['accepted'],round(float(score),3),q['reasons'],flush=True)
    (OUT/'manifest.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    allowed={r['id']+'.png' for r in results}
    for folder in ('accepted','rejected'):
        for child in ('cutouts','contexts','masks'):
            for p in (OUT/folder/child).glob('*.png'):
                if p.name not in allowed:
                    p.unlink()
    accepted=[r for r in results if r['quality']['accepted']]
    summary=dict(candidate_count=len(results),accepted=len(accepted),
                 by_site=dict(Counter(r['site'] for r in accepted)),
                 rejected_reasons=dict(Counter(e for r in results for e in r['quality']['reasons'])),
                 distinct_airframes=len(set(r['serial'] for r in accepted)),
                 quality_gates=GATES,review_status='automatic_only_not_human_approved',
                 source_domain='NAIP orthorectified aerial imagery')
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    sheet_w,sheet_h=500,260
    sheet=Image.new('RGB',(sheet_w*4,sheet_h*((len(results)+3)//4)),(220,220,220))
    draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(results):
        cut=Image.open(r['cutout_path']).convert('RGBA')
        cut.thumbnail((240,230))
        src=Image.open(r['source_crop_path']).convert('RGB')
        src.thumbnail((240,230))
        x=(i%4)*sheet_w;y=(i//4)*sheet_h
        sheet.paste(src,(x+(240-src.width)//2,y+(230-src.height)//2))
        sheet.paste(cut,(x+250+(240-cut.width)//2,y+(230-cut.height)//2),cut)
        draw.text((x+5,y+233),f"{r['site']} {r['acquisition_datetime'][:10]} {'PASS' if r['quality']['accepted'] else 'FAIL'}",fill='black')
    sheet.save(OUT/'review_contact_sheet.jpg',quality=92)
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
