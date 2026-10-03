#!/usr/bin/env python3
"""Publish SR-71 v2 cutouts with source and v1 side-by-side comparison."""
from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path('/data3/tianzhibei/derived/aircraft_copypaste_final_20260928/extra/SR71_NAIP_sources_20261003')
CONFIG = Path('/data2/tianzhibei/dataset-viewer/datasets.json')


def paste_contained(canvas, image, box, alpha=False):
    x,y,w,h = box
    image = ImageOps.contain(image,(w,h))
    at=(x+(w-image.width)//2,y+(h-image.height)//2)
    canvas.paste(image,at,image if alpha else None)


def main():
    old = {r['id']:r for r in json.loads((ROOT/'extraction_v1/manifest.json').read_text())}
    rows = json.loads((ROOT/'extraction_v2/manifest.json').read_text())
    review = ROOT/'dashboard_v2'/'v1_v2_review'
    cutouts = ROOT/'dashboard_v2'/'cutouts'
    for d in (review,cutouts): d.mkdir(parents=True,exist_ok=True)
    valid={r['id'] for r in rows}
    for directory in (review,cutouts):
        for p in directory.glob('*.jpg'):
            if p.stem not in valid: p.unlink()
    for r in rows:
        stem=r['id']
        context=Image.open(r['source_crop_path']).convert('RGB')
        v1=Image.open(old[stem]['cutout_path']).convert('RGBA')
        v2=Image.open(r['cutout_path']).convert('RGBA')
        canvas=Image.new('RGB',(1200,460),(215,215,215))
        paste_contained(canvas,context,(10,30,380,390))
        paste_contained(canvas,v1,(410,30,380,390),True)
        paste_contained(canvas,v2,(810,30,380,390),True)
        d=ImageDraw.Draw(canvas)
        d.text((10,8),f"SR-71A {r['serial']} | {r['site']} | {r['acquisition_datetime'][:10]}",fill='black')
        d.text((10,430),'NAIP source',fill='black')
        d.text((410,430),'v1 SAM2',fill='black')
        d.text((810,430),f"v2 {r['refinement']['method']} | REVIEW",fill=(100,0,0))
        canvas.save(review/f'{stem}.jpg',quality=94)
        only=Image.new('RGB',(420,420),(215,215,215))
        paste_contained(only,v2,(10,10,400,400),True)
        only.save(cutouts/f'{stem}.jpg',quality=94)
    cfg=json.loads(CONFIG.read_text())
    ids={'extra-sr71-naip-v2-review-20261003','extra-sr71-naip-v2-cutouts-20261003'}
    cfg['datasets']=[d for d in cfg['datasets'] if d['id'] not in ids]
    n=len(rows)
    cfg['datasets'].extend([
        dict(id='extra-sr71-naip-v2-review-20261003',name=f'SR-71 · v2精修对照 · {n}组',
             group='extra',root=str(review),format='images',training=False,
             note='左：NAIP原图局部；中：v1 SAM2；右：v2精修透明crop。Castle去除红褐底座，其他地点按明度去底座/去碎片。所有样本待人工确认，未用于训练。'),
        dict(id='extra-sr71-naip-v2-cutouts-20261003',name=f'SR-71 · v2透明crop预览 · {n}个',
             group='extra',root=str(cutouts),format='images',training=False,
             note='灰底仅为展板预览；真正RGBA在extraction_v2/cutouts。仍需人工审核边缘和机身完整性，不能直接作为YF-12A训练图。'),
    ])
    temp=CONFIG.with_suffix('.json.tmp')
    temp.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    os.replace(temp,CONFIG)
    print(json.dumps(dict(published=n,review=str(review),cutouts=str(cutouts)),ensure_ascii=False))


if __name__=='__main__':main()
