#!/usr/bin/env python3
"""Publish source/cutout review pairs to the existing extra viewer."""
from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path('/data3/tianzhibei/derived/aircraft_copypaste_final_20260928/extra/SR71_NAIP_sources_20261003')
CONFIG = Path('/data2/tianzhibei/dataset-viewer/datasets.json')


def main():
    rows = json.loads((ROOT/'extraction_v1/manifest.json').read_text())
    n=len(rows)
    passed=sum(r['quality']['accepted'] for r in rows)
    review = ROOT/'dashboard'/'review'
    raw = ROOT/'dashboard'/'raw'
    cutouts = ROOT/'dashboard'/'cutouts'
    for d in (review,raw,cutouts): d.mkdir(parents=True,exist_ok=True)
    valid={r['id'] for r in rows}
    for directory,extension in ((review,'.jpg'),(raw,'.png'),(cutouts,'.jpg')):
        for old in directory.glob('*'+extension):
            if old.stem not in valid:
                old.unlink()
    for r in rows:
        stem = r['id']
        link = raw/f'{stem}.png'
        if not link.exists(): link.symlink_to(r['image_path'])
        context = Image.open(r['source_crop_path']).convert('RGB')
        crop = Image.open(r['cutout_path']).convert('RGBA')
        tile = Image.new('RGB',(800,435),(218,218,218))
        left = ImageOps.contain(context,(380,380))
        right = ImageOps.contain(crop,(380,380))
        tile.paste(left,(10+(380-left.width)//2,25+(380-left.height)//2))
        tile.paste(right,(410+(380-right.width)//2,25+(380-right.height)//2),right)
        d = ImageDraw.Draw(tile)
        tag = 'AUTO PASS · REVIEW REQUIRED' if r['quality']['accepted'] else 'AUTO FAIL · REVIEW REQUIRED'
        d.text((10,5),f"{r['site']} / {r['serial']} / {r['acquisition_datetime'][:10]}",fill=(0,0,0))
        d.text((10,410),'NAIP aerial context',fill=(0,0,0))
        d.text((410,410),f"SAM2 cutout  |  {tag}",fill=(120,0,0))
        tile.save(review/f'{stem}.jpg',quality=94)
        only=Image.new('RGB',(400,400),(215,215,215))
        on=ImageOps.contain(crop,(370,370))
        only.paste(on,((400-on.width)//2,(400-on.height)//2),on)
        only.save(cutouts/f'{stem}.jpg',quality=94)
    cfg=json.loads(CONFIG.read_text())
    ds=cfg['datasets']
    ids={'extra-sr71-naip-sources-20261003','extra-sr71-naip-crop-review-20261003','extra-sr71-naip-cutouts-20261003'}
    ds[:]=[d for d in ds if d['id'] not in ids]
    entries=[
        dict(id='extra-sr71-naip-sources-20261003',name=f'SR-71 · NAIP遥感原图 · {n}个独立画面',
             group='extra',root=str(raw),format='images',training=False,
             note=f'7架室外展机、{n}个像素不重复的遥感画面；NAIP正射航空遥感。March室内不可见、Eglin 2019影像空白已排除；Eglin另有2对不同日期但像素相同的瓦片已去重。Palmdale附近另有A-12，不要混标。'),
        dict(id='extra-sr71-naip-crop-review-20261003',name=f'SR-71 · crop case审核 · {n}组原图/抠图',
             group='extra',root=str(review),format='images',training=False,
             note=f'左：无框原始遥感局部；右：SAM2透明前景叠在灰底。共{n}组，自动质检{passed}通过、{n-passed}失败；部分自动通过仍粘连水泥底座或阴影，均未人工批准、不可直接训练。'),
        dict(id='extra-sr71-naip-cutouts-20261003',name=f'SR-71 · 透明crop单独预览 · {n}个',
             group='extra',root=str(cutouts),format='images',training=False,
             note=f'灰底仅用于展板预览；真正RGBA图在extraction_v1/accepted或rejected/cutouts。{n}个含{n-passed}个自动不合格，均待人工确认。'),
    ]
    ds.extend(entries)
    tmp=CONFIG.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
    os.replace(tmp,CONFIG)
    print(json.dumps(dict(published=len(rows),raw=str(raw),review=str(review),cutouts=str(cutouts)),ensure_ascii=False))


if __name__=='__main__': main()
