#!/usr/bin/env python3
"""Render a small, non-training preview of the current SAM2-mask Copy-Paste path."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

RECORDS = Path('/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/qa/sam2_tiny_50/records.json')
MASKS = RECORDS.parent
DIOR_IMAGES = Path('/data2/tianzhibei/LAE-1M/LAE-FOD/DIOR/JPEGImages-trainval')
OUT = Path('/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/qa/viewer_preview_current')
NAMES = ['B-1B', 'B-52', 'C-130', 'C-17', 'C-5', 'E-3', 'F-15', 'F-16', 'F/A-18']
MAP = {'B-1':'B-1B','B-52':'B-52','C-130':'C-130','C-17':'C-17','C-5':'C-5',
       'B-1_Lancer':'B-1B','B-52_Stratofortress':'B-52','C-130_Hercules':'C-130',
       'C-17_Globemaster':'C-17','C-5_Galaxy':'C-5','E-3_Sentry':'E-3',
       'F-15_Eagle':'F-15','F-16_Falcon':'F-16','F-18_Hornit':'F/A-18'}

def label(path: Path, cls: int, box: tuple[int,int,int,int], size: tuple[int,int]) -> None:
    x1,y1,x2,y2=box; w,h=size
    path.write_text(f'{cls} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n')

def main() -> None:
    rng=random.Random(20260923)
    recs=json.loads(RECORDS.read_text())
    selected=[]
    for r in recs:
        raw=Path(r['path']).parent.name
        if r['selected'] and raw in MAP and r['selected_area_ratio'] >= .015:
            selected.append((r, MAP[raw]))
    # Varied but deterministic, six samples max; no claim of production quality.
    picks=[]
    seen=set()
    for r,c in selected:
        if c not in seen:
            picks.append((r,c)); seen.add(c)
        if len(picks)==6: break
    dirs=[]
    for kind in ('cutouts','composites'):
        base=OUT/kind; (base/'train/images').mkdir(parents=True,exist_ok=True); (base/'train/labels').mkdir(parents=True,exist_ok=True)
        (base/'data.yaml').write_text('names:\n' + ''.join(f'  {i}: {n}\n' for i,n in enumerate(NAMES)))
    bgs=sorted(DIOR_IMAGES.glob('*.jpg'))
    for i,(r,name) in enumerate(picks,1):
        im=Image.open(r['path']).convert('RGB'); mask=Image.open(MASKS/f"{r['id']:03d}.png").convert('L')
        box=mask.getbbox()
        if box is None: continue
        fg=im.crop(box); alpha=mask.crop(box)
        # Cutout preview: neutral display background makes residual source context visible.
        canvas=Image.new('RGB',(512,512),(232,232,232)); scale=min(420/fg.width,420/fg.height)
        sz=(round(fg.width*scale),round(fg.height*scale)); fg=fg.resize(sz,Image.Resampling.LANCZOS); alpha=alpha.resize(sz,Image.Resampling.LANCZOS)
        px,py=(512-sz[0])//2,(512-sz[1])//2; canvas.paste(fg,(px,py),alpha)
        stem=f'{i:02d}__{name.replace("/","-")}'
        canvas.save(OUT/'cutouts/train/images'/f'{stem}.jpg',quality=95)
        label(OUT/'cutouts/train/labels'/f'{stem}.txt',NAMES.index(name),(px,py,px+sz[0],py+sz[1]),canvas.size)
        # Composite preview on a real DIOR scene.  Use a scale appropriate for an 800px scene.
        bg=Image.open(bgs[(i*173)%len(bgs)]).convert('RGB'); target_long=[112,138,126,132,166,118][i-1]
        sc=target_long/max(fg.size); psize=(max(8,round(fg.width*sc)),max(8,round(fg.height*sc)))
        pf=fg.resize(psize,Image.Resampling.LANCZOS); pa=alpha.resize(psize,Image.Resampling.LANCZOS)
        x,y=[(96,110),(535,105),(128,500),(515,510),(330,292),(590,330)][i-1]
        bg.paste(pf,(x,y),pa); bg.save(OUT/'composites/train/images'/f'{stem}.jpg',quality=95)
        label(OUT/'composites/train/labels'/f'{stem}.txt',NAMES.index(name),(x,y,x+psize[0],y+psize[1]),bg.size)
    (OUT/'README.txt').write_text('仅供8999质检预览：来自当前SAM2 mask与DIOR背景的6组离线示例；非训练数据。\n')
    print('preview samples',len(picks),'root',OUT)
if __name__=='__main__': main()
