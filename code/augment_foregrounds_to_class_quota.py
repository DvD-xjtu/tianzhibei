#!/usr/bin/env python3
"""Expand accepted MTARSI or MAR20 foregrounds to a per-class variant quota."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from augment_mtarsi_foregrounds_v1 import sha256, transform, write_png

BASE = Path('/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1')
CLASSES_13 = ['C-17','C-130','KC-135','KC-10','A-10','F-15','F-16','F/A-18',
              'F-35','B-1B','B-52H','E-2','E-3']
MTARSI_CLASSES = ['C-17','C-130','A-10','F-15','F-16','F/A-18','F-35','B-1B','B-52H','E-2','E-3']
MAR20_NAMES = ['A1 SU-35','A2 C-130','A3 C-17','A4 C-5','A5 F-16','A6 TU-160','A7 E-3',
               'A8 B-52','A9 P-3C','A10 B-1B','A11 E-8','A12 TU-22','A13 F-15','A14 KC-135',
               'A15 F-22','A16 F/A-18','A17 TU-95','A18 KC-10','A19 SU-34','A20 SU-24']
MAR20_MAP = {'A2 C-130':'C-130','A3 C-17':'C-17','A5 F-16':'F-16','A7 E-3':'E-3',
             'A8 B-52':'B-52H','A10 B-1B':'B-1B','A13 F-15':'F-15','A14 KC-135':'KC-135',
             'A16 F/A-18':'F/A-18','A18 KC-10':'KC-10'}
MTARSI_MAP = {'C-17':'C-17','C-130':'C-130','A-10':'A-10','F-15':'F-15','F-16':'F-16',
              'F/A-18':'F/A-18','F-35':'F-35','B-1B':'B-1B','B-52':'B-52H','E-2':'E-2','E-3':'E-3'}


def load_sources(kind: str, classes: list[str]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    if kind == 'mtarsi':
        root = BASE/'foreground_all_15taxonomy_v6'
        for path in sorted(root.glob('shard_*/manifest_quality_gated_v2.json')):
            for row in json.loads(path.read_text()):
                cls = MTARSI_MAP.get(row.get('class'))
                if (row.get('quality_gate',{}).get('accepted') and cls in classes
                        and row.get('source') in {'fixed','innar_train'}):
                    groups[cls].append({'id':row['id'],'class':cls,'source':row['source'],
                        'rgb_path':row['source_path'],'rgba_path':row['cutout_path'],
                        'source_manifest':str(path)})
    else:
        path = BASE/'foreground_mar20_500_v1/manifest_quality_gated.json'
        for row in json.loads(path.read_text()):
            cls = MAR20_MAP.get(row.get('class_name'))
            if row.get('quality_gate',{}).get('accepted') and cls in classes:
                groups[cls].append({'id':row['id'],'class':cls,'source':'mar20_train',
                    'rgb_path':row['source_crop_path'],'rgba_path':row['cutout_path'],
                    'source_manifest':str(path)})
    return groups


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',choices=['mtarsi','mar20'],required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--per-class',type=int,default=1400)
    ap.add_argument('--seed',type=int,default=20260928)
    args=ap.parse_args()
    classes=MTARSI_CLASSES if args.source=='mtarsi' else list(MAR20_MAP.values())
    if args.per_class<1 or (args.output.exists() and any(args.output.iterdir())):
        ap.error('per-class must be positive and output must be new or empty')
    groups=load_sources(args.source,classes)
    missing=[c for c in classes if not groups[c]]
    if missing: raise RuntimeError(f'no accepted sources for classes: {missing}')
    py=random.Random(args.seed); nrng=np.random.default_rng(args.seed)
    cut_dir=args.output/'cutouts'; ctx_dir=args.output/'source_context'
    records=[]; rejected=Counter(); source_counts={c:len(groups[c]) for c in classes}
    for ci,cls in enumerate(classes):
        sources=groups[cls].copy(); py.shuffle(sources)
        made=0; attempts=0; limit=args.per_class*3
        while made<args.per_class and attempts<limit:
            src=sources[attempts%len(sources)]; variant_index=attempts//len(sources)+1; attempts+=1
            try:
                context=np.asarray(Image.open(src['rgb_path']).convert('RGB'))
                rgba=np.asarray(Image.open(src['rgba_path']).convert('RGBA'))
            except (OSError,ValueError):
                rejected['source_read_error']+=1; continue
            if context.shape[:2]!=rgba.shape[:2]:
                rejected['shape_mismatch']+=1; continue
            angle=py.uniform(0,360); zoom=py.uniform(.8,1.2)
            brightness=py.uniform(-3,3); contrast=py.uniform(.97,1.03); noise=py.uniform(0,1.2)
            ctx,var,params=transform(context,rgba,angle,zoom,brightness,contrast,noise,nrng)
            alpha=var[:,:,3]>127
            n,_,stats,_=cv2.connectedComponentsWithStats(alpha.astype(np.uint8),8)
            areas=sorted(stats[1:n,cv2.CC_STAT_AREA].tolist(),reverse=True)
            total=sum(areas); secondary=sum(areas[1:])/max(total,1)
            if total<100 or not areas or areas[0]/total<.90 or secondary>.07 or alpha.mean()>.45:
                rejected['transformed_mask_gate']+=1; continue
            stem=f'{cls.replace("/","-")}__{src["id"]}__v{variant_index:04d}'
            cp=cut_dir/f'{stem}.png'; xp=ctx_dir/f'{stem}.png'
            write_png(cp,var); write_png(xp,ctx)
            records.append({'id':stem,'class':cls,'source':src['source'],'source_path':str(xp),
                'cutout_path':str(cp),'mask_path':'','foreground_pool':f'{args.source}_13class_quota_v1',
                'base_foreground_id':src['id'],'augmentation':params,'paste_scale_factor':zoom,
                'source_manifest':src['source_manifest'],'source_cutout_sha256':sha256(Path(src['rgba_path'])),
                'variant_sha256':sha256(cp),'mask_area_ratio':float(alpha.mean()),
                'component_count':int(n-1),'secondary_component_ratio':float(secondary),'gate':'passed'})
            made+=1
        if made<args.per_class:
            raise RuntimeError(f'{cls}: generated {made}/{args.per_class} variants after {attempts} attempts')
        print(json.dumps({'class':cls,'sources':source_counts[cls],'variants':made,'attempts':attempts},ensure_ascii=False),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    summary={'pipeline_version':f'{args.source}-foreground-13class-quota-v1','source':args.source,
        'per_class_quota':args.per_class,'total_variants':len(records),'classes':dict(Counter(r['class'] for r in records)),
        'base_sources_by_class':source_counts,'rejected_attempts':dict(rejected),'seed':args.seed,
        'transform':{'rotation_deg':[0,360],'zoom':[.8,1.2],'brightness':[-3,3],
                     'contrast':[.97,1.03],'noise_sigma':[0,1.2]},
        'mask_gate':{'min_alpha_pixels':100,'min_dominant_component_ratio':.90,
                     'max_secondary_component_ratio':.07,'max_alpha_area_ratio':.45}}
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__': main()
