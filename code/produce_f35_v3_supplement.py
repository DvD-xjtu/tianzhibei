#!/usr/bin/env python3
"""1,000 v3-compatible mixed images, each containing at least one F-35."""
from __future__ import annotations
import argparse,json,multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

from produce_final_dataset import FINAL_ROOT,SUBSETS,source_values,json_write
from produce_extra_engine import run_subset,RULES,GATES
from synthesize_copypaste_v6 import atomic_bytes,digest

V3=FINAL_ROOT/'final_v3_20261001'
ROOT=V3/'extra/F-35_1000_20261002'
VERSION='final-v3-F35-supplement-v1'


def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def prepare(root,count):
    root.mkdir(parents=True,exist_ok=True)
    base=rows(V3/'foreground_pool.jsonl')
    selected=rows(FINAL_ROOT/'extra/class_supplement_v1/batches/F-35/foreground_pool.jsonl')
    known={r['source_image_path'] for r in selected}
    for r in rows(FINAL_ROOT/'v2_preview_20261001/foreground_pool.jsonl'):
        if r['class_name']=='F-35' and r['source_image_path'] not in known:
            selected.append(r);known.add(r['source_image_path'])
    test=[r for r in rows(FINAL_ROOT/'crop-metadata/heldout_reference_pool.jsonl')
          if r['class_name']=='F-35' and r['source']=='innar_test']
    good=[]
    for r in test:
        try:m=source_values(r['cutout_path'],r['source_crop_path'])
        except (OSError,ValueError):continue
        if m['source_background_like_fraction']>GATES['max_source_background_like_fraction'] or m['secondary_mask_fraction']>GATES['max_secondary_mask_fraction']:
            continue
        r.update(source_path=r['source_crop_path'],source_metrics=m,source_split='test',
                 training_eligible=True,source_domain='remote_sensing_dataset',
                 source_scene_key='MTARSI-INNAR/'+Path(r['source_image_path']).stem)
        good.append(r)
    good.sort(key=lambda r:(-r['selected_score'],r['id']))
    selected+=good[:len(test)//2]
    assert len(test)==7 and len(selected)==10 and len({r['source_image_path'] for r in selected})==10
    assert len({r['id'] for r in base+selected})==len(base+selected)
    atomic_bytes(root/'foreground_pool.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in base+selected).encode())
    atomic_bytes(root/'taxonomy.json',(V3/'taxonomy.json').read_bytes())
    json_write(root/'source_summary.json',dict(F35_original_cases=len(selected),F35_train_cases=8,
        F35_test_available=len(test),F35_test_selected=len(good[:len(test)//2]),
        F35_test_ids=[r['id'] for r in good[:len(test)//2]],F35_source_counts=dict(Counter(r['source'] for r in selected))))
    json_write(root/'config.json',dict(version=VERSION,stem_prefix='v3_f35',mixed_classes=True,
        primary_class='F-35',requested_images=count,quality_gate=GATES,rules=RULES,
        foreground_pool_sha256=digest(root/'foreground_pool.jsonl'),
        definition='Every accepted image contains at least one F-35; remaining slots may contain other v3 classes.'))
    snap=root/'code_snapshot';snap.mkdir(exist_ok=True)
    for name in ('produce_f35_v3_supplement.py','produce_extra_engine.py','produce_final_dataset.py',
                 'synthesize_copypaste_v6.py','synthesize_lae1m_v9.py','filter_copypaste_v7.py'):
        p=Path(__file__).parent/name
        atomic_bytes(snap/name,p.read_bytes())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--count',type=int,default=1000);ap.add_argument('--workers',type=int,default=6)
    args=ap.parse_args();prepare(args.root,args.count)
    tasks=[]
    for i,ds in enumerate(SUBSETS):
        n=args.count//6+int(i<args.count%6)
        progress=args.root/'shards'/ds/'progress.json'
        if progress.exists():
            p=json.loads(progress.read_text())
            if p.get('completed') and p.get('goal')==n:continue
        tasks.append((ds,str(args.root),n,20261002+11*i))
    with ProcessPoolExecutor(max_workers=min(args.workers,len(tasks)),mp_context=multiprocessing.get_context('spawn')) as ex:
        future={ex.submit(run_subset,t):t for t in tasks}
        for f in as_completed(future):print(future[f][0],f.result(),flush=True)


if __name__=='__main__':main()
