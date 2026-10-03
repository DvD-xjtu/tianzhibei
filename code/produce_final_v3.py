#!/usr/bin/env python3
"""Generate balanced class-specific final-v3 LAE composites with v2 paste rules."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path

from filter_copypaste_v7 import GATES
from produce_extra_engine import run_subset, CLASSES_13, RULES
from produce_final_dataset import FINAL_ROOT, SUBSETS, taxonomy, json_write, source_values
from synthesize_copypaste_v6 import atomic_bytes, digest

ROOT = FINAL_ROOT/'final_v3_20261001'
VERSION = 'final-v3-balanced-13class'
CLASSES = [c for c in CLASSES_13 if c not in ('F-35', 'YF-12A')]


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def quota(cls, ds, per_class):
    if per_class != 1500:
        i = SUBSETS.index(ds)
        return per_class//len(SUBSETS)+int(i<per_class%len(SUBSETS))
    if cls == 'E-2':
        return 0 if ds == 'xView' else 300
    if ds == 'xView':
        return 271
    # Rotate the single 245 quota across the 12 other classes.
    i = SUBSETS.index(ds)
    return 245 if i == CLASSES.index(cls)%5 else 246


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    base = FINAL_ROOT/'v2_preview_20261001'
    rows = read_rows(base/'foreground_pool.jsonl')
    rows = [r for r in rows if r['class_name'] in CLASSES]
    counts = Counter(r['class_name'] for r in rows)
    # Two genuinely new E-2 source photographs survived the extra-v1 source review.
    known = {r['source_image_path'] for r in rows if r['class_name']=='E-2'}
    extra = read_rows(FINAL_ROOT/'extra/class_supplement_v1/batches/E-2/foreground_pool.jsonl')
    for r in extra:
        if r['source_image_path'] not in known:
            rows.append(r); known.add(r['source_image_path'])
    # E-2 is far below the requested ~100 originals. Use only quality-passing test
    # cutouts, capped at half of the 10 E-2 test originals. Never use validation.
    held = [r for r in read_rows(FINAL_ROOT/'crop-metadata/heldout_reference_pool.jsonl')
            if r['class_name']=='E-2' and r['source']=='innar_test']
    candidates = []
    for r in held:
        try:
            m = source_values(r['cutout_path'], r['source_crop_path'])
        except (OSError, ValueError):
            continue
        if m['source_background_like_fraction'] > GATES['max_source_background_like_fraction'] or m['secondary_mask_fraction'] > GATES['max_secondary_mask_fraction']:
            continue
        r.update(source_path=r['source_crop_path'], source_metrics=m,
                 source_split='test', training_eligible=True,
                 source_domain='remote_sensing_dataset',
                 source_scene_key='MTARSI-INNAR/'+Path(r['source_image_path']).stem)
        candidates.append(r)
    candidates.sort(key=lambda r:(-r['selected_score'],r['id']))
    selected = candidates[:len(held)//2]
    rows += selected
    assert len(selected) <= len(held)//2
    heli = read_rows(FINAL_ROOT/'extra/source_expansion/selected/pool_Helicopter.jsonl')
    assert len(heli)==100 and all(r['source_split']=='train' for r in heli)
    rows += heli
    assert len({r['id'] for r in rows})==len(rows)
    names = CLASSES_13 + [n for n in taxonomy()[13:] if n not in ('LAE/Helicopter','LAE/helicopter')]
    for cls in CLASSES:
        batch=root/'batches'/cls
        batch.mkdir(parents=True,exist_ok=True)
        subset=[r for r in rows if r['class_name']==cls]
        assert subset,cls
        atomic_bytes(batch/'foreground_pool.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in subset).encode())
        json_write(batch/'taxonomy.json',names)
        json_write(batch/'config.json',dict(version=VERSION,stem_prefix='v3',rules=RULES,quality_gate=GATES,
            source_class=cls,requested_images=1500,
            requested_per_subdataset={ds:quota(cls,ds,1500) for ds in SUBSETS},
            test_original_limit=len(held)//2,source_pool_sha256=digest(batch/'foreground_pool.jsonl')))
    json_write(root/'source_pool_summary.json',dict(classes=CLASSES,source_counts=dict(Counter(r['class_name'] for r in rows)),
        E2_test_available=len(held),E2_test_selected=len(selected),E2_test_ids=[r['id'] for r in selected],
        E2_new_extra_originals=len(known)-counts['E-2'],
        sources_by_class={c:dict(Counter(r['source'] for r in rows if r['class_name']==c)) for c in CLASSES}))
    snapshot=root/'code_snapshot';snapshot.mkdir(exist_ok=True)
    for name in ('produce_final_v3.py','produce_extra_engine.py','produce_final_dataset.py','synthesize_copypaste_v6.py','synthesize_lae1m_v9.py','filter_copypaste_v7.py'):
        p=Path(__file__).parent/name
        atomic_bytes(snapshot/name,p.read_bytes())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--count-per-class',type=int,default=1500)
    ap.add_argument('--workers',type=int,default=12)
    ap.add_argument('--classes',nargs='+',default=CLASSES)
    ap.add_argument('--subsets',nargs='+',default=SUBSETS)
    ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args()
    prepare(args.root)
    if args.prepare_only:return
    tasks=[]
    for cls in args.classes:
        assert cls in CLASSES
        for i,ds in enumerate(args.subsets):
            assert ds in SUBSETS
            n=quota(cls,ds,args.count_per_class)
            if n==0:continue
            out=args.root/'batches'/cls/'shards'/ds
            if (out/'progress.json').exists() and json.loads((out/'progress.json').read_text()).get('completed') and json.loads((out/'progress.json').read_text()).get('goal')==n:
                continue
            tasks.append((ds,str(args.root/'batches'/cls),n,20261001+1000*CLASSES.index(cls)+i))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as ex:
        futures={ex.submit(run_subset,t):t for t in tasks}
        for future in as_completed(futures):
            print(futures[future][1],futures[future][0],future.result(),flush=True)


if __name__=='__main__':main()
