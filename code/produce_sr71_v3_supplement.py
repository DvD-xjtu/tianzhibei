#!/usr/bin/env python3
"""Produce 500 v3-compatible YF-12A proxy composites from SR-71A NAIP v2 masks."""
from __future__ import annotations

import argparse
import json
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from produce_final_dataset import FINAL_ROOT, SUBSETS, source_values, json_write
from produce_extra_engine import run_subset, RULES, GATES
from synthesize_copypaste_v6 import atomic_bytes, digest

V3 = FINAL_ROOT / 'final_v3_20261001'
ROOT = V3 / 'extra' / 'SR71_proxy_YF12A_500_20261003'
SOURCE = FINAL_ROOT / 'extra/SR71_NAIP_sources_20261003/extraction_v2/manifest.json'
VERSION = 'final-v3-SR71A-proxy-YF12A-supplement-v1'
HOLDOUT_SITES = frozenset({'beale', 'edwards'})


def read_jsonl(path: Path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def prepare(root: Path, count: int):
    root.mkdir(parents=True, exist_ok=True)
    base = read_jsonl(V3/'foreground_pool.jsonl')
    source = json.loads(SOURCE.read_text())
    assert len(source) == 38 and len({r['id'] for r in source}) == 38
    assert len({r['serial'] for r in source}) == 7
    training, heldout, rejected = [], [], []
    for r in source:
        row = dict(r, id='sr71_v2__'+r['id'], class_name='YF-12A',
                   source='NAIP_SR71_v2', source_dataset='NAIP',
                   source_split='test_reserved' if r['site'] in HOLDOUT_SITES else 'train',
                   source_scene_key='NAIP/SR71/'+r['serial'],
                   source_path=r['source_crop_path'], source_image_path=r['image_path'],
                   source_url=r['original_cog_url'],
                   true_model='SR-71A', proxy_for='YF-12A',
                   source_domain='NAIP_orthorectified_aerial',
                   training_eligible=r['site'] not in HOLDOUT_SITES)
        if r['site'] in HOLDOUT_SITES:
            heldout.append(row)
            continue
        try:
            metrics=source_values(row['cutout_path'],row['source_path'])
        except (OSError,ValueError) as e:
            rejected.append(dict(id=row['id'],serial=row['serial'],reason=str(e)))
            continue
        row['source_metrics']=metrics
        if metrics['source_background_like_fraction'] > GATES['max_source_background_like_fraction']:
            rejected.append(dict(id=row['id'],serial=row['serial'],reason='source_background_leak',metrics=metrics))
            continue
        if metrics['secondary_mask_fraction'] > GATES['max_secondary_mask_fraction']:
            rejected.append(dict(id=row['id'],serial=row['serial'],reason='mask_fragments',metrics=metrics))
            continue
        training.append(row)
    assert len(training)==27 and len(heldout)==9 and len(rejected)==2
    assert not ({r['serial'] for r in training} & {r['serial'] for r in heldout})
    assert not ({r['id'] for r in base} & {r['id'] for r in training})
    assert all(r['true_model']=='SR-71A' and r['proxy_for']=='YF-12A' for r in training+heldout)
    atomic_bytes(root/'foreground_pool.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in base+training).encode())
    atomic_bytes(root/'sr71_training_cases.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in training).encode())
    atomic_bytes(root/'sr71_heldout_cases.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in heldout).encode())
    json_write(root/'sr71_rejected_cases.json',rejected)
    atomic_bytes(root/'taxonomy.json',(V3/'taxonomy.json').read_bytes())
    json_write(root/'source_split_summary.json',dict(source_cases=len(source),training_cases=len(training),
        reserved_test_cases=len(heldout),rejected_train_cases=len(rejected),
        training_airframes=sorted({r['serial'] for r in training}),
        reserved_test_airframes=sorted({r['serial'] for r in heldout}),
        training_by_site=dict(Counter(r['site'] for r in training)),
        heldout_by_site=dict(Counter(r['site'] for r in heldout)),
        rejected_ids=[r['id'] for r in rejected],
        split_unit='physical_airframe_serial; all years of same airframe in one split',
        heldout_policy='reserved source crops only; no detector test images created'))
    json_write(root/'config.json',dict(version=VERSION,stem_prefix='v3_sr71_proxy',mixed_classes=True,
        primary_class='YF-12A',requested_images=count,quality_gate=GATES,rules=RULES,
        foreground_pool_sha256=digest(root/'foreground_pool.jsonl'),
        true_model='SR-71A',competition_proxy_label='YF-12A',
        definition='Every accepted image contains at least one SR-71A crop annotated with the YF-12A proxy class ID; other slots may contain v3 classes.',
        source_split_policy='5 airframes train, 2 airframes reserved; no shared airframe across splits'))
    snap=root/'code_snapshot';snap.mkdir(exist_ok=True)
    for name in ('produce_sr71_v3_supplement.py','produce_extra_engine.py','produce_final_dataset.py',
                 'synthesize_copypaste_v6.py','synthesize_lae1m_v9.py','filter_copypaste_v7.py',
                 'refine_sr71_naip_v2.py'):
        p=Path(__file__).parent/name
        atomic_bytes(snap/name,p.read_bytes())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--count',type=int,default=500)
    ap.add_argument('--workers',type=int,default=6)
    args=ap.parse_args()
    assert args.count>=6
    prepare(args.root,args.count)
    tasks=[]
    for i,ds in enumerate(SUBSETS):
        n=args.count//6+int(i<args.count%6)
        progress=args.root/'shards'/ds/'progress.json'
        if progress.exists():
            p=json.loads(progress.read_text())
            if p.get('completed') and p.get('goal')==n:continue
        tasks.append((ds,str(args.root),n,20261003+11*i))
    with ProcessPoolExecutor(max_workers=min(args.workers,len(tasks)),
                             mp_context=multiprocessing.get_context('spawn')) as executor:
        futures={executor.submit(run_subset,t):t for t in tasks}
        for future in as_completed(futures):
            print(futures[future][0],future.result(),flush=True)


if __name__=='__main__':
    main()
