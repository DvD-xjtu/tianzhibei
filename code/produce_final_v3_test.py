#!/usr/bin/env python3
"""Build a source-disjoint synthetic test batch for final v3."""
from __future__ import annotations

import argparse
import json
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from produce_final_dataset import FINAL_ROOT, json_write
from produce_extra_engine import GATES, RULES, run_subset
from produce_final_v3_mixed import ROOT as V3
from produce_f35_v3_supplement import ROOT as F35
from produce_sr71_v3_supplement import ROOT as SR71
from synthesize_copypaste_v6 import atomic_bytes, digest

ROOT = V3/'test_case_limited_1500_20261003'
HELI = V3/'test_2000_20261003'/'helicopter_sources'
SHARDS = [f'FAIR1M_valid_{i}' for i in range(6)]
RARE = {'F-35', 'E-2'}
CLASSES = json.loads((V3/'taxonomy.json').read_text())[:15]
WEIGHTS = {c: 1.0 for c in CLASSES}
WEIGHTS.update({'F-35': .12, 'E-2': .12, 'YF-12A': .4, 'Helicopter': .6})
HELI_VISUAL_REJECT = {'test_SIMD_2456_0': 'helipad_not_aircraft',
                      'test_SIMD_3104_2': 'incomplete_or_wrong_object'}


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def prepare(root: Path, count: int):
    root.mkdir(parents=True,exist_ok=True)
    quality = {r['id']:r for r in rows(V3/'combined_21500_sr71_proxy'/'heldout_source_quality_audit.jsonl')}
    refs=rows(FINAL_ROOT/'crop-metadata'/'heldout_reference_pool.jsonl')
    sr=rows(SR71/'sr71_heldout_cases.jsonl')
    heli=rows(HELI/'extraction'/'pool_None.jsonl')

    train_ids=set();train_sources=set();train_sr_serials=set()
    for batch in (V3,F35,SR71):
        for p in (batch/'shards').glob('*/manifest.jsonl'):
            for image in rows(p):
                for instance in image['instances']:
                    train_ids.add(instance['foreground_id'])
                    train_sources.add(instance['source_image_path'])
    train_sr_serials={r['serial'] for r in rows(SR71/'sr71_training_cases.jsonl')}

    pool=[];rejected=[]
    for r in refs+sr:
        q=quality[r['id']]
        if not q['passes_v3_source_gate']:
            rejected.append(dict(id=r['id'],reason='v3_source_quality_gate'))
            continue
        if r['id'] in train_ids or r['source_image_path'] in train_sources:
            rejected.append(dict(id=r['id'],reason='used_in_training'))
            continue
        if r['class_name'] not in RARE and r['source']=='innar_valid':
            rejected.append(dict(id=r['id'],reason='reserve_valid_for_future_validation'))
            continue
        if r['class_name'] in RARE and r['source']!='innar_valid':
            rejected.append(dict(id=r['id'],reason='rare_class_requires_clean_valid_source'))
            continue
        if r.get('true_model')=='SR-71A':
            assert r['serial'] not in train_sr_serials
        r=dict(r,source_path=r.get('source_path',r.get('source_crop_path')),
               source_metrics=q['metrics'],test_eligible=True,
               training_eligible=False)
        r.setdefault('source_scene_key',r['source_dataset']+'/'+Path(r['source_image_path']).stem)
        pool.append(r)
    for r in heli:
        if r['id'] in HELI_VISUAL_REJECT:
            rejected.append(dict(id=r['id'],reason=HELI_VISUAL_REJECT[r['id']]))
            continue
        if r['source_image_path'] in train_sources:
            rejected.append(dict(id=r['id'],reason='source_used_in_training'))
            continue
        r=dict(r,source_path=r['source_crop_path'],test_eligible=True,
               training_eligible=False,visual_review='manual_pass_2026-10-03')
        pool.append(r)
    assert len({r['id'] for r in pool})==len(pool)
    assert set(CLASSES)=={r['class_name'] for r in pool}
    assert not {r['id'] for r in pool}&train_ids
    assert not {r['source_image_path'] for r in pool}&train_sources
    assert all(Path(r['cutout_path']).is_file() and Path(r['source_path']).is_file() for r in pool)
    assert Counter(r['class_name'] for r in pool)['F-35']==1
    assert Counter(r['class_name'] for r in pool)['E-2']==2
    assert Counter(r['class_name'] for r in pool)['YF-12A']==8

    atomic_bytes(root/'foreground_pool.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pool).encode())
    atomic_bytes(root/'taxonomy.json',(V3/'taxonomy.json').read_bytes())
    json_write(root/'foreground_rejections.json',rejected)
    json_write(root/'foreground_summary.json',dict(cases=len(pool),by_class=dict(Counter(r['class_name'] for r in pool)),
        by_source=dict(Counter(r['source'] for r in pool)),rejected_by_reason=dict(Counter(r['reason'] for r in rejected)),
        rare_classes=sorted(RARE),sr71_proxy_heldout_airframes=sorted({r['serial'] for r in pool if r['class_name']=='YF-12A'}),
        train_source_paths_disjoint=True,train_crop_ids_disjoint=True))
    single_use_classes=sorted(c for c,n in Counter(r['class_name'] for r in pool).items() if n>=300)
    json_write(root/'config.json',dict(version='final-v3-synthetic-test-case-limited-v2',stem_prefix='v3_test',
        mixed_classes=True,requested_images=count,classes=CLASSES,rare_classes=sorted(RARE),
        class_sampling_weights=WEIGHTS,background_dataset='FAIR1M',background_split='valid',
        background_shards=6,foreground_shards=6,output_split='test',rules=RULES,quality_gate=GATES,
        max_case_uses=3,single_use_classes=single_use_classes,
        foreground_pool_sha256=digest(root/'foreground_pool.jsonl'),
        source_isolation='exact crop ID and source path; SR-71 airframe serial; FAIR1M valid_ background tiles',
        evaluation_scope='synthetic copy-paste test, not an independent real-image benchmark'))
    snap=root/'code_snapshot';snap.mkdir(exist_ok=True)
    for name in ('produce_final_v3_test.py','prepare_v3_test_helicopter_jobs.py',
                 'extract_source_expansion.py','produce_extra_engine.py',
                 'synthesize_lae1m_v9.py','synthesize_copypaste_v6.py','filter_copypaste_v7.py'):
        p=Path(__file__).parent/name
        atomic_bytes(snap/name,p.read_bytes())
    return Counter(r['class_name'] for r in pool)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--count',type=int,default=1500)
    ap.add_argument('--workers',type=int,default=6)
    args=ap.parse_args()
    assert args.count>=6
    print('test foreground pool',prepare(args.root,args.count),flush=True)
    tasks=[]
    for i,shard in enumerate(SHARDS):
        n=args.count//6+int(i<args.count%6)
        progress=args.root/'shards'/shard/'progress.json'
        if progress.exists():
            p=json.loads(progress.read_text())
            if p.get('completed') and p.get('goal')==n:continue
        tasks.append((shard,str(args.root),n,20261003+29*i))
    with ProcessPoolExecutor(max_workers=min(args.workers,len(tasks)),
                             mp_context=multiprocessing.get_context('spawn')) as ex:
        future={ex.submit(run_subset,t):t for t in tasks}
        for f in as_completed(future): print(future[f][0],f.result(),flush=True)


if __name__=='__main__': main()
