#!/usr/bin/env python3
"""Final v3: about 20k multi-class composites, balanced across LAE subsets."""
from __future__ import annotations
import argparse,json,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed

from produce_final_dataset import FINAL_ROOT,SUBSETS,json_write
from produce_final_v3 import CLASSES
from produce_extra_engine import run_subset,RULES,GATES
from synthesize_copypaste_v6 import atomic_bytes,digest

ROOT=FINAL_ROOT/'final_v3_20261001'
SOURCE=FINAL_ROOT/'extra/v3_single_class_trial_20261001'
VERSION='final-v3-mixed-13class'


def prepare(root,count):
    root.mkdir(parents=True,exist_ok=True)
    source_summary=json.loads((SOURCE/'source_pool_summary.json').read_text())
    assert source_summary['classes']==CLASSES
    pool=[]
    for cls in CLASSES:
        path=SOURCE/'batches'/cls/'foreground_pool.jsonl'
        rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        assert rows and all(r['class_name']==cls for r in rows)
        pool.extend(rows)
    assert len({r['id'] for r in pool})==len(pool)
    atomic_bytes(root/'foreground_pool.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pool).encode())
    atomic_bytes(root/'taxonomy.json',(SOURCE/'batches'/CLASSES[0]/'taxonomy.json').read_bytes())
    atomic_bytes(root/'source_pool_summary.json',(SOURCE/'source_pool_summary.json').read_bytes())
    json_write(root/'config.json',dict(version=VERSION,stem_prefix='v3',mixed_classes=True,
        requested_images=count,classes=CLASSES,subsets=SUBSETS,
        class_sampling='sqrt(usable original crop count), with feedback for successful instances',
        foreground_pool_sha256=digest(root/'foreground_pool.jsonl'),rules=RULES,quality_gate=GATES,
        test_crop_policy='E-2 only: 4 of 10 available test originals; all other classes train/unsplit'))
    snapshot=root/'code_snapshot';snapshot.mkdir(exist_ok=True)
    for name in ('produce_final_v3_mixed.py','produce_final_v3.py','produce_extra_engine.py',
                 'produce_final_dataset.py','synthesize_copypaste_v6.py','synthesize_lae1m_v9.py','filter_copypaste_v7.py'):
        p=Path(__file__).parent/name
        atomic_bytes(snapshot/name,p.read_bytes())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--count',type=int,default=20000);ap.add_argument('--workers',type=int,default=16)
    args=ap.parse_args();assert args.count>=6
    prepare(args.root,args.count)
    tasks=[]
    for i,ds in enumerate(SUBSETS):
        n=args.count//6+int(i<args.count%6)
        progress=args.root/'shards'/ds/'progress.json'
        if progress.exists():
            p=json.loads(progress.read_text())
            if p.get('completed') and p.get('goal')==n:continue
        tasks.append((ds,str(args.root),n,20261001+11*i))
    with ProcessPoolExecutor(max_workers=min(args.workers,len(tasks)),mp_context=multiprocessing.get_context('spawn')) as ex:
        future={ex.submit(run_subset,t):t for t in tasks}
        for f in as_completed(future):print(future[f][0],f.result(),flush=True)


if __name__=='__main__':main()
