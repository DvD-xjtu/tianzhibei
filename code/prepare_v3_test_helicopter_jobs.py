#!/usr/bin/env python3
"""Prepare new helicopter cutout jobs from SIMD validation and unseen DOTA val scenes."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from produce_final_dataset import FINAL_ROOT
from produce_final_v3_mixed import ROOT as V3
from produce_f35_v3_supplement import ROOT as F35
from produce_sr71_v3_supplement import ROOT as SR71

OUT = FINAL_ROOT/'final_v3_20261001'/'test_2000_20261003'/'helicopter_sources'
SIMD = Path('/data3/tianzhibei/datasets/external_aircraft_supplement/SIMD')
DOTA = Path('/data2/tianzhibei/LAE-1M/LAE-FOD/DOTAv2')


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main():
    train_dota_scenes=set()
    used_fg=set()
    for root in (V3,F35,SR71):
        for p in (root/'shards').glob('*/manifest.jsonl'):
            for r in rows(p):
                if r['subdataset']=='DOTAv2':
                    train_dota_scenes.add(re.sub(r'_\d{4}$','',Path(r['background_image_path']).stem))
                used_fg.update(i['foreground_id'] for i in r['instances'])
    for root in (V3,F35,SR71):
        for r in rows(root/'foreground_pool.jsonl'):
            if r['id'] in used_fg and r.get('source_scene_key','').startswith('DOTAv2/'):
                train_dota_scenes.add(r['source_scene_key'].split('/',1)[1])

    jobs=[]
    for r in json.loads((SIMD/'helicopter_inventory.json').read_text()):
        if r['source_split']!='held_out': continue
        image=SIMD/'validation'/f"{r['image_stem']}.jpg"
        if not image.is_file():
            # Three manually held-out objects belong to SIMD's official train
            # partition; keep this test source strictly in official validation.
            continue
        b=r['bbox']
        jobs.append(dict(id=f"test_SIMD_{r['image_stem']}_{r['object_index']}",
            class_name='Helicopter',image_path=str(image),
            bbox_xyxy=[float(b[k]) for k in ('xmin','ymin','xmax','ymax')],
            source='simd_validation',source_dataset='SIMD_validation',source_split='valid',
            source_scene_key=f"SIMD_validation/{r['image_stem']}",
            source_url='https://github.com/ihians/simd',training_eligible=False,
            test_eligible=True,annotation_path=r['annotation_path']))

    data=json.loads((DOTA/'DOTAv2_val.json').read_text())
    images={r['id']:r for r in data['images']}
    by_scene=defaultdict(list)
    for ann in data['annotations']:
        if ann['category_id']!=12: continue
        im=images[ann['image_id']]
        scene=re.sub(r'_\d{4}$','',Path(im['file_name']).stem)
        if scene in train_dota_scenes: continue
        x,y,w,h=ann['bbox']
        if min(w,h)<24: continue
        by_scene[scene].append(dict(id=f"test_DOTAv2_{ann['id']}",
            class_name='Helicopter',image_path=str(DOTA/'images'/im['file_name']),
            bbox_xyxy=[x,y,x+w,y+h],source='dotav2_val_unseen_scene',
            source_dataset='DOTAv2_validation',source_split='valid',
            source_scene_key=f'DOTAv2/{scene}',training_eligible=False,
            test_eligible=True,annotation_id=ann['id'],
            annotation_manifest=str(DOTA/'DOTAv2_val.json')))
    for group in by_scene.values(): jobs.extend(group)
    assert len({r['id'] for r in jobs})==len(jobs)
    assert all(Path(r['image_path']).is_file() for r in jobs)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'source_jobs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in jobs))
    (OUT/'source_summary.json').write_text(json.dumps(dict(
        simd_validation=sum(r['source_dataset']=='SIMD_validation' for r in jobs),
        dota_val_unseen_scene=sum(r['source_dataset']=='DOTAv2_validation' for r in jobs),
        dota_scenes=sorted(by_scene),train_dota_scenes_checked=len(train_dota_scenes)),indent=2))
    print((OUT/'source_summary.json').read_text())


if __name__=='__main__': main()
