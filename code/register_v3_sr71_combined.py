#!/usr/bin/env python3
"""Add the audited SR-71 proxy supplement to the existing v3 training view."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from produce_final_dataset import SUBSETS, json_write
from produce_final_v3_mixed import ROOT as V3
from produce_sr71_v3_supplement import ROOT as SR71
from synthesize_copypaste_v6 import atomic_bytes, digest

BASE = V3/'combined_21000'
ROOT = V3/'combined_21500_sr71_proxy'


def link(source: Path, destination: Path):
    assert source.is_file()
    destination.parent.mkdir(parents=True,exist_ok=True)
    target=source.resolve()
    if destination.is_symlink():
        assert destination.resolve()==target
    else:
        assert not destination.exists()
        os.symlink(target,destination)


def main():
    assert json.loads((BASE/'summary.json').read_text())['images']==21000
    sr=json.loads((SR71/'summary.json').read_text())
    assert sr['images']==500 and sr['audit_passed']
    assert (BASE/'taxonomy.json').read_bytes()==(SR71/'taxonomy.json').read_bytes()
    ROOT.mkdir(parents=True,exist_ok=True)
    atomic_bytes(ROOT/'taxonomy.json',(BASE/'taxonomy.json').read_bytes())
    rows=[];counts=Counter();seen=set()
    for s in (BASE/'index.jsonl').read_text().splitlines():
        if not s:continue
        row=json.loads(s);stem=row['id']
        assert stem not in seen;seen.add(stem)
        for key,directory in (('image_path','train/images'),('label_path','train/labels'),('mask_path','instance_masks')):
            source=Path(row[key]);destination=ROOT/directory/source.name
            link(source,destination);row[key]=str(destination)
        rows.append(row);counts[row['batch']]+=1
    for ds in SUBSETS:
        manifest=SR71/'shards'/ds/'manifest.jsonl'
        for s in manifest.read_text().splitlines():
            if not s:continue
            r=json.loads(s);stem=r['id']
            assert stem not in seen;seen.add(stem)
            for key,directory in (('image_path','train/images'),('label_path','train/labels'),('instance_mask_path','instance_masks')):
                link(Path(r[key]),ROOT/directory/Path(r[key]).name)
            rows.append(dict(id=stem,batch='SR71_proxy_YF12A_supplement',subdataset=ds,
                image_path=str(ROOT/'train/images'/Path(r['image_path']).name),
                label_path=str(ROOT/'train/labels'/Path(r['label_path']).name),
                mask_path=str(ROOT/'instance_masks'/Path(r['instance_mask_path']).name),
                provenance_manifest=str(manifest),
                classes=sorted({i['class_name'] for i in r['instances']}),
                proxy_true_model='SR-71A',proxy_target_label='YF-12A'))
            counts['SR71_proxy_YF12A_supplement']+=1
    assert len(rows)==21500 and counts=={'base_v3':20000,'F35_supplement':1000,'SR71_proxy_YF12A_supplement':500}
    assert sum(r['batch']=='SR71_proxy_YF12A_supplement' and 'YF-12A' in r['classes'] for r in rows)==500
    atomic_bytes(ROOT/'index.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode())
    names=json.loads((ROOT/'taxonomy.json').read_text())
    yaml='path: '+str(ROOT)+'\ntrain: train/images\nnames:\n'
    yaml+=''.join(f'  {i}: {json.dumps(n,ensure_ascii=False)}\n' for i,n in enumerate(names))
    yaml+='# Train only; independent val/test images must be provided separately.\n'
    atomic_bytes(ROOT/'dataset_train_only.yaml',yaml.encode())
    json_write(ROOT/'summary.json',dict(images=len(rows),batches=dict(counts),
        SR71_proxy_images=500,SR71_true_model='SR-71A',proxy_label='YF-12A',
        taxonomy_sha256=digest(ROOT/'taxonomy.json'),
        view_type='symlinks; all original batches preserved',
        image_dir=str(ROOT/'train/images'),label_dir=str(ROOT/'train/labels'),
        mask_dir=str(ROOT/'instance_masks'),index=str(ROOT/'index.jsonl'),
        note='Training view only. SR-71 v2 crop holdout is airframe-disjoint; no independent detector test split is included.'))
    print(ROOT,len(rows),dict(counts))


if __name__=='__main__':main()
