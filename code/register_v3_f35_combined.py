#!/usr/bin/env python3
"""Create a symlink-based 21k training view over base v3 plus F-35 supplement."""
from __future__ import annotations
from collections import Counter
import json,os
from pathlib import Path

from produce_final_dataset import SUBSETS,json_write
from produce_final_v3_mixed import ROOT as V3
from produce_f35_v3_supplement import ROOT as F35
from synthesize_copypaste_v6 import atomic_bytes,digest

ROOT=V3/'combined_21000'


def main():
    names=json.loads((V3/'taxonomy.json').read_text())
    assert names==json.loads((F35/'taxonomy.json').read_text())
    ROOT.mkdir(parents=True,exist_ok=True)
    atomic_bytes(ROOT/'taxonomy.json',(V3/'taxonomy.json').read_bytes())
    counts=Counter();seen=set();rows=[]
    for batch,label in ((V3,'base_v3'),(F35,'F35_supplement')):
        for ds in SUBSETS:
            manifest=batch/'shards'/ds/'manifest.jsonl'
            for s in manifest.read_text().splitlines():
                if not s.strip():continue
                r=json.loads(s);stem=r['id']
                assert stem not in seen;seen.add(stem)
                for key,subdir in (('image_path','train/images'),('label_path','train/labels'),('instance_mask_path','instance_masks')):
                    source=Path(r[key]);assert source.is_file()
                    dest=ROOT/subdir/source.name
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    if dest.is_symlink():assert dest.resolve()==source.resolve()
                    else:
                        assert not dest.exists()
                        os.symlink(source,dest)
                rows.append(dict(id=stem,batch=label,subdataset=ds,image_path=str(ROOT/'train/images'/Path(r['image_path']).name),
                    label_path=str(ROOT/'train/labels'/Path(r['label_path']).name),
                    mask_path=str(ROOT/'instance_masks'/Path(r['instance_mask_path']).name),
                    provenance_manifest=str(manifest),classes=sorted({i['class_name'] for i in r['instances']})))
                counts[label]+=1
    assert counts=={'base_v3':20000,'F35_supplement':1000}
    assert sum('F-35' in r['classes'] for r in rows)==1000
    atomic_bytes(ROOT/'index.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode())
    yaml='path: '+str(ROOT)+'\ntrain: train/images\n'
    yaml+='names:\n'+''.join(f'  {i}: {json.dumps(n,ensure_ascii=False)}\n' for i,n in enumerate(names))
    yaml+='# Add an independent val split before model evaluation.\n'
    atomic_bytes(ROOT/'dataset_train_only.yaml',yaml.encode())
    json_write(ROOT/'summary.json',dict(images=len(rows),batches=dict(counts),F35_images=1000,
        taxonomy_sha256=digest(ROOT/'taxonomy.json'),view_type='symlinks; original batches preserved',
        image_dir=str(ROOT/'train/images'),label_dir=str(ROOT/'train/labels'),
        mask_dir=str(ROOT/'instance_masks'),index=str(ROOT/'index.jsonl'),
        note='Training view only; no independent validation split is included.'))
    print(ROOT,len(rows),counts)


if __name__=='__main__':main()
