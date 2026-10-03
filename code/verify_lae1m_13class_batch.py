#!/usr/bin/env python3
"""Structural/source audit for large LAE batches using the 13-class taxonomy."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from synthesize_copypaste_v6 import digest
from synthesize_lae1m_13class_batch import CLASSES_13, MTARSI_CLASSES, MAR20_CLASSES
ALLOWED={'DIOR','DOTAv2','FAIR1M','NWPU','RSOD','xView'}

def audit(root:Path,expected_dataset:str|None=None)->dict:
    records=[json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines() if s.strip()]
    summary=json.loads((root/'qa/summary.json').read_text()); config=json.loads((root/'generation_config.json').read_text())
    ds=summary['subdataset']; source=config['foreground_source'].lower()
    classes=MTARSI_CLASSES if source=='mtarsi' else MAR20_CLASSES
    ids={c:CLASSES_13.index(c) for c in classes}
    assert ds in ALLOWED and (expected_dataset is None or ds==expected_dataset)
    assert summary['generated']==config['count']==len(records) and summary['train_ready'] is False
    assert len({r['id'] for r in records})==len(records)
    counts=Counter()
    for r in records:
        assert r['pipeline_version']==f'copy-paste-{source}-13class-lae-v1'
        assert r['subdataset']==ds and r['background_dataset']==ds and r['background_split']=='train'
        assert r['class_name'] in ids and r['target_class_id']==ids[r['class_name']]
        assert '/LAE-1M/LAE-FOD/' in r['background_image_path']
        if ds=='FAIR1M': assert Path(r['background_image_path']).name.startswith('train_')
        for pk,hk in [('background_image_path','background_image_sha256'),('background_label_path','background_label_sha256'),
          ('foreground_path','foreground_sha256'),('image_path','image_sha256'),('label_path','label_sha256'),
          ('instance_mask_path','instance_mask_sha256'),('original_annotations_path','original_annotations_sha256')]:
            assert digest(Path(r[pk]))==r[hk],(r['id'],pk)
        im=Image.open(r['image_path']).convert('RGB'); w,h=im.size
        mask=cv2.imread(r['instance_mask_path'],cv2.IMREAD_GRAYSCALE)
        assert mask is not None and mask.shape==(h,w) and mask.max()==255
        x,y,bw,bh=cv2.boundingRect(cv2.findNonZero((mask>0).astype(np.uint8)))
        assert r['bbox_xyxy']==[x,y,x+bw,y+bh]
        tok=Path(r['label_path']).read_text().strip().split()
        assert len(tok)==5 and int(tok[0])==ids[r['class_name']]
        v=np.array([float(t) for t in tok[1:]])
        assert np.isfinite(v).all() and ((v>=0)&(v<=1)).all()
        ann=json.loads(Path(r['original_annotations_path']).read_text()); assert ann['split']=='train'
        counts[r['class_name']]+=1
    assert dict(counts)==summary['classes']
    return {'source':source,'subdataset':ds,'verified':len(records),'classes':dict(counts),'errors':0,'train_ready':False}

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('batch',type=Path); ap.add_argument('--dataset'); a=ap.parse_args()
    print(json.dumps(audit(a.batch,a.dataset),ensure_ascii=False))
