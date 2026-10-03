#!/usr/bin/env python3
"""Verify the 500-image SR-71 proxy supplement and its airframe holdout."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from produce_final_dataset import SUBSETS, json_write
from produce_sr71_v3_supplement import ROOT
from filter_copypaste_v7 import rejection_reasons
from synthesize_copypaste_v6 import yolo_hbb_line


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def audit(root=ROOT, expected=500):
    taxonomy=json.loads((root/'taxonomy.json').read_text())
    names={name:i for i,name in enumerate(taxonomy)}
    assert names['YF-12A']==13
    pool_rows=read(root/'foreground_pool.jsonl')
    pool={r['id']:r for r in pool_rows}
    assert len(pool)==len(pool_rows)
    training=read(root/'sr71_training_cases.jsonl')
    heldout=read(root/'sr71_heldout_cases.jsonl')
    assert len(training)==27 and len(heldout)==9
    assert not ({r['serial'] for r in training}&{r['serial'] for r in heldout})
    assert not ({r['id'] for r in training}&{r['id'] for r in heldout})
    rows=[];attempts={};subsets={}
    for i,ds in enumerate(SUBSETS):
        shard=root/'shards'/ds
        progress=json.loads((shard/'progress.json').read_text())
        quota=expected//6+int(i<expected%6)
        assert progress['completed'] and progress['goal']==quota and progress['images']==quota
        part=read(shard/'manifest.jsonl')
        assert len(part)==quota
        rows.extend(part);attempts[ds]=progress['attempts'];subsets[ds]=quota
    assert len(rows)==expected and len({r['id'] for r in rows})==expected
    uses=Counter();by_class=Counter();actual=Counter();mixed=0;foreground_ids=set()
    train_ids={r['id'] for r in training};heldout_ids={r['id'] for r in heldout}
    for r in rows:
        assert r['quality_gate_passed'] and 1<=r['actual_count']<=r['requested_count']<=r['estimated_capacity']<=6
        assert any(i['class_name']=='YF-12A' for i in r['instances'])
        assert Path(r['image_path']).is_file() and Path(r['label_path']).is_file() and Path(r['instance_mask_path']).is_file()
        assert hashlib.sha256(Path(r['image_path']).read_bytes()).hexdigest()==r['image_sha256']
        assert hashlib.sha256(Path(r['label_path']).read_bytes()).hexdigest()==r['label_sha256']
        mask=np.asarray(Image.open(r['instance_mask_path']))
        assert mask.shape==(r['height'],r['width'])
        assert set(np.unique(mask))==set(range(r['actual_count']+1))
        labels=Path(r['label_path']).read_text().splitlines()
        originals=[]
        for ann in r['original_annotations']:
            x,y,w,h=ann['bbox_xywh']
            box=[max(0,x),max(0,y),min(r['width'],x+w),min(r['height'],y+h)]
            if box[2]<=box[0] or box[3]<=box[1]:continue
            name='LAE/'+ann['category_name']
            if name in ('LAE/helicopter','LAE/Helicopter'):name='Helicopter'
            originals.append(yolo_hbb_line(names[name],box,r['width'],r['height']))
        assert labels[r['actual_count']:]==originals
        seen=set()
        for j,inst in enumerate(r['instances'],1):
            fid=inst['foreground_id'];assert fid in pool and fid not in heldout_ids
            assert inst['class_name']==pool[fid]['class_name']
            assert inst['target_class_id']==names[inst['class_name']]
            assert labels[j-1]==yolo_hbb_line(names[inst['class_name']],inst['bbox_xyxy'],r['width'],r['height'])
            yy,xx=np.where(mask==j)
            assert [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]==inst['bbox_xyxy']
            assert not rejection_reasons(inst['automatic_metrics'])
            if fid in train_ids:
                assert inst['class_name']=='YF-12A'
                assert inst['true_model']=='SR-71A' and inst['proxy_for']=='YF-12A'
                assert pool[fid]['serial'] not in {x['serial'] for x in heldout}
                uses[fid]+=1
            by_class[inst['class_name']]+=1;seen.add(inst['class_name']);foreground_ids.add(fid)
        mixed+=len(seen)>1;actual[r['actual_count']]+=1
    assert uses and set(uses)==train_ids
    assert len(rows)==500 and len(uses)==27
    summary=dict(images=len(rows),proxy_instances=by_class['YF-12A'],
        mixed_class_images=mixed,used_sr71_training_cases=len(uses),
        reserved_sr71_test_cases=len(heldout),training_airframes=len({r['serial'] for r in training}),
        reserved_test_airframes=len({r['serial'] for r in heldout}),
        training_case_uses=dict(uses),instances_by_class=dict(by_class),
        actual_paste_count_histogram=dict(actual),subdatasets=subsets,
        attempts=attempts,overall_image_pass_rate=round(expected/sum(attempts.values()),4),
        heldout_source_ids_used=0,distinct_foreground_ids=len(foreground_ids),
        audit_passed=True,training_ready=True,
        source_truth='SR-71A',proxy_label='YF-12A')
    json_write(root/'summary.json',summary)
    board=Image.new('RGB',(1200,340*6),'white');draw=ImageDraw.Draw(board)
    for i,r in enumerate(rows[:12]):
        im=Image.open(r['image_path']).convert('RGB')
        pen=ImageDraw.Draw(im)
        for inst in r['instances']:
            pen.rectangle(inst['bbox_xyxy'],outline='#ff3333',width=3)
            pen.text(tuple(inst['bbox_xyxy'][:2]),inst['class_name'],fill='#ff3333')
        im.thumbnail((590,305));x=(i%2)*600;y=(i//2)*340
        board.paste(im,(x,y+30));draw.text((x+5,y+5),r['id'],fill='black')
    (root/'qa').mkdir(exist_ok=True)
    board.save(root/'qa/SR71_proxy_review.jpg',quality=92)
    print(json.dumps({k:summary[k] for k in ('images','proxy_instances','mixed_class_images','used_sr71_training_cases','reserved_sr71_test_cases','overall_image_pass_rate','audit_passed')},ensure_ascii=False))
    return summary


if __name__=='__main__':audit()
