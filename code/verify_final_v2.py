#!/usr/bin/env python3
"""Independent on-disk audit and review sheets for final v2."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from filter_copypaste_v7 import metrics, rejection_reasons
from synthesize_copypaste_v6 import digest, box_overlap, yolo_hbb_line
from produce_final_dataset import json_write


def audit(root, expected=None):
    rows=[json.loads(l) for p in sorted((root/'shards').glob('*/manifest.jsonl')) for l in p.read_text().splitlines()]
    names=json.loads((root/'taxonomy.json').read_text()); name_ids={n:i for i,n in enumerate(names)}
    pool={r['id']:r for r in map(json.loads,(root/'foreground_pool.jsonl').read_text().splitlines())}
    counts=Counter(); requested=Counter(); capacity=Counter(); classes=Counter(); sources=Counter(); datasets=Counter()
    bases={}; scales=[]; gains=[]; unique_backgrounds=set(); shortfalls=0
    assert len({r['id'] for r in rows})==len(rows)
    if expected is not None: assert len(rows)==expected,(len(rows),expected)
    for row in rows:
        assert 1 <= row['actual_count'] <= row['requested_count'] <= row['estimated_capacity'] <= 6
        assert row['actual_count']==len(row['instances'])
        assert row['image_sha256']==digest(Path(row['image_path']))
        assert row['label_sha256']==digest(Path(row['label_path']))
        image=np.asarray(Image.open(row['image_path']).convert('RGB'))
        mask=np.asarray(Image.open(row['instance_mask_path']))
        w,h=row['width'],row['height']; assert image.shape==(h,w,3) and mask.shape==(h,w)
        assert set(np.unique(mask))==set(range(row['actual_count']+1))
        original=[]; expected_labels=[]
        for ann in row['original_annotations']:
            x,y,ww,hh=ann['bbox_xywh']; b=(max(0,x),max(0,y),min(w,x+ww),min(h,y+hh))
            if b[2]>b[0] and b[3]>b[1]:
                original.append(b); expected_labels.append(yolo_hbb_line(name_ids['LAE/'+ann['category_name']],b,w,h))
        labels=Path(row['label_path']).read_text().splitlines()
        assert labels[row['actual_count']:]==expected_labels
        assert len({i['foreground_id'] for i in row['instances']})==row['actual_count']
        patches=[]; total_area=0
        background=np.asarray(Image.open(row['background_image_path']).convert('RGB'))
        # Recompute old file-based metrics independently with per-instance masks below.
        for j,inst in enumerate(row['instances'],1):
            src=pool[inst['foreground_id']]; assert src['training_eligible']
            assert src['source'] in ('fixed','innar_train','mar20_train')
            assert inst['foreground_path']==src['cutout_path'] and inst['foreground_source_path']==src['source_path']
            assert not rejection_reasons(inst['automatic_metrics'])
            hard=mask==j; yy,xx=np.where(hard); b=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
            assert b==inst['bbox_xyxy']; bw,bh=b[2]-b[0],b[3]-b[1]
            assert min(bw,bh)>=16 and hard.sum()>=100
            assert row['scale_model']['lower']<=math.sqrt(bw*bh)<=row['scale_model']['upper']
            assert 0<inst['scale']<=2; total_area+=bw*bh
            rect=inst['patch_xyxy']; x,y,x2,y2=rect
            assert x>=8 and y>=8 and x2<=w-8 and y2<=h-8 and max(x2-x,y2-y)<=min(w,h)/3
            assert not any(box_overlap(rect,other,8) for other in original+patches)
            patches.append(rect)
            assert labels[j-1]==yolo_hbb_line(inst['target_class_id'],b,w,h)
            assert names[inst['target_class_id']]==inst['class_name']
            # Same gate definition, recomputed from disk without generator's output_metrics.
            bg_lab=cv2.cvtColor(background,cv2.COLOR_RGB2LAB)
            fg_lab=cv2.cvtColor(image,cv2.COLOR_RGB2LAB)
            bg_sat=float(np.median(np.ptp(background[hard].astype(float),axis=1)))
            fg_sat=float(np.median(np.ptp(image[hard].astype(float),axis=1)))
            values=dict(inst['automatic_metrics'],paste_lightness_delta=float(np.median(fg_lab[:,:,0][hard]))-float(np.median(bg_lab[:,:,0][hard])),
                background_saturation=bg_sat,paste_saturation=fg_sat,paste_saturation_delta=fg_sat-bg_sat)
            assert not rejection_reasons(values)
            assert all(abs(values[k]-inst['automatic_metrics'][k])<1e-5 for k in values)
            classes[inst['class_name']]+=1; sources[inst['foreground_source']]+=1
            bases.setdefault(inst['class_name'],set()).add(inst['foreground_id']); scales.append(math.sqrt(bw*bh)); gains.append(inst['blend']['lightness_gain'])
        assert total_area<=.20*w*h
        counts[row['actual_count']]+=1; requested[row['requested_count']]+=1; capacity[row['estimated_capacity']]+=1
        datasets[row['subdataset']]+=1; unique_backgrounds.add(row['source_background_id']); shortfalls+=row['count_shortfall']>0
    report=dict(images=len(rows),instances=sum(classes.values()),actual_count_histogram=dict(counts),
        requested_count_histogram=dict(requested),capacity_histogram=dict(capacity),instances_by_class=dict(classes),
        unique_crops_by_class={k:len(v) for k,v in bases.items()},foreground_sources=dict(sources),subdatasets=dict(datasets),
        unique_backgrounds=len(unique_backgrounds),images_below_requested=shortfalls,
        linear_size_px_quantiles=np.quantile(scales,[0,.1,.5,.9,1]).tolist(),
        lightness_gain_quantiles=np.quantile(gains,[0,.1,.5,.9,1]).tolist(),
        audit_passed=True,checks=['hashes','labels preserve original GT','mask to HBB alignment','train-only foreground manifest',
        '1 <= actual <= requested <= capacity <= 6','distinct crop per image','edge and mutual clearance',
        'size and area bounds','final JPEG per-instance photometric gates'],visual_review='contact sheets; user review pending')
    json_write(root/'summary.json',report)
    # Select examples by actual count; boxes show only newly pasted instances.
    for n in range(1,7):
        selected=[r for r in rows if r['actual_count']==n][:8]
        if not selected: continue
        sheet=Image.new('RGB',(1200,math.ceil(len(selected)/2)*340),'white'); draw=ImageDraw.Draw(sheet)
        for k,r in enumerate(selected):
            im=Image.open(r['image_path']).convert('RGB'); d=ImageDraw.Draw(im)
            for inst in r['instances']:
                d.rectangle(inst['bbox_xyxy'],outline='#ff3a30',width=2)
                d.text(tuple(inst['bbox_xyxy'][:2]),inst['class_name'],fill='#ff3a30')
            im.thumbnail((590,300)); x=(k%2)*600; y=(k//2)*340
            sheet.paste(im,(x,y+30)); draw.text((x+5,y+5),f"{r['id']} capacity={r['estimated_capacity']} request={r['requested_count']} actual={n}",fill='black')
        out=root/'qa'/f'count_{n}.jpg'; out.parent.mkdir(exist_ok=True); sheet.save(out,quality=92)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); ap.add_argument('--expected',type=int)
    a=ap.parse_args(); audit(a.root,a.expected)
