#!/usr/bin/env python3
"""Audit the 1k F-35 supplement and write a Chinese production report."""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json,math
import numpy as np
from PIL import Image,ImageDraw

from produce_f35_v3_supplement import ROOT
from produce_final_dataset import SUBSETS,json_write
from filter_copypaste_v7 import rejection_reasons
from synthesize_copypaste_v6 import yolo_hbb_line


def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def audit(root=ROOT,expected=1000):
    pool=read(root/'foreground_pool.jsonl');by_id={r['id']:r for r in pool}
    names={n:i for i,n in enumerate(json.loads((root/'taxonomy.json').read_text()))}
    rows=[];subsets={};attempts={};failures={}
    for i,ds in enumerate(SUBSETS):
        shard=root/'shards'/ds
        p=json.loads((shard/'progress.json').read_text())
        n=expected//6+int(i<expected%6)
        assert p['completed'] and p['images']==n and p['goal']==n,(ds,p)
        part=read(shard/'manifest.jsonl');assert len(part)==n
        rows+=part;subsets[ds]=n;attempts[ds]=p['attempts'];failures[ds]=p['failures']
    assert len(rows)==expected and len({r['id'] for r in rows})==expected
    classes=Counter();images_by_class=Counter();used=defaultdict(set);actual=Counter();mixed=0;test_used=set()
    for row in rows:
        assert row['quality_gate_passed'] and 1<=row['actual_count']<=row['requested_count']<=row['estimated_capacity']<=6
        assert any(i['class_name']=='F-35' for i in row['instances'])
        ip=Path(row['image_path']);lp=Path(row['label_path']);mp=Path(row['instance_mask_path'])
        assert all(p.is_file() for p in (ip,lp,mp))
        assert hashlib.sha256(ip.read_bytes()).hexdigest()==row['image_sha256']
        assert hashlib.sha256(lp.read_bytes()).hexdigest()==row['label_sha256']
        mask=np.asarray(Image.open(mp));assert mask.shape==(row['height'],row['width'])
        assert set(np.unique(mask))==set(range(row['actual_count']+1))
        labels=lp.read_text().splitlines();assert len({i['foreground_id'] for i in row['instances']})==row['actual_count']
        originals=[]
        for ann in row['original_annotations']:
            x,y,w,h=ann['bbox_xywh'];b=[max(0,x),max(0,y),min(row['width'],x+w),min(row['height'],y+h)]
            if b[2]<=b[0] or b[3]<=b[1]:continue
            name='LAE/'+ann['category_name']
            if name in ('LAE/helicopter','LAE/Helicopter'):name='Helicopter'
            originals.append(yolo_hbb_line(names[name],b,row['width'],row['height']))
        assert labels[row['actual_count']:]==originals
        seen=set()
        for j,inst in enumerate(row['instances'],1):
            cls=inst['class_name'];fid=inst['foreground_id'];assert fid in by_id and by_id[fid]['class_name']==cls
            assert labels[j-1]==yolo_hbb_line(names[cls],inst['bbox_xyxy'],row['width'],row['height'])
            yy,xx=np.where(mask==j)
            assert [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]==inst['bbox_xyxy']
            assert not rejection_reasons(inst['automatic_metrics'])
            classes[cls]+=1;used[cls].add(fid);seen.add(cls)
            if cls=='F-35' and by_id[fid].get('source_split')=='test':test_used.add(fid)
        images_by_class.update(seen);mixed+=len(seen)>1;actual[row['actual_count']]+=1
    assert len(used['F-35'])==10 and len(test_used)==2
    summary=dict(images=expected,new_instances=sum(classes.values()),F35_instances=classes['F-35'],
        F35_images=images_by_class['F-35'],F35_original_cases_available=10,
        F35_original_cases_used=len(used['F-35']),F35_test_cases_used=len(test_used),
        mixed_class_images=mixed,subdatasets=subsets,attempts=attempts,
        overall_image_pass_rate=round(expected/sum(attempts.values()),4),
        instances_by_class=dict(classes),images_by_class=dict(images_by_class),
        unique_cases_by_class={c:len(v) for c,v in used.items()},actual_count_histogram=dict(actual),
        failures_by_subdataset=failures,audit_passed=True)
    json_write(root/'summary.json',summary)
    make_board(root/'qa/F-35_review.jpg',rows[:12])
    lines=['# Final v3：F-35 类别补充 1,000 张','',
        f'落盘：`{root}`。这是与原 final v3 兼容、单独存放的增量批次，沿用同一 `taxonomy.json`。真实训练图在 `shards/<LAE子集>/train/images`，标签在 `train/labels`，新增实例 mask 在 `instance_masks`；展板红框不写入训练 JPG。','',
        f'完成 {expected:,} 张图，每张至少含 1 架 F-35；F-35 新增实例 {classes["F-35"]:,} 个。{mixed:,} 张图还含其他机型。图级通过率 {summary["overall_image_pass_rate"]:.2%}（通过图 / 尝试背景）；逐图哈希、标签、mask 与质量门审计通过。','',
        f'F-35 原始可用 crop case 共 10 个，**实际全部用到**：8 个训练来源、2 个测试来源。测试来源原有 7 个 case，本批用了 2 个（28.57%），未超过 50%。这些测试来源不应再用于独立测试。10 个原始 case 经旋转、尺度、明暗和色度匹配后重复组合，1,000 张图不代表 1,000 个独立飞机来源。','',
        '| LAE 子集 | 成图 | 尝试 | 通过率 |','|---|---:|---:|---:|']
    for ds in SUBSETS:lines.append(f'| {ds} | {subsets[ds]} | {attempts[ds]} | {subsets[ds]/attempts[ds]:.2%} |')
    lines+=['','其他机型在本批中的实例数：'+ '、'.join(f'{c} {n}' for c,n in classes.items() if c!='F-35')+'。',
        '“可用”表示通过自动质量门及文件完整性审计；`qa/F-35_review.jpg` 为抽样视觉展板，尚非逐张人工审核。详细来源及每次贴图参数见 `foreground_pool.jsonl` 和各 shard 的 `manifest.jsonl`。']
    (root/'F-35_1000_中文报告.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:summary[k] for k in ('images','F35_instances','F35_original_cases_used','mixed_class_images','overall_image_pass_rate','audit_passed')},ensure_ascii=False,indent=2))
    return summary


def make_board(path,rows):
    path.parent.mkdir(exist_ok=True)
    sheet=Image.new('RGB',(1200,math.ceil(len(rows)/2)*340),'white');d=ImageDraw.Draw(sheet)
    for j,r in enumerate(rows):
        im=Image.open(r['image_path']).convert('RGB');di=ImageDraw.Draw(im)
        for inst in r['instances']:
            di.rectangle(inst['bbox_xyxy'],outline='#ff3333',width=3)
            di.text(tuple(inst['bbox_xyxy'][:2]),inst['class_name'],fill='#ff3333')
        im.thumbnail((590,305));x=(j%2)*600;y=(j//2)*340
        sheet.paste(im,(x,y+30));d.text((x+5,y+5),r['id'],fill='black')
    sheet.save(path,quality=92)


if __name__=='__main__':audit()
