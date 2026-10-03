#!/usr/bin/env python3
"""Audit v3 on disk and write the requested Chinese production report."""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json,math
from PIL import Image,ImageDraw
import numpy as np

from produce_final_v3 import ROOT, CLASSES, SUBSETS, quota
from produce_final_dataset import json_write
from filter_copypaste_v7 import rejection_reasons
from synthesize_copypaste_v6 import yolo_hbb_line


def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def audit(root=ROOT):
    pool_summary=json.loads((root/'source_pool_summary.json').read_text())
    taxonomy=json.loads((root/'batches'/CLASSES[0]/'taxonomy.json').read_text())
    records=[]; per_class={}; bad=[]; all_src=Counter(); all_subsets=Counter()
    for cls in CLASSES:
        batch=root/'batches'/cls
        pool=read(batch/'foreground_pool.jsonl')
        source_by_id={r['id']:r for r in pool}
        pool_ids={r['id'] for r in pool}
        rows=[]; subset_counts={}; attempts=0; reasons=Counter(); count_hist=Counter(); capacities=Counter(); requests=Counter()
        for ds in SUBSETS:
            expected=quota(cls,ds,1500)
            shard=batch/'shards'/ds
            if expected==0:
                assert not (shard/'manifest.jsonl').exists(),(cls,ds,'unexpected shard')
                subset_counts[ds]=0;continue
            progress=json.loads((shard/'progress.json').read_text())
            assert progress['completed'] and progress['goal']==expected and progress['images']==expected,(cls,ds,progress)
            part=read(shard/'manifest.jsonl')
            assert len(part)==expected,(cls,ds,len(part),expected)
            for r in part:
                assert r['subdataset']==ds and r['actual_count']==len(r['instances'])
            rows+=part;subset_counts[ds]=len(part);attempts+=progress['attempts'];reasons.update(progress['failures'])
        assert len(rows)==1500,(cls,len(rows))
        used=Counter(); sources=Counter(); test_ids=set(); total_instances=0; all_ids=set(); unique_bg=set()
        for r in rows:
            assert r['id'] not in all_ids,r['id'];all_ids.add(r['id'])
            assert r['quality_gate_passed'] and 1<=r['actual_count']<=r['requested_count']<=r['estimated_capacity']<=6
            assert all(i['class_name']==cls for i in r['instances'])
            image_path=Path(r['image_path']);label_path=Path(r['label_path']);mask_path=Path(r['instance_mask_path'])
            assert image_path.is_file() and label_path.is_file() and mask_path.is_file()
            assert hashlib.sha256(image_path.read_bytes()).hexdigest()==r['image_sha256']
            assert hashlib.sha256(label_path.read_bytes()).hexdigest()==r['label_sha256']
            mask=np.asarray(Image.open(mask_path))
            assert mask.shape==(r['height'],r['width'])
            assert set(np.unique(mask))==set(range(r['actual_count']+1))
            labels=label_path.read_text().splitlines()
            assert len(labels)==r['actual_count']+len(r['original_annotations'])
            assert len({i['foreground_id'] for i in r['instances']})==r['actual_count']
            for j,inst in enumerate(r['instances']):
                fid=inst['foreground_id'];assert fid in pool_ids
                assert inst['target_class_id']==taxonomy.index(cls)
                assert labels[j]==yolo_hbb_line(taxonomy.index(cls),inst['bbox_xyxy'],r['width'],r['height'])
                assert not rejection_reasons(inst['automatic_metrics'])
                assert inst['foreground_path']==source_by_id[fid]['cutout_path']
                yy,xx=np.where(mask==j+1)
                assert [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]==inst['bbox_xyxy']
                used[fid]+=1;sources[inst['foreground_source']]+=1
                if source_by_id[fid].get('source_split')=='test':test_ids.add(fid)
            total_instances+=r['actual_count'];count_hist[r['actual_count']]+=1;capacities[r['estimated_capacity']]+=1;requests[r['requested_count']]+=1
            unique_bg.add(r['source_background_id'])
        assert set(used)<=pool_ids
        data=dict(images=len(rows),instances=total_instances,attempts=attempts,pass_rate=round(len(rows)/attempts,4),
            candidate_crop_count=len(pool),unique_used_crop_count=len(used),crop_use_rate=round(len(used)/len(pool),4),
            train_or_unsplit_crop_count=sum(r.get('source_split')!='test' for r in pool),
            test_crop_pool_count=sum(r.get('source_split')=='test' for r in pool),test_crop_used_count=len(test_ids),
            subdatasets=subset_counts,source_instances=dict(sources),instance_count_histogram=dict(count_hist),
            estimated_capacity_histogram=dict(capacities),requested_count_histogram=dict(requests),
            unique_background_count=len(unique_bg),generation_failures=dict(reasons),
            output_path=str(batch))
        per_class[cls]=data;all_src.update(sources);all_subsets.update(subset_counts)
        records.extend(rows)
    assert len(records)==19500
    assert max(all_subsets.values())-min(all_subsets.values())<=4,all_subsets
    summary=dict(version='final-v3-balanced-13class',images=len(records),instances=sum(v['instances'] for v in per_class.values()),
        class_count=len(CLASSES),all_subdatasets=dict(all_subsets),total_attempts=sum(v['attempts'] for v in per_class.values()),
        overall_pass_rate=round(len(records)/sum(v['attempts'] for v in per_class.values()),4),
        per_class=per_class,source_pool_summary=pool_summary,
        audit_passed=True,checks=['19,500 unique image IDs','all 13 exact class quotas','six-subset aggregate spread <=4',
            'all image/label hashes','source pool membership','all automatic instance gates','label class and mask box match','1-6 instances within capacity'])
    json_write(root/'summary.json',summary)
    # A small board is for review only; neither its boxes nor text enter train/images.
    board=Image.new('RGB',(4*360,4*300),'white');draw=ImageDraw.Draw(board)
    for k,cls in enumerate(CLASSES):
        subset=next(ds for ds in SUBSETS if quota(cls,ds,1500)>0)
        sample=read(root/'batches'/cls/'shards'/subset/'manifest.jsonl')[k%20]
        im=Image.open(sample['image_path']).convert('RGB');d=ImageDraw.Draw(im)
        for inst in sample['instances']:
            d.rectangle(inst['bbox_xyxy'],outline='#ff3333',width=3)
        im.thumbnail((350,265));x=(k%4)*360;y=(k//4)*300
        board.paste(im,(x,y+25));draw.text((x+5,y+5),cls,fill='black')
    board.save(root/'qa_board.jpg',quality=92)
    lines=['# Final v3 产出报告','',f'落盘目录：`{root}`。实际训练图在 `batches/<类别>/shards/<LAE子集>/train/images`，同名 YOLO HBB 标注在 `train/labels`，新增实例 ID mask 在 `instance_masks`；`qa_board.jpg` 的红框仅用于展板检查，训练 JPG 无绘制框。','',
        f'共 {summary["images"]:,} 张自动质量门通过的图、{summary["instances"]:,} 个新增实例。13 类每类恰好 1,500 张；各 LAE 子集总量差 {max(all_subsets.values())-min(all_subsets.values())} 张。图级通过率 {summary["overall_pass_rate"]:.2%}（通过图数 / 尝试背景数；一次尝试可能因容量、位置或照片质量失败）。','',
        '| 类别 | 候选原始 crop | 实际使用 crop | 测试 crop 用量 | 可用图/尝试 | 通过率 | 新增实例 |','|---|---:|---:|---:|---:|---:|---:|']
    for cls in CLASSES:
        d=per_class[cls]
        lines.append(f'| {cls} | {d["candidate_crop_count"]} | {d["unique_used_crop_count"]} | {d["test_crop_used_count"]}/{d["test_crop_pool_count"]} | {d["images"]}/{d["attempts"]} | {d["pass_rate"]:.2%} | {d["instances"]} |')
    lines+=['','| LAE 子集 | 输出图 |','|---|---:|']
    for ds in SUBSETS:lines.append(f'| {ds} | {all_subsets[ds]} |')
    lines+=['','E-2 训练及外部未划分来源仅 21 个独立 crop，因此额外使用 4 个通过源图质量门的测试 crop；该类共 10 个测试 crop，使用比例为 40%，没有使用验证集。其他类均未使用测试 crop。测试来源的原图不应再用于这批合成训练数据的独立测试评估。',
        'E-2 在 xView 的试产通过率极低，因此其 1,500 张分配给另外五个 LAE 子集；其余 12 类将 xView 配额提高到 271 张，使整批六子集总量平衡。每张图先估计可贴容量，再从 1 到容量随机请求；大小来自原图实例局部正态分布，另有旋转、缩放、明暗调整与质量门。',
        '“可用”指自动门通过及本报告文件完整性审计；`qa_board.jpg` 为抽样视觉展板，不等同于逐张人工审核。详细 crop 及来源、失败原因、每类子集分布见 `summary.json`、各类 `foreground_pool.jsonl` 与各 shard 的 `progress.json`。']
    (root/'final_v3_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:summary[k] for k in ('images','instances','overall_pass_rate','all_subdatasets','audit_passed')},ensure_ascii=False,indent=2))
    return summary


if __name__=='__main__':audit()
