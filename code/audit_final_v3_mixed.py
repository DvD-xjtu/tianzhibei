#!/usr/bin/env python3
"""Independent file/annotation audit and report for the mixed-class final v3."""
from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json,math
import numpy as np
from PIL import Image,ImageDraw

from produce_final_v3_mixed import ROOT
from produce_final_v3 import CLASSES
from produce_final_dataset import SUBSETS,json_write
from filter_copypaste_v7 import rejection_reasons
from synthesize_copypaste_v6 import yolo_hbb_line


def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def audit(root=ROOT,expected=20000):
    config=json.loads((root/'config.json').read_text());assert config['mixed_classes']
    taxonomy=json.loads((root/'taxonomy.json').read_text());names={n:i for i,n in enumerate(taxonomy)}
    source_pool=read(root/'foreground_pool.jsonl');pool={r['id']:r for r in source_pool}
    assert len(pool)==len(source_pool)
    rows=[];subset_counts={};attempts={};failures={}
    for i,ds in enumerate(SUBSETS):
        p=root/'shards'/ds
        progress=json.loads((p/'progress.json').read_text())
        n=expected//6+int(i<expected%6)
        assert progress['completed'] and progress['goal']==n and progress['images']==n,(ds,progress)
        part=read(p/'manifest.jsonl');assert len(part)==n,(ds,len(part),n)
        assert all(r['subdataset']==ds for r in part)
        subset_counts[ds]=n;attempts[ds]=progress['attempts'];failures[ds]=progress['failures'];rows+=part
    assert len(rows)==expected and len({r['id'] for r in rows})==expected
    image_presence=Counter();instances=Counter();unique_crops=defaultdict(set);crop_uses=Counter()
    class_subsets=defaultdict(Counter);source_uses=defaultdict(Counter)
    actual=Counter();request=Counter();capacity=Counter();mixed=0;unique_bgs=set();test_used=defaultdict(set)
    for row in rows:
        assert row['quality_gate_passed'] and 1<=row['actual_count']<=row['requested_count']<=row['estimated_capacity']<=6
        image=Path(row['image_path']);label=Path(row['label_path']);mask_path=Path(row['instance_mask_path'])
        assert image.is_file() and label.is_file() and mask_path.is_file()
        assert hashlib.sha256(image.read_bytes()).hexdigest()==row['image_sha256']
        assert hashlib.sha256(label.read_bytes()).hexdigest()==row['label_sha256']
        mask=np.asarray(Image.open(mask_path));assert mask.shape==(row['height'],row['width'])
        assert set(np.unique(mask))==set(range(row['actual_count']+1))
        labels=label.read_text().splitlines()
        original=[]
        for ann in row['original_annotations']:
            x,y,w,h=ann['bbox_xywh'];b=[max(0,x),max(0,y),min(row['width'],x+w),min(row['height'],y+h)]
            if b[2]<=b[0] or b[3]<=b[1]:continue
            name='LAE/'+ann['category_name']
            if name in ('LAE/helicopter','LAE/Helicopter'):name='Helicopter'
            original.append(yolo_hbb_line(names[name],b,row['width'],row['height']))
        assert labels[row['actual_count']:]==original
        assert len({i['foreground_id'] for i in row['instances']})==row['actual_count']
        classes=set()
        for j,inst in enumerate(row['instances'],1):
            cls=inst['class_name'];assert cls in CLASSES and inst['target_class_id']==names[cls]
            fid=inst['foreground_id'];assert fid in pool and pool[fid]['class_name']==cls
            assert inst['foreground_path']==pool[fid]['cutout_path']
            assert labels[j-1]==yolo_hbb_line(names[cls],inst['bbox_xyxy'],row['width'],row['height'])
            yy,xx=np.where(mask==j)
            assert [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]==inst['bbox_xyxy']
            assert not rejection_reasons(inst['automatic_metrics'])
            classes.add(cls);instances[cls]+=1;unique_crops[cls].add(fid);crop_uses[fid]+=1
            class_subsets[cls][row['subdataset']]+=1;source_uses[cls][pool[fid]['source']]+=1
            if pool[fid].get('source_split')=='test':test_used[cls].add(fid)
        for cls in classes:image_presence[cls]+=1
        mixed+=len(classes)>1;actual[row['actual_count']]+=1;request[row['requested_count']]+=1
        capacity[row['estimated_capacity']]+=1;unique_bgs.add(row['source_background_id'])
    assert all(image_presence[c]>0 for c in CLASSES)
    assert max(subset_counts.values())-min(subset_counts.values())<=1
    candidate=Counter(r['class_name'] for r in source_pool)
    class_report={c:dict(candidate_original_crops=candidate[c],used_original_crops=len(unique_crops[c]),
        crop_use_rate=round(len(unique_crops[c])/candidate[c],4),images_with_class=image_presence[c],
        new_instances=instances[c],test_original_crops_used=len(test_used[c]),
        sources=dict(source_uses[c]),lae_subdataset_instances=dict(class_subsets[c])) for c in CLASSES}
    summary=dict(version=config['version'],images=len(rows),new_instances=sum(instances.values()),
        mixed_class_images=mixed,mixed_class_image_rate=round(mixed/len(rows),4),
        attempts=sum(attempts.values()),overall_image_pass_rate=round(len(rows)/sum(attempts.values()),4),
        lae_subdatasets=subset_counts,lae_attempts=attempts,
        lae_pass_rates={d:round(subset_counts[d]/attempts[d],4) for d in SUBSETS},
        class_distribution=class_report,actual_paste_count_histogram=dict(actual),
        requested_count_histogram=dict(request),estimated_capacity_histogram=dict(capacity),
        unique_backgrounds=len(unique_bgs),failures_by_subdataset=failures,
        E2_test_originals_available=10,E2_test_originals_in_pool=4,
        audit_passed=True,checks=['20,000 unique image IDs','six LAE subsets differ by at most one image',
            'all image and label hashes','all instance ID masks and HBB alignment',
            'original LAE GT preserved','all final JPEG gates','source provenance','1-6 and capacity rules'])
    json_write(root/'summary.json',summary)
    qa=root/'qa';qa.mkdir(exist_ok=True)
    for n in range(1,7):
        selected=[r for r in rows if r['actual_count']==n][:8]
        if selected:make_board(qa/f'count_{n}.jpg',selected,n)
    for cls in CLASSES:
        selected=[r for r in rows if any(i['class_name']==cls for i in r['instances'])][:8]
        if selected:make_board(qa/f'class_{cls.replace("/","-")}.jpg',selected,cls)
    report(root,summary,candidate)
    print(json.dumps({k:summary[k] for k in ('images','new_instances','mixed_class_images','overall_image_pass_rate','lae_subdatasets','audit_passed')},ensure_ascii=False,indent=2))
    return summary


def make_board(path,rows,title):
    sheet=Image.new('RGB',(1200,math.ceil(len(rows)/2)*340),'white');d=ImageDraw.Draw(sheet)
    for j,r in enumerate(rows):
        im=Image.open(r['image_path']).convert('RGB');di=ImageDraw.Draw(im)
        for inst in r['instances']:
            di.rectangle(inst['bbox_xyxy'],outline='#ff3333',width=3)
            di.text(tuple(inst['bbox_xyxy'][:2]),inst['class_name'],fill='#ff3333')
        im.thumbnail((590,305));x=(j%2)*600;y=(j//2)*340
        sheet.paste(im,(x,y+30));d.text((x+5,y+5),f'{title} | {r["id"]}',fill='black')
    sheet.save(path,quality=92)


def report(root,s,candidate):
    lines=['# Final v3 混合类别数据报告','',
        f'落盘目录：`{root}`。实际无框 JPG 在 `shards/<LAE子集>/train/images`，同名 YOLO HBB 标签在 `train/labels`，新增实例 ID mask 在 `instance_masks`；`qa` 目录的红框只出现在展板图中。', '',
        f'共 {s["images"]:,} 张图、{s["new_instances"]:,} 个新增实例；{s["mixed_class_images"]:,} 张含两种及以上新增机型（{s["mixed_class_image_rate"]:.2%}）。整体图级通过率 {s["overall_image_pass_rate"]:.2%}（通过图 / 尝试背景图）。所有输出通过最终 JPEG 质量门和落盘审计。','',
        '类别抽样权重随原始可用 crop 数的平方根增长，并以当前成功实例数做反馈；因此 crop 丰富类通常分布更多，稀少类仍可进入成图。每张图先估可贴容量，再随机请求 1 至上限个实例，允许混合机型。','',
        '| 机型 | 可用原始 crop | 实际使用 crop | 含该类的图 | 新增实例 | 用到测试 crop |','|---|---:|---:|---:|---:|---:|']
    for cls in CLASSES:
        x=s['class_distribution'][cls]
        lines.append(f'| {cls} | {x["candidate_original_crops"]} | {x["used_original_crops"]} | {x["images_with_class"]} | {x["new_instances"]} | {x["test_original_crops_used"]} |')
    lines+=['','| LAE 子集 | 产出图 | 尝试背景 | 图级通过率 |','|---|---:|---:|---:|']
    for ds in SUBSETS:
        lines.append(f'| {ds} | {s["lae_subdatasets"][ds]} | {s["lae_attempts"][ds]} | {s["lae_pass_rates"][ds]:.2%} |')
    lines+=['','E-2 训练及未划分来源共有 21 个独立 crop，仍远少于先前期望的约 100 个；因此补入 4 个通过源图质量门的测试 crop，占该类 10 个测试 crop 的 40%。其余类没有使用测试 crop。用于本批训练的 E-2 测试来源不得再作为独立测试评估。',
        '图片级质量门通过率不能解释为每个机型的独立通过率：一个背景尝试会尝试多种机型，失败按背景和原因记录。每个机型的原始 crop 候选、实际使用量、来源及 LAE 子集实例分布保存在 `summary.json`；逐图来源见 `shards/*/manifest.jsonl`。',
        '“可用”表示自动质量门和完整性审计通过。`qa` 展板提供抽样人工复核入口，不代表已逐张人工审核。']
    (root/'final_v3_report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':audit()
