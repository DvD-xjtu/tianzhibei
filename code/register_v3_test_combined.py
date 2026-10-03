#!/usr/bin/env python3
"""Audit and register final-v3 train + synthetic-test symlink view."""
from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw

from produce_final_dataset import json_write
from produce_final_v3_mixed import ROOT as V3
from produce_final_v3_test import ROOT as TEST, SHARDS, RARE
from synthesize_copypaste_v6 import atomic_bytes, digest

TRAIN=V3/'combined_21500_sr71_proxy'
ROOT=V3.parent.parent/'final_v3_20261003'/'combined_21500_train_1500_test'
LEGACY=V3/'combined_21500_train_1500_test'


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def link(source,destination):
    source=Path(source);destination=Path(destination)
    assert source.is_file(),source
    destination.parent.mkdir(parents=True,exist_ok=True)
    resolved=source.resolve()
    if destination.is_symlink():
        assert destination.resolve()==resolved
    else:
        assert not destination.exists(),destination
        os.symlink(resolved,destination)


def preview(items):
    first={}
    selected=set()
    for cls in json.loads((TEST/'taxonomy.json').read_text())[:15]:
        match=next((r for r in items if r['id'] not in selected and
                    any(i['class_name']==cls for i in r['instances'])),None)
        if match is None:
            match=next((r for r in items if any(i['class_name']==cls for i in r['instances'])),None)
        if match is not None:
            first[cls]=match;selected.add(match['id'])
    assert len(first)==15
    ordered=[first[c] for c in json.loads((TEST/'taxonomy.json').read_text())[:15]]
    random.Random(20261003).shuffle(ordered)
    side=320;canvas=Image.new('RGB',(side*4,side*4),(235,235,235));draw=ImageDraw.Draw(canvas)
    for index,row in enumerate(ordered):
        im=Image.open(row['image_path']).convert('RGB')
        scale=min((side-10)/im.width,(side-30)/im.height)
        image=im.resize((round(im.width*scale),round(im.height*scale)))
        ox=(index%4)*side+(side-image.width)//2
        oy=(index//4)*side+20+(side-30-image.height)//2
        canvas.paste(image,(ox,oy))
        for inst in row['instances']:
            x1,y1,x2,y2=inst['bbox_xyxy']
            draw.rectangle((ox+x1*scale,oy+y1*scale,ox+x2*scale,oy+y2*scale),outline=(255,35,35),width=2)
        classes=','.join(sorted({i['class_name'] for i in row['instances']}))
        draw.text(((index%4)*side+5,(index//4)*side+3),classes[:40],fill=(0,0,0))
    out=ROOT/'preview'/'test_contact_sheet_with_boxes.jpg'
    out.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(out,quality=92)


def main():
    train=rows(TRAIN/'index.jsonl')
    assert len(train)==21500
    train_inventory=json.loads((TRAIN/'inventory_15class.json').read_text())
    assert train_inventory['images']==21500
    assert train_inventory['new_pasted_instances']==48701
    release_at=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S CST')
    pool={r['id']:r for r in rows(TEST/'foreground_pool.jsonl')}
    taxonomy=json.loads((TRAIN/'taxonomy.json').read_text())
    assert (TRAIN/'taxonomy.json').read_bytes()==(TEST/'taxonomy.json').read_bytes()
    test=[]
    for shard in SHARDS:
        progress=json.loads((TEST/'shards'/shard/'progress.json').read_text())
        assert progress['completed'],shard
        test.extend(rows(TEST/'shards'/shard/'manifest.jsonl'))
    assert len(test)==1500 and len({r['id'] for r in test})==1500
    assert not {r['id'] for r in test}&{r['id'] for r in train}
    assert all(r['background_split']=='valid' and r['subdataset']=='FAIR1M' for r in test)
    assert all(Path(r['background_image_path']).name.startswith('valid_') for r in test)
    assert all(r['quality_gate_passed'] and 1<=r['actual_count']<=r['requested_count']<=r['estimated_capacity']<=6 for r in test)
    assert all(len(r['instances'])==r['actual_count'] for r in test)

    train_backgrounds=set()
    train_fg_ids=set();train_fg_sources=set();train_sr_serials=set()
    for r in train:
        manifest=Path(r['provenance_manifest'])
        # The index records the shard, not the specific manifest line. The
        # foreground audit uses the three source manifests below instead.
        assert manifest.is_file()
    for source_root in (V3,V3/'extra/F-35_1000_20261002',V3/'extra/SR71_proxy_YF12A_500_20261003'):
        for p in (source_root/'shards').glob('*/manifest.jsonl'):
            for r in rows(p):
                train_backgrounds.add(r['background_image_path'])
                for i in r['instances']:
                    train_fg_ids.add(i['foreground_id'])
                    train_fg_sources.add(i['source_image_path'])
    assert not {r['background_image_path'] for r in test}&train_backgrounds

    used=defaultdict(set);images_by_class=Counter();instances_by_class=Counter()
    hist=Counter();backgrounds=Counter();test_fg_sources=set();test_fg_ids=set();case_uses=Counter()
    for r in test:
        backgrounds[r['background_image_path']]+=1
        hist[r['actual_count']]+=1
        present=set()
        labels=Path(r['label_path']).read_text().strip().splitlines()
        assert len(labels)>=r['actual_count']
        assert Path(r['image_path']).is_file() and Path(r['instance_mask_path']).is_file()
        for i in r['instances']:
            fid=i['foreground_id'];c=i['class_name']
            assert fid in pool and pool[fid]['class_name']==c
            assert i['target_class_id']==taxonomy.index(c)
            assert fid not in train_fg_ids and i['source_image_path'] not in train_fg_sources
            used[c].add(fid);present.add(c);instances_by_class[c]+=1
            case_uses[fid]+=1
            test_fg_ids.add(fid);test_fg_sources.add(i['source_image_path'])
        for c in present:images_by_class[c]+=1
    assert set(taxonomy[:15])==set(used)
    assert sum(instances_by_class.values())==sum(k*v for k,v in hist.items())
    assert not test_fg_ids&train_fg_ids and not test_fg_sources&train_fg_sources
    assert max(case_uses.values())<=3
    case_use_hist=Counter(case_uses.values())
    single_use_classes=set(json.loads((TEST/'config.json').read_text())['single_use_classes'])
    assert all(n==1 for fid,n in case_uses.items() if pool[fid]['class_name'] in single_use_classes)
    sr_reserved={r['serial'] for r in rows(V3/'extra/SR71_proxy_YF12A_500_20261003'/'sr71_heldout_cases.jsonl')}
    sr_train={r['serial'] for r in rows(V3/'extra/SR71_proxy_YF12A_500_20261003'/'sr71_training_cases.jsonl')}
    assert not sr_reserved&sr_train
    assert all(pool[fid]['serial'] in sr_reserved for fid in used['YF-12A'])

    ROOT.mkdir(parents=True,exist_ok=True)
    if not LEGACY.exists():
        os.symlink(ROOT,LEGACY)
    assert LEGACY.resolve()==ROOT.resolve()
    source_links={'training_view':TRAIN,'test_generation':TEST,
                  'helicopter_crops':TEST/'helicopter_sources'}
    for name,source in source_links.items():
        dest=ROOT/'sources'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.is_symlink():
            assert dest.resolve()==source.resolve()
        else:
            assert not dest.exists()
            os.symlink(source.resolve(),dest)
    atomic_bytes(ROOT/'taxonomy.json',(TRAIN/'taxonomy.json').read_bytes())
    train_index=[]
    for r in train:
        copy=dict(r)
        for key,folder in (('image_path','train/images'),('label_path','train/labels'),('mask_path','instance_masks/train')):
            original=Path(r[key]);dest=ROOT/folder/original.name
            link(original,dest);copy[key]=str(dest)
        train_index.append(copy)
    test_index=[]
    for r in test:
        copy=dict(id=r['id'],subdataset=r['subdataset'],source_background_id=r['source_background_id'],
                  background_image_path=r['background_image_path'],background_split=r['background_split'],
                  actual_count=r['actual_count'],classes=sorted({i['class_name'] for i in r['instances']}),
                  provenance_manifest=str(TEST/'shards'/next(s for s in SHARDS if r['id'].startswith('v3_test__'+s+'__'))/'manifest.jsonl'))
        for key,out_key,folder in (('image_path','image_path','test/images'),('label_path','label_path','test/labels'),('instance_mask_path','mask_path','instance_masks/test')):
            original=Path(r[key]);dest=ROOT/folder/original.name
            link(original,dest);copy[out_key]=str(dest)
        test_index.append(copy)
    atomic_bytes(ROOT/'index_train.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in train_index).encode())
    atomic_bytes(ROOT/'index_test.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in test_index).encode())
    yml='path: '+str(ROOT)+'\ntrain: train/images\ntest: test/images\nnames:\n'
    yml+=''.join(f'  {i}: {json.dumps(n,ensure_ascii=False)}\n' for i,n in enumerate(taxonomy))
    atomic_bytes(ROOT/'dataset.yaml',yml.encode())
    summary=dict(train_images=len(train),test_images=len(test),taxonomy_total=len(taxonomy),
        release_date='2026-10-03',readme_updated_at=release_at,
        train_batches=train_inventory['batches'],
        train_pasted_instances=train_inventory['new_pasted_instances'],
        train_mixed_class_images=train_inventory['mixed_class_images'],
        train_paste_count_histogram=train_inventory['actual_paste_count_histogram'],
        train_foreground_pool_cases=train_inventory['frozen_foreground_pool_cases'],
        train_foreground_used_cases=train_inventory['used_foreground_cases'],
        train_foreground_unused_cases=train_inventory['unused_foreground_cases'],
        train_classes={c:train_inventory['classes'][c] for c in taxonomy[:15]},
        target_classes=taxonomy[:15],rare_classes=sorted(RARE),
        test_pasted_instances=sum(instances_by_class.values()),test_unique_backgrounds=len(backgrounds),
        test_background_repeats=sum(v-1 for v in backgrounds.values()),
        test_paste_count_histogram={str(i):hist[i] for i in range(1,7)},
        test_foreground_pool_cases=len(pool),test_foreground_used_cases=sum(len(s) for s in used.values()),
        max_case_use=max(case_uses.values()),single_use_classes=sorted(single_use_classes),
        case_use_histogram={str(i):case_use_hist[i] for i in range(1,4)},
        train_test_foreground_id_overlap=0,train_test_source_path_overlap=0,
        train_test_background_path_overlap=0,sr71_airframe_overlap=0,
        test_classes={c:dict(pool_cases=sum(r['class_name']==c for r in pool.values()),
            used_cases=len(used[c]),images=images_by_class[c],instances=instances_by_class[c],
            rare=c in RARE,proxy_true_model='SR-71A' if c=='YF-12A' else None) for c in taxonomy[:15]},
        test_source_root=str(TEST),view_type='symlink',
        legacy_compatibility_path=str(LEGACY),
        taxonomy_sha256=digest(ROOT/'taxonomy.json'),
        limitation='Synthetic pasted test only; not independent real-scene detection evaluation.')
    json_write(ROOT/'summary.json',summary)
    json_write(TEST/'summary.json',summary)
    preview(test)
    train_class_table='\n'.join(
        f"| {c}{'（稀少类）' if c in RARE else '（SR-71代理）' if c=='YF-12A' else ''} | "
        f"{train_inventory['classes'][c]['pool_cases']:,} | "
        f"{train_inventory['classes'][c]['used_cases']:,} | "
        f"{train_inventory['classes'][c]['unused_pool_cases']:,} | "
        f"{train_inventory['classes'][c]['images_with_class']:,} | "
        f"{train_inventory['classes'][c]['pasted_instances']:,} |"
        for c in taxonomy[:15])
    train_hist_table='\n'.join(
        f"| {i} | {train_inventory['actual_paste_count_histogram'][str(i)]:,} |"
        for i in range(1,7))
    train_batch_table='\n'.join(f'| {name} | {count:,} |'
                                for name,count in train_inventory['batches'].items())
    class_table='\n'.join(f"| {c}{'（稀少类）' if c in RARE else '（SR-71代理）' if c=='YF-12A' else ''} | {summary['test_classes'][c]['pool_cases']:,} | {len(used[c]):,} | {summary['test_classes'][c]['pool_cases']-len(used[c]):,} | {images_by_class[c]:,} | {instances_by_class[c]:,} |" for c in taxonomy[:15])
    hist_table='\n'.join(f'| {i} | {hist[i]:,} |' for i in range(1,7))
    readme=f'''# Final v3（2026-10-03）：训练与合成测试数据集

发布日期：**2026-10-03**；本 README 更新于 **{release_at}**。

本目录是统一数据集视图。图像文件本身没有可见框，只有 `preview/` 的检查展板画了框。下文的“实例”只统计本项目新增的前 15 类贴图，不包含背景图原有的 LAE 标注。

| 划分 | 图像 | 新增贴图实例 | 冻结前景池 crop | 实际使用 crop | 池内未用 crop | 背景来源 |
|---|---:|---:|---:|---:|---:|---|
| 训练 | {len(train):,} | {train_inventory['new_pasted_instances']:,} | {train_inventory['frozen_foreground_pool_cases']:,} | {train_inventory['used_foreground_cases']:,} | {train_inventory['unused_foreground_cases']:,} | LAE 六子集训练图 |
| 合成测试 | {len(test):,} | {sum(instances_by_class.values()):,} | {len(pool):,} | {sum(len(s) for s in used.values()):,} | {len(pool)-sum(len(s) for s in used.values()):,} | FAIR1M `valid_` 图 |

## 训练集

训练图由三批组成，统一注册到 `train/`；三批的冻结 crop 共同形成上表的训练前景池：

| 批次 | 图像数 |
|---|---:|
{train_batch_table}

训练集包含 **{train_inventory['new_pasted_instances']:,} 个新增实例**，其中 **{train_inventory['mixed_class_images']:,} 张**含至少两种新增机型。每张图按可用空位估计容量、请求 1–6 个 crop，再经过尺度、位置、明暗与质量门；下表为实际贴入数，不计原有 LAE 标注。

| 实际新增数/图 | 图像数 |
|---:|---:|
{train_hist_table}

| 类别 | 训练前景池 crop | 实际用到 crop | 池内未用 crop | 含该类的训练图 | 新增实例 |
|---|---:|---:|---:|---:|---:|
{train_class_table}

训练侧 YF-12A 标签使用 **27 个 SR-71A 外形代理 crop**；F-35、E-2 均为稀少来源，其中已有 2 个 F-35、4 个 E-2 原生 `test` 来源 crop 曾进入训练。这 6 个 crop 及其源图没有进入下方合成测试集。

## 路径与结构

绝对路径：`{ROOT}`。

```text
{ROOT.name}/
├── README.md                 本文件，口径、来源、风险与目录说明
├── dataset.yaml              YOLO 数据入口；train/test 路径和完整 138 类名称
├── taxonomy.json             类别 ID 顺序；前 15 项是本项目目标机型
├── summary.json              可机器读取的统计与隔离审计结果
├── index_train.jsonl         每张训练图的文件路径、批次与来源清单
├── index_test.jsonl          每张测试图的文件路径、背景和类别清单
├── train/
│   ├── images/               21,500 张训练图的符号链接；原图无框
│   └── labels/               对应 YOLO HBB 标注，含新增及原有 LAE 实例
├── test/
│   ├── images/               1,500 张合成测试图的符号链接；原图无框
│   └── labels/               对应 YOLO HBB 标注，含新增及原有 LAE 实例
├── instance_masks/
│   ├── train/                训练图逐实例新增前景掩膜，PNG
│   └── test/                 测试图逐实例新增前景掩膜，PNG
├── preview/
│   └── test_contact_sheet_with_boxes.jpg  仅供人工检查的带框展板
└── sources/
    ├── training_view/        原始训练批次索引与统计的兼容链接
    ├── test_generation/      测试集冻结前景池、逐图 manifest 与参数的链接
    └── helicopter_crops/     新提取直升机透明 crop 与检查图的链接
```

`train/images`、`test/images` 和掩膜文件使用符号链接复用原始生成批次；`sources/` 提供对应批次的入口。移动数据时应保留链接目标，或重新运行注册链接脚本。`index_train.jsonl`、`index_test.jsonl` 可逐图查来源与落盘文件。

直升机透明 crop 在调整测试 crop 使用次数规则前已提取。规则调整前的 2,000 张试产图不符合本版限制，**不在本目录的 train/test 中**，不能用于评估。

## 合成测试集

测试集有 **{len(test):,} 张**图、**{sum(instances_by_class.values()):,} 个新增实例**，从 {len(pool):,} 个冻结测试前景 crop 中用到 {sum(len(s) for s in used.values()):,} 个；{len(backgrounds):,} 个背景均只使用一次。与训练集使用同一套粘贴和标签生成流程，测试前景来源另行隔离。

- 测试背景：FAIR1M 处理索引中的 `valid_` 遥感切片，实际用到 **{len(backgrounds):,} 个不同背景**；训练 v3 使用 `train_` 切片，背景文件路径交集为 0。
- 常规机型前景：MTARSI-INNAR/MAR20 原生 `test` 来源，经过现有源图质量门；不同于 v3 训练使用的 crop ID 和源图路径。
- **F-35、E-2 标记为稀少类**：原生 `test` 没有通过当前质量门且未被训练使用的 crop，分别改用 `valid` 中 1、2 个合格来源。为降低重复样本偏差，合成时下调其采样权重；实际图像数见下表。
- YF-12A：使用封存的 **SR-71A 外形代理**，不是 YF-12A 真机；8 个质量合格 crop 来自 2 架未进入训练的机体，因而无需改动已有训练集。
- Helicopter：从 [SIMD 官方验证集](https://github.com/ihians/simd)及 DOTA v2 未用于现有训练合成的验证场景新提取，经过自动质量门和人工展板筛选；使用 {summary['test_classes']['Helicopter']['pool_cases']} 个透明 crop。
- 每张图先估计可放置空位，随机请求 1–6 个并经过大小、碰撞、明暗与粘贴质量门；实际贴入数量见下表。原有 LAE 标注保留，标签类别 ID 与训练集一致。
- **每个原始测试 crop 全局最多使用 3 次**；前景池不少于 300 个 crop 的类别，每个 crop 最多使用 1 次。各分片的前景池按 crop ID 哈希切开，避免跨分片超限。实际有 {case_use_hist[1]:,} 个 crop 用 1 次、{case_use_hist[2]:,} 个用 2 次、{case_use_hist[3]:,} 个用 3 次。

| 类别 | 测试前景池 crop | 实际用到 crop | 池内未用 crop | 含该类的测试图 | 新增实例 |
|---|---:|---:|---:|---:|---:|
{class_table}

| 实际新增数/图 | 图像数 |
|---:|---:|
{hist_table}

## 使用注意

这批数据是**合成测试集**，可检查 copy-paste 生成数据上的表现，不能替代全新真实遥感场景上完成完整标注的独立比赛测试。F-35、E-2 和 YF-12A 代理的源图很少，同一 crop 会在多个测试图中出现；这些类别的逐图成绩不等同于独立样本数量。YF-12A 标签对应的真实机型是 SR-71A，应单独报告代理结果。

`dataset.yaml` 保留 v3 原有的 **138 类** taxonomy：前 15 类是本项目目标机型，其余 `LAE/*` 是背景原有标注。只统计 15 类时应按类别 ID 0–14 过滤，不能直接把其余 123 类重编号后丢入同一个标签文件。当前没有独立 `val/` 图集。

已检查训练/测试前景 crop ID、前景源图路径、背景图路径均无交集；SR-71 代理按机体编号隔离。旧来源跨数据集的近重复场景尚未完成全量视觉审计，且 DOTA 验证切片可能与其未被本项目使用的原始训练切片共享大图场景。
'''
    atomic_bytes(ROOT/'README.md',readme.encode())
    test_readme=f'''# Final v3（2026-10-03）合成测试批次源目录

本 README 更新于 **{release_at}**。本目录生成 {len(test):,} 张合成测试图，已注册至 `{ROOT}`。训练与测试两侧的完整类别分布、文件夹结构、使用限制和合成说明见 `{ROOT/'README.md'}`。

```text
{TEST.name}/
├── README.md                 本文件
├── config.json               合成参数、稀少类权重与来源隔离口径
├── taxonomy.json             与 final v3 训练完全相同的 138 类 ID 表
├── foreground_pool.jsonl     冻结后的测试前景 crop 清单
├── foreground_summary.json   测试前景来源与各类数量
├── foreground_rejections.json 质量门或隔离检查剔除记录
├── summary.json              最终结果与隔离审计
├── helicopter_sources/       符号链接；SIMD/DOTA 直升机透明 crop 与检查展板
├── code_snapshot/            本次合成使用的脚本快照
└── shards/                   6 个 FAIR1M valid 背景分片
    └── FAIR1M_valid_*/
        ├── backgrounds.json  合格背景及大小先验
        ├── manifest.jsonl    逐图来源、贴图位置和质量指标
        ├── progress.json     完成数与失败原因
        ├── test/images/      原始合成图，无可见框
        ├── test/labels/      YOLO HBB 标签
        ├── instance_masks/   新增前景的逐实例掩膜
        └── preview_labels/   仅新增实例的标签侧车
```
'''
    atomic_bytes(TEST/'README.md',test_readme.encode())
    print(json.dumps(dict(path=str(ROOT),train=len(train),test=len(test),
        foreground_cases=sum(len(s) for s in used.values()),unique_backgrounds=len(backgrounds),
        instances=sum(instances_by_class.values())),ensure_ascii=False))


if __name__=='__main__':main()
