#!/usr/bin/env python3
"""Build a reproducible 15-class final-v3 inventory and Chinese Markdown report."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from produce_final_dataset import FINAL_ROOT, SUBSETS, source_values, json_write
from produce_extra_engine import GATES
from produce_final_v3_mixed import ROOT as V3
from produce_f35_v3_supplement import ROOT as F35
from produce_sr71_v3_supplement import ROOT as SR71
from register_v3_sr71_combined import ROOT as COMBINED
from synthesize_copypaste_v6 import atomic_bytes

OUT = Path(__file__).resolve().parents[1] / 'docs' / 'Final_v3_15类数据与crop剩余统计_2026-10-03.md'
BATCHES = [('基础v3', V3), ('F-35增量', F35), ('SR-71代理增量', SR71)]


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def test_quality(row):
    result=dict(id=row['id'],class_name=row['class_name'],source=row.get('source'),
                source_image_path=row.get('source_image_path'))
    try:
        m=source_values(row['cutout_path'],row['source_crop_path'])
        reasons=[]
        if m['source_background_like_fraction']>GATES['max_source_background_like_fraction']:
            reasons.append('source_background_leak')
        if m['secondary_mask_fraction']>GATES['max_secondary_mask_fraction']:
            reasons.append('mask_fragments')
        result.update(metrics=m,reasons=reasons,passes_v3_source_gate=not reasons)
    except (OSError,ValueError) as exc:
        result.update(metrics=None,reasons=[str(exc)],passes_v3_source_gate=False)
    return result


def main():
    cv2.setNumThreads(1)
    taxonomy=json.loads((V3/'taxonomy.json').read_text())
    classes=taxonomy[:15]
    assert classes==['C-17','C-130','KC-135','KC-10','A-10','F-15','F-16','F/A-18',
                     'F-35','B-1B','B-52H','E-2','E-3','YF-12A','Helicopter']
    assert (V3/'taxonomy.json').read_bytes()==(F35/'taxonomy.json').read_bytes()==(SR71/'taxonomy.json').read_bytes()

    pool_by_id={};pool_origin={}
    for batch,root in BATCHES:
        for r in rows(root/'foreground_pool.jsonl'):
            old=pool_by_id.get(r['id'])
            if old is not None:
                assert old['class_name']==r['class_name'] and old['cutout_path']==r['cutout_path']
            else:
                pool_by_id[r['id']]=r;pool_origin[r['id']]=batch

    image_count=Counter();images_by_class=Counter();instances_by_class=Counter()
    used=defaultdict(set);use_frequency=Counter();counts=Counter();requested=Counter();capacities=Counter()
    mixed=0;all_image_ids=set();all_sources=set();used_by_batch=defaultdict(set)
    test_source_used=defaultdict(set)
    for batch,root in BATCHES:
        batch_count=0
        for ds in SUBSETS:
            for r in rows(root/'shards'/ds/'manifest.jsonl'):
                batch_count+=1
                assert r['id'] not in all_image_ids
                all_image_ids.add(r['id'])
                assert 1<=r['actual_count']<=r['requested_count']<=r['estimated_capacity']<=6
                counts[r['actual_count']]+=1;requested[r['requested_count']]+=1;capacities[r['estimated_capacity']]+=1
                present=set()
                for inst in r['instances']:
                    fid=inst['foreground_id'];assert fid in pool_by_id
                    cls=inst['class_name'];assert cls==pool_by_id[fid]['class_name']
                    assert cls in classes and inst['target_class_id']==classes.index(cls)
                    used[cls].add(fid);use_frequency[fid]+=1;instances_by_class[cls]+=1;present.add(cls)
                    used_by_batch[batch].add(fid)
                    all_sources.add((cls,pool_by_id[fid].get('source_image_path')))
                    if pool_by_id[fid].get('source_split')=='test':test_source_used[cls].add(fid)
                for cls in present:images_by_class[cls]+=1
                mixed+=len(present)>1
        image_count[batch]=batch_count
    assert dict(image_count)=={'基础v3':20000,'F-35增量':1000,'SR-71代理增量':500}
    assert len(all_image_ids)==21500 and sum(counts.values())==21500
    assert sum(k*v for k,v in counts.items())==sum(instances_by_class.values())
    assert sum(instances_by_class.values())==48701

    reference=rows(FINAL_ROOT/'crop-metadata/heldout_reference_pool.jsonl')
    sr_reserved=rows(SR71/'sr71_heldout_cases.jsonl')
    assert len(reference)==7284 and len(sr_reserved)==9
    assert len({r['id'] for r in reference+sr_reserved})==len(reference)+len(sr_reserved)
    test_rows=reference+sr_reserved
    assert not ({r['id'] for r in sr_reserved}&set(use_frequency))
    held_original={r['id'] for r in reference}
    consumed=set(use_frequency)&held_original
    assert len(consumed)==6
    assert Counter(pool_by_id[fid]['class_name'] for fid in consumed)=={'E-2':4,'F-35':2}
    train_serials={r['serial'] for r in rows(SR71/'sr71_training_cases.jsonl')}
    reserved_serials={r['serial'] for r in sr_reserved}
    assert not (train_serials&reserved_serials)

    # This second gate is the one actually used before v3 paste, beyond the
    # original SAM foreground gate recorded in heldout_reference_pool.
    with ThreadPoolExecutor(max_workers=12) as pool:
        quality=list(pool.map(test_quality,test_rows))
    quality_by_id={r['id']:r for r in quality}
    atomic_bytes(COMBINED/'heldout_source_quality_audit.jsonl',
                 ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in quality).encode())
    assert len(quality_by_id)==len(test_rows)

    raw_total=Counter(r['class_name'] for r in test_rows)
    raw_consumed=Counter(r['class_name'] for r in test_rows if r['id'] in consumed)
    gate_pass=Counter(r['class_name'] for r in quality if r['passes_v3_source_gate'])
    gate_pass_consumed=Counter(r['class_name'] for r in quality
                               if r['id'] in consumed and r['passes_v3_source_gate'])
    test_sources=defaultdict(Counter)
    for r in test_rows:test_sources[r['class_name']][r['source']]+=1
    native_test_sources={'mar20_test','innar_test'}
    native_test=[r for r in reference if r['source'] in native_test_sources]
    native_valid=[r for r in reference if r['source']=='innar_valid']
    assert len(native_test)+len(native_valid)==len(reference)
    native_test_raw=Counter(r['class_name'] for r in native_test)
    native_test_used=Counter(r['class_name'] for r in native_test if r['id'] in consumed)
    native_test_quality=Counter(r['class_name'] for r in quality
                                if r['source'] in native_test_sources and r['passes_v3_source_gate'])
    native_valid_raw=Counter(r['class_name'] for r in native_valid)
    native_valid_quality=Counter(r['class_name'] for r in quality
                                 if r['source']=='innar_valid' and r['passes_v3_source_gate'])
    proxy_quality=sum(r['passes_v3_source_gate'] for r in quality if r['source']=='NAIP_SR71_v2')
    candidate=Counter(r['class_name'] for r in pool_by_id.values())
    for cls in classes:
        assert len(used[cls])<=candidate[cls]
        assert gate_pass_consumed[cls]<=gate_pass[cls]
    assert sum(raw_total.values())==7293
    assert sum(gate_pass_consumed.values())==6

    summary=dict(version='final-v3-combined-21500-sr71-proxy',images=21500,
        batches=dict(image_count),new_pasted_instances=sum(instances_by_class.values()),
        mixed_class_images=mixed,actual_paste_count_histogram={str(k):counts[k] for k in range(1,7)},
        requested_count_histogram={str(k):requested[k] for k in range(1,7)},
        estimated_capacity_histogram={str(k):capacities[k] for k in range(1,7)},
        frozen_foreground_pool_cases=len(pool_by_id),used_foreground_cases=sum(len(s) for s in used.values()),
        unused_foreground_cases=len(pool_by_id)-sum(len(s) for s in used.values()),
        classes={cls:dict(pool_cases=candidate[cls],used_cases=len(used[cls]),
                          unused_pool_cases=candidate[cls]-len(used[cls]),
                          images_with_class=images_by_class[cls],pasted_instances=instances_by_class[cls],
                          heldout_raw=raw_total[cls],heldout_consumed=raw_consumed[cls],
                          heldout_raw_remaining=raw_total[cls]-raw_consumed[cls],
                          heldout_pass_v3_source_gate=gate_pass[cls],
                          heldout_pass_consumed=gate_pass_consumed[cls],
                          heldout_pass_remaining=gate_pass[cls]-gate_pass_consumed[cls],
                          native_test_raw=native_test_raw[cls],
                          native_test_used=native_test_used[cls],
                          native_test_remaining=native_test_raw[cls]-native_test_used[cls],
                          native_test_pass_remaining=native_test_quality[cls]-native_test_used[cls],
                          native_valid_raw=native_valid_raw[cls],
                          native_valid_pass=native_valid_quality[cls],
                          heldout_sources=dict(test_sources[cls])) for cls in classes},
        native_test_raw_total=len(native_test),native_test_used_total=len(consumed),
        native_test_remaining_total=len(native_test)-len(consumed),
        native_test_pass_remaining_total=sum(native_test_quality.values())-len(consumed),
        native_valid_raw_total=len(native_valid),
        native_valid_pass_total=sum(native_valid_quality.values()),
        sr71_proxy_reserved_raw=len(sr_reserved),sr71_proxy_reserved_pass=proxy_quality,
        test_reference_raw_total=len(test_rows),test_reference_consumed=len(consumed),
        test_reference_remaining=len(test_rows)-len(consumed),
        test_reference_pass_v3_source_gate=sum(gate_pass.values()),
        test_reference_pass_remaining=sum(gate_pass.values())-len(consumed),
        sr71_proxy_training_airframes=sorted(train_serials),
        sr71_proxy_reserved_airframes=sorted(reserved_serials),
        original_test_source_ids_used=sorted(consumed),
        detector_test_images_created=0,
        counting_rule='Distinct foreground IDs in frozen v3 pools; training use from 21,500 manifest instances. Heldout raw is MTARSI/MAR20 test+valid references plus 9 SR-71 proxy reserves. Quality-pass uses current v3 source gate.',
        audit_scope='Exact crop IDs and original source image paths; legacy near-duplicate scenes not fully audited.')
    json_write(COMBINED/'inventory_15class.json',summary)

    lines=['# Final v3：15 类数据、crop 使用与测试来源剩余统计','',
        '统计基准：`final_v3_20261001/combined_21500_sr71_proxy` 的 **21,500 张实际训练图**，由基础 v3 20,000 张、F-35 增量 1,000 张、SR-71 代理增量 500 张组成。统计依据为三批冻结的 `foreground_pool.jsonl` 与全部逐图 `shards/*/manifest.jsonl`，同一 crop 重复贴多次只算 1 个原始 case。', '',
        '## 图像与贴图分布','',
        f'共 **{summary["images"]:,} 张**、**{summary["new_pasted_instances"]:,} 个新增贴图实例**；**{mixed:,} 张**含至少两种新增机型。下表的“1–6”是每张图**实际成功贴入**的实例数，原有 LAE 标注不计入。','',
        '| 实际新增实例数/图 | 1 | 2 | 3 | 4 | 5 | 6 | 合计 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
        '| 实际贴入的图像数 | '+' | '.join(f'{counts[i]:,}' for i in range(1,7))+f' | {sum(counts.values()):,} |',
        '| 空位容量上限对应的图像数 | '+' | '.join(f'{capacities[i]:,}' for i in range(1,7))+f' | {sum(capacities.values()):,} |',
        '| 按容量随机请求的图像数 | '+' | '.join(f'{requested[i]:,}' for i in range(1,7))+f' | {sum(requested.values()):,} |','',
        '容量上限先按背景可用空位估计；请求数在此上限内抽样；实际数还受到贴入位置和质量门限制。因此容量、请求和成功贴入三个分布不同。','',
        '## 15 类训练前景池与实际使用','',
        '“前景池”只指 final v3 三批**冻结且允许抽样**的唯一 crop case，不包括早期被质量门剔除的候选；“剩余未使用”是池内没有出现在这 21,500 张训练图中的 case。SR-71 的真实机型和 YF-12A 训练代理必须分开理解。','',
        '| 类别 | v3 前景池 | 已用于训练 | 池内剩余未使用 | 含该类的图 | 新增实例 |',
        '|---|---:|---:|---:|---:|---:|']
    for cls in classes:
        d=summary['classes'][cls]
        label='YF-12A（SR-71代理）' if cls=='YF-12A' else cls
        lines.append(f'| {label} | {d["pool_cases"]:,} | {d["used_cases"]:,} | {d["unused_pool_cases"]:,} | {d["images_with_class"]:,} | {d["pasted_instances"]:,} |')
    lines += [f'| **合计** | **{summary["frozen_foreground_pool_cases"]:,}** | **{summary["used_foreground_cases"]:,}** | **{summary["unused_foreground_cases"]:,}** | — | **{summary["new_pasted_instances"]:,}** |','',
        '## 15 类原生测试来源剩余','',
        '本表只统计 MTARSI-INNAR `test` 与 MAR20 `test` 中已整理的前景 crop，**不把 `valid` 混入 test**。“通过质量门”是按 v3 实际采用的源图背景泄漏比例 ≤0.30、次级掩膜碎片比例 ≤0.02 复算，仍需人工审查。它们是前景来源，**不是已生成的目标检测测试图**。','',
        '| 类别 | test 原数 | 已用于训练 | test 剩余 | 质量门通过且未用 |',
        '|---|---:|---:|---:|---:|']
    for cls in classes:
        d=summary['classes'][cls]
        lines.append(f'| {cls} | {d["native_test_raw"]:,} | {d["native_test_used"]:,} | {d["native_test_remaining"]:,} | {d["native_test_pass_remaining"]:,} |')
    lines += [f'| **合计** | **{len(native_test):,}** | **{len(consumed):,}** | **{len(native_test)-len(consumed):,}** | **{sum(native_test_quality.values())-len(consumed):,}** |','',
        '## 验证来源与 SR-71 代理留出','',
        '下表单列 MTARSI-INNAR `valid` 来源；它们未计入上表的 test 数量，也尚未组成独立验证图集。','',
        '| 类别 | valid 原数 | 质量门通过且未用 |',
        '|---|---:|---:|']
    for cls in classes:
        d=summary['classes'][cls]
        lines.append(f'| {cls} | {d["native_valid_raw"]:,} | {d["native_valid_pass"]:,} |')
    lines += [f'| **合计** | **{len(native_valid):,}** | **{sum(native_valid_quality.values()):,}** |','',
        f'另外，SR-71 代理按机身编号封存 **{len(sr_reserved)} 个** crop（Beale 61-7963 的 6 个、Edwards 61-7955 的 3 个），其中 **{proxy_quality} 个**通过当前源图质量门；训练使用数为 0，与训练侧 5 架机体完全隔离。它们不是原生 YF-12A 测试 crop，**真实 YF-12A 的遥感留出仍为 0**。直升机也尚无已整理的原生 test/valid crop。', '',
        'E-2 原生 test 来源已有 **4 个**进入训练，F-35 已有 **2 个**进入训练；这 6 个及其原始源图不能再用于独立测试。两类在“质量门通过且未用”的原生 test 列均为 **0**，不要把各自的 `valid` 来源误计为 test。', '',
        f'若把原生 test、原生 valid、SR-71 代理封存三种来源合并，仅作“留出前景储备”口径，则原始共 {len(test_rows):,} 个，扣除训练占用 6 个后余 {len(test_rows)-len(consumed):,} 个；其中 {sum(gate_pass.values())-len(consumed):,} 个通过当前源图质量门。此合并数不可称为正式测试集规模。', '',
        '## 测试集状态与口径','',
        '当前统一数据集只有 `train/images`、`train/labels`、`instance_masks`；**没有独立 `val/test` 目标检测图集**。上表是前景来源储备，不能直接当成测试图数量。正式测试需另取与训练前景源图、机体和 LAE 背景场景均隔离的真实遥感图，并完成全图标注；仅把留出 crop 再贴在训练背景上不能作为独立测试。', '',
        '本报告只保证冻结池中的 crop ID 和已知原始源图路径口径；旧 MTARSI/MAR20 全量跨场景近重复或同机体重复未做完整审计。另需注意，SR-71 只是 YF-12A 的外形代理，代理留出集不能替代真实 YF-12A 的比赛测试。', '',
        '## 落盘与复算','',
        f'- 统一训练视图：`{COMBINED}`；`dataset_train_only.yaml` 没有测试路径。',
        f'- 逐类机器统计：`{COMBINED}/inventory_15class.json`。',
        f'- 留出来源逐例质量审计：`{COMBINED}/heldout_source_quality_audit.jsonl`。',
        f'- 历史来源重叠审计：`{COMBINED}/split_audit.json`。',
        '- 复算脚本：`code/report_final_v3_inventory.py`。','']
    OUT.write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(OUT),images=summary['images'],instances=summary['new_pasted_instances'],
        pool=summary['frozen_foreground_pool_cases'],used=summary['used_foreground_cases'],
        unused=summary['unused_foreground_cases'],heldout_remaining=summary['test_reference_remaining'],
        heldout_quality_remaining=summary['test_reference_pass_remaining']),ensure_ascii=False))


if __name__=='__main__':main()
