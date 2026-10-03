#!/usr/bin/env python3
"""Complete the original foreground pool; never generate pasted/augmented images.

One decision per source image (MTARSI) or annotated object (MAR20). Held-out
objects are archived but excluded from training_pool.jsonl. Resume by decision
key, not by counting files. Rejected images are not retained in the new pool.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

BASE = Path('/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1')
ROOT = Path('/data3/tianzhibei/derived/aircraft_copypaste_final_20260928')
CLASSES = ['C-17', 'C-130', 'KC-135', 'KC-10', 'A-10', 'F-15', 'F-16',
           'F/A-18', 'F-35', 'B-1B', 'B-52H', 'E-2', 'E-3']
MAR_MAP = {1:'C-130', 2:'C-17', 4:'F-16', 6:'E-3', 7:'B-52H',
           9:'B-1B', 12:'F-15', 13:'KC-135', 15:'F/A-18', 17:'KC-10'}
GATES = dict(min_score=.50, min_dominant=.90, max_secondary=.07,
             max_mask_area=.45, min_bbox_fill=.10)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(source, path, obj=None):
    return f'{source}|{path}|{obj if obj is not None else "image"}'


def identity(row):
    return hashlib.sha256(row['key'].encode()).hexdigest()[:20]


def archive(root, row, cutout, context, mask=None):
    from PIL import Image
    import numpy as np
    with Image.open(cutout) as im:
        if im.mode != 'RGBA' or not np.asarray(im.getchannel('A')).any():
            raise ValueError(f'invalid RGBA foreground: {cutout}')
        size = im.size
    with Image.open(context) as im:
        if im.size != size:
            raise ValueError(f'context/cutout shape mismatch: {context}')
    stem = f'{row["source"]}__{identity(row)}'
    dest = root/'crop-image'/row['class_name'].replace('/', '-')/f'{stem}.png'
    ctx = root/'crop-metadata'/'source-context'/f'{stem}{Path(context).suffix.lower()}'
    for src, dst in [(Path(cutout), dest), (Path(context), ctx)]:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
    row.update(id=stem, cutout_path=str(dest), source_crop_path=str(ctx),
               crop_size=list(size), original_cutout_path=str(cutout),
               training_eligible=row['source'] in {'fixed', 'innar_train', 'mar20_train'})


def prepare(root):
    from extract_mtarsi_foregrounds import candidates
    meta = root/'crop-metadata'
    meta.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        (root/'crop-image'/cls.replace('/', '-')).mkdir(parents=True, exist_ok=True)
    mtarsi = {}
    for p in sorted((BASE/'foreground_all_15taxonomy_v6').glob('shard_*/manifest_quality_gated_v2.json')):
        for rec in json.loads(p.read_text()):
            cls = 'B-52H' if rec['class'] == 'B-52' else rec['class']
            if cls not in CLASSES:
                continue
            k = key(rec['source'], rec['source_path'])
            if k in mtarsi:
                raise RuntimeError(f'duplicate MTARSI input: {k}')
            row = dict(key=k, source=rec['source'], source_dataset='MTARSI',
                       source_image_path=rec['source_path'], class_name=cls,
                       quality_gate=rec['quality_gate'], selected_score=rec['selected_score'],
                       original_manifest=str(p), extraction_reused=True)
            if row['quality_gate']['accepted']:
                archive(root, row, rec['cutout_path'], rec['source_path'])
            mtarsi[k] = row
    inventory = {key(src, str(p)): dict(key=key(src,str(p)), source=src,
        source_dataset='MTARSI', source_image_path=str(p),
        class_name='B-52H' if cls=='B-52' else cls,
        training_eligible=src in {'fixed','innar_train'})
        for cls, items in candidates().items() if cls in CLASSES or cls == 'B-52'
        for src, p in items}
    if set(mtarsi)-set(inventory):
        raise RuntimeError('historical MTARSI contains files missing from current inventory')
    missing_mtarsi = [r for k,r in inventory.items() if k not in mtarsi]
    (meta/'mtarsi-decisions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in mtarsi.values()))
    sampled = {}
    for rec in json.loads((BASE/'foreground_mar20_500_v1/manifest_quality_gated.json').read_text()):
        if rec['class_id'] not in MAR_MAP:
            continue
        sampled[key('mar20_train', rec['source_image_path'], rec['object_index_in_label'])] = rec
    jobs, reused = missing_mtarsi.copy(), []
    for split in ['train', 'test']:
        data = Path('/data3/tianzhibei/datasets/MAR20/yolo_obb')/split
        for label in sorted((data/'labels').glob('*.txt')):
            image = data/'images'/f'{label.stem}.jpg'
            if not image.is_file():
                raise FileNotFoundError(image)
            for oi, line in enumerate(label.read_text().splitlines()):
                f = line.split()
                if not f:
                    continue
                ci = int(f[0])
                if ci not in MAR_MAP:
                    continue
                if len(f) != 9:
                    raise ValueError(f'invalid OBB: {label}:{oi+1}')
                source = f'mar20_{split}'
                k = key(source, str(image), oi)
                row = dict(key=k, source=source, source_dataset='MAR20', source_split=split,
                           source_image_path=str(image), source_label_path=str(label),
                           object_index_in_label=oi, class_name=MAR_MAP[ci],
                           source_polygon_normalized=[float(x) for x in f[1:]],
                           training_eligible=split=='train')
                if k in sampled:
                    old = sampled[k]
                    row.update(quality_gate=old['quality_gate'], selected_score=old['selected_score'],
                               crop_xyxy_in_source=old['crop_xyxy_in_source'],
                               prompt_box_xyxy_in_crop=old['prompt_box_xyxy_in_crop'],
                               extraction_reused=True)
                    if row['quality_gate']['accepted']:
                        archive(root, row, old['cutout_path'], old['source_crop_path'])
                    reused.append(row)
                else:
                    jobs.append(row)
    (meta/'mar20-reused-decisions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reused))
    (meta/'pending-inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in jobs))
    write_json(meta/'config.json', dict(pipeline_version='final-crop-image-v1',
        mtarsi_reused_pipeline='foreground_all_15taxonomy_v6 / quality_gate_v2',
        mar20_reused_pipeline='foreground_mar20_500_v1', classes=CLASSES, gates=GATES, crop_context=1.8,
        sam_checkpoint=str(BASE/'checkpoints/sam2.1_hiera_large.pt'),
        augmentation=False, visual_review=False, rejected_images_retained=False,
        mtarsi_candidates=len(inventory), mtarsi_reused=len(mtarsi), mtarsi_pending=len(missing_mtarsi),
        mar20_candidates=len(jobs)-len(missing_mtarsi)+len(reused),
        mar20_reused=len(reused), mar20_pending=len(jobs)-len(missing_mtarsi),
        warning='Held-out test/valid crops are reference only; use training_pool.jsonl, never glob crop-image for training.',
        class_aliases={'B-52':'B-52H (existing project alias, not verified subtype)',
                       'F/A-18':'F-A-18 (folder slug)'}))
    print(json.dumps(dict(mtarsi=len(inventory), mtarsi_passed_reused=sum(r['quality_gate']['accepted'] for r in mtarsi.values()),
        mtarsi_pending=len(missing_mtarsi), mar20_total=len(jobs)-len(missing_mtarsi)+len(reused),
        mar20_reused=len(reused), total_pending=len(jobs)),ensure_ascii=False),flush=True)


def worker(root, shard, shards):
    import numpy as np
    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from extract_mar20_foregrounds import crop_and_prompt
    from extract_mtarsi_foregrounds import choose_conservative_mask
    from filter_mar20_foregrounds import quality
    from filter_foreground_batch import quality as mtarsi_quality
    torch.set_num_threads(2)
    meta = root/'crop-metadata'
    output = meta/f'worker-{shard}-decisions.jsonl'
    done = {r['key'] for r in lines(output)}
    jobs = [r for i,r in enumerate(lines(meta/'pending-inputs.jsonl')) if i%shards==shard]
    predictor = SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',
        str(BASE/'checkpoints/sam2.1_hiera_large.pt'), device='cuda'))
    cached_path, original = None, None
    started = time.time()
    with output.open('a', buffering=1) as stream:
        for n,row in enumerate(jobs,1):
            if row['key'] in done:
                continue
            if cached_path != row['source_image_path']:
                with Image.open(row['source_image_path']) as im:
                    original = im.convert('RGB')
                cached_path = row['source_image_path']
            if row['source_dataset']=='MAR20':
                crop, prompt, box = crop_and_prompt(original,row['source_polygon_normalized'],1.8)
            else:
                crop = original
                prompt = [.05*crop.width,.05*crop.height,.95*crop.width,.95*crop.height]
                box = [0,0,crop.width,crop.height]
            image = np.array(crop, dtype=np.uint8, copy=True)
            t = time.perf_counter()
            with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
                predictor.set_image(image)
                masks,scores,_ = predictor.predict(box=np.asarray(prompt,dtype=np.float32),multimask_output=True)
            mask,score,selection = choose_conservative_mask(masks,scores)
            mask = mask.astype(bool)
            gate = quality(mask,score,GATES) if row['source_dataset']=='MAR20' else mtarsi_quality(mask,score,**GATES)
            row.update(quality_gate=gate,selected_score=float(score),selection_detail=selection,
                       crop_xyxy_in_source=box,prompt_box_xyxy_in_crop=prompt,
                       inference_ms=(time.perf_counter()-t)*1000,extraction_reused=False)
            if gate['accepted']:
                stem = f'{row["source"]}__{identity(row)}'
                cp = root/'crop-image'/row['class_name'].replace('/','-')/f'{stem}.png'
                xp = meta/'source-context'/f'{stem}.jpg'
                xp.parent.mkdir(parents=True,exist_ok=True)
                # Atomic accepted-only output; the alpha channel is the native SAM mask.
                temp = cp.with_suffix('.tmp')
                Image.fromarray(np.dstack((image,mask.astype(np.uint8)*255)),'RGBA').save(temp,format='PNG')
                temp.replace(cp)
                temp = xp.with_suffix('.tmp')
                crop.save(temp,format='JPEG',quality=95)
                temp.replace(xp)
                row.update(id=stem,cutout_path=str(cp),source_crop_path=str(xp),crop_size=list(crop.size))
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            if n%100==0 or n==len(jobs):
                write_json(meta/f'worker-{shard}-progress.json',dict(completed=n,total=len(jobs),elapsed_seconds=time.time()-started))
                print(f'shard {shard}: {n}/{len(jobs)} ({time.time()-started:.1f}s)',flush=True)


def report(root):
    meta = root/'crop-metadata'
    rows = lines(meta/'mtarsi-decisions.jsonl')+lines(meta/'mar20-reused-decisions.jsonl')
    for p in sorted(meta.glob('worker-*-decisions.jsonl')):
        rows += lines(p)
    if len({r['key'] for r in rows}) != len(rows):
        raise RuntimeError('duplicate source decisions')
    config = json.loads((meta/'config.json').read_text())
    expected = config['mtarsi_candidates']+config['mar20_candidates']
    approved = [r for r in rows if r['quality_gate']['accepted']]
    for row in approved:
        if not Path(row['cutout_path']).is_file() or not Path(row['source_crop_path']).is_file():
            raise FileNotFoundError(row['key'])
    paths = {r['cutout_path'] for r in approved}
    actual = {str(p) for p in (root/'crop-image').glob('*/*.png')}
    # While workers are running a PNG may precede its journal append. Full
    # reconciliation is strict only after every candidate decision is present.
    if paths-actual or (len(rows)==expected and actual-paths):
        raise RuntimeError(f'crop inventory mismatch: missing={len(paths-actual)}, orphan={len(actual-paths)}')
    for filename,subset in [('accepted_pool.jsonl',approved),
        ('training_pool.jsonl',[r for r in approved if r['training_eligible']]),
        ('heldout_reference_pool.jsonl',[r for r in approved if not r['training_eligible']])]:
        (meta/filename).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in subset))
    stats = []
    for cls in CLASSES:
        cr = [r for r in rows if r['class_name']==cls]
        entry = dict(class_name=cls)
        for ds in ['MTARSI','MAR20']:
            dr = [r for r in cr if r['source_dataset']==ds]
            entry[ds] = dict(candidates=len(dr),passed=sum(r['quality_gate']['accepted'] for r in dr),
                training_passed=sum(r['quality_gate']['accepted'] and r['training_eligible'] for r in dr))
        entry.update(passed=sum(r['quality_gate']['accepted'] for r in cr),
                     training_passed=sum(r['quality_gate']['accepted'] and r['training_eligible'] for r in cr))
        stats.append(entry)
    summary = dict(completed=len(rows)==expected,expected=expected,processed=len(rows),
                   accepted=len(approved),training_accepted=sum(r['training_eligible'] for r in approved),
                   classes=stats,by_source={s:dict(candidates=sum(r['source']==s for r in rows),
                       passed=sum(r['source']==s and r['quality_gate']['accepted'] for r in rows))
                       for s in sorted({r['source'] for r in rows})})
    if summary['completed']:
        from PIL import Image
        from collections import defaultdict
        def inspect(row):
            with Image.open(row['cutout_path']) as im:
                if im.mode != 'RGBA' or not im.getchannel('A').getbbox():
                    raise ValueError(f'invalid crop: {row["cutout_path"]}')
                size = im.size
                h = hashlib.sha256(f'{im.width}x{im.height}|RGBA|'.encode()+im.tobytes()).hexdigest()
            with Image.open(row['source_crop_path']) as im:
                if im.size != size:
                    raise ValueError(f'invalid context size: {row["id"]}')
            return row,h
        groups = defaultdict(list)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for row,h in pool.map(inspect,approved):
                groups[(row['class_name'],h)].append(row)
        hash_classes = defaultdict(set)
        for cls,h in groups:
            hash_classes[h].add(cls)
        conflicting = {h for h,classes in hash_classes.items() if len(classes)>1}
        unique_train = []
        strict_train = []
        overlaps = []
        for (cls,h),items in sorted(groups.items()):
            train = [r for r in items if r['training_eligible']]
            held = [r for r in items if not r['training_eligible']]
            if train:
                unique_train.append(train[0])
                if not held and h not in conflicting:
                    strict_train.append(train[0])
            if train and held:
                overlaps.append(dict(class_name=cls,rgba_pixel_sha256=h,ids=[r['id'] for r in items]))
        summary.update(unique_rgba_pixel_count=len(hash_classes),unique_training_crop_count=len(unique_train),
                       exact_train_heldout_content_overlap=len(overlaps),
                       exact_content_class_conflicts=len(conflicting),
                       unique_training_by_class=dict(Counter(r['class_name'] for r in unique_train)),
                       strict_unique_training_crop_count=len(strict_train),
                       strict_unique_training_by_class=dict(Counter(r['class_name'] for r in strict_train)))
        id_source = {r['id']:r['source'] for r in approved}
        summary['exact_overlap_sources'] = dict(Counter(
            ' / '.join(sorted({id_source[i] for i in g['ids']})) for g in overlaps))
        write_json(meta/'exact-content-audit.json',dict(
            duplicate_groups=[dict(class_name=cls,rgba_pixel_sha256=h,ids=[r['id'] for r in items])
                              for (cls,h),items in groups.items() if len(items)>1],
            train_heldout_overlaps=overlaps))
        (meta/'training_pool_unique.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in unique_train))
        (meta/'training_pool_no_exact_heldout_overlap.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in strict_train))
    write_json(meta/'summary.json',summary)
    doc = ['# Final 原始 crop-image 全量统计','',
        f'处理状态：{"已完成" if summary["completed"] else "处理中"}；已处理 {len(rows):,}/{expected:,} 个原始候选。',
        f'质量门通过 **{len(approved):,}** 个原始 crop 记录；其中训练来源 **{summary["training_accepted"]:,}** 个（精确内容去重另列，不含扰动副本）。',
        '', '| 类别 | MTARSI 候选 | MTARSI 通过 | Mar-20 候选 | Mar-20 通过 | 合计通过 | 训练可用 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in stats:
        a,b = s['MTARSI'],s['MAR20']
        doc.append(f'| {s["class_name"]} | {a["candidates"]} | {a["passed"]} | {b["candidates"]} | {b["passed"]} | {s["passed"]} | {s["training_passed"]} |')
    mt = [r for r in rows if r['source_dataset']=='MTARSI']
    mar = [r for r in rows if r['source_dataset']=='MAR20']
    doc.append(f'| **合计** | **{len(mt)}** | **{sum(r["quality_gate"]["accepted"] for r in mt)}** | **{len(mar)}** | **{sum(r["quality_gate"]["accepted"] for r in mar)}** | **{len(approved)}** | **{summary["training_accepted"]}** |')
    doc += ['', f'整体工程通过率：{len(approved)/len(rows):.2%}（通过 / 实际处理候选）。各类数量不含旋转、缩放或边缘扰动的派生副本。']
    if summary['completed']:
        doc += ['', f'额外校验：全部通过 PNG 均成功解码、RGBA alpha 非空，且上下文尺寸一致。逐像素完全相同的 crop 去重后，训练池为 **{summary["unique_training_crop_count"]:,}** 张（`training_pool_unique.jsonl`）；全池像素内容唯一数为 {summary["unique_rgba_pixel_count"]:,}。',
                f'两版 MTARSI 存在大量完全相同 crop，不能简单相加当作独立多样性。训练来源去重共减少 {summary["training_accepted"]-summary["unique_training_crop_count"]} 个重复记录。',
                f'训练/held-out 之间完全相同 RGBA 的重叠组：{summary["exact_train_heldout_content_overlap"]}。这是精确像素核对，不代表已排除同场景、同飞机或近似重复泄漏。',
                '重叠来源：'+ '；'.join(f'{sources}：{count} 组' for sources,count in summary['exact_overlap_sources'].items())+'。',
                f'完全相同 RGBA 却标为不同类别的内容冲突：{summary["exact_content_class_conflicts"]} 组。额外排除 held-out 精确重叠及上述冲突后，保守训练池为 **{summary["strict_unique_training_crop_count"]:,}** 张，入口为 `training_pool_no_exact_heldout_overlap.jsonl`。仅从训练清单中排除，不删除归档图片。',
                '', '| 类别 | 训练 crop 精确像素去重后 | 再排除 held-out 精确重叠 |','|---|---:|---:|']
        doc += [f'| {cls} | {summary["unique_training_by_class"].get(cls,0)} | {summary["strict_unique_training_by_class"].get(cls,0)} |' for cls in CLASSES]
    doc += ['', '| 来源 | 原始候选 | 通过 | 通过率 |','|---|---:|---:|---:|']
    for source,s in summary['by_source'].items():
        doc.append(f'| {source} | {s["candidates"]} | {s["passed"]} | {s["passed"]/s["candidates"]:.2%} |')
    doc += ['', '## 目录与使用注意','',f'- 透明 PNG：`{root}/crop-image/<类别>/`，共 13 个类别目录；F/A-18 的目录名为 `F-A-18`。',
        f'- 完整训练来源记录：`{meta}/training_pool.jsonl`。建议下一轮使用去重且排除 held-out 精确重叠的 `training_pool_no_exact_heldout_overlap.jsonl`；测试/验证仅供参考，记录于 `heldout_reference_pool.jsonl`，不得直接 glob 整个 crop-image 混入训练。',
        '- 每个 PNG 对应一个原始 MTARSI 图片或一个 Mar-20 标注实例，不含旋转/缩放副本；同一架飞机跨照片的潜在重复未做身份去重。',
        f'- 当前 MTARSI 目录比旧全量清单多 {config.get("mtarsi_pending",0)} 个未处理候选（本次为 fixed/E-2 的 10 张），已纳入补抠；因此当前候选总数与旧文档可能不同。',
        '- MTARSI 复用既有全量 SAM2.1-large 抠图及 v2 质量门判定，并对当前目录新增/遗漏候选补抠；Mar-20 复用已有抽样结果并补齐其余 train/test 实例。',
        '- Mar-20：OBB 外接框 1.8 倍上下文裁剪 → 框提示 SAM2.1-large → 几何约束选择最高分 mask → 3×3 闭运算 → 自动质量门。',
        '- 质量门：SAM 分数 ≥0.50、主连通块占比 ≥0.90、次连通块占比 ≤0.07、mask 面积占比 ≤0.45、外接框填充率 ≥0.10。MTARSI 保留既有 4 连通实现，Mar-20 使用既有 8 连通实现。',
        '- 未做人工/AI 视觉筛选；通过工程门不等于逐张语义正确保证。不通过的图片不在本批落盘，仅保留判定与原因，既有测试数据不删除。',
        '- 沿用项目已有类别映射 B-52→B-52H，不代表已逐实例核实 H 亚型。F-35、E-2、A-10 没有对应 Mar-20 补充来源，仍需关注稀少类。',
        '- 不恢复已暂停的 paste，不修改先前 12,899 张合成训练图；此次只扩大、归档原始前景池。',
        '- 下一轮 paste 应指定新的训练前景清单；旧版使用的抽样前景/扰动池不会自动变成全量，也未在此轮重生成扰动图。',
        '- 之前“使用原始 crop 数”统计的是某次合成成功输出实际用到的不同前景，不能当作数据集全部可用前景数。', '']
    Path('/home/wangduyun/aircraft-copypaste-lae/docs/final产出_crop-image_全量原始前景统计.md').write_text('\n'.join(doc))
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode',choices=['prepare','worker','run','report'])
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--shard',type=int,default=0)
    ap.add_argument('--shards',type=int,default=4)
    args = ap.parse_args()
    if args.mode=='prepare': prepare(args.root)
    elif args.mode=='worker': worker(args.root,args.shard,args.shards)
    elif args.mode=='report': report(args.root)
    else:
        prepare(args.root)
        procs = []
        for i in range(args.shards):
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
            log = (args.root/'crop-metadata'/f'worker-{i}.log').open('a')
            proc = subprocess.Popen([sys.executable,__file__,'worker','--root',str(args.root),
                '--shard',str(i),'--shards',str(args.shards)],env=env,stdout=log,stderr=subprocess.STDOUT)
            log.close()
            procs.append(proc)
        while any(p.poll() is None for p in procs):
            time.sleep(20)
            processed = sum(len(lines(p)) for p in (args.root/'crop-metadata').glob('worker-*-decisions.jsonl'))
            print(f'new foreground decisions: {processed}',flush=True)
        summary = report(args.root)
        if any(p.returncode for p in procs) or not summary['completed']:
            raise RuntimeError(f'workers incomplete: {[p.returncode for p in procs]}')


if __name__=='__main__': main()
