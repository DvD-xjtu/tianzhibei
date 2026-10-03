#!/usr/bin/env python3
"""Audit final quotas, publish 200 examples/class, and write a Chinese report."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import random
import shutil
import time
import urllib.request

import cv2
import numpy as np
from PIL import Image
from produce_final_dataset import FINAL_ROOT, CLASSES_13, SUBSETS, json_write, GATES, class_quotas
from filter_copypaste_v7 import rejection_reasons
from synthesize_copypaste_v6 import digest, atomic_bytes

CONFIG = Path('/data2/tianzhibei/dataset-viewer/datasets.json')
DOC = Path('/home/wangduyun/aircraft-copypaste-lae/docs/final产出_13类各1000张_正式数据集.md')

def symlink(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve(): return
    if target.exists() or target.is_symlink(): raise RuntimeError(f'refusing to replace {target}')
    target.symlink_to(source)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=FINAL_ROOT)
    ap.add_argument('--skip-reload', action='store_true')
    args = ap.parse_args()
    root = args.root.resolve()
    names = json.loads((root/'taxonomy.json').read_text())
    rows, progress = [], []
    for ds in SUBSETS:
        for cls in CLASSES_13:
            if class_quotas(cls)[ds] == 0: continue
            folder = root/'shards'/ds/cls.replace('/', '-')
            p = json.loads((folder/'progress.json').read_text())
            if not p['completed']: raise RuntimeError(f'incomplete quota: {ds}/{cls}')
            part = [json.loads(s) for s in (folder/'manifest.jsonl').read_text().splitlines()]
            assert len(part) == p['quota_by_class'][cls] == p['passed_by_class'][cls] == class_quotas(cls)[ds]
            rows.extend(part)
            progress.append(p)
    byclass = defaultdict(list)
    seen_ids, hashes = set(), set()
    source_counts, bg_counts = Counter(), Counter()
    for r in rows:
        cls = r['class_name']
        assert r['id'] not in seen_ids and r['image_sha256'] not in hashes
        seen_ids.add(r['id']); hashes.add(r['image_sha256'])
        image, label, mask = (Path(r[k]) for k in ('image_path', 'label_path', 'instance_mask_path'))
        assert digest(image) == r['image_sha256'] and digest(label) == r['label_sha256']
        with Image.open(image) as im:
            assert im.size == (r['width'], r['height'])
            im.verify()
        m = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        assert m is not None and m.shape == (r['height'], r['width'])
        x, y, w, h = cv2.boundingRect(cv2.findNonZero((m > 0).astype('uint8')))
        assert [x, y, x+w, y+h] == r['bbox_xyxy']
        lines = label.read_text().splitlines()
        assert int(lines[0].split()[0]) == CLASSES_13.index(cls)
        assert len(lines) == 1 + sum(1 for a in r['original_annotations']
            if min(r['width'], a['bbox_xywh'][0]+a['bbox_xywh'][2]) > max(0,a['bbox_xywh'][0])
            and min(r['height'],a['bbox_xywh'][1]+a['bbox_xywh'][3]) > max(0,a['bbox_xywh'][1]))
        for line in lines:
            tokens = line.split()
            assert len(tokens) == 5 and 0 <= int(tokens[0]) < len(names)
            coords = np.array([float(t) for t in tokens[1:]])
            assert np.isfinite(coords).all() and ((coords >= 0) & (coords <= 1)).all()
            assert coords[2] > 0 and coords[3] > 0
        assert r['quality_gate_passed'] and not rejection_reasons(r['automatic_metrics'])
        assert r['background_split'] == 'train'
        if r['subdataset'] == 'FAIR1M': assert Path(r['background_image_path']).name.startswith('train_')
        byclass[cls].append(r)
        source_counts[r['foreground_source']] += 1
        bg_counts[r['subdataset']] += 1
        symlink(image, root/'train/images'/image.name)
        symlink(label, root/'train/labels'/label.name)
    assert len(rows) == 13000 and all(len(byclass[c]) == 1000 for c in CLASSES_13)
    atomic_bytes(root/'manifest.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows).encode())
    yaml = f'path: {root}\ntrain: train/images\nval: null\n# Set val to a separate untouched evaluation dataset before training with validation.\nnames:\n'
    yaml += ''.join(f'  {i}: {json.dumps(n, ensure_ascii=False)}\n' for i, n in enumerate(names))
    atomic_bytes(root/'data.yaml', yaml.encode())
    attempts, checked, reasons, generation = Counter(), Counter(), Counter(), Counter()
    for p in progress:
        attempts.update(p['attempts_by_class']); checked.update(p['composites_checked_by_class'])
        reasons.update(p['rejection_reasons']); generation.update(p['generation_failures'])
    class_stats = {}
    for cls, part in byclass.items():
        class_stats[cls] = {'passed': len(part), 'attempts': attempts[cls],
            'composites_checked': checked[cls], 'engineering_pass_rate': len(part)/checked[cls],
            'end_to_end_yield': len(part)/attempts[cls],
            'foreground_sources': dict(Counter(r['foreground_source'] for r in part)),
            'unique_base_foregrounds': len({(r['foreground_source'], r['foreground_base_id']) for r in part}),
            'unique_variants': len({(r['foreground_source'], r['foreground_id']) for r in part}),
            'unique_backgrounds': len({r['source_background_id'] for r in part}),
            'subdatasets': {ds: sum(r['subdataset']==ds for r in part) for ds in SUBSETS}}
    size_bytes = sum(Path(r['image_path']).stat().st_size+Path(r['label_path']).stat().st_size+
        Path(r['instance_mask_path']).stat().st_size for r in rows)
    summary = {'complete': True, 'passed': len(rows), 'classes': class_stats,
        'composites_checked': sum(checked.values()), 'engineering_pass_rate': len(rows)/sum(checked.values()),
        'generation_attempts': sum(attempts.values()), 'end_to_end_yield': len(rows)/sum(attempts.values()),
        'subdatasets': dict(bg_counts), 'foreground_sources': dict(source_counts),
        'rejection_reasons_nonexclusive': dict(reasons), 'generation_failures': dict(generation),
        'disk_bytes_images_labels_masks': size_bytes, 'rejected_images_saved': 0,
        'elapsed_wall_seconds': round(time.time()-json.loads((root/'production_config.json').read_text()).get('started_unix', (root/'production_config.json').stat().st_mtime)),
        'unique_image_hashes': len(hashes), 'quality_gate': GATES,
        'structural_audit': {'verified': len(rows), 'errors': 0}, 'visual_filtering_performed': False,
        'original_LAE_labels_included': True, 'label_classes_total': len(names),
        'dashboard_examples_per_class': 200, 'dashboard_examples_total': 2600}
    json_write(root/'summary.json', summary)

    # Round robin over both foreground sources and six backgrounds for review.
    view = root/'dashboard'
    splits = []
    for ci, cls in enumerate(CLASSES_13):
        buckets = defaultdict(list)
        for r in byclass[cls]: buckets[(r['subdataset'], r['foreground_source'])].append(r)
        rng = random.Random(20260928+ci)
        for group in buckets.values(): rng.shuffle(group)
        selected = []
        while len(selected) < 200:
            for key in sorted(buckets):
                if buckets[key]: selected.append(buckets[key].pop())
                if len(selected) == 200: break
        slug = cls.replace('/', '-')
        for r in selected:
            src = Path(r['image_path'])
            symlink(src, view/slug/'images'/src.name)
            # Only mark the inserted fine-grained aircraft on the dashboard.
            atomic_bytes(view/slug/'labels'/f'{src.stem}.txt',
                (Path(r['label_path']).read_text().splitlines()[0]+'\n').encode())
        splits.append({'name': f'{cls} · 200例', 'images': f'{slug}/images',
            'labels': f'{slug}/labels', 'label_prefix': ''})
    cfg = json.loads(CONFIG.read_text())
    backup = CONFIG.with_name(f'datasets.json.backup-final-{datetime.now():%Y%m%d-%H%M%S}')
    shutil.copy2(CONFIG, backup)
    for d in cfg['datasets']:
        if 'copy-paste-整合' in d.get('group', ''): d['group'] = 'final产出'
    cfg['datasets'] = [d for d in cfg['datasets'] if d['id'] not in {
        'copypaste-final-13class-20260928', 'copypaste-integrated-mtarsi-10samples',
        'copypaste-integrated-mar20-10samples'}]
    cfg['datasets'].append({'id': 'copypaste-final-13class-20260928', 'name': '2026-09-28 · 13类正式产出（每类200例）',
        'group': 'final产出', 'root': str(view), 'format': 'yolo', 'names': CLASSES_13,
        'training': False, 'split_filter': True, 'splits': splits,
        'note': '正式数据集13,000张，每类1,000张；本展板每类抽样200张，只标记新增飞机。工程质量门通过，无人工/AI视觉筛选。'})
    json_write(CONFIG, cfg)

    table = ['| 类别 | 通过图 | 已合成并过滤 | 通过率 | 端到端尝试 | 实际使用原始前景数 |',
             '|---|---:|---:|---:|---:|---:|']
    for cls in CLASSES_13:
        s = class_stats[cls]
        table.append(f'| {cls} | {s["passed"]} | {s["composites_checked"]} | {s["engineering_pass_rate"]:.2%} | {s["attempts"]} | {s["unique_base_foregrounds"]} |')
    dist = ['| 类别 | '+' | '.join(SUBSETS)+' |', '|---|'+'---:|'*6]
    for cls in CLASSES_13:
        dist.append('| '+cls+' | '+' | '.join(str(class_stats[cls]['subdatasets'][ds]) for ds in SUBSETS)+' |')
    doc = f'''# final产出：13类各1,000张正式数据集

完成时间：{datetime.now():%Y-%m-%d %H:%M}。本批全部由程序生成、过滤，未做人工或多模态视觉筛选。

## 结果与位置

- 总计 **13,000张**，13类各 **1,000张**，均通过当前 `engineering-v7-1` 质量门。
- 正式目录：`{root}`。与原有 `lae1m_mtarsi_copypaste_v1` 下的测试数据分开存放。
- `train/images`、`train/labels` 为统一训练入口；真实文件在 `shards/子集/类别/`，统一入口与展板使用软链接，不重复占用图片空间。
- `data.yaml`：YOLO水平框配置；`manifest.jsonl`：来源、前景扰动、位置、质量指标和原标注；`summary.json`：完整统计；`instance_masks` 位于各分片。
- 图片、标签与mask合计约 **{size_bytes/1024**3:.2f} GiB**。未通过图片保存 **0张**；只保留拒绝原因和数量统计。
- 从本批启动到全量审计完成约 **{summary['elapsed_wall_seconds']/60:.1f}分钟**。
- 全量检查13,000张：数量、唯一性、校验和、图片解码、mask/框一致性、标签范围与类别编号、质量门指标、训练来源，错误0项。

## 每类产出与过滤

{'\n'.join(table)}

已完成配额批次中，合成图片的通过率：**{summary['engineering_pass_rate']:.2%}**（13,000 / {sum(checked.values()):,}）；包含前景预检查、尺度与位置失败的产出率：**{summary['end_to_end_yield']:.2%}**（13,000 / {sum(attempts.values()):,}）。统计不含提前终止的E-2/xView可行性探测；该探测通过0张，未保存拒绝图。源mask与位置明显不合格时直接在内存中提前结束该次尝试；各项阈值与已审阅版本相同，未为补数量降低阈值。

## 六个LAE子集分布

{'\n'.join(dist)}

12类在六库各166或167张；E-2在xView连续探测未获得通过图，故分配给其他五库各200张。每类合计仍为1,000张，全批六库仍接近均衡。背景仅使用训练部分中原标注有飞机的遥感场景，FAIR1M排除了valid文件。

## 前景与扰动

MTARSI沿用v9方法，Mar-20沿用v1方法。KC-135、KC-10来自Mar-20；A-10、F-35、E-2来自MTARSI；共同类别混合两个来源，实际比例见`summary.json`。

复用已有质量门通过的增强前景：旋转0–360°、缩放0.8–1.2、轻量亮度/对比度/噪声扰动。粘贴时再加入±15°转向，保留预乘alpha缩放、Lab颜色匹配、锐度适应，边缘羽化sigma在0.50–0.80之间轻微变化；不通过侵蚀mask来裁掉机翼。尺寸按照各背景子集的原飞机框尺度估计，在原飞机附近的相似地表择位，并避让全部原有目标。

稀少类仍需注意独立前景数量：生成1,000个不同组合不等于1,000架不同飞机。表中列出本批实际使用的原始前景数；F-35等少样本类会较多复用原crop的不同扰动。仅使用MTARSI fixed/INNAR train、Mar-20 train前景；未混入INNAR test/valid或Mar-20 test。

## 标签与训练使用

前13个类别编号固定为：`{', '.join(CLASSES_13)}`（0–12）。每张新增一个细型号目标，同时将背景原有目标按原标签保留为`LAE/原类别名`，总计{len(names)}个标签类。这避免背景里原飞机和其他目标成为未标注负样本；原有泛飞机标签不猜测具体型号。训练时需读取本批`data.yaml`的完整类别表，不能直接套只有13个类别的配置。若仅做13型号任务，需要训练端对辅助原标签制定忽略/联合监督策略。

本批只提供训练增强集，验证应使用独立的未增强数据。HBB标注可直接用于水平框检测；OBB任务需另作导出。B-52到B-52H沿用前期已确定的类名映射，未重新识别原数据集的子型号。

## 展板

8999看板一级标题：**final产出**；二级：**2026-09-28 · 13类正式产出（每类200例）**。每类展示200张，共2,600张，按来源和LAE子集轮转抽样；默认全不选，按类别多选，保留全选/全不选。图片为插入后的结果，展板只mark新增飞机；正式训练标签仍包含原图目标。
'''
    atomic_bytes(DOC, doc.encode())
    print(json.dumps({'verified': len(rows), 'dashboard': 2600, 'root': str(root), 'document': str(DOC)}, ensure_ascii=False), flush=True)
    if not args.skip_reload:
        req = urllib.request.Request('http://127.0.0.1:8999/api/reload', method='POST')
        with urllib.request.urlopen(req, timeout=600) as response: print(response.read().decode(), flush=True)

if __name__ == '__main__': main()
