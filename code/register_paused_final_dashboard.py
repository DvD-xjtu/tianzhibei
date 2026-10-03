#!/usr/bin/env python3
"""Publish existing passed images without resuming generation or requiring full quotas."""
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import random
import shutil

from produce_final_dataset import FINAL_ROOT, CLASSES_13, json_write
from synthesize_copypaste_v6 import atomic_bytes, digest
from filter_copypaste_v7 import rejection_reasons

CONFIG = Path('/data2/tianzhibei/dataset-viewer/datasets.json')


def main():
    byclass = defaultdict(list)
    ids = set()
    for path in sorted(FINAL_ROOT.glob('shards/*/*/manifest.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['id'] in ids: raise RuntimeError(f'duplicate ID: {row["id"]}')
            ids.add(row['id'])
            assert row['quality_gate_passed'] and not rejection_reasons(row['automatic_metrics'])
            byclass[row['class_name']].append(row)
    view = FINAL_ROOT/'dashboard_paused_20260928'
    splits, selected_rows = [], []
    for ci, cls in enumerate(CLASSES_13):
        if len(byclass[cls]) < 200: raise RuntimeError(f'{cls}: fewer than 200 passed images')
        rng = random.Random(20260928+ci)
        buckets = defaultdict(list)
        for row in byclass[cls]: buckets[(row['subdataset'], row['foreground_source'])].append(row)
        for bucket in buckets.values(): rng.shuffle(bucket)
        used_bases = Counter()
        selected = []
        while len(selected) < 200:
            for key in sorted(buckets):
                bucket = buckets[key]
                if not bucket: continue
                # Spread review examples over libraries/sources, prioritising distinct original crops.
                i = min(range(len(bucket)), key=lambda j: used_bases[(bucket[j]['foreground_source'], bucket[j]['foreground_base_id'])])
                row = bucket.pop(i)
                selected.append(row)
                used_bases[(row['foreground_source'], row['foreground_base_id'])] += 1
                if len(selected) == 200: break
        slug = cls.replace('/', '-')
        for row in selected:
            image = Path(row['image_path'])
            label = Path(row['label_path'])
            assert digest(image) == row['image_sha256'] and digest(label) == row['label_sha256']
            first = label.read_text().splitlines()[0]
            assert int(first.split()[0]) == CLASSES_13.index(cls)
            dest = view/slug/'images'/image.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_symlink():
                assert dest.resolve() == image.resolve()
            elif dest.exists():
                raise RuntimeError(f'refusing to replace {dest}')
            else:
                dest.symlink_to(image)
            # Mark only the pasted fine-grained aircraft; full training labels stay untouched.
            atomic_bytes(view/slug/'labels'/f'{image.stem}.txt', (first+'\n').encode())
            selected_rows.append(row)
        assert len(list((view/slug/'images').glob('*.jpg'))) == 200
        splits.append({'name': f'{cls} · 200例', 'images': f'{slug}/images',
                       'labels': f'{slug}/labels', 'label_prefix': ''})
    atomic_bytes(view/'selected_manifest.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selected_rows).encode())
    counts = {cls: len(byclass[cls]) for cls in CLASSES_13}
    json_write(view/'summary.json', {'data_generation_paused': True, 'actual_passed': sum(counts.values()),
        'passed_by_class': counts, 'displayed_by_class': {cls: 200 for cls in CLASSES_13},
        'displayed_total': len(selected_rows), 'images_copied': 0, 'new_images_generated': 0,
        'selection': 'library/source round-robin with original-crop diversity priority',
        'displayed_files_hash_verified': len(selected_rows), 'visual_review_performed': False})
    cfg = json.loads(CONFIG.read_text())
    backup = CONFIG.with_name(f'datasets.json.backup-paused-final-{datetime.now():%Y%m%d-%H%M%S}')
    shutil.copy2(CONFIG, backup)
    replaced_ids = {'copypaste-final-13class-20260928', 'copypaste-integrated-mtarsi-10samples',
                    'copypaste-integrated-mar20-10samples'}
    cfg['datasets'] = [d for d in cfg['datasets'] if d['id'] not in replaced_ids]
    for d in cfg['datasets']:
        if 'copy-paste-整合' in d.get('group', ''): d['group'] = 'final产出'
    cfg['datasets'].append({'id': 'copypaste-final-13class-20260928',
        'name': '2026-09-28 · 当前通过批次（12,899张；每类展示200）',
        'group': 'final产出', 'root': str(view), 'format': 'yolo', 'names': CLASSES_13,
        'training': False, 'split_filter': True, 'splits': splits,
        'note': '生成已暂停。当前通过12,899张：12类各1,000张，E-2为899张。每类展示200张，勾选类别查看；MTARSI v9＋Mar-20 v1。只mark新增飞机，未做人工视觉筛选。'})
    json_write(CONFIG, cfg)
    print(json.dumps({'group': 'final产出', 'actual_passed': sum(counts.values()),
                      'displayed': len(selected_rows), 'backup': str(backup)}, ensure_ascii=False))


if __name__ == '__main__': main()
