#!/usr/bin/env python3
"""Register audited final v2 preview under the existing final产出 group."""
from datetime import datetime
import json
from pathlib import Path
import shutil
from produce_final_v2 import DEFAULT_ROOT
from produce_final_dataset import CLASSES_13, json_write
from synthesize_copypaste_v6 import atomic_bytes

CONFIG=Path('/data2/tianzhibei/dataset-viewer/datasets.json')
DATASET_ID='copypaste-final-v2-20261001'


def main():
    root=DEFAULT_ROOT
    summary=json.loads((root/'summary.json').read_text())
    assert summary['audit_passed'] and summary['images']==1000
    rows=[json.loads(l) for p in sorted((root/'shards').glob('*/manifest.jsonl')) for l in p.read_text().splitlines()]
    view=root/'dashboard'; splits=[]
    for n in range(1,7):
        subset=[r for r in rows if r['actual_count']==n]
        if not subset: continue
        for r in subset:
            src=Path(r['image_path']); dest=view/f'count_{n}'/'images'/src.name
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.is_symlink(): assert dest.resolve()==src.resolve()
            elif dest.exists(): raise RuntimeError(f'cannot overwrite {dest}')
            else: dest.symlink_to(src)
            labels=Path(r['label_path']).read_text().splitlines()[:n]
            atomic_bytes(view/f'count_{n}'/'labels'/f'{src.stem}.txt',('\n'.join(labels)+'\n').encode())
        splits.append(dict(name=f'新增{n}架 · {len(subset)}张',images=f'count_{n}/images',labels=f'count_{n}/labels',label_prefix=''))
    entry=dict(id=DATASET_ID,name=f"v2 · 容量约束多实例与正态尺度（1,000张／{summary['instances']}架）",
        group='final产出',root=str(view),format='yolo',names=CLASSES_13,training=False,split_filter=True,splits=splits,
        note='先试排布估计容量，再随机请求1至容量（最多6架）；混合原始crop；局部GT截断正态尺度；额外明暗扰动。按实际新增数量筛选，只显示新增飞机框。预览待人工验收。')
    cfg=json.loads(CONFIG.read_text())
    backup=CONFIG.with_name(f'datasets.json.backup-final-v2-{datetime.now():%Y%m%d-%H%M%S}')
    shutil.copy2(CONFIG,backup)
    cfg['datasets']=[d for d in cfg['datasets'] if d['id']!=DATASET_ID]+[entry]
    json_write(CONFIG,cfg); json_write(root/'dashboard_registration.json',dict(entry=entry,backup=str(backup)))
    print(json.dumps(dict(id=DATASET_ID,images=len(rows),splits=splits),ensure_ascii=False))


if __name__=='__main__':main()
