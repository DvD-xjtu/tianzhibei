#!/usr/bin/env python3
"""Register the SR-71 proxy v3 increment for visual review."""
from __future__ import annotations

import json
import os
from pathlib import Path

from produce_final_dataset import SUBSETS
from produce_sr71_v3_supplement import ROOT

CONFIG=Path('/data2/tianzhibei/dataset-viewer/datasets.json')
ID='copypaste-final-v3-sr71-proxy-500-20261003'


def main():
    summary=json.loads((ROOT/'summary.json').read_text())
    assert summary['audit_passed'] and summary['images']==500
    names=json.loads((ROOT/'taxonomy.json').read_text())
    data=json.loads(CONFIG.read_text())
    data['datasets']=[d for d in data['datasets'] if d['id']!=ID]
    data['datasets'].append(dict(id=ID,name='Final v3 · SR-71代理YF-12A · 500张',
        group='final产出',root=str(ROOT),format='yolo',names=names,
        training=True,split_filter=True,
        splits=[dict(name=f'{ds} · {summary["subdatasets"][ds]}张',
                     images=f'shards/{ds}/train/images',
                     labels=f'shards/{ds}/preview_labels',label_prefix='')
                for ds in SUBSETS],
        note='真实机型SR-71A，训练代理标签YF-12A（ID 13）；38个v2源case按机体划分，27个训练、9个封存测试来源、2个源图质检剔除。500张均至少含1架SR-71代理。框只由展板绘制；真实训练JPG无框，完整训练标签在shards/*/train/labels。'))
    temp=CONFIG.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    os.replace(temp,CONFIG)
    print(ID,ROOT)


if __name__=='__main__':main()
