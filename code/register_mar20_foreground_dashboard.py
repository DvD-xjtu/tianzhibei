#!/usr/bin/env python3
"""Register the automatically accepted MAR20 cutouts as a separate viewer group."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

CONFIG = Path('/data2/tianzhibei/dataset-viewer/datasets.json')
BATCH = Path('/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/foreground_mar20_500_v1')
DATA_ID = 'mar20-foreground-sam2-v1-500'
GROUP = 'Mar-20抠图参考'


def main() -> None:
    summary = json.loads((BATCH / 'qa/summary.json').read_text())
    cfg = json.loads(CONFIG.read_text())
    backup = CONFIG.with_name(f'datasets.json.backup-mar20-cutout-{datetime.now():%Y%m%d-%H%M%S}')
    shutil.copy2(CONFIG, backup)
    cfg['datasets'] = [d for d in cfg['datasets'] if d.get('id') != DATA_ID]
    names = [
        'A1 SU-35', 'A2 C-130', 'A3 C-17', 'A4 C-5', 'A5 F-16',
        'A6 TU-160', 'A7 E-3', 'A8 B-52', 'A9 P-3C', 'A10 B-1B',
        'A11 E-8', 'A12 TU-22', 'A13 F-15', 'A14 KC-135', 'A15 F-22',
        'A16 F/A-18', 'A17 TU-95', 'A18 KC-10', 'A19 SU-34', 'A20 SU-24',
    ]
    cfg['datasets'].append({
        'id': DATA_ID,
        'name': f'Mar-20 SAM2 抠图 · 自动门通过（{summary["passed"]}/{summary["candidate_count"]}）',
        'group': GROUP, 'root': str(BATCH / 'viewer_auto_accepted'),
        'format': 'yolo', 'names': names, 'training': False,
        'note': '透明前景抠图的中性底色参考图；仅自动质量门筛选，未人工审核。',
        'splits': [{'name': 'automatic_pass', 'images': 'train/images',
                    'labels': 'train/labels', 'label_prefix': ''}],
    })
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'backup': str(backup), 'id': DATA_ID, 'group': GROUP,
                      'passed': summary['passed'], 'candidates': summary['candidate_count']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
