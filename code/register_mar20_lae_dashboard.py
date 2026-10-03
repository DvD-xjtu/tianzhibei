#!/usr/bin/env python3
"""Register MAR20→LAE v1 and separate legacy MTARSI paste versions in viewer."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

CONFIG = Path('/data2/tianzhibei/dataset-viewer/datasets.json')
BASE = Path('/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1')
ACCEPTED = BASE/'paste_mar20_v1_lae_engineering_pass'
AGGREGATE = BASE/'viewer_version_aggregates'/'copy-paste-mar-20-v1'
SUBSETS = ('DIOR', 'DOTAv2', 'FAIR1M', 'NWPU', 'RSOD', 'xView')
NAMES = [
    'A1 SU-35', 'A2 C-130', 'A3 C-17', 'A4 C-5', 'A5 F-16',
    'A6 TU-160', 'A7 E-3', 'A8 B-52', 'A9 P-3C', 'A10 B-1B',
    'A11 E-8', 'A12 TU-22', 'A13 F-15', 'A14 KC-135', 'A15 F-22',
    'A16 F/A-18', 'A17 TU-95', 'A18 KC-10', 'A19 SU-34', 'A20 SU-24',
]


def main() -> None:
    cfg = json.loads(CONFIG.read_text())
    for subset in SUBSETS:
        source = ACCEPTED/subset/'train'
        if not (source/'images').is_dir() or not (source/'labels').is_dir():
            raise SystemExit(f'missing accepted output: {source}')
    backup = CONFIG.with_name(f'datasets.json.backup-mar20-lae-v1-{datetime.now():%Y%m%d-%H%M%S}')
    shutil.copy2(CONFIG, backup)
    # Historical copy-paste batches use MTARSI-derived foregrounds; keep them
    # grouped together and make the new MAR20-source version independently selectable.
    for item in cfg.get('datasets', []):
        if item.get('group') == '🔬 copy-paste':
            item['group'] = '🔬 copy-paste-mtarsi'
        if item.get('id') == 'copypaste-v9-lae1m':
            item['name'] = '🧩 Copy-Paste v9-mtarsi · LAE-1M（工程通过）'
            item['group'] = '🔬 copy-paste-mtarsi'
    cfg['datasets'] = [d for d in cfg.get('datasets', []) if d.get('id') != 'copypaste-mar20-lae-v1']
    splits = []
    for subset in SUBSETS:
        view_root = AGGREGATE/subset/'train'
        view_root.mkdir(parents=True, exist_ok=True)
        for leaf in ('images', 'labels'):
            link = view_root/leaf
            target = ACCEPTED/subset/'train'/leaf
            if link.is_symlink() and link.resolve() == target.resolve():
                continue
            if link.exists() or link.is_symlink():
                raise SystemExit(f'refusing to replace existing path: {link}')
            link.symlink_to(target, target_is_directory=True)
        splits.append({'name': subset, 'images': f'{subset}/train/images',
                       'labels': f'{subset}/train/labels', 'label_prefix': ''})
    cfg['datasets'].append({
        'id': 'copypaste-mar20-lae-v1',
        'name': '🧩 Copy-Paste Mar-20 v1 · LAE-1M（工程通过）',
        'group': '🔬 copy-paste-mar-20', 'root': str(AGGREGATE),
        'format': 'yolo', 'training': False, 'split_filter': True,
        'names': NAMES, 'splits': splits,
        'note': 'Mar-20 train 前景；旋转、0.8–1.2 缩放和轻量颜色/噪声扰动；只登记工程质量门通过图，未人工审核。',
    })
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'backup': str(backup), 'legacy_group': '🔬 copy-paste-mtarsi',
                      'new_group': '🔬 copy-paste-mar-20', 'new_id': 'copypaste-mar20-lae-v1',
                      'subdatasets': list(SUBSETS)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
