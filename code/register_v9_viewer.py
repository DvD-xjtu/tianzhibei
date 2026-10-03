#!/usr/bin/env python3
"""Register a single filtered v9 LAE view in the shared 8999 viewer."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

CONFIG = Path('/data2/tianzhibei/dataset-viewer/datasets.json')
ROOT = Path('/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/paste_v9_lae1m_150pass_engineering_pass')
BASE = Path('/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1')
AGGREGATES = BASE / 'viewer_version_aggregates'
DATASETS = ('DIOR', 'DOTAv2', 'FAIR1M', 'NWPU', 'RSOD', 'xView')
NAMES = [
    'A1 SU-35', 'A2 C-130', 'A3 C-17', 'A4 C-5', 'A5 F-16',
    'A6 TU-160', 'A7 E-3', 'A8 B-52', 'A9 P-3C', 'A10 B-1B',
    'A11 E-8', 'A12 TU-22', 'A13 F-15', 'A14 KC-135', 'A15 F-22',
    'A16 F/A-18', 'A17 TU-95', 'A18 KC-10', 'A19 SU-34', 'A20 SU-24',
]
VERSION_IDS = {
    'v5': {'mtarsi-dior-paste-v7-before-after-40', 'mtarsi-dior-paste-v7-final-40'},
    'v6': {
        'mtarsi-dior-copypaste-v6-comparison-20', 'mtarsi-dior-copypaste-v6-final-20',
        'mtarsi-mar20-copypaste-v6-comparison-20', 'mtarsi-mar20-copypaste-v6-final-20',
        'mtarsi-mar20-copypaste-v6-curated-comparison-20',
        'mtarsi-mar20-copypaste-v6-curated-final-20',
    },
    'v7': {
        'mtarsi-mar20-copypaste-v7-candidates-200', 'mtarsi-mar20-copypaste-v7-auto-pass-56',
        'mtarsi-mar20-copypaste-v7-auto-pass-obb-56',
    },
    'v8': {
        'mtarsi-mar20-copypaste-v8-aug-candidates-500',
        'mtarsi-mar20-copypaste-v8-aug-engineering-pass-138',
        'mtarsi-mar20-copypaste-v8-aug-engineering-pass-obb-138',
    },
}
FOREGROUND_IDS = {
    'mtarsi-fixed', 'mtarsi-innar', 'cp-preview-cutouts-v1', 'cp-preview-composites-v1',
    'mtarsi-foreground-review-v3', 'mtarsi-foreground-auto-large-v4-40',
    'mtarsi-foreground-auto-large-v4-quality-gate',
    'mtarsi-foreground-auto-large-v4-100-quality-gate',
    'mtarsi-foreground-all-v6-quality-gate',
}


def make_view_link(version: str, source_name: str, source_root: Path) -> dict:
    split_root = AGGREGATES / version / source_name / 'train'
    split_root.mkdir(parents=True, exist_ok=True)
    for leaf in ('images', 'labels'):
        link = split_root / leaf
        target = source_root / 'train' / leaf
        if not target.is_dir():
            raise SystemExit(f'missing {version} {source_name} {leaf}: {target}')
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise SystemExit(f'refusing to replace existing path: {link}')
        link.symlink_to(target, target_is_directory=True)
    return {'name': source_name, 'images': f'{source_name}/train/images',
            'labels': f'{source_name}/train/labels', 'label_prefix': ''}


def legacy_version(version: str, sources: list[tuple[str, Path]], split_filter: bool) -> dict:
    splits = [make_view_link(version, name, source) for name, source in sources]
    return {
        'id': f'copypaste-{version}',
        'name': f'🧩 Copy-Paste {version} · 合成图',
        'group': '🔬 copy-paste', 'root': str(AGGREGATES / version),
        'format': 'yolo', 'training': False, 'names': NAMES, 'splits': splits,
        **({'split_filter': True} if split_filter else {}),
    }


def main() -> None:
    cfg = json.loads(CONFIG.read_text())
    for dataset in DATASETS:
        if not (ROOT / dataset / 'train/images').is_dir():
            raise SystemExit(f'missing accepted images: {ROOT / dataset}')
    backup = CONFIG.with_name(f'datasets.json.backup-v9-{datetime.now():%Y%m%d-%H%M%S}')
    shutil.copy2(CONFIG, backup)
    legacy_ids = set().union(*VERSION_IDS.values())
    registered_version_ids = {f'copypaste-v{version}' for version in ('5', '6', '7', '8')}
    legacy = [
        legacy_version('v5', [('DIOR', BASE / 'paste_preview_fixed_v7_40' / 'final')], False),
        legacy_version('v6', [
            ('DIOR', BASE / 'paste_v6_dior_20_r2' / 'final'),
            ('MAR20', BASE / 'paste_v6_mar20_20_r2' / 'final'),
            ('MAR20-curated', BASE / 'paste_v6_mar20_40_curated_v3'),
        ], False),
        legacy_version('v7', [('MAR20', BASE / 'paste_v7_mar20_200_engineering_pass')], False),
        legacy_version('v8', [('MAR20', BASE / 'paste_v8_mar20_aug_500_engineering_pass')], False),
    ]
    cfg['datasets'] = [d for d in cfg['datasets']
                       if d.get('id') not in legacy_ids
                       and d.get('id') not in registered_version_ids
                       and not d.get('id', '').startswith('v9-lae-')
                       and d.get('id') != 'copypaste-v9-lae1m']
    for dataset in cfg['datasets']:
        if dataset.get('id') in FOREGROUND_IDS:
            dataset['group'] = '📦 MTARSI前景/抠图参考'
            dataset['training'] = False
    cfg['datasets'].extend(legacy)
    cfg['datasets'].append({
        'id': 'copypaste-v9-lae1m',
        'name': '🧩 Copy-Paste v9 · LAE-1M（工程通过）',
        'group': '🔬 copy-paste',
        'root': str(ROOT),
        'format': 'yolo',
        'training': False,
        'split_filter': True,
        'names': NAMES,
        'splits': [
            {'name': dataset, 'images': f'{dataset}/train/images',
             'labels': f'{dataset}/train/labels', 'label_prefix': ''}
            for dataset in DATASETS
        ],
    })
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'backup': str(backup), 'versions': ['v5', 'v6', 'v7', 'v8', 'v9'],
                      'v9_subdatasets': list(DATASETS)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
