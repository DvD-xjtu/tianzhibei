#!/usr/bin/env python3
"""Fail-closed structural audit for v6 smoke outputs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from synthesize_copypaste_v6 import CLASS_ID, digest, parse_mar20_label


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(root: Path) -> dict:
    records = read_jsonl(root / 'manifest.jsonl')
    backgrounds = read_jsonl(root / 'background_train.jsonl')
    summary = json.loads((root / 'qa/summary.json').read_text())
    config = json.loads((root / 'generation_config.json').read_text())
    assert len(records) == len(backgrounds) == summary['generated'] == config['count']
    assert summary['background_manifest_sha256'] == digest(root / 'background_train.jsonl')
    assert len({r['id'] for r in records}) == len(records)
    assert len({r['foreground_id'] for r in records}) == len(records)
    assert len({r['background_id'] for r in records}) == len(records)
    mode = config['mode']
    assert mode in {'dior', 'mar20'}
    for r, bg in zip(records, backgrounds):
        assert r['mode'] == mode and r['background_split'] == 'train'
        assert r['foreground_source'] in {'fixed', 'innar_train'}
        assert 'Test' not in r['foreground_source_path'] and 'valid' not in r['foreground_source_path']
        assert r['mar20_class_id'] == CLASS_ID[r['class_name']]
        assert r['train_ready'] == (mode == 'mar20')
        assert r['background_id'] == bg['background_id']
        assert digest(Path(r['background_image_path'])) == bg['image_sha256']
        assert digest(Path(r['background_label_path'])) == bg['label_sha256'] == r['background_label_sha256']
        assert digest(Path(r['foreground_path'])) == r['foreground_sha256']
        for field, sha in [('image_path', 'image_sha256'), ('label_path', 'label_sha256'),
                           ('instance_mask_path', 'instance_mask_sha256')]:
            assert digest(Path(r[field])) == r[sha], (r['id'], field)
        image = Image.open(r['image_path'])
        width, height = image.size
        mask = cv2.imread(r['instance_mask_path'], cv2.IMREAD_GRAYSCALE)
        assert mask is not None and mask.shape == (height, width) and mask.max() == 255
        x, y, w, h = cv2.boundingRect(cv2.findNonZero((mask > 0).astype('uint8')))
        box = r['bbox_xyxy']
        assert box == [x, y, x+w, y+h], (r['id'], box, (x, y, x+w, y+h))
        assert 0 <= x < x+w <= width and 0 <= y < y+h <= height
        assert 0 < r['actual_hbb_area_ratio'] < 1
        assert Path(r['image_path']).stem == Path(r['label_path']).stem
        lines = [line.strip() for line in Path(r['label_path']).read_text().splitlines() if line.strip()]
        if mode == 'mar20':
            assert '/train/' in r['background_image_path']
            original, anns = parse_mar20_label(Path(r['background_label_path']), width, height)
            assert lines[:-1] == original
            assert len(lines) == len(anns) + 1
            for ann in anns:
                ax1, ay1, ax2, ay2 = ann['bbox']
                assert ax2 + 8 <= x or x+w + 8 <= ax1 or ay2 + 8 <= y or y+h + 8 <= ay1
            assert len(lines[-1].split()) == 9
        else:
            assert len(lines) == 1 and len(lines[0].split()) == 5
            original_file = root / 'original_dior_annotations' / f'{r["id"]}.json'
            original = json.loads(original_file.read_text())
            assert len(original) >= 2
            for ann in original:
                ax, ay, aw, ah = ann['bbox']
                assert ax+aw+8 <= x or x+w+8 <= ax or ay+ah+8 <= y or y+h+8 <= ay
        for line in lines:
            values = line.split()
            assert 0 <= int(values[0]) < 20
            coords = [float(v) for v in values[1:]]
            assert all(np.isfinite(v) and 0 <= v <= 1 for v in coords)
        assert int(lines[-1].split()[0]) == r['mar20_class_id']
    assert dict(Counter(r['class_name'] for r in records)) == summary['classes']
    assert dict(Counter(r['foreground_source'] for r in records)) == summary['foreground_sources']
    return {'mode': mode, 'verified': len(records), 'classes': summary['classes'],
            'train_ready': mode == 'mar20', 'errors': 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
