#!/usr/bin/env python3
"""Structural and split audit for one v9 LAE-1M preview batch."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from synthesize_copypaste_v6 import CLASS_ID, digest

ALLOWED = {'DIOR', 'DOTAv2', 'FAIR1M', 'NWPU', 'RSOD', 'xView'}


def audit(root: Path, expected_dataset: str | None = None) -> dict:
    records = [json.loads(x) for x in (root/'manifest.jsonl').read_text().splitlines() if x.strip()]
    summary = json.loads((root/'qa/summary.json').read_text())
    config = json.loads((root/'generation_config.json').read_text())
    assert summary['generated'] == config['count'] == len(records)
    assert summary['mode'] == 'lae1m_preview' and summary['train_ready'] is False
    dataset = summary['subdataset']
    assert dataset in ALLOWED and config['dataset'] == dataset
    if expected_dataset:
        assert dataset == expected_dataset
    assert len({r['id'] for r in records}) == len(records)
    assert len({r['foreground_id'] for r in records}) == len(records)
    classes = Counter()
    for r in records:
        assert r['mode'] == 'lae1m_preview' and r['subdataset'] == dataset
        assert r['background_dataset'] == dataset and r['background_split'] == 'train'
        assert r['foreground_source'] in {'fixed', 'innar_train'}
        assert 'valid' not in r['foreground_source_path'].lower()
        assert '/LAE-1M/LAE-FOD/' in r['background_image_path']
        if dataset == 'FAIR1M':
            assert Path(r['background_image_path']).name.startswith('train_')
        for pathkey, hashkey in (('background_image_path', 'background_image_sha256'),
                                 ('background_label_path', 'background_label_sha256'),
                                 ('foreground_path', 'foreground_sha256'),
                                 ('image_path', 'image_sha256'),
                                 ('label_path', 'label_sha256'),
                                 ('instance_mask_path', 'instance_mask_sha256'),
                                 ('original_annotations_path', 'original_annotations_sha256')):
            assert digest(Path(r[pathkey])) == r[hashkey], (r['id'], pathkey)
        image = Image.open(r['image_path']).convert('RGB')
        width, height = image.size
        mask = cv2.imread(r['instance_mask_path'], cv2.IMREAD_GRAYSCALE)
        assert mask is not None and mask.shape == (height, width) and mask.max() == 255
        x, y, w, h = cv2.boundingRect(cv2.findNonZero((mask > 0).astype(np.uint8)))
        assert r['bbox_xyxy'] == [x, y, x+w, y+h]
        tokens = Path(r['label_path']).read_text().strip().split()
        assert len(tokens) == 5 and int(tokens[0]) == r['mar20_class_id'] == CLASS_ID[r['class_name']]
        vals = np.asarray([float(v) for v in tokens[1:]])
        assert np.isfinite(vals).all() and ((vals >= 0) & (vals <= 1)).all()
        ann = json.loads(Path(r['original_annotations_path']).read_text())
        assert ann['dataset'] == dataset and ann['split'] == 'train'
        assert r['train_ready'] is False
        classes[r['class_name']] += 1
    assert dict(classes) == summary['classes']
    return {'subdataset': dataset, 'verified': len(records), 'classes': dict(classes), 'errors': 0,
            'unique_backgrounds': len({r['source_background_id'] for r in records}),
            'train_ready': False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('batch', type=Path)
    parser.add_argument('--dataset')
    args = parser.parse_args()
    print(json.dumps(audit(args.batch, args.dataset), ensure_ascii=False))


if __name__ == '__main__':
    main()
