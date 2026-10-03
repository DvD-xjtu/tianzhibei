#!/usr/bin/env python3
"""Structural/source-split audit for the MAR20-foreground LAE preview."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from synthesize_copypaste_v6 import digest
from synthesize_lae1m_mar20_v1 import CLASS_ID

ALLOWED = {'DIOR', 'DOTAv2', 'FAIR1M', 'NWPU', 'RSOD', 'xView'}


def audit(root: Path) -> dict:
    records = [json.loads(x) for x in (root/'manifest.jsonl').read_text().splitlines() if x.strip()]
    summary = json.loads((root/'qa/summary.json').read_text())
    config = json.loads((root/'generation_config.json').read_text())
    dataset = summary['subdataset']
    assert dataset in ALLOWED and config['dataset'] == dataset
    assert summary['generated'] == config['count'] == len(records)
    assert summary['mode'] == 'lae1m_preview' and summary['train_ready'] is False
    assert len({x['id'] for x in records}) == len(records)
    classes = Counter()
    for r in records:
        assert r['pipeline_version'] == 'copy-paste-mar-20-v1-lae-preview-1'
        assert r['subdataset'] == dataset and r['background_split'] == 'train'
        assert r['foreground_source'] == 'mar20_train'
        assert r['class_name'] in CLASS_ID and r['mar20_class_id'] == CLASS_ID[r['class_name']]
        assert '/LAE-1M/LAE-FOD/' in r['background_image_path']
        if dataset == 'FAIR1M':
            assert Path(r['background_image_path']).name.startswith('train_')
        for pathkey, hashkey in (
            ('background_image_path','background_image_sha256'), ('background_label_path','background_label_sha256'),
            ('foreground_path','foreground_sha256'), ('image_path','image_sha256'),
            ('label_path','label_sha256'), ('instance_mask_path','instance_mask_sha256'),
            ('original_annotations_path','original_annotations_sha256')):
            assert digest(Path(r[pathkey])) == r[hashkey], (r['id'], pathkey)
        image = Image.open(r['image_path']).convert('RGB')
        width, height = image.size
        mask = cv2.imread(r['instance_mask_path'], cv2.IMREAD_GRAYSCALE)
        assert mask is not None and mask.shape == (height, width) and mask.max() == 255
        x,y,w,h = cv2.boundingRect(cv2.findNonZero((mask > 0).astype(np.uint8)))
        assert r['bbox_xyxy'] == [x,y,x+w,y+h]
        tokens = Path(r['label_path']).read_text().strip().split()
        assert len(tokens) == 5 and int(tokens[0]) == CLASS_ID[r['class_name']]
        values = np.asarray([float(v) for v in tokens[1:]])
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        assert json.loads(Path(r['original_annotations_path']).read_text())['split'] == 'train'
        classes[r['class_name']] += 1
    assert dict(classes) == summary['classes']
    return {'subdataset': dataset, 'verified': len(records), 'classes': dict(classes),
            'errors': 0, 'train_ready': False}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('batch', type=Path); args = ap.parse_args()
    print(json.dumps(audit(args.batch), ensure_ascii=False))
