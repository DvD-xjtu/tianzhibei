#!/usr/bin/env python3
"""Independent structural and provenance audit for curated v6 MAR20 output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from synthesize_copypaste_v6 import digest, parse_mar20_label


def lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    summary = json.loads((root / 'qa/summary.json').read_text())
    records = [json.loads(line) for line in lines(root / 'manifest.jsonl')]
    decisions = [json.loads(line) for line in lines(root / 'qa/decisions.jsonl')]
    accepted = {r['id'] for r in decisions if r['accepted']}
    exported = summary.get('accepted', summary.get('passed'))
    assert len(records) == exported == len(accepted)
    assert len(decisions) == summary['source_count']
    assert {r['id'] for r in records} == accepted
    assert len({r['foreground_id'] for r in records}) == len(records)
    assert len({r['background_id'] for r in records}) == len(records)
    assert len(list((root / 'train/images').glob('*.jpg'))) == len(records)
    assert len(list((root / 'train/labels').glob('*.txt'))) == len(records)
    assert len(list((root / 'instance_masks').glob('*.png'))) == len(records)
    assert len(list((root / 'comparison/train/images').glob('*.jpg'))) == len(records)
    assert len(list((root / 'comparison/train/labels').glob('*.txt'))) == len(records)
    for record in records:
        assert record['mode'] == 'mar20' and record['train_ready']
        assert record['review_status'] in {'visual_approved', 'engineering_pass_unreviewed'}
        assert record['foreground_source'] in {'fixed', 'innar_train'}
        for field, checksum in [('image_path', 'image_sha256'), ('label_path', 'label_sha256'),
                                ('instance_mask_path', 'instance_mask_sha256')]:
            path = Path(record[field])
            assert path.is_relative_to(root), (record['id'], field)
            assert digest(path) == record[checksum], (record['id'], field)
        image = Image.open(record['image_path'])
        width, height = image.size
        mask = cv2.imread(record['instance_mask_path'], cv2.IMREAD_GRAYSCALE)
        assert mask is not None and mask.shape == (height, width)
        x, y, w, h = cv2.boundingRect(cv2.findNonZero((mask > 0).astype('uint8')))
        assert record['bbox_xyxy'] == [x, y, x+w, y+h]
        original, _ = parse_mar20_label(Path(record['background_label_path']), width, height)
        label_lines = lines(Path(record['label_path']))
        assert label_lines[:-1] == original and len(label_lines[-1].split()) == 9
        assert int(label_lines[-1].split()[0]) == record['mar20_class_id']
        assert all(np.isfinite(float(v)) and 0 <= float(v) <= 1 for line in label_lines
                   for v in line.split()[1:])
    print(json.dumps({'accepted': len(records), 'rejected': summary['rejected'],
                      'verified': len(records), 'errors': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
