#!/usr/bin/env python3
"""Record visual QA decisions and build a viewer payload for approved cutouts."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True, type=Path)
    parser.add_argument('--approved', required=True, nargs='+', help='two-digit batch ids, e.g. 01 02')
    args = parser.parse_args()
    approved = set(args.approved)
    manifest_path = args.batch / 'manifest.json'
    records = json.loads(manifest_path.read_text())
    viewer = args.batch / 'viewer_approved' / 'train'
    (viewer / 'images').mkdir(parents=True, exist_ok=True)
    (viewer / 'labels').mkdir(parents=True, exist_ok=True)
    for record in records:
        record['approved'] = record['id'].split('__', 1)[0] in approved
        if not record['approved']:
            continue
        stem = record['id']
        shutil.copy2(args.batch / 'viewer' / 'images' / f'{stem}.jpg', viewer / 'images' / f'{stem}.jpg')
        shutil.copy2(args.batch / 'viewer' / 'labels' / f'{stem}.txt', viewer / 'labels' / f'{stem}.txt')
    manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    shutil.copy2(args.batch / 'viewer' / 'data.yaml', args.batch / 'viewer_approved' / 'data.yaml')
    print(json.dumps({'approved': sum(r['approved'] for r in records), 'total': len(records)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
