#!/usr/bin/env python3
"""Apply the frozen v7 engineering quality gate to one v9 LAE preview batch."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from filter_copypaste_v7 import GATES, metrics, rejection_reasons
from synthesize_copypaste_v6 import atomic_bytes, digest
from verify_lae1m_v9 import audit


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--append', action='store_true',
                        help='append a supplemental batch to an existing accepted output')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.append:
        parser.error('output must be new or empty')
    prior_records, prior_decisions, prior_summary = [], [], None
    if args.append:
        if not (args.output/'manifest.jsonl').is_file() or not (args.output/'qa/summary.json').is_file():
            parser.error('--append requires an existing filtered output')
        prior_records = [json.loads(x) for x in (args.output/'manifest.jsonl').read_text().splitlines() if x.strip()]
        prior_decisions = [json.loads(x) for x in (args.output/'qa/decisions.jsonl').read_text().splitlines() if x.strip()]
        prior_summary = json.loads((args.output/'qa/summary.json').read_text())
    input_audit = audit(args.input)
    records = [json.loads(x) for x in (args.input/'manifest.jsonl').read_text().splitlines() if x.strip()]
    accepted, decisions = [], []
    primary, all_reasons = Counter(), Counter()
    for record in records:
        value = metrics(record)
        reasons = rejection_reasons(value)
        all_reasons.update(reasons)
        if reasons:
            primary[reasons[0]] += 1
        passed = not reasons
        decisions.append({'id': record['id'], 'subdataset': record['subdataset'],
                          'class_name': record['class_name'], 'accepted': passed,
                          'reasons': reasons, 'metrics': {k: round(v, 6) for k, v in value.items()}})
        if not passed:
            continue
        stem = record['id']
        paths = {'image_path': args.output/'train/images'/f'{stem}.jpg',
                 'label_path': args.output/'train/labels'/f'{stem}.txt',
                 'instance_mask_path': args.output/'instance_masks'/f'{stem}.png',
                 'comparison_path': args.output/'comparison/train/images'/f'{stem}.jpg',
                 'comparison_label_path': args.output/'comparison/train/labels'/f'{stem}.txt',
                 'original_annotations_path': args.output/'original_annotations'/f'{stem}.json'}
        copy_file(Path(record['image_path']), paths['image_path'])
        copy_file(Path(record['label_path']), paths['label_path'])
        copy_file(Path(record['instance_mask_path']), paths['instance_mask_path'])
        copy_file(args.input/'comparison/train/images'/f'{stem}.jpg', paths['comparison_path'])
        copy_file(args.input/'comparison/train/labels'/f'{stem}.txt', paths['comparison_label_path'])
        copy_file(Path(record['original_annotations_path']), paths['original_annotations_path'])
        item = dict(record)
        for key, path in paths.items():
            item[key] = str(path)
        item.update({'review_status': 'engineering_pass_unreviewed',
                     'automatic_metrics': {k: round(v, 6) for k, v in value.items()},
                     'source_manifest_path': str(args.input/'manifest.jsonl')})
        for key, hashkey in (('image_path', 'image_sha256'), ('label_path', 'label_sha256'),
                             ('instance_mask_path', 'instance_mask_sha256'),
                             ('original_annotations_path', 'original_annotations_sha256')):
            if digest(Path(item[key])) != item[hashkey]:
                raise RuntimeError(f'copy checksum mismatch: {stem}: {key}')
        accepted.append(item)
    copy_file(args.input/'final/data.yaml', args.output/'data.yaml')
    copy_file(args.input/'comparison/data.yaml', args.output/'comparison/data.yaml')
    combined_records = prior_records + accepted
    combined_decisions = prior_decisions + decisions
    atomic_bytes(args.output/'manifest.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in combined_records).encode())
    atomic_bytes(args.output/'qa/decisions.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in combined_decisions).encode())
    previous_sources = prior_summary['source_count'] if prior_summary else 0
    previous_passed = prior_summary['passed'] if prior_summary else 0
    prior_all = Counter(prior_summary.get('rejection_reasons_all', {})) if prior_summary else Counter()
    prior_primary = Counter(prior_summary.get('rejection_primary_reasons', {})) if prior_summary else Counter()
    prior_classes = Counter(prior_summary.get('passed_classes', {})) if prior_summary else Counter()
    all_reasons.update(prior_all)
    primary.update(prior_primary)
    prior_classes.update(Counter(x['class_name'] for x in accepted))
    total_sources = previous_sources + len(records)
    total_passed = previous_passed + len(accepted)
    summary = {'source_count': total_sources, 'passed': total_passed,
               'rejected': total_sources-total_passed,
               'engineering_pass_rate': total_passed/total_sources if total_sources else 0,
               'subdataset': input_audit['subdataset'],
               'passed_classes': dict(prior_classes),
               'rejection_reasons_all': dict(all_reasons), 'rejection_primary_reasons': dict(primary),
               'filter_version': 'engineering-v7-1', 'thresholds': GATES,
               'human_or_ai_visual_review_performed_on_this_batch': False,
               'train_ready': False, 'original_annotations_preserved_separately': True,
               'appended_batch_source_count': len(records),
               'source_manifest_sha256': digest(args.input/'manifest.jsonl')}
    atomic_bytes(args.output/'qa/summary.json', (json.dumps(summary, ensure_ascii=False, indent=2)+'\n').encode())
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
