#!/usr/bin/env python3
"""Audit foreground source overlap against reserved test references in v3."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from produce_final_dataset import FINAL_ROOT, SUBSETS, json_write
from produce_final_v3_mixed import ROOT as V3
from produce_f35_v3_supplement import ROOT as F35
from produce_sr71_v3_supplement import ROOT as SR71
from register_v3_sr71_combined import ROOT as COMBINED


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main():
    held=rows(FINAL_ROOT/'crop-metadata/heldout_reference_pool.jsonl')
    held_by_id={r['id']:r for r in held}
    used_ids=set();used_by_class=defaultdict(set);used_by_batch=defaultdict(set)
    for batch,root in (('base_v3',V3),('F35_supplement',F35),('SR71_proxy_supplement',SR71)):
        for ds in SUBSETS:
            for r in rows(root/'shards'/ds/'manifest.jsonl'):
                for inst in r['instances']:
                    fid=inst['foreground_id']
                    used_ids.add(fid)
                    if fid in held_by_id:
                        used_by_class[held_by_id[fid]['class_name']].add(fid)
                        used_by_batch[batch].add(fid)
    overlap=used_ids&set(held_by_id)
    assert len(overlap)==6
    assert {k:len(v) for k,v in used_by_class.items()}=={'E-2':4,'F-35':2}
    pool_by_id={}
    for root in (V3,F35,SR71):
        pool_by_id.update({r['id']:r for r in rows(root/'foreground_pool.jsonl')})
    used_source_paths={pool_by_id[fid].get('source_image_path') for fid in used_ids}
    held_source_paths={r.get('source_image_path') for r in held}
    source_path_overlap=(used_source_paths&held_source_paths)-{None}
    assert len(source_path_overlap)==6
    sr_train=rows(SR71/'sr71_training_cases.jsonl')
    sr_held=rows(SR71/'sr71_heldout_cases.jsonl')
    train_serials={r['serial'] for r in sr_train}
    held_serials={r['serial'] for r in sr_held}
    assert len(sr_train)==27 and len(sr_held)==9 and not (train_serials&held_serials)
    assert not (used_ids&{r['id'] for r in sr_held})
    report=dict(combined_training_images=21500,original_heldout_reference_cases=len(held),
        originally_heldout_cases_used_for_training=len(overlap),
        originally_heldout_used_by_class={k:len(v) for k,v in used_by_class.items()},
        originally_heldout_used_by_batch={k:len(v) for k,v in used_by_batch.items()},
        originally_heldout_used_ids=sorted(overlap),
        originally_heldout_source_image_paths_reused=len(source_path_overlap),
        original_heldout_reference_cases_not_used=len(held)-len(overlap),
        sr71_training_cases=len(sr_train),sr71_reserved_cases=len(sr_held),
        sr71_train_airframes=sorted(train_serials),sr71_reserved_airframes=sorted(held_serials),
        sr71_reserved_cases_used_in_training=0,
        detector_test_images_created=0,
        audit_scope='Exact case IDs and original source-image paths; near-duplicate scene matching was not performed for legacy data.',
        audit_note='Existing v3 uses 4 E-2 and 2 F-35 original test-reference crops for training; these six must be excluded from any independent evaluation. SR-71 new split is airframe-disjoint. No independent detector test set exists yet.')
    json_write(COMBINED/'split_audit.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
