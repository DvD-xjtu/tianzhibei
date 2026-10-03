#!/usr/bin/env python3
"""Detached completion watcher: audit and report after all generation shards finish."""
from __future__ import annotations
import json,os,time,traceback
from pathlib import Path
from audit_final_v3_mixed import audit
from produce_final_v3_mixed import ROOT
from produce_final_dataset import SUBSETS,json_write


def main():
    root=ROOT
    pid=json.loads((root/'generation_process.json').read_text())['pid']
    while True:
        progress=[]
        for ds in SUBSETS:
            p=root/'shards'/ds/'progress.json'
            progress.append(json.loads(p.read_text()) if p.exists() else {})
        if all(p.get('completed') and p.get('images')==p.get('goal') for p in progress):break
        if not Path(f'/proc/{pid}').exists():
            raise RuntimeError('generation process exited before all shards completed; rerun produce_final_v3_mixed.py to resume')
        time.sleep(30)
    summary=audit(root)
    json_write(root/'completion_status.json',dict(status='complete',images=summary['images'],
        audit_passed=summary['audit_passed'],report=str(root/'final_v3_report.md')))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        json_write(ROOT/'completion_status.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc()))
        raise
