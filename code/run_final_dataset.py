#!/usr/bin/env python3
"""Run all 78 class/subdataset quotas with bounded parallelism."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from produce_final_dataset import FINAL_ROOT, CLASSES_13, SUBSETS, taxonomy, json_write, class_quotas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=FINAL_ROOT)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--wait-existing', action='store_true', help='wait for already-running shards to finish before quota top-up')
    args = ap.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    json_write(root/'taxonomy.json', taxonomy())
    config_path = root/'production_config.json'
    started = json.loads(config_path.read_text()).get('started_unix', config_path.stat().st_mtime) if config_path.exists() else time.time()
    json_write(config_path, {
        'started_unix': started,
        'target_classes': CLASSES_13, 'images_per_class': 1000, 'total_images': 13000,
        'subdatasets': SUBSETS, 'workers': args.workers, 'seed': 20260928,
        'foreground_versions': {'mtarsi': 'v9', 'mar20': 'v1'},
        'label_format': 'YOLO HBB', 'quality_gate': 'engineering-v7-1',
        'rejected_image_retention': False, 'human_or_ai_visual_filtering': False,
        'class_subdataset_quotas': {cls: class_quotas(cls) for cls in CLASSES_13}})
    script = Path(__file__).with_name('produce_final_dataset.py')
    logs = root/'logs'
    logs.mkdir(exist_ok=True)
    def job(ds, cls):
        progress = root/'shards'/ds/cls.replace('/', '-')/'progress.json'
        if args.wait_existing and progress.exists():
            deadline = time.monotonic()+3600
            while not json.loads(progress.read_text()).get('completed'):
                if time.monotonic() > deadline: raise RuntimeError(f'waiting shard timed out: {ds}/{cls}')
                time.sleep(5)
        if progress.exists():
            saved = json.loads(progress.read_text())
            if saved.get('completed') and saved['passed_by_class'].get(cls, 0) == class_quotas(cls)[ds]:
                return ds, cls, 0
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        with (logs/f'{ds}__{cls.replace("/", "-")}.log').open('a') as log:
            result = subprocess.run([sys.executable, str(script), '--root', str(root),
                '--dataset', ds, '--class-name', cls], stdout=log, stderr=subprocess.STDOUT, env=env)
        return ds, cls, result.returncode
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, ds, cls) for cls in CLASSES_13 for ds in SUBSETS if class_quotas(cls)[ds] > 0]
        for f in as_completed(futures):
            ds, cls, status = f.result()
            print(json.dumps({'dataset': ds, 'class': cls, 'exit_code': status}), flush=True)
            if status: failures.append([ds, cls, status])
    if failures:
        json_write(root/'failed_jobs.json', failures)
        raise SystemExit(f'{len(failures)} workers failed; rerun resumes incomplete quotas')
    subprocess.run([sys.executable, str(Path(__file__).with_name('publish_final_dataset.py')),
        '--root', str(root)], check=True)

if __name__ == '__main__': main()
