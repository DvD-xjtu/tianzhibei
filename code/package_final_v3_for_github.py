#!/usr/bin/env python3
"""Package final v3 images and frozen crop pools into portable tar shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path


DATASET = Path('/data3/tianzhibei/derived/final_v3_20261003/combined_21500_train_1500_test')
V3 = Path('/data3/tianzhibei/derived/aircraft_copypaste_final_20260928/final_v3_20261001')


def lines(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, records):
    with path.open('w') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def tar_files(target: Path, entries):
    with tarfile.open(target, 'w', dereference=True) as archive:
        for source, name in entries:
            if not source.is_file():
                raise FileNotFoundError(source)
            archive.add(source, arcname=name, recursive=False)
    print(f'{target.name}: {target.stat().st_size:,} bytes', flush=True)


def safe_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value)


def crop_pool(paths):
    by_id = {}
    for path in paths:
        for row in lines(path):
            old = by_id.get(row['id'])
            if old is not None and (old['class_name'] != row['class_name'] or old['cutout_path'] != row['cutout_path']):
                raise ValueError(f'inconsistent crop ID: {row["id"]}')
            by_id[row['id']] = row
    return [by_id[key] for key in sorted(by_id)]


def prepare_metadata(dataset: Path, stage: Path):
    out = stage / 'dataset'
    out.mkdir(parents=True, exist_ok=True)
    for name in ('summary.json', 'taxonomy.json'):
        (out / name).write_bytes((dataset / name).read_bytes())
    yaml = (dataset / 'dataset.yaml').read_text()
    yaml = re.sub(r'^path:.*$', 'path: .', yaml, count=1, flags=re.MULTILINE)
    (out / 'dataset.yaml').write_text(yaml)
    (out / 'README_generation.md').write_bytes((dataset / 'README.md').read_bytes())
    (out / 'README.md').write_text(
        '# Final v3 可下载数据包\n\n'
        '将本仓库 `data/` 中全部 tar 文件解压到同一目录后，`dataset/` 即为数据集根目录。'
        '图像、标注和掩膜均为真实文件，不依赖服务器上的符号链接。\n\n'
        '训练集 21,500 张，合成测试集 1,500 张。`dataset.yaml` 中的 `path: .` '
        '按数据集目录解释；若训练框架另有路径规则，请改成解压后的绝对路径。'
        '完整统计及来源限制见 `README_generation.md`。\n\n'
        '`index_train.jsonl` 和 `index_test.jsonl` 的图像、标注及掩膜路径相对于 `dataset/`。'
        '其中 `background_image_path` 和 `provenance_manifest` 为原始生成环境的来源记录，'
        '对应的上游原图及中间 manifest 未打包。\n', encoding='utf-8')
    for split in ('train', 'test'):
        records = []
        for row in lines(dataset / f'index_{split}.jsonl'):
            row = dict(row)
            for key, directory in (('image_path', f'{split}/images'),
                                   ('label_path', f'{split}/labels'),
                                   ('mask_path', f'instance_masks/{split}')):
                row[key] = f'{directory}/{Path(row[key]).name}'
            records.append(row)
        expected = 21500 if split == 'train' else 1500
        if len(records) != expected:
            raise ValueError(f'{split}: {len(records)} != {expected}')
        write_jsonl(out / f'index_{split}.jsonl', records)
    return out


def package_crops(v3: Path, data_dir: Path, stage: Path):
    pools = {
        'train': crop_pool((v3 / 'foreground_pool.jsonl',
                            v3 / 'extra/F-35_1000_20261002/foreground_pool.jsonl',
                            v3 / 'extra/SR71_proxy_YF12A_500_20261003/foreground_pool.jsonl')),
        'test': crop_pool((v3 / 'test_case_limited_1500_20261003/foreground_pool.jsonl',)),
    }
    if (len(pools['train']), len(pools['test'])) != (4933, 5156):
        raise ValueError('frozen crop pool counts changed')
    entries = []
    all_ids = set()
    for split, rows in pools.items():
        manifest = []
        for row in rows:
            if row['id'] in all_ids:
                raise ValueError(f'train/test crop overlap: {row["id"]}')
            all_ids.add(row['id'])
            cutout = Path(row['cutout_path'])
            context = Path(row.get('source_path') or row['source_crop_path'])
            stem = safe_name(row['id'])
            cls = safe_name(row['class_name'])
            cutout_rel = f'crops/{split}/{cls}/cutouts/{stem}{cutout.suffix.lower()}'
            context_rel = f'crops/{split}/{cls}/contexts/{stem}{context.suffix.lower()}'
            entries.extend(((cutout, cutout_rel), (context, context_rel)))
            record = dict(row)
            record['cutout_path'] = cutout_rel
            record['source_path'] = context_rel
            record['source_crop_path'] = context_rel
            record.pop('original_cutout_path', None)
            manifest.append(record)
        path = stage / 'crops' / f'manifest_{split}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(path, manifest)
        entries.append((path, f'crops/{path.name}'))
    (stage / 'crops' / 'README.md').write_text(
        '# Final v3 冻结 crop 池\n\n'
        '训练前景池 4,933 个、测试前景池 5,156 个，按 crop ID 隔离。'
        '每个 case 有透明 PNG 和原始裁片；manifest 中的 `cutout_path`、'
        '`source_path` 相对于解压根目录。`source_image_path` 是未打包的上游原图路径，'
        '仅用于记录来源。测试集每个 crop 最多使用 3 次。\n', encoding='utf-8')
    entries.append((stage / 'crops' / 'README.md', 'crops/README.md'))
    tar_files(data_dir / 'crop_cases.tar', entries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--v3', type=Path, default=V3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data_dir = args.output / 'data'
    stage = args.output / '.bundle_metadata'
    data_dir.mkdir(parents=True, exist_ok=True)
    stage.mkdir(parents=True, exist_ok=True)
    metadata = prepare_metadata(args.dataset, stage)

    images = sorted((args.dataset / 'train/images').glob('*.jpg'))
    if len(images) != 21500:
        raise ValueError(f'train images: {len(images)} != 21500')
    for group, start in enumerate(range(0, len(images), 5000)):
        chunk = images[start:start + 5000]
        tar_files(data_dir / f'train_images_{group:02d}.tar',
                  ((p, f'dataset/train/images/{p.name}') for p in chunk))

    entries = []
    for split, folder, expected in (('train', 'labels', 21500),
                                    ('test', 'images', 1500),
                                    ('test', 'labels', 1500),
                                    ('instance_masks/train', '', 21500),
                                    ('instance_masks/test', '', 1500)):
        root = args.dataset / split / folder
        files = sorted(p for p in root.iterdir() if p.is_file())
        if len(files) != expected:
            raise ValueError(f'{root}: {len(files)} != {expected}')
        entries.extend((p, f'dataset/{split}/{folder}/{p.name}'.replace('//', '/')) for p in files)
    entries.extend((p, f'dataset/{p.name}') for p in metadata.iterdir() if p.is_file())
    preview = args.dataset / 'preview/test_contact_sheet_with_boxes.jpg'
    entries.append((preview, f'dataset/preview/{preview.name}'))
    tar_files(data_dir / 'dataset_other.tar', entries)
    package_crops(args.v3, data_dir, stage)

    with (data_dir / 'SHA256SUMS').open('w') as handle:
        for path in sorted(data_dir.glob('*.tar')):
            sha = hashlib.sha256()
            with path.open('rb') as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    sha.update(block)
            handle.write(f'{sha.hexdigest()}  {path.name}\n')
    print('package complete', flush=True)


if __name__ == '__main__':
    main()
