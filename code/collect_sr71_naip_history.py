#!/usr/bin/env python3
"""Build a small, provenance-preserving SR-71 NAIP time-series source set."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageOps


ROOT = Path('/data3/tianzhibei/derived/aircraft_copypaste_final_20260928/extra/SR71_NAIP_sources_20261003')
STAC = 'https://planetarycomputer.microsoft.com/api/stac/v1/search'
DATA = 'https://planetarycomputer.microsoft.com/api/data/v1/item/bbox'
SITES = [
    dict(id='barksdale', lon=-93.6815850286, lat=32.5113580853, serial='61-7967',
         name='Barksdale Global Power Museum', true_model='SR-71A'),
    dict(id='palmdale', lon=-118.0858354658, lat=34.6028212605, serial='61-7973',
         name='Blackbird Airpark', true_model='SR-71A'),
    dict(id='lackland', lon=-98.620466667, lat=29.389833333, serial='61-7979',
         name='Joint Base San Antonio Lackland', true_model='SR-71A'),
    dict(id='march', lon=-117.2661328, lat=33.8828316, serial='61-7975',
         name='March Field Air Museum', true_model='SR-71A'),
    dict(id='eglin', lon=-86.5620722222, lat=30.4659972222, serial='61-7959',
         name='Air Force Armament Museum', true_model='SR-71A'),
    dict(id='edwards', lon=-117.92053399, lat=34.9123695, serial='61-7955',
         name='Air Force Flight Test Museum', true_model='SR-71A'),
    dict(id='castle', lon=-120.577783333, lat=37.3640888889, serial='61-7960',
         name='Castle Air Museum', true_model='SR-71A'),
    dict(id='beale', lon=-121.390083333, lat=39.1136833333, serial='61-7963',
         name='Beale AFB Heritage Park', true_model='SR-71A'),
]


def fetch(job: tuple[dict, dict]) -> dict:
    site, item = job
    lon, lat = site['lon'], site['lat']
    bbox = [lon - .0015, lat - .0012, lon + .0015, lat + .0012]
    bbox_str = ','.join(map(str, bbox))
    url = f'{DATA}/{bbox_str}/1200x960.png'
    params = [('collection', 'naip'), ('item', item['id']), ('assets', 'image'),
              ('bidx', 1), ('bidx', 2), ('bidx', 3)]
    out = ROOT / 'images' / f"{site['id']}_{item['properties']['datetime'][:10]}_{site['serial']}.png"
    if not out.exists():
        response = requests.get(url, params=params, timeout=90)
        response.raise_for_status()
        if not response.headers.get('content-type', '').startswith('image/'):
            raise RuntimeError(f'Unexpected response: {response.text[:300]}')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(response.content)
    with Image.open(out) as im:
        im.verify()
    return dict(id=out.stem, site=site['id'], name=site['name'], serial=site['serial'],
                true_model=site['true_model'], image_path=str(out),
                acquisition_datetime=item['properties']['datetime'],
                gsd_m=item['properties'].get('gsd'), stac_item_id=item['id'],
                stac_item_url=item['links'][0]['href'],
                original_cog_url=item['assets']['image']['href'],
                crop_bbox_wgs84=bbox, source_domain='NAIP_orthorectified_aerial',
                stage='raw_unsegmented_source', training_eligible=False,
                note='Same static airframe at this site across years; not independent aircraft.')


def main() -> None:
    jobs = []
    for site in SITES:
        lon, lat = site['lon'], site['lat']
        box = f'{lon-.0001},{lat-.0001},{lon+.0001},{lat+.0001}'
        response = requests.get(STAC, params=dict(collections='naip', bbox=box, limit=100), timeout=30)
        response.raise_for_status()
        for item in response.json()['features']:
            jobs.append((site, item))
    with ThreadPoolExecutor(max_workers=5) as pool:
        rows = list(pool.map(fetch, jobs))
    rows.sort(key=lambda r: (r['site'], r['acquisition_datetime']))
    (ROOT / 'historical_manifest.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')

    # Review sheet is diagnostic only; original per-date images remain untouched.
    w, h = 300, 270
    sheet = Image.new('RGB', (w * 4, h * ((len(rows) + 3) // 4)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        with Image.open(row['image_path']) as im:
            tile = ImageOps.contain(im.convert('RGB'), (w - 8, h - 25))
        x, y = (i % 4) * w, (i // 4) * h
        sheet.paste(tile, (x + (w - tile.width)//2, y + (h - 25 - tile.height)//2))
        draw.text((x + 5, y + h - 22), f"{row['site']} {row['acquisition_datetime'][:10]}", fill='black')
    sheet.save(ROOT / 'historical_contact_sheet.jpg', quality=92)
    print(json.dumps(dict(total=len(rows), by_site={s['id']:sum(r['site']==s['id'] for r in rows)
                                        for s in SITES}, root=str(ROOT)), ensure_ascii=False))


if __name__ == '__main__':
    main()
