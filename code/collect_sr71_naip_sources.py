#!/usr/bin/env python3
"""Collect verified SR-71 examples from public-domain USGS NAIP orthoimagery."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import requests
from PIL import Image


ROOT = Path('/data3/tianzhibei/derived/aircraft_copypaste_final_20260928/extra/SR71_NAIP_sources_20261003')
CONFIG = Path('/data2/tianzhibei/dataset-viewer/datasets.json')
SERVICE = 'https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer'
SITES = [
    dict(id='barksdale_61-7967', name='Barksdale Global Power Museum',
         serial='61-7967', lon=-93.6815850286, lat=32.5113580853,
         reference='https://www.barksdale.af.mil/News/Display/Article/1204379/barksdale-global-power-museum-more-than-just-history/'),
    dict(id='palmdale_61-7973', name='Blackbird Airpark',
         serial='61-7973', lon=-118.0858354658, lat=34.6028212605,
         reference='https://flighttestmuseum.org/blackbird-airpark/'),
]


def main() -> None:
    images = ROOT / 'images'
    images.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    records = []
    for site in SITES:
        lon, lat = site['lon'], site['lat']
        # Broad context preserves nearby landmarks and lets the user inspect the model.
        bbox = [lon - .0015, lat - .0012, lon + .0015, lat + .0012]
        params = dict(f='image', bbox=','.join(map(str, bbox)), bboxSR=4326,
                      imageSR=4326, size='1200,960', format='jpg')
        response = session.get(SERVICE + '/exportImage', params=params, timeout=90)
        response.raise_for_status()
        path = images / (site['id'] + '.jpg')
        path.write_bytes(response.content)
        with Image.open(path) as im:
            im.verify()

        identify = dict(f='json', geometry=json.dumps(dict(x=lon, y=lat,
                        spatialReference=dict(wkid=4326))), geometryType='esriGeometryPoint',
                        returnCatalogItems='true', returnGeometry='false')
        info = session.get(SERVICE + '/identify', params=identify, timeout=60).json()
        features = info.get('catalogItems', {}).get('features', [])
        source = next((f['attributes'] for f in features if f['attributes'].get('Category') == 1), {})
        records.append(dict(**site, image_path=str(path), bbox_wgs84=bbox,
                            image_size=[1200, 960], source_service=SERVICE,
                            source_request_url=response.url, acquisition_date=source.get('acquisition_date'),
                            source_tile=source.get('Name'), source_year=source.get('Year'),
                            source_resolution_m=source.get('resolution_value'),
                            source_agency=source.get('agency'), true_model='SR-71A',
                            source_domain='NAIP_orthorectified_aerial',
                            stage='raw_unsegmented_source', training_eligible=False))

    (ROOT / 'manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'README.md').write_text(
        '# SR-71 遥感原图候选（NAIP）\n\n'
        '两张独立地点的美国 USGS/USDA NAIP 正射航空遥感图，均为露天静态展机；'
        '属于遥感图，但不是卫星图，也不是已抠出的透明前景。'
        'Barksdale 为 SR-71A 61-7967；Palmdale 上方偏右的 Blackbird 为 SR-71A 61-7973，'
        '下方偏左的是 A-12，不能当成 SR-71。'
        '详情、采集日期和原图请求链接见 manifest.json。'
        '原始图无框，未纳入 final v3 训练。\n\n'
        'USGS NAIP 来源与使用条款：'
        'https://www.usgs.gov/centers/eros/science/usgs-eros-archive-aerial-photography-national-agriculture-imagery-program-naip\n'
    )

    cfg = json.loads(CONFIG.read_text())
    entry = dict(id='extra-sr71-naip-sources-20261003', name='SR-71 · NAIP真实遥感原图（2处）',
                 group='extra', root=str(images), format='images', training=False,
                 note='USGS/USDA NAIP正射航空遥感；Barksdale 61-7967、Palmdale 61-7973。'
                      'Palmdale另一架是A-12。原图无框，尚未分割，未并入训练集。')
    backup = CONFIG.with_name(f'datasets.json.backup-sr71-naip-{datetime.now():%Y%m%d-%H%M%S}')
    shutil.copy2(CONFIG, backup)
    cfg['datasets'] = [d for d in cfg['datasets'] if d.get('id') != entry['id']] + [entry]
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(images=len(records), root=str(ROOT), dashboard_id=entry['id'],
                          backup=str(backup), records=records), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
