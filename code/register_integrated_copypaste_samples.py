#!/usr/bin/env python3
"""Expose 10 accepted examples per foreground-source × LAE-subdataset cell."""
from __future__ import annotations
import json, shutil
from datetime import datetime
from pathlib import Path

CONFIG=Path('/data2/tianzhibei/dataset-viewer/datasets.json')
BASE=Path('/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1')
SOURCES={
 'mtarsi': {'root':BASE/'paste_integrated_mtarsi_10k_per_subset','title':'MTARSI · v9 方法 · 展示样例'},
 'mar20': {'root':BASE/'paste_integrated_mar20_10k_per_subset','title':'Mar-20 · v1 方法 · 展示样例'},
}
SUBSETS=('DIOR','DOTAv2','FAIR1M','NWPU','RSOD','xView')
CLASSES=['C-17','C-130','KC-135','KC-10','A-10','F-15','F-16','F/A-18','F-35','B-1B','B-52H','E-2','E-3']

def balanced_ten(manifest:Path)->list[dict]:
 rows=[json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
 by={c:[] for c in CLASSES}
 for r in rows:
  if r['class_name'] in by: by[r['class_name']].append(r)
 for c in by: by[c].sort(key=lambda r:r['id'])
 chosen=[]; depth=0
 while len(chosen)<10:
  progressed=False
  for cls in CLASSES:
   if depth<len(by[cls]):
    chosen.append(by[cls][depth]); progressed=True
    if len(chosen)==10: break
  if not progressed: break
  depth+=1
 if len(chosen)<10: raise SystemExit(f'only {len(chosen)} passed records in {manifest.parent}')
 return chosen

def main()->None:
 cfg=json.loads(CONFIG.read_text()); entries=[]
 for source,meta in SOURCES.items():
  splits=[]
  for ds in SUBSETS:
   accepted=meta['root']/ds
   manifest=accepted/'manifest.jsonl'
   if not manifest.is_file(): raise SystemExit(f'missing accepted manifest {manifest}')
   selected=balanced_ten(manifest)
   split_name=f'{ds} · 10例'
   split_root=BASE/'viewer_integrated_samples'/source/ds
   for leaf,key in [('images','image_path'),('labels','label_path')]:
    out=split_root/leaf; out.mkdir(parents=True,exist_ok=True)
    for n,row in enumerate(selected,1):
     source_path=Path(row[key]); target=out/f'{n:02d}__{source_path.name}'
     if target.is_symlink() and target.resolve()==source_path.resolve(): continue
     if target.exists() or target.is_symlink(): raise SystemExit(f'refusing to replace {target}')
     target.symlink_to(source_path,target_is_directory=False)
   splits.append({'name':split_name,'images':f'{source}/{ds}/images',
                  'labels':f'{source}/{ds}/labels','label_prefix':''})
  entries.append({'id':f'copypaste-integrated-{source}-10samples',
    'name':meta['title']+'（六子集×10）','group':'🧩 copy-paste-整合',
    'root':str(BASE/'viewer_integrated_samples'),'format':'yolo','training':False,
    'split_filter':True,'names':CLASSES,'splits':splits,
    'note':'每个 LAE 子集按类别轮转抽取 10 张工程质量门通过样例；仅展示抽样，完整批次另见对应版本。'})
 backup=CONFIG.with_name(f'datasets.json.backup-integrated-{datetime.now():%Y%m%d-%H%M%S}')
 shutil.copy2(CONFIG,backup)
 ids={x['id'] for x in entries}; cfg['datasets']=[d for d in cfg['datasets'] if d.get('id') not in ids]
 cfg['datasets'].extend(entries); CONFIG.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'backup':str(backup),'entries':[x['id'] for x in entries],
                   'samples_per_entry':60,'total_samples':120,'group':'🧩 copy-paste-整合'},ensure_ascii=False))

if __name__=='__main__':main()
