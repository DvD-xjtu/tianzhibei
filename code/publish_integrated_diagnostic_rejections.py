#!/usr/bin/env python3
"""Publish engineering-rejected composites as a temporary 8999 review area."""
from __future__ import annotations
import json, shutil
from datetime import datetime
from pathlib import Path

BASE=Path('/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1')
VIEW=BASE/'paste_integrated_diagnostic_rejected_view'
CONFIG=Path('/data2/tianzhibei/dataset-viewer/datasets.json')
SUBSETS=('DIOR','DOTAv2','FAIR1M','NWPU','RSOD','xView')
CLASSES=['C-17','C-130','KC-135','KC-10','A-10','F-15','F-16','F/A-18','F-35','B-1B','B-52H','E-2','E-3']
SOURCES={
 'mtarsi': {'label':'MTARSI · v9','candidate':BASE/'paste_integrated_diagnostic_mtarsi_100_candidates',
           'passed':BASE/'paste_integrated_diagnostic_mtarsi_100_pass'},
 'mar20': {'label':'Mar-20 · v1','candidate':BASE/'paste_integrated_diagnostic_mar20_100_candidates',
           'passed':BASE/'paste_integrated_diagnostic_mar20_100_pass'},
}

def main()->None:
 cfg=json.loads(CONFIG.read_text()); splits=[]; totals={}
 for src,meta in SOURCES.items():
  total_in=total_pass=total_reject=0
  for ds in SUBSETS:
   inp=meta['candidate']/ds; out=meta['passed']/ds
   decisions=[json.loads(s) for s in (out/'qa/decisions.jsonl').read_text().splitlines() if s.strip()]
   total_in+=len(decisions)
   passed_rows=[json.loads(s) for s in (out/'manifest.jsonl').read_text().splitlines() if s.strip()]
   passed_ids={r['id'] for r in passed_rows}; total_pass+=len(passed_ids)
   rejected=[d for d in decisions if not d['accepted']]
   total_reject+=len(rejected)
   root=VIEW/src/ds
   pass_root=root/'passed'
   for leaf in ('images','labels'):
    (root/leaf).mkdir(parents=True,exist_ok=True)
    (pass_root/leaf).mkdir(parents=True,exist_ok=True)
   records={r['id']:r for r in (json.loads(line) for line in (inp/'manifest.jsonl').read_text().splitlines() if line.strip())}
   for sample_id in passed_ids:
    for leaf,ext in (('images','.jpg'),('labels','.txt')):
     source_path=out/'train'/leaf/f'{sample_id}{ext}'
     dest=pass_root/leaf/f'{sample_id}{ext}'
     if source_path.is_file() and not dest.is_file(): shutil.copy2(source_path,dest)
   splits.append({'name':f'{meta["label"]} · {ds} · 通过 {len(passed_ids)}',
                  'images':f'{src}/{ds}/passed/images','labels':f'{src}/{ds}/passed/labels',
                  'label_prefix':''})
   for d in rejected:
    reasons=d.get('reasons') or ['unspecified']
    token='+'.join(reasons)
    # Keep image/label stems paired; prefix makes rejected examples obvious in the panel.
    if d['id'] not in records: raise RuntimeError(f'missing candidate manifest record: {d["id"]}')
    stem=f'REJECT__{token}__{d["id"]}'
    image_dest=root/'images'/f'{stem}.jpg'; label_dest=root/'labels'/f'{stem}.txt'
    for source_path,dest in ((inp/'final/train/images'/f'{d["id"]}.jpg',image_dest),
                             (inp/'final/train/labels'/f'{d["id"]}.txt',label_dest)):
     if dest.is_file():
      if source_path.read_bytes()!=dest.read_bytes(): raise RuntimeError(f'content mismatch: {dest}')
     else: shutil.copy2(source_path,dest)
   splitname=f'{meta["label"]} · {ds} · 未通过 {len(rejected)}'
   splits.append({'name':splitname,'images':f'{src}/{ds}/images','labels':f'{src}/{ds}/labels','label_prefix':''})
  totals[src]={'candidate':total_in,'passed':total_pass,'rejected':total_reject,
               'pass_rate':total_pass/total_in if total_in else 0}
 entries=[{'id':'copypaste-integrated-diagnostic-20260928',
   'name':'2026-09-28 · Copy-Paste 自动拒绝案例诊断','group':'exta-临时',
   'root':str(VIEW),'format':'yolo','training':False,'split_filter':True,
   'names':CLASSES,'splits':splits,
   'note':'临时诊断区：并列展示 MTARSI v9 方法与 Mar-20 v1 方法的工程通过/拒绝样例；子集可全选/全不选，也可只选“通过”拆分做对照。通过仅指自动质量门判定，不代表人工确认。'}]
 backup=CONFIG.with_name(f'datasets.json.backup-integrated-diagnostic-{datetime.now():%Y%m%d-%H%M%S}')
 shutil.copy2(CONFIG,backup)
 ids={d['id'] for d in entries}
 cfg['datasets']=[d for d in cfg['datasets'] if d.get('id') not in ids
                  and not d.get('id','').startswith('copypaste-integrated-diagnostic-')]
 cfg['datasets'].extend(entries); CONFIG.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'backup':str(backup),'entries':[d['id'] for d in entries],
                   'group':'exta-临时','secondary_title':entries[0]['name'],
                   'split_options':len(splits),'totals':totals},ensure_ascii=False))

if __name__=='__main__': main()
