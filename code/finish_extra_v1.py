#!/usr/bin/env python3
import json,shutil,hashlib
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
from produce_extra_engine import CLASSES_13,RULES,GATES
from produce_final_dataset import json_write
from verify_extra_v1 import audit
ROOT=Path('/data3/tianzhibei/derived/aircraft_copypaste_final_20260928/extra/class_supplement_v1')
CONFIG=Path('/data2/tianzhibei/dataset-viewer/datasets.json')
ID='copypaste-extra-v1-20261001'
def main():
 roots=[ROOT/'batches'/c for c in ['F-35','E-2','Helicopter','YF-12A']]
 (ROOT/'shards').mkdir(exist_ok=True)
 pool=[]
 for p in roots:
  pool += [json.loads(l) for l in (p/'foreground_pool.jsonl').read_text().splitlines()]
  for shard in (p/'shards').iterdir():
   dest=ROOT/'shards'/f'{p.name}__{shard.name}'
   if not dest.exists():dest.symlink_to(shard,target_is_directory=True)
 (ROOT/'foreground_pool.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pool))
 shutil.copy2(roots[0]/'taxonomy.json',ROOT/'taxonomy.json')
 summary=audit(ROOT,400)
 rows=[json.loads(l) for p in sorted((ROOT/'shards').glob('*/manifest.jsonl')) for l in p.read_text().splitlines()]
 (ROOT/'manifest.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in rows))
 for a in rows:
  for fld,key in [('images','image_path'),('labels','label_path'),('instance_masks','instance_mask_path')]:
   src=Path(a[key]);dest=ROOT/'train'/fld/src.name;dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():dest.symlink_to(src)
 names=json.loads((ROOT/'taxonomy.json').read_text())
 (ROOT/'data.yaml').write_text('path: '+str(ROOT)+'\ntrain: train/images\nnames: '+json.dumps(names,ensure_ascii=False)+'\n')
 view=ROOT/'dashboard';splits=[];crop_splits=[]
 for cls in ['F-35','E-2','Helicopter','YF-12A']:
  selected=[r for r in rows if r['instances'][0]['class_name']==cls]
  for r in selected:
   src=Path(r['image_path']);dest=view/cls/'images'/src.name;dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():dest.symlink_to(src)
   label=view/cls/'labels'/f'{src.stem}.txt';label.parent.mkdir(exist_ok=True);label.write_text('\n'.join(Path(r['label_path']).read_text().splitlines()[:r['actual_count']])+'\n')
  title=f'{cls} · {len(selected)}张'+(' · 航空斜俯拍试验（1个来源）' if cls=='YF-12A' else '')
  splits.append(dict(name=title,images=f'{cls}/images',labels=f'{cls}/labels',label_prefix=''))
  # Context crop sheets make foreground edge quality reviewable at native resolution.
  examples=selected[:12];sheet=Image.new('RGB',(1200,((len(examples)+3)//4)*330),'white');draw=ImageDraw.Draw(sheet)
  for i,r in enumerate(examples):
   im=Image.open(r['image_path']);b=r['instances'][0]['bbox_xyxy'];pad=max(b[2]-b[0],b[3]-b[1])*.55
   tile=im.crop((max(0,int(b[0]-pad)),max(0,int(b[1]-pad)),min(im.width,int(b[2]+pad)),min(im.height,int(b[3]+pad))))
   tile.thumbnail((294,295));x=i%4*300;y=i//4*330;sheet.paste(tile,(x,y));draw.text((x,y+299),r['id'].replace('extra_v1__',''),fill='black')
  sheet.save(ROOT/'qa'/f'pasted_{cls}.jpg',quality=95)
  crops=[r for r in pool if r['class_name']==cls]
  for r in crops:
   fg=Image.open(r['cutout_path']).convert('RGBA');alpha=np.array(fg.getchannel('A'));yy,xx=np.where(alpha>127);b=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
   canvas=Image.new('RGB',fg.size,(210,210,210));canvas.paste(fg,(0,0),fg)
   p=ROOT/'crop_dashboard'/cls/'images'/f'{r["id"]}.png';p.parent.mkdir(parents=True,exist_ok=True);canvas.save(p)
   from synthesize_copypaste_v6 import yolo_hbb_line
   l=ROOT/'crop_dashboard'/cls/'labels'/f'{r["id"]}.txt';l.parent.mkdir(exist_ok=True);l.write_text(yolo_hbb_line(CLASSES_13.index(cls),b,*fg.size)+'\n')
  crop_splits.append(dict(name=f'{cls} · {len(crops)}个 crop',images=f'{cls}/images',labels=f'{cls}/labels',label_prefix=''))
 summary.update(images_by_class=dict(Counter(r['instances'][0]['class_name'] for r in rows)),available_crops=dict(Counter(r['class_name'] for r in pool)),source_domain_note='YF-12A: one NASA historical elevated oblique photo, not satellite imagery. MTARSI external downloads duplicate existing sources; no new independent MTARSI origins claimed.',xview_excluded_reason='Current categories table and annotation IDs appear offset; apparent helicopter crops were fixed-wing aircraft. Excluded from this version.',training_ready=False)
 json_write(ROOT/'summary.json',summary)
 json_write(ROOT/'config.json',dict(version='extra-class-supplement-v1',rules=RULES,quality_gate=GATES,training_ready=False,source_selection='qa/visual_selection_*.json; source pool frozen after crop review',foreground_pool='foreground_pool.jsonl',taxonomy='taxonomy.json'))
 entries=[dict(id=ID,name='类别补充：v1 · 400张',group='extra',root=str(view),format='yolo',names=CLASSES_13,training=False,split_filter=True,splits=splits,note='四类单独预览；容量先估计后随机请求1–6架；YF-12A为NASA真实斜俯拍、单一来源的域差异试验。框仅展板绘制，训练图无框。'),dict(id=ID+'-crops',name='类别补充：v1 · 原始crop与分割',group='extra',root=str(ROOT/'crop_dashboard'),format='yolo',names=CLASSES_13,training=False,split_filter=True,splits=crop_splits,note='透明前景在中性灰底显示；YF-12A来自NASA EC71-2679，非遥感数据。')]
 cfg=json.loads(CONFIG.read_text());backup=CONFIG.with_name(f'datasets.json.backup-extra-v1-{datetime.now():%Y%m%d-%H%M%S}');shutil.copy2(CONFIG,backup)
 cfg['datasets']=[d for d in cfg['datasets'] if d['id'] not in [e['id'] for e in entries]]+entries;json_write(CONFIG,cfg);json_write(ROOT/'dashboard_registration.json',dict(entries=entries,backup=str(backup)))
 (ROOT/'code_snapshot').mkdir(exist_ok=True)
 for name in ['prepare_extra_v1.py','extract_extra_v1.py','refine_extra_masks.py','select_extra_pool.py','produce_extra_engine.py','run_extra_v1.py','verify_extra_v1.py','finish_extra_v1.py']:
  shutil.copy2(Path(__file__).parent/name,ROOT/'code_snapshot'/name)
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
