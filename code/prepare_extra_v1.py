#!/usr/bin/env python3
"""Fetch rare-class originals, exclude held-out duplicates, index local helicopters."""
import json, hashlib, re
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw

ROOT=Path('/data3/tianzhibei/derived/aircraft_copypaste_final_20260928/extra/class_supplement_v1')
INNAR=Path('/data3/tianzhibei/datasets/MTARSI-INNAR/extracted/MTARSI-INNAR')
LAE=Path('/data2/tianzhibei/LAE-1M/LAE-FOD')

def write(name,data):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def pixelhash(p):
 im=Image.open(p).convert('RGB');return hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()

def phash(p):
 a=np.asarray(Image.open(p).convert('L').resize((32,32)),np.float32)
 d=cv2.dct(a)[:8,:8];return (d>np.median(d[1:])).flatten()

def fetch():
 jobs=[]
 for cls in ['F-35','E-2']:
  url=f'https://huggingface.co/api/datasets/amistele/MTARSI-fixed/tree/main/{cls}?limit=1000'
  items=[]
  while url:
   r=requests.get(url,timeout=40);r.raise_for_status();items+=r.json();url=r.links.get('next',{}).get('url')
  write(f'external/{cls}/listing.json',items)
  for item in items:
   if Path(item['path']).suffix.lower() in ('.jpg','.png','.jpeg'):
    jobs.append((cls,item))
 def one(job):
  cls,item=job;p=ROOT/'external'/item['path'];url='https://huggingface.co/datasets/amistele/MTARSI-fixed/resolve/main/'+item['path']
  if not p.exists():
   for retry in range(3):
    try:
     r=requests.get(url,timeout=45);r.raise_for_status();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(r.content);break
    except requests.RequestException:
     if retry==2:raise
  Image.open(p).verify()
  return dict(class_name=cls,image_path=str(p),source='mtarsi_fixed_external',source_url=url,
      source_dataset='MTARSI-fixed',source_split='external_unsplit',pixel_sha256=pixelhash(p),
      sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 with ThreadPoolExecutor(max_workers=8) as ex:external=list(ex.map(one,jobs))
 write('external/download_manifest.json',external)
 return external

def main():
 ROOT.mkdir(parents=True,exist_ok=True);external=fetch();print('downloaded',Counter(r['class_name'] for r in external),flush=True)
 held=[];local=[]
 for cls,folder in [('F-35','F-35_JSF'),('E-2','E-2_Hawkeye')]:
  for split in ['Train','Test','valid']:
   for p in sorted((INNAR/split/folder).glob('*')):
    if p.suffix.lower() not in ('.jpg','.jpeg','.png'):continue
    r=dict(class_name=cls,image_path=str(p),source='innar_'+split.lower(),source_dataset='MTARSI-INNAR',source_split=split.lower(),pixel_sha256=pixelhash(p))
    (local if split=='Train' else held).append(r)
 for p in sorted(Path('/data3/tianzhibei/datasets/MTARSI-fixed/E-2').glob('*')):
  if p.suffix.lower() in ('.jpg','.png','.jpeg'):local.append(dict(class_name='E-2',image_path=str(p),source='fixed',source_dataset='MTARSI-fixed',source_split='external_unsplit',pixel_sha256=pixelhash(p)))
 hh={r['pixel_sha256'] for r in held};hp=[(r['class_name'],phash(r['image_path'])) for r in held]
 seen=set();chosen=[];excluded=[]
 for r in local+external:
  reason=None
  if r['pixel_sha256'] in hh:reason='exact_heldout_overlap'
  elif r['pixel_sha256'] in seen:reason='duplicate_original_pixels'
  else:
   ph=phash(r['image_path'])
   if any(c==r['class_name'] and np.count_nonzero(ph!=h)<=4 for c,h in hp):reason='near_heldout_overlap_phash_le4'
  if reason:excluded.append(dict(r,reason=reason));continue
  seen.add(r['pixel_sha256']);r['source_scene_key']=r['source_dataset']+'/'+r['pixel_sha256'];chosen.append(r)
 write('rare_source_exclusions.json',excluded)
 # Helicopters: preserve parent group, reject clipped targets and global-coordinate tile duplicates.
 heli=[];dropped=[]
 for ds,manifest,idir in [('DOTAv2','DOTAv2/processed_LAE-1M_DOTAv2_train.json','DOTAv2/images'),('xView','xview/processed_LAE-1M_Xview_train_1024_05.json','xview/train_images_1024_05')]:
  data=json.loads((LAE/manifest).read_text());cats={c['id']:c['name'] for c in data['categories']};ims={i['id']:i for i in data['images']}
  for a in data['annotations']:
   if cats.get(a['category_id'],'').lower()!='helicopter':continue
   im=ims[a['image_id']];p=LAE/idir/im['file_name'];x,y,w,h=a['bbox'];W,H=im['width'],im['height']
   if min(w,h)<24 or x<2 or y<2 or x+w>W-2 or y+h>H-2:
    dropped.append(dict(dataset=ds,image=str(p),annotation=a['id'],reason='small_or_tile_boundary'));continue
   stem=p.stem;scene=re.sub(r'_\d{4}$','',stem) if ds=='DOTAv2' else '_'.join(stem.split('_')[:-4])
   global_box=None
   if ds=='xView':
    ox,oy,_,_=map(int,stem.split('_')[-4:]);global_box=[x+ox,y+oy,x+ox+w,y+oy+h]
   heli.append(dict(class_name='Helicopter',image_path=str(p),source=ds.lower()+'_train',source_dataset=ds,
      source_split='train',source_scene_key=ds+'/'+scene,annotation_id=a['id'],bbox_xyxy=[x,y,x+w,y+h],global_box=global_box,
      annotation_manifest=str(LAE/manifest)))
 def iou(a,b):
  inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-6)
 accepted=[];seen_crop=set()
 for r in heli:
  if r['global_box'] and any(t['source_scene_key']==r['source_scene_key'] and t['global_box'] and iou(t['global_box'],r['global_box'])>.65 for t in accepted):
   dropped.append(dict(r,reason='overlapping_tile_same_instance'));continue
  with Image.open(r['image_path']) as im:
   crop=im.convert('RGB').crop(tuple(map(round,r['bbox_xyxy'])))
   key=hashlib.sha256(str(crop.size).encode()+crop.tobytes()).hexdigest()
  if key in seen_crop:dropped.append(dict(r,reason='exact_crop_duplicate'));continue
  seen_crop.add(key);accepted.append(r)
 # Limit dense same-scene contributions while using every available source scene.
 groups=defaultdict(list)
 for r in accepted:groups[r['source_scene_key']].append(r)
 balanced=[]
 for key in sorted(groups):
  items=sorted(groups[key],key=lambda r:-(r['bbox_xyxy'][2]-r['bbox_xyxy'][0])*(r['bbox_xyxy'][3]-r['bbox_xyxy'][1]))
  balanced+=items[:6]
 write('helicopter_exclusions.json',dropped)
 jobs=chosen+balanced
 for n,r in enumerate(jobs):r['id']=f"extra_{r['class_name']}_{n:04d}";r['training_eligible']=True
 (ROOT/'source_jobs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in jobs))
 write('source_summary.json',dict(downloaded=dict(Counter(r['class_name'] for r in external)),
    rare_exclusions=dict(Counter(r['reason'] for r in excluded)),jobs_by_class=dict(Counter(r['class_name'] for r in jobs)),
    helicopter_after_tile_dedup=len(accepted),helicopter_scene_groups=len(groups),helicopter_jobs=len(balanced),
    external_unique_jobs=dict(Counter(r['class_name'] for r in chosen if r['source']=='mtarsi_fixed_external'))))
 # Source review sheet, all rare images and the first scene-diverse helicopter examples.
 for cls in ['F-35','E-2','Helicopter']:
  items=[r for r in jobs if r['class_name']==cls]
  for page in range((len(items)+35)//36):
   batch=items[page*36:(page+1)*36];canvas=Image.new('RGB',(1200,((len(batch)+5)//6)*180),'white');draw=ImageDraw.Draw(canvas)
   for i,r in enumerate(batch):
    with Image.open(r['image_path']) as im:
     im=im.convert('RGB')
     if 'bbox_xyxy' in r:
      x,y,x2,y2=r['bbox_xyxy'];pad=max(x2-x,y2-y)*.3;im=im.crop((max(0,int(x-pad)),max(0,int(y-pad)),min(im.width,int(x2+pad)),min(im.height,int(y2+pad))))
     im.thumbnail((195,150));x=(i%6)*200;y=(i//6)*180;canvas.paste(im,(x,y));draw.text((x,y+152),r['id'],fill='black')
   out=ROOT/'qa'/f'sources_{cls}_{page}.jpg';out.parent.mkdir(exist_ok=True);canvas.save(out)
 print((ROOT/'source_summary.json').read_text(),flush=True)

if __name__=='__main__':main()
