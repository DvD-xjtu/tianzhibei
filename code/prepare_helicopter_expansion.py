import json,hashlib,re,xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from PIL import Image,ImageDraw
ROOT=Path('/data3/tianzhibei/derived/aircraft_copypaste_final_20260928/extra/source_expansion')
LAE=Path('/data2/tianzhibei/LAE-1M/LAE-FOD');SIMD=Path('/data3/tianzhibei/datasets/external_aircraft_supplement/SIMD')
def main():
 raw=[]
 for ds,manifest,images,cid in [('DOTAv2','DOTAv2/processed_LAE-1M_DOTAv2_train.json','DOTAv2/images',12),('xView','xview/processed_LAE-1M_Xview_train_1024_05.json','xview/train_images_1024_05',4)]:
  data=json.loads((LAE/manifest).read_text());ims={i['id']:i for i in data['images']}
  for a in data['annotations']:
   if a['category_id']!=cid:continue
   im=ims[a['image_id']];p=LAE/images/im['file_name'];x,y,w,h=a['bbox'];stem=p.stem;scene=re.sub(r'_\d{4}$','',stem) if ds=='DOTAv2' else '_'.join(stem.split('_')[:-4])
   gb=None
   if ds=='xView':
    ox,oy,_,_=map(int,stem.split('_')[-4:]);gb=[x+ox,y+oy,x+w+ox,y+h+oy]
   raw.append(dict(id=f'exp_{ds}_{a["id"]}',class_name='Helicopter',image_path=str(p),bbox_xyxy=[x,y,x+w,y+h],width=im['width'],height=im['height'],source=ds.lower()+'_train',source_dataset=ds,source_scene_key=ds+'/'+scene,source_split='train',global_box=gb,annotation_id=a['id'],annotation_manifest=str(LAE/manifest),category_mapping_note='xView original annotations are one-based; category4 visually verified helicopters' if ds=='xView' else 'DOTA category12 helicopter'))
 if (SIMD/'train').exists():
  files={p.stem:p for p in (SIMD/'train').rglob('*') if p.suffix.lower() in ['.jpg','.png','.jpeg']}
  for a in json.loads((SIMD/'helicopter_inventory.json').read_text()):
   if a['source_split']!='train' or a['image_stem'] not in files:continue
   p=files[a['image_stem']];im=Image.open(p);b=a['bbox'];raw.append(dict(id=f'exp_SIMD_{a["image_stem"]}_{a["object_index"]}',class_name='Helicopter',image_path=str(p),bbox_xyxy=[float(b[k]) for k in ['xmin','ymin','xmax','ymax']],width=im.width,height=im.height,source='simd_train',source_dataset='SIMD',source_scene_key='SIMD/'+a['image_stem'],source_split='train',global_box=None,annotation_path=a['annotation_path'],source_url='https://github.com/ihians/simd'))
 def iou(a,b):
  it=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));return it/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-it+1e-6)
 seen=set();accepted=[];reject=[]
 for r in raw:
  x,y,x2,y2=r['bbox_xyxy'];reason=None
  if min(x2-x,y2-y)<24 or min(x,y)<2 or x2>r['width']-2 or y2>r['height']-2:reason='small_or_clipped'
  elif r['global_box'] and any(a['source_scene_key']==r['source_scene_key'] and a['global_box'] and iou(a['global_box'],r['global_box'])>.6 for a in accepted):reason='same_instance_overlapping_tiles'
  if not reason:
   im=Image.open(r['image_path']).convert('RGB').crop(tuple(map(round,r['bbox_xyxy'])));h=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
   if h in seen:reason='duplicate_pixels'
   else:seen.add(h);r['crop_pixel_sha256']=h
  if reason:reject.append(dict(id=r['id'],reason=reason));continue
  r['training_eligible']=True;accepted.append(r)
 (ROOT/'source_jobs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in accepted));(ROOT/'helicopter_inventory_summary.json').write_text(json.dumps(dict(raw=dict(Counter(r['source_dataset'] for r in raw)),eligible=dict(Counter(r['source_dataset'] for r in accepted)),rejected=dict(Counter(r['reason'] for r in reject))),indent=2))
 (ROOT/'qa').mkdir(exist_ok=True);print((ROOT/'helicopter_inventory_summary.json').read_text(),flush=True)
if __name__=='__main__':main()
