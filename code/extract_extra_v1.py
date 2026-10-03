#!/usr/bin/env python3
import json, argparse
from pathlib import Path
from collections import Counter
import numpy as np
import cv2, torch
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from extract_mtarsi_foregrounds import choose_conservative_mask
from extract_mar20_foregrounds import crop_and_prompt
from filter_mar20_foregrounds import quality
from produce_final_dataset import source_values
ROOT=Path('/data3/tianzhibei/derived/aircraft_copypaste_final_20260928/extra/class_supplement_v1')
G=dict(min_score=.5,min_dominant=.90,max_secondary=.02,max_mask_area=.45,min_bbox_fill=.10)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--class-name');args=ap.parse_args()
 torch.set_num_threads(2);cv2.setNumThreads(1)
 model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml','/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/checkpoints/sam2.1_hiera_large.pt',device='cuda')
 predictor=SAM2ImagePredictor(model)
 jobs=[json.loads(l) for l in (ROOT/'source_jobs.jsonl').read_text().splitlines()]
 if args.class_name:jobs=[r for r in jobs if r['class_name']==args.class_name]
 out=ROOT/'extraction';out.mkdir(exist_ok=True)
 decisions=[];good=[]
 for n,r in enumerate(jobs):
  im=Image.open(r['image_path']).convert('RGB');w,h=im.size
  if 'bbox_xyxy' in r:
   x,y,x2,y2=r['bbox_xyxy'];poly=[x/w,y/h,x2/w,y/h,x2/w,y2/h,x/w,y2/h]
   im,box,rect=crop_and_prompt(im,poly,1.8)
  else:
   box=[.05*w,.05*h,.95*w,.95*h];rect=[0,0,w,h]
  a=np.asarray(im)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   predictor.set_image(a);masks,scores,_=predictor.predict(box=np.asarray(box,np.float32),multimask_output=True)
  mask,score,selection=choose_conservative_mask(masks,scores)
  q=quality(mask,score,G)
  paths={k:str(out/folder/(r['id']+'.png')) for k,folder in [('cutout_path','cutouts'),('source_path','contexts'),('mask_path','masks')]}
  for p in paths.values():Path(p).parent.mkdir(exist_ok=True)
  Image.fromarray(a).save(paths['source_path']);Image.fromarray(np.dstack([a,mask.astype('uint8')*255])).save(paths['cutout_path']);Image.fromarray(mask.astype('uint8')*255).save(paths['mask_path'])
  try:
   metrics=source_values(paths['cutout_path'],paths['source_path'])
   if metrics['source_background_like_fraction']>.30:q['reasons'].append('source_background_leak')
   if metrics['secondary_mask_fraction']>.02:q['reasons'].append('mask_fragments')
  except ValueError as e:metrics={};q['reasons'].append(str(e))
  q['accepted']=not q['reasons']
  rec=dict(r,**paths,source_crop_path=paths['source_path'],source_image_path=r['image_path'],source_metrics=metrics,quality=q,selected_score=score,prompt_box_xyxy=list(map(float,box)),context_rect=rect,selection=selection,source_domain=r.get('source_domain','remote_sensing_dataset'))
  decisions.append(rec)
  if q['accepted']:good.append(rec)
  if (n+1)%20==0:print(args.class_name,n+1,len(good),flush=True)
 (out/f'decisions_{args.class_name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in decisions))
 (out/f'pool_{args.class_name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in good))
 for page in range((len(good)+23)//24):
  batch=good[page*24:(page+1)*24];canvas=Image.new('RGB',(1200,((len(batch)+5)//6)*180),(210,210,210));draw=ImageDraw.Draw(canvas)
  for i,r in enumerate(batch):
   fg=Image.open(r['cutout_path']);fg.thumbnail((196,150));x=i%6*200;y=i//6*180;canvas.paste(fg,(x,y),fg);draw.text((x,y+154),r['id'],fill='black')
  canvas.save(ROOT/'qa'/f'cutouts_{args.class_name}_{page}.jpg')
 print(args.class_name,'DONE',len(good),len(decisions),Counter(x for r in decisions for x in r['quality']['reasons']),flush=True)
if __name__=='__main__':main()
