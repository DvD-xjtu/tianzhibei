import json,shutil
from pathlib import Path
from collections import Counter
import numpy as np,cv2
from PIL import Image,ImageDraw
from produce_final_dataset import source_values
from filter_mar20_foregrounds import quality
ROOT=Path('/data3/tianzhibei/derived/aircraft_copypaste_final_20260928/extra/class_supplement_v1')
G=dict(min_score=.5,min_dominant=.90,max_secondary=.02,max_mask_area=.45,min_bbox_fill=.10)
allowed={'F-35':{1,6,14,20,25,27,105},'E-2':{172,178,181,183,184,186,193,208,209,213,225},'Helicopter':{228,232,236,251,253,263,264,275,277,279,282,285,286,296,300,301,302,303,305,308,309,310,311,312,320},'YF-12A':set()}
for cls in allowed:
 records=[json.loads(l) for l in (ROOT/'extraction'/f'decisions_{cls}.jsonl').read_text().splitlines()]
 pool=[];reviews=[]
 for r in records:
  keep=cls=='YF-12A' or int(r['id'].split('_')[-1]) in allowed[cls]
  if not keep:reviews.append(dict(id=r['id'],accepted=False,reason='not_selected_after_visual_review_or_wrong_xview_category_mapping'));continue
  if cls in ['F-35','YF-12A']:
   rgba=np.array(Image.open(ROOT/'refinement'/f'{r["id"]}_2.png'));meta=json.loads((ROOT/'refinement'/f'{r["id"]}.json').read_text());r['selected_score']=meta['scores'][2];r['refined_prompt']=meta
   Image.fromarray(rgba[:,:,:3]).save(r['source_path']);r['context_rect']=[0,0,rgba.shape[1],rgba.shape[0]]
  else:rgba=np.array(Image.open(r['cutout_path']))
  mask=rgba[:,:,3]>127
  n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
  mask=labels==(1+np.argmax(stats[1:,cv2.CC_STAT_AREA]))
  rgba[:,:,3]=mask.astype('uint8')*255
  Image.fromarray(rgba).save(r['cutout_path']);Image.fromarray(rgba[:,:,3]).save(r['mask_path'])
  q=quality(mask,r['selected_score'],G);metrics=source_values(r['cutout_path'],r['source_path']);r['source_metrics']=metrics
  if metrics['source_background_like_fraction']>.30:q['reasons'].append('source_background_leak')
  q['accepted']=not q['reasons'];r['quality']=q;r['visual_review']='selected after source and silhouette inspection; largest connected component retained';r['source_domain']=r.get('source_domain','remote_sensing_dataset')
  if q['accepted']:pool.append(r)
  reviews.append(dict(id=r['id'],accepted=q['accepted'],reasons=q['reasons']))
 (ROOT/'extraction'/f'pool_{cls}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pool))
 (ROOT/'qa'/f'visual_selection_{cls}.json').write_text(json.dumps(reviews,indent=2))
 canvas=Image.new('RGB',(1200,max(1,(len(pool)+5)//6)*180),(210,210,210));d=ImageDraw.Draw(canvas)
 for i,r in enumerate(pool):
  fg=Image.open(r['cutout_path']);fg.thumbnail((196,150));x=i%6*200;y=i//6*180;canvas.paste(fg,(x,y),fg);d.text((x,y+154),r['id'],fill='black')
 canvas.save(ROOT/'qa'/f'selected_{cls}.jpg');print(cls,len(pool),[(r['id'],r.get('reasons')) for r in reviews if r.get('reasons')])
