import json
from pathlib import Path
import numpy as np,cv2,torch
from PIL import Image,ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
ROOT=Path('/data3/tianzhibei/derived/aircraft_copypaste_final_20260928/extra/class_supplement_v1')
torch.set_num_threads(2)
p=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml','/data3/tianzhibei/derived/lae1m_mtarsi_copypaste_v1/checkpoints/sam2.1_hiera_large.pt',device='cuda'))
for cls,ids in [('F-35',[1,2,6,14,20,25,27,105]),('YF-12A',[])]:
 jobs=[json.loads(l) for l in (ROOT/'source_jobs.jsonl').read_text().splitlines() if json.loads(l)['class_name']==cls]
 for r in jobs:
  if cls=='F-35' and int(r['id'].split('_')[-1]) not in ids:continue
  im=np.array(Image.open(r['image_path']).convert('RGB'));h,w=im.shape[:2]
  if cls=='F-35':
   pts=np.array([[.5*w,.5*h],[.1*w,.1*h],[.9*w,.1*h],[.1*w,.9*h],[.9*w,.9*h]],np.float32);lab=np.array([1,0,0,0,0]);box=np.array([.15*w,.15*h,.85*w,.85*h],np.float32)
  else:
   pts=np.array([[410,420],[300,160],[220,145],[480,160],[150,225],[205,290],[530,255],[375,620],[440,705]],np.float32);lab=np.array([1,1,1,1,0,0,0,0,0]);box=np.array([100,8,582,740],np.float32)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   p.set_image(im);m,s,_=p.predict(box=box,point_coords=pts,point_labels=lab,multimask_output=True)
  out=ROOT/'refinement';out.mkdir(exist_ok=True)
  panels=[]
  for i,mask in enumerate(m):
   rgba=Image.fromarray(np.dstack([im,mask.astype('uint8')*255]));rgba.save(out/f'{r["id"]}_{i}.png');rgba.thumbnail((320,400));can=Image.new('RGB',(330,430),(210,210,210));can.paste(rgba,(0,0),rgba);ImageDraw.Draw(can).text((0,405),f'{i} score={s[i]:.3f}',fill='black');panels.append(can)
  sheet=Image.new('RGB',(990,430),'white')
  for i,can in enumerate(panels):sheet.paste(can,(330*i,0))
  sheet.save(out/f'{r["id"]}.jpg')
  (out/f'{r["id"]}.json').write_text(json.dumps(dict(scores=s.tolist(),points=pts.tolist(),labels=lab.tolist(),box=box.tolist())))
  print(r['id'],s,flush=True)
