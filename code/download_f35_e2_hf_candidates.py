import requests,json,hashlib
from pathlib import Path
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path('/data3/tianzhibei/datasets/external_aircraft_supplement/Illia56_Military-Aircraft-Detection')
base='https://huggingface.co/datasets/Illia56/Military-Aircraft-Detection/resolve/main/'
rows=[]
for cls in ['F35','E2']:
 u='https://huggingface.co/api/datasets/Illia56/Military-Aircraft-Detection/tree/main/crop/'+cls+'?recursive=true&expand=false';r=requests.get(u,timeout=60);r.raise_for_status();a=r.json();print(cls,len(a),flush=True)
 for x in a:rows.append((cls,x))
def get(v):
 cls,a=v;rel=a['path'];p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists() or p.stat().st_size!=a['size']:
  r=requests.get(base+quote(rel,safe='/'),timeout=120);r.raise_for_status();p.write_bytes(r.content)
 h=hashlib.sha256(p.read_bytes()).hexdigest();return dict(class_name=cls,path=str(p),source_path=rel,size=p.stat().st_size,sha256=h,valid_size=p.stat().st_size==a['size'],source_url=base+quote(rel,safe='/'),source_domain='unverified_photography')
out=[]
with ThreadPoolExecutor(max_workers=12) as ex:
 for future in as_completed([ex.submit(get,v) for v in rows]):out.append(future.result())
(ROOT/'candidate_manifest.json').write_text(json.dumps(out,indent=2));print('DONE',len(out),sum(x['valid_size'] for x in out),flush=True)
