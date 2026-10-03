#!/usr/bin/env python3
import json,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from produce_extra_engine import run_subset, CLASSES_13, RULES, GATES, VERSION
from produce_final_dataset import taxonomy,json_write
ROOT=Path('/data3/tianzhibei/derived/aircraft_copypaste_final_20260928/extra/class_supplement_v1')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--classes',nargs='+',default=['F-35','E-2','Helicopter','YF-12A']);ap.add_argument('--count',type=int,default=100);args=ap.parse_args()
 names=CLASSES_13+[n for n in taxonomy()[13:] if n not in ['LAE/Helicopter','LAE/helicopter']]
 tasks=[]
 for c in args.classes:
  root=ROOT/'batches'/c;root.mkdir(parents=True,exist_ok=True)
  pool=(ROOT/'extraction'/f'pool_{c}.jsonl').read_text();assert pool.strip(),c
  (root/'foreground_pool.jsonl').write_text(pool);json_write(root/'taxonomy.json',names)
  json_write(root/'config.json',dict(version=VERSION,rules=RULES,quality_gate=GATES,source_class=c,requested_images=args.count))
  datasets=['DOTAv2'] if c=='Helicopter' else ['DIOR','DOTAv2','FAIR1M','NWPU','RSOD']
  for i,ds in enumerate(datasets):
   if (root/'shards'/ds).exists():continue
   tasks.append((ds,str(root),args.count//len(datasets)+int(i<args.count%len(datasets)),20261001+100*CLASSES_13.index(c)+i))
 with ProcessPoolExecutor(max_workers=6) as ex:
  for result in ex.map(run_subset,tasks):print(result,flush=True)
if __name__=='__main__':main()
