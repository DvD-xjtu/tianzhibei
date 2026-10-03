"""Identify repeated annotated instances using image registration, not aircraft shape."""
import json,itertools
from pathlib import Path
from collections import defaultdict,Counter
import cv2,numpy as np
R=Path('/data3/tianzhibei/derived/aircraft_copypaste_final_20260928/extra/source_expansion')
def main():
 rows=[json.loads(l) for p in (R/'extraction').glob('pool_*.jsonl') for l in p.read_text().splitlines()]
 groups=defaultdict(list)
 for r in rows:groups[r['source_scene_key'] if r['source_dataset']=='DOTAv2' else r['source_dataset']].append(r)
 sift=cv2.SIFT_create(nfeatures=2500); matcher=cv2.BFMatcher();links=[];cv2.setNumThreads(2)
 for gn,rs in groups.items():
  paths=sorted(set(r['image_path'] for r in rs));feat={}
  for p in paths:
   a=cv2.imread(p,0);k,d=sift.detectAndCompute(a,None);feat[p]=(np.float32([v.pt for v in k]),d)
  for p,q in itertools.combinations(paths,2):
   kp,dp=feat[p];kq,dq=feat[q]
   if dp is None or dq is None:continue
   good=[m for z in matcher.knnMatch(dp,dq,k=2) if len(z)==2 for m,n in [z] if m.distance<.65*n.distance]
   if len(good)<12:continue
   src=np.float32([kp[m.queryIdx] for m in good]);dst=np.float32([kq[m.trainIdx] for m in good])
   M,inlier=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=3000)
   if M is None or inlier.sum()<12 or inlier.mean()<.4:continue
   # Require spatially distributed evidence, avoiding alignment based on similar aircraft alone.
   pts=src[inlier.ravel().astype(bool)]
   if np.ptp(pts[:,0])<100 or np.ptp(pts[:,1])<100:continue
   for a in [r for r in rs if r['image_path']==p]:
    b=a['bbox_xyxy'];corn=np.array([[b[0],b[1],1],[b[2],b[1],1],[b[2],b[3],1],[b[0],b[3],1]])@M.T;bb=[*corn.min(0),*corn.max(0)]
    for c in [r for r in rs if r['image_path']==q]:
     bc=c['bbox_xyxy'];inter=max(0,min(bb[2],bc[2])-max(bb[0],bc[0]))*max(0,min(bb[3],bc[3])-max(bb[1],bc[1]));union=(bb[2]-bb[0])*(bb[3]-bb[1])+(bc[2]-bc[0])*(bc[3]-bc[1])-inter
     if inter/max(union,1)>.5:links.append(dict(a=a['id'],b=c['id'],iou=inter/union,inliers=int(inlier.sum()),matrix=M.tolist()))
  print(gn,len(paths),'links',len(links),flush=True)
 parent={r['id']:r['id'] for r in rows}
 def find(a):
  while parent[a]!=a:a=parent[a]
  return a
 for l in links:parent[find(l['b'])]=find(l['a'])
 clusters=defaultdict(list)
 for r in rows:clusters[find(r['id'])].append(r['id'])
 out=dict(method='SIFT background registration, affine RANSAC >=12 inliers and bbox IoU >0.5; exact pixels and xView global coordinates also filtered upstream',links=links,clusters=list(clusters.values()),counts=dict(raw=len(rows),registered_unique=len(clusters)))
 (R/'qa'/'registration_dedup.json').write_text(json.dumps(out,indent=2));print(out['counts'])
if __name__=='__main__':main()
