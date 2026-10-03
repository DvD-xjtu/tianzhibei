import csv,json
from pathlib import Path
from collections import defaultdict
from PIL import Image,ImageDraw
R=Path('/data3/tianzhibei/datasets/RarePlanes');O=Path('/data3/tianzhibei/derived/aircraft_copypaste_final_20260928/extra/source_expansion/rareplanes_screening');O.mkdir(parents=True,exist_ok=True)
meta={r['loc_id']:r for r in csv.DictReader(open('/tmp/RarePlanes_Public_Metadata.csv'))}
for target,name in [(3,'fighter'),(5,'transport_awac')]:
 rows=[]
 for p in sorted((R/'labels').glob('*.txt')):
  ip=R/'extracted/PS-RGB_tiled'/f'{p.stem}.png'
  if not ip.exists():continue
  for n,l in enumerate(p.read_text().splitlines()):
   parts=l.split()
   if not parts or int(parts[0])!=target:continue
   x,y,w,h=map(float,parts[1:5]);rows.append(dict(tile=p.stem,instance=n,loc_id=p.stem.split('_')[0],xywh=[x,y,w,h],image=str(ip)))
 (O/f'{name}_candidates.json').write_text(json.dumps(rows,indent=2))
 for start in range(0,len(rows),48):
  sheet=Image.new('RGB',(1200,960),'white');d=ImageDraw.Draw(sheet)
  for i,row in enumerate(rows[start:start+48]):
   im=Image.open(row['image']).convert('RGB');x,y,w,h=row['xywh'];W,H=im.size;cx,cy=x*W,y*H;bw,bh=w*W,h*H;pad=max(bw,bh)*.4;box=(max(0,int(cx-bw/2-pad)),max(0,int(cy-bh/2-pad)),min(W,int(cx+bw/2+pad)),min(H,int(cy+bh/2+pad)))
   cut=im.crop(box);cut.thumbnail((195,130));xx=i%6*200;yy=i//6*120;sheet.paste(cut,(xx+(200-cut.width)//2,yy));d.text((xx+2,yy+102),f'{start+i} L{row["loc_id"]} {meta.get(row["loc_id"],{}).get("Air_Field","")[:12]}',fill='black')
  sheet.save(O/f'{name}_{start//48:02}.jpg',quality=90)
 print(name,len(rows),'sheets',(len(rows)+47)//48)
