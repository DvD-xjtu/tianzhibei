import json
from pathlib import Path
from collections import Counter,defaultdict
from PIL import Image,ImageDraw
R=Path('/data3/tianzhibei/derived/aircraft_copypaste_final_20260928/extra/source_expansion')
rows=[json.loads(l) for p in (R/'extraction').glob('pool_*.jsonl') for l in p.read_text().splitlines()]
reject={f'exp_DOTAv2_{i}':'visual_incomplete_or_ground_attached' for i in [845,90345,34209,64606,113453,136639,136640,27202,27204,683,684,685,31565,31581,31590,35365,35366,56107,97877,97878,99512,100552,121482,145459,145460,156953,174530,202365,202369,219316,223092]}
for i in ['0227_2','0254_0','0751_0','3819_0','2457_0','2459_0','2455_0','0247_0','0249_0','0248_0','0248_1','0080_0','0080_1','0080_2']:reject['exp_SIMD_'+i]='visual_helipad_line_attached'
for i in [35723,131840,134299,233756,238806,238992,239257]:reject['exp_xView_'+str(i)]='visual_incomplete_or_ground_attached'
lookup={r['id']:r for r in rows}
for cl in json.loads((R/'qa'/'registration_dedup.json').read_text())['clusters']:
 usable=sorted([i for i in cl if i not in reject],key=lambda i:lookup[i]['selected_score'],reverse=True)
 for i in usable[1:]:reject[i]='registered_same_instance_as:'+usable[0]
accepted=[r for r in rows if r['id'] not in reject]
# Round-robin among scenes reduces dependence on dense storage yards.
groups=defaultdict(list)
for r in accepted:groups[r['source_scene_key']].append(r)
selected=[]
while groups and len(selected)<100:
 for key in list(groups):
  selected.append(groups[key].pop(0))
  if not groups[key]:del groups[key]
  if len(selected)==100:break
for r in selected:r['review_status']='visual_pass_and_registration_deduplicated';r['review_note']='Distinct image instances; similar stored helicopters are not augmented copies. Residual cross-dataset/time duplicates cannot be ruled out.'
(R/'selected').mkdir(exist_ok=True)
(R/'selected'/'pool_Helicopter.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
(R/'qa'/'visual_rejections.json').write_text(json.dumps(reject,indent=2))
summary=dict(candidate_crops=len(rows),post_review_available=len(accepted),selected=len(selected),selected_by_dataset=dict(Counter(r['source_dataset'] for r in selected)),rejected=len(reject),quota_scope='Helicopter only; other three classes remain below target',cross_dataset_duplicate_audit='not exhaustive',composites_generated=False)
(R/'selected'/'summary.json').write_text(json.dumps(summary,indent=2))
for start in range(0,len(selected),25):
 sheet=Image.new('RGB',(1250,900),(210,210,210));d=ImageDraw.Draw(sheet)
 for j,r in enumerate(selected[start:start+25]):
  im=Image.open(r['cutout_path']).convert('RGBA');box=im.getchannel('A').getbbox();im=im.crop(box);im.thumbnail((210,145));x=j%5*250;y=j//5*180;sheet.paste(im,(x+(250-im.width)//2,y),im);d.text((x+3,y+150),r['id'],fill='black')
 sheet.save(R/'selected'/f'Helicopter_{start//25+1}.jpg',quality=95)
print(json.dumps(summary,indent=2))
