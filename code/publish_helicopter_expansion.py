import json,shutil
from pathlib import Path
from datetime import datetime
from PIL import Image
R=Path('/data3/tianzhibei/derived/aircraft_copypaste_final_20260928/extra/source_expansion')
P=R/'preview_Helicopter';rows=[json.loads(l) for l in (P/'shards/DOTAv2/manifest.jsonl').read_text().splitlines()];pool=[json.loads(l) for l in (R/'selected/pool_Helicopter.jsonl').read_text().splitlines()];names=json.loads((P/'taxonomy.json').read_text())
for r in rows:
 for folder,key in [('images','image_path'),('labels','label_path'),('instance_masks','instance_mask_path')]:
  src=Path(r[key]);dst=P/'train'/folder/src.name;dst.parent.mkdir(parents=True,exist_ok=True)
  if not dst.exists():dst.symlink_to(src)
 src=Path(r['image_path']);dst=P/'dashboard/images'/src.name;dst.parent.mkdir(parents=True,exist_ok=True)
 if not dst.exists():dst.symlink_to(src)
 dst=P/'dashboard/labels'/f'{src.stem}.txt';dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text('\n'.join(Path(r['label_path']).read_text().splitlines()[:r['actual_count']])+'\n')
(P/'data.yaml').write_text('path: '+str(P)+'\ntrain: train/images\nnames: '+json.dumps(names)+'\n')
for r in pool:
 im=Image.open(r['cutout_path']).convert('RGBA');bg=Image.new('RGB',im.size,(210,210,210));bg.paste(im,(0,0),im);dst=R/'crop_dashboard/Helicopter'/f'{r["id"]}.png';dst.parent.mkdir(parents=True,exist_ok=True);bg.save(dst)
entries=[dict(id='extra-v1-helicopter-expanded-crops',name='类别补充：v1 · 直升机扩充100个crop',group='extra',root=str(R/'crop_dashboard'),format='classification',training=False,note='DOTA73 + SIMD20 + xView7；已视觉筛选与同源背景配准去重，跨来源重复审计非穷尽。'),dict(id='extra-v1-helicopter-expanded-preview',name='类别补充：v1 · 直升机扩充预览100张',group='extra',root=str(P/'dashboard'),format='yolo',names=names,training=False,splits=[dict(name='Helicopter',images='images',labels='labels',label_prefix='')],note='215新增实例，使用99/100个crop。框仅展板绘制。其他三类仍未达到每类100个原始crop。')]
c=Path('/data2/tianzhibei/dataset-viewer/datasets.json');cfg=json.loads(c.read_text());shutil.copy2(c,c.with_name('datasets.json.backup-heli-expansion-'+datetime.now().strftime('%Y%m%d-%H%M%S')));cfg['datasets']=[d for d in cfg['datasets'] if d['id'] not in {e['id'] for e in entries}]+entries;c.write_text(json.dumps(cfg,ensure_ascii=False,indent=2));(R/'dashboard_registration.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2))
(R/'README.txt').write_text('类别补充 v1：原始crop扩充，阶段产物，四类目标尚未全部完成。\n直升机筛选池100个：DOTA73、外部SIMD20、修正类别映射后的xView7。原始train来源；SAM自动筛选、视觉审核、精确像素及同源背景配准去重。跨数据集/跨时间重复尚非穷尽审核。\n预览100张、215实例、实际使用99个crop，检查通过；原始图不含框，GT标签单独保存。展板在extra分组中。\nF-35、E-2、YF-12A仍未获得接近100个独立可用crop；既有400张四类预览不能充当原始来源数量。\nMilitary-RSOD按用户指示暂缓，继续其他来源。OPT-Aircraft已下载，799个螺旋桨类文件已筛查，e321仅列为E-2候选，未计入合格池。\n')
print('Registered 2 entries; reload pending')
