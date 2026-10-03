#!/usr/bin/env python3
"""Final v2: distinct raw crops, bounded local normal scale, sequential occupancy."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import time
import cv2
import numpy as np
from PIL import Image
from produce_final_dataset import FINAL_ROOT, CLASSES_13, SUBSETS, taxonomy, json_write, source_values
from synthesize_lae1m_v9 import load_lae_backgrounds
from synthesize_copypaste_v6 import (atomic_bytes, atomic_mask, box_overlap, foreground_premult,
    resize_premult, harmonize, rgb_to_lab, surface_reference, yolo_hbb_line, digest)
from filter_copypaste_v7 import GATES, rejection_reasons

VERSION = 'final-v2-local-normal-multi-instance-1'
POOL = FINAL_ROOT/'crop-metadata/training_pool_no_exact_heldout_overlap.jsonl'
DEFAULT_ROOT = FINAL_ROOT/'v2_preview_20261001'
RULES = dict(requested_count=[1, 'min(6, estimated_capacity)'], capacity_packings=3,
    capacity_trials_per_slot=8, max_trials_per_slot=24, image_attempt_factor=40,
    edge_margin_px=8, clearance_px=8, min_hbb_side_px=16, min_mask_pixels=100,
    max_patch_side_fraction=1/3, max_upscale=2.0, max_added_hbb_area_fraction=.20,
    local_size_window=[.6, 1.6], local_min_n=5, cv_bounds=[.08, .35],
    subset_size_quantiles=[.05, .95], rotation_deg=[0, 360], lightness_gain=[.90, 1.10],
    lightness_offset=[-3, 3], max_new_lightness_clipping_fraction=.02,
    feather_sigma=[.50, .80], jpeg_quality=95)


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    provenance={}
    for name in ('produce_final_v2.py','verify_final_v2.py','produce_final_dataset.py',
                 'synthesize_copypaste_v6.py','synthesize_lae1m_v9.py','filter_copypaste_v7.py',
                 'augment_foregrounds_to_class_quota.py'):
        p=Path(__file__).parent/name
        atomic_bytes(root/'code_snapshot'/name,p.read_bytes()); provenance[name]=digest(p)
    good, bad = [], []
    for line in POOL.read_text().splitlines():
        r = json.loads(line)
        assert r['training_eligible'] and r['source'] in ('fixed', 'innar_train', 'mar20_train')
        r['source_path'] = r['source_crop_path']
        try:
            v = source_values(r['cutout_path'], r['source_path'])
            reasons = []
            if v['source_background_like_fraction'] > GATES['max_source_background_like_fraction']:
                reasons.append('source_background_leak')
            if v['secondary_mask_fraction'] > GATES['max_secondary_mask_fraction']:
                reasons.append('mask_fragments')
            r['source_metrics'] = v
        except (ValueError, OSError) as e:
            reasons = [str(e)]
        if reasons:
            bad.append({'id': r['id'], 'class_name': r['class_name'], 'reasons': reasons})
        else:
            good.append(r)
    for name, rows in [('foreground_pool', good), ('foreground_rejections', bad)]:
        atomic_bytes(root/(name+'.jsonl'), ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows).encode())
    json_write(root/'taxonomy.json', taxonomy())
    json_write(root/'config.json', dict(version=VERSION, rules=RULES, quality_gate=GATES,
        foreground_manifest=str(POOL), foreground_manifest_sha256=digest(POOL),
        usable_foregrounds=dict(Counter(r['class_name'] for r in good)),
        source_precheck_rejected=len(bad), visual_review='preview pending user review',
        scale_definition='sqrt(original aircraft HBB area); preserve crop aspect ratio',
        code_sha256=provenance,
        count_policy='estimate geometric/surface capacity first; uniform 1..capacity; bounded retries; no forced filling'))
    print('source precheck', len(good), 'usable', flush=True)


def scale_model(boxes, w, h, prior):
    values = np.sqrt([(b[2]-b[0])*(b[3]-b[1]) for b in boxes])
    # Trim extreme local values only when enough original aircraft exist.
    fit = values
    if len(values) >= 10:
        q = np.quantile(values, [.1, .9]); fit = values[(values >= q[0]) & (values <= q[1])]
    mu = float(np.mean(fit))
    cv = float(np.std(fit, ddof=1)/mu) if len(fit) >= 5 else prior['cv']
    cv = float(np.clip(cv, *RULES['cv_bounds']))
    lo = max(16., .6*mu, prior['q05']*math.sqrt(w*h))
    hi = min(1.6*mu, prior['q95']*math.sqrt(w*h), min(w, h)/3)
    return dict(n=len(values), fit_n=len(fit), mean=mu, std=mu*cv,
        lower=lo, upper=hi, variance_source='local' if len(fit) >= 5 else 'subset_cv')


def sample_size(model, rng):
    if model['lower'] > model['upper']:
        return None
    for _ in range(64):
        s = rng.gauss(model['mean'], model['std'])
        if model['lower'] <= s <= model['upper']:
            return s
    return None


def placement(before, lab, ref, occupied, hosts, size, rng):
    """Fresh candidates for each attempt; full patch clearance protects alpha feather."""
    h, w = before.shape[:2]; fw, fh = size
    candidates = set()
    for host in sorted(hosts, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)[:8]:
        cx, cy = (host[0]+host[2])/2, (host[1]+host[3])/2
        side = max(host[2]-host[0], host[3]-host[1], fw, fh)
        for radius in (1.4, 1.9, 2.4):
            phase = rng.uniform(0, 30)
            for degrees in range(0, 360, 30):
                a = math.radians(degrees+phase)
                candidates.add((round(cx+side*radius*math.cos(a)-fw/2),
                                round(cy+side*radius*math.sin(a)-fh/2)))
    candidates = sorted(candidates); rng.shuffle(candidates)
    for x, y in candidates:
        rect = (x, y, x+fw, y+fh)
        if x < 8 or y < 8 or x+fw > w-8 or y+fh > h-8: continue
        if any(box_overlap(rect, b, 8) for b in occupied): continue
        patch = lab[y:y+fh, x:x+fw]
        texture = float(patch[:, :, 0].std())
        if texture > GATES['max_placement_texture_std_lab']: continue
        color = float(np.linalg.norm(np.median(patch.reshape(-1, 3), axis=0)-ref))
        if color > GATES['max_placement_color_distance_lab']: continue
        rgb = before[y:y+fh, x:x+fw]; r, g, b = rgb.astype(float).transpose(2, 0, 1)
        vegetation = float(((g > r*1.08) & (g > b*1.05) & (g-r > 5)).mean())
        if vegetation > GATES['max_placement_vegetation_fraction']: continue
        edges = float((cv2.Canny(rgb, 70, 150) > 0).mean())
        if edges > GATES['max_placement_edge_fraction']: continue
        return x, y, dict(texture_std_lab=texture, color_distance_lab=color,
                          vegetation_fraction=vegetation, edge_fraction=edges)
    return None


def output_metrics(before, after, hard, source, pm):
    bg, fg = before[hard], after[hard]
    bl = cv2.cvtColor(bg.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB)
    fl = cv2.cvtColor(fg.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB)
    bs = float(np.median(np.ptp(bg.astype(float), axis=1)))
    fs = float(np.median(np.ptp(fg.astype(float), axis=1)))
    return dict(source, placement_texture_std_lab=pm['texture_std_lab'],
        placement_vegetation_fraction=pm['vegetation_fraction'], placement_edge_fraction=pm['edge_fraction'],
        placement_color_distance_lab=pm['color_distance_lab'],
        paste_lightness_delta=float(np.median(fl[:, :, 0]))-float(np.median(bl[:, :, 0])),
        paste_saturation_delta=fs-bs, background_saturation=bs, paste_saturation=fs)


def estimate_capacity(before, lab, ref, bg, model, templates, rng):
    """Conservative feasible greedy packing, not an exact maximum or a quality guarantee.

    Use actual rotated source shapes and the same scale/surface/collision limits as paste.
    Keep the best of three packings, capped at six. No original or synthetic GT enters fitting.
    """
    h,w = before.shape[:2]; best=[]
    for _ in range(RULES['capacity_packings']):
        occupied=list(bg['boxes']); packed=[]; area=0
        for slot in range(6):
            found=False
            for _ in range(RULES['capacity_trials_per_slot']):
                t=rng.choice(templates); target=sample_size(model,rng)
                if target is None: continue
                scale=target/t['linear']; size=(round(t['w']*scale),round(t['h']*scale))
                if scale > 2 or min(size)<16 or max(size)>min(w,h)/3: continue
                if min(t['bw'],t['bh'])*scale<16 or t['pixels']*scale*scale<100: continue
                if area+target*target>.20*w*h: continue
                pos=placement(before,lab,ref,occupied,bg['plane_boxes'],size,rng)
                if pos is None: continue
                x,y,_=pos; rect=[x,y,x+size[0],y+size[1]]
                occupied.append(rect); packed.append(rect); area+=target*target; found=True; break
            if not found: break
        if len(packed)>len(best): best=packed
        if len(best)==6: break
    return len(best),best


def encode(image):
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok: raise RuntimeError('JPEG encode failed')
    decoded = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return buf.tobytes(), decoded


def run_subset(args):
    ds, root, count, seed = args; root = Path(root)
    cv2.setNumThreads(1); rng = random.Random(seed)
    groups = defaultdict(list)
    for line in (root/'foreground_pool.jsonl').read_text().splitlines():
        r = json.loads(line); groups[r['class_name']].append(r)
    classes = sorted(groups)
    # Shape-only templates from the frozen training pool, balanced over classes.
    templates=[]
    for cls in classes:
        for item in rng.sample(groups[cls],min(8,len(groups[cls]))):
            pmul,a,_=foreground_premult(item,rng.uniform(0,360))
            yy,xx=np.where(a>.5); bw=int(xx.max()-xx.min()+1); bh=int(yy.max()-yy.min()+1)
            templates.append(dict(w=pmul.shape[1],h=pmul.shape[0],bw=bw,bh=bh,
                pixels=int((a>.5).sum()),linear=math.sqrt(bw*bh)))
    names = json.loads((root/'taxonomy.json').read_text()); name_ids = {n:i for i,n in enumerate(names)}
    backgrounds, ratios, manifest_path = load_lae_backgrounds(ds)
    sizes = np.sqrt(ratios); trimmed = sizes[(sizes >= np.quantile(sizes, .1)) & (sizes <= np.quantile(sizes, .9))]
    prior = dict(q05=float(np.quantile(sizes, .05)), q95=float(np.quantile(sizes, .95)),
        cv=float(np.std(trimmed)/np.mean(trimmed)))
    out = root/'shards'/ds; out.mkdir(parents=True)
    json_write(out/'backgrounds.json', dict(manifest=manifest_path, sha256=digest(Path(manifest_path)),
        eligible_ids=[b['id'] for b in backgrounds], prior=prior, seed=seed))
    rng.shuffle(backgrounds)
    used = Counter(); counts = Counter(); failures = Counter(); requests = Counter(); actual = Counter(); capacities=Counter()
    attempts = 0; passed = 0; start = time.monotonic()
    def progress(done=False):
        json_write(out/'progress.json', dict(completed=done, images=passed, goal=count, attempts=attempts,
            requested_all_attempts=dict(requests), estimated_capacities=dict(capacities), actual_counts=dict(actual), instances_by_class=dict(counts),
            failures=dict(failures), seconds=round(time.monotonic()-start, 1)))
    with (out/'manifest.jsonl').open('w', buffering=1) as handle:
        while passed < count and attempts < count*RULES['image_attempt_factor']:
            bg = backgrounds[attempts % len(backgrounds)]
            if attempts and attempts % len(backgrounds) == 0: rng.shuffle(backgrounds)
            attempts += 1
            before = np.asarray(Image.open(bg['image']).convert('RGB')); h,w = before.shape[:2]
            model = scale_model(bg['plane_boxes'], w, h, prior)
            if model['lower'] > model['upper']:
                failures['empty_scale_interval'] += 1; continue
            lab = rgb_to_lab(before); ref = surface_reference(before, bg['plane_boxes'])[0]
            capacity,capacity_boxes=estimate_capacity(before,lab,ref,bg,model,templates,rng)
            capacities[capacity]+=1
            if not capacity:
                failures['zero_capacity']+=1
                if attempts%10==0: progress()
                continue
            requested=rng.randint(1,capacity); requests[requested]+=1
            occupied = list(bg['boxes']); composed = before.copy(); instances = []; selected = set(); area_used = 0
            for slot in range(requested):
                accepted = False
                for trial in range(RULES['max_trials_per_slot']):
                    cls = rng.choices(classes, weights=[1/math.sqrt(counts[c]+10) for c in classes])[0]
                    options = [r for r in rng.sample(groups[cls], min(8, len(groups[cls]))) if r['id'] not in selected]
                    if not options: failures['no_distinct_crop'] += 1; continue
                    item = min(options, key=lambda r: used[r['id']])
                    angle = rng.uniform(0, 360)
                    pmul, alpha, context = foreground_premult(item, angle)
                    ys,xs = np.where(alpha > .5)
                    source_size = math.sqrt((xs.max()-xs.min()+1)*(ys.max()-ys.min()+1))
                    target = sample_size(model, rng)
                    if target is None: failures['normal_draw_failed'] += 1; continue
                    scale = target/source_size
                    size = (round(pmul.shape[1]*scale), round(pmul.shape[0]*scale))
                    if scale > 2 or min(size) < 16 or max(size) > min(w,h)/3:
                        failures['scale_bounds'] += 1; continue
                    fg, mask = resize_premult(pmul, alpha, size); hard = mask > .5
                    bx,by,bw,bh = cv2.boundingRect(cv2.findNonZero(hard.astype('uint8')))
                    effective = math.sqrt(bw*bh)
                    if min(bw,bh) < 16 or hard.sum() < 100 or not model['lower'] <= effective <= model['upper']:
                        failures['effective_size_bounds'] += 1; continue
                    if area_used+bw*bh > .20*w*h:
                        failures['image_area_budget'] += 1; continue
                    pos = placement(before, lab, ref, occupied, bg['plane_boxes'], size, rng)
                    if pos is None: failures['no_safe_surface_position'] += 1; continue
                    x,y,pm = pos; roi = composed[y:y+size[1], x:x+size[0]]
                    fg, _, blend = harmonize(fg, mask, context, roi)
                    flab = rgb_to_lab(fg); gain = rng.uniform(.9, 1.1); offset = rng.uniform(-3, 3)
                    original_l = flab[:, :, 0].copy(); new_l = original_l*gain+offset
                    clipping = float((((new_l < 0) | (new_l > 100)) & ((original_l > 0) & (original_l < 100)))[hard].mean())
                    if clipping > .02: failures['brightness_clipping'] += 1; continue
                    flab[:, :, 0] = np.clip(new_l, 0, 100)
                    fg = np.clip(cv2.cvtColor(flab, cv2.COLOR_LAB2RGB)*255, 0, 255)
                    feather = rng.uniform(.5, .8); soft = np.clip(cv2.GaussianBlur(mask.astype('float32'), (0,0), feather), 0, 1)
                    patch = np.clip(fg*soft[:,:,None]+roi*(1-soft[:,:,None]), 0, 255).astype('uint8')
                    value = output_metrics(before[y:y+size[1], x:x+size[0]], patch, hard, item['source_metrics'], pm)
                    reasons = rejection_reasons(value)
                    if reasons: failures.update(reasons); continue
                    composed[y:y+size[1], x:x+size[0]] = patch
                    box = [x+bx,y+by,x+bx+bw,y+by+bh]; rect = [x,y,x+size[0],y+size[1]]
                    blend.update(lightness_gain=gain, lightness_offset=offset, new_clipping_fraction=clipping, feather_sigma=feather)
                    instances.append(dict(class_name=cls, target_class_id=CLASSES_13.index(cls),
                        foreground_id=item['id'], foreground_source=item['source'], foreground_path=item['cutout_path'],
                        foreground_source_path=item['source_path'], source_image_path=item['source_image_path'],
                        rotation_deg=angle, scale=scale, sampled_linear_size=target, actual_linear_size=effective,
                        bbox_xyxy=box, patch_xyxy=rect, placement=pm, blend=blend,
                        source_metrics=item['source_metrics'], _hard=hard, _patch=patch, slot=slot, trials=trial+1))
                    occupied.append(rect); selected.add(item['id']); area_used += bw*bh; accepted = True; break
                if not accepted: failures['slot_retry_budget_exhausted'] += 1
            if not instances:
                failures['zero_instances'] += 1
                if attempts % 10 == 0: progress()
                continue
            # Recheck every instance in the actual final JPEG; remove failures and encode again.
            while instances:
                encoded, decoded = encode(composed); survivors = []
                for inst in instances:
                    x,y,x2,y2 = inst['patch_xyxy']
                    v = output_metrics(before[y:y2,x:x2], decoded[y:y2,x:x2], inst['_hard'], inst['source_metrics'], inst['placement'])
                    reasons = rejection_reasons(v)
                    if reasons: failures.update('final_jpeg:'+r for r in reasons)
                    else: inst['automatic_metrics'] = v; survivors.append(inst)
                if len(survivors) == len(instances): break
                instances = survivors; composed = before.copy()
                for inst in instances:
                    x,y,x2,y2 = inst['patch_xyxy']; composed[y:y2,x:x2] = inst['_patch']
            if not instances: failures['zero_after_jpeg'] += 1; continue
            passed += 1; stem = f'v2__{ds}__{passed:04d}'
            ip = out/'train/images'/f'{stem}.jpg'; lp = out/'train/labels'/f'{stem}.txt'
            labels = []; mask_ids = np.zeros((h,w), np.uint8)
            for i, inst in enumerate(instances, 1):
                x,y,x2,y2 = inst['patch_xyxy']; mask_ids[y:y2,x:x2][inst.pop('_hard')] = i
                inst.pop('_patch'); inst['mask_id'] = i
                labels.append(yolo_hbb_line(inst['target_class_id'], inst['bbox_xyxy'], w, h))
                counts[inst['class_name']] += 1; used[inst['foreground_id']] += 1
            display_labels = '\n'.join(labels)+'\n'
            for ann in bg['annotations']:
                x,y,ww,hh = ann['bbox_xywh']; b = (max(0,x),max(0,y),min(w,x+ww),min(h,y+hh))
                if b[2]>b[0] and b[3]>b[1]: labels.append(yolo_hbb_line(name_ids['LAE/'+ann['category_name']], b,w,h))
            lb = ('\n'.join(labels)+'\n').encode(); mp = out/'instance_masks'/f'{stem}.png'
            atomic_bytes(ip, encoded); atomic_bytes(lp, lb); atomic_mask(mp, mask_ids)
            atomic_bytes(out/'preview_labels'/f'{stem}.txt', display_labels.encode())
            actual[len(instances)] += 1
            row = dict(id=stem, pipeline_version=VERSION, subdataset=ds, seed=seed, background_attempt=attempts,
                source_background_id=bg['id'], background_image_path=str(bg['image']), background_label_path=str(bg['label']),
                background_split='train', width=w, height=h, image_path=str(ip), label_path=str(lp),
                instance_mask_path=str(mp), image_sha256=hashlib.sha256(encoded).hexdigest(),
                label_sha256=hashlib.sha256(lb).hexdigest(), estimated_capacity=capacity, capacity_probe_boxes=capacity_boxes,
                requested_count=requested, actual_count=len(instances),
                count_shortfall=requested-len(instances), scale_model=model, instances=instances,
                original_annotations=bg['annotations'], quality_gate_passed=True, visual_review_performed=False,
                status='preview_pending_review')
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
            if passed % 10 == 0: progress(); print(ds, passed, '/', count, 'attempts', attempts, flush=True)
        progress(passed == count)
    if passed != count: raise RuntimeError(f'{ds}: only {passed}/{count} images within attempt budget')
    return ds, passed


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--count', type=int, default=1000); ap.add_argument('--seed', type=int, default=20261001)
    ap.add_argument('--workers', type=int, default=6); args=ap.parse_args()
    if args.count < 6: ap.error('count must be at least six')
    cv2.setNumThreads(1); prepare(args.root)
    cfg=json.loads((args.root/'config.json').read_text()); cfg.update(seed=args.seed, requested_images=args.count)
    json_write(args.root/'config.json',cfg)
    tasks=[(ds,str(args.root),args.count//6+int(i<args.count%6),args.seed+i) for i,ds in enumerate(SUBSETS)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for result in ex.map(run_subset,tasks): print(result,flush=True)


if __name__ == '__main__': main()
