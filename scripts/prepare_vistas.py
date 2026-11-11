import argparse,json,zipfile,collections
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);w=p.parse_args().workspace;o=w/'reports/vistas_preparation';o.mkdir(parents=True,exist_ok=True);dest=w/'data/Vistas/v2_selected';dest.mkdir(exist_ok=True)
targets=['bench','bike-rack','cctv-camera','fire-hydrant','junction-box','mailbox','parking-meter','phone-booth','street-light','trash-can','water-valve','catch-basin','manhole','sign--store','sign--advertisement','sign--information','support--utility-pole','support--pole']
stats={};mapping={};manifests={};invalid=[]
with zipfile.ZipFile(w/'data/Vistas/mapillary-vistas-dataset_public_v2.0.zip') as z:
 for split in ['training','validation']:
  c=json.loads(z.read(f'{split}/v2.0/panoptic/panoptic_2020.json'));cats={x['id']:x['name'] for x in c['categories']};images={x['id']:x for x in c['images']};assert len(images)==len(c['images']);bucket=collections.defaultdict(list);rows=[]
  for ann in c['annotations']:
   im=images[ann['image_id']]
   for seg in ann['segments_info']:
    name=cats[seg['category_id']]
    if name not in ['object--'+t for t in targets]:continue
    x,y,bw,bh=seg['bbox']
    if bw<=0 or bh<=0 or seg['area']<=0:
     invalid.append({'split':split,'image_id':ann['image_id'],'segment':seg});continue
    r={'image_id':ann['image_id'],'file_name':im['file_name'],'segment_id':seg['id'],'category':name,'category_id':seg['category_id'],'bbox':seg['bbox'],'area':seg['area'],'iscrowd':seg['iscrowd'],'width':im['width'],'height':im['height']};rows.append(r);bucket[name].append(r)
  stats[split]={}
  for name,rr in bucket.items():
   valid=[r for r in rr if not r['iscrowd']];scale=[r['area']/(r['width']*r['height']) for r in valid];short=[min(r['bbox'][2:])*1024/max(r['width'],r['height']) for r in valid]
   stats[split][name]={'segments':len(rr),'noncrowd':len(valid),'images':len({r['image_id'] for r in valid}),'median_relative_mask_area':float(np.median(scale)) if scale else None,'short_side_below16_at_long1024':float(np.mean(np.array(short)<16)) if short else None}
  (o/f'{split}_instances.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8');mapping[split]=cats
 (o/'statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf8');(o/'category_mapping.json').write_text(json.dumps(mapping,indent=2),encoding='utf8');print(json.dumps(stats['validation']),flush=True)
 (o/'invalid_segments.json').write_text(json.dumps(invalid,indent=2),encoding='utf8')
 selected=[i for i in z.infolist() if not i.is_dir() and (i.filename in ['README','LICENSE','config_v2.0.json','demo.py'] or any(i.filename.startswith(s+'/images/') or i.filename.startswith(s+'/v2.0/panoptic/') for s in ['training','validation']))]
 for j,i in enumerate(selected):
  target=(dest/i.filename).resolve();assert target.is_relative_to(dest.resolve())
  z.extract(i,dest)
  if (j+1)%5000==0:print('extracted',j+1,'/',len(selected),flush=True)
 (o/'extraction.json').write_text(json.dumps({'files':len(selected),'bytes':sum(i.file_size for i in selected),'destination':str(dest)},indent=2),encoding='utf8')
print('DONE',flush=True)


