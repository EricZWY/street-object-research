"""Fixed development pilot; no fitting, prompt tuning, or validation-set selection."""
import argparse,json,hashlib,time,collections
from pathlib import Path
import numpy as np
import torch,ultralytics
from ultralytics import YOLOWorld
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();w=a.workspace;o=w/'runs/vistas_detection_pilot';o.mkdir(parents=True,exist_ok=True)
config=json.loads((w/'reports/vistas_preparation/class_split_v1.json').read_text());cats=config['seen']+config['unseen'];prompts=['street light','electrical junction box','trash can','fire hydrant','manhole cover','storm drain','utility pole','store sign','surveillance camera','bench','mailbox','bicycle rack','information sign'];assert len(cats)==len(prompts)
rows=[json.loads(x) for x in (w/'reports/vistas_preparation/training_instances.jsonl').read_text().splitlines()];by=collections.defaultdict(list)
for r in rows:
 if r['category'] in cats and not r['iscrowd']:by[r['file_name']].append(r)
selected=set()
for c in cats:
 pool=[f for f,rs in by.items() if any(r['category']==c for r in rs)];selected.update(sorted(pool,key=lambda f:hashlib.sha256(('pilot-v1'+f).encode()).hexdigest())[:3])
selected=sorted(selected);manifest={'selection':'3 images per class by SHA256(pilot-v1+filename), union; training development only','images':selected,'categories':cats,'prompts':prompts,'imgsz':960,'conf':0.1,'nms_iou':0.7,'match_iou':0.5,'max_det':300,'precision':'FP32','batch':1,'ultralytics':ultralytics.__version__,'torch':torch.__version__}
(o/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.set_per_process_memory_fraction(.6)
f=w/'weights/YOLOWorld/yolov8s-worldv2.pt';model=YOLOWorld(str(f));model.set_classes(prompts);torch.cuda.reset_peak_memory_stats()
counts={c:{'gt':0,'pred':0,'tp':0} for c in cats};times=[]
def iou(a,b):
 lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:]);v=np.maximum(hi-lo,0).prod();return float(v/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-v+1e-9))
with (o/'predictions.jsonl').open('w',encoding='utf8') as out:
 for j,name in enumerate(selected):
  torch.cuda.synchronize();start=time.perf_counter()
  result=model.predict(str(w/'data/Vistas/v2_selected/training/images'/name),imgsz=960,conf=.1,iou=.7,max_det=300,device=0,half=False,verbose=False)[0]
  torch.cuda.synchronize();elapsed=time.perf_counter()-start;times.append(elapsed)
  boxes=result.boxes.xyxy.cpu().numpy();cls=result.boxes.cls.cpu().numpy().astype(int);scores=result.boxes.conf.cpu().numpy();assert np.isfinite(boxes).all()
  gt=[]
  for r in by[name]:
   x,y,bw,bh=r['bbox'];gt.append({'category':r['category'],'box':[x,y,x+bw,y+bh]})
  for k,c in enumerate(cats):
   g=[x['box'] for x in gt if x['category']==c];used=set();ids=sorted(np.where(cls==k)[0],key=lambda i:-scores[i]);tp=0
   for idx in ids:
    candidates=[(iou(boxes[idx],b),t) for t,b in enumerate(g) if t not in used]
    if candidates:
     v,t=max(candidates)
     if v>=.5:used.add(t);tp+=1
   counts[c]['gt']+=len(g);counts[c]['pred']+=len(ids);counts[c]['tp']+=tp
  record={'file_name':name,'seconds':elapsed,'gt':gt,'predictions':[{'category':cats[k],'box':b.tolist(),'score':float(s)} for b,k,s in zip(boxes,cls,scores)]};out.write(json.dumps(record)+'\n');out.flush()
  if j<4:result.save(filename=str(o/f'preview_{j}.jpg'))
  print(f'{j+1}/{len(selected)} predictions={len(boxes)}',flush=True)
for c,v in counts.items():v['recall']=v['tp']/v['gt'] if v['gt'] else None;v['precision']=v['tp']/v['pred'] if v['pred'] else None
summary={'n_images':len(selected),'classes':counts,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'median_end_to_end_seconds_excluding_first':float(np.median(times[1:])),'checkpoint_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'note':'development feasibility only; fixed-threshold greedy diagnostic, not COCO AP; invalid/crowd ignore-region handling not implemented'}
(o/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2))

