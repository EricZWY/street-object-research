from pathlib import Path
import json,numpy as np
from collections import defaultdict
from PIL import Image,ImageDraw
import argparse,hashlib,torch
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);R=p.parse_args().workspace
O=R/'reports/cvos_full';O.mkdir(parents=True,exist_ok=True)
def read(name):return [json.loads(s) for s in (R/'runs'/name/'predictions.jsonl').read_text().splitlines()]
def stat(rows):
 v=np.array([r['iou'] for r in rows]);return {'n':len(v),'mean_iou':float(v.mean()) if len(v) else None,'acc50':float((v>.5).mean()) if len(v) else None,'acc25':float((v>.25).mean()) if len(v) else None,'zero_iou':int((v==0).sum())}
v=read('cvos_svi_val');t=read('cvos_svi_test');c=read('cvos_svi_test_center')
assert len(v)==923 and len(t)==len(c)==973
assert all(a['id']==b['id'] for a,b in zip(t,c))
base=R/'data/CVOGL/CVOGL_SVI'
train=torch.load(base/'CVOGL_SVI_train.pth',weights_only=True)
qh={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (base/'query').glob('*.jpg')}
train_hashes={qh[r[1]] for r in train};di={r['index'] for r in t if qh[r['query']] in train_hashes}
res={'val':stat(v),'test':stat(t),'center_prompt':stat(c),'duplicate_query':stat([r for r in t if r['index'] in di]),'other_query':stat([r for r in t if r['index'] not in di]),'classes':{},'sizes':{}}
for cl in sorted({r['class'] for r in t}):res['classes'][cl]=stat([r for r in t if r['class']==cl])
def area(r):
 b=r['ground_truth'];return (b[2]-b[0])*(b[3]-b[1])/1024**2
for lo,hi,label in [(0,.01,'<1%'),(.01,.04,'1%-4%'),(.04,2,'>=4%')]:res['sizes'][label]=stat([r for r in t if lo<=area(r)<hi])
res['paired_prompt']={'actual_success_center_fail':sum(a['iou']>.5 and b['iou']<=.5 for a,b in zip(t,c)),'actual_fail_center_success':sum(a['iou']<=.5 and b['iou']>.5 for a,b in zip(t,c)),'mean_iou_delta':float(np.mean([a['iou']-b['iou'] for a,b in zip(t,c)]))}
# Observable error geometry, not causal labels.
fail=[r for r in t if r['iou']<=.5]
center_out=0;partial=0
for r in fail:
 b=r['ground_truth'];p=r['prediction'];x=(p[0]+p[2])/2;y=(p[1]+p[3])/2
 center_out+=not(b[0]<=x<=b[2] and b[1]<=y<=b[3]);partial+=r['iou']>0
res['failure_geometry']={'failures':len(fail),'predicted_center_outside_truth':int(center_out),'positive_overlap_but_fail':int(partial)}
res['timing']={'test_median_seconds':float(np.median([r['seconds'] for r in t]))}
(O/'analysis.json').write_text(json.dumps(res,indent=2),encoding='utf8')
# Deterministic examples: 3 zero-overlap by target size, 2 largest near misses, 1 success.
chosen=[]
for lo,hi in [(0,.01),(.01,.04),(.04,2)]:
 pool=[r for r in t if r['iou']==0 and lo<=area(r)<hi]
 if pool:chosen.append(pool[len(pool)//2])
chosen+=sorted([r for r in t if .25<r['iou']<=.5],key=lambda r:-r['iou'])[:2]
chosen+=sorted([r for r in t if r['iou']>.8],key=lambda r:r['index'])[:1]
import torch
data=torch.load(R/'data/CVOGL/CVOGL_SVI/CVOGL_SVI_test.pth',weights_only=True)
for j,r in enumerate(chosen):
 row=data[r['index']];q=Image.open(R/'data/CVOGL/CVOGL_SVI/query'/r['query']).convert('RGB');sat=Image.open(R/'data/CVOGL/CVOGL_SVI/satellite'/r['reference']).convert('RGB')
 x,y=row[4];ImageDraw.Draw(q).ellipse((x-5,y-5,x+5,y+5),fill='red')
 dr=ImageDraw.Draw(sat);dr.rectangle(r['ground_truth'],outline='lime',width=5);dr.rectangle(r['prediction'],outline='red',width=5)
 sat=sat.resize((512,512));panel=Image.new('RGB',(1024,550),'white');panel.paste(q,(0,100));panel.paste(sat,(512,35));ImageDraw.Draw(panel).text((8,8),f"test[{r['index']}] {r['class']} IoU={r['iou']:.4f}; green=GT red=prediction",fill='black');panel.save(O/f'case_{j}.jpg')
(O/'cases.json').write_text(json.dumps(chosen,indent=2),encoding='utf8');print(json.dumps(res,indent=2))

