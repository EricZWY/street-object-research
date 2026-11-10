from pathlib import Path
import json,torch,hashlib,itertools
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw
import argparse
parser=argparse.ArgumentParser(description='Audit local CVOGL SVI data and fixed-box baseline')
parser.add_argument('--workspace',type=Path,required=True,help='Contains data/CVOGL; outputs audit_20261110')
root=parser.parse_args().workspace; out=root/'audit_20261110';out.mkdir(exist_ok=True)
base=root/'data/CVOGL/CVOGL_SVI'
D={s:torch.load(base/f'CVOGL_SVI_{s}.pth',weights_only=True) for s in ['train','val','test']}
res={'splits':{},'overlaps':{},'image_errors':[]}; sizes={}; hashes={}
for folder in ['query','satellite']:
 for p in (base/folder).glob('*'):
  if not p.is_file():continue
  try:
   with Image.open(p) as im: im.load();sizes[p.name]=im.size
   hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
  except Exception as e: res['image_errors'].append([p.name,str(e)])
res['image_count']=len(sizes);res['dimensions']=dict(Counter(str(v) for v in sizes.values()))
for s,d in D.items():
 bad=[];area=[];clickbad=[];missing=[]
 for i,r in enumerate(d):
  q,t=r[1:3]; b=np.array(r[5],dtype=float)
  if q not in sizes or t not in sizes:missing.append(i);continue
  w,h=sizes[t];a=(b[2]-b[0])*(b[3]-b[1]);area.append(a/w/h)
  if not np.isfinite(b).all() or b[2]<=b[0] or b[3]<=b[1] or min(b)<0 or b[2]>w or b[3]>h:bad.append(i)
  x,y=r[4]
  if not(0<=x<512 and 0<=y<256):clickbad.append(i)
 res['splits'][s]={'records':len(d),'classes':dict(Counter(r[7] for r in d)),'unique_query':len(set(r[1] for r in d)),'unique_reference':len(set(r[2] for r in d)),'invalid_bbox_indices':bad,'invalid_prompt_indices':clickbad,'missing_indices':missing,'bbox_area_fraction_quantiles':np.quantile(area,[0,.25,.5,.75,1]).tolist()}
for a,b in itertools.combinations(D,2):
 res['overlaps'][a+'_'+b]={k:len(set(r[i] for r in D[a])&set(r[i] for r in D[b])) for k,i in [('id',0),('query',1),('reference',2)]}
 for k,i in [('query_hash',1),('reference_hash',2)]:res['overlaps'][a+'_'+b][k]=len(set(hashes.get(r[i]) for r in D[a])&set(hashes.get(r[i]) for r in D[b]))
# fixed box learned only from training, normalized reference coordinates
boxes=np.array([np.array(r[5])/np.array([*sizes[r[2]],*sizes[r[2]]]) for r in D['train']]);pred=np.median(boxes,axis=0)
res['baseline']={'training_median_box':pred.tolist()}
for s in ['val','test']:
 gt=np.array([np.array(r[5])/np.array([*sizes[r[2]],*sizes[r[2]]]) for r in D[s]])
 inter=np.maximum(0,np.minimum(gt[:,2:],pred[2:])-np.maximum(gt[:,:2],pred[:2])).prod(1)
 iou=inter/((gt[:,2:]-gt[:,:2]).prod(1)+(pred[2:]-pred[:2]).prod()-inter)
 err=np.linalg.norm((gt[:,:2]+gt[:,2:])/2-.5,axis=1)
 res['baseline'][s]={'n':len(gt),'mean_iou':float(iou.mean()),'success_iou_ge_05':float((iou>=.5).mean()),'center_guess_median_error_image_width':float(np.median(err))}
res['sample_record']=D['test'][0]
(out/'results.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf8')
# deterministic spread over test, 12 examples
ids=np.linspace(0,len(D['test'])-1,12,dtype=int); sheet=Image.new('RGB',(960,12*200),'white');draw=ImageDraw.Draw(sheet)
for j,i in enumerate(ids):
 r=D['test'][i];q=Image.open(base/'query'/r[1]).convert('RGB');t=Image.open(base/'satellite'/r[2]).convert('RGB')
 dr=ImageDraw.Draw(q);x,y=r[4];x=x*q.width/512;y=y*q.height/256;dr.ellipse((x-8,y-8,x+8,y+8),fill='red')
 ImageDraw.Draw(t).rectangle(r[5],outline='red',width=5)
 q.thumbnail((640,170));t.thumbnail((270,170));sheet.paste(q,(0,j*200+25));sheet.paste(t,(660,j*200+25));draw.text((5,j*200+4),f'test[{i}] {r[7]}',fill='black')
sheet.save(out/'samples.jpg');print(json.dumps(res,ensure_ascii=False,indent=2))

