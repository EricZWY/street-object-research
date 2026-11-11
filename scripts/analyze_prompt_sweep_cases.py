"""Select largest gain/loss diagnostic cases after a fixed development sweep."""
import argparse,json,collections
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();w=a.workspace;o=w/'reports/vistas_sweep_v2';summary=json.loads((o/'summary.json').read_text());tag=max(summary,key=lambda t:summary[t]['tuned']['AP']);r=w/'runs'/('vistas_sweep_v2_'+tag);manifest=json.loads((r/'manifest.json').read_text());gt=json.loads((r/'ground_truth.json').read_text());base=json.loads((r/'baseline_predictions.json').read_text());tuned=json.loads((r/'tuned_predictions.json').read_text())
def box(x):
 a,b,c,d=x;return np.array([a,b,a+c,b+d])
def iou(a,b):
 lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:]);v=np.maximum(hi-lo,0).prod();return v/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-v+1e-9)
def matches(pred,truth):
 used=set()
 for rr in sorted(pred,key=lambda x:-x['score']):
  pool=[(iou(box(rr['bbox']),np.array(g['box'])),j) for j,g in enumerate(truth) if g['class']==rr['category_id'] and j not in used]
  if pool:
   score,j=max(pool)
   if score>=.5:used.add(j)
 return len(used)
rows=[]
for i,k in enumerate(manifest['dev']):
 b=[x for x in base if x['image_id']==i and x['score']>=.1];t=[x for x in tuned if x['image_id']==i and x['score']>=.1];nb=matches(b,gt[k]);nt=matches(t,gt[k]);rows.append({'id':i,'key':k,'base_tp':nb,'tuned_tp':nt,'delta':nt-nb})
cases=[max(rows,key=lambda x:x['delta']),min(rows,key=lambda x:x['delta'])]
for j,rec in enumerate(cases):
 k=rec['key'];im=Image.open(w/'data/Vistas/v2_selected/training/images'/f'{k}.jpg').convert('RGB');panel=Image.new('RGB',(1800,760),'white');draw=ImageDraw.Draw(panel)
 for col,(name,preds) in enumerate([('baseline',base),('tuned',tuned)]):
  img=im.copy();d=ImageDraw.Draw(img)
  for g in gt[k]:d.rectangle(g['box'],outline='lime',width=5)
  for pred in preds:
   if pred['image_id']==rec['id'] and pred['score']>=.1:d.rectangle(box(pred['bbox']).tolist(),outline='red',width=5)
  img.thumbnail((900,675));panel.paste(img,(col*900,45));draw.text((col*900+5,8),f'{tag} {name} | green GT, red predictions >=0.1',fill='black')
 draw.text((5,735),f"Selected by delta matched targets @IoU0.5: {rec['base_tp']} -> {rec['tuned_tp']}; diagnostic, not representative",fill='black');panel.save(o/f'case_{j}.jpg')
(o/'case_selection.json').write_text(json.dumps({'model_selected_by_development_AP':tag,'cases':cases},indent=2),encoding='utf8')
# Actual exposures: balanced selection does not imply equal object counts.
counts={}
for name in summary:
 rr=w/'runs'/('vistas_sweep_v2_'+name);mm=json.loads((rr/'manifest.json').read_text());gg=json.loads((rr/'ground_truth.json').read_text());cc=collections.Counter();unique=set()
 for ep in range(1,mm['epochs']+1):
  order=json.loads((rr/f'epoch_{ep}_sample_order.json').read_text())
  for k in order:unique.add(k);cc.update(x['class'] for x in gg[k])
 counts[name]={'distinct_training_images_actually_seen':len(unique),'instance_exposures':{c:cc[i] for i,c in enumerate(mm['categories'])}}
(o/'sampling_exposures.json').write_text(json.dumps(counts,indent=2),encoding='utf8');print(json.dumps({'selected':tag,'cases':cases,'sampling':counts},indent=2))

