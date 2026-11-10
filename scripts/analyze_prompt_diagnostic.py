import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args()
r=a.workspace/'runs/prompt_diagnostic';out=a.workspace/'reports/prompt_diagnostic';out.mkdir(parents=True,exist_ok=True)
rows={m:[json.loads(s) for s in (r/m/'predictions.jsonl').read_text().splitlines()] for m in ['actual','center','far','zero']}
assert all(len(v)==40 for v in rows.values())
stats={};details=[]
for m,rr in rows.items():
 assert [x['id'] for x in rr]==[x['id'] for x in rows['actual']]
 changes=[];feat={k:[] for k in ['combined','fused']}
 for i,(x,b) in enumerate(zip(rr,rows['actual'])):
  d=float(np.max(np.abs(np.array(x['prediction'])-b['prediction'])));changes.append(d)
  with np.load(r/m/f'features_{i}.npz') as f,np.load(r/'actual'/f'features_{i}.npz') as g:
   for k in feat:feat[k].append(float(np.linalg.norm(f[k]-g[k])/max(np.linalg.norm(g[k]),1e-12)))
  details.append({'mode':m,'index':i,'max_box_delta_px':d,'relative_l2':{k:feat[k][-1] for k in feat}})
 stats[m]={'n':40,'acc50':sum(x['iou']>.5 for x in rr)/40,'mean_iou':float(np.mean([x['iou'] for x in rr])),'box_delta_median_px':float(np.median(changes)),'box_delta_max_px':max(changes),'boxes_changed_lt_1px':sum(x<1 for x in changes),'feature_relative_l2_median':{k:float(np.median(v)) for k,v in feat.items()}}
(out/'summary.json').write_text(json.dumps(stats,indent=2),encoding='utf8');(out/'paired_details.json').write_text(json.dumps(details,indent=2),encoding='utf8');print(json.dumps(stats,indent=2))

