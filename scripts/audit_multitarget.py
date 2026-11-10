import argparse,json,hashlib,itertools
from pathlib import Path
from collections import defaultdict
import torch
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);w=p.parse_args().workspace;d=w/'data/CVOGL/CVOGL_SVI';o=w/'reports/multitarget_audit';o.mkdir(parents=True,exist_ok=True)
cache={}
def digest(folder,name):
 key=(folder,name)
 if key not in cache:cache[key]=hashlib.sha256((d/folder/name).read_bytes()).hexdigest()
 return cache[key]
groups=defaultdict(list);qgroups=defaultdict(list)
for split in ['train','val','test']:
 for i,r in enumerate(torch.load(d/f'CVOGL_SVI_{split}.pth',weights_only=True)):
  x={'split':split,'index':i,'id':r[0],'query':r[1],'reference':r[2],'click':list(r[4]),'box':list(r[5]),'class':r[7]}
  qh=digest('query',r[1]);rh=digest('satellite',r[2]);groups[(qh,rh)].append(x);qgroups[qh].append(x)
pairs=[]
for key,rows in groups.items():
 for a,b in itertools.combinations(rows,2):
  if a['click']!=b['click'] and a['box']!=b['box']:pairs.append({'a':a,'b':b})
summary={'total_rows':sum(map(len,groups.values())),'unique_image_pairs':len(groups),'repeated_image_pair_groups':sum(len(v)>1 for v in groups.values()),'different_click_and_box_pairs':len(pairs),'val_only_pairs':sum(x['a']['split']==x['b']['split']=='val' for x in pairs),'repeated_query_groups':sum(len(v)>1 for v in qgroups.values())}
(o/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');(o/'candidate_pairs.json').write_text(json.dumps(pairs,indent=2),encoding='utf8');print(json.dumps(summary));print(json.dumps(pairs[:3]))

query_candidates=[]
for rows in qgroups.values():
 for a,b in itertools.combinations(rows,2):
  if a['click']!=b['click']:query_candidates.append({'a':a,'b':b})
summary['same_query_different_click_pairs']=len(query_candidates)
summary['val_same_query_different_click_pairs']=sum(x['a']['split']==x['b']['split']=='val' for x in query_candidates)
(o/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
(o/'query_candidates.json').write_text(json.dumps(query_candidates,indent=2),encoding='utf8')
print(json.dumps(summary));print(json.dumps(query_candidates[:2]))

