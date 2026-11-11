import argparse,json,collections
from pathlib import Path
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);w=p.parse_args().workspace;o=w/'runs/vistas_prompt_v1';m=json.loads((o/'manifest.json').read_text());gt=json.loads((o/'ground_truth.json').read_text());b=json.loads((o/'baseline_predictions.json').read_text());t=json.loads((o/'tuned_predictions.json').read_text());stats={}
for split in ['train','dev']:
 cnt=collections.Counter(r['class'] for k in m[split] for r in gt[k]);stats[split]={c:cnt[i] for i,c in enumerate(m['categories'])}
(o/'sample_counts.json').write_text(json.dumps(stats,indent=2),encoding='utf8');print(json.dumps(stats))
for j,c in enumerate([0,3]):
 idx=next(i for i,k in enumerate(m['dev']) if any(r['class']==c for r in gt[k]));k=m['dev'][idx];im=Image.open(w/'data/Vistas/v2_selected/training/images'/f'{k}.jpg').convert('RGB');panel=Image.new('RGB',(1600,650),'white');dr=ImageDraw.Draw(panel)
 for col,(tag,pp) in enumerate([('baseline',b),('tuned',t)]):
  copy=im.copy();d=ImageDraw.Draw(copy)
  for r in gt[k]:
   if r['class']==c:d.rectangle(r['box'],outline='lime',width=5)
  for r in pp:
   if r['image_id']==idx and r['category_id']==c and r['score']>=.1:
    x,y,bw,bh=r['bbox'];d.rectangle([x,y,x+bw,y+bh],outline='red',width=5)
  copy.thumbnail((800,600));panel.paste(copy,(col*800,40));dr.text((col*800+8,8),tag+' '+m['categories'][c]+' green=GT red=pred (score>=0.1)',fill='black')
 panel.save(o/f'comparison_{j}.jpg')

