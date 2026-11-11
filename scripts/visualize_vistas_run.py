"""Deterministic development samples; reusable after every training run."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def font(n):return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n)
def render_run(w,r,out=None,base_path=None,tuned_path=None,labels=('Zero-shot','Tuned'),caption=None):
    m=read(r/'manifest.json');gt=read(r/'ground_truth.json');base=read(base_path or r/'baseline_predictions.json');tuned=read(tuned_path or r/'tuned_predictions.json');root=w/'data/Vistas/v2_selected/training/images';out=out or r/'visual_samples';out.mkdir(exist_ok=True)
    keys=sorted(m['dev'],key=lambda k:hashlib.sha256(('fixed-visual-v1'+k).encode()).hexdigest())[:4]
    names=[c.split('--')[-1] for c in m['categories']];selection={'rule':'SHA256 fixed-visual-v1 + image ID, first 4; per-class crop SHA256 fixed-crop-v1 + ID/index; independent of prediction','score_threshold':.1,'full_image_indices':[m['dev'].index(k) for k in keys],'crops':[]}
    def panel(k,mode,preds,region=None):
        im=Image.open(root/(k+'.jpg')).convert('RGB');region=region or (0,0,im.width,im.height);im=im.crop(region);zoom=min(590/im.width,360/im.height);im=im.resize((max(1,round(im.width*zoom)),max(1,round(im.height*zoom))),Image.Resampling.LANCZOS);scale=im.width/(region[2]-region[0]);d=ImageDraw.Draw(im);idx=m['dev'].index(k)
        boxes=[(g['box'],names[g['class']],None) for g in gt[k]] if mode=='GT' else [([p['bbox'][0],p['bbox'][1],p['bbox'][0]+p['bbox'][2],p['bbox'][1]+p['bbox'][3]],names[p['category_id']],p['score']) for p in preds if p['image_id']==idx and p['score']>=.1]
        for b,lab,s in boxes:
            if b[2]<=region[0] or b[0]>=region[2] or b[3]<=region[1] or b[1]>=region[3]:continue
            b=[(b[0]-region[0])*scale,(b[1]-region[1])*scale,(b[2]-region[0])*scale,(b[3]-region[1])*scale];color='#00ff66' if mode=='GT' else '#ffcc00';d.rectangle(b,outline=color,width=2)
            label=lab+(f' {s:.2f}' if s is not None else '');x=max(0,min(b[0],im.width-100));y=max(0,min(b[1],im.height-18));bb=d.textbbox((x,y),label,font=font(12));d.rectangle(bb,fill='black');d.text((x,y),label,fill=color,font=font(12))
        tile=Image.new('RGB',(600,400),'#f3f4f6');tile.paste(im,((600-im.width)//2,32));ImageDraw.Draw(tile).text((10,7),mode,font=font(18),fill='black');return tile
    sheet=Image.new('RGB',(1800,1660),'white');d=ImageDraw.Draw(sheet);d.text((15,8),(caption or r.name)+' | fixed sample; predictions score >= 0.10',font=font(22),fill='black')
    for row,k in enumerate(keys):
        for col,(mode,preds) in enumerate([('GT',[]),(labels[0],base),(labels[1],tuned)]):sheet.paste(panel(k,mode,preds),(col*600,50+row*400))
    sheet.save(out/'fixed_samples.jpg',quality=92)
    crops=[]
    for ci,c in enumerate(m['categories']):
        pool=[(k,j,g) for k in m['dev'] for j,g in enumerate(gt[k]) if g['class']==ci]
        k,j,g=min(pool,key=lambda x:hashlib.sha256(('fixed-crop-v1'+x[0]+str(x[1])).encode()).hexdigest());im=Image.open(root/(k+'.jpg'));b=g['box'];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2;ww=max(120,(b[2]-b[0])*2);hh=max(120,(b[3]-b[1])*2);reg=(max(0,int(cx-ww/2)),max(0,int(cy-hh/2)),min(im.width,int(cx+ww/2)),min(im.height,int(cy+hh/2)))
        selection['crops'].append({'class':c,'image_index':m['dev'].index(k),'gt_index':j,'region':reg})
        row=Image.new('RGB',(1800,440),'white');ImageDraw.Draw(row).text((10,5),c+' (fixed class sample; crop for display only)',font=font(20),fill='black')
        for col,(mode,preds) in enumerate([('GT',[]),(labels[0],base),(labels[1],tuned)]):row.paste(panel(k,mode,preds,reg),(col*600,35))
        crops.append(row)
    for page,start in enumerate(range(0,len(crops),4)):
        batch=crops[start:start+4];sheet=Image.new('RGB',(1800,440*len(batch)),'white')
        for j,row in enumerate(batch):sheet.paste(row,(0,j*440))
        sheet.save(out/f'class_crops_{page+1}.jpg',quality=92)
    (out/'selection_local.json').write_text(json.dumps(selection,indent=2),encoding='utf8')
    print('Visual samples saved: '+str(out),flush=True)
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,required=True);ap.add_argument('--run-name',required=True);a=ap.parse_args();render_run(a.workspace,a.workspace/'runs'/a.run_name)

