"""Freeze the comparison matrix before inspecting any new results."""
import json,hashlib,urllib.request,time,zipfile
from pathlib import Path
w=Path(r'E:\CodexWorkspace\StreetObject_DataPlan');o=w/'reports/vistas_world_matrix_v4';o.mkdir(exist_ok=True)
old=json.loads((w/'runs/vistas_sweep_v2_highres/manifest.json').read_text())
prompts={'original':old['prompts'],
'short':['street light','electrical cabinet','garbage bin','fire hydrant','manhole cover','storm drain grate','utility pole','shop sign','security camera','bench','mailbox','bike rack','information sign'],
'descriptive':['street lamp lighting fixture','outdoor electrical junction box','public trash bin','street fire hydrant','round metal manhole cover in the road','storm drain grate at the curb','utility pole carrying overhead wires','storefront shop sign','mounted surveillance camera','outdoor public bench','street mailbox for posting letters','metal bicycle parking rack','public information signboard']}
plan={'created':time.strftime('%Y-%m-%d %H:%M:%S'),'models':['s','m','l'],'prompt_sets':prompts,'ensemble':'L2-normalized mean of the three normalized CLIP text embeddings per category; 13 queries, not 39 detections','cases':[{'case':f'{model}_{prompt}','model':model,'prompt':prompt} for model in ['s','m','l'] for prompt in ['original','short','descriptive','ensemble']]+[{'case':'s_tuned5','model':'s','prompt':'tuned5'},{'case':'s_tuned30','model':'s','prompt':'tuned30'}],'fixed':{'image_size':960,'tile_original_pixels':1024,'stride':768,'include_full_image':True,'per_view_max_det':1000,'final_max_det':1000,'confidence':.001,'nms_iou':.7,'batch_size':1,'precision':'FP32','cuda_memory_fraction':.6,'cpu_threads':4,'coco_maxDets':[1,10,100],'dev_images':len(old['dev']),'epochs':0,'optimizer':None,'seed':17},'selection':'No prompt revision after scores; full 3x4 comparison. S tuned5/30 are supervised references, not pure model-size comparisons. Same 56 development images; no official validation. Primary metric AP50; report seen and novel separately.','sources':['https://docs.ultralytics.com/models/yolo-world/','https://github.com/ultralytics/assets/releases/tag/v8.3.0']}
p=o/'plan.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(plan,indent=2),encoding='utf8')
weights=[]
for size in ['s','m','l']:
    dest=w/'weights/YOLOWorld'/f'yolov8{size}-worldv2.pt';url=f'https://github.com/ultralytics/assets/releases/download/v8.3.0/{dest.name}'
    if not dest.exists():
        part=dest.with_suffix('.pt.part');print('Downloading '+url,flush=True)
        with urllib.request.urlopen(url,timeout=60) as resp,part.open('wb') as f:
            total=int(resp.headers.get('Content-Length',0));done=0;last=time.monotonic()
            while True:
                data=resp.read(1024*1024)
                if not data:break
                f.write(data);done+=len(data)
                if time.monotonic()-last>5:print(f'{dest.name}: {done/2**20:.1f}/{total/2**20:.1f} MiB',flush=True);last=time.monotonic()
        assert not total or done==total
        with zipfile.ZipFile(part) as z:assert z.testzip() is None
        part.rename(dest)
    weights.append({'size':size,'filename':dest.name,'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'source':url})
    print(json.dumps(weights[-1]),flush=True)
(o/'weights.json').write_text(json.dumps(weights,indent=2),encoding='utf8')

