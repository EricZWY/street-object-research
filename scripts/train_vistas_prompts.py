"""Class-specific prompt tuning pilot. Frozen detector; no novel training images."""
import argparse,json,hashlib,collections,time,contextlib,io,copy,sys,platform,traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import torch,cv2,ultralytics
from ultralytics import YOLOWorld
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.cfg import get_cfg
from ultralytics.utils.ops import non_max_suppression
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--epochs',type=int,default=3);p.add_argument('--run-name',default='vistas_prompt_'+datetime.now().strftime('%Y%m%d_%H%M%S'));p.add_argument('--lr',type=float,default=.002);p.add_argument('--imgsz',type=int,default=640);p.add_argument('--train-per-class',type=int,default=32);p.add_argument('--dev-per-class',type=int,default=8);p.add_argument('--sampler',choices=['uniform','class_cycle'],default='uniform');a=p.parse_args();w=a.workspace
if a.epochs<1 or a.lr<=0 or a.imgsz%32 or a.imgsz<320 or min(a.train_per_class,a.dev_per_class)<1:p.error('Invalid training settings')
if Path(a.run_name).name!=a.run_name or a.run_name in ('.','..'):p.error('run-name must be one directory name')
o=w/'runs'/a.run_name;o.mkdir(parents=True,exist_ok=False)
run_started=time.perf_counter()
def log(s):
 s=datetime.now().astimezone().isoformat(timespec='seconds')+' '+str(s)
 print(s,flush=True)
 with (o/'run.log').open('a',encoding='utf8') as f:f.write(str(s)+'\n')
def save(name,obj):(o/name).write_text(json.dumps(obj,indent=2),encoding='utf8')
phase_starts={}
def progress(phase,done,total,detail=''):
 now=time.perf_counter();start=phase_starts.setdefault(phase,now);elapsed=now-start
 eta=elapsed/done*(total-done) if done>=3 and elapsed>0 else None
 state={'phase':phase,'done':done,'total':total,'percent':100*done/total,'phase_elapsed_seconds':elapsed,'run_elapsed_seconds':now-run_started,'phase_eta_seconds':eta,'status':'running'}
 save('progress.json',state)
 estimate=f'{eta:.0f}s' if eta is not None else 'estimating'
 log(f'{phase} {done}/{total} ({100*done/total:.1f}%) | elapsed={elapsed:.1f}s | phase ETA={estimate} | {detail}')
def failed(kind,value,tb):
 save('status.json',{'status':'failed','error':str(value)})
 log('FAILED '+''.join(traceback.format_exception(kind,value,tb)))
sys.excepthook=failed
save('status.json',{'status':'running','started':datetime.now().astimezone().isoformat()})
log('RUN '+str(o)+' | ETA covers current phase only; setup/COCO evaluation/save excluded from training ETA')
phase_starts['prepare']=time.perf_counter()

cfg=json.loads((w/'reports/vistas_preparation/class_split_v1.json').read_text());cats=cfg['seen']+cfg['unseen'];ns=len(cfg['seen']);prompts=['street light fixture','electrical junction box','trash can','fire hydrant','manhole cover','storm drain','utility pole','store sign','surveillance camera','bench','mailbox','bicycle rack','information sign'];root=w/'data/Vistas/v2_selected/training';raw=json.loads((root/'v2.0/panoptic/panoptic_2020.json').read_text());cid={x['id']:cats.index(x['name']) for x in raw['categories'] if x['name'] in cats};imeta={x['id']:x for x in raw['images']};ann={x['image_id']:x for x in raw['annotations']};group={k:[s for s in v['segments_info'] if s['category_id'] in cid] for k,v in ann.items()}
def order(keys,salt):return sorted(keys,key=lambda x:hashlib.sha256((salt+str(x)).encode()).hexdigest())
eligible=[k for k,ss in group.items() if ss and all(cid[s['category_id']]<ns and not s['iscrowd'] for s in ss)]
train=set()
for c in range(ns):train.update(order([k for k in eligible if any(cid[s['category_id']]==c for s in group[k])],'train-prompt-v2')[:a.train_per_class])
# Exclude identical image bytes from development, not just file IDs.
hashes={k:hashlib.sha256((root/'images'/imeta[k]['file_name']).read_bytes()).hexdigest() for k in train};train_hash=set(hashes.values());dev=set()
prior=json.loads((w/'runs/vistas_prompt_v1/manifest.json').read_text());old_ids=set(prior['train']+prior['dev'])
old_ids.update(Path(f).stem for f in json.loads((w/'runs/vistas_detection_pilot/manifest.json').read_text())['images'])
for c in range(len(cats)):
 pool=order([k for k,ss in group.items() if k not in train and k not in old_ids and any(cid[s['category_id']]==c for s in ss) and not any(s['iscrowd'] for s in ss)],'dev-prompt-v2')
 added=0
 for k in pool:
  h=hashlib.sha256((root/'images'/imeta[k]['file_name']).read_bytes()).hexdigest()
  if h in train_hash:continue
  dev.add(k);added+=1
  if added==a.dev_per_class:break
train=sorted(train);dev=sorted(dev);records={};checks=[]
for j,k in enumerate(train+dev):
 img=cv2.imread(str(root/'v2.0/panoptic'/ann[k]['file_name']));rgb=img[:,:,::-1].astype(np.int32);ids=rgb[:,:,0]+256*rgb[:,:,1]+65536*rgb[:,:,2];rr=[]
 for s in group[k]:
  yy,xx=np.where(ids==s['id'])
  if len(xx)==0:continue
  assert len(xx)==s['area']
  box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)];rr.append({'class':cid[s['category_id']],'box':box,'area':len(xx)})
 records[k]=rr
 if (j+1)%10==0 or j+1==len(train)+len(dev):progress('prepare',j+1,len(train)+len(dev))
manifest={'train':train,'dev':dev,'categories':cats,'prompts':prompts,'epochs':a.epochs,'lr':a.lr,'seed':17,'image_size':a.imgsz,'batch':1,'precision':'FP32','train_selection':f'{a.train_per_class} images per seen class, SHA256 order, union; excludes annotated novel classes/crowds','dev_selection':f'{a.dev_per_class} images per class, union; excludes previously inspected IDs and training IDs/image SHA256','bbox':'mask min inclusive max exclusive; positive pixel-area retained','sampler':a.sampler,'optimizer':'AdamW only seen embedding, weight_decay=0','ultralytics':ultralytics.__version__,'torch':torch.__version__,'official_validation_used':False};save('manifest.json',manifest);save('ground_truth.json',records);log(f'Train {len(train)}, development {len(dev)}')
torch.manual_seed(17);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.6)
yolo=YOLOWorld(str(w/'weights/YOLOWorld/yolov8s-worldv2.pt'));yolo.set_classes(prompts);m=yolo.model;m.clip_model=None;m.to('cuda').eval()
for q in m.parameters():q.requires_grad_(False)
initial=m.txt_feats.detach().clone().to('cuda').float();learn=torch.nn.Parameter(initial[:,:ns].clone());novel=initial[:,ns:].clone();optimizer=torch.optim.AdamW([learn],lr=a.lr,weight_decay=0);m.args=get_cfg();head=m.model[-1];head.nc=ns;criterion=v8DetectionLoss(m)
parameters={'manifest':manifest,'optimizer_defaults':optimizer.defaults,'loss_gains':{k:getattr(m.args,k) for k in ['box','cls','dfl']},'scheduler':None,'warmup_steps':0,'gradient_accumulation_steps':1,'gradient_clip_norm':10,'augmentation':'none; OpenCV linear resize, centered pad114 to configured input size, BGR to RGB, divide255','num_workers':0,'cpu_threads':4,'cuda_memory_fraction':0.6,'trainable_parameters':learn.numel(),'frozen_detector_eval_mode':True,'prompt_normalization':'L2 each forward','checkpoint_sha256':hashlib.sha256((w/'weights/YOLOWorld/yolov8s-worldv2.pt').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':sys.version,'platform':platform.platform(),'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'evaluation':{'imgsz':a.imgsz,'conf':0.001,'nms_iou':0.7,'inference_max_det':300,'coco_maxDets':[1,10,100],'iou_range':[0.5,0.95,0.05]},'checkpoint_selection':'last epoch; no early stopping'}
save('resolved_parameters.json',parameters)
table='| Parameter | Value |\n|---|---|\n'+''.join(f'| {k} | {v} |\n' for k,v in {'epochs':a.epochs,'batch size':1,'learning rate':a.lr,'input size':a.imgsz,'sampler':a.sampler,'train images':len(train),'development images':len(dev),'optimizer':'AdamW','trainable parameters':learn.numel(),'precision':'FP32','seed':17,'weight decay':0,'gradient clip':10,'memory fraction':.6}.items())
(o/'report.md').write_text('# Training run '+a.run_name+'\n\nStatus: running.\n\n## Key parameters\n\n'+table+'\n## Detailed resolved parameters\n\n```json\n'+json.dumps(parameters,indent=2)+'\n```\n',encoding='utf8')
log('PARAMETERS '+str(o/'resolved_parameters.json'))

def digest():
 h=hashlib.sha256()
 for name,t in m.state_dict().items():h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
before=digest();assert not any(p.requires_grad for p in m.parameters());log(f'Trainable prompt parameters: {learn.numel()}, detector trainable: 0')
def get(k):
 img=cv2.imread(str(root/'images'/imeta[k]['file_name']));h,ww=img.shape[:2];scale=a.imgsz/max(h,ww);nw,nh=round(ww*scale),round(h*scale);left=(a.imgsz-nw)//2;top=(a.imgsz-nh)//2;res=cv2.resize(img,(nw,nh));canvas=np.full((a.imgsz,a.imgsz,3),114,dtype=np.uint8);canvas[top:top+nh,left:left+nw]=res
 tensor=torch.from_numpy(canvas[:,:,::-1].copy()).permute(2,0,1)[None].to('cuda').float()/255
 return tensor,(nw/ww,nh/h,left,top)
def evaluate(tag,features):
 phase_starts[tag]=time.perf_counter()
 head.nc=len(cats);m.eval();gt={'info':{},'images':[],'categories':[{'id':i,'name':c} for i,c in enumerate(cats)],'annotations':[]};pred=[];aid=1;times=[]
 with torch.inference_mode():
  for i,k in enumerate(dev):
   x,(sx,sy,l,t)=get(k);torch.cuda.synchronize();st=time.perf_counter();out=m.predict(x,txt_feats=features);ds=non_max_suppression(out[0],conf_thres=.001,iou_thres=.7,max_det=300,nc=len(cats))[0].cpu().numpy();torch.cuda.synchronize();times.append(time.perf_counter()-st)
   progress(tag,i+1,len(dev))
   gt['images'].append({'id':i,'width':imeta[k]['width'],'height':imeta[k]['height']})
   for r in records[k]:
    x1,y1,x2,y2=r['box'];gt['annotations'].append({'id':aid,'image_id':i,'category_id':r['class'],'bbox':[x1,y1,x2-x1,y2-y1],'area':(x2-x1)*(y2-y1),'iscrowd':0});aid+=1
   for b in ds:
    x1,x2=np.clip((b[[0,2]]-l)/sx,0,imeta[k]['width']);y1,y2=np.clip((b[[1,3]]-t)/sy,0,imeta[k]['height'])
    if x2>x1 and y2>y1:pred.append({'image_id':i,'category_id':int(b[5]),'bbox':[float(x1),float(y1),float(x2-x1),float(y2-y1)],'score':float(b[4])})
 save(tag+'_predictions.json',pred);save('dev_coco.json',gt)
 with contextlib.redirect_stdout(io.StringIO()):
  co=COCO();co.dataset=gt;co.createIndex();dt=co.loadRes(pred);ev=COCOeval(co,dt,'bbox');ev.evaluate();ev.accumulate();ev.summarize()
 pr=ev.eval['precision'];per={}
 for i,c in enumerate(cats):
  v=pr[:,:,i,0,2];v50=pr[0,:,i,0,2];per[c]={'AP':float(v[v>=0].mean()) if (v>=0).any() else None,'AP50':float(v50[v50>=0].mean()) if (v50>=0).any() else None}
 seen=float(np.mean([per[c]['AP'] for c in cats[:ns]]));unseen=float(np.mean([per[c]['AP'] for c in cats[ns:]]));summary={'AP':float(ev.stats[0]),'AP50':float(ev.stats[1]),'seen_AP':seen,'unseen_AP':unseen,'harmonic_AP':2*seen*unseen/(seen+unseen) if seen+unseen else 0,'classes':per,'median_forward_nms_seconds':float(np.median(times[1:]))};save(tag+'_summary.json',summary);log(f'{tag}: '+json.dumps({k:v for k,v in summary.items() if k!="classes"}));return summary
save('train_script_snapshot.py.json',{'source':Path(__file__).read_text(encoding='utf-8-sig')})
baseline=evaluate('baseline',initial);head.nc=ns;torch.cuda.reset_peak_memory_stats();lossrows=[]
phase_starts['training']=time.perf_counter()
for ep in range(a.epochs):
 head.nc=ns;m.eval();rng=np.random.default_rng(17+ep);order_=rng.permutation(len(train));tot=[]
 if a.sampler=='class_cycle':
  pools={c:[j for j,k in enumerate(train) if any(r['class']==c for r in records[k])] for c in range(ns)}
  order_=np.array([rng.choice(pools[(step+ep)%ns]) for step in range(len(train))])
 save(f'epoch_{ep+1}_sample_order.json',[train[int(j)] for j in order_])
 for step,j in enumerate(order_):
  k=train[j];x,(sx,sy,l,t)=get(k);boxes=[];labels=[]
  for r in records[k]:
   assert r['class']<ns;x1,y1,x2,y2=r['box'];boxes.append([((x1+x2)*sx/2+l)/a.imgsz,((y1+y2)*sy/2+t)/a.imgsz,(x2-x1)*sx/a.imgsz,(y2-y1)*sy/a.imgsz]);labels.append(r['class'])
  batch={'batch_idx':torch.zeros(len(labels),device='cuda'),'cls':torch.tensor(labels,device='cuda').view(-1,1),'bboxes':torch.tensor(boxes,device='cuda').view(-1,4)}
  optimizer.zero_grad();features=torch.nn.functional.normalize(learn,dim=-1);out=m.predict(x,txt_feats=features);loss,parts=criterion(out,batch);assert torch.isfinite(loss);loss.backward();assert learn.grad is not None and torch.isfinite(learn.grad).all();torch.nn.utils.clip_grad_norm_([learn],10);optimizer.step();tot.append(float(loss.detach()));lossrows.append({'epoch':ep+1,'step':step+1,'loss':tot[-1]})
  progress('training',ep*len(train)+step+1,a.epochs*len(train),f'epoch={ep+1}/{a.epochs} step={step+1}/{len(train)} loss={tot[-1]:.4f} lr={optimizer.param_groups[0]["lr"]} cuda_alloc={torch.cuda.memory_allocated()/2**20:.0f}MiB')
 log(f'epoch {ep+1} mean loss {np.mean(tot):.4f}');torch.save({'seen_embeddings':learn.detach().cpu(),'initial_embeddings':initial.cpu(),'epoch':ep+1},o/f'prompt_epoch_{ep+1}.pt');save('loss.json',lossrows)
finalfeatures=torch.cat([torch.nn.functional.normalize(learn.detach(),dim=-1),novel],dim=1);after=digest();assert before==after;assert torch.equal(finalfeatures[:,ns:],initial[:,ns:]);tuned=evaluate('tuned',finalfeatures)
audit={'detector_state_unchanged':before==after,'frozen_state_sha256':before,'novel_embeddings_bitwise_unchanged':True,'trainable_parameters':learn.numel(),'seen_embedding_l2_change':float((finalfeatures[:,:ns]-initial[:,:ns]).norm()),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20};save('audit.json',audit);torch.save({'text_embeddings':finalfeatures.cpu(),'prompts':prompts},o/'tuned_prompts.pt');log('COMPLETE '+json.dumps(audit))

save('status.json',{'status':'completed','elapsed_seconds':time.perf_counter()-run_started})
with (o/'report.md').open('a',encoding='utf8') as f:f.write('\n## Results\n\nStatus: completed.\n\n```json\n'+json.dumps({'baseline':baseline,'tuned':tuned,'audit':audit},indent=2)+'\n```\n')

report_path=o/'report.md'
report_path.write_text(report_path.read_text(encoding='utf8').replace('Status: running.','Status: completed.',1),encoding='utf8')

