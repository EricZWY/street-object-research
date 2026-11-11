"""A single prespecified model/prompt case, full image plus tiles."""
import argparse,contextlib,hashlib,io,json,sys,time,traceback,platform
from pathlib import Path
import cv2,numpy as np,torch,ultralytics
from torchvision.ops import batched_nms
from ultralytics import YOLOWorld
from ultralytics.utils.ops import non_max_suppression
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from visualize_vistas_run import render_run
ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,required=True);ap.add_argument('--case',required=True);a=ap.parse_args();w=a.workspace;report=w/'reports/vistas_world_matrix_v4'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
plan=read(report/'plan.json');case=next(c for c in plan['cases'] if c['case']==a.case);r=w/'runs'/('vistas_world_v4_'+a.case);r.mkdir(exist_ok=False);beg=time.perf_counter()
def save(n,x):(r/n).write_text(json.dumps(x,indent=2),encoding='utf8')
def log(s):
    line=time.strftime('%H:%M:%S')+' '+s;print(line,flush=True)
    with (r/'run.log').open('a',encoding='utf8') as f:f.write(line+'\n')
def fail(t,v,tb):save('status.json',{'status':'failed','error':str(v)});log(''.join(traceback.format_exception(t,v,tb)))
sys.excepthook=fail;save('status.json',{'status':'running'})
old=w/'runs/vistas_sweep_v2_highres';m=read(old/'manifest.json');gt=read(old/'dev_coco.json');root=w/'data/Vistas/v2_selected/training/images'
torch.manual_seed(17);torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.6)
weight=w/'weights/YOLOWorld'/f"yolov8{case['model']}-worldv2.pt";y=YOLOWorld(str(weight));prompt=case['prompt']
if prompt=='ensemble':
    texts=[txt for p in ['original','short','descriptive'] for txt in plan['prompt_sets'][p]];y.set_classes(texts);feats=y.model.txt_feats.detach().clone().reshape(1,3,13,-1);features=torch.nn.functional.normalize(feats.mean(1),dim=-1)
else:
    texts=plan['prompt_sets'][prompt] if prompt in plan['prompt_sets'] else plan['prompt_sets']['original'];y.set_classes(texts);features=y.model.txt_feats.detach().clone()
if prompt in ['tuned5','tuned30']:
    cp=w/'runs'/('vistas_sweep_v2_highres' if prompt=='tuned5' else 'vistas_epoch30_v3')/'tuned_prompts.pt';features=torch.load(cp,map_location='cpu',weights_only=False)['text_embeddings']
model=y.model;model.clip_model=None;model.model[-1].nc=13;model.to('cuda').eval();features=features.float().to('cuda');assert features.shape==(1,13,512)
for p in model.parameters():p.requires_grad_(False)
params={'case':case,'fixed':plan['fixed'],'prompts':texts,'embedding_rule':plan['ensemble'] if prompt=='ensemble' else prompt,'weight_sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),'model_parameter_count':sum(p.numel() for p in model.parameters()),'torch':torch.__version__,'ultralytics':ultralytics.__version__,'python':sys.version,'platform':platform.platform(),'gpu':torch.cuda.get_device_name(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'dev_list_sha256':hashlib.sha256(json.dumps(m['dev']).encode()).hexdigest(),'ground_truth_sha256':hashlib.sha256((old/'dev_coco.json').read_bytes()).hexdigest(),'embedding_sha256':hashlib.sha256(features.cpu().numpy().tobytes()).hexdigest(),'pretraining_caveat':'Released checkpoints vary in learned weights; this is a same-family checkpoint scale comparison, not a controlled pretraining study.'}
save('resolved_parameters.json',params);save('source_split_manifest.json',m);save('manifest.json',{**m,'epochs':0,'training_performed':False,'train':[],'run_type':'inference-only','source_split_run':old.name,'actual_prompt_configuration':'resolved_parameters.json','source_prompt_checkpoint_epochs':5 if prompt=='tuned5' else 30 if prompt=='tuned30' else 0});save('ground_truth.json',read(old/'ground_truth.json'));save('dev_coco.json',gt);save('source_snapshot.json',{'source':Path(__file__).read_text(encoding='utf8')})
torch.cuda.reset_peak_memory_stats();log('START '+a.case+' '+str(params['model_parameter_count'])+' parameters')
def infer(img):
    h,ww=img.shape[:2];s=960/max(h,ww);nw,nh=round(ww*s),round(h*s);l,t=(960-nw)//2,(960-nh)//2;canvas=np.full((960,960,3),114,np.uint8);canvas[t:t+nh,l:l+nw]=cv2.resize(img,(nw,nh));x=torch.from_numpy(canvas[:,:,::-1].copy()).permute(2,0,1)[None].to('cuda').float()/255
    with torch.inference_mode():raw=model.predict(x,txt_feats=features)[0];out=non_max_suppression(raw,conf_thres=.001,iou_thres=.7,max_det=1000,nc=13)[0].cpu().numpy()
    out[:,[0,2]]=np.clip((out[:,[0,2]]-l)/(nw/ww),0,ww);out[:,[1,3]]=np.clip((out[:,[1,3]]-t)/(nh/h),0,h);return out
def pos(n):return sorted(set(list(range(0,max(n-1024,0)+1,768))+[max(n-1024,0)]))
pred=[];timing=[];start=time.perf_counter()
for idx,k in enumerate(m['dev']):
    st=time.perf_counter();im=cv2.imread(str(root/(k+'.jpg')));pieces=[infer(im)]
    for top in pos(im.shape[0]):
        for left in pos(im.shape[1]):
            tile=infer(im[top:top+1024,left:left+1024]);tile[:,[0,2]]+=left;tile[:,[1,3]]+=top;pieces.append(tile)
    arr=np.concatenate(pieces);arr=arr[(arr[:,2]>arr[:,0])&(arr[:,3]>arr[:,1])];inds=batched_nms(torch.from_numpy(arr[:,:4]),torch.from_numpy(arr[:,4]),torch.from_numpy(arr[:,5]),.7).numpy();arr=arr[inds[:1000]]
    for b in arr:pred.append({'image_id':idx,'category_id':int(b[5]),'bbox':[float(b[0]),float(b[1]),float(b[2]-b[0]),float(b[3]-b[1])],'score':float(b[4])})
    timing.append(time.perf_counter()-st);elapsed=time.perf_counter()-start;eta=elapsed/(idx+1)*(len(m['dev'])-idx-1);save('progress.json',{'done':idx+1,'total':56,'elapsed_seconds':elapsed,'eta_seconds':eta});log(f'{idx+1}/56 | views={len(pieces)} | elapsed={elapsed:.1f}s ETA={eta:.1f}s | CUDA={torch.cuda.memory_allocated()/2**20:.0f} MiB')
save('predictions_local.json',pred)
with contextlib.redirect_stdout(io.StringIO()):
    co=COCO();co.dataset=gt;co.createIndex();dt=co.loadRes(pred);ev=COCOeval(co,dt,'bbox');ev.evaluate();ev.accumulate();ev.summarize()
pr=ev.eval['precision'];per={c:{'AP50':float(pr[0,:,i,0,2][pr[0,:,i,0,2]>=0].mean()),'AP':float(pr[:,:,i,0,2][pr[:,:,i,0,2]>=0].mean())} for i,c in enumerate(m['categories'])}
summary={'AP50':float(ev.stats[1]),'AP':float(ev.stats[0]),'seen_AP50':float(np.mean([per[c]['AP50'] for c in m['categories'][:8]])),'novel_AP50':float(np.mean([per[c]['AP50'] for c in m['categories'][8:]])),'classes':per,'median_end_to_end_seconds':float(np.median(timing[1:])),'inference_total_seconds':sum(timing),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'parameter_count':params['model_parameter_count']}
if prompt=='tuned5':assert abs(summary['AP50']-read(w/'reports/vistas_diagnostic_v3/scale_probe.json')['results']['tiled1000']['AP50'])<1e-7
save('summary.json',summary);save('timings.json',timing);log('RESULT '+json.dumps({k:v for k,v in summary.items() if k!='classes'}))
render_run(w,old,out=r/'visual_samples',base_path=w/'reports/vistas_diagnostic_v3/tiled1000_predictions_local.json',tuned_path=r/'predictions_local.json',labels=('S tuned5 reference',a.case),caption='Fixed full image + tiles: '+a.case)
md='# '+a.case+'\n\n|Parameter|Value|\n|---|---|\n|Model|YOLO-World v2 '+case['model'].upper()+'|\n|Prompt|'+prompt+'|\n|Epoch|0 (inference only; tuned references reuse prior checkpoints)|\n|Batch|1|\n|Input|960|\n|Original-pixel tile / stride|1024 / 768|\n|Precision|FP32|\n|Max detections|1000|\n\n## Results\n```json\n'+json.dumps(summary,indent=2)+'\n```\n\n## Full parameters\n```json\n'+json.dumps(params,indent=2)+'\n```\n\n![Fixed samples](visual_samples/fixed_samples.jpg)\n'
(r/'report.md').write_text(md,encoding='utf8');save('status.json',{'status':'completed','elapsed_seconds':time.perf_counter()-beg});log('COMPLETE')

