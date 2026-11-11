"""Inference-only probes: output caps, original-pixel tiles, GT-crop CLIP proxy."""
import argparse,collections,contextlib,hashlib,io,json,time
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from torchvision.ops import batched_nms
from ultralytics import YOLOWorld
from ultralytics.utils.ops import non_max_suppression
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from diagnose_vistas_detection import overlaps

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,required=True);a=ap.parse_args();w=a.workspace;r=w/'runs/vistas_sweep_v2_highres';o=w/'reports/vistas_diagnostic_v3';o.mkdir(exist_ok=True);m=read(r/'manifest.json');gt=read(r/'dev_coco.json');records=read(r/'ground_truth.json');root=w/'data/Vistas/v2_selected/training/images';beg=time.perf_counter()
def save(n,x):(o/n).write_text(json.dumps(x,indent=2),encoding='utf8')
def log(s):print(time.strftime('%H:%M:%S')+' '+s,flush=True)
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.6)
y=YOLOWorld(str(w/'weights/YOLOWorld/yolov8s-worldv2.pt'));y.set_classes(m['prompts']);model=y.model;model.clip_model=None;model.to('cuda').eval();features=torch.load(r/'tuned_prompts.pt',map_location='cuda',weights_only=False)['text_embeddings'].float().to('cuda')
torch.cuda.reset_peak_memory_stats()
def infer(img):
    h,ww=img.shape[:2];s=960/max(h,ww);nw,nh=round(ww*s),round(h*s);l,t=(960-nw)//2,(960-nh)//2;canvas=np.full((960,960,3),114,np.uint8);canvas[t:t+nh,l:l+nw]=cv2.resize(img,(nw,nh));x=torch.from_numpy(canvas[:,:,::-1].copy()).permute(2,0,1)[None].to('cuda').float()/255
    with torch.inference_mode():raw=model.predict(x,txt_feats=features)[0];out=non_max_suppression(raw.clone(),conf_thres=.001,iou_thres=.7,max_det=1000,nc=13)[0].cpu().numpy();raw=raw[0].transpose(0,1).cpu().numpy()
    out[:,[0,2]]=np.clip((out[:,[0,2]]-l)/(nw/ww),0,ww);out[:,[1,3]]=np.clip((out[:,[1,3]]-t)/(nh/h),0,h)
    return out,raw,(nw/ww,nh/h,l,t)
def serialize(arr,idx):return [{'image_id':idx,'category_id':int(b[5]),'bbox':[float(b[0]),float(b[1]),float(b[2]-b[0]),float(b[3]-b[1])],'score':float(b[4])} for b in arr if b[2]>b[0] and b[3]>b[1]]
def positions(n):return sorted(set(list(range(0,max(n-1024,0)+1,768))+[max(n-1024,0)]))
pred={n:[] for n in ['full300','full1000','tiled300','tiled1000']};rawcov=[];timings=[];counts=[]
for idx,k in enumerate(m['dev']):
    st=time.perf_counter();img=cv2.imread(str(root/(k+'.jpg')));full,raw,(sx,sy,l,t)=infer(img);full_seconds=time.perf_counter()-st
    pb=raw[:,:4].copy();pb[:,:2]-=pb[:,2:]/2;pb[:,0]=(pb[:,0]-l)/sx;pb[:,1]=(pb[:,1]-t)/sy;pb[:,2]/=sx;pb[:,3]/=sy
    rawmax=raw[:,4:].max(1);ov=overlaps([g['box'] for g in records[k]],pb[rawmax>=.001]);rawcov.extend((ov.max(1)>=.5).tolist() if ov.shape[1] else [False]*len(records[k]))
    pieces=[full];num=0;st=time.perf_counter()
    for top in positions(img.shape[0]):
        for left in positions(img.shape[1]):
            tile,_,_=infer(img[top:top+1024,left:left+1024]);tile[:,[0,2]]+=left;tile[:,[1,3]]+=top;pieces.append(tile);num+=1
    merged=np.concatenate(pieces);valid=(merged[:,2]>merged[:,0])&(merged[:,3]>merged[:,1]);merged=merged[valid]
    inds=batched_nms(torch.from_numpy(merged[:,:4]),torch.from_numpy(merged[:,4]),torch.from_numpy(merged[:,5]),.7).numpy();merged=merged[inds]
    for name,arr in [('full300',full[:300]),('full1000',full[:1000]),('tiled300',merged[:300]),('tiled1000',merged[:1000])]:pred[name].extend(serialize(arr,idx))
    counts.append({'image_index':idx,'full_after_nms':len(full),'tiles':num,'merged_after_nms':len(merged)})
    timings.append({'full_seconds':full_seconds,'tile_extra_seconds':time.perf_counter()-st})
    log(f'Inference {idx+1}/56 | tiles={num} | full candidates={len(full)} | elapsed={time.perf_counter()-beg:.0f}s | ETA={(time.perf_counter()-beg)/(idx+1)*(56-idx-1):.0f}s')
summaries={}
for name,pp in pred.items():
    save(name+'_predictions_local.json',pp)
    with contextlib.redirect_stdout(io.StringIO()):
        co=COCO();co.dataset=gt;co.createIndex();dt=co.loadRes(pp);ev=COCOeval(co,dt,'bbox');ev.evaluate();ev.accumulate();ev.summarize()
    pr=ev.eval['precision'];per={c:{'AP50':float(pr[0,:,i,0,2][pr[0,:,i,0,2]>=0].mean())} for i,c in enumerate(m['categories'])};summary={'AP50':float(ev.stats[1]),'AP':float(ev.stats[0]),'seen_AP50':float(np.mean([per[c]['AP50'] for c in m['categories'][:8]])),'novel_AP50':float(np.mean([per[c]['AP50'] for c in m['categories'][8:]])),'classes':per}
    by=collections.defaultdict(list)
    for p in pp:by[p['image_id']].append(p)
    rows=[]
    for i,k in enumerate(m['dev']):
        pp1=by[i];ov=overlaps([g['box'] for g in records[k]],[x['bbox'] for x in pp1]);meta=gt['images'][i];scale=960/max(meta['width'],meta['height'])
        for j,g in enumerate(records[k]):
            same=np.array([x['category_id']==g['class'] for x in pp1]);v=ov[j,same];rows.append({'class':g['class'],'short':min(g['box'][2]-g['box'][0],g['box'][3]-g['box'][1])*scale,'any':bool(ov[j].max()>=.5) if ov.shape[1] else False,'same':bool(v.max()>=.5) if len(v) else False})
    summary['candidate_coverage']={q:float(np.mean([x[q] for x in rows])) for q in ['any','same']};summary['short_under8_coverage']={q:float(np.mean([x[q] for x in rows if x['short']<8])) for q in ['any','same']};summary['macro_candidate_coverage']={q:float(np.mean([np.mean([x[q] for x in rows if x['class']==c]) for c in range(13)])) for q in ['any','same']}
    summaries[name]=summary;log(name+' '+json.dumps({k:v for k,v in summary.items() if k!='classes'}))
original=read(r/'tuned_summary.json')['AP50'];assert abs(summaries['full300']['AP50']-original)<1e-7,(summaries['full300']['AP50'],original)
save('scale_probe.json',{'parameters':{'source_run':r.name,'input':960,'tile_original_pixels':1024,'tile_stride':768,'overlap_nominal':.25,'global_full_image_included':True,'tile_nms_max_det':1000,'global_max_dets_compared':[300,1000],'score':.001,'nms_iou':.7,'coco_maxDets':[1,10,100],'model':'YOLO-World-v2-S','prompt_checkpoint_epoch':5,'precision':'FP32','cuda_fraction':.6,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'results':summaries,'raw_pre_nms_any_coverage':float(np.mean(rawcov)),'counts':counts,'timings':timings,'full300_reproduced':True,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20})
# Proxy: frozen CLIP crop classification, NOT detector regional logits or detection AP.
del model,y,features;torch.cuda.empty_cache()
import clip
clipmodel,preprocess=clip.load('ViT-B/32',device='cuda');clipmodel.eval()
with torch.inference_mode():text=clipmodel.encode_text(clip.tokenize(m['prompts']).to('cuda'));text=text/text.norm(dim=-1,keepdim=True)
crop_rows=[]
for idx,k in enumerate(m['dev']):
    im=Image.open(root/(k+'.jpg')).convert('RGB')
    for j,g in enumerate(records[k]):
        x1,y1,x2,y2=g['box'];cx=(x1+x2)/2;cy=(y1+y2)/2;hw=(x2-x1)*.625;hh=(y2-y1)*.625;crop=im.crop((max(0,int(cx-hw)),max(0,int(cy-hh)),min(im.width,int(np.ceil(cx+hw))),min(im.height,int(np.ceil(cy+hh)))))
        with torch.inference_mode():v=clipmodel.encode_image(preprocess(crop)[None].to('cuda'));v=v/v.norm(dim=-1,keepdim=True);order=(v@text.T)[0].argsort(descending=True).cpu().tolist()
        crop_rows.append({'class':g['class'],'predicted':order[0],'top1':order[0]==g['class'],'top3':g['class'] in order[:3]})
    if (idx+1)%8==0:log(f'GT crop proxy {idx+1}/56')
per={c:{'n':sum(x['class']==i for x in crop_rows),**{q:float(np.mean([x[q] for x in crop_rows if x['class']==i])) for q in ['top1','top3']}} for i,c in enumerate(m['categories'])}
save('crop_proxy.json',{'method':'Frozen CLIP ViT-B/32, original-pixel GT box expanded 1.25x, CLIP standard center-crop preprocessing, 13 supplied texts, all 844 objects. Closed-set recognition proxy, not detector accuracy; center crop can remove ends of elongated objects.','classes':per,'macro_top1':float(np.mean([x['top1'] for x in per.values()])),'macro_top3':float(np.mean([x['top3'] for x in per.values()])),'confusion':[[sum(x['class']==i and x['predicted']==j for x in crop_rows) for j in range(13)] for i in range(13)]})
log('COMPLETE '+str(o));save('probe_status.json',{'status':'completed','elapsed_seconds':time.perf_counter()-beg})

