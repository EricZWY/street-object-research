"""SVI box evaluation using pinned upstream CVOS; no segmentation labels needed."""
import argparse, hashlib, json, sys, time, subprocess
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image, ImageDraw

REV='289602a033d5353d06987a2d56a808e9c938cdad'

def main():
 p=argparse.ArgumentParser(__doc__)
 p.add_argument('--upstream',type=Path,required=True)
 p.add_argument('--data',type=Path,required=True,help='CVOGL_SVI directory')
 p.add_argument('--checkpoint',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--split',choices=['val','test'],default='val')
 p.add_argument('--limit',type=int,default=8,help='0 evaluates entire split')
 p.add_argument('--device',default='cuda')
 p.add_argument('--prompt',choices=['actual','center','far','zero'],default='actual')
 p.add_argument('--memory-fraction',type=float,default=0.6)
 a=p.parse_args()
 if a.limit<0:p.error('limit must be nonnegative')
 revision=subprocess.check_output(['git','-C',str(a.upstream),'rev-parse','HEAD'],text=True).strip()
 if revision!=REV:raise RuntimeError('Unexpected upstream revision: '+revision)
 sys.path.insert(0,str(a.upstream.resolve()))
 from model.TROGeo import TROGeo
 from utils.utils import eval_iou_acc
 a.output.mkdir(parents=True,exist_ok=True)
 torch.manual_seed(0);torch.set_num_threads(4)
 device=torch.device(a.device)
 if not 0<a.memory_fraction<=1:p.error('memory fraction must be in (0,1]')
 if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
 checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
 state={k.removeprefix('module.'):v for k,v in checkpoint['state_dict'].items()}
 # The complete task checkpoint replaces every backbone tensor; skip redundant ImageNet download.
 original=torchvision.models.swin_s
 with patch('torchvision.models.swin_s',side_effect=lambda **kw: original(weights=None)):
  model=TROGeo()
 # Shared backbone is registered twice; reject inconsistent duplicated tensors.
 for k,v in state.items():
  if k.startswith('query_model.'):
   other='reference_model.'+k[len('query_model.'):]
   if other in state and not torch.equal(v,state[other]):raise RuntimeError('Shared backbone differs: '+k)
 model.load_state_dict(state,strict=True)
 key_count=len(state);epoch=checkpoint.get('epoch');del checkpoint,state
 model=model.to(device).eval()
 if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
 features={}
 def capture(name):
  def hook(module,inputs,output):
   features[name]=output.detach().float().cpu().numpy()
  return hook
 model.combine_clickptns_conv.register_forward_hook(capture('combined'))
 model.cross_attention.register_forward_hook(capture('fused'))
 transform=Compose([ToTensor(),Normalize([.485,.456,.406],[.229,.224,.225])])
 data=torch.load(a.data/f'CVOGL_SVI_{a.split}.pth',weights_only=True)
 n=len(data) if a.limit==0 else min(a.limit,len(data))
 anchors=torch.tensor([[37,41],[78,84],[96,215],[129,129],[194,82],[198,179],[246,280],[395,342],[550,573]][::-1],dtype=torch.float32,device=device)
 yy,xx=np.mgrid[:256,:512]; results=[]
 for i,row in enumerate(data[:n]):
  with Image.open(a.data/'query'/row[1]) as im:q=im.convert('RGB')
  with Image.open(a.data/'satellite'/row[2]) as im:r=im.convert('RGB')
  if q.size!=(512,256) or r.size!=(1024,1024):raise ValueError('Unexpected image size')
  x,y=map(int,row[4])
  if a.prompt=='center':x,y=256,128
  if a.prompt=='far':x,y=(511 if x<256 else 0),(255 if y<128 else 0)
  prompt=((1-np.sqrt((yy-y)**2+(xx-x)**2)/np.hypot(256,512))**2).astype('float32')
  if a.prompt=='zero':prompt[:]=0
  gt=torch.tensor(np.array(row[5],dtype=int),dtype=torch.float32,device=device)[None]
  if device.type=='cuda':torch.cuda.synchronize()
  started=time.perf_counter()
  with torch.inference_mode():
   boxes,masks=model(transform(q)[None].to(device),transform(r)[None].to(device),torch.from_numpy(prompt)[None].to(device))
   boxes=boxes.reshape(1,9,5,*boxes.shape[-2:])
   if not torch.isfinite(boxes).all():raise RuntimeError('Nonfinite model output')
   # Official decoder/IoU; dummy cell IDs affect only unused center-cell accuracy.
   dummy=torch.zeros(1,device=device,dtype=torch.long)
   acc,_,iou,_,pred,_=eval_iou_acc(boxes,gt,anchors,dummy,dummy,1024,[.5,.25])
  if device.type=='cuda':torch.cuda.synchronize()
  record={'index':i,'id':row[0],'class':row[7],'query':row[1],'reference':row[2],'iou':float(iou),'prediction':pred[0].cpu().tolist(),'ground_truth':gt[0].cpu().tolist(),'seconds':time.perf_counter()-started}
  np.savez_compressed(a.output/f'features_{i}.npz',**features)
  record['click_xy']=[x,y] if a.prompt!='zero' else None
  record['prompt_range']=[float(prompt.min()),float(prompt.max())]
  results.append(record)
  if i<4:
   draw=ImageDraw.Draw(r);draw.rectangle(row[5],outline='lime',width=4);draw.rectangle(record['prediction'],outline='red',width=4);r.save(a.output/f'prediction_{i}.jpg')
  if (i+1)%50==0 or i==n-1: print(f'{i+1}/{n} IoU={float(iou):.4f}',flush=True)
  with (a.output/'predictions.jsonl').open('a' if i else 'w',encoding='utf8') as f:f.write(json.dumps(record)+'\n')
 summary={'upstream_revision':revision,'checkpoint_sha256':hashlib.file_digest(a.checkpoint.open('rb'),'sha256').hexdigest(),'strict_state_keys':key_count,'epoch':epoch,'torch':torch.__version__,'torchvision':torchvision.__version__,'split':a.split,'evaluated':n,'total_split':len(data),'sampling':'first N; smoke test only' if n<len(data) else 'entire official split','prompt':a.prompt,'memory_fraction':a.memory_fraction,'peak_cuda_reserved_mib':torch.cuda.max_memory_reserved()/2**20 if device.type=='cuda' else None,'precision':'float32','batch_size':1,'mean_iou':float(np.mean([z['iou'] for z in results])),'acc_iou_gt_05':sum(z['iou']>.5 for z in results)/n,'acc_iou_gt_025':sum(z['iou']>.25 for z in results)/n,'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None}
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()

