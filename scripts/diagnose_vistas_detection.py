"""Audit saved detections. Coverage is an optimistic diagnostic, not AP/recall."""
import argparse, collections, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def overlaps(gt, pred):
    g=np.asarray(gt,float).reshape(-1,4); p=np.asarray(pred,float).reshape(-1,4)
    if not len(p): return np.zeros((len(g),0))
    p[:,2:]+=p[:,:2]
    inter=np.maximum(np.minimum(g[:,None,2:],p[None,:,2:])-np.maximum(g[:,None,:2],p[None,:,:2]),0).prod(-1)
    return inter/(np.prod(g[:,2:]-g[:,:2],-1)[:,None]+np.prod(p[:,2:]-p[:,:2],-1)[None,:]-inter+1e-12)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,required=True);a=ap.parse_args();w=a.workspace
    out=w/'reports/vistas_diagnostic_v3';out.mkdir(exist_ok=True)
    result={}; detail=[]
    for tag in ['control','lowlr','balanced','highres']:
        r=w/'runs'/('vistas_sweep_v2_'+tag);m=read(r/'manifest.json');gt=read(r/'ground_truth.json');co=read(r/'dev_coco.json');meta={x['id']:x for x in co['images']}
        losses=read(r/'loss.json');epochs=[{'epoch':e,'mean_loss':float(np.mean([x['loss'] for x in losses if x['epoch']==e]))} for e in sorted(set(x['epoch'] for x in losses))]
        res={'epochs':epochs,'methods':{}}
        for mode in ['baseline','tuned']:
            pp=read(r/(mode+'_predictions.json'));by=collections.defaultdict(list)
            for p in pp:by[p['image_id']].append(p)
            rows=[]
            for i,k in enumerate(m['dev']):
                ss=gt[k];pr=by[i];ov=overlaps([s['box'] for s in ss],[p['bbox'] for p in pr]);scale=m['image_size']/max(meta[i]['width'],meta[i]['height'])
                for j,g in enumerate(ss):
                    same=np.array([p['category_id']==g['class'] for p in pr]);bw=g['box'][2]-g['box'][0];bh=g['box'][3]-g['box'][1]
                    row={'image_index':i,'gt_index':j,'class':g['class'],'input_short_side':min(bw,bh)*scale,'input_sqrt_area':np.sqrt(bw*bh)*scale}
                    for t in [.001,.01,.1]:
                        eligible=np.array([p['score']>=t for p in pr]);v=ov[j,eligible];cv=ov[j,eligible&same]
                        row[f'any_{t}']=float(v.max()) if len(v) else 0.;row[f'same_{t}']=float(cv.max()) if len(cv) else 0.
                    rows.append(row)
            def summarize(rr):
                return {'n':len(rr),**{f'{kind}_coverage_iou{io}_score{t}':sum(x[f'{kind}_{t}']>=io for x in rr)/len(rr) if rr else None for kind in ['any','same'] for io in [.1,.3,.5] for t in [.001,.01,.1]}}
            summ=read(r/(mode+'_summary.json'));classap=summ['classes']
            sizebins=[('short<8',lambda x:x['input_short_side']<8),('short8-16',lambda x:8<=x['input_short_side']<16),('short16-32',lambda x:16<=x['input_short_side']<32),('short>=32',lambda x:x['input_short_side']>=32)]
            res['methods'][mode]={'all':summarize(rows),'size_bins':{n:summarize([x for x in rows if f(x)]) for n,f in sizebins},'classes':{c:{**summarize([x for x in rows if x['class']==j]),'AP50':classap[c]['AP50']} for j,c in enumerate(m['categories'])},'AP50':summ['AP50'],'seen_AP50':float(np.mean([classap[c]['AP50'] for c in m['categories'][:8]])),'novel_AP50':float(np.mean([classap[c]['AP50'] for c in m['categories'][8:]]))}
            if tag=='highres' and mode=='tuned':detail=rows
        result[tag]=res
    (out/'diagnosis.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    (out/'gt_diagnostics_local.json').write_text(json.dumps(detail,indent=2),encoding='utf8')
    fig,ax=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    for tag,r in result.items(): ax[0].plot([x['epoch'] for x in r['epochs']],[x['mean_loss'] for x in r['epochs']],marker='o',label=tag)
    ax[0].set(xlabel='Epoch',ylabel='Mean training loss',title='Five epochs do not establish convergence');ax[0].legend()
    rr=result['highres']['methods']['tuned']['size_bins'];xx=np.arange(len(rr));
    for offset,kind,label in [(-.18,'any','Any-class candidate'),(.18,'same','Correct-class candidate')]:ax[1].bar(xx+offset,[100*v[f'{kind}_coverage_iou0.5_score0.001'] if v['n'] else 0 for v in rr.values()],width=.36,label=label)
    ax[1].set(xticks=xx,xticklabels=[k+'\nn='+str(v['n']) for k,v in rr.items()],ylabel='GT coverage at IoU 0.5 (%)',ylim=(0,100),title='960 tuned: candidate coverage by input size');ax[1].legend()
    fig.savefig(out/'diagnosis.png',dpi=160);plt.close(fig)
    print(json.dumps({k:{'loss':v['epochs'],'baseline':v['methods']['baseline']['AP50'],'tuned':{x:v['methods']['tuned'][x] for x in ['AP50','seen_AP50','novel_AP50','all','size_bins']}} for k,v in result.items()},indent=2))

if __name__=='__main__':main()

