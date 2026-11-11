"""Aggregate fixed prompt-tuning comparisons, without rerunning inference."""
import argparse,json,hashlib,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);w=p.parse_args().workspace;o=w/'reports/vistas_sweep_v2';o.mkdir(exist_ok=True)
labels=['control','lowlr','balanced','highres'];data={}
for tag in labels:
 r=w/'runs'/('vistas_sweep_v2_'+tag)
 def read(n):return json.loads((r/n).read_text())
 assert read('status.json')['status']=='completed'
 data[tag]={'manifest':read('manifest.json'),'baseline':read('baseline_summary.json'),'tuned':read('tuned_summary.json'),'audit':read('audit.json'),'status':read('status.json'),'params':read('resolved_parameters.json'),'loss':read('loss.json')}
ref=data['control']['manifest'];gtref=json.loads((w/'runs/vistas_sweep_v2_control/ground_truth.json').read_text())
for tag,d in data.items():
 assert d['manifest']['train']==ref['train'] and d['manifest']['dev']==ref['dev']
 assert d['audit']['detector_state_unchanged'] and d['audit']['novel_embeddings_bitwise_unchanged']
 assert json.loads((w/'runs'/('vistas_sweep_v2_'+tag)/'ground_truth.json').read_text())==gtref
 if tag!='highres':assert abs(d['baseline']['AP']-data['control']['baseline']['AP'])<1e-9
safe={}
for tag,d in data.items():
 param=json.loads(json.dumps(d['params']));param['manifest'].pop('train');param['manifest'].pop('dev');param['manifest']['train_count']=len(ref['train']);param['manifest']['dev_count']=len(ref['dev']);param['manifest']['sample_id_list_sha256']=hashlib.sha256(json.dumps({'train':ref['train'],'dev':ref['dev']},sort_keys=True).encode()).hexdigest()
 safe[tag]={'baseline':d['baseline'],'tuned':d['tuned'],'audit':d['audit'],'status':d['status'],'parameters':param}
(o/'summary.json').write_text(json.dumps(safe,indent=2),encoding='utf8')
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
x=np.arange(4);width=.25
for off,(key,name) in enumerate([('seen_AP','Seen AP'),('unseen_AP','Unseen AP'),('AP','Overall AP')]):axes[0].bar(x+(off-1)*width,[data[t]['tuned'][key]*100 for t in labels],width,label=name)
axes[0].set_xticks(x,labels);axes[0].set_ylabel('COCO bbox AP (0-100)');axes[0].set_title('Fixed final epoch, development set');axes[0].legend()
for tag,d in data.items():
 means=[np.mean([r['loss'] for r in d['loss'] if r['epoch']==ep]) for ep in range(1,ref['epochs']+1)];axes[1].plot(range(1,ref['epochs']+1),means,marker='o',label=tag)
axes[1].set_xlabel('Epoch');axes[1].set_ylabel('Mean training loss');axes[1].set_title('Loss not directly comparable across samplers/resolutions');axes[1].legend();fig.tight_layout();fig.savefig(o/'comparison.png',dpi=150);plt.close(fig)
lines=['# 独立提示微调改进实验（2026-11-11）','','## 关键参数','','| 方案 | Epoch | Batch | LR | 输入 | 图片采样 | 训练/开发图片 |','|---|---:|---:|---:|---:|---|---:|']
for t,d in data.items():
 m=d['manifest'];lines.append(f"|{t}|{m['epochs']}|{m['batch']}|{m['lr']}|{m['image_size']}|{m['sampler']}|{len(m['train'])}/{len(m['dev'])}|")
lines+=['','共同参数：YOLO-World v2 Small；8已见/5未见；4096可训练提示参数；检测网络冻结并eval；FP32；AdamW，betas=(0.9,0.999)，eps=1e-8，weight_decay=0；梯度裁剪10；无调度/预热/梯度累积/数据增强；seed17；CPU线程4；CUDA分配器限额60%；框损失7.5、分类0.5、DFL1.5。','所有训练在独立可见终端串行运行，逐步记录日志与阶段ETA；固定末轮，没有事后挑最佳epoch。','','## 开发结果（0—100）','','| 方案 | 已见AP | 未见AP | 总体AP | AP50 | 本分辨率零样本总体AP |','|---|---:|---:|---:|---:|---:|']
for t,d in data.items():
 z=d['tuned'];lines.append(f"|{t}|{z['seen_AP']*100:.2f}|{z['unseen_AP']*100:.2f}|{z['AP']*100:.2f}|{z['AP50']*100:.2f}|{d['baseline']['AP']*100:.2f}|")
lines+=['','## 逐类别AP','','| 类别 | 640零样本 | control | lowlr | balanced | 960零样本 | highres |','|---|---:|---:|---:|---:|---:|---:|']
for c in ref['categories']:
 vals=[data['control']['baseline']['classes'][c]['AP']]+[data[t]['tuned']['classes'][c]['AP'] for t in labels[:3]]+[data['highres']['baseline']['classes'][c]['AP'],data['highres']['tuned']['classes'][c]['AP']];lines.append('|'+c+'|'+'|'.join(f'{v*100:.2f}' for v in vals)+'|')
lines+=['','## 耗时与显存','','|方案|总耗时/秒|峰值allocated/MiB|峰值reserved/MiB|','|---|---:|---:|---:|']
for t,d in data.items():lines.append(f"|{t}|{d['status']['elapsed_seconds']:.1f}|{d['audit']['peak_allocated_mib']:.1f}|{d['audit']['peak_reserved_mib']:.1f}|")
lines+=['','## 公平性和局限','四组训练/开发ID、GT完全一致，640三组零样本AP一致，冻结网络和未见嵌入审计全部通过。开发样本排除先前检查过的ID及本轮训练ID/同字节影像，但未保证空间或近重复独立。正式validation未参与。','按类选图合并会造成共现实例不均衡。class_cycle仅平衡被抽选类别次数，不保证每类实例数相同，允许重复图；每轮总更新步数相同。','高分辨率方案同时改变训练与推理尺寸，因此先比较其对应零样本基线，再比较微调增益；不能把相对640的所有提升归因于prompt tuning。','单随机种子、有限开发集，无显著性主张；本轮用于选择后续方案，不替代最终独立测试。候选样本、像素派生框和全部权重留在本地。','评测采用COCO bbox AP IoU .50:.95，maxDets100，推理阈值.001/NMS.7/max300；没有分割评测。','','## 完整参数附录','各运行report.md也包含关键参数表和完整配置；本汇总略去原始样本ID，仅保留数量和清单哈希。']
for t,d in safe.items():lines+=['','### '+t,'','```json',json.dumps(d['parameters'],ensure_ascii=False,indent=2),'```']
interpretation=(o/'interpretation.md').read_text(encoding='utf8') if (o/'interpretation.md').exists() else ''
lines.insert(2,interpretation)
(o/'report.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps({t:{'baseline':d['baseline']['AP'],'tuned':d['tuned']['AP'],'seen':d['tuned']['seen_AP'],'unseen':d['tuned']['unseen_AP']} for t,d in data.items()},indent=2))

