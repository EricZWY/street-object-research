"""Aggregate the prespecified S/M/L x prompt matrix without raw data publication."""
import argparse,json,hashlib,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,required=True);a=ap.parse_args();w=a.workspace;o=w/'reports/vistas_world_matrix_v4'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def fmt(x):return f'{100*x:.2f}'
plan=read(o/'plan.json');weights=read(o/'weights.json');results={};parameters={};hashes=set();fixedset=set()
for case in plan['cases']:
    name=case['case'];r=w/'runs'/('vistas_world_v4_'+name);assert read(r/'status.json')['status']=='completed',name
    results[name]=read(r/'summary.json');parameters[name]=read(r/'resolved_parameters.json');hashes.add((parameters[name]['dev_list_sha256'],parameters[name]['ground_truth_sha256']));fixedset.add(json.dumps(parameters[name]['fixed'],sort_keys=True))
assert len(hashes)==len(fixedset)==1
prior=read(w/'reports/vistas_diagnostic_v3/scale_probe.json')['results']['tiled1000']['AP50'];assert abs(results['s_tuned5']['AP50']-prior)<1e-7
assert abs(results['s_original']['parameter_count']-12759880)<1
best=max((c['case'] for c in plan['cases'] if c['prompt'] not in ['tuned5','tuned30']),key=lambda name:results[name]['AP50']);bestnovel=max(results,key=lambda name:results[name]['novel_AP50']);bestall=max(results,key=lambda name:results[name]['AP50'])
promptnames=['original','short','descriptive','ensemble'];fig,ax=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
for axis,key,title in zip(ax,['AP50','seen_AP50','novel_AP50'],['All AP50','Seen AP50','Novel AP50']):
    mat=np.array([[100*results[f'{model}_{prompt}'][key] for prompt in promptnames] for model in ['s','m','l']]);im=axis.imshow(mat,vmin=0,vmax=50,cmap='Blues');axis.set(xticks=range(4),xticklabels=promptnames,yticks=range(3),yticklabels=['S','M','L'],title=title)
    for i in range(3):
        for j in range(4):axis.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',color='white' if mat[i,j]>30 else 'black',fontsize=11)
fig.colorbar(im,ax=ax,shrink=.8,label='AP50 (%) [target: 50]');fig.savefig(o/'matrix.png',dpi=160);fig.savefig(o/'matrix.svg');plt.close(fig)
modeldeltas={};promptdeltas={}
for prompt in promptnames:modeldeltas[prompt]={model:{k:results[f'{model}_{prompt}'][k]-results[f's_{prompt}'][k] for k in ['AP50','seen_AP50','novel_AP50']} for model in ['m','l']}
for model in ['s','m','l']:promptdeltas[model]={prompt:{k:results[f'{model}_{prompt}'][k]-results[f'{model}_original'][k] for k in ['AP50','seen_AP50','novel_AP50']} for prompt in promptnames[1:]}
summary={'plan':plan,'weights':weights,'results':results,'parameters':parameters,'same_prompt_model_deltas':modeldeltas,'same_model_prompt_deltas':promptdeltas,'selected_on_development':{'best_unadapted':best,'best_all':bestall,'best_novel':bestnovel},'verification':{'same_development_and_ground_truth':True,'same_inference_settings':True,'s_tuned5_AP50_reproduced':True}}
(o/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
lines=['# 固定全图＋切片的提示词与模型规模对照','',
f"本轮完成12组预训练推理对照（S/M/L × 四种提示）及2组S微调参照。预训练组开发集最高为{best}，总体AP50={fmt(results[best]['AP50'])}%；全部组最高为{bestall}，AP50={fmt(results[bestall]['AP50'])}%。本轮没有新增训练。",'']
if (o/'interpretation.md').exists():lines += [(o/'interpretation.md').read_text(encoding='utf8'),'']
lines+=['## 关键参数','',
'|参数|固定设置|','|---|---|','|模型|YOLO-World v2 S/M/L，官方预训练权重|','|本轮Epoch / 学习率|0 / 不适用（仅推理）|','|Batch size|1，各实验串行|','|每个视图输入|960×960 letterbox|','|推理视图|全图＋原图1024像素切片|','|切片步长 / 重叠|768 / 名义25%|','|提示数量|13类、每类1个向量；ensemble为向量平均|','|置信度 / NMS IoU|0.001 / 0.7|','|每个视图 / 合并后最大输出|1000 / 1000|','|COCO maxDets|[1,10,100]，各组一致|','|精度 / CUDA分配器限额|FP32 / 60%|','|开发集|同56张、844个GT，8已见＋5未见|','|CPU线程 / seed|4 / 17|','|固定可视化|4张全图＋每类1局部，显示阈值0.1|','',
'S tuned5/tuned30复用此前训练出的提示向量：141张训练图、5/30轮、batch1、lr0.0002、960输入、4096个提示参数、网络冻结；不能与未适配M/L直接解释成纯参数量差异。S/M/L同提示对照属于已发布checkpoint的规模对照，未重新控制各模型预训练过程。',
'','## 全部结果','',
'|模型/提示|总体AP50|已见AP50|未见AP50|AP50:95|单图中位秒|峰值allocated MiB|峰值reserved MiB|','|---|---:|---:|---:|---:|---:|---:|---:|']
for name,x in results.items():lines.append(f"|{name}|{fmt(x['AP50'])}|{fmt(x['seen_AP50'])}|{fmt(x['novel_AP50'])}|{fmt(x['AP'])}|{x['median_end_to_end_seconds']:.3f}|{x['peak_allocated_mib']:.1f}|{x['peak_reserved_mib']:.1f}|")
lines+=['','![三种规模与四种提示](matrix.png)','',
'计时为全图读取、预处理、全部视图推理和合并后处理，不含模型加载、文本编码、COCO汇总和可视化；排除首图。显存为模型加载后重置计数的PyTorch峰值，不是整张GPU占用。',
'','## 单因素比较','',
'|固定提示|M−S总体AP50点|L−S总体AP50点|M−S未见AP50点|L−S未见AP50点|','|---|---:|---:|---:|---:|']
for pr,dd in modeldeltas.items():lines.append(f"|{pr}|{fmt(dd['m']['AP50'])}|{fmt(dd['l']['AP50'])}|{fmt(dd['m']['novel_AP50'])}|{fmt(dd['l']['novel_AP50'])}|")
lines+=['','|固定模型|short−original AP50点|descriptive−original AP50点|ensemble−original AP50点|','|---|---:|---:|---:|']
for model,dd in promptdeltas.items():lines.append('|'+model.upper()+'|'+'|'.join(fmt(dd[p]['AP50']) for p in promptnames[1:])+'|')
lines+=['','## 逐类别AP50','',
'|类别|S original|M original|L original|预训练最高组合 '+best+'|S tuned5|S tuned30|','|---|---:|---:|---:|---:|---:|---:|']
for c in results['s_original']['classes']:lines.append('|'+c+'|'+'|'.join(fmt(results[n]['classes'][c]['AP50']) for n in ['s_original','m_original','l_original',best,'s_tuned5','s_tuned30'])+'|')
lines+=['','## 预先固定的提示词','',
'原提示、简短类名和描述性提示如下；ensemble按类别将三种CLIP归一化向量平均后再次L2归一化，保持13个查询，不增加检测查询数量。中途没有根据分数更改提示，也没有逐类别拼装最佳结果。',
'','|类别|original|short|descriptive|','|---|---|---|---|']
cats=list(results['s_original']['classes'])
for i,c in enumerate(cats):lines.append('|'+c+'|'+'|'.join(plan['prompt_sets'][p][i] for p in ['original','short','descriptive'])+'|')
lines+=['','## 固定抽样结果','',
'每组目录下visual_samples保存真值、S tuned5参照、本组预测，样本按固定哈希选择，不依赖效果。局部图只是显示裁剪，不是额外推理。绿色真值、黄色预测，显示阈值0.1；未显示的低置信度候选仍参与AP评测。',
'',f"![预训练最高组合抽样](../../runs/vistas_world_v4_{best}/visual_samples/fixed_samples.jpg)",'',f"![预训练最高组合局部](../../runs/vistas_world_v4_{best}/visual_samples/class_crops_1.jpg)",'',
'## 验证、局限与研究贡献','',
'14组均使用相同开发图清单、COCO真值哈希和推理设置；S tuned5精确复现上轮AP50。官方validation未使用。这里只能称为相对本任务未进行监督适配的预训练推理，不能声称预训练从未见过这些类别。',
'','当前开发集已用于连续实验决策，最高组合有选择偏差；部分类别GT仅10—20个，微小差异不能解释为显著提升。没有新训练，所以本轮未判断M/L的提示微调或检测头训练潜力。没有变更类别、排除困难目标或降低IoU门槛来追求50%。',
'','本轮为研究A选择设施检测的提示策略、预训练基座及后续适配方式提供依据；不代表Logo识别、分割或地理定位已有对应提升。原始预测、影像和权重仅保留本地。',
'','## 来源与完整参数','',
'预训练模型：[Ultralytics官方模型说明](https://docs.ultralytics.com/models/yolo-world/)；[固定v8.3.0权重发布](https://github.com/ultralytics/assets/releases/tag/v8.3.0)。各权重SHA256和实际参数量见下方。','',
'```json',json.dumps({'weights':weights,'fixed':plan['fixed'],'per_case':parameters},indent=2),'```']
(o/'report.md').write_text('\n'.join(lines),encoding='utf8')
pub='\n'.join(lines).replace('(matrix.png)','(figures/vistas-world-matrix-v4.svg)');pub=re.sub(r'!\[[^\]]*\]\(\.\./\.\./runs/[^\n]+\)','（固定抽样影像见本地实验目录，数据集影像不上传公开仓库。）',pub);(o/'report_public.md').write_text(pub,encoding='utf8')
print(json.dumps({'best_unadapted':best,'best_all':bestall,'best_novel':bestnovel,'results':{k:{f:v[f] for f in ['AP50','seen_AP50','novel_AP50','median_end_to_end_seconds','peak_allocated_mib']} for k,v in results.items()}},indent=2))

