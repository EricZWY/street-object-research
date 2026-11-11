"""Build reproducible diagnostic report, curves and fixed visual comparisons."""
import collections,hashlib,json
from pathlib import Path
import argparse,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from visualize_vistas_run import render_run
ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,required=True);a=ap.parse_args();w=a.workspace;o=w/'reports/vistas_diagnostic_v3';r=w/'runs/vistas_epoch30_v3';old=w/'runs/vistas_sweep_v2_highres'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(n,x):(o/n).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf8')
def pct(v):return f'{100*v:.2f}'
d=read(o/'diagnosis.json');p=read(o/'scale_probe.json');c=read(o/'crop_proxy.json');m=read(r/'manifest.json');params=read(r/'resolved_parameters.json');audit=read(r/'audit.json');loss=read(r/'loss.json');checkpoints=[]
assert read(r/'status.json')['status']=='completed' and read(o/'probe_status.json')['status']=='completed'
assert read(old/'manifest.json')['train']==m['train'] and read(old/'manifest.json')['dev']==m['dev']
assert read(old/'loss.json')==loss[:705]
for ep in [0,5,10,15,20,25,30]:
    ss=read(r/(('baseline' if ep==0 else 'tuned' if ep==30 else f'epoch_{ep}')+'_summary.json'))
    checkpoints.append({'epoch':ep,'AP50':ss['AP50'],'seen_AP50':ss['seen_AP50'],'novel_AP50':ss['unseen_AP50'],'mean_train_loss':None if ep==0 else float(np.mean([x['loss'] for x in loss if x['epoch']==ep]))})
assert abs(checkpoints[1]['AP50']-p['results']['full300']['AP50'])<1e-10
fig,axs=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
for key,label in [('AP50','All'),('seen_AP50','Seen'),('novel_AP50','Novel')]:axs[0].plot([x['epoch'] for x in checkpoints],[100*x[key] for x in checkpoints],marker='o',label=label)
axs[0].set(xlabel='Epoch',ylabel='Development AP50 (%)',ylim=(0,30),title='Longer prompt training: modest gain');axs[0].legend()
epochs=sorted(set(x['epoch'] for x in loss));axs[1].plot(epochs,[np.mean([x['loss'] for x in loss if x['epoch']==ep]) for ep in epochs],marker='.');axs[1].set(xlabel='Epoch',ylabel='Mean training loss',title='Training loss keeps declining')
names=list(p['results']);values=[100*p['results'][n]['AP50'] for n in names];bars=axs[2].bar(names,values,color=['#94a3b8','#64748b','#38bdf8','#0284c7']);axs[2].bar_label(bars,fmt='%.2f');axs[2].set(ylabel='Development AP50 (%)',ylim=(0,55),title='Fixed epoch-5 weights: scale vs cap');axs[2].axhline(50,color='#b91c1c',linestyle='--',label='Target 50');axs[2].legend()
fig.savefig(o/'epoch_and_scale.png',dpi=150);plt.close(fig)
render_run(w,old,out=o/'scale_visuals',base_path=o/'full300_predictions_local.json',tuned_path=o/'tiled1000_predictions_local.json',labels=('Full image, cap 300','Full + tiles, cap 1000'),caption='Scale probe: same epoch-5 weights')
# Evidence and publication-safe metadata (no image IDs/predictions).
safeparams=json.loads(json.dumps(params));safeparams['manifest'].pop('train');safeparams['manifest'].pop('dev');safeparams['manifest']['train_count']=len(m['train']);safeparams['manifest']['dev_count']=len(m['dev'])
for split in ['train','dev']:safeparams['manifest'][split+'_list_sha256']=hashlib.sha256(json.dumps(m[split]).encode()).hexdigest()
timing={'full_median_seconds':float(np.median([x['full_seconds'] for x in p['timings'][1:]])),'full_plus_tiles_median_seconds':float(np.median([x['full_seconds']+x['tile_extra_seconds'] for x in p['timings'][1:]])),'tiles_per_image_min':min(x['tiles'] for x in p['counts']),'tiles_per_image_max':max(x['tiles'] for x in p['counts'])}
summary={'epoch_checkpoints':checkpoints,'scale_results':p['results'],'scale_parameters':p['parameters'],'timings':timing,'crop_proxy':c,'training_parameters':safeparams,'training_audit':audit,'checks':{'first_5_epochs_loss_exactly_reproduced':True,'first_5_epochs_AP50_exactly_reproduced':True,'same_train_dev':True,'same_full300_inference_reproduced':p['full300_reproduced']},'raw_pre_nms_any_coverage':p['raw_pre_nms_any_coverage']};save('summary.json',summary)
lines=['# Vistas设施检测诊断：优先解决尺度与识别问题，增加轮次只能带来有限改善','',
'截至2026-11-11。本轮完成既有结果诊断、30轮提示微调对照、候选框上限检查、全图加切片推理检查和真实框裁剪识别代理实验。所有结果来自同一56张开发图（844个目标、13类），正式validation未使用。',
'','## 结论与下一步','',
'1. **优先保留全图上下文并加入原图切片。** 在固定第5轮权重下，AP50从14.91升至18.92（同样最终300框上限）；提高到1000框为19.10。尺度干预比单纯增加训练轮次的收益更大，但当前距离50%仍很远。',
'2. **将30轮作为下一轮受控实验的观察窗口，并每5轮评测。** 第5轮14.91，第30轮16.12；第25轮16.39是本开发集观察到的最高值，并非独立测试确认的最佳epoch。损失继续降低而AP50波动，不能据此把epoch无限增加，也不能宣称30轮已完全收敛。',
'3. **切片没有解决未见类识别。** 已见AP50由18.24升至25.07，未见由9.57到9.55。先在固定尺度策略下对照同义词/目标范围描述，并以更大预训练模型做小规模同条件对照；目前没有更大模型实验，无法断言参数量是主因。',
'4. **暂不全面解冻检测头。** 先完成尺度、prompt及预训练模型的单因素对照，再根据仍然存在的定位/分类错误决定是否训练框回归或有限视觉适配。',
'','## 关键参数','',
'|项目|30轮对照|尺度/候选框诊断|','|---|---|---|','|模型|YOLO-World v2 Small|相同模型，第5轮权重|','|Epoch|30；每5轮评测；报告固定末轮|不训练|','|Batch size|1|1，切片串行|','|学习率|0.0002|不适用|','|输入尺寸|960×960 letterbox|每个全图/切片均960×960|','|训练/开发图|141/56|同56张开发图|','|可训练参数|4096个已见类提示参数，检测网络冻结|0|','|优化器|AdamW，weight decay 0|不适用|','|采样/seed|uniform / 17|固定全开发集|','|切片原图尺寸/步长|无|1024/768，名义重叠25%，保留全图|','|推理阈值/NMS IoU|0.001/0.7|0.001/0.7|','|输出上限|300/图|300及1000/图|','|COCO评测maxDets|[1,10,100]|[1,10,100]，标准设置不变|',
'','训练采用FP32，无增强、预热、调度、梯度累积，梯度裁剪10；冻结网络eval；CUDA分配器限额60%；均在独立可见终端运行。完整配置见文末及每个run的resolved_parameters.json。',
'','## Epoch是否太少','',
'|Epoch|总体AP50|已见AP50|未见AP50|该轮训练平均损失|','|---:|---:|---:|---:|---:|']
for x in checkpoints:lines.append(f"|{x['epoch']}|{pct(x['AP50'])}|{pct(x['seen_AP50'])}|{pct(x['novel_AP50'])}|{x['mean_train_loss'] if x['mean_train_loss'] is not None else '—'}|")
lines+=['','前5轮的逐步loss及第5轮AP50与旧实验完全一致，训练/开发样本相同，支持把变化归于训练时长这一受控设置。末轮网络冻结哈希、未见嵌入不变审计均通过。未见嵌入不变并不保证未见检测分数完全不变。','', '![训练曲线与尺度干预](epoch_and_scale.png)','', '## 目标尺度和候选框诊断','',
'960输入下465/844（55.09%）个目标短边不足8像素。短边用于诊断细小/细长对象，并不等同于COCO面积小目标。下表是每个GT是否存在至少一个IoU≥0.5候选框，不惩罚误报，不做一对一匹配，因此是乐观的**候选覆盖率**，不是AP或标准召回率。阈值为0.001，模型只查询当前13类；不是类别无关proposal网络。',
'','|输入后目标短边|GT数|任意类候选覆盖率|正确类候选覆盖率|','|---|---:|---:|---:|']
for name,x in d['highres']['methods']['tuned']['size_bins'].items():lines.append(f"|{name}|{x['n']}|{pct(x['any_coverage_iou0.5_score0.001'])}|{pct(x['same_coverage_iou0.5_score0.001'])}|")
lines+=['','![原5轮诊断](diagnosis.png)','',
'53/56张图曾达到300框上限，但放宽至1000后AP50仅14.91→14.93。全图NMS前、分数≥0.001的任意类候选覆盖率43.48%，与1000框后的42.89%接近，说明低分截断/NMS不是全图漏检的主要解释；这仍不是所有可能分数阈值下的理论上限。',
'','|推理方式（均第5轮）|总体AP50|已见AP50|未见AP50|任意类候选覆盖率|正确类候选覆盖率|','|---|---:|---:|---:|---:|---:|']
for name,x in p['results'].items():lines.append(f"|{name}|{pct(x['AP50'])}|{pct(x['seen_AP50'])}|{pct(x['novel_AP50'])}|{pct(x['candidate_coverage']['any'])}|{pct(x['candidate_coverage']['same'])}|")
lines+=['',
'上述覆盖率按实例汇总，路灯402个、电线杆150个，不能直接与按类别平均的AP50做差。按类别平均的任意类/正确类覆盖率：full300为52.40%/41.38%，tiled1000为80.12%/67.42%。切片显著增加可用候选，但新增低置信度候选与误报、排序问题使AP50收益较小；不能用候选覆盖率宣称达到了50%目标。',
'',f"推理计时中位数：全图{timing['full_median_seconds']:.3f}秒/图，全图+切片{timing['full_plus_tiles_median_seconds']:.3f}秒/图；每图{timing['tiles_per_image_min']}—{timing['tiles_per_image_max']}片。计时含磁盘读取、预处理、前向和后处理，不含后续COCO汇总；首图排除。",'',
'## 逐类变化与语义风险','',
'|类别|开发GT数|5轮全图AP50|30轮全图AP50|5轮切片AP50|真实框CLIP Top1|','|---|---:|---:|---:|---:|---:|']
final=read(r/'tuned_summary.json')
for name in m['categories']:lines.append(f"|{name}|{c['classes'][name]['n']}|{pct(p['results']['full300']['classes'][name]['AP50'])}|{pct(final['classes'][name]['AP50'])}|{pct(p['results']['tiled1000']['classes'][name]['AP50'])}|{pct(c['classes'][name]['top1'])}|")
lines+=['',
'切片使路灯12.23→27.11、接线箱12.79→26.69、消防栓18.10→41.20；但井盖28.32→24.79、排水口10.68→3.81，存在上下文损失/边缘截断/误报等待区分因素。摄像头、邮箱仍接近零。没有把切片+30轮组合效果作为已验证结论。',
'',f"真实框裁剪使用冻结CLIP ViT-B/32、原图GT框扩大1.25倍、标准CLIP中心裁剪预处理、13个原始文本。类别平均Top1={pct(c['macro_top1'])}%，Top3={pct(c['macro_top3'])}%。这只是不依赖候选框发现的识别代理；CLIP并非检测器区域分类头，且中心裁剪会丢失细长目标端部，因此不能用此数值宣称检测器分类准确率或能力上限。它提示不能只解决尺度而忽视外观与语义差异。",'',
'固定类样本目视检查可见：路灯GT是灯具部分、接线箱有壁挂/落地形态差异，垃圾箱样本被识为邮箱；这些支持核对类别描述和对象范围，但不是整套标注正确性的全面人工审计。',
'','## 可视化抽样','',
'以后每次训练自动保存4张固定抽样全图，以及每类1个固定局部目标，共13个局部样本。按图像ID哈希抽样，选择不依赖预测效果；同数据划分的各次训练可直接比较。局部仅用于展示，不是额外局部推理。绿色为真值，黄色为预测；显示阈值0.1，与AP评测候选阈值0.001不同。低于0.1的候选不会显示。',
'','[30轮完整抽样图](../../runs/vistas_epoch30_v3/visual_samples/fixed_samples.jpg)','',
'![30轮每类局部样本第一页](../../runs/vistas_epoch30_v3/visual_samples/class_crops_1.jpg)','',
'![全图与切片相同样本](scale_visuals/class_crops_1.jpg)','',
'其余局部页见相应visual_samples和scale_visuals目录；包含漏检和误检，不仅选成功案例。影像及原始逐条数据仅保留本地，不上传公开仓库。',
'','## 对研究整体的贡献与边界','',
'本轮把设施识别低分拆分成可验证的问题：尺度干预有实际收益，增加epoch有有限收益，候选数量上限影响很小，未见类别识别仍未解决。下一轮优先在全图+切片框架下做prompt描述与更强预训练模型对照；30轮、每5轮评测作为适配实验的观察窗口。保持seen/novel划分，不为提高分数事后移除小目标或困难类别。',
'','该证据仅作用于研究A的设施检测可行性，不等于Logo识别、分割或地理位置精度改善。单种子、小开发集、部分类别仅10—20个目标，排名不稳定；本开发集已用于多次决策，应保留正式validation用于后续冻结设置后的评测。',
'','## 完整配置与复现','',
'源脚本：scripts/diagnose_vistas_detection.py、probe_vistas_scale.py、summarize_vistas_diagnostic.py、visualize_vistas_run.py、train_vistas_prompts.py。每次训练仍经Start-Training.ps1启动可见终端。运行目录runs/vistas_epoch30_v3保存全部30轮提示权重、每轮采样顺序、逐步loss、每5轮预测、完整参数、训练脚本快照与日志。','',
'```json',json.dumps({'training':safeparams,'training_audit':audit,'scale_probe':p['parameters'],'crop_proxy_method':c['method'],'validation':summary['checks']},indent=2,ensure_ascii=False),'```']
(o/'report.md').write_text('\n'.join(lines),encoding='utf8')
print(json.dumps({'epochs':checkpoints,'timing':timing,'report':str(o/'report.md')},indent=2))

