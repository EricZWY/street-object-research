# Vistas设施检测诊断：优先解决尺度与识别问题，增加轮次只能带来有限改善

截至2026-11-11。本轮完成既有结果诊断、30轮提示微调对照、候选框上限检查、全图加切片推理检查和真实框裁剪识别代理实验。所有结果来自同一56张开发图（844个目标、13类），正式validation未使用。

## 结论与下一步

1. **优先保留全图上下文并加入原图切片。** 在固定第5轮权重下，AP50从14.91升至18.92（同样最终300框上限）；提高到1000框为19.10。尺度干预比单纯增加训练轮次的收益更大，但当前距离50%仍很远。
2. **将30轮作为下一轮受控实验的观察窗口，并每5轮评测。** 第5轮14.91，第30轮16.12；第25轮16.39是本开发集观察到的最高值，并非独立测试确认的最佳epoch。损失继续降低而AP50波动，不能据此把epoch无限增加，也不能宣称30轮已完全收敛。
3. **切片没有解决未见类识别。** 已见AP50由18.24升至25.07，未见由9.57到9.55。先在固定尺度策略下对照同义词/目标范围描述，并以更大预训练模型做小规模同条件对照；目前没有更大模型实验，无法断言参数量是主因。
4. **暂不全面解冻检测头。** 先完成尺度、prompt及预训练模型的单因素对照，再根据仍然存在的定位/分类错误决定是否训练框回归或有限视觉适配。

## 关键参数

|项目|30轮对照|尺度/候选框诊断|
|---|---|---|
|模型|YOLO-World v2 Small|相同模型，第5轮权重|
|Epoch|30；每5轮评测；报告固定末轮|不训练|
|Batch size|1|1，切片串行|
|学习率|0.0002|不适用|
|输入尺寸|960×960 letterbox|每个全图/切片均960×960|
|训练/开发图|141/56|同56张开发图|
|可训练参数|4096个已见类提示参数，检测网络冻结|0|
|优化器|AdamW，weight decay 0|不适用|
|采样/seed|uniform / 17|固定全开发集|
|切片原图尺寸/步长|无|1024/768，名义重叠25%，保留全图|
|推理阈值/NMS IoU|0.001/0.7|0.001/0.7|
|输出上限|300/图|300及1000/图|
|COCO评测maxDets|[1,10,100]|[1,10,100]，标准设置不变|

训练采用FP32，无增强、预热、调度、梯度累积，梯度裁剪10；冻结网络eval；CUDA分配器限额60%；均在独立可见终端运行。完整配置见文末及每个run的resolved_parameters.json。

## Epoch是否太少

|Epoch|总体AP50|已见AP50|未见AP50|该轮训练平均损失|
|---:|---:|---:|---:|---:|
|0|9.13|8.64|9.90|—|
|5|14.91|18.24|9.57|5.561657352650419|
|10|15.40|18.98|9.68|5.4632218391337295|
|15|15.72|19.47|9.72|5.410835839332418|
|20|15.43|19.00|9.72|5.345933975057399|
|25|16.39|20.61|9.63|5.313101942657579|
|30|16.12|20.16|9.65|5.273671842635946|

前5轮的逐步loss及第5轮AP50与旧实验完全一致，训练/开发样本相同，支持把变化归于训练时长这一受控设置。末轮网络冻结哈希、未见嵌入不变审计均通过。未见嵌入不变并不保证未见检测分数完全不变。

（图表及抽样影像见本地同名报告；数据集影像不上传公开仓库。）

## 目标尺度和候选框诊断

960输入下465/844（55.09%）个目标短边不足8像素。短边用于诊断细小/细长对象，并不等同于COCO面积小目标。下表是每个GT是否存在至少一个IoU≥0.5候选框，不惩罚误报，不做一对一匹配，因此是乐观的**候选覆盖率**，不是AP或标准召回率。阈值为0.001，模型只查询当前13类；不是类别无关proposal网络。

|输入后目标短边|GT数|任意类候选覆盖率|正确类候选覆盖率|
|---|---:|---:|---:|
|short<8|465|10.32|9.46|
|short8-16|191|67.02|61.26|
|short16-32|109|87.16|76.15|
|short>=32|79|91.14|88.61|

（图表及抽样影像见本地同名报告；数据集影像不上传公开仓库。）

53/56张图曾达到300框上限，但放宽至1000后AP50仅14.91→14.93。全图NMS前、分数≥0.001的任意类候选覆盖率43.48%，与1000框后的42.89%接近，说明低分截断/NMS不是全图漏检的主要解释；这仍不是所有可能分数阈值下的理论上限。

|推理方式（均第5轮）|总体AP50|已见AP50|未见AP50|任意类候选覆盖率|正确类候选覆盖率|
|---|---:|---:|---:|---:|---:|
|full300|14.91|18.24|9.57|40.64|37.20|
|full1000|14.93|18.28|9.58|42.89|38.98|
|tiled300|18.92|24.86|9.41|64.45|60.78|
|tiled1000|19.10|25.07|9.55|73.58|69.19|

上述覆盖率按实例汇总，路灯402个、电线杆150个，不能直接与按类别平均的AP50做差。按类别平均的任意类/正确类覆盖率：full300为52.40%/41.38%，tiled1000为80.12%/67.42%。切片显著增加可用候选，但新增低置信度候选与误报、排序问题使AP50收益较小；不能用候选覆盖率宣称达到了50%目标。

推理计时中位数：全图0.090秒/图，全图+切片0.390秒/图；每图1—35片。计时含磁盘读取、预处理、前向和后处理，不含后续COCO汇总；首图排除。

## 逐类变化与语义风险

|类别|开发GT数|5轮全图AP50|30轮全图AP50|5轮切片AP50|真实框CLIP Top1|
|---|---:|---:|---:|---:|---:|
|object--street-light|402|12.23|13.95|27.11|40.05|
|object--junction-box|14|12.79|22.49|26.69|21.43|
|object--trash-can|33|33.88|32.95|45.07|33.33|
|object--fire-hydrant|12|18.10|21.57|41.20|50.00|
|object--manhole|45|28.32|24.93|24.79|13.33|
|object--catch-basin|17|10.68|14.63|3.81|35.29|
|object--support--utility-pole|150|13.55|12.37|16.64|62.00|
|object--sign--store|86|16.38|18.36|15.25|50.00|
|object--cctv-camera|20|0.37|0.59|0.65|35.00|
|object--bench|22|34.15|35.26|36.87|54.55|
|object--mailbox|11|0.32|0.49|0.31|18.18|
|object--bike-rack|22|8.20|7.34|6.06|50.00|
|object--sign--information|10|4.83|4.57|3.88|20.00|

切片使路灯12.23→27.11、接线箱12.79→26.69、消防栓18.10→41.20；但井盖28.32→24.79、排水口10.68→3.81，存在上下文损失/边缘截断/误报等待区分因素。摄像头、邮箱仍接近零。没有把切片+30轮组合效果作为已验证结论。

真实框裁剪使用冻结CLIP ViT-B/32、原图GT框扩大1.25倍、标准CLIP中心裁剪预处理、13个原始文本。类别平均Top1=37.17%，Top3=59.68%。这只是不依赖候选框发现的识别代理；CLIP并非检测器区域分类头，且中心裁剪会丢失细长目标端部，因此不能用此数值宣称检测器分类准确率或能力上限。它提示不能只解决尺度而忽视外观与语义差异。

固定类样本目视检查可见：路灯GT是灯具部分、接线箱有壁挂/落地形态差异，垃圾箱样本被识为邮箱；这些支持核对类别描述和对象范围，但不是整套标注正确性的全面人工审计。

## 可视化抽样

以后每次训练自动保存4张固定抽样全图，以及每类1个固定局部目标，共13个局部样本。按图像ID哈希抽样，选择不依赖预测效果；同数据划分的各次训练可直接比较。局部仅用于展示，不是额外局部推理。绿色为真值，黄色为预测；显示阈值0.1，与AP评测候选阈值0.001不同。低于0.1的候选不会显示。

30轮完整抽样图位于本地 runs/vistas_epoch30_v3/visual_samples/fixed_samples.jpg

（图表及抽样影像见本地同名报告；数据集影像不上传公开仓库。）

（图表及抽样影像见本地同名报告；数据集影像不上传公开仓库。）

其余局部页见相应visual_samples和scale_visuals目录；包含漏检和误检，不仅选成功案例。影像及原始逐条数据仅保留本地，不上传公开仓库。

## 对研究整体的贡献与边界

本轮把设施识别低分拆分成可验证的问题：尺度干预有实际收益，增加epoch有有限收益，候选数量上限影响很小，未见类别识别仍未解决。下一轮优先在全图+切片框架下做prompt描述与更强预训练模型对照；30轮、每5轮评测作为适配实验的观察窗口。保持seen/novel划分，不为提高分数事后移除小目标或困难类别。

该证据仅作用于研究A的设施检测可行性，不等于Logo识别、分割或地理位置精度改善。单种子、小开发集、部分类别仅10—20个目标，排名不稳定；本开发集已用于多次决策，应保留正式validation用于后续冻结设置后的评测。

## 完整配置与复现

源脚本：scripts/diagnose_vistas_detection.py、probe_vistas_scale.py、summarize_vistas_diagnostic.py、visualize_vistas_run.py、train_vistas_prompts.py。每次训练仍经Start-Training.ps1启动可见终端。运行目录runs/vistas_epoch30_v3保存全部30轮提示权重、每轮采样顺序、逐步loss、每5轮预测、完整参数、训练脚本快照与日志。

```json
{
  "training": {
    "manifest": {
      "categories": [
        "object--street-light",
        "object--junction-box",
        "object--trash-can",
        "object--fire-hydrant",
        "object--manhole",
        "object--catch-basin",
        "object--support--utility-pole",
        "object--sign--store",
        "object--cctv-camera",
        "object--bench",
        "object--mailbox",
        "object--bike-rack",
        "object--sign--information"
      ],
      "prompts": [
        "street light fixture",
        "electrical junction box",
        "trash can",
        "fire hydrant",
        "manhole cover",
        "storm drain",
        "utility pole",
        "store sign",
        "surveillance camera",
        "bench",
        "mailbox",
        "bicycle rack",
        "information sign"
      ],
      "epochs": 30,
      "lr": 0.0002,
      "seed": 17,
      "image_size": 960,
      "batch": 1,
      "precision": "FP32",
      "train_selection": "32 images per seen class, SHA256 order, union; excludes annotated novel classes/crowds",
      "dev_selection": "8 images per class, union; excludes previously inspected IDs and training IDs/image SHA256",
      "bbox": "mask min inclusive max exclusive; positive pixel-area retained",
      "sampler": "uniform",
      "optimizer": "AdamW only seen embedding, weight_decay=0",
      "ultralytics": "8.3.93",
      "torch": "2.5.1+cu121",
      "official_validation_used": false,
      "train_count": 141,
      "dev_count": 56,
      "train_list_sha256": "b04c305d84a84e82c9deb948eab4fbdbcba4da37024b24f1415a44180481edd0",
      "dev_list_sha256": "a0caa9158d7e187cc1728897a24fa8c1b0eee82121e20dae968463720a2bd5c7"
    },
    "optimizer_defaults": {
      "lr": 0.0002,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-08,
      "weight_decay": 0,
      "amsgrad": false,
      "foreach": null,
      "maximize": false,
      "capturable": false,
      "differentiable": false,
      "fused": null
    },
    "loss_gains": {
      "box": 7.5,
      "cls": 0.5,
      "dfl": 1.5
    },
    "scheduler": null,
    "warmup_steps": 0,
    "gradient_accumulation_steps": 1,
    "gradient_clip_norm": 10,
    "augmentation": "none; OpenCV linear resize, centered pad114 to configured input size, BGR to RGB, divide255",
    "num_workers": 0,
    "cpu_threads": 4,
    "cuda_memory_fraction": 0.6,
    "trainable_parameters": 4096,
    "frozen_detector_eval_mode": true,
    "prompt_normalization": "L2 each forward",
    "checkpoint_sha256": "9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792",
    "script_sha256": "3d055de58763b939aa47628ca18095611b14bdda12201de02afc12d170e2fd2d",
    "python": "3.11.5 | packaged by Anaconda, Inc. | (main, Sep 11 2023, 13:26:23) [MSC v.1916 64 bit (AMD64)]",
    "platform": "Windows-10-10.0.19045-SP0",
    "cuda_runtime": "12.1",
    "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU",
    "evaluation": {
      "imgsz": 960,
      "conf": 0.001,
      "nms_iou": 0.7,
      "inference_max_det": 300,
      "coco_maxDets": [
        1,
        10,
        100
      ],
      "iou_range": [
        0.5,
        0.95,
        0.05
      ]
    },
    "checkpoint_selection": "last epoch; periodic development metrics are diagnostic, no early stopping",
    "eval_every": 5,
    "visualization": "4 fixed hash-selected dev images plus one fixed crop per class; score>=0.1"
  },
  "training_audit": {
    "detector_state_unchanged": true,
    "frozen_state_sha256": "79b971d42e0b7184b4398a1a1547d1427d036a51280ccefaffce2d2bc2e26f42",
    "novel_embeddings_bitwise_unchanged": true,
    "trainable_parameters": 4096,
    "seen_embedding_l2_change": 1.0773062705993652,
    "peak_allocated_mib": 549.52294921875,
    "peak_reserved_mib": 754.0
  },
  "scale_probe": {
    "source_run": "vistas_sweep_v2_highres",
    "input": 960,
    "tile_original_pixels": 1024,
    "tile_stride": 768,
    "overlap_nominal": 0.25,
    "global_full_image_included": true,
    "tile_nms_max_det": 1000,
    "global_max_dets_compared": [
      300,
      1000
    ],
    "score": 0.001,
    "nms_iou": 0.7,
    "coco_maxDets": [
      1,
      10,
      100
    ],
    "model": "YOLO-World-v2-S",
    "prompt_checkpoint_epoch": 5,
    "precision": "FP32",
    "cuda_fraction": 0.6,
    "script_sha256": "ac08e45a0abb77fb7b7de9a1e7d07a90d61395a445b2eb449d4b69af2931476d"
  },
  "crop_proxy_method": "Frozen CLIP ViT-B/32, original-pixel GT box expanded 1.25x, CLIP standard center-crop preprocessing, 13 supplied texts, all 844 objects. Closed-set recognition proxy, not detector accuracy; center crop can remove ends of elongated objects.",
  "validation": {
    "first_5_epochs_loss_exactly_reproduced": true,
    "first_5_epochs_AP50_exactly_reproduced": true,
    "same_train_dev": true,
    "same_full300_inference_reproduced": true
  }
}
```
