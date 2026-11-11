# 独立提示微调改进实验（2026-11-11）

## 本轮结论与研究贡献
本轮取得可继续验证的正向结果，优先候选为960输入、lr=0.0002的类别独立提示微调。它在同分辨率下相对零样本总体AP由6.49升至9.64（+3.16点），已见类6.32→11.49（+5.17点），未见类6.75→6.69（-0.06点）。候选根据本开发集选出，尚非独立测试确认的最佳方案。
研究A新增证据：冻结网络仅学习4096个提示参数，可改善已见设施的检测；保存较多输入像素与提示适配可以同时带来收益。未见性能大体接近相同分辨率零样本，但摄像头/邮箱仍很弱，不能宣称广泛开放词汇问题已解决。
研究B未新增定位证据；此次提升不等于地理位置精度提升。Logo也没有在本轮重新评测。研究主线暂不调整，下一步需独立样本/多种子检验，同时继续补充实例定位真值。

## 各方向判断
- 降低学习率：总体AP6.11→6.32（+0.21点）。更新幅度更小，只有轻微开发收益，无显著性结论。
- 类别循环采样：在低学习率上6.32→6.60（+0.28点），已见AP7.83→8.30；未见3.90→3.87。循环抽选使消防栓实例曝光170→291、垃圾箱360→499，但电线杆也因共现而增多。实际仅抽到训练池中的136/141张，不能称作完全均衡。
- 提高分辨率：960零样本本身为6.49，高于640零样本4.66。960微调进一步达到9.64；不能将全部相对640收益都归于训练，也没有把训练/推理尺寸的作用分别隔离。
- 所有已见类别并非同时获益：960消防栓AP17.69→15.12，垃圾箱22.58→23.01近乎持平；井盖6.12→18.50、接线箱0→9.08改善较明显。下一步要查少样本、边界与置信度排序问题。

## 案例与下一阶段
已目视核验两张本地对照图。case_0.jpg是960方案中固定阈值(score≥0.1、IoU≥0.5)匹配数增益最大的开发图：正确匹配0→7，但预测框也明显增多，出现疑似误检；不能只展示新增检出而忽略误报。case_1.jpg是增益最小图（0→0，属于持续失败而非退化），远处/细小设施仍未形成正确匹配。该选择用于诊断，不代表总体分布；AP采用0.001阈值候选，不能与图片展示阈值混淆。
下一步优先固定960低学习率方案，在独立开发检查样本及多个随机种子上复核；再测试960与类别循环采样的组合，不能假定两者收益可相加。之后冻结完整设置，进行正式validation评测。现阶段不增加模型结构或分割模块。当前总体AP仍偏低，需保留漏检、误检与目标范围不一致三类问题的单独诊断。


## 关键参数

| 方案 | Epoch | Batch | LR | 输入 | 图片采样 | 训练/开发图片 |
|---|---:|---:|---:|---:|---|---:|
|control|5|1|0.002|640|uniform|141/56|
|lowlr|5|1|0.0002|640|uniform|141/56|
|balanced|5|1|0.0002|640|class_cycle|141/56|
|highres|5|1|0.0002|960|uniform|141/56|

共同参数：YOLO-World v2 Small；8已见/5未见；4096可训练提示参数；检测网络冻结并eval；FP32；AdamW，betas=(0.9,0.999)，eps=1e-8，weight_decay=0；梯度裁剪10；无调度/预热/梯度累积/数据增强；seed17；CPU线程4；CUDA分配器限额60%；框损失7.5、分类0.5、DFL1.5。
所有训练在独立可见终端串行运行，逐步记录日志与阶段ETA；固定末轮，没有事后挑最佳epoch。

## 开发结果（0—100）

| 方案 | 已见AP | 未见AP | 总体AP | AP50 | 本分辨率零样本总体AP |
|---|---:|---:|---:|---:|---:|
|control|7.47|3.94|6.11|9.75|4.66|
|lowlr|7.83|3.90|6.32|9.88|4.66|
|balanced|8.30|3.87|6.60|10.23|4.66|
|highres|11.49|6.69|9.64|14.91|6.49|

## 逐类别AP

| 类别 | 640零样本 | control | lowlr | balanced | 960零样本 | highres |
|---|---:|---:|---:|---:|---:|---:|
|object--street-light|0.04|4.11|3.80|4.09|0.09|6.71|
|object--junction-box|0.00|4.33|5.48|6.26|0.00|9.08|
|object--trash-can|16.87|15.70|16.64|16.69|22.58|23.01|
|object--fire-hydrant|14.42|15.69|15.33|16.50|17.69|15.12|
|object--manhole|4.09|6.81|10.26|11.30|6.12|18.50|
|object--catch-basin|0.00|1.92|1.11|0.21|0.00|5.48|
|object--support--utility-pole|2.59|4.54|5.08|4.93|2.30|5.33|
|object--sign--store|2.12|6.63|4.99|6.44|1.78|8.70|
|object--cctv-camera|0.14|0.00|0.00|0.00|0.13|0.07|
|object--bench|18.00|17.90|17.07|17.96|27.48|26.23|
|object--mailbox|0.06|0.00|0.00|0.01|0.28|0.18|
|object--bike-rack|0.90|0.55|0.85|0.78|1.86|2.76|
|object--sign--information|1.29|1.25|1.59|0.60|4.00|4.20|

## 耗时与显存

|方案|总耗时/秒|峰值allocated/MiB|峰值reserved/MiB|
|---|---:|---:|---:|
|control|207.9|306.0|692.0|
|lowlr|199.9|306.0|692.0|
|balanced|200.5|306.0|692.0|
|highres|205.3|549.5|754.0|

## 公平性和局限
四组训练/开发ID、GT完全一致，640三组零样本AP一致，冻结网络和未见嵌入审计全部通过。开发样本排除先前检查过的ID及本轮训练ID/同字节影像，但未保证空间或近重复独立。正式validation未参与。
按类选图合并会造成共现实例不均衡。class_cycle仅平衡被抽选类别次数，不保证每类实例数相同，允许重复图；每轮总更新步数相同。
高分辨率方案同时改变训练与推理尺寸，因此先比较其对应零样本基线，再比较微调增益；不能把相对640的所有提升归因于prompt tuning。
单随机种子、有限开发集，无显著性主张；本轮用于选择后续方案，不替代最终独立测试。候选样本、像素派生框和全部权重留在本地。
评测采用COCO bbox AP IoU .50:.95，maxDets100，推理阈值.001/NMS.7/max300；没有分割评测。

## 完整参数附录
各运行report.md也包含关键参数表和完整配置；本汇总略去原始样本ID，仅保留数量和清单哈希。

### control

```json
{
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
    "epochs": 5,
    "lr": 0.002,
    "seed": 17,
    "image_size": 640,
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
    "sample_id_list_sha256": "e0ef01b416c8b39de2dfa4f7daa96e94d25625324c79f7237f3b5657cc04f7d2"
  },
  "optimizer_defaults": {
    "lr": 0.002,
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
  "script_sha256": "2f0c0a3811d8bed2d6b6f280b07c5456edb46d515ec9479fa21347d8c1552999",
  "python": "3.11.5 | packaged by Anaconda, Inc. | (main, Sep 11 2023, 13:26:23) [MSC v.1916 64 bit (AMD64)]",
  "platform": "Windows-10-10.0.19045-SP0",
  "cuda_runtime": "12.1",
  "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU",
  "evaluation": {
    "imgsz": 640,
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
  "checkpoint_selection": "last epoch; no early stopping"
}
```

### lowlr

```json
{
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
    "epochs": 5,
    "lr": 0.0002,
    "seed": 17,
    "image_size": 640,
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
    "sample_id_list_sha256": "e0ef01b416c8b39de2dfa4f7daa96e94d25625324c79f7237f3b5657cc04f7d2"
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
  "script_sha256": "2f0c0a3811d8bed2d6b6f280b07c5456edb46d515ec9479fa21347d8c1552999",
  "python": "3.11.5 | packaged by Anaconda, Inc. | (main, Sep 11 2023, 13:26:23) [MSC v.1916 64 bit (AMD64)]",
  "platform": "Windows-10-10.0.19045-SP0",
  "cuda_runtime": "12.1",
  "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU",
  "evaluation": {
    "imgsz": 640,
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
  "checkpoint_selection": "last epoch; no early stopping"
}
```

### balanced

```json
{
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
    "epochs": 5,
    "lr": 0.0002,
    "seed": 17,
    "image_size": 640,
    "batch": 1,
    "precision": "FP32",
    "train_selection": "32 images per seen class, SHA256 order, union; excludes annotated novel classes/crowds",
    "dev_selection": "8 images per class, union; excludes previously inspected IDs and training IDs/image SHA256",
    "bbox": "mask min inclusive max exclusive; positive pixel-area retained",
    "sampler": "class_cycle",
    "optimizer": "AdamW only seen embedding, weight_decay=0",
    "ultralytics": "8.3.93",
    "torch": "2.5.1+cu121",
    "official_validation_used": false,
    "train_count": 141,
    "dev_count": 56,
    "sample_id_list_sha256": "e0ef01b416c8b39de2dfa4f7daa96e94d25625324c79f7237f3b5657cc04f7d2"
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
  "script_sha256": "2f0c0a3811d8bed2d6b6f280b07c5456edb46d515ec9479fa21347d8c1552999",
  "python": "3.11.5 | packaged by Anaconda, Inc. | (main, Sep 11 2023, 13:26:23) [MSC v.1916 64 bit (AMD64)]",
  "platform": "Windows-10-10.0.19045-SP0",
  "cuda_runtime": "12.1",
  "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU",
  "evaluation": {
    "imgsz": 640,
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
  "checkpoint_selection": "last epoch; no early stopping"
}
```

### highres

```json
{
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
    "epochs": 5,
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
    "sample_id_list_sha256": "e0ef01b416c8b39de2dfa4f7daa96e94d25625324c79f7237f3b5657cc04f7d2"
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
  "script_sha256": "2f0c0a3811d8bed2d6b6f280b07c5456edb46d515ec9479fa21347d8c1552999",
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
  "checkpoint_selection": "last epoch; no early stopping"
}
```
