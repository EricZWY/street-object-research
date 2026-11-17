# 本轮改动的文件说明（Grounding DINO-T 基线 + 已见类扩样）

本轮只新增文件，没有修改任何既有脚本、既有 `runs/`、`reports/` 产物，也没有删除任何东西。所有新增命名都带 `_ds` 后缀，结果目录与之前完全分开，便于和 GPT 侧的实验区分。

## 1. 新增代码

|文件|作用|
|---|---|
|`scripts/download_gdino_ds.py`|从 Hugging Face 下载 `IDEA-Research/grounding-dino-tiny`，逐文件记录字节数与 SHA256|
|`scripts/probe_pools_ds.py`|只读探针：统计 Vistas 里每个已见类可用的“全已见类标注图”数量，以及每类取前 N 张并集后的规模|
|`scripts/evaluate_gdino_ds.py`|Grounding DINO-T 零样本推理基线（不训练），几何与置信度口径同 YOLO-World 组|
|`scripts/train_seen_expand_ds.py`|已见类扩样训练（`--mode prompt` 只学 8 条已见提示向量；`--mode head` 另加检测头）|
|`scripts/summarize_ds.py`|汇总 `runs_ds/` 下的结果，出对比表、逐类表、对比图，可同时写入公开仓库|
|`scripts/Start-GdinoEval_ds.ps1`|独立可见窗口跑 GDINO 评测，日志与 ETA 写 `runs_ds/console_logs/`|
|`scripts/Start-SeenExpand_ds.ps1`|独立可见窗口跑扩样训练，日志与 ETA 写 `runs_ds/console_logs/`|

## 2. 新增目录（均在工作区，不进公开仓库）

|目录|内容|
|---|---|
|`runs_ds/`|本轮所有运行目录（每次运行一个子目录，`mkdir(exist_ok=False)`，拒绝覆盖）|
|`runs_ds/console_logs/`|每轮的终端日志与退出码|
|`reports_ds/`|本轮汇总报告、对比图、汇总 JSON|
|`deps_ds/`|隔离安装的 `transformers 4.46.3` / `tokenizers 0.20.3` / `regex`|
|`weights/GDino_T_ds/`|Grounding DINO-T 权重与下载清单|

## 3. 依赖处理

环境里原有 `transformers 4.30.1`，没有 `GroundingDinoForObjectDetection`，但直接升级有破坏 YOLO-World 提示微调链路的可能。因此没有改动环境，改为隔离安装：

```powershell
python -m pip install --target E:\CodexWorkspace\StreetObject_DataPlan\deps_ds --no-deps transformers==4.46.3 tokenizers==0.20.3 regex
```

`evaluate_gdino_ds.py` 在导入 transformers 之前把 `deps_ds` 插到 `sys.path[0]`，所以同一台机器上两套 transformers 互不影响。

这样也避开了 Grounding DINO 官方仓库 `MultiScaleDeformableAttention` 的 CUDA 编译（Windows 上要装 MSVC 工具链，容易失败）。transformers 4.46.3 自带纯 PyTorch 版实现，加载时会打印

```text
Could not load the custom kernel for multi-scale deformable attention: ...
```

这是预期的降级路径，不影响结果，只是慢一些。

## 4. 两个实现细节（踩过坑，写下来避免复现）

**下载进度不能用文件大小判断。** Hugging Face 下载器会预分配整个文件大小并按块乱序写入，所以“文件已达 689,359,096 字节”不代表下完；中途读头部全是 0 也不是文件损坏，只是那一段还没写。判断进度要按固定间隔抽样看哪些块非零。另外 `.cache/huggingface/download/*.incomplete` 支持断点续传，如果续接一个被中断的稀疏文件，会瞬间“完成”并沿用不完整内容，必须整个删掉缓存重下。本轮第二次下载耗时 765.6 秒，`model.safetensors` SHA256 为 `bc14aae4ebb73c48291f36d351696fb57bda62ab206506037a1931f8703cab09`。

**训练时的文本列数必须等于 `head.nc`。** YOLO-World 的 `WorldDetect` 里类别通道来自对比头对文本矩阵的投影，所以文本矩阵有几行，检测头就输出几个类别通道。训练只用已见类时，文本矩阵必须是 8 行（不是 13 行拼上未见类），否则 `xi.view(shape[0], self.nc + self.reg_max * 4, -1)` 会因通道数不匹配报错 `shape '[1, 72, -1]' is invalid for input of size 69300`。未见类向量只在评测时拼接。

**评测时必须把文本列换回 13 行。** 同一个原因在评测方向也会出错：`head.nc=13` 而文本矩阵还是训练时的 8 行，会报 `shape '[1, 77, -1]' is invalid for input of size 1036800`（1036800 = 72×14400）。因此 `evaluate()` 开头必须重新拼出 8＋5＝13 行的文本矩阵，再进入推理。第一次 10 轮扩样运行就是在第 5 轮评测时踩到这个错误中止的，该运行目录 `runs_ds/seen_expand_prompt_c160_e10_ds` 按“不删既有产物”原则原样保留，实际报告用的是修复后重跑的同配置运行 `..._e10b_ds`。

## 5. 复现命令

```powershell
# 1) 下载权重（只需一次）
python scripts\download_gdino_ds.py --workspace E:\CodexWorkspace\StreetObject_DataPlan

# 2) 已见类可用池探测（只读，可随时重跑）
python scripts\probe_pools_ds.py --workspace E:\CodexWorkspace\StreetObject_DataPlan

# 3) GDINO 零样本基线（独立窗口，约 10 分钟）
powershell -ExecutionPolicy Bypass -File scripts\Start-GdinoEval_ds.ps1 -Workspace E:\CodexWorkspace\StreetObject_DataPlan

# 4) 扩样训练（独立窗口；先 prepare 一次，后续用 --reuse-split 省去重解码）
python scripts\train_seen_expand_ds.py --workspace E:\CodexWorkspace\StreetObject_DataPlan --run-name prepare_split_c160_ds --train-per-class 160 --prepare-only
powershell -ExecutionPolicy Bypass -File scripts\Start-SeenExpand_ds.ps1 -Mode prompt -Epochs 10 -TrainPerClass 160 -ReuseSplit E:\CodexWorkspace\StreetObject_DataPlan\runs_ds\prepare_split_c160_ds
powershell -ExecutionPolicy Bypass -File scripts\Start-SeenExpand_ds.ps1 -Mode prompt -Epochs 30 -TrainPerClass 160 -ReuseSplit E:\CodexWorkspace\StreetObject_DataPlan\runs_ds\prepare_split_c160_ds
powershell -ExecutionPolicy Bypass -File scripts\Start-SeenExpand_ds.ps1 -Mode head   -Epochs 10 -TrainPerClass 160 -ReuseSplit E:\CodexWorkspace\StreetObject_DataPlan\runs_ds\prepare_split_c160_ds

# 5) 汇总（并写入公开仓库的 results/ 与 docs/figures/）
python scripts\summarize_ds.py --workspace E:\CodexWorkspace\StreetObject_DataPlan --repo E:\DSWS\street-object-research
```

## 6. 口径一致性

两个实验都与既有的 `vistas_world_matrix_v4` 组共用同一批 56 张开发图、同一份 COCO 真值（844 个框）、同一套推理几何（960 letterbox、原图 1024 像素切片、步长 768、全图＋切片、每视图与合并后各 1000）、同一组阈值（conf 0.001、NMS 0.7）、同一 `maxDets [1,10,100]`、FP32、CUDA 分配上限 0.6、线程 4、seed 17。因此分数可以直接和 `s_original 11.77`、`s_tuned5 19.10`、`s_tuned30 19.99`、`l_original 16.12` 比较。

## 7. 不进入公开仓库的内容

影像、标注、逐条预测 JSON、可视化图、权重、日志、`deps_ds/` 都留在本地工作区。仓库只放脚本、聚合结果 JSON、统计图和说明文档。
