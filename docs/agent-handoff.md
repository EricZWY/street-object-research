# 实验思路与 agent 接手说明
更新：2026-11-10。先阅读本文件，再阅读各实验报告和脚本；以磁盘真实产物为准。

## 研究目标
主题：基于多源影像的街道关键小地物位置确定与属性提取。
A：设施类别识别、Logo品牌识别；B：跨视角场景检索及目标位置确定。属性与位置通过实例ID关联。
不扩展为变化检测。设施采用明确定义的地面代表点；Logo对应入口或立面落地点，建筑框中心不能替代。CVOGL主要是较大地物的框标注，只承担B的基础方法验证。

## 工作区与环境
本机工作区 E:/CodexWorkspace/StreetObject_DataPlan。
研究材料 E:/LLM_Memory/基于多源影像的街道关键小地物位置确定与属性提取研究。
公开仓库 https://github.com/EricZWY/street-object-research 。用户已授权保存实验代码与报告；不要上传数据集影像、权重、原始逐条数据或凭据。
Python D:/Anaconda/envs/pytorch/python.exe；Python3.11、torch2.5.1+cu121、torchvision0.20.1+cu121，RTX3060 Laptop 6GiB，内存16GB。
外部代码 external/CVOS-Code，固定版本289602a033d5353d06987a2d56a808e9c938cdad。
数据 data/CVOGL/CVOGL_SVI；权重 weights/CVOS/model_svi_model_best.pth.tar。
该权重SHA256：036095a7a65fe5451545a014673e6b1b64c528e40ac9a6ad730a76d69d439e5b。
Vistas在下载中；接手时重新确认下载状态，不假定可用。上海区域及后续一两个街区的自采尚未落实。

工作区可能位于接手agent的默认写入范围外：沿用环境的权限机制申请执行已授权的写入，不能假定当前目录就是工作区。不要删除现有产物。

## 已完成
1. 数据审查及初步基线：scripts/audit_cvogl.py及既有报告。
2. SVI公开权重严格加载成功，736项匹配。scripts/evaluate_cvos.py为正式推理脚本，跳过多余ImageNet下载但严格载入任务权重。
3. 全量val923、test973，以及test中心提示对照完成：
   val成功率46.59%，test47.28%，中心提示47.28%（IoU严格>0.5）。
   reports/cvos_full/report.md为完整报告，analysis.json为汇总，cases.json及case_0.jpg至case_5.jpg为案例。
   runs/cvos_svi_val、runs/cvos_svi_test、runs/cvos_svi_test_center保存逐条结果。
   test422例IoU=0；961/973条在中心替换后框坐标最大变化<1像素。
   30条test查询文件与train字节相同，其余943条成功率46.98%；这不等于空间独立测试。
4. 本轮提示诊断：scripts/diagnose_cvos_prompt.py，val前40条，actual/center/far/zero四组。
   far为离实际点击最远的图像角点，仅作为扰动，不保证该角点对应另一个已标注目标；zero为全零提示图，可能分布外。
   runs/prompt_diagnostic/<mode>保存逐条预测和features_*.npz。
   两个hook分别记录combine_clickptns_conv及cross_attention完整输出；CPU保存，单次推理batch1。
   scripts/analyze_prompt_diagnostic.py --workspace <工作区>生成reports/prompt_diagnostic汇总和成对差异。
   本轮最终结论见reports/prompt_diagnostic/report.md。先检查四组各40条及报告是否存在，再决定是否续跑；不要无理由重复GPU计算。

## 方法与资源约束
继续FP32、batch1、inference_mode、单任务串行GPU，分配器限额0.6。先小样本再全量。禁止把特征张量积存在GPU上。
正式全量推理峰值allocated约593MiB、reserved902MiB；整卡占用包含系统，不能与分配器统计混同。
本轮带hook、CPU特征保存和压缩，其耗时不能与正式推理直接比较。
点击格式原始(x,y)，转换距离图时按(y,x)，图像512×256；参考图1024×1024。作者eval_iou_acc严格阈值>0.5、>0.25。
每个模式的预测仍对原始目标真值评分，仅衡量扰动鲁棒性/敏感性，不是新点击目标的正确率。
前40条为确定性诊断样本，不是随机代表性估计；不使用测试集调整方案。

## 下一步实验计划
第一优先：根据提示诊断报告决定是否进一步验证同图多目标。输入提示进入计算与“模型能够按目标切换”是两回事。为同一场景中至少两个目标建立独立配对真值，明确期望切换后的位置，才能评价实例条件定位。不要仅凭一次消融声称创新成立。
第二优先：制定小设施与Logo位置代表点标注协议，记录遮挡、入口歧义、影像日期/分辨率/坐标系统和点位依据；准备少量上海配对样本。
第三优先：Vistas下载后检查版本、标签配置及完整性，完成设施类别选择与识别基线。Logo已有其他验证，不必重复投入无关分类工作。
后续：A输出区域和实例ID→B场景检索→实例代表点定位；分开报告检索Recall@K、条件定位、端到端定位以及属性—位置关联。真实地理精度用米制误差及阈值成功率，并用独立街区检验泛化。

## 接手执行规则
先核对本轮报告和summary，再读代码；核对用户最新指令。保留有价值的正负结果、样本选择规则、环境与权重指纹。报告必须解释对研究整体的贡献和缺口，避免将CVOGL刷分变成独立课题。若环境/数据缺失，明确具体所需项；其余工作继续完成。
