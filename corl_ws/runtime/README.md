# Semantic Anchoring for Robotic Action Representations 复现

目标论文：Xu et al., [arXiv:2607.13597](https://arxiv.org/abs/2607.13597)。本目录实现论文的训练期语义锚定模块，并保留此前的本机 MuJoCo 自检。SO-101 抓取场景**不是**论文的 LIBERO / SimplerEnv 基准。

## LIBERO 环境（已配置并验证）

```bash
source /home/cyd/ca_repro/activate_libero.sh
python /home/cyd/ca_repro/check_libero.py --lerobot
```

上述命令已验证 `libero_spatial` 的真实任务可以重置、执行一步动作，并返回主视角和腕部相机两路 RGB 图像。环境位于 `.venv/`；它继承现有 LeRobot Python 3.12/CUDA 环境的基础包，仅把 LIBERO 新依赖安装到项目虚拟环境。已安装 LeRobot 0.6.2、hf-libero 0.1.4、robosuite 1.4.0、MuJoCo 3.8.1、transformers 5.5.4；LIBERO 资产位于 `/home/cyd/.cache/libero/assets`，配置在 `libero_config/config.yaml`。`python -m pip check` 已通过。

同一终端可用 `python /home/cyd/ca_repro/check_anchor.py --device cuda --paper-shape` 检查语义锚定模块的 CUDA 前后向传播。GPU 在常规终端可见；当前受限沙箱内可能不可见。

环境的依赖来源是本机 `/home/cyd/lerobot` 的 `pi,libero` extras。重新创建时需要先建立基于 `/home/cyd/miniforge3/envs/lerobot/bin/python` 的 `--system-site-packages` venv，再在其中安装 `-e '/home/cyd/lerobot[pi,libero]'` 和 `typeguard`。首次运行还需从 [LeRobot 的 LIBERO 资产仓库](https://huggingface.co/datasets/lerobot/libero-assets)下载约 422 MB 场景资产。

## 已实现

`semantic_anchor.py` 实现论文式 (4)–(8)：共享/私有双通道、重建损失、平方余弦去相关、可学习查询的八头注意力池化、双向 InfoNCE，以及与主模型动作损失的加权组合。输入为第 10 层动作 token 特征 `[B,T,d]` 与冻结的 EgoHOD-Large 文本嵌入 `[B,768]`。默认配置按 π0 设置：`d=1024`、`ds=dp=512`、`λalign=0.1`、`λr=0.01`、`λd=0.075`；SpatialVLA 需要 `d=2304`、`λalign≈0.5`。

论文未给出双层 MLP 隐层宽度/激活函数及温度初值；当前实现分别采用输出维度/GELU/0.07。附录写明八头注意力，正文式 (3) 则写成单查询形式；当前实现遵照附录。

```bash
/home/cyd/miniforge3/envs/mujoco/bin/python /home/cyd/ca_repro/check_anchor.py
```

在能够访问显卡设备的终端，还可运行 `.../check_anchor.py --device cuda --paper-shape`；已在本机 RTX 4060 上通过损失、优化步骤及 π0 特征维度检查。这只测试辅助模块，不包含完整 π0 的显存占用。

`with_action_loss()` 返回可反向传播的总损失；训练后丢弃 `SemanticAnchor`，部署策略无须依赖它。本机 LeRobot 的 π0 已增加可选 `action_feature_layer=10` 输出；默认调用方式保持原有返回值。`lerobot_pi0_step.py` 将此输出与动作损失、EgoHOD 文本嵌入接成一个优化步骤。真实训练仍需准备带指令的 LIBERO 批次与 EgoHOD 权重/嵌入。

训练时可分别创建 `AdamW(policy.parameters(), lr=2.5e-5)` 与 `AdamW(anchor.parameters(), lr=1e-4, weight_decay=0.01)`，再把真实批次和对应指令嵌入传给 `lerobot_pi0_step.train_step()`。嵌入应与模型在同一设备。论文还要求 BF16、batch 32、30k 步；这里只提供单步接口，未伪造数据或模型权重。

## 另有 SO-101 物理仿真自检

```bash
MUJOCO_GL=egl /home/cyd/miniforge3/envs/mujoco/bin/python /home/cyd/ca_repro/smoke_sim.py --check-camera
```

它检查已有 SO-101 场景的动力学和手腕相机。该场景可用于代码调试，但论文表 1 的结果必须在 LIBERO / SimplerEnv 上评估。

## 论文结果的资源要求

- π0 + LIBERO：50 条示范/任务、全参数训练 30k 步、batch 32、2 GPU、EgoHOD-Large 文本嵌入；在四套 LIBERO 任务上评测。
- π0 + BridgeData V2 / SimplerEnv：50k 步、batch 128、2 GPU；SimplerEnv Visual Matching 协议。
- SpatialVLA + BridgeData V2 / SimplerEnv：4B 模型、LoRA rank 32、约 22k 步、batch 192、2 GPU。
- 诊断性 probing 使用另一冻结文本编码器 Qwen3-VL-Embedding；与训练期 EgoHOD 目标不同。

主机有一张 NVIDIA GeForce RTX 4060 Laptop GPU（8 GB），新环境中的 PyTorch 已验证可执行 CUDA 张量计算。论文表 3 使用 2 张 GPU 全参数微调 π0；单卡 8 GB 只能考虑降低 batch、梯度累积、梯度检查点或只训练部分参数。这属于资源受限的近似实验，不能当作论文表 1 的严格复现。LIBERO 仿真环境和资产已经可用，但训练示范数据、π0/EgoHOD 权重仍未准备；作者的 [SemanticMN 仓库](https://github.com/xy02-05/SemanticMN) 当前仅有 README 和展示素材，因此本目录尚不声称复现论文成功率。π0 特征接口已通过语法检查，但尚未用真实 π0 权重完成前向验证。
