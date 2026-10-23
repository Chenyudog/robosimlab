# π0 的 LIBERO-Goal 表示图

目标：先生成预训练 π0 的动作表示 t-SNE 图，再用同样样本比较普通微调 checkpoint。
当前阶段不训练 semantic anchoring，也没有普通微调后的 π0 权重。

## 已完成的本地结果（2026-10-22）

- 全部 10 个 Goal 任务，每类 10 条轨迹，每条 8 帧，共 100 条轨迹 / 800 帧。
- 原始 1024D 特征的 cosine Silhouette：`0.1644805371761322`。
- 实际 CUDA 推理通过，峰值分配显存约 6.58 GB；所有轨迹特征通过形状、有限值、均值一致性检查。
- 输出目录：`outputs/pretrained_goal/`。
- `pretrained_goal_tsne_compact.png`：截图风格方图；`pretrained_goal_tsne.png`：完整任务图例版。
- 特征和坐标：`features_and_tsne.npz`；样本清单：`manifest.json`；指标：`metrics.json`。
- 软件版本、源码哈希和权重头指纹：工作目录下的 `environment.json`。

## 输入与实现

- 本地权重：`/home/cyd/robosimlab/model/pi0/model.safetensors`，LeRobot π0 基础模型。
- 数据：`/home/cyd/datasets/libero_pi`，`physical-intelligence/libero`，revision `a4336d589d589045d1c56423ffdf3b88a0e19b1f`。
- π0 源码：`/home/cyd/lerobot/src`，在运行时显式设置导入路径，不修改已有模型源码。
- Python：`/home/cyd/miniforge3/envs/lerobot/bin/python`。
- 分词器：`https://storage.googleapis.com/big_vision/paligemma_tokenizer.model`。
- 状态归一化：`https://storage.googleapis.com/openpi-assets/checkpoints/pi0_libero/assets/physical-intelligence/libero/norm_stats.json`，使用 q01/q99。
- 套件任务名来源：`https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/benchmark/libero_suite_task_map.py`。

严格按任务名匹配 Goal；这份数据的对应 ID 为 10–19。作者快照的绘图脚本写死 ID 20–29
并排除 28，与本地数据的套件映射不一致，不直接沿用。保留完整十类。

## 提取协议

1. 种子 0，每类随机选 10 条轨迹；每条均匀采 8 帧，包含首尾帧。
2. 每帧读取主视角、腕部图像和 8D 状态；图像 resize 到 224，映射到 [-1,1]。
3. 状态按官方 LIBERO q01/q99 归一化后补零到 32D；同一指令经 PaliGemma tokenizer 编码。
4. 图像和语言作为 prefix；标准高斯动作噪声起步，执行 10 步 Euler 去噪。
5. hook 动作 expert 的 `layers[10]`（零起始编号，即第 11 个 block）输出。
6. 只取最后 50 个动作 token，对 token 求平均；取最后一次去噪调用（t=0.1、更新前）的特征。
7. 再对每条轨迹的采样帧求平均，得到一个 1024D 向量。
8. t-SNE：cosine、PCA 初始化、seed=0、2000 iter；perplexity 与官方脚本公式一致。
9. Silhouette 用原始 1024D 特征与 cosine 距离计算，不在二维坐标上计算。

每个输出目录保存 `manifest.json`、逐轨迹 NPZ、合并特征、PNG、`metrics.json`。
逐轨迹文件原子写入，重跑时跳过已完成文件。配置不一致会拒绝复用目录。
权重逐参数校验名字与形状，缺失参数即报错。未使用的 vocabulary 输出 heads 不放入 GPU；
PaliGemma 的 tied input embedding 从 checkpoint 的 lm_head 权重恢复。
两个有效相机输入直接参与 prefix，省略被完全 mask 的第三相机占位。

## 运行

```bash
bash /home/cyd/pi0_representation/run_pretrained.sh
```

扩大到所有 Goal 轨迹、每条 8 帧，使用独立输出目录：

```bash
cd /home/cyd/pi0_representation
PYTHONPATH=/home/cyd/lerobot/src /home/cyd/miniforge3/envs/lerobot/bin/python extract_pretrained.py \
  --episodes-per-task 0 --frames-per-episode 8 --output outputs/pretrained_goal_all
/home/cyd/miniforge3/envs/lerobot/bin/python plot_features.py outputs/pretrained_goal_all
```

`--frames-per-episode 0` 使用全部帧，会显著增加运行时间。

## 解释边界

这是一份使用本地 LeRobot π0、明确采样协议的复现实验，不保证重现论文的 0.27 数值或相同点形。
第一张图采用同一批离线示范图像上的模型推理特征，不是闭环环境 rollout 成功率实验。
不对特定类别或模型的二维坐标做额外变换。所有任务都保留。
要得到 Vanilla FT 图，仍需同一基础模型在 LIBERO 上普通微调后的 checkpoint。
