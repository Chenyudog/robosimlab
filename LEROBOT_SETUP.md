## 本地 LeRobot 环境

框架源码位于 `lerobot/`，独立 Git 仓库，版本 `0.6.2`（提交 `64b23178`）。
原来的 `/home/cyd/lerobot` 保留；Conda 的 `lerobot` 环境使用本工作区的可编辑安装。
`lerobot/` 在外层仓库中忽略，框架更新在它自己的仓库内管理。

```bash
conda activate lerobot
cd /home/cyd/mujoco_ws
python task/check_environment.py
```

重新绑定源码和安装 π0、MuJoCo 依赖：

```bash
python -m pip install --no-build-isolation -e './lerobot[pi]' mujoco
```

检查脚本验证框架导入路径、π0 配置读取及 Franka 物理步进，不加载完整权重。
下一步是添加 Franka 抓取场景和观测/动作适配层，再验证模型推理。


## 验证结果

- LeRobot 从本工作区的 `lerobot/src/lerobot` 加载。
- π0 类导入与本地配置读取成功；完整权重尚未加载。
- MuJoCo 3.15.0 已安装，Franka 的 8 个执行器和 1 个相机加载成功，100 步物理仿真通过。
- 当前 CUDA 不可用，模型推理前需要解决 GPU 可用性。
- `pip check` 发现环境中已有的 `generate-parameter-library-py` 缺少 `typeguard`；上述验证通过。
