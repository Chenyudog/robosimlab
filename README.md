# 一、 corl_ws
## 文件内容解释
本文件夹用于发表2027corl论文
### /data 用于存放LIBERO下载的东西 
/data/chunk-000 
主要存放.parquet文件 其主要内容是vla的输入文件 包括 image action state等等
/data/meta 
episodes 为存放1693条示范轨迹的任务列表
info 用于存放任务参数
tasks 存放任务字典
### docs agent工作文档
### models  用于存放模型
一个是fine-tune模型
一个是还未fine-tuned的模型
### reference
中间产物 可忽略
### runtime

### scripts

### third_party
第三方代码

## 任务编码

以下为本地 LIBERO 四个相关套件的默认顺序，编号直接对应 --task-id，不是数据集 tasks.jsonl 的 task_index。每个任务内再用 --episode-index 选择初始状态。

### libero_spatial（空间关系）

- 0：拿起盘子和小烤碗之间的黑碗，放到盘子上。
- 1：拿起小烤碗旁边的黑碗，放到盘子上。
- 2：拿起桌子中央的黑碗，放到盘子上。
- 3：拿起饼干盒上的黑碗，放到盘子上。
- 4：拿起木柜最上层抽屉里的黑碗，放到盘子上。
- 5：拿起小烤碗上的黑碗，放到盘子上。
- 6：拿起饼干盒旁边的黑碗，放到盘子上。
- 7：拿起炉灶上的黑碗，放到盘子上。
- 8：拿起盘子旁边的黑碗，放到盘子上。
- 9：拿起木柜上的黑碗，放到盘子上。

### libero_object（目标物体）

- 0：拿起字母汤罐头，放进篮子。
- 1：拿起奶油奶酪，放进篮子。
- 2：拿起沙拉酱，放进篮子。
- 3：拿起烧烤酱，放进篮子。
- 4：拿起番茄沙司（ketchup），放进篮子。
- 5：拿起番茄酱（tomato sauce），放进篮子。
- 6：拿起黄油，放进篮子。
- 7：拿起牛奶，放进篮子。
- 8：拿起巧克力布丁，放进篮子。
- 9：拿起橙汁，放进篮子。

### libero_goal（操作目标）

- 0：打开柜子中间的抽屉。
- 1：将碗放到炉灶上。
- 2：将葡萄酒瓶放到柜子顶部。
- 3：打开最上面的抽屉，将碗放进去。
- 4：将碗放到柜子顶部。
- 5：将盘子推到炉灶前方。
- 6：将奶油奶酪放进碗里。
- 7：打开炉灶。
- 8：将碗放到盘子上。
- 9：将葡萄酒瓶放到架子上。

### libero_10（多步骤任务）

- 0：将字母汤罐头和番茄酱都放进篮子。（LIVING_ROOM_SCENE2）
- 1：将奶油奶酪盒和黄油都放进篮子。（LIVING_ROOM_SCENE2）
- 2：打开炉灶，将摩卡壶放到炉灶上。（KITCHEN_SCENE3）
- 3：将黑碗放进柜子最下面的抽屉，并关上抽屉。（KITCHEN_SCENE4）
- 4：将白色杯子放在左边盘子上，将黄白色杯子放在右边盘子上。（LIVING_ROOM_SCENE5）
- 5：拿起书，放进收纳架后面的隔间。（STUDY_SCENE1）
- 6：将白色杯子放在盘子上，将巧克力布丁放在盘子右侧。（LIVING_ROOM_SCENE6）
- 7：将字母汤罐头和奶油奶酪盒都放进篮子。（LIVING_ROOM_SCENE1）
- 8：将两个摩卡壶都放到炉灶上。（KITCHEN_SCENE8）
- 9：将黄白色杯子放进微波炉，并关上炉门。（KITCHEN_SCENE6）

## 运行方式

```bash
cd /home/cyd/robosimlab/corl_ws

export VLA_LEGACY_PI0=1
export VLA_LEROBOT_SRC="$PWD/third_party/lerobot_v044/src"
export PYTHONPATH="$VLA_LEROBOT_SRC"
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$PWD/runtime/libero_config"
export HF_HUB_OFFLINE=1

runtime/legacy_pi0/bin/python -u scripts/run_baseline.py \
  --model-dir models/pi0_libero_finetuned \
  --suite libero_spatial \
  --task-id 0 \
  --episode-index 0 \
  --seed 7 \
  --max-steps 220 \
  --replan-steps 50 \
  --snapshot-step 10 \
  --vision-fp32 \
  --output "outputs/reproduce_$(date +%Y%m%d_%H%M%S)"
```

### 参数含义

suite：选择任务套件，上面列出本数据集涉及的4个套件。
task-id：选择套件内的任务，可能同时改变场景布局、物体和目标。
episode-index：选择当前任务的预设初始状态，可能改变物体位置和机械臂姿态；已检查的 Spatial task0 状态0–5之间，物体位置差可达约3厘米。
seed：固定随机种子；模型、环境版本和其余参数也应保持一致。
max-steps：最多执行220步，成功后提前结束。
replan-steps：每预测一个动作块执行50步，再根据新画面预测。
snapshot-step：在第10步保存中途快照。
vision-fp32：视觉部分使用FP32；当前实现同时将大型词嵌入放在CPU，删除此选项不一定省显存。


# 二、其他文件
    本文件夹用于vla模型的仿真
