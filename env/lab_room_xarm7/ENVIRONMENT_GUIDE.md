# xArm7 仿真环境维护与修改指南

更新日期：2026-10-06

本文用于继续维护 `/home/cyd/mujoco_ws/env/lab_room_xarm7`。它说明当前场景的文件关系、坐标约定、修改方法和验证顺序。实际 XML 和脚本优先于本文中的历史说明。

## 1. 当前场景和文件职责

当前场景使用 xArm7、平行夹爪、ALOHA 桌、两个固定篮子、七个物体和五个相机。场景目录是：

```text
/home/cyd/mujoco_ws/env/lab_room_xarm7
```

| 内容 | 文件 |
| --- | --- |
| 房间、桌子、篮子、物体、灯光、固定相机 | `scene.xml` |
| 机械臂、夹爪、关节、执行器、TCP、腕部相机 | `xarm7.xml` |
| 统一加载、home reset、物理落稳 | `scene_state.py` |
| 打开交互式 MuJoCo 窗口 | `view_scene.py` |
| 渲染五个相机预览 | `capture_scene.py` |
| 检查竖直向下 IK 可达性 | `check_reach.py` |
| IK 专家抓取和放置 | `expert.py` |
| 当前实际加载的机械臂、桌子和物体网格 | `assets/xarm7_meshes/` |
| 房间和物体贴图 | `assets/textures/` |
| ALOHA 桌贴图 | `assets/aloha_table/small_meta_table_diffuse.png` |
| 当前预览图 | `fixed_camera.png`、`wrist_camera.png`、`table_view.png`、`room_overview.png`、`door_view.png` |

`robot/ufactory_xarm7/` 是原始机器人模型包，不是当前实验室场景直接读取的 XML。`/home/cyd/ca_repro` 是独立的论文复现工程，使用另一套 Python 环境；修改这里的场景不需要修改它们。

## 2. 使用正确的 Python 环境

当前场景使用 Conda 环境 `mujoco`：

```bash
/home/cyd/miniforge3/envs/mujoco/bin/python
```

已核实的版本是 Python 3.10.20、MuJoCo 3.14.0、NumPy 1.26.0。可以先检查：

```bash
/home/cyd/miniforge3/envs/mujoco/bin/python -c \
'import sys, mujoco, numpy; print(sys.version.split()[0], mujoco.__version__, numpy.__version__)'
```

有桌面显示时，打开交互窗口：

```bash
/home/cyd/miniforge3/envs/mujoco/bin/python \
  /home/cyd/mujoco_ws/env/lab_room_xarm7/view_scene.py
```

也可以先执行 `conda activate mujoco`，再进入场景目录运行 `python view_scene.py`。离屏渲染使用 `MUJOCO_GL=egl`：

```bash
MUJOCO_GL=egl /home/cyd/miniforge3/envs/mujoco/bin/python \
  /home/cyd/mujoco_ws/env/lab_room_xarm7/capture_scene.py
```

`view_scene.py` 会读取完整的 `home`，先推进默认 1 秒物理，再打开窗口。直接打开 XML 不会执行这个统一的落稳流程。

## 3. 修改前的安全操作

当前仓库有未提交的场景、脚本、资产、截图和旧目录删除记录。不要用下面这些命令清理工作区：

```bash
git reset --hard
git checkout -- .
git clean -fd
```

修改前可以把当前场景备份到仓库外：

```bash
cp -a /home/cyd/mujoco_ws/env/lab_room_xarm7 \
  "/home/cyd/lab_room_xarm7_backup_$(date +%Y%m%d_%H%M%S)"
git -C /home/cyd/mujoco_ws status --short
```

建议一次只改一类内容：先改布局，再改相机，再改控制或抓取逻辑。XML 修改后必须重新创建 `MjModel`；已经打开的 MuJoCo 窗口不会自动热更新。

## 4. 坐标、单位和当前关键尺寸

- 长度单位是米，关节角单位是弧度。
- MuJoCo 四元数顺序是 `w x y z`。
- 世界 Z 轴向上；机械臂面向桌子时，世界 +X 指向桌子，世界 +Y 是机器人左侧，世界 -Y 是机器人右侧。
- `body`、`geom`、`site` 和相机的 `pos` 通常是相对于父 body 的局部坐标。
- `box size` 是半尺寸。例如 `size="0.1 0.2 0.01"` 的完整尺寸是 `0.2 × 0.4 × 0.02` 米。

当前关键位置：

- 机械臂基座：`(0.25, 0, 0.62)`。
- 桌子 body：`(0.82, 0, 0.62)`。
- 桌面顶面约为 `z=0.620787`。
- 桌子已旋转 90°：桌子局部 +X 大致对应世界 +Y，局部 +Y 大致对应世界 -X。
- 左篮子世界位置约为 `(0.72, +0.45, 0.631587)`。
- 右篮子世界位置约为 `(0.72, -0.45, 0.631587)`。

不要把 body 原点、geom 中心和桌面顶面混为同一个高度。调整物体时，应根据实际 collision geom 的最低点检查支撑关系。

## 5. 修改物体位置或朝向

物体在 `scene.xml` 的 `worldbody` 下，是带 `freejoint` 的动态 body。当前物体名称和初始位置如下：

| 物体 body | 初始位置 `(x, y, z)` |
| --- | --- |
| `cola_can` | `(0.56, -0.22, 0.661084003)` |
| `glass_bottle_1` | `(0.56, 0, 0.685644374)` |
| `cereal_box` | `(0.72, -0.20, 0.695749001)` |
| `milk_carton` | `(0.56, 0.22, 0.681803403)` |
| `bread` | `(0.70, 0.02, 0.644038370)` |
| `fruit_lemon_1` | `(0.72, 0.22, 0.640715999)` |
| `robosuite_plate` | `(0.95, 0, 0.630787000)` |

修改初始摆放时必须同时修改：

1. 对应 body 的 `pos` 和 `quat`；
2. `scene.xml` 最后 `home` keyframe 中对应的 freejoint 数值。

例如把可乐罐的 X 坐标改为 0.60：

```xml
<body name="cola_can" pos="0.60 -0.22 0.661084003">
```

同时把 `home` 中可乐罐对应的一行改为：

```text
0.60 -0.22 0.661084003 1 0 0 0
```

只修改 body 而不修改 home，运行 `scene_state.py` 或其他入口时会被旧 keyframe 重置。旋转物体后，还要重新检查网格最低点，因为同一高度不一定仍能贴合桌面。

当前 `home.qpos` 的顺序是：13 个机器人和夹爪坐标，然后依照 `worldbody` 关节顺序排列的 7 个物体 freejoint，每个 freejoint 是 `xyz + wxyz` 共 7 个数。当前模型统计为 `nq=62`、`nv=55`、`nu=8`。新增一个 freejoint 会增加 qpos 和速度自由度，但不会自动增加控制量。

带 `freejoint` 的物体必须是 `worldbody` 的直接子 body。不要把它放进桌子 body 下，否则会触发 MuJoCo 的 `free joint can only be used on top level` 错误。固定篮子没有 freejoint，可以作为桌子的子 body。

## 6. 增加、删除或替换物体

建议按以下顺序操作：

1. 从已有模型库选择网格，记录来源和许可证。
2. 把实际使用的网格复制到 `assets/xarm7_meshes/objects/meshes/`。
3. 在 `scene.xml` 的 `<asset>` 中声明 mesh、texture 和 material。
4. 在 `worldbody` 下添加名称唯一的 body、freejoint 和 geom。
5. 在 `home.qpos` 中按正确关节顺序添加或删除对应的 `xyz + wxyz`。
6. 在 `check_reach.py` 的 `OBJECTS` 或 `PLACE_TARGETS` 中同步名称。
7. 如果要使用专家抓取，还要更新 `expert.py` 的 `OBJECTS`、抓取方向和抓取高度逻辑。
8. 重新检查视觉网格、碰撞网格、质量、摩擦和桌面接触。

`xarm7.xml` 的全局 compiler 是：

```xml
<compiler angle="radian" autolimits="true" meshdir="assets/xarm7_meshes"/>
```

因此 `scene.xml` 中：

```xml
<mesh file="objects/meshes/new_object.stl"/>
```

实际会从 `assets/xarm7_meshes/objects/meshes/new_object.stl` 加载。网格不要只放在外层 `assets/objects/meshes/`，否则当前 meshdir 找不到它。

可视 geom 和碰撞 geom 可以分开。玻璃瓶当前就是可视 geom 禁止碰撞、质量为 0，另有透明 collision geom 参与物理。调整模型时要同时检查这两部分，避免留下看不见的旧碰撞体。

当前带孔方板 `robosuite_plate` 是约 0.34 m 的方形目标，不是普通餐盘；它适合做放置目标。历史专家测试中可乐罐、麦片盒、牛奶盒和面包可用竖直抓取，玻璃瓶和柠檬不适合当前的竖直夹取。修改后必须重新测试，不能直接沿用这个历史结论。

## 7. 修改桌子、篮子或机器人基座

桌子同时包含可视 mesh 和独立的 `table_collision` box。改变桌子位置、尺寸或旋转后，要同时核对：

- 可视网格边界；
- `table_collision` 的位置和半尺寸；
- `table_top` site；
- 桌面顶面高度；
- 物品和篮子的支撑高度。

不要只把渲染图调好而保留旧碰撞平面。

两个篮子是桌子的固定子 body，名字是 `left_bin` 和 `right_bin`。移动桌子会带动它们，但不会带动 `worldbody` 下的自由物体。固定篮子放高以后不会自动掉落，需要直接修改它的局部位置。

机器人基座在 `xarm7.xml` 的 `link_base`，支撑柱在 `scene.xml` 的 `robot_pedestal`。移动或升降机械臂时，要同步支撑柱、home TCP、腕部相机、可达性和轨迹碰撞。改变桌面高度后，物品高度、篮底高度、专家抓取高度和运输高度都必须重新核查。

## 8. 修改相机

固定相机 `fixed_camera` 在 `scene.xml` 的 `worldbody` 下，当前为：

```xml
<camera name="fixed_camera" mode="fixed"
        pos="1.39 0 2.10"
        xyaxes="0 1 0 -0.81799036 0 0.57523193"
        fovy="58"/>
```

它位于桌子另一侧，水平位置满足当前布局中的：

```text
camera_xy = 2 * table_center_xy - robot_base_xy
```

腕部相机 `wrist_camera` 在 `xarm7.xml` 的 `xarm_gripper_base_link` 下，会随机械臂末端运动。当前为：

```xml
<camera name="wrist_camera"
        pos="-0.12 0 0.12"
        xyaxes="0 1 0 0.629812 0 -0.776748"
        fovy="85"/>
```

`xyaxes` 的前 3 个数是相机局部 X 轴，后 3 个数是局部 Y 轴；相机视线沿局部 -Z。不要把目标点直接填进 `xyaxes`。修改后至少检查 home、抓取前、闭合、抬升和篮子上方几个姿态。腕部相机需要同时看到两根手指和操作区域；固定相机需要看到桌面物体、篮子和机械臂。

重新渲染预览：

```bash
MUJOCO_GL=egl /home/cyd/miniforge3/envs/mujoco/bin/python \
  /home/cyd/mujoco_ws/env/lab_room_xarm7/capture_scene.py
```

它会覆盖本目录五张同名的 1280×720 PNG，并检查五个相机存在以及腕部深度数组有效。这个检查不判断遮挡，也不判断物体是否与桌面接触；生成图片后要实际查看。

## 9. 修改机械臂 home 和控制

当前机械臂 home 关节角为：

```text
[0, -0.7714, 0, 0.8223, 0, 1.5938, -3.1416]
```

TCP site 名称是 `link_tcp`。修改机械臂 home 时要同步 `scene.xml` 中：

- home qpos 的前 7 个关节值；
- home qpos 中夹爪机构的值；
- home ctrl 中前 7 个位置目标；
- home ctrl 的第 8 个夹爪命令。

夹爪控制范围是 `[0, 255]`：0 表示张开，255 表示闭合。前 7 个执行器是关节位置目标形式，不要把 `data.ctrl` 当作关节力矩，也不要把夹爪数值当作弧度。

物理步长是 `0.002 s`。专家脚本的控制频率是 20 Hz，每个控制周期约推进 25 个物理步。修改物理步长或控制频率后，需要重新检查轨迹插值、跟踪误差和接触稳定性。

## 10. 修改后必须执行的验证顺序

### 10.1 先加载和观察 home

```bash
/home/cyd/miniforge3/envs/mujoco/bin/python \
  /home/cyd/mujoco_ws/env/lab_room_xarm7/view_scene.py
```

检查物品是否落在桌面上，是否悬空、穿透、滚出桌面或互相穿插。

### 10.2 检查竖直 IK 可达性

```bash
/home/cyd/miniforge3/envs/mujoco/bin/python \
  /home/cyd/mujoco_ws/env/lab_room_xarm7/check_reach.py
```

当前脚本检查物体、篮子和带孔板的多个目标，位置误差阈值为 3 mm、倾斜阈值为 3°；失败时退出码为 1。它只检查运动学，不检查从 home 到目标的整段路径是否撞桌子或篮子，也不证明夹爪能夹住物体。新增物体后要同步更新 `OBJECTS`。

### 10.3 重新生成并查看相机图

```bash
MUJOCO_GL=egl /home/cyd/miniforge3/envs/mujoco/bin/python \
  /home/cyd/mujoco_ws/env/lab_room_xarm7/capture_scene.py
```

不要把旧 PNG 当作新 XML 的结果。需要保留旧图时，先把它们复制到工作区外。

### 10.4 用专家脚本测试实际抓放

专家脚本支持三个选项：`--object`、`--bin` 和 `--snapshots DIR`。例如：

```bash
MUJOCO_GL=egl /home/cyd/miniforge3/envs/mujoco/bin/python \
  /home/cyd/mujoco_ws/env/lab_room_xarm7/expert.py \
  --object cola_can --bin right_bin \
  --snapshots /home/cyd/mujoco_ws/env/lab_room_xarm7/snapshots
```

`--object` 当前可选 `cola_can`、`glass_bottle_1`、`cereal_box`、`milk_carton`、`bread`、`fruit_lemon_1`；`--bin` 可选 `left_bin` 和 `right_bin`。`success=True` 才表示这一次抓放通过；失败时脚本返回退出码 1。`contacts` 会记录接触，但不是所有接触都会自动否决 success，因此要查看终端输出和快照。

改变布局、物体朝向或夹爪参数后，至少测试两个篮子和所有计划作为目标的物体。带随机化的实验要使用多个随机种子重新测试；已有历史成功记录不能替代新布局验证。

### 10.5 检查修改内容

```bash
git -C /home/cyd/mujoco_ws diff -- env/lab_room_xarm7
git -C /home/cyd/mujoco_ws status --short
```

把“本次刚运行得到的结果”和 README/交接文档中的历史结果分开记录。当前目录没有完整持久化的物理回归测试套件。

## 11. 常见问题

| 现象 | 优先检查 |
| --- | --- |
| 物体改完位置又回到旧位置 | body 和 `home` 是否同时更新 |
| keyframe 长度错误或物体出现在原点 | freejoint 是否加入 home，顺序是否正确 |
| 网格找不到 | 是否放入 `assets/xarm7_meshes/` 对应的 meshdir 路径 |
| 看起来在桌上但会下落或悬空 | collision geom 的实际最低点和桌面碰撞高度 |
| IK 通过但抓放失败 | 抓取高度、朝向、指垫开口、掌部碰撞、摩擦和整段轨迹 |
| 移动篮子后左右方向反了 | 桌子局部坐标经过 90° 旋转，检查局部坐标到世界坐标的转换 |
| 相机图片没有变化 | 是否重新编译模型并重新运行 `capture_scene.py` |
| EGL 无法渲染 | 检查驱动；有桌面时可尝试 `MUJOCO_GL=glfw`，不要为此修改论文复现环境 |

## 12. 本次清理记录

本次只移除了确认没有当前代码引用、且与实际加载副本逐字节一致的 8 个网格：

- `assets/objects/meshes/` 下的 `bottle.stl`、`bread.stl`、`can.msh`、`cereal.msh`、`lemon.msh`、`milk.msh`；空目录一并移除。
- `assets/aloha_table/tablelegs.obj` 和 `assets/aloha_table/tabletop.obj`。

它们没有永久删除，备份位于：

```text
/home/cyd/mujoco_cleanup_backup_20261006_VGsuOi/assets/
```

实际加载的 `assets/xarm7_meshes/` 网格、桌子贴图、所有场景脚本、预览图以及没有确认重复的贴图均保留。旧的 `env/pick_and_place_xarm7` 删除状态也保留，没有恢复。

仓库根目录新增 `.gitignore`，忽略 Python 缓存、pytest 缓存和本场景的 `snapshots/`。它没有忽略网格格式文件。

当前场景仍只有场景、IK 专家和检查基础；本文不代表已经完成 Gymnasium 封装、示范采集、训练策略或论文评测。

需要恢复本次移出的 8 个网格时，可以执行下面的命令；`-n` 表示不覆盖已存在的同名文件：

```bash
cp -an /home/cyd/mujoco_cleanup_backup_20261006_VGsuOi/assets/. \
  /home/cyd/mujoco_ws/env/lab_room_xarm7/assets/
```

## 13. 本次清理后的实际验证

以下是 2026-10-06 本次新运行的结果，不是之前交接文档中的历史记录：

- `load_scene()` 成功加载并落稳，`nq=62`、`nv=55`、`nu=8`、5 个相机、1 个 keyframe，qpos/qvel 有限，MuJoCo warning 数量为 0。
- `check_reach.py` 的 33 个目标全部通过，输出 `0 unreachable`。
- `expert.py --object cola_can --bin right_bin` 成功，输出 `success=True`、`contacts=[]`。
- 五路 RGB 均成功离屏渲染为 1280×720，腕部深度形状正确且全部有限，范围约 0.079–1.849 米。
- `git diff --check` 通过。

本次渲染只在内存中检查，没有覆盖原来的五张预览图，也没有重新进行图像目视验收。没有启动交互窗口、重跑所有物体/篮子组合或历史随机化测试。场景 XML 和控制脚本没有因本次清理而改变。
