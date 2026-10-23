# 当前执行计划与证据边界

## 当前主线

使用官方 `lerobot/pi0_libero_finetuned`，固定 revision `45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f`。模型卡明确注明其在 LIBERO 上训练并评测。完整文件 SHA256 验证后才运行。

模型、处理器统计和来源放在 `models/pi0_libero_finetuned/`，下载直接写入一个可恢复文件，避免重复合并大文件造成额外磁盘开销。`model.download.part` 是预先分配的文件，其文件长度不能当作下载进度；应查看 `download_progress/*.done` 和下载日志。

## 自动执行链

1. 下载并校验官方 finetuned 模型。
2. 使用配套处理器统计、FP32 视觉路径，对 task0 的三个官方初始状态运行正常任务。
3. 若至少两条成功，从成功轨迹第20步的快照做配对实验。
4. 先验证中途快照的可重复恢复，再分别运行原状态、目标左移2厘米、目标右移2厘米。
5. 三个分支均逐步获取最新画面并重新推理，使用相同噪声安排。新增物体穿透的分支标为无效。

状态文件：`outputs/pilot_status.json`。

日志：`logs/finetuned_download.log`、`logs/finetuned_pilot.log`。

普通任务未达到门槛时，流程停在基线检查，不把它算作反馈机制发现。配对实验是单快照 pilot，不是论文级泛化证据，也没有开始 SAE 修复训练。

## 已完成的对照

先前的 `pi0_libero_base` 已完成下载和校验，但在目前本地实现的多个普通任务诊断中均未成功。FP32视觉与OpenPI额外delta动作变换的兼容性对照也没有成功。它保留作来源和对照，不再当作可执行基线。

这些失败不能直接归因于某个内部机制，也不能单凭模型名断言它应该达到某个成功率。首条完整视频见 `outputs/baseline_spatial0_seed7_run4/rollout.mp4`。

归一化检查发现：base checkpoint 内部统计对应原始动作，而官方旧 OpenPI pi0_libero 配置包含额外 delta 变换。这两套约定不能混用。当前 finetuned 路径使用它自己配套的 pre/postprocessor 统计，默认不施加 OpenPI 额外 delta 变换。

来源：[官方 finetuned 模型](https://huggingface.co/lerobot/pi0_libero_finetuned)、[OpenPI 数据变换配置](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)。

## 推理版本兼容性核对

现代推理组合中的 finetuned 三条基线未成功，尚不能推断模型机制。已建立独立 legacy 环境，固定 LeRobot v0.4.4 与其 pyproject 要求的 `fix/lerobot_openpi` Transformers 分支。两者 revision 记录在 `runtime/legacy_backend_sources.json`。仿真依赖复用已通过检查的 MuJoCo 3.8.1，而非 conda 全局的另一版本。

当前日志：`logs/legacy_aligned_pilot.log`。状态文件仍为 `outputs/pilot_status.json`。使用 `run_pilot_legacy.sh` 启动；重复实验时设置新的 `VLA_RUN_PREFIX`，避免覆盖已有结果。

## 已验证基线及下一轮控制检查（2026-10-22）

legacy h50 普通任务在 initial0/1/2 中分别成功77步、成功79步、失败220步。此为小样本诊断，不能视为正式成功率。

第一轮配对使用 h1，原状态与左移2厘米均失败，右移2厘米因引入物体穿插而无效。与 h50 普通基线的执行间隔不同，不能归因于位移或反馈机制。结果与描述图位于 `outputs/legacy_h50_paired_pilot/`。

下一轮预先固定为三分支均 h50、位移±5毫米，仍从同一快照开始且各分支重新预测动作块；这不是原基线待执行动作队列的精确续跑。脚本 `scripts/run_matched_horizon_pilot.py`，日志 `logs/matched_h50_5mm.log`，结果 `outputs/legacy_h50_paired_5mm/`。仍检查穿插，先看未位移分支是否完成任务，再解释位移影响。

### h50、5毫米初始状态0结果

`outputs/legacy_h50_paired_5mm/summary.json`：nominal 成功212步，minus_x/plus_x 均在220步内未完成，三者均通过初始穿插检查。描述性图 `response_comparison.png`。此现象不能证明内部反馈机制缺陷；还需重复、终止预算与动作执行间隔对照。已继续 initial1 同协议重复，日志 `logs/matched_h50_5mm_initial1.log`，输出 `outputs/legacy_h50_paired_5mm_initial1/`。

### initial1 重复及执行间隔对照

initial1 三分支：nominal 成功62步，minus_x 未完成220步，plus_x 成功75步。两个快照汇总仅为 nominal 2/2、minus_x 0/2、plus_x 1/2，样本太小，不代表总体成功率。下一步预先固定相同两个快照、相同±5毫米和随机种子规则，只将执行间隔从50改为10，检查现象是否由长动作块解释。脚本 `scripts/run_horizon_control.py`，日志 `logs/horizon_control_h10.log`。若 h10 普通分支失败，该快照不可用于证明位移特异性失败。

## 扩展批处理（2026-10-22）

已启动 `vla-feedback-expanded-h50.service`，固定 task0、initial0..19、seed7、h50、220步预算、snapshot10，复用原有 initial0..2；新增17条普通任务。完成后对所有普通任务成功的快照执行 h50、±5毫米配对，复用已完成的 initial0/1 配对。选择不依赖扰动结果。协议 `outputs/expanded_h50/protocol.json`，实时状态 `outputs/expanded_h50/status.json`，汇总同目录，日志 `logs/expanded_h50.log`。此为单任务诊断，不能报告为完整LIBERO成功率。

扩展20条普通基线完成，10条成功。配对 initial3 在3步重复回放中最大状态差1.119897108207979e-09，超过固定1e-9门槛；保留原门槛并标记该快照 invalid_replay，不计作策略失败。批处理已恢复，继续其他成功快照；原错误保存在 paired_initial3/validation_error.json。

## 激活探针试运行（2026-10-23）

恢复旧项目路径兼容链接 `vla_feedback -> corl_ws`，保留原配置和已有环境。启动服务 `corl-activation-probe-v1`：20条task0轨迹自然h50规划边界，提取0/5/11/17层视觉语言两相机分别池化激活，以及动作专家去噪0/9步动作token池化激活。标签为目标相对末端world xyz，仅供评估，不输入模型。按initial0..11训练、12..15验证、16..19测试；Ridge超参数与候选激活只按验证集选择，加入状态、常数和训练标签打乱对照。尚不证明因果机制。

输出 `outputs/probe_pilot_v1/`。采集后自动拟合探针，生成 `probe_results.json` 和 `PROBE_RESULTS.md`。第一状态执行expert第11层第一去噪步的恒等替换和全块置零烟雾检查，验证hook正确性，尚未做跨分支donor替换或闭环修复。日志使用 `journalctl --user -u corl-activation-probe-v1.service`。

本轮首次加载因GPU峰值显存不足失败，错误保留 `outputs/probe_pilot_v1/load_error_attempt1.json`。修复 `run_baseline.py` 加载器：FP32视觉模式下词嵌入直接加载到CPU，最后只迁移buffer，不再将全部参数重新迁移GPU；不改变权重与计算精度。重启后成功采集。第一条技术检查：expert11/denoise0恒等替换的动作最大差0；全块置零的动作最大差0.18208，说明hook生效，不证明特征语义。已排队服务 `corl-activation-patch-v1`，探针完成后在验证集选层，并对16..19初始观测做位移donor替换、恒等与通道打乱对照；仅分析动作敏感性，尚未闭环干预。

探针采集已完成20条轨迹70个规划时刻，train39/val14/test17。验证集选择expert17的denoise0，测试坐标RMSE0.0629米；state对照0.0343米。不能将此解读为独立目标信息或机制证据，存在任务阶段和机器人状态相关性。实际数值见 `outputs/probe_pilot_v1/PROBE_RESULTS.md`。替换实验已自动启动。

激活替换试运行已完成：16/18/19有效，17因穿插排除；3个恒等替换动作差均为0，donor首动作L2差约0.00142..0.00249，具体各维和完整动作块保存在 `outputs/activation_patch_v1/`。此为单次推理动作敏感性，非闭环修复。报告 `PATCH_RESULTS.md`。下一步需要固定机器人状态的成组目标位移采集，以控制探针的状态/阶段相关性。

## 固定状态位移探针v2（2026-10-23）

启动 `corl-fixed-state-probe-v2`。20个新initial20..39，在初始状态settle10步后固定机器人、同seed7噪声、同语言，只改变目标world x或y，分别±5/±10毫米。不执行物理步，采集相对nominal激活差和动作差；目标标签不进入策略。逐分支检查机器人state不变、目标精确位移、图像变化、与其他可动对象（含子body）的穿插。训练20..31、验证32..35、测试36..39按场景隔离。对照包括state_delta恒零、图像RGB均值变化和场景内标签打乱。标签是2维真实位移，而非原先受阶段影响的绝对相对位置。仍为单任务、少量场景的readout检查，不是因果机制或闭环成功验证。脚本 `scripts/run_fixed_state_probes.py`，输出 `outputs/fixed_state_probe_v2/`。

v2运行修正：首次推理OOM，启用本进程PyTorch expandable_segments，未改变精度。第二次被state相等断言拦截；CPU环境单独检查定位为env.step返回obs与再次sim.forward后EEF观测约6.5微米差。原位置也执行sim.forward+强制观测刷新后，位移分支EEF差严格为0。已统一刷新顺序，未放宽1e-12检查。两次错误均归档在v2输出目录，不计作模型结果。

v2完成：20场景151有效分支，9穿插排除，train91/val29/test31。验证选择prefix0，测试坐标RMSE4.388毫米，常数5.531毫米，图像均值2.416毫米。动作专家均值池化未稳定超过常数。不能声称语义机制或动作专家位置盲；池化可能丢失空间/时间信息。中文报告 `outputs/fixed_state_probe_v2/RESULTS_ZH.md`，图 `comparison.png`。下一步先改进空间/时间token读出并加入非目标物体位移对照，后续使用新场景评估。

## v3空间/时间token与非目标位移（2026-10-23）

用户要求以后在项目根目录写当前进度，已建立 `当前进度.md`，v3程序自动更新实时段。启动服务 `corl-spatial-specificity-v3`。训练initial20..31、验证32..35、全新测试40..49，共26场景。目标akita_black_bowl_1与非目标akita_black_bowl_2分别做X/Y±5/±10毫米。视觉层0/5/11/17保留每相机4×4空间网格；动作专家保留action位置0/10/25/49在denoise0/9的特征。固定噪声和机器人state、统一强制刷新、排除穿插。只用目标位移训练decoder，在相同场景/位移且双方有效的分支比较目标与非目标读出大小。不能把位置不同引起的选择性解释成语言目标绑定；无闭环修复结论。输出 `outputs/spatial_specificity_v3/`。

v3首次加载与其他GPU进程重叠（错误显示另一进程2.8GiB），OOM，未停止其他进程。改为仅此v3设置VLA_VISION_CPU=1：FP32视觉塔与投影CPU执行，语言/动作专家仍GPU BF16，词嵌入CPU；不降精度或改变权重，CPU/GPU数值内核差异已记录，v3样本统一此配置。加载前要求空闲显存5400MiB，否则每5秒等待最多30分钟并更新根目录进度。首次协议与错误保留，未产生数据，不混合配置。

用户要求v3视觉切回GPU，已停止CPU视觉加载（尚无样本），恢复GPU FP32视觉与GPU BF16语言/动作专家；词嵌入仍CPU。等待空闲显存门槛6900MiB，旧CPU协议归档，不混合样本。
