"""Descriptive plot and Chinese summary; does not select models by test performance."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];OUT=R/'outputs/spatial_specificity_v3'
d=json.loads((OUT/'results.json').read_text());rows=d['probes'];best=next(r for r in rows if r['feature']==d['selected_by_validation'])
fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
axs[0].bar([r['feature'] for r in rows],[r['target_test_rmse_mm'] for r in rows]);axs[0].axhline(d['constant_test_rmse_mm'],color='red',linestyle='--',label='Training-mean baseline')
axs[0].set_ylabel('Target displacement coordinate RMSE (mm)');axs[0].legend();axs[0].tick_params(axis='x',rotation=75)
x=np.arange(len(rows));axs[1].bar(x-.2,[r['matched_mean_target_decoded_norm_mm'] for r in rows],width=.4,label='Target shifted');axs[1].bar(x+.2,[r['matched_mean_distractor_decoded_norm_mm'] for r in rows],width=.4,label='Distractor shifted')
axs[1].set_xticks(x,[r['feature'] for r in rows],rotation=75);axs[1].set_ylabel('Mean decoded target displacement norm (mm)');axs[1].legend()
fig.suptitle('Exploratory spatial readouts: ten held-out scenes, matched valid shifts');fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
validity=[json.loads(p.read_text()) for p in OUT.glob('*_validity.json')];valid=sum(r['valid'] for r in validity);invalid=sum(len(r['invalid']) for r in validity)
text=f'''# v3空间与非目标位移读出结果

26场景，目标及非目标黑碗各8种位移；有效{valid}分支，无效{invalid}分支。训练/验证/测试按场景隔离：{d['samples']}（此处只统计目标样本），测试初始状态40..49为新场景。

验证集选出的激活：`{best['feature']}`。测试目标位移坐标RMSE {best['target_test_rmse_mm']:.3f}毫米，常数对照 {d['constant_test_rmse_mm']:.3f}毫米。同一个探针在双方有效的{best['matched_samples']}组位移中，目标位移读出平均范数 {best['matched_mean_target_decoded_norm_mm']:.3f}毫米，非目标位移读出平均范数 {best['matched_mean_distractor_decoded_norm_mm']:.3f}毫米。

这些数值是读出能力和特定对照下的响应差异，不是闭环任务成功率，也不证明语义绑定、反馈机制或修复。另一只黑碗位于不同空间位置，差异可能来自空间位置或可见性。v2与v3使用不同测试场景，不能直接将误差差值解释为空间池化的改进。完整候选与简单图像均值对照见 `RESULTS.md`。

下一步依据结果决定：若空间读出仍弱，先检查token定位与数据；若读出可靠，再用同一物体切换语言目标等对照排除位置捷径，并验证特征干预的闭环行为。尚不开始SAE训练。
'''
(OUT/'RESULTS_ZH.md').write_text(text)
p=R/'当前进度.md';s=p.read_text();start=s.index('## 正在进行');end=s.index('## 下一步与判断标准')
s=s[:start]+f'''## 最新完成：v3

- 26场景、{valid}个有效位移分支；保留视觉4×4空间网格和动作token时间位置。
- 新测试场景40..49；验证选择 `{best['feature']}`，测试目标位移RMSE {best['target_test_rmse_mm']:.3f}毫米，常数对照 {d['constant_test_rmse_mm']:.3f}毫米。
- 非目标位移对照已完成；结果只反映读出，不能直接证明语义或因果机制。
- 详细说明：`outputs/spatial_specificity_v3/RESULTS_ZH.md`。

'''+s[end:]
s=s.replace('- `outputs/fixed_state_probe_v2/RESULTS_ZH.md`：最新已完成实验说明。','- `outputs/fixed_state_probe_v2/RESULTS_ZH.md`：上一轮固定状态实验。')
p.write_text(s)
print(text)
