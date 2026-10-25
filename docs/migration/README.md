# Go2 方法迁移说明

## 来源与范围

目标仓库基线为 `1425b15`，工作分支 `codex/go2-methods-migration`。保留全部 Tracking 及 Go2 Flat/Rough；移除其他机器人速度任务、模型和 deploy/simulate 程序。Tracking 依赖的 G1/G1-23DoF 模型与动作保留。删除清单见 [removal-manifest.json](../removal-manifest.json)。

源端为用户提供的 `unitree_rl_lab` 工作区。DreamWaQ baseline、AMP、StandUp V2.1 使用读取时的源文件，包括源工作区未提交修改；空翻使用用户指定的 V1.6 最后训练快照恢复源码。详细路径、基线及 SHA-256 见 [source-manifest.json](source-manifest.json)。远端仅作只读读取。

## 实现与物理适配

| 方法 | 观测和算法 | 保留的关键行为 |
| --- | --- | --- |
| DreamWaQ | actor 45、历史 225、privileged 202；critic 247；速度 3 + 潜变量 16 | VAE 编解码、速度监督/重构/KL 与 PPO 共用优化器、训练随机上下文、推理确定性上下文、7 类地形及课程 |
| AMP | actor 45、critic 60、AMP 特征 30 | 原 10 段动作、FL/FR/RL/RR 特征次序、0.6 任务 + 0.4 风格奖励、BCE/R1、回放缓冲区、自动 reset 前的终止特征捕获、判别器/优化器/归一化保存恢复 |
| StandUp V2.1 | actor 45、critic 86 | 后腿直立移动、23 项源奖励、原批次高度门控、随机子步延迟、严格超时条件、镜像损失与命令课程 |
| Backflip V1.6 | actor 60、critic 64 | 21 项源奖励、空翻相位及俯仰累计、FR/FL/RR/RL 次序、一个控制步延迟、2 秒回合；保留源 reset 时目标缓存语义 |

四种迁移任务均使用物理步长 0.005 s、decimation 4。Go2HV 电机保留源扭矩/速度包络；PD 分别为 DreamWaQ/AMP 25/0.5、站立 40/1、空翻 70/3。空翻的关节次序显式覆盖原生 JointPositionAction 的解析顺序。

原生 MuJoCo 模型的 calf body 同时包含足球。非足碰撞奖励仅选非足几何体，并先按源 body 汇总力向量，再进行阈值与历史判断，避免把正常脚支撑误计为小腿碰撞。保留源端启用的自碰撞及未过滤接触报告。质量与质心随机化使用 mjlab 对应事件。

接触几何体和求解器来自 MuJoCo，无法与源 USD/PhysX 完全等同。MuJoCo 单个滑动摩擦系数近似源静/动摩擦，无法逐项复制源材质桶与混合模式；有弹性系数随机化的任务通过 MuJoCo `solref` 阻尼映射近似，其范围和映射有数值测试。站立保留源 point-mass 随机化方式。这些物理差异可能改变训练速度和最终行为。

没有修改已安装的 mjlab/RSL-RL，也没有以自定义包覆盖 `rsl_rl`。DreamWaQ 使用 native PPO 扩展，AMP 将判别器优化接入同一 Adam 更新；各方法的扩展留在独立包内。脚本从任意启动目录优先导入本仓库，避免本机其他同名 `src` 编辑安装包被误导入。

## 训练预算与验证

默认轮数及证据见 [training-budgets.md](training-budgets.md)。本次未读取到完整远端训练曲线，不能确认实际平台期；按已保存的所选版本训练配置设置预算。

本机验证记录见 [verification.json](verification.json)。CPU 检查覆盖源公式、维度与顺序、历史 reset、判别器/估计器更新、镜像、延迟控制、接触聚合和导出元数据。CUDA 冒烟检查覆盖环境交互、超时 reset、两轮 PPO、检查点恢复及 ONNX 模型检查。Tracking 使用由原 CSV 前 15 帧转换出的短参考动作，只验证工具链与接口。

短测试没有训练出成熟策略，没有确认空翻/站立成功率，也没有测得完整训练平台期。新实现应重新训练；本次没有转换旧 Isaac policy 检查点。用少量环境验证后，用户可按显存和吞吐设置训练规模，并同时观察奖励、回合长度、速度/姿态/落地等任务指标。

## 许可证

保留目标仓库 `LICENCE`。DreamWaQ/RSL 以及 AMP 算法来源的许可证保存在各方法目录；站立保留 `SOURCE_LICENSE` 和原始来源清单。AMP 示例数据来源见 `src/assets/motions/go2/amp/PROVENANCE.md`；代码的 BSD 许可证不自动授予动作数据独立许可。
