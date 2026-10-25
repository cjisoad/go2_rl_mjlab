# 默认训练轮数依据（本地缓存审计，2026-10-24）

服务器已按用户说明关闭，后续不再尝试连接。本次只能读取已经下载的源码和训练参数快照，不能确认真实训练平台期。缓存中没有远端 TensorBoard event 文件或标量曲线；`max_iterations`、运行目录名称和模型编号均不能代替收敛证据。

## 建议配置

所选源版本的 DreamWaQ baseline 和 AMP 都配置为 50,000 轮，但用户指定迁入本仓库后的默认轮数分别改为 5,000 和 10,000；这是训练预算偏好调整，不表示已依据收敛曲线确认较短轮数足够。AMP 训练默认并行环境数恢复为 2,048；Flat、Rough、DreamWaQ、StandUp、Backflip 为 4,096。回放仍使用较少环境。Backflip/StandUp 轮数和保留 Tracking 配置不变。以下数值均指 PPO 更新次数，每次采样 24 个控制步。

| 任务 | 建议默认 max_iterations | 证据与解释 |
| --- | ---: | --- |
| Unitree-Go2-DreamWaQ | 5,000 | 用户指定默认值。所选 baseline 源配置为 50,000（不是 Exp2 的 20,000）；降低后的默认值不代表已确认平台期。 |
| Unitree-Go2-AMP | 10,000 | 用户指定默认值。所选 AMP 源配置为 50,000；缺少策略奖励、风格奖励和判别器曲线，降低后的默认值不代表已确认平台期。 |
| Unitree-Go2-StandUp | 6,000 | 最新 V2.1 的 `go2_standup_v21_cfg.py:8` 明确覆盖父 V2 的 15,000。不要误用旧直立方法的 100,000 / 50 步 rollout。 |
| Unitree-Go2-Backflip-V1.6 | 1,000 | 源配置及三份精确 V1.6 运行 `params/agent.yaml:4` 均为 1,000；最后一份所选运行 `2026-07-04_23-21-24_backflip_pitch_progress_v1.6`，rollout 24、resume false。这是已配置预算，未证明执行完毕或达到平台。 |
| Unitree-Go2-Flat | 2,000 | 用户指定默认值；无可比较的远端曲线。 |
| Unitree-Go2-Rough | 3,000 | 用户指定默认值；不把 DreamWaQ 版本的预算当作普通 Go2 PPO 的收敛证据。 |
| Unitree-G1-Tracking | 30,001 | 保留目标仓库原 `src/tasks/tracking/config/g1/rl_cfg.py:46`；不同动作数据会改变收敛速度。 |
| Unitree-G1-Tracking-No-State-Estimation | 30,001 | 与原配置共享预算，没有该变体的曲线证据。 |
| Unitree-G1-23Dof-Tracking | 30,001 | 保留目标仓库原 `src/tasks/tracking/config/g1_23dof/rl_cfg.py:46`。 |
| Unitree-G1-23Dof-Tracking-No-State-Estimation | 30,001 | 与原配置共享预算，没有该变体的曲线证据。 |

四个 Tracking 任务保留原目标仓库 30,001 的默认值。Flat、Rough、DreamWaQ、AMP 的新轮数是用户指定值，并非平台期推断。

## 已核实的日志参数

以下三份缓存均为 `run_name: backflip_pitch_progress_v1.6`、`max_iterations: 1000`、`num_steps_per_env: 24`、`resume: false`：

- `logs/rsl_rl/unitree_go2_backflip/2026-07-04_19-42-18_backflip_pitch_progress_v1.6/params/agent.yaml`
- `logs/rsl_rl/unitree_go2_backflip/2026-07-04_21-28-58_backflip_pitch_progress_v1.6/params/agent.yaml`
- `logs/rsl_rl/unitree_go2_backflip/2026-07-04_23-21-24_backflip_pitch_progress_v1.6/params/agent.yaml`

没有将 V1.6re、V1.6re2 或 V1.7 的模型表现当作精确 V1.6 的平台期依据。缓存中的站立文档描述 V2 的 15,000，所选 V2.1 源配置的 6,000 具有更直接的版本对应关系。

## 平台期结论的边界

所有任务的 `plateau_confirmed` 均为 false，`plateau_iteration` 均为空。当前建议是基于现有证据的初始训练预算，不声称“已确认第 N 轮收敛”。源端 Isaac Lab 与目标 mjlab 的接触求解、机器人模型和训练实现不同，相同预算也不保证相同收敛轮数。改变并行环境数还会改变每轮总采样数；例如 4096 环境、24 步为每轮 98,304 个 transition。

后续若补充完整远端 TensorBoard 曲线，应同时检查训练奖励、平均回合长度和任务指标：DreamWaQ/Go2 的速度误差与地形难度、AMP 的任务/风格奖励与判别器指标、站立的姿态/支撑/速度指标、空翻的完整旋转与落地成功率、Tracking 的运动跟踪误差。仅总奖励平坦不足以判定任务成功。当前未取得这些曲线，因此未拟合或生成伪造的平台期图。

## 审计材料

[training-budgets.json](training-budgets.json) 保存全部任务的预算、来源位置、行号、SHA-256 和三份运行参数证据。
[source-manifest.json](source-manifest.json) 保存完整来源清单及所选版本说明。原服务器绝对路径是历史来源标识，不表示本仓库附带原始服务器目录。

本次核验了 10 个任务预算、3 份 V1.6 参数快照，缓存的远端 TensorBoard event 文件为 0；因此没有可发布的训练曲线数据。
