# Go2 右前腿操作任务

两个任务在原有速度指令之外增加右前脚位置指令。四足任务基于 DreamWaQ，双足任务基于 StandUp V2.1 后腿站立。参考本机 Cjisoad_mjlab 的位置误差指数奖励、放回阶段权重和操作腿奖励屏蔽思路；没有引入其界面与课程管理系统。

| 任务 ID | 基础任务和地面 | 默认训练轮数 | 并行环境 |
| --- | --- | ---: | ---: |
| Unitree-Go2-Quad-Pedipulation | DreamWaQ，含台阶、斜坡、箱格等粗糙地形 | 5000 | 4096 |
| Unitree-Go2-Biped-Pedipulation | StandUp V2.1，平地后腿站立与行走 | 6000 | 4096 |

## 命令和坐标

速度命令沿用基础任务：四足为 twist，双足为 base_velocity。新增 foot_position 为三维米制坐标，以 base_link 原点为原点，轴方向随机身转动。站立后也仍使用机身坐标，而不是世界坐标或水平航向坐标；无需外部定位来构造该命令。脚端是 FR site 的位置，不含球形脚的朝向。

actor 在原有 45 维观测末尾附加位置命令三维，得到 48 维。命令 manager 使用两个独立 term，合起来提供速度3维＋位置3维。四足历史为5帧，共240维；双足 critic 为89维。actor 只使用 IMU角速度、重力方向、关节位置和速度、上一动作，以及两组命令。实际脚端世界位置、真值线速度、接触力和地形数据仅用于奖励、critic 或 DreamWaQ 估计器监督。

## 阶段与放回

每个6秒周期包括休息2秒、平滑伸出1秒、保持2秒、平滑收回1秒。每个新周期有65%概率采样操作目标，其余周期保持基础运动。首次周期开始前，四足等待1秒、双足等待3秒。阶段完全按时间推进，不读取真实接触或站立高度来生成 actor 的命令。

训练目标通过实际 Go2 模型的正运动学生成：在正常关节范围内扰动右前腿，再筛选小范围位置目标。四足额外要求目标比正常脚位抬高4–12厘米。该采样保证目标端点有对应关节解，不保证每个地形上都无碰撞，也不代表策略已学会到达它。

四足收回至 DreamWaQ 实际默认姿态对应的脚位，收回后段奖励位置恢复及地面接触；操作时保留其他三腿的支撑约束。地面奖励使用独立、只匹配 terrain 的传感器，机器人自碰撞不能算作落脚。

双足收回至站立姿态的前腿收拢位置，不奖励前脚落地。脚端跟踪和放回奖励仅在该环境机身已接近后腿站立且高度足够时生效。操作时放松右前腿的默认姿态及前腿成对约束，保留左前腿和后腿约束。该单侧任务关闭原 StandUp 的左右镜像损失。

末端跟踪使用 exp(-位置误差平方和/0.05²)，权重2；收回后段的回位奖励权重1。四足地面恢复与其他三脚支撑奖励权重各0.5。参数位于 src/tasks/go2/pedipulation/env_cfg.py。奖励与物理配置都通过独立工厂生成，原始任务不受修改。

## 训练

在 WSL 中执行：

```bash
cd /home/e980/go2_rl_mjlab
source /home/e980/miniforge3/etc/profile.d/conda.sh
conda activate cjisoad_mjlab
python scripts/train.py Unitree-Go2-Quad-Pedipulation
python scripts/train.py Unitree-Go2-Biped-Pedipulation
```

分别写入 logs/rsl_rl/go2_quad_pedipulation 和 logs/rsl_rl/go2_biped_pedipulation。服务器队列已按用户要求扩展为原六项任务之后依次训练四足、双足操作任务；原运行进程完成后由接续进程启动新增任务。

仿真或后续部署控制器设置目标的接口：

```python
command = env.command_manager.get_term("foot_position")
ids = torch.arange(env.num_envs, device=env.device)
# 演示使用已验证端点可达的目标；实际外部命令也需要满足运动范围。
command.set_target(command.targets[0].expand(env.num_envs, -1), ids)
# goal 是完整目标，command.command 是经平滑插值后的当前位置参考。
# 手动目标跨周期保留，每周期按相同时序伸出和收回。
command.clear_target(ids)  # 回到休息参考，下个自动周期恢复随机采样。
```

## 验证

```bash
python -m unittest discover -s tests
python scripts/check_tasks.py Unitree-Go2-Quad-Pedipulation
python scripts/check_tasks.py Unitree-Go2-Biped-Pedipulation
python scripts/check_pedipulation.py
```

check_tasks 验证重置、观测和奖励有限、2轮 PPO 更新、检查点读回及 ONNX。check_pedipulation 保持未训练环境存活，覆盖完整指令周期，验证阶段和数值稳定性；不会把未训练机器人跌倒误认为训练策略失败。短测不能证明策略收敛、末端操作成功率或实机性能。

本机验证记录：59项单元测试、两个新任务和原DreamWaQ的GPU短程训练/读回/导出均通过，两个新任务完整命令周期通过；详见 [验证结果](pedipulation-verification.json)。
