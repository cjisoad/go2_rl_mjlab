# Go2 键盘控制

键盘模式可以同时给机器狗发布速度命令、给右前腿发布末端位置命令，使用当前下载的策略推理。仅在播放时替换命令生成器，不改变训练配置、actor 观测维度或检查点。

## 在 WSL 中启动

```bash
source /home/e980/miniforge3/etc/profile.d/conda.sh
conda activate cjisoad_mjlab
cd /home/e980/go2_rl_mjlab
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
```

四足操作：

```bash
python scripts/play.py Unitree-Go2-Quad-Pedipulation \
  --checkpoint-file logs/selected_policies/2026-10-25/Unitree-Go2-Quad-Pedipulation/model_4950.pt \
  --num-envs 1 --viewer native --keyboard True
```

双足操作：

```bash
python scripts/play.py Unitree-Go2-Biped-Pedipulation \
  --checkpoint-file logs/selected_policies/2026-10-25/Unitree-Go2-Biped-Pedipulation/model_4900.pt \
  --num-envs 1 --viewer native --keyboard True
```

键盘模式使用仓库自有的 GLFW 窗口和 MuJoCo 渲染器，已关闭 MuJoCo Simulate 内置快捷键，按键只执行下表中的命令。鼠标左键拖动转动视角、右键拖动平移、滚轮缩放；按住 Shift 可切换拖动方向。

点击仿真窗口，使它获得键盘焦点，使用英文输入法。一次只运行一个播放窗口。布尔参数沿用 mjlab 的显式写法 `--keyboard True`。省略 `--num-envs` 时，键盘模式也会使用单环境；当前不支持 Viser 浏览器键盘控制。

其他速度策略可以使用同样的参数，例如：

```bash
python scripts/play.py Unitree-Go2-Flat \
  --checkpoint-file logs/selected_policies/2026-10-25/Unitree-Go2-Flat/model_1900.pt \
  --num-envs 1 --viewer native --keyboard True
```

Flat、Rough、DreamWaQ、AMP、StandUp 支持机器狗速度命令；只有 Quad/Biped Pedipulation 的 actor 接收脚端目标，才启用右前腿控制。Backflip 和 Tracking 没有兼容的速度命令，使用键盘模式会在创建仿真前给出错误说明。

## 键位

| 按键 | 命令 | 每次按键的变化 |
| --- | --- | --- |
| W / S | 前进速度增加 / 减少 | ±0.1 m/s |
| A / D | 向左速度增加 / 减少 | ±0.1 m/s |
| Q / E | 左转角速度增加 / 减少 | ±0.1 rad/s |
| X | 行走速度三轴归零 | 不改变脚端目标 |
| I / K | 右前脚目标沿机身 +X / -X 移动 | ±1 cm |
| J / L | 右前脚目标沿机身 +Y / -Y 移动 | ±1 cm |
| U / O | 右前脚目标沿机身 +Z / -Z 移动 | ±1 cm |
| H | 右前脚平滑放回 | 不改变行走速度 |
| Space | 暂停 / 继续 | |
| Enter | 重置机器人、速度归零、脚端回到默认位置 | |
| - / = | 降低 / 提高播放速度 | |
| R / P | 切换调试可视化 / 奖励曲线（仓库定义） | |
| Esc | 关闭播放窗口 | |

这些是**增量命令**：例如按 W 三次，前进速度变为 0.3 m/s，松开后继续保持；按 S 减小该值，按 X 直接清零。不是松键即停的游戏式控制。速度到达当前任务的播放范围边界后不再增加。键盘模式不会启动 MuJoCo Simulate 的默认快捷键处理器，因此行走和脚端按键不会同时改变显示选项。

脚端坐标以机身为基准，正常四足姿态下 +X 朝头部、+Y 朝左、+Z 朝上；双足站起时这些轴也随机身旋转，并非世界坐标方向。StandUp/Biped 的速度保持源策略的站立坐标约定：前进跟踪机身 -Z 速度，转向跟踪机身 X 角速度；横移训练范围是 0，因此 A/D 在这两个任务中不会产生横移命令。

## 脚端目标与放回

每次脚端按键移动请求位置 1 cm。请求位置先限制在训练工作空间的包围盒内，再投影到训练时“默认位置到目标位置”的参考线段集合上，因此实际目标可能不同于请求位置；窗口同时显示 requested（请求）、goal（投影后的目标）、reference（当前平滑参考）。参考在 1 秒内平滑移动到目标并保持，不再执行原来的 6 秒随机循环。重新指定目标从当前参考开始，按 H 则用 1 秒放回默认位置。

Quad 前 1 秒、Biped 前 3 秒沿用原任务的姿态准备时间；这段时间可以输入目标，参考会在准备时间结束后开始移动。命令由键盘和时钟决定，不向 actor 加入真实末端位置、接触力等额外状态。机器人因跌倒或超时重置时，速度和脚端目标也清零 / 归位。

键盘只指定策略目标。实际脚端是否跟上、机器狗是否保持平衡，仍取决于策略能力；在原来短时伸腿训练上长时间保持目标属于不同的时间分布。

## 验证

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
python scripts/check_keyboard_control.py
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA python scripts/check_keyboard_control.py --native --task Quad-Pedipulation
```

CUDA 检查加载本地下载的真实检查点，检查速度保持、脚端保持超过 6 秒、actor 接收到的目标、放回、重置以及动作数值有限。可用 `--task Rough --task DreamWaQ --task AMP --task StandUp` 检查其他任务。检查时关闭终止条件以完整执行目标保持 / 放回阶段，因此这是控制接口验证，不是策略成功率评估。原生窗口检查还验证按键入队、状态文字渲染、暂停和 Enter 重置，随后自动关闭测试窗口。

2026-10-25 本机验证结果：80 项测试通过；此前 7 个速度任务 CUDA 检查通过；快捷键修复后 Flat/Quad/Biped 三个窗口检查通过，包括已注册的 GLFW 回调、渲染选项不受控制键影响、暂停、重置、鼠标缩放和 Esc 关闭。记录保存在 `logs/keyboard_validation/2026-10-25/shortcut-fix/`。

粗糙地形键盘播放现在默认使用 [62×10 m 长条测试场地](play-strip.md)，保留原有全部七种类型。使用 `--play-terrain original` 切回原地图。默认关闭调试标记、阴影和反射；R 可临时打开调试标记。
