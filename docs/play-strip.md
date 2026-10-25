# 长条测试场地

Rough、DreamWaQ、Quad Pedipulation 的键盘播放默认使用专用长条场地。训练配置和检查点不变，actor 的高度扫描输入仍然计算完整物理地形。

## 场地布局

长度 **62 m**，宽度 **10 m**。中间是与训练相同的 8 m 宽地形，两侧各有 1 m 平地肩带；外围还有 1 m 平地缓冲。沿世界 +X 方向排列：6 m 平地起点，然后每类地形各占 8 m，原来的七种类型每种都生成一次。

| 世界 X 区间（m） | Rough | DreamWaQ / Quad Pedipulation |
| --- | --- | --- |
| -31 ～ -25 | 平地起点 | 平地起点 |
| -25 ～ -17 | 平地 | 平地 |
| -17 ～ -9 | 金字塔上楼梯 | 随机粗糙地面 |
| -9 ～ -1 | 倒金字塔楼梯 | 正向坡面 |
| -1 ～ 7 | 正向坡面 | 倒置坡面 |
| 7 ～ 15 | 倒置坡面 | 箱格地形 |
| 15 ～ 23 | 随机粗糙地面 | 金字塔上楼梯 |
| 23 ～ 31 | 波浪地形 | 倒金字塔楼梯 |

出生位置固定在 `(-28, 0)`，机身朝 +X，按 W 就是沿场地向前走。按 A/D 可以横向移动。窗口显示当前地形区段。重置仍回到起点，保留跌倒终止；最长一小时的播放回合避免在走完场地前被原来 20 秒超时打断。

地形种类按源配置的字典顺序固定排列，随机种子为 42，难度固定为 0.55。继续使用原生地形生成函数和原来的坡度、台阶范围、粗糙度、箱格参数；这是一个固定中等难度样本，并不是训练全集。当前 Rough 楼梯每级约 5.5 cm，DreamWaQ/Quad 每级约 14.9 cm。Native 箱格要求方形区域，因此各段保持原来的 8×8 m 几何，额外的宽度由两侧平地肩带提供。

## 启动

原来的键盘启动命令直接生效。也可以明确选择长条场地：

```bash
source /home/e980/miniforge3/etc/profile.d/conda.sh
conda activate cjisoad_mjlab
cd /home/e980/go2_rl_mjlab
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python scripts/play.py Unitree-Go2-Rough \
  --checkpoint-file logs/selected_policies/2026-10-25/Unitree-Go2-Rough/model_2999.pt \
  --num-envs 1 --viewer native --keyboard True --play-terrain strip
```

将最后一个参数改为 `--play-terrain original`，可恢复原来的地图。`auto` 是默认值：键盘播放且任务有生成地形时选择长条；其他情况保持原场景。Flat、AMP、StandUp、Biped Pedipulation、Backflip 在 auto 下继续使用平地。

键盘模式默认关闭阴影、反射、调试标记；按 R 可临时显示调试信息，大量射线标记会增加绘制开销。只为显示临时过滤远处静态地形，碰撞和扫描仍保留完整场地。滚轮拉远镜头会扩大绘制范围，可以查看整条场地；近距离跟随机器人时主要绘制附近区段。

## 验证与性能记录

测试使用实际 MuJoCo 几何和向下射线，检查出生点、边界衔接、横向地面覆盖、固定随机样本和训练配置不变；渲染过滤检查模型分组与射线高度在绘制后保持不变。

本机 RTX 4060 Laptop / WSLg / Mesa D3D12 NVIDIA 的原始 Rough 地图测试，单独完整显示刷新中位耗时约 2357 ms；关闭调试、阴影、反射后约 144 ms。长条加低开销显示设置的第一次测试约 33 ms，后续再增加远地形绘制过滤。单帧显示耗时不包含策略推理和 GPU 仿真，不能作为完整播放帧率。最终完整播放测量和测试记录见仓库 `logs/play_strip_validation/2026-10-25/`。

最终短时完整播放测量（单 CPU 线程）：Rough 23.8 FPS / 0.59× 实时，DreamWaQ 27.6 FPS / 0.71× 实时，Quad Pedipulation 23.5 FPS / 0.63× 实时。86 项测试通过。该机器上尚未达到满速实时，不将单独渲染帧耗时等同于完整播放速度。
