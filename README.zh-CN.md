# TritNet-MCU（tritnetmcu）

[![ci](https://github.com/weifeng233/tritnet-mcu/actions/workflows/ci.yml/badge.svg)](https://github.com/weifeng233/tritnet-mcu/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

简体中文 | [English](README.md)

一个纯 Zig 的训练、量化、部署框架，面向三元权重 {-1, 0, +1} 的 MCU
模型，附带四个可复现的样例。除 libc 外零依赖：一条命令完成 fp32
预训练、BitNet 风格量化感知训练（QAT）、留出集验证、三元打包导出，
以及验证打包产物无损的往返校验。每个样例在笔记本上数秒完成。

框架本体在 `src/framework.zig`（RNG、三元编解码、absmean QAT 量化器、
Adam 与余弦调度、打包模型容器、计时）。`examples/` 下是四个不同的
模型家族，`examples/*/c/` 的 C 引擎负责部署验证：只凭打包产物复现
训练器算出的全部指标。

## 流水线

每个样例在同一进程里跑完五个阶段：

1. fp32 预训练：Adam 加余弦学习率调度、梯度裁剪，能用 SIMD 的地方用
   `@Vector` 的 AVX2/NEON 内核。
2. BitNet 三元 QAT：每个线性投影按 absmean 缩放三元化
   （`scale = mean(|W|)`，绝对值超过 `scale/2` 的权重取 `+1/-1`，其余为
   `0`），用直通估计器（STE）训练。梯度按 `|W| < scale` 掩码，激活按
   token 做 8 比特量化，嵌入与偏置保持 fp32。
3. 验证：在留出集上对比 fp32、朴素 PTQ 与 QAT，并统计 trit 分布、
   延迟与存储。
4. 三元打包：每字节 5 个 trit
   （[compilade 的方案](https://compilade.net/blog/ternary-packing)，
   llama.cpp 为 BitNet 模型用的也是它）。三进制基数编码加向上取整
   重标定，解码只用乘法与移位，不含除法。
5. 往返校验：重新加载打包文件、解码、再评估，指标必须与导出前逐位
   一致。

### 框架模块

| 模块 | 职责 |
| --- | --- |
| `framework/rng.zig` | 确定性 xorshift 流：uniform / gauss / range |
| `framework/ternary.zig` | 每字节 5 trit 的权重编解码（pack5 / unpack5） |
| `framework/quant.zig` | absmean 量化、STE 梯度掩码、8 比特激活量化、wired-only 缩放变体 |
| `framework/adam.zig` | 8 宽 SIMD Adam、余弦学习率调度、偏置修正 |
| `framework/packed_model.zig` | 小端模型文件写入器/读取器、共享的三元矩阵记录 |
| `framework/timing.zig` | 可移植单调时钟 |

新增样例只需要写模型侧的代码：前向与反向传播，以及基于这些原语的
导出布局。

## 构建与运行

```bash
zig build --release=fast                # 框架样例 + fp32 基线 + C 对照
zig build test                          # 框架 + 样例单元测试
zig build --release=fast example-gpt    # 样例 1：micro-GPT 人名
zig build --release=fast example-cnn    # 样例 2：TinyML 手势 1D-CNN
zig build --release=fast example-ncps   # 样例 3：CfC/LTC 神经回路策略
zig build --release=fast example-timestral  # 样例 4：Timestral-5M 量化
zig build --release=fast run            # 仅 fp32 基线（microGPT）
make deploy-check                       # 对全部打包产物做 C 部署校验
make timestral-check                    # 样例 4 与 PyTorch 的端到端数值校验
make sketch                             # 样例 4 的绘图预测网页（http://127.0.0.1:8000）
```

依赖：Zig 0.16.0、`make`，Python 3.11+ 只给工具链用
（`examples/*/python/`、`examples/timestral_sketch/deploy/`）。Zig 部分零依赖，
Python 依赖由 `uv` 锁定：

```bash
uv sync --extra eval                 # 作图与指标
uv sync --extra qat --extra eval     # Timestral 微调
uv sync --extra serve                # 绘图服务
```

## 文档

| 文档 | 回答什么问题 |
| --- | --- |
| [`docs/ARTIFACT-FORMAT.md`](docs/ARTIFACT-FORMAT.md) | 五种二进制格式逐字段说明，以及三元编解码的黄金测试向量 |
| [`docs/TIMESTRAL.md`](docs/TIMESTRAL.md) | 1.6 bit/权重在 Timestral-5M 上到底付出多少代价（实测） |
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | 吞吐与其测量条件 |
| [`THIRD_PARTY.md`](THIRD_PARTY.md) | 所有上游来源及其许可证 |
| [`CHANGELOG.md`](CHANGELOG.md) | 改了什么，哪些数字变了 |

## 设计决策

下面五条选择决定了仓库其余部分的形状。每段只写结论，支撑它的数字在
上面的文档里。

流水线用 Zig，C 只出现在第二实现的位置。训练与部署同一种语言，就
消掉了「在我机器上是对的」那个边界。`zig build` 同时是构建系统、测试
运行器和交叉编译器，`@Vector` 一份源码产出 AVX2 或 NEON。C 引擎留着，
因为它的职责是与 Zig 对拍。

权重编解码一个字节装五个 trit。一个 trit 携带 log2(3) ≈ 1.585 bit，而
`3^5 = 243 ≤ 256`，于是一个字节装五个、1.6 bit/权重，解码是一次乘法、
一次移位、一次掩码，不含除法（目标 Cortex-M 核没有整数除法）。改成
2 bit/权重，体积会大 25%，解码却并不更便宜。

每个要出厂的产物配一个 C 引擎。往返检查的读方和写方是同一份代码，
格式被共同误解时照样能完美往返。所以每个引擎只读打包文件，自己重算
指标，再与训练器写进产物 footer 的数字比较，不一致就让
`make deploy-check` 失败。

标度用 absmean，不用学习出来的阈值。`scale = mean(|W|)` 加固定 0.5
阈值，使产物成为权重的确定性函数：一条命令、没有标定语料、没有优化器
状态要版本化。absmean 不够好的地方，这里的答案是更多训练（QAT），
而不是更聪明的量化器。

fp32 后端保留。量化后的数字离开同进程的基线就没有意义。两个后端回答
同一个接口，1.77 bit 的代价因此画在同一张图上，而不是只写在表格里。

## 样例 1：`gpt_names`，字符级 micro-GPT

microGPT-C 里的微型字符级 Transformer（4,192 参数：1 层、4 头、
16 维嵌入、64 维 MLP、ReLU²、RMSNorm），在约 3.2 万人名上训练并生成
新人名。以三元权重训练并部署后，模型文件约为 fp32 权重的 1/22。

以下结果在 examples/gpt_names/data/names.txt 上测得（80/20 训练/留出划分，
20000 步 fp32 预训练加 5000 步 QAT，Ryzen 9 7940H 上共约 1 秒）：

```
eval (avg nats/char, float activations)
  fp32  train 2.1944   val 2.1987
  ptq   val 2.4948   (ternary, no QAT)
  qat   train 2.2722   val 2.2700
  trits: -1 35.8%   0 26.5%   +1 37.7%

storage
  packed 5-trits/byte  :    761 bytes  (比 fp32 小 22.0x，含 scale 后 1.74 bits/trit)
  model file           :   3579 bytes (嵌入层保持 fp32)
packed round-trip: OK (lossless)

summary
  fp32 val 2.1987 | ptq val 2.4948 | qat val 2.2700 | qat 恢复 fp32->ptq 差距的 75.9%
```

```bash
zig build --release=fast example-gpt    # 写出 examples/gpt_names/modeldata/tritnetmcu_packed.bin
```

## 样例 2：`gesture_cnn`，TinyML 传感器分类器

EdgeImpulse 与 MLPerf-Tiny 里的 1D-CNN 任务：从三轴加速度计窗口识别四类
手势（静止、摇晃、画圈、敲击）。数据由固定种子确定性生成，带随机相位、
幅度抖动与逐样本噪声，不需要外部数据集。模型 2,524 参数，每次推理
54,208 次乘加：

```
input 3x128
Conv1D(3 -> 8, k=8)  + ReLU + MaxPool(2)   -> 8x60
Conv1D(8 -> 16, k=4) + ReLU + MaxPool(2)   -> 16x28
Flatten -> Dense(448 -> 4) + softmax
```

偏置保持 fp32（2,524 个参数里的 28 个），所有卷积核与全连接权重都
三元化。样例的单元测试把卷积与池化内核与直接参考实现对照，另外做一次
端到端的数值梯度检查。

Ryzen 9 7940H 实测（4000 样本，80/20 划分，2000 步 fp32 预训练加 1500 步
QAT，共约 11 秒）：

```
eval (accuracy, float activations)
  fp32  train 100.00%   val 100.00%   (val loss 0.0014)
  ptq   val 65.75%   (ternary, no QAT)
  qat   train 100.00%   val 100.00%
  trits: -1 42.6%   0 21.2%   +1 36.2%

storage
  packed 5-trits/byte  :    672 bytes  (比 fp32 小 15.0x)

latency (host CPU, float activations)
  fp32 weights         :  45.67 us/inference  (1187.0M MACs/s)
  ternary weights      :  42.93 us/inference  (1262.6M MACs/s)

packed round-trip: OK (lossless)   qat 恢复 fp32->ptq 损失差距的 99.9%
```

朴素三元化把验证准确率降到 65.75%，QAT 又把它拉回 100%，此时每个内核
都以 1.6 bit/权重的 {-1, 0, +1} trit 运行。

```bash
zig build --release=fast example-cnn    # 写出 examples/gesture_cnn/modeldata/cnn1d_packed.bin + cnn1d_valset.bin
make cnn-c                              # C 部署校验（或：make deploy-check）
```

## 样例 3：`liquid_sine`，神经回路策略（CfC/LTC）

把 [ncps](https://github.com/mlech26l/ncps) 栈（Lechner 与 Hasani 的
Neural Circuit Policies）移植到框架上。这类连续时间 RNN 的稀疏布线
（wiring）由感觉、中间、指令与运动神经元组成回路。从参考实现移植的
内容：

- `wirings.NCP` / `AutoNCP`：随机三层回路构造器（感觉扇出、中间扇出、
  指令层循环突触、运动扇入、±1 极性、可达性修复），以及
  `wirings.FullyConnected`
- 三种模式（default / pure / no_gate）的 `torch.CfCCell`，含逐神经元
  时间尺度，以及 `WiredCfCCell` 层叠：ff1/ff2 投影按布线邻接关系掩码，
  时间门投影保持稠密，与 ncps 一致
- `torch.LTCCell`：原始液态时间常数 ODE，每步用 6 次欧拉不动点迭代求解。
  BPTT 在全部迭代步上展开链式法则，另有有限差分校验。
- `torch_cfc_sinusoidal.py` 任务（sin/cos 输入、倍频正弦输出、MSE），
  推广成 1,600 条随机相位、频率抖动的序列

任务设定是 48 步序列，输入 [0, 3·pi·k] 上的 [sin, cos]，目标是同一波形
的倍频正弦。有两处设计让布线在量化后仍然有效：稀疏性直接编码在 trit
取值里（被掩码的突触潜值为零，trit 也就是零，打包产物不需要额外的掩码
文件），absmean 缩放只对已布线的突触计算，未布线突触的零值不参与。
一次运行训练四个模型，对应 ncps 的 sinusoidal 示例：

| 模型 | 参数 | fp32 验证 MSE | ptq 验证 | qat 验证 | qat 恢复 | 打包 |
| --- | --- | --- | --- | --- | --- | --- |
| CfC(2→32, backbone 64, proj 1) | 14,689 | 0.000520 | 0.448876 | 0.007895 | 98.4% | 3,779 B（压缩 15.5 倍）|
| WiredCfC + FullyConnected(8, out 1) | 352 | 0.002416 | 0.354496 | 0.029022 | 92.4% | 280 B（压缩 5.0 倍）|
| WiredCfC + NCP(16 inter, 8 cmd, 1 motor) | 2,056 | 0.001180 | 0.514103 | 0.019580 | 96.4% | 964 B（压缩 8.5 倍）|
| LTC + NCP(16 inter, 8 cmd, 1 motor) | 2,781 | 0.000815 | 0.544995 | 0.001454 | 99.9% | 8,623 B |

Ryzen 9 7940H 实测（1,600 条序列，80/20 划分，共约 4.5 分钟）。四个模型里
LTC 受三元化影响最大：朴素 PTQ 把它的 MSE 推到 0.545，连续时间动力学对
三元化很敏感；QAT 补回 99.9% 的差距，最终 0.001454，约为同规模布线 CfC 的
1/13，一次 48 步推理在 ODE 内核上 2.8 ms 完成。（LTC 的压缩比只有 1.3，
因为大部分参数是 fp32 的 ODE 物理参数，只有 675 个突触强度是三元。）
所有打包产物都无损往返；在与 ncps 示例完全相同的输入序列上，部署权重下
的 QAT CfC 以 0.000603 MSE 跟踪倍频正弦。

与参考实现的差异记录在 `examples/liquid_sine/liquid_sine.zig`：布线 RNG
用框架的 xorshift、按布线实例播种（构造算法相同，具体突触选择不同），
权重用 Xavier-Gauss 初始化，未布线骨干是 64 单元（参考实现为 128），
mixed_memory（fp32 LSTM 增强）没有移植。

```bash
zig build --release=fast example-ncps   # 写出 examples/liquid_sine/modeldata/ncps_{cfc,fc,ncp,ltc}_packed.bin + ncps_valset.bin
make ncps-check                         # 对全部四个产物做 C 部署校验
```

## 样例 4：`timestral_sketch`，部署 Mistral 的 Timestral-5M 再量化它

这个样例部署的模型来自别处：Mistral AI 的时间序列基础模型
[Timestral-5M](https://huggingface.co/mistralai/Timestral-5M)（520 万参数、
8 层 patch Transformer、每步 21 个分位数）。样例把它放到
[sketch-a-series](https://huggingface.co/spaces/townbox/sketch-a-series)
的交互背后：用鼠标画一条曲线，模型把它延续成一条分位数扇形带。然后用
本仓库的技术栈（absmean 三元量化、每字节 5 个 trit、打包产物、Zig 推理
引擎）量化同一个模型，并让同一个网页同时请求 fp32 与三元两个后端，把
1.77 bit/权重的代价画在同一张图上。

Zig 引擎是完整的前向移植：逐 patch 因果归一化、`[Enc ; Dec]` 注意力加
RoPE 与 sigmoid 门、SwiGLU、单调分位数头、advancing refine 帧、clampsym。
它在 fp32 权重上与 PyTorch 的相对误差为 2.4·10⁻⁶，所以后文的差距来自
量化，不是移植 bug。

```
fp32（safetensors，PyTorch）       20,843,960 B
三元产物（Zig，1.77 bit/权重）     1,152,070 B   （小 18.1 倍）

24 条留出笔画，256 步上下文，96 步预测
  变体          pinball   MAE     MASE    cov90   单次请求
  fp32          0.0711   0.217   5.89    0.961    31-44 ms
  PTQ 三元      0.2023   0.522  17.55    0.777       -
  QAT 三元      0.1071   0.293   8.91    0.901    16-22 ms
```

朴素训练后量化让误差涨到三倍。2500 步量化感知微调（4 线程 CPU 约 9.5
分钟，`examples/timestral_sketch/python/timestral_qat.py`）补回其中 73% 的
差距。保得最好的是速写本来的样子：趋势与随机游走离 fp32 不到 10%，需要
保住振幅的笔画（手绘曲线、正弦）则明显被压平，误差是 2.2 到 2.8 倍。
网页把两条中位数曲线叠在一起画，代价直接可见。

```bash
make fetch-timestral                    # 下载官方权重（21 MB）
make timestral-qat quantize-timestral   # 微调 → 量化 + 分位数宽度标定
make timestral-check                    # 与 PyTorch 在同样笔画上的数值校验
make sketch                             # 打开 http://127.0.0.1:8000
```

网页（`examples/timestral_sketch/web/sketch-a-series.html`）、服务端
（`examples/timestral_sketch/deploy/app.py`）与完整记录（打包格式、哪些
地方会坏、为什么去掉梯度掩码、分位数宽度标定）见
[docs/TIMESTRAL.md](docs/TIMESTRAL.md) 与
[examples/timestral_sketch/deploy/README.md](examples/timestral_sketch/deploy/README.md)。

## C 部署引擎

`examples/*/c/` 里是相互独立的 C 部署引擎。它们只加载打包模型和导出的
验证集，解码 trit 后按加、跳过、减的方式运行模型，不用逐权重的浮点表，
再重新算训练器的验证指标做比对：

```bash
make cnn-c         # cnn1d-c：CNN 的准确率 + 混淆矩阵一致性
make ncps-check    # ncps-c：全部四个 CfC/LTC 产物的 val-MSE 一致性
make deploy-check  # 两者均执行
```

每个引擎的输出必须与训练器一致，不一致时构建目标失败。

## fp32 基线

`examples/gpt_names/fp32.zig`（以及对应的 C 实现
`examples/gpt_names/c/microgpt.c`）是 fp32 预训练路径的对照参考：最初的
microGPT，只有 fp32，在同一份人名数据上训练。训练与推理用两条独立的
前向路径。`gpt_forward` 为反向传播保存激活，`gpt_forward_infer` 是专用的
单 token 路径，logits 与之相差不超过 fp32 舍入误差。
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) 介绍了这条路径的原理与瓶颈。

| 机器 | 后端 | tok/sec | 条件 |
| --- | --- | --- | --- |
| AMD Ryzen 9 7940H | Zig fp32，AVX2 | 8,441,952 | 5 次中位数，高性能电源方案，2026-10-30 |
| AMD Ryzen 9 7940H | C fp32，AVX2 | 8,217,680 | 同一轮、同一编译参数 |

同样两个二进制在本机「平衡」电源方案下只有约 400 万 tok/sec，仅这一项
设置就差约 2 倍，比仓库里任何代码改动的幅度都大。不带条件说明的吞吐
数字不可比，所以条件写进了表里。

## 旧版 PyTorch 路径

`examples/gpt_names/python/train_qat.py` 是最初的 PyTorch QAT 训练器，
`examples/gpt_names/python/export.py` 是它的 C 头文件导出器（生成
`examples/gpt_names/c/model.h`，由 `examples/gpt_names/c/inference.c` 使用）。
Zig 框架已经取代这两个脚本，留下它们是因为检查点产物
（`examples/gpt_names/modeldata/bitnet_gpt.pth`、
`examples/gpt_names/modeldata/bitnet_gpt_weights.npz`）记录了框架所沿用的
量化语义。启发本方案的两个项目
[BitNetMCU](https://github.com/cpldcpu/BitNetMCU) 与
[microGPT-C](https://github.com/vixhal-baraiya/microgpt-c) 的署名见
[THIRD_PARTY.md](THIRD_PARTY.md)，那里也说明了为什么两个都不再内嵌。

## 目录结构

```
src/framework.zig    框架入口：聚合六个核心模块
  src/framework/       rng / ternary / quant / adam / packed_model / timing
examples/            每个模型一个目录，代码、数据、产物都在自己的目录里
  gpt_names/           gpt_names.zig、fp32.zig（基线）、data/、c/、python/、modeldata/、web/
  gesture_cnn/         gesture_cnn.zig、c/、modeldata/、web/
  liquid_sine/         liquid_sine.zig、c/、modeldata/
  timestral_sketch/    timestral_sketch.zig、timestral/（移植模块）、deploy/、python/、
                       reference/、modeldata/、web/
docs/                格式规范、性能说明、Timestral 专题
```

每个样例把产物写进自己的 `modeldata/`，样例之间共享的只有
`src/framework/`。
