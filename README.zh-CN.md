# TritNet-MCU（tritnetmcu）

[![ci](https://github.com/weifeng233/tritnet-mcu/actions/workflows/ci.yml/badge.svg)](https://github.com/weifeng233/tritnet-mcu/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

简体中文 | [English](README.md)

一个纯 Zig 的**训练 → 量化 → 部署框架**，面向三元权重 {-1, 0, +1} 的
MCU 模型，附带四个完整可复现的样例。除 libc 外零依赖：一条命令完成
fp32 预训练、BitNet 风格量化感知训练（QAT）、留出集验证、三元打包
导出，以及验证打包产物无损的往返校验 —— 每个样例在笔记本电脑上
数秒内完成。

框架本体在 `src/framework.zig`（RNG、三元编解码、absmean QAT 量化器、
Adam + 余弦调度、打包模型容器、计时）。`examples/` 目录提供四个刻意
不同的模型家族样例；`c/` 中的 C 引擎用于部署验证，仅凭打包产物复现
训练器的全部指标。

## 流水线

每个样例在同一进程内依次完成相同的五个阶段：

1. **fp32 预训练** —— Adam + 余弦学习率调度、梯度裁剪、SIMD（经
   `@Vector` 的 AVX2/NEON）内核。
2. **BitNet 三元 QAT** —— 每个线性投影按 absmean 缩放三元化
   （`scale = mean(|W|)`，绝对值超过 `scale/2` 的权重取 `+1/-1`，其余为
   `0`），用直通估计器（STE）训练；梯度按 `|W| < scale` 掩码，激活按
   token 做 8 比特量化，嵌入/偏置保持 fp32。
3. **验证** —— 留出集上对比 fp32 / 朴素 PTQ / QAT，外加 trit 分布、
   延迟与存储核算。
4. **三元打包** —— 每字节 5 个 trit
   （[compilade 的方案](https://compilade.net/blog/ternary-packing)，即
   llama.cpp 为 BitNet 模型所用的方案）：三进制基数编码与向上取整
   重标定，解码使用乘法与移位实现，不包含除法运算。
5. **往返校验** —— 重新加载打包文件、解码、再评估；指标必须与导出前
   的模型逐位一致。

### 框架模块

| 模块 | 职责 |
| --- | --- |
| `framework/rng.zig` | 确定性 xorshift 流：uniform / gauss / range |
| `framework/ternary.zig` | 每字节 5 trit 的权重编解码（pack5 / unpack5） |
| `framework/quant.zig` | absmean 量化、STE 梯度掩码、8 比特激活量化、wired-only 缩放变体 |
| `framework/adam.zig` | 8 宽 SIMD Adam、余弦学习率调度、偏置修正 |
| `framework/packed_model.zig` | 小端模型文件写入器/读取器、共享的三元矩阵记录 |
| `framework/timing.zig` | 可移植单调时钟 |

新增一个样例只需要写模型侧代码：前向/反向传播，以及基于这些原语的
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

依赖：**Zig 0.16.0**、`make`，Python 3.11+ 仅用于工具链（`python/`、`deploy/`）。
Zig 部分零依赖。Python 依赖由 `uv` 锁定：

```bash
uv sync --extra eval                 # 作图与指标
uv sync --extra qat --extra eval     # Timestral 微调
uv sync --extra serve                # 绘图服务
```

## 文档

| 文档 | 回答什么问题 |
| --- | --- |
| [`docs/ARTIFACT-FORMAT.md`](docs/ARTIFACT-FORMAT.md) | 五种二进制格式逐字段说明，以及三元编解码的黄金测试向量 |
| [`docs/OWNERSHIP.md`](docs/OWNERSHIP.md) | 哪些是我写的、哪些是移植、哪些是内嵌 |
| [`docs/TIMESTRAL.md`](docs/TIMESTRAL.md) | 1.6 bit/权重在 Timestral-5M 上到底付出多少代价（实测） |
| [`docs/TIMESTRAL-TRADEOFFS.zh-CN.md`](docs/TIMESTRAL-TRADEOFFS.zh-CN.md) | 上述取舍的中文版本 |
| [`docs/adr/`](docs/adr/README.md) | 为什么用 Zig、为什么 5 trits、为什么双实现、为什么 absmean、为什么保留 fp32 |
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | 吞吐与其测量条件（[中文](docs/PERFORMANCE.zh-CN.md)） |
| [`THIRD_PARTY.md`](THIRD_PARTY.md) | 所有上游来源及其许可证 |
| [`CHANGELOG.md`](CHANGELOG.md) | 改了什么，哪些数字变了 |

## 样例 1 — `gpt_names`：字符级 micro-GPT

基于 microGPT-C 的微型字符级 Transformer（4,192 参数：1 层、4 头、
16 维嵌入、64 维 MLP、ReLU²、RMSNorm），在约 3.2 万人名数据上训练并
生成新人名。以三元权重训练并部署后，模型文件体积约为 fp32 权重的
1/22。

以下结果在 data/names.txt 上测得（80/20 训练/留出划分，20000 步 fp32
预训练 + 5000 步 QAT，Ryzen 9 7940H 上共约 1 秒）：

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
zig build --release=fast example-gpt    # 写出 modeldata/tritnetmcu_packed.bin
```

## 样例 2 — `gesture_cnn`：TinyML 传感器分类器

典型的 TinyML 1D-CNN 任务（EdgeImpulse / MLPerf-Tiny 风格）：基于
三轴加速度计窗口的 4 类手势识别 —— 静止（rest）/ 摇晃（shake）/
画圈（circle）/ 敲击（tap）。数据由固定种子确定性生成（随机相位、
幅度抖动与逐样本噪声），无需外部数据集即可复现。2,524 参数、每次
推理 54,208 次乘加：

```
input 3x128
Conv1D(3 -> 8, k=8)  + ReLU + MaxPool(2)   -> 8x60
Conv1D(8 -> 16, k=4) + ReLU + MaxPool(2)   -> 16x28
Flatten -> Dense(448 -> 4) + softmax
```

偏置保持 fp32（2,524 参数中的 28 个）；所有卷积核与全连接权重均
三元化。样例内的单元测试将卷积/池化内核与直接参考实现逐一对照，
并做端到端数值梯度检查。

Ryzen 9 7940H 上的实测（4000 样本，80/20 划分，2000 步 fp32 预训练 +
1500 步 QAT，共约 11 秒）：

```
eval (accuracy, float activations)
  fp32  train 100.00%   val 100.00%   (val loss 0.0014)
  ptq   val 65.75%   (ternary, no QAT)
  qat   train 100.00%   val 100.00%
  trits: -1 42.6%   0 21.2%   +1 36.2%

storage
  packed 5-trits/byte  :    672 bytes  (比 fp32 小 15.0x)

latency (host CPU, float activations)
  ternary weights      :  41.1 us/inference  (1320M MACs/s)

packed round-trip: OK (lossless)   qat 恢复 fp32->ptq 损失差距的 99.9%
```

朴素三元化使验证准确率降至 65.75%；QAT 将其恢复至 100%，此时每个
卷积核与全连接层均以每权重 1.6 比特的 {-1, 0, +1} trit 运行。

```bash
zig build --release=fast example-cnn    # 写出 modeldata/cnn1d_packed.bin + cnn1d_valset.bin
make cnn-c                              # C 部署校验（或：make deploy-check）
```

## 样例 3 — `liquid_sine`：神经回路策略（CfC/LTC）

将 [ncps](https://github.com/mlech26l/ncps) 栈（Lechner 与 Hasani 的
神经回路策略，Neural Circuit Policies）移植到框架上：连续时间 RNN，
其稀疏"布线"（wiring）是由感觉、中间、指令与运动神经元构成的回路。
从参考实现移植的内容：

- `wirings.NCP` / `AutoNCP` —— 随机三层回路构造器（感觉扇出、中间扇出、
  指令层循环突触、运动扇入、±1 极性、可达性修复），外加
  `wirings.FullyConnected`
- 三种模式（default / pure / no_gate）的 `torch.CfCCell`，含逐神经元
  时间尺度（timespans），以及 `WiredCfCCell` 层叠 —— 其 ff1/ff2 投影按
  布线邻接关系掩码，时间门投影保持稠密，与 ncps 完全一致
- `torch.LTCCell` —— 原始液态时间常数 ODE，每步以 6 次欧拉不动点迭代
  求解；BPTT 在全部迭代步上展开链式法则（有限差分校验通过）
- `torch_cfc_sinusoidal.py` 任务（sin/cos 输入、倍频正弦输出、MSE）
  推广为 1,600 条随机相位、频率抖动的序列数据集

任务设定：48 步序列，输入 [0, 3·pi·k] 上的 [sin, cos]，目标是同一波形
的倍频正弦。有两个设计使布线在量化后仍然有效：稀疏性直接编码于
trit 取值（被掩码的突触潜值为零，因而 trit 为零 —— 打包产物不需要
额外的掩码文件）；absmean 缩放只对已布线的突触计算，未布线突触的
零值不参与缩放因子计算。一次运行训练四个模型（对应 ncps 的
sinusoidal 示例）：

| 模型 | 参数 | fp32 验证 MSE | ptq 验证 | qat 验证 | qat 恢复 | 打包 |
| --- | --- | --- | --- | --- | --- | --- |
| CfC(2→32, backbone 64, proj 1) | 14,689 | 0.000520 | 0.448876 | 0.007895 | 98.4% | 3,779 B（压缩 15.5 倍）|
| WiredCfC + FullyConnected(8, out 1) | 352 | 0.002416 | 0.354496 | 0.029022 | 92.4% | 280 B（压缩 5.0 倍）|
| WiredCfC + NCP(16 inter, 8 cmd, 1 motor) | 2,056 | 0.001180 | 0.514103 | 0.019580 | 96.4% | 964 B（压缩 8.5 倍）|
| LTC + NCP(16 inter, 8 cmd, 1 motor) | 2,781 | 0.000815 | 0.544995 | 0.001454 | 99.9% | 8,623 B |

Ryzen 9 7940H 实测（1,600 条序列，80/20 划分，共约 4.5 分钟）。四个
模型中，LTC 受三元化影响最大：朴素 PTQ 使其 MSE 恶化至 0.545，
连续时间动力学对三元化敏感；QAT 恢复了 99.9% 的差距，最终 MSE 为
0.001454，约为同规模布线 CfC 的 1/13，同时以 ODE 内核在 2.8 ms 内
完成一次 48 步推理。（LTC 的打包压缩比仅为 1.3：其大部分参数为
fp32 的 ODE 物理参数，仅 675 个突触强度为三元。）所有打包产物均
无损往返；在与 ncps 示例完全相同的输入序列上，部署权重下的 QAT CfC
以 0.000603 MSE 跟踪倍频正弦。

与参考实现的差异（记录在 `examples/liquid_sine.zig`，均为有意设计）：
布线 RNG 采用框架的 xorshift、按布线实例播种（构造算法相同，具体
突触选择不同）；权重用 Xavier-Gauss 初始化；未布线骨干为 64 单元
（参考实现为 128）；mixed_memory（fp32 LSTM 增强）未移植。

```bash
zig build --release=fast example-ncps   # 写出 modeldata/ncps_{cfc,fc,ncp,ltc}_packed.bin + ncps_valset.bin
make ncps-check                         # 对全部四个产物做 C 部署校验
```

## 样例 4 — `timestral_sketch`：部署 Mistral Timestral-5M，再用本框架量化

这个样例部署的不是本仓库训练的模型，而是 Mistral AI 的时间序列基础模型
[Timestral-5M](https://huggingface.co/mistralai/Timestral-5M)（520 万参数、8 层
patch Transformer、每步 21 个分位数），并复刻
[sketch-a-series](https://huggingface.co/spaces/townbox/sketch-a-series) 的交互：
用鼠标画一条曲线，模型把它延续成一条分位数扇形带。随后用本仓库的技术栈
（absmean 三元量化、每字节 5 个 trit、打包产物、Zig 推理引擎）量化同一个模型，
并让**同一个网页**同时请求 fp32 与三元两个后端，把 1.77 bit/权重的代价画在同一张图上。

Zig 引擎是完整的前向移植（逐 patch 因果归一化、`[Enc ; Dec]` 注意力 + RoPE +
sigmoid 门、SwiGLU、单调分位数头、advancing refine 帧、clampsym），在 fp32 权重上
与 PyTorch 的相对误差为 2.4·10⁻⁶——所以两者的差异只来自量化。

```
fp32（safetensors，PyTorch）       20,843,960 B
三元产物（Zig，1.77 bit/权重）     1,152,070 B   （小 18.1 倍）

24 条留出笔画，256 步上下文，96 步预测
  变体          pinball   MAE     MASE    cov90   单次请求
  fp32          0.0711   0.217   5.89    0.961    31-44 ms
  PTQ 三元      0.2023   0.522  17.55    0.777       -
  QAT 三元      0.1071   0.293   8.91    0.901    16-22 ms
```

朴素训练后量化会让误差涨到三倍；2500 步量化感知微调（4 线程 CPU 约 9.5 分钟，
`python/timestral_qat.py`）把其中 73% 的差距补回来。趋势类和随机游走类笔画与 fp32
相差不超过 10%，而需要保住振幅的笔画（手绘曲线、正弦）明显被压平（误差 2.2–2.8 倍）——
网页把两条中位数曲线叠在一起画，这个代价是可见的，而不是被藏起来。

```bash
make fetch-timestral                    # 下载官方权重（21 MB）
make timestral-qat quantize-timestral   # 微调 → 量化 + 分位数宽度标定
make timestral-check                    # 与 PyTorch 在同样笔画上的数值校验
make sketch                             # 打开 http://127.0.0.1:8000
```

网页（`web/sketch-a-series.html`）、服务端（`deploy/app.py`）以及完整记录
（打包格式、哪些地方会坏、为什么去掉梯度掩码、分位数宽度标定）见
[docs/TIMESTRAL.md](docs/TIMESTRAL.md) 与 [deploy/README.md](deploy/README.md)。

## C 部署引擎

`c/` 是相互独立的 C 部署引擎 —— 引擎加载打包模型和导出的验证集，
解码 trit 后以加/跳过/减（add/skip/subtract）方式运行模型，不使用
逐权重的浮点表，然后重新计算训练器的验证指标并比对：

```bash
make cnn-c         # cnn1d-c：CNN 的准确率 + 混淆矩阵一致性
make ncps-check    # ncps-c：全部四个 CfC/LTC 产物的 val-MSE 一致性
make deploy-check  # 两者均执行
```

每个引擎的输出指标必须与训练器一致；不一致时构建目标失败。

## fp32 基线

`src/fp32.zig`（及对应的 C 实现 `c/microgpt.c`）是 fp32 预训练路径的
对照参考 —— 最初的 microGPT，仅 fp32，在同一份人名数据上训练。训练
与推理使用两条独立的前向路径 —— `gpt_forward` 为反向传播保存激活，
`gpt_forward_infer` 是专用的单 token 路径，logits 与之相差不超过 fp32
舍入误差。[docs/PERFORMANCE.zh-CN.md](docs/PERFORMANCE.zh-CN.md)
（英文版 [PERFORMANCE.md](docs/PERFORMANCE.md)）介绍了该路径的原理与
瓶颈。

| 机器 | 后端 | tok/sec |
| --- | --- | --- |
| AMD Ryzen 9 7940H | AVX2 | 5,491,565 |

## 旧版 PyTorch 路径

`python/train_qat.py` 是最初的 PyTorch QAT 训练器，`python/export.py`
是它的 C 头文件导出器（生成 `c/model.h`，由 `c/inference.c` 消费）。
Zig 框架已取代它们；保留这两个脚本是因为其检查点产物
（`modeldata/bitnet_gpt.pth`、`modeldata/bitnet_gpt_weights.npz`）记录
了框架所沿用的量化语义。启发本方案的两个项目
[BitNetMCU](https://github.com/cpldcpu/BitNetMCU) 与
[microGPT-C](https://github.com/vixhal-baraiya/microgpt-c) 的署名见
[THIRD_PARTY.md](THIRD_PARTY.md)；该文件也说明了它们为何都不再内嵌。

## 目录结构

```
src/framework.zig    框架入口：聚合六个核心模块
  src/framework/       rng / ternary / quant / adam / packed_model / timing
src/fp32.zig         fp32 参考基线（最初的 microGPT）
src/timestral.zig    样例 4：Timestral-5M 部署模块
  src/timestral/       model / engine / weights / artifact / eval / fixture
examples/            样例，每个样例一个二进制
  gpt_names.zig        字符级 micro-GPT（names.txt 人名）
  gesture_cnn.zig      TinyML 1D-CNN 手势分类器
  liquid_sine.zig      CfC/LTC 神经回路策略（倍频正弦）
  timestral_sketch.zig Timestral-5M：量化 / 评估 / 预测 / 服务
c/                   C 部署引擎（cnn1d、ncps）+ fp32 C 对照
deploy/              sketch-a-series 服务端（FastAPI）及其说明
python/              旧版 PyTorch QAT 路径，以及 Timestral 工具（QAT、数值校验、作图）
reference/           内嵌参考项目：timestral/（Apache-2.0）、sketch-a-series/
web/                 基准测试的单页可视化，以及 sketch-a-series 网页
data/ modeldata/ docs/
```
