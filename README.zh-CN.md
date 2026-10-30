# TritNet-MCU（tritnetmcu）

[![ci](https://github.com/weifeng233/tritnet-mcu/actions/workflows/ci.yml/badge.svg)](https://github.com/weifeng233/tritnet-mcu/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

简体中文 | [English](README.md)

一个使用 Zig 实现的训练、量化、部署框架，面向三元权重 {-1, 0, +1} 的 MCU
模型。除 libc 外无外部依赖。一条命令完成 fp32 预训练、BitNet 风格量化
感知训练（QAT）、留出集验证、三元打包导出与无损往返校验。样例在笔记本上
的运行时间为数秒。

框架代码位于 `src/framework.zig`（RNG、三元编解码、absmean QAT 量化器、
Adam 优化器与余弦学习率调度、打包模型容器、计时）。`examples/` 下为四个
模型样例。`examples/*/c/` 下的 C 引擎负责部署验证：根据打包产物重新计算
训练阶段的全部指标。

## 流程

每个样例在同一进程内依次执行五个阶段：

1. fp32 预训练：Adam 优化器、余弦学习率调度、梯度裁剪；模型支持时使用
   `@Vector`（AVX2/NEON）内核。
2. BitNet 三元 QAT：每个线性投影按 absmean 缩放进行三元化
   （`scale = mean(|W|)`，绝对值大于 `scale/2` 的权重取 `+1` 或 `-1`，其余取
   `0`），使用直通估计器（STE）训练。梯度按 `|W| < scale` 掩码，激活值按
   token 量化为 8 位，嵌入与偏置保持 fp32。
3. 验证：在留出集上比较 fp32、朴素 PTQ 与 QAT，统计 trit 分布、延迟与
   存储占用。
4. 三元打包：每字节 5 个 trit
   （[compilade 的方案](https://compilade.net/blog/ternary-packing)，
   llama.cpp 的 BitNet 权重使用同一方案）。三进制编码加向上取整重标定，
   解码使用乘法与移位，不含除法。
5. 往返校验：重新加载打包文件、解码并评估，指标与导出前一致。

### 框架模块

| 模块 | 内容 |
| --- | --- |
| `framework/rng.zig` | 确定性 xorshift 流：uniform / gauss / range |
| `framework/ternary.zig` | 每字节 5 trit 的权重编解码（pack5 / unpack5） |
| `framework/quant.zig` | absmean 量化、STE 梯度掩码、8 位激活量化、wired-only 缩放变体 |
| `framework/adam.zig` | 8 宽 SIMD Adam、余弦学习率调度、偏置修正 |
| `framework/packed_model.zig` | 小端模型文件写入器/读取器、共享的三元矩阵记录 |
| `framework/timing.zig` | 可移植单调时钟 |

新增样例只需实现模型相关代码：前向与反向传播，以及基于上述原语的导出
布局。

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

依赖：Zig 0.16.0、`make`；Python 3.11+ 仅用于工具脚本
（`examples/*/python/`、`examples/timestral_sketch/deploy/`）。Zig 部分无外部依赖，
Python 依赖由 `uv` 锁定：

```bash
uv sync --extra eval                 # 作图与指标
uv sync --extra qat --extra eval     # Timestral 微调
uv sync --extra serve                # 绘图服务
```

## 文档

| 文档 | 内容 |
| --- | --- |
| [`docs/ARTIFACT-FORMAT.md`](docs/ARTIFACT-FORMAT.md) | 五种二进制格式的逐字段说明，以及三元编解码的黄金测试向量 |
| [`docs/TIMESTRAL.md`](docs/TIMESTRAL.md) | 1.6 bit/权重在 Timestral-5M 上的实测代价 |
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | 吞吐量与测量条件 |
| [`THIRD_PARTY.md`](THIRD_PARTY.md) | 上游来源与许可证 |
| [`CHANGELOG.md`](CHANGELOG.md) | 变更内容与数值变化 |

## 设计决策

以下五条设计选择：

训练与部署使用同一种语言，消除了「在我机器上成立、换一台机器不成立」
这一来源的差异。`zig build` 同时提供构建、测试与交叉编译，`@Vector` 由
一份源码生成 AVX2 或 NEON 指令。C 引擎的用途是与 Zig 实现交叉验证。

权重编解码每字节存储 5 个 trit。一个 trit 携带 log2(3) ≈ 1.585 bit，
`3^5 = 243 ≤ 256`，因此每字节可存 5 个 trit，即 1.6 bit/权重。解码为一次
乘法、一次移位与一次掩码，不含除法（目标 Cortex-M 核没有整数除法指令）。
采用 2 bit/权重时文件体积增加 25%，解码开销不变。

每个发布产物配一个 C 引擎。往返检查的读取与写入为同一份代码，共同的
格式误解不会被发现。C 引擎只读取打包文件，重新计算指标，并与训练器写入
产物 footer 的数值比较；不一致时 `make deploy-check` 失败。

量化标度采用 absmean：`scale = mean(|W|)` 加固定 0.5 阈值，使产物成为权重
的确定函数，一条命令即可生成，无需标定语料与优化器状态。absmean 精度
不足时，通过增加训练（QAT）改善。

fp32 后端保留。量化指标需要同进程的 fp32 基线才有意义。两个后端提供
同一接口，1.77 bit/权重的代价在同一张图上显示。

## 样例

| 样例 | 模型 | 参数量 |
| --- | --- | --- |
| [`gpt_names`](examples/gpt_names/README.md) | 字符级 micro-GPT（人名数据） | 4,192 |
| [`gesture_cnn`](examples/gesture_cnn/README.md) | TinyML 手势 1D-CNN | 2,524 |
| [`liquid_sine`](examples/liquid_sine/README.md) | CfC/LTC 神经回路策略 | 352 至 14,689 |
| [`timestral_sketch`](examples/timestral_sketch/README.md) | Timestral-5M 量化与部署 | 5,208,960 |

## C 部署引擎

`examples/*/c/` 为独立的 C 部署引擎。引擎读取打包模型与验证集，解码 trit
后按加、跳过、减的方式运行模型，不使用逐权重浮点表；随后重新计算训练器
的验证指标并比较：

```bash
make cnn-c         # cnn1d-c：CNN 的准确率 + 混淆矩阵一致性
make ncps-check    # ncps-c：全部四个 CfC/LTC 产物的 val-MSE 一致性
make deploy-check  # 两者均执行
```

指标不一致时构建目标失败。

## fp32 基线

`examples/gpt_names/fp32.zig`（C 实现：
`examples/gpt_names/c/microgpt.c`）是 fp32 预训练路径的对照实现：最初的
microGPT，仅 fp32，在同一份人名数据上训练。训练与推理使用两条独立的
前向路径。`gpt_forward` 为反向传播保存激活值；`gpt_forward_infer` 为单
token 推理路径，其 logits 与训练路径的差异在 fp32 舍入误差范围内。
[docs/PERFORMANCE.md](docs/PERFORMANCE.md) 说明该路径的原理与瓶颈。

| 机器 | 后端 | tok/sec | 条件 |
| --- | --- | --- | --- |
| AMD Ryzen 9 7940H | Zig fp32，AVX2 | 8,441,952 | 5 次中位数，高性能电源方案，2026-10-30 |
| AMD Ryzen 9 7940H | C fp32，AVX2 | 8,217,680 | 同一轮、同一编译参数 |

同样两个二进制在本机「平衡」电源方案下约为 400 万 tok/sec。仅电源方案
一项即造成约 2 倍差异，大于仓库内全部代码优化的幅度。未标注条件的吞吐
数字不可比较，表中包含该条件。

## 旧版 PyTorch 路径

`examples/gpt_names/python/train_qat.py` 是最初的 PyTorch QAT 训练器，
`examples/gpt_names/python/export.py` 是它的 C 头文件导出器（生成
`examples/gpt_names/c/model.h`，由 `examples/gpt_names/c/inference.c` 使用）。
Zig 框架已取代这两个脚本。保留原因是检查点产物
（`examples/gpt_names/modeldata/bitnet_gpt.pth`、
`examples/gpt_names/modeldata/bitnet_gpt_weights.npz`）记录了框架沿用的
量化语义。本方案参考的两个项目
[BitNetMCU](https://github.com/cpldcpu/BitNetMCU) 与
[microGPT-C](https://github.com/vixhal-baraiya/microgpt-c) 的署名见
[THIRD_PARTY.md](THIRD_PARTY.md)，该文件同时说明两个项目均未内嵌。

## 目录结构

```
src/framework.zig    框架入口：聚合六个核心模块
  src/framework/       rng / ternary / quant / adam / packed_model / timing
examples/            每个模型一个目录，代码、数据、产物均位于该目录
  gpt_names/           gpt_names.zig、fp32.zig（基线）、data/、c/、python/、modeldata/、web/
  gesture_cnn/         gesture_cnn.zig、c/、modeldata/、web/
  liquid_sine/         liquid_sine.zig、c/、modeldata/
  timestral_sketch/    timestral_sketch.zig、timestral/（移植模块）、deploy/、python/、
                       reference/、modeldata/、web/
docs/                格式规范、性能说明、Timestral 专题
```

每个样例将产物写入自身目录下的 `modeldata/`；样例之间共享的代码为
`src/framework/`。
