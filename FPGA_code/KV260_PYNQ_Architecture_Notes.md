# KV260 + Ubuntu 24.04 + Minimal PYNQ Runtime 架构笔记

本文从初学者视角说明 KV260、Zynq、Ubuntu、PYNQ、自定义 FPGA Overlay 以及数据传输组件之间的关系。重点是分清“谁运行软件、谁执行硬件计算、谁负责控制、谁负责搬运数据”。

## 1. KV260 平台概述

KV260 的全称是 **Kria KV260 Vision AI Starter Kit**。它是一套开发套件，使用 **Kria K26 System-on-Module（SOM，系统级模块）**。K26 的核心器件属于 **Zynq UltraScale+ MPSoC（Multiprocessor System-on-Chip，多处理器片上系统）** 系列。

Zynq UltraScale+ MPSoC 同时包含：

- **PS（Processing System，处理系统）**：包含 ARM 处理器、内存控制器和常用外设等。
- **PL（Programmable Logic，可编程逻辑）**：可以被配置成自定义数字硬件，也就是通常所说的 FPGA 部分。

最重要的关系是：

```text
Zynq UltraScale+ MPSoC
=
PS + PL
=
ARM 处理系统 + FPGA 可编程逻辑
```

PL 是 Zynq MPSoC 的一部分，不能把 PL 本身等同于整个 Zynq MPSoC。

```text
KV260
  |
  `-- Kria K26 SOM
        |
        `-- Zynq UltraScale+ MPSoC
              |
              +-------------------+
              |                   |
             PS                  PL
      Processing System   Programmable Logic
              |                   |
          ARM CPU               FPGA
```

## 2. PS 与 PL 的组成和职责

### PS：运行软件的处理系统

PS 是 **Processing System**。其中包括 ARM Cortex-A53 等处理器，以及内存控制器和其他固定功能模块。在本文讨论的系统里，软件栈运行在 PS 上：

```text
ARM CPU
   -> Ubuntu Server 24.04
   -> Python
   -> PYNQ
```

PS 可以访问 DDR 内存。Python 程序、PYNQ Runtime 和操作系统都使用这套处理系统。

### PL：实现自定义硬件的可编程逻辑

PL 是 **Programmable Logic**，即 FPGA 部分。根据硬件设计，它可以实现：

- 自定义 FPGA Accelerator（FPGA 加速器）
- CNN Accelerator（卷积神经网络加速器）
- HLS IP（High-Level Synthesis IP，高层次综合 IP）
- AXI DMA
- AXI GPIO
- BRAM（Block RAM，块 RAM）
- DSP（Digital Signal Processing，数字信号处理）单元
- 自定义 Verilog/VHDL IP

```text
             PS                                  PL
  +----------------------+          +--------------------------+
  | ARM Cortex-A53       |          | FPGA                     |
  |                      |          |                          |
  | Ubuntu / Linux       |<-- AXI -->| 自定义硬件 IP            |
  | Python               |          | AXI DMA                 |
  | PYNQ                 |          | HLS IP                  |
  +----------------------+          +--------------------------+
             |
             v
            DDR
```

**AXI（Advanced eXtensible Interface）** 是 PS 与 PL 之间非常重要的一组片上通信接口。控制寄存器访问、内存映射通信和流式数据传输都可能通过不同类型的 AXI 接口完成。

## 3. Ubuntu 在系统中的位置

Ubuntu Server 24.04 是运行在 PS 的 ARM CPU 上的 Linux 操作系统。它管理 CPU、内存、设备驱动、进程和文件系统，并为 Python、PYNQ 等软件提供运行环境。

普通电脑可类比为：

```text
Intel / AMD CPU
      -> Windows / Linux
      -> Python / 软件
```

KV260 上本文关注的软件路径是：

```text
ARM CPU
   -> Ubuntu Server 24.04
   -> Python
   -> PYNQ
```

**Ubuntu 不是 FPGA。Ubuntu 运行在 ARM CPU 上；FPGA 是 PL 中的可编程硬件。**

## 4. Kria 结构

这些名称处在不同层次：

```text
AMD / Xilinx
      |
      +-- Zynq UltraScale+ MPSoC
      |
      `-- Kria
            |
            `-- K26 SOM
                  |
                  `-- KV260 Starter Kit
```

- **Zynq UltraScale+ MPSoC**：把处理系统和可编程逻辑集成在同一器件中的架构，即本笔记所关心的 ARM CPU + FPGA。
- **Kria K26**：基于 Zynq UltraScale+ MPSoC 构建的 SOM，除核心器件外还集成了内存等模块级资源。
- **KV260**：使用 K26 SOM 的开发套件，并提供载板、接口和开发环境。

**Kria 不是 PYNQ。** Kria 是 AMD/Xilinx 的硬件平台系列；PYNQ 是运行在软件侧、帮助 Python 控制 FPGA 的框架。

## 5. PYNQ 软件框架

PYNQ 是运行在 Linux/Python 环境中的软件框架。它让 Python 程序更方便地完成以下工作：

- 配置 FPGA
- 控制 FPGA 中的 IP
- 访问硬件寄存器
- 使用 DMA（Direct Memory Access，直接内存访问）
- 分配适合设备/DMA 访问的数据缓冲区
- 管理 Overlay

```text
ARM CPU
   -> Ubuntu
   -> Python
   -> PYNQ
   -> 控制 PL / FPGA
```

必须分清“控制”和“计算”：

```text
PYNQ != FPGA 计算

PYNQ = 控制 FPGA

FPGA = 真正执行硬件计算
```

以 CNN 推理为例：

```text
Python / PYNQ
       |
       | 准备输入数据
       v
      DMA
       |
       v
CNN FPGA Accelerator
       | 真正执行卷积、MAC、矩阵运算
       v
      DMA
       |
       v
Python 获取结果
```

PYNQ 的价值之一，是把许多底层细节封装为较易用的 Python 接口。否则，应用程序往往需要自己处理 `/dev/mem`、`mmap`、物理地址、DMA 驱动、缓存同步、`ioctl` 以及其他 Linux FPGA 操作。PYNQ 并不会消除所有硬件约束，但能显著减少应用层的重复工作。

## 6. CNN 模型到 FPGA 的大致流程

训练完成的 CNN 通常不能简单理解为“直接放进 Vivado”。从软件模型到可在 FPGA 上运行的加速器，中间通常还需要模型转换、量化、编译或硬件实现。大致流程如下：

```text
训练 CNN
PyTorch / TensorFlow
        -> 训练后的模型
           .pt / .onnx / 其他格式
        -> 量化 / 编译 / HLS / FPGA 硬件化
        -> FPGA Accelerator
        -> Vivado Block Design
        -> Synthesis
        -> Implementation
        -> Generate Bitstream
        -> .bit + .hwh
```

具体工具链取决于模型和加速器方案，上图只描述概念流程。

`.bit` 和 `.hwh` 并不是 CNN 专属文件。任何相应的 Vivado FPGA 设计都可能生成这些文件，例如：

```text
FFT
 -> Vivado
 -> .bit + .hwh
```

或者：

```text
CNN Accelerator
 -> Vivado
 -> .bit + .hwh
```

## 7. `.bit` FPGA 位流文件

`.bit` 是 **FPGA Bitstream（FPGA 位流）**。它是真正用于配置 PL 的文件。

```text
Vivado Hardware Design
        -> Synthesis
        -> Implementation
        -> Generate Bitstream
        -> design.bit
        -> 配置 FPGA
```

`.bit` 决定设计中的 LUT、Flip-Flop、DSP、BRAM、Routing、IP 和 AXI 连接等资源如何被配置。容易记忆的说法是：

```text
.bit
=
给 FPGA 看
=
告诉 FPGA“你要变成什么硬件”
```

可以把 `.bit` 粗略类比为 CPU 软件中的 executable（可执行文件），因为两者都是构建流程的可运行产物。但技术上二者不同：FPGA bitstream 用于配置硬件资源，不是由普通 CPU 逐条执行的程序指令。

## 8. `.hwh` 硬件描述元数据

`.hwh` 是 Vivado 导出的 **Hardware Handoff（硬件交接）/ Hardware Description Metadata（硬件描述元数据）**。它不负责真正配置 FPGA，而是向 PYNQ 等软件描述当前硬件设计。

其中可能包含：

- IP 名称
- IP Base Address（基地址）
- Address Range（地址范围）
- Register（寄存器）信息
- AXI Interface
- Interrupt（中断）
- GPIO
- DMA
- Clock（时钟）
- 硬件连接关系

例如：

```text
axi_dma_0
Base Address: 0xA0000000

cnn_accel_0
Base Address: 0xA0010000
```

以上地址仅用于解释格式，不代表当前项目中的真实地址。容易记忆的说法是：

```text
.hwh
=
给 PYNQ 看
=
告诉 PYNQ“FPGA 里面有什么”
```

`.bit` 和 `.hwh` 最好来自同一次 Vivado Hardware Design 构建。不要混用：

```text
新版 design.bit
+
旧版 design.hwh
```

否则，IP 名称、地址或硬件结构可能不一致，软件可能访问错误的硬件资源。

## 9. `.bit` 和 `.hwh` 的关系

| 文件 | 主要使用者 | 作用 |
| --- | --- | --- |
| `.bit` | FPGA | 配置 FPGA 逻辑 |
| `.hwh` | PYNQ / 软件 | 描述 FPGA 中有哪些 IP、地址和结构 |

```text
.bit = FPGA 本身的硬件配置
.hwh = 这套硬件设计的软件说明书
```

两者描述的是同一套硬件设计的不同侧面：一个让 PL 形成该硬件，一个让软件理解该硬件。

## 10. PYNQ Overlay

PYNQ 的 **Overlay** 是 Python 对一套 FPGA 硬件设计的封装和管理入口。例如：

```python
from pynq import Overlay

overlay = Overlay("design.bit")
```

通常将同名的 `design.bit` 与 `design.hwh` 放在一起。Overlay 主要完成两类工作：

```text
design.bit
     |
     `-- 配置 FPGA

design.hwh
     |
     `-- 解析 FPGA Hardware Design
```

解析元数据后，PYNQ 可以知道设计中存在的 IP，例如：

```text
axi_dma_0
cnn_accel_0
axi_gpio_0
...
```

因此，根据设计中的实际命名和 PYNQ 驱动绑定情况，Python 可能可以直接取得相应对象：

```python
dma = overlay.axi_dma_0
cnn = overlay.cnn_accel_0
```

一句话理解：

```text
Overlay
=
FPGA 硬件设计在 Python 世界中的对象
```

也可以用房子类比：

```text
.bit
=
真正把房子建成某个结构

.hwh
=
房子的平面图

Overlay
=
读取平面图并管理整个房子的管理员
```

## 11. MMIO 寄存器控制

MMIO 是 **Memory-Mapped I/O（内存映射输入/输出）**。它把硬件 IP 的控制寄存器映射到处理器可访问的地址空间，使 CPU/Python 能以读写地址的方式控制 FPGA IP。

MMIO 常用于：

- START、STOP
- 状态查询
- 参数设置
- 地址设置
- `width`、`height`、`threshold`、`mode` 等配置

例如：

```python
ip.write(0x10, 640)
ip.write(0x18, 480)
ip.write(0x00, 1)
```

如果这些偏移量在该 IP 的寄存器映射中分别代表宽度、高度和启动位，就可以概念性地理解为：

```text
width = 640
height = 480
START = 1
```

实际偏移量和位定义必须以当前 IP 的寄存器文档或 `.hwh` 信息为准。

```text
MMIO
=
少量数据
+
控制命令
+
状态读取
```

最简单地说：

```text
MMIO = 控制 FPGA
```

## 12. `allocate()` 缓冲区分配

PYNQ 提供：

```python
from pynq import allocate
```

它主要用于准备 FPGA 或 DMA 能够访问的数据 Buffer（缓冲区）。例如：

```python
input_buffer = allocate(
    shape=(1024,),
    dtype="uint32"
)
```

普通 NumPy Array 通常只是面向 CPU 程序的数组：

```text
Python
 -> Virtual Memory
 -> 普通 Linux 内存
```

DMA 则需要能够按平台规则正确定位和访问底层内存。PYNQ `allocate()` 创建的缓冲区带有设备访问所需的信息和能力，可用于 CPU 与 DMA/FPGA 交换数据；具体平台还可能需要按 PYNQ API 规则进行缓存同步。

```text
allocate
=
为 DMA / FPGA 准备数据内存
```

## 13. DMA 数据传输

DMA 是 **Direct Memory Access（直接内存访问）**，主要用于高速搬运大量数据：

```text
DDR
 |
 v
DMA
 |
 v
FPGA Accelerator
 |
 v
DMA
 |
 v
DDR
```

DMA 的主要任务不是计算，而是数据搬运；真正的计算仍由 FPGA Accelerator 完成。

例如，CNN 输入图片可能包含大量像素。如果通过 MMIO 一个像素一个像素写入，效率通常很低。因此应区分：

```text
MMIO
=
控制 / 少量数据

DMA
=
大量数据传输
```

## 14. MMIO、`allocate()`、DMA、FPGA 的关系

```text
MMIO
=
控制 FPGA

allocate
=
准备 DMA 可以访问的内存

DMA
=
高速搬数据

FPGA
=
真正执行计算
```

一个简化的完整推理路径是：

```text
Python
   |
   | allocate input_buffer
   v
  DDR
   |
   v
  DMA
   |
   v
CNN FPGA Accelerator
   |
   v
  DMA
   |
   v
output_buffer
   |
   v
Python 获得结果
```

在这条路径旁边，Python/PYNQ 通常还会通过 MMIO 配置参数、启动加速器并读取状态。也就是说，MMIO 形成“控制路径”，DMA 形成“大数据路径”。

## 15. 最终完整架构图

```text
                               KV260
                                 |
                    Kria K26 / Zynq MPSoC
                                 |
             +-------------------+-------------------+
             |                                       |
             PS                                      PL
       Processing System                      Programmable Logic
             |                                       |
          ARM CPU                                  FPGA
             |                               (.bit 配置 FPGA)
      Ubuntu Server 24.04                            |
             |                                       |
          Python                                     |
             |                                       |
           PYNQ                                      |
             |                                       |
          Overlay -----------------------------------+
             |
        读取 .hwh
        识别 FPGA 中的 IP
             |
       +-----+----------+
       |                |
     MMIO           allocate
       |                |
   控制寄存器        DMA Buffer
       |                |
       |               DDR
       |                |
       |               DMA -----------+
       |                               |
       +------------------------------>|
                              FPGA Accelerator
                                       |
                                      计算
```

这张图可以按四条线理解：

1. Ubuntu、Python 和 PYNQ 运行在 PS 的 ARM CPU 上。
2. `.bit` 配置 PL，使它成为特定的硬件设计。
3. Overlay 读取 `.hwh`，让 Python 理解并访问设计中的 IP。
4. MMIO 负责控制，`allocate()` 准备缓冲区，DMA 搬运数据，FPGA Accelerator 执行计算。

## 16. 最终速记表

| 名称 | 是什么 | 最简单理解 |
| --- | --- | --- |
| KV260 | 开发板/开发套件 | 整个平台 |
| Kria K26 | SOM | KV260 的核心计算模块 |
| Zynq UltraScale+ MPSoC | SoC | ARM 处理系统 + FPGA 可编程逻辑 |
| PS | Processing System | ARM CPU 所在部分 |
| PL | Programmable Logic | FPGA 部分 |
| Ubuntu | 操作系统 | 运行在 ARM CPU 上 |
| Python | 编程环境 | 运行控制程序 |
| PYNQ | Python FPGA Framework | 让 Python 容易控制 FPGA |
| Vivado | FPGA 开发工具 | 把硬件设计实现到 FPGA |
| `.bit` | Bitstream | 告诉 FPGA 变成什么硬件 |
| `.hwh` | Hardware Metadata | 告诉 PYNQ FPGA 里面有什么 |
| Overlay | PYNQ 对 FPGA Design 的封装 | 加载 bit + 解析 hwh |
| MMIO | Memory-Mapped I/O | 控制 FPGA 寄存器 |
| `allocate` | Buffer Allocation | 准备 DMA/FPGA 可访问内存 |
| DMA | Direct Memory Access | 快速搬运大量数据 |
| FPGA Accelerator | 硬件计算模块 | 真正执行 CNN、FFT 等计算 |

核心总结：

```text
Ubuntu
=
PS 中 ARM CPU 的操作系统

PYNQ
=
让 Python 方便控制 FPGA 的软件框架

.bit
=
给 FPGA 看，决定 FPGA 变成什么硬件

.hwh
=
给 PYNQ 看，告诉它 FPGA 里面有什么

Overlay
=
把 .bit + .hwh 封装成 Python 可以方便操作的 FPGA Design

MMIO
=
控制

allocate
=
准备内存

DMA
=
搬数据

FPGA
=
做计算
```
