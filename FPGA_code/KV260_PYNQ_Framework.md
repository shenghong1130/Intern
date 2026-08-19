# KV260 Minimal PYNQ 平台框架设计

本文档用于记录计划设计的 KV260 软件平台框架。基础概念与各组件的工作原理记录在 `KV260_PYNQ_Architecture_Notes.md` 中。

## 1. 设计目标

保留已经安装好的 KV260 Ubuntu Server 24.04 及 AMD 原生底层组件，再安装满足 Python 控制自定义硬件所需的 Minimal PYNQ Runtime。

目标是让 Python 能够加载自己的 FPGA Design，并通过 PYNQ 控制其中的 FPGA Accelerator、DMA 和其他 IP。

## 2. 整体平台框架

```text
已经安装好的 KV260 Ubuntu Server 24.04
        |
        +-- 保留 Ubuntu / AMD 原生组件
        |     Kernel
        |     FPGA Manager
        |     Device Tree
        |     XRT / ZOCL
        |     CMA / Memory
        |
        +-- 安装 Minimal PYNQ Runtime
        |     PYNQ
        |     Overlay
        |     MMIO
        |     allocate
        |     DMA
        |
        +-- 加载自己的设计文件
        |     design.bit
        |     design.hwh
        |
        `-- Python 控制 FPGA Accelerator
```

## 3. 保留的底层平台组件

计划保留 Ubuntu Server 24.04 与 AMD 原生平台提供的底层能力，包括：

- Kernel
- FPGA Manager
- Device Tree
- XRT / ZOCL
- CMA / Memory

这部分作为操作系统、FPGA 配置、设备描述、运行时和内存管理的基础。

## 4. Minimal PYNQ Runtime 范围

需要的是最小 PYNQ Runtime，能够正常使用：

```python
from pynq import Overlay
from pynq import MMIO
from pynq import allocate
```

并能够操作自定义 Overlay 中的 DMA/IP。

## 5. 自定义 FPGA Design

平台需要能够加载由自己的 Vivado Hardware Design 生成的一组匹配文件：

```text
design.bit
design.hwh
```

其中，`design.bit` 用于配置 FPGA，`design.hwh` 用于让 PYNQ 识别硬件设计中的 IP、地址和连接信息。

## 6. 不属于当前目标的组件

当前目标不是搭建完整的 Kria-PYNQ Notebook 环境，因此不要求：

- Jupyter
- JupyterLab
- Notebook
- DPU-PYNQ examples
- 官方 Base Overlay
- Composable Pipeline
- 各种 Demo

## 7. 预期控制路径

```text
Python
   |
   v
PYNQ Overlay
   |
   +-- MMIO 控制寄存器
   |
   +-- allocate 准备数据缓冲区
   |
   +-- DMA 搬运输入和输出数据
   |
   `-- FPGA Accelerator 执行计算
```

本文只描述目标架构，不假设这些组件的具体版本、安装方法或当前运行状态。
