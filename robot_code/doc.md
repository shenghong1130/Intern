# robot_code 两个文件夹及 TonyPi 原厂系统功能总结

## 1. 整体结构

```text
robot_code/
├── robotall/
└── robot_tonypi/
```

整个机器人系统可以简单理解为：

> **robotall = 项目自己的底层能力 / 基础功能封装**
> **robot_tonypi = 项目的上层应用 / 具体任务逻辑**
> **TonyPi 原厂系统 = 提供动作文件、动作执行接口和底层舵机控制接口**

---

# 2. robotall

## 作用

`robotall` 主要负责机器人的**底层控制和基础功能**。

可以理解为项目自己的“基础能力层”。

主要包括：

* 机器人基础动作相关功能
* 基础运动控制
* NFC / I2C 等底层通信
* 其他硬件相关操作
* 将底层操作封装成上层可以直接调用的函数

### 使用方式

上层程序一般不需要重新实现这些底层功能，而是调用 `robotall` 提供的接口。

```text
上层程序
   ↓
调用 robotall
   ↓
robotall 调用底层系统 / 硬件接口
   ↓
机器人执行操作
```

因此可以简单理解为：

> **robotall 解决的是“机器人具有什么基础能力”。**

---

# 3. robot_tonypi

## 作用

`robot_tonypi` 主要负责机器人的**上层程序和具体任务逻辑**。

它建立在底层功能之上，用于完成更加具体的机器人任务。

可能包括：

* 调用机器人基础动作
* 机器人任务流程
* AprilTag 定位
* 路径规划
* 导航
* 比赛逻辑
* 根据当前环境决定机器人下一步做什么

可以理解为机器人的“应用层”。

```text
robot_tonypi
      ↓
判断现在应该做什么
      ↓
调用底层功能
      ↓
机器人执行动作
```

因此：

> **robot_tonypi 解决的是“机器人什么时候做什么”。**

---

# 4. TonyPi 原厂系统

除了项目自己的 `robotall` 和 `robot_tonypi`，TonyPi 本身还提供了一套**原厂的软件和动作系统**。

其中比较重要的是：

```text
TonyPi 原厂系统
│
├── .d6a 动作文件
│
├── hiwonder.ActionGroupControl
│       │
│       └── AGC.runActionGroup(...)
│
└── Controller
        │
        └── set_pwm_servo_pulse(...)
```

---

# 5. `.d6a` 动作文件

`.d6a` 可以理解为 TonyPi 已经制作好的**动作数据 / 动作组**。

例如：

```text
stand.d6a
go_forward_fast.d6a
turn_right.d6a
...
```

这些文件保存了机器人执行某个完整动作时所需要的舵机运动数据。

例如：

```text
go_forward_fast.d6a
        ↓
记录多个舵机的运动
        ↓
按照一定的时间顺序执行
        ↓
机器人完成“快速前进”
```

因此：

> **`.d6a` 是“具体怎么动”的动作数据。**

---

# 6. AGC.runActionGroup()

TonyPi 提供：

```python
import hiwonder.ActionGroupControl as AGC
```

然后可以使用：

```python
AGC.runActionGroup(...)
```

执行 `.d6a` 动作组。

例如：

```python
AGC.runActionGroup('stand')
```

或者：

```python
AGC.runActionGroup('go_forward_fast', times=10)
```

可以理解为：

```text
AGC.runActionGroup('go_forward_fast')
              ↓
      找到对应动作组
              ↓
     go_forward_fast.d6a
              ↓
       读取动作数据
              ↓
       控制多个舵机
              ↓
          机器人运动
```

所以：

> **AGC 是“执行动作组”的接口，而 `.d6a` 是“动作组的数据”。**

---

# 7. Controller.set_pwm_servo_pulse()

除了执行完整的 `.d6a` 动作组之外，TonyPi 还可以通过 `Controller` **直接控制某一个舵机**。

例如：

```python
ctl.set_pwm_servo_pulse(2, 1200, 1000)
```

这种方式和 `.d6a` 动作组不同。

它不是：

```text
读取一个完整的动作文件
```

而是：

```text
直接指定某个舵机
        ↓
设置 PWM 参数
        ↓
让这个舵机运动
```

因此：

> **`Controller` 更适合直接控制单个舵机，而 `AGC.runActionGroup()` 更适合执行已经制作好的完整动作。**

例如：

```text
Controller
    ↓
2号舵机
    ↓
调整头部角度
```

这种操作就不需要制作一个 `.d6a` 动作组。

---

# 8. 两种机器人控制方式

TonyPi 原厂系统实际上可以看成两条控制路径：

```text
                    TonyPi 原厂系统
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ↓                           ↓
       动作组控制                    单舵机控制
             │                           │
             ↓                           ↓
   AGC.runActionGroup()             Controller
             │                           │
             ↓                           ↓
         .d6a 文件              set_pwm_servo_pulse(...)
             │                           │
             ↓                           ↓
     执行完整动作组                 直接控制舵机
             │                           │
             └─────────────┬─────────────┘
                           ↓
                        机器人
```

---

# 9. 整个项目的关系

综合起来，可以理解成：

```text
                         robot_tonypi
                    ┌──────────────────┐
                    │   上层任务逻辑     │
                    │                  │
                    │ AprilTag 定位     │
                    │ 路径规划          │
                    │ 导航              │
                    │ 比赛流程          │
                    └────────┬─────────┘
                             │
                             ↓
                    调用项目底层功能
                             │
                             ↓
                         robotall
                    ┌──────────────────┐
                    │   基础能力封装     │
                    │                  │
                    │ 基础动作          │
                    │ 硬件通信          │
                    │ NFC / I2C        │
                    │ 其他底层功能       │
                    └────────┬─────────┘
                             │
                 ┌───────────┴───────────┐
                 ↓                       ↓
          TonyPi 原厂接口             其他硬件接口
                 │
        ┌────────┴─────────┐
        ↓                  ↓
       AGC             Controller
        │                  │
        ↓                  ↓
     .d6a 文件        PWM 舵机控制
        │                  │
        └────────┬─────────┘
                 ↓
              机器人
```

---

# 10. 最简单的理解

| 部分                     | 主要作用            | 可以理解成          |
| ---------------------- | --------------- | -------------- |
| `.d6a`                 | 保存已经制作好的动作数据    | **动作本身**       |
| `AGC.runActionGroup()` | 执行 `.d6a` 动作组   | **播放动作**       |
| `Controller`           | 直接控制指定舵机        | **手动控制舵机**     |
| `robotall`             | 项目自己的底层能力封装     | **机器人有什么能力**   |
| `robot_tonypi`         | 上层任务、定位、导航、比赛逻辑 | **机器人什么时候做什么** |

---

# 11. 一句话总结

整个项目可以记成：

```text
.d6a
↓
“动作数据”

AGC.runActionGroup()
↓
“执行完整动作”

Controller.set_pwm_servo_pulse()
↓
“直接控制某个舵机”

robotall
↓
“把底层能力封装起来，供上层调用”

robot_tonypi
↓
“利用这些能力完成具体的定位、导航和比赛任务”
```

因此以后写自己的比赛程序时，核心思路是：

```text
                 自己的比赛逻辑
                       ↓
                 robot_tonypi
                       ↓
              调用已有底层能力
                       ↓
                    robotall
                       ↓
          ┌────────────┴────────────┐
          ↓                         ↓
   AGC.runActionGroup()        Controller
          ↓                         ↓
       .d6a动作                 单舵机控制
          ↓                         ↓
          └───────────┬─────────────┘
                      ↓
                    机器人
```

> **重点：不需要重新制作已经存在的基础动作，也不需要重新实现底层舵机控制。首先找到现有接口，然后在上层组合这些能力，实现自己的比赛逻辑。**
