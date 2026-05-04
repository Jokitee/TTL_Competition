# Link-Combat STM32 AI Controller

这是一个为 STM32 单片机设计的轻量级神经网络 AI 控制器，用于在 Link-Combat 机甲对战中实现智能走位、精准瞄准与自动防御。该项目包含 Python 端的强化学习/遗传算法训练环境，以及可以直接部署至单片机的 C 语言推理引擎。

## 🌟 核心特性 (Features)

* **极低资源占用**：13 -> 48 -> 32 -> 3 的轻量级前馈神经网络。Flash 占用仅约 9.4KB，完美适配 STM32F103C8T6 等资源受限的 MCU。
* **智能攻击角度修正**：AI 模型在环境冷却期间会疯狂寻找最佳射击角度，只要冷却完毕就会开火，倒逼其实现 "完美死盯" 敌方的瞄准能力。
* **智能护盾躲避与平移**：AI 模型自带子弹威胁预测，面对敌方射来的子弹时，会自主判断是进行**平移侧滑规避**，还是直接**180度转身用护盾硬接子弹**。
* **一键导出 C 代码**：训练完毕后，提供脚本将模型 `.pkl` 权重一键转换为 C 语言的 `model_weights.h` 数组。

## 📂 目录结构

* `game_env.py`: 机甲对战的物理与逻辑环境，包含完整的碰撞、伤害、开火判定及奖惩函数（Reward Shaping）。
* `train_ppo.py` / `train_ga.py`: 分别使用 PPO 强化学习和遗传算法 (GA) 训练 AI 的脚本。
* `export_to_c.py`: 将训练后的最优模型导出为 STM32 可用的头文件。
* `nn_inference.c`: 纯 C 语言编写的单片机神经网络前向推理引擎及特征工程算子。
* `uart_ai_handler.c`: STM32 串口 JSON 解析与调度器。

## 🚀 快速上手 (Quick Start)

### 1. 训练你的 AI 模型
你可以选择使用 PPO 或者 GA 进行训练。进入项目目录并运行：
```bash
python train_ppo.py
```
*提示：在 `game_env.py` 中，我们强制了 `fr=1`，让 AI 能够专注于学习走位(mv)与旋转(rt)。*

### 2. 导出模型到 C 语言头文件
训练完成后，会生成 `.pkl` 文件。运行导出脚本：
```bash
python export_to_c.py
```
这会在当前目录生成一个 `model_weights.h`。

### 3. 部署到 STM32 工程
将以下三个文件拷贝至你的 STM32 工程目录中（如 `Core/Src` 及 `Core/Inc`）：
1. `model_weights.h` (刚刚生成的权重文件)
2. `nn_inference.c`、`nn_inference.h` (推理引擎及特征提取)
3. `uart_ai_handler.c`、`uart_ai_handler.h`(串口通信调度)

### 4. STM32 代码集成
在你的 `main.c` 中开启 UART1 接收中断：

```c
#include "uart_ai_handler.h"

int main(void) {
    // ... 其他初始化代码 ...
    
    // 初始化 AI 串口接收
    AI_UART_Init();
    
    while (1) {
        // ... 主循环 ...
    }
}
```

在你的中断处理文件 `stm32f1xx_it.c` (或直接在 `main.c` 底部) 添加回调对接：
```c
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
    if (huart->Instance == USART1) {
        AI_UART_RxCallback(); 
    }
}
```

## 📡 UART 通信协议

STM32 通过 UART1 接收裁判系统或上位机发来的当前战局数据，随后立即回传 AI 动作。

**上位机发给 STM32 (100ms 频率)**:
```json
{"p1":{"x":50.0,"y":50.0,"a":45.0,"hp":100},"p2":{"x":450.0,"y":450.0,"a":225.0,"hp":100}}
```

**STM32 发给上位机**:
```json
{"mv":1,"rt":-1,"fr":1}
```

## 🧠 神经网络架构设计

* **输入层 (13维)**: 包含双方相对距离、角度偏差、相对坐标差、双方血量、自身冷却状态、暴露系数、距墙距离等。
* **隐藏层**: 
  - L1: 48神经元 (Tanh激活)
  - L2: 32神经元 (Tanh激活)
* **输出层 (3维)**:
  - `mv`: 前后移动 (-1 到 1)
  - `rt`: 左右旋转 (-1 到 1)
  - `fr`: 是否开火 (目前在训练中被重写为持续开火)

## 💡 开发提示
* 如果你需要在 C 语言中维护冷却时间（避免 AI 发射频率超过游戏限制），请在 `uart_ai_handler.c` 中结合定时器中断自行对 `self_cd` 进行递减。
* `combat_logic.c` 是一个不依赖神经网络的硬编码简单逻辑，如果你使用 AI 模型，则无需将该文件烧录进单片机。


## 改进方向

当前的训练结果偏于直线solo，没有智能决策，