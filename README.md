# 射击对战 AI — 灵动博弈系统 v5.0 (Combat AI v5.0)

基于 **PPO + 课程学习 + 分支网络架构** 打造的高精度 2D 格斗 AI 项目。  
训练完成的权重可一键导出为 C 头文件，直接部署到 **STM32 等嵌入式硬件**。

---

## 🧠 v5.0 核心架构：解耦分支网络 (Branched Architecture)

```
输入层 (16维观测)
    │
    ▼
[共享感知层] 16 → 64  (Shared Perception)
  提取通用几何与物理特征
    │
    ├──────────────────────────────┐
    ▼                              ▼
[进攻火控分支] 64 → 48        [战术走位分支] 64 → 48
 Offense Branch                  Tactical Branch
 专精炮塔旋转 & 开火时机          专精底盘机动 & 绕后侧翼
    │                              │
    ├── rt_head  → 3 (炮塔旋转)    └── mv_head → 3 (底盘移动)
    └── fire_head → 1 (开火)
```

> 传统单链网络中"射击梯度"和"走位梯度"互相干扰，导致 AI 既走不好也打不准。分支架构让两种技能**独立进化**，实现"走位时不忘瞄准，开火时不停位移"的双核战术执行力。

---

## 🎓 4 阶段课程学习

| 阶段 | 步数范围 | 训练内容 | AI 状态 |
|------|---------|---------|--------|
| **Stage 0** | 0 ~ 100万 | 近程快反 (距离 50-150) | 固定位置，练习 360° 极速转头 |
| **Stage 1** | 100 ~ 300万 | 中程精准 (距离 150-300) | 固定位置，学习稳定追踪 |
| **Stage 2** | 300 ~ 500万 | 远程预判 (距离 300-500) | 固定位置，强化打提前量能力 |
| **Stage 3** | 500万+ | 全量实战 | 自由移动，自博弈死斗 |

---

## 📂 核心文件说明

| 文件 | 说明 |
|------|------|
| `train_ppo.py` | 主训练脚本，4 阶段课程学习，权重保存至 `best_model_v5_branched.pkl` |
| `game_env.py` | 物理引擎 + 完整奖励塑形（含预判、侧翼、风筝逻辑） |
| `reward_config.py` | 奖励分值配置，调整此文件可改变 AI 战术风格 |
| `human_vs_ai.py` | 可视化对战工具，**按 'M' 键切换到 Human vs AI 亲自对战** |
| `human_vs_ai_train.py` | 人机对战实时 PPO 在线学习 |
| `realtime_plot.py` | 训练命中率实时监控图表 |
| `migrate_weights.py` | 旧架构权重迁移工具（架构升级时保留训练成果） |
| `export_to_c.py` | 将权重导出为 `model_weights.h` C 头文件 |
| `nn_inference.c/h` | STM32 C 推理引擎（与 Python 完全对齐） |
| `uart_ai_handler.c/h` | STM32 UART 通信 + 推理调度 |
| `combat_logic.c` | 备用规则策略（用于回退/条试） |

---

## 🚀 训练使用流程

### 第一步：（可选）从旧版本迁移权重
如果之前已有训练好的旧版模型，先运行迁移工具保留已有成果：
```powershell
python migrate_weights.py
```
> 迁移完成后输出 `best_model_v5_branched.pkl`，感知层和分支层权重完美继承，输出头约 10~30 万步重新收敛。  
> 若已有 `best_model_v5_branched.pkl`，脚本自动跳过，不会覆盖。

### 第二步：开启特训
```powershell
python train_ppo.py
```
> 权重每 50 万步自动保存至 `best_model_v5_branched.pkl`。

### 第三步：实时监控命中率
新开一个终端运行：
```powershell
python realtime_plot.py
```
> 在图表窗口按 **'C'** 清理历史数据，重新统计。

### 第四步：人机对战验证
```powershell
python human_vs_ai.py
```
> 启动后按 **'M'** 切换模式，直到出现 `Human vs AI`。  
> **W/S** 前后移动 | **A/D** 左右旋转 | **空格/F** 开火

### 第五步：（可选）人机实时对练
边打边训练，AI 会实时学习您的战术：
```powershell
python human_vs_ai_train.py
```
> 每积累 1024 步自动触发一次 PPO 更新，训练结果实时保存。

---

## 📦 硬件部署流程 (STM32)

### 第一步：导出 C 权重文件
训练完成后，运行：
```powershell
python export_to_c.py
```
自动生成 `model_weights.h`，内含所有神经网络权重的 C 浮点数组。

### 第二步：加入 STM32 工程
将以下 4 个文件复制到 STM32 工程的 `Core/Src` 或 `AI/` 目录：

```
model_weights.h      ← 网络权重（由 export_to_c.py 生成）
nn_inference.c       ← 神经网络推理引擎（分支架构）
nn_inference.h       ← 推理接口头文件
uart_ai_handler.c    ← UART 通信 + 推理调度主控
uart_ai_handler.h    ← 调度接口头文件
```

### 第三步：初始化 UART 中断
在 `main.c` 中调用：
```c
#include "uart_ai_handler.h"

// 在 main() 中的 while(1) 之前调用一次
AI_UART_Init();
```

### 第四步：注册中断回调
在 `stm32xxxx_it.c` 的 `HAL_UART_RxCpltCallback` 中添加：
```c
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
    if (huart->Instance == UART5) {   // 根据实际 UART 修改
        AI_UART_RxCallback();
    }
}
```

### 第五步：配置阵营
在 `uart_ai_handler.c` 顶部修改宏定义（决定我方是 P1 还是 P2）：
```c
#define AI_PLAY_AS_P2  1   // 0 = 我方是 P1，1 = 我方是 P2
```

### 通信协议
| 方向 | 格式 | 示例 |
|------|------|------|
| 上位机 → STM32 | `{"p1":{"x":50,"y":50,"a":45,"hp":100},"p2":{...}}\n` | 每帧发送，以 `\n` 结尾 |
| STM32 → 上位机 | `{"mv":1,"rt":0,"fr":1}\n` | `mv`/`rt` ∈ {-1,0,1}，`fr` ∈ {0,1} |

### 输出动作说明
| 字段 | 含义 | 取值 |
|------|------|------|
| `mv` | 底盘前后移动 | `-1`(后退) / `0`(停止) / `+1`(前进) |
| `rt` | 炮塔左右旋转 | `-1`(右转) / `0`(不转) / `+1`(左转) |
| `fr` | 开火 | `0`(不开火) / `1`(开火) |

---

## 💡 常见问题

**Q: AI 不开枪 / 命中率为 0？**  
A: 检查 `reward_config.py` 中 `FIRE_PERFECT_AIM` 是否为正值（建议 ≥ 30.0）。若 `FIRE_BAD_AIM_PENALTY` 负值过大，AI 会因怕扣分而永远不敢开枪。

**Q: AI 只转圈不打人？**  
A: 降低 `EXPOSURE_REWARD_SCALE` 和 `TRACK_AIM_WEIGHT`，提高 `HIT_ENEMY` 权重，迫使 AI 追求实质伤害而非保持"占位姿势"。

**Q: 训练/图表卡顿？**  
A: 减少 `train_ppo.py` 中的 `N_ENVS`（16 → 8 或 4）；`realtime_plot.py` 已内置末尾 2000 行采样，图表无论日志多大都不会卡。

**Q: STM32 接收到乱码或解析失败？**  
A: 确认上位机发送的 JSON 格式与协议完全一致，字段名不能有多余空格，末尾必须有 `\n`。检查波特率是否与 CubeMX 中 UART 配置一致。