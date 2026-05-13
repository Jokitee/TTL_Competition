import matplotlib.pyplot as plt
import json
import os
import time
import numpy as np
from matplotlib.animation import FuncAnimation

# ============================================================
# 实时监控配置
# ============================================================
LOG_FILE = "E:\\Test_FIre\\reward_logs.jsonl"
REFRESH_MS = 2000  # 每2秒刷新一次
SMOOTH_WEIGHT = 0.9  # 平滑权重

def smooth(data, weight=0.9):
    if not data: return []
    smoothed = []
    last = data[0]
    for val in data:
        last = last * weight + val * (1 - weight)
        smoothed.append(last)
    return smoothed

def animate(i):
    if not os.path.exists(LOG_FILE):
        return

    history = []
    try:
        with open(LOG_FILE, "r") as f:
            for line in f:
                if line.strip():
                    history.append(json.loads(line.strip()))
    except Exception:
        return

    if not history:
        return

    categories = ["Aiming", "Dodging", "Tracking", "Damage"]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    episodes = list(range(1, len(history) + 1))

    for idx, cat in enumerate(categories):
        y_raw = []
        for ep in history:
            total_abs = sum(abs(ep[c]) for c in categories) + 1e-6
            pct = (abs(ep[cat]) / total_abs) * 100.0
            y_raw.append(pct)
        
        y_smoothed = smooth(y_raw, SMOOTH_WEIGHT)
        
        axs[idx].clear()
        axs[idx].plot(episodes, y_raw, color=colors[idx], alpha=0.2)
        axs[idx].plot(episodes, y_smoothed, color=colors[idx], linewidth=2, label=f"{cat} Smoothed")
        axs[idx].set_title(f"{cat} 能力占比 (%)")
        axs[idx].set_ylim(0, 100)
        axs[idx].set_ylabel("%")
        axs[idx].grid(True, linestyle='--', alpha=0.5)
        axs[idx].legend(loc='upper right')

    axs[3].set_xlabel("对局数 (Episodes)")
    plt.tight_layout()

# 初始化画布
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
fig.suptitle('AI 训练能力拟合实时监控 (Real-time Fitness Monitor)', fontsize=16)

ani = FuncAnimation(fig, animate, interval=REFRESH_MS)

print(f"正在实时监控: {LOG_FILE}")
print("提示: 请保持训练脚本开启并设置 analytics.ENABLE_ANALYTICS = True")
plt.show()
