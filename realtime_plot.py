import matplotlib.pyplot as plt
import json
import time
import os
from collections import deque

# 配置
LOG_PATH = "reward_logs.jsonl"
MAX_POINTS = 500  # 图表最多显示的采样点数

def load_data_fast():
    """优化后的数据读取：只采集最后一部分数据，防止文件过大卡顿"""
    x, y_total = [], []
    y_close, y_mid, y_long = [], [], []
    
    if not os.path.exists(LOG_PATH):
        return x, y_total, y_close, y_mid, y_long

    try:
        with open(LOG_PATH, "r") as f:
            # 仅读取最后 2000 行，避免全量解析
            lines = f.readlines()
            if len(lines) > 2000:
                lines = lines[-2000:]
            
            # 采样显示，保持图表流畅
            step = max(1, len(lines) // MAX_POINTS)
            for i, line in enumerate(lines[::step]):
                data = json.loads(line)
                x.append(i)
                y_total.append(data.get("total_reward", 0))
                y_close.append(data.get("hr_close", 0) * 100)
                y_mid.append(data.get("hr_mid", 0) * 100)
                y_long.append(data.get("hr_long", 0) * 100)
    except Exception as e:
        print(f"Read error: {e}")
    
    return x, y_total, y_close, y_mid, y_long

def on_key(event):
    if event.key == 'c':
        print("Cleaning logs...")
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "w") as f: f.write("")
        plt.clf()

def main():
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.canvas.mpl_connect('key_press_event', on_key)

    while True:
        x, y_total, y_close, y_mid, y_long = load_data_fast()
        
        if x:
            ax1.clear()
            ax1.plot(x, y_total, label="Total Reward", color='blue', linewidth=1)
            ax1.set_title("Training Progress (Fast Mode - Press 'C' to clear)")
            ax1.legend(loc='upper left')

            ax2.clear()
            ax2.plot(x, y_close, label="Close Range", color='green', alpha=0.8)
            ax2.plot(x, y_mid, label="Mid Range", color='orange', alpha=0.8)
            ax2.plot(x, y_long, label="Long Range", color='red', alpha=0.8)
            ax2.set_ylabel("Accuracy (%)")
            ax2.set_ylim(-5, 105)
            ax2.legend(loc='upper left')
            ax2.grid(True, linestyle='--', alpha=0.5)

        plt.pause(1) # 降低刷新频率到 1s
        time.sleep(0.5)

if __name__ == "__main__":
    main()
