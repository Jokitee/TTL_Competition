import matplotlib.pyplot as plt
import json
import os
import numpy as np

class RewardAnalyzer:
    """辅助分析工具：统计并绘制 AI 在训练中各维度的能力拟合度折线图"""
    def __init__(self, log_file="E:\\Test_FIre\\reward_logs.jsonl"):
        self.log_file = log_file
        self.data = {
            "Aiming": 0.0,
            "Dodging": 0.0,
            "Tracking": 0.0,
            "Damage": 0.0,
            "Count": 0
        }
        # 确保目录存在，并可选择在此处清空旧日志
        # if os.path.exists(self.log_file): os.remove(self.log_file)

    def add_reward(self, category, amount):
        if category in self.data:
            self.data[category] += amount

    def step_count(self):
        self.data["Count"] += 1

    def save_log(self):
        """每局结束时将当前对局的得分权重作为一行 JSON 追加到文件中"""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(self.data) + "\n")
        
        # 清空当前内存，准备下一局
        self.data = {k: 0.0 for k in self.data}
        self.data["Count"] = 0

    @staticmethod
    def plot_distribution(log_file="E:\\Test_FIre\\reward_logs.jsonl", save_path="E:\\Test_FIre\\reward_distribution.png"):
        """读取所有的对局日志，绘制4个折线统计图 (横轴: 局数, 纵轴: 百分比)"""
        if not os.path.exists(log_file):
            print("未找到分析数据，请确保 ENABLE_ANALYTICS 已开启并进行了训练。")
            return
            
        history = []
        with open(log_file, "r") as f:
            for line in f:
                if line.strip():
                    history.append(json.loads(line.strip()))
                    
        if not history:
            return

        categories = ["Aiming", "Dodging", "Tracking", "Damage"]
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        # 简单的滑动平均函数，防止单局数据抖动太大
        def smooth(data, weight=0.85):
            smoothed = []
            last = data[0]
            for val in data:
                last = last * weight + val * (1 - weight)
                smoothed.append(last)
            return smoothed
        
        episodes = list(range(1, len(history) + 1))
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial'] 
        plt.rcParams['axes.unicode_minus'] = False 
        
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        fig.suptitle('模型各能力拟合度折线图 (Fitness Percentage Over Episodes)', fontsize=16)
        
        for i, (cat, color) in enumerate(zip(categories, colors)):
            # 计算单局某类别占当局所有得分总绝对值的百分比
            y_raw = []
            for ep in history:
                total_abs = sum(abs(ep[c]) for c in categories) + 1e-6
                pct = (abs(ep[cat]) / total_abs) * 100.0
                y_raw.append(pct)
            
            y_smoothed = smooth(y_raw)
            
            axs[i].plot(episodes, y_raw, color=color, alpha=0.3, label='Raw')
            axs[i].plot(episodes, y_smoothed, color=color, linewidth=2, label='Smoothed')
            
            axs[i].set_title(f"{cat} (目标/规避/追踪/伤害) 拟合百分比")
            axs[i].set_ylabel("百分比 (%)")
            axs[i].set_ylim(0, 100)
            axs[i].grid(True, linestyle='--', alpha=0.6)
            axs[i].legend(loc='upper right')
            
        axs[3].set_xlabel("对局数 (Episodes)")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # 给大标题留出空间
        plt.savefig(save_path)
        print(f"\n[Analytics] 拟合度折线图已生成并保存至: {save_path}")

# 全局实例
analyzer = RewardAnalyzer()
ENABLE_ANALYTICS = False
