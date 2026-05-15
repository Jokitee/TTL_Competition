import json
import os
import time

ENABLE_ANALYTICS = True

class Analytics:
    def __init__(self, log_path="reward_logs.jsonl"):
        self.log_path = log_path
        self.current_episode = {
            "Aiming": 0.0, "Damage": 0.0, "Tracking": 0.0, "Dodging": 0.0,
            "hits_close": 0, "shots_close": 0,
            "hits_mid": 0, "shots_mid": 0,
            "hits_long": 0, "shots_long": 0
        }

    def clear_logs(self):
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
        with open(self.log_path, "w") as f:
            pass

    def add_reward(self, category, value):
        if category in self.current_episode:
            self.current_episode[category] += value

    def log_shot(self, dist, hit=False):
        if dist < 150:
            self.current_episode["shots_close"] += 1
            if hit: self.current_episode["hits_close"] += 1
        elif dist < 300:
            self.current_episode["shots_mid"] += 1
            if hit: self.current_episode["hits_mid"] += 1
        else:
            self.current_episode["shots_long"] += 1
            if hit: self.current_episode["hits_long"] += 1

    def save_log(self):
        def calc_rate(hits, shots):
            return hits / shots if shots > 0 else 0.0

        hr_c = calc_rate(self.current_episode["hits_close"], self.current_episode["shots_close"])
        hr_m = calc_rate(self.current_episode["hits_mid"], self.current_episode["shots_mid"])
        hr_l = calc_rate(self.current_episode["hits_long"], self.current_episode["shots_long"])
        
        log_data = {
            "timestamp": time.time(),
            "Aiming": self.current_episode["Aiming"],
            "Damage": self.current_episode["Damage"],
            "Tracking": self.current_episode["Tracking"],
            "Dodging": self.current_episode["Dodging"],
            "hr_close": hr_c, "hr_mid": hr_m, "hr_long": hr_l,
            "total_reward": sum([self.current_episode[k] for k in ["Aiming", "Damage", "Tracking", "Dodging"]])
        }
        
        # 强制刷盘，确保实时性能看到数据
        with open(self.log_path, "a") as f:
            f.write(json.dumps(log_data) + "\n")
            f.flush()
        
        # 控制台反馈
        print(f"  [Analytics] Episode Saved | HR Close: {hr_c:.1%} | Mid: {hr_m:.1%} | Long: {hr_l:.1%}")
        
        for k in self.current_episode:
            self.current_episode[k] = 0.0

analyzer = Analytics()
