import json
import math
import pickle
import os
import sys
import numpy as np
import torch

# ─── 配置参数 ──────────────────────────────────────────
MODEL_PATH = "E:\\Test_FIre\\best_model_ppo.pkl"

# 环境常量，与训练时保持一致
ARENA_SIZE = 500.0
PLAYER_RADIUS = 20.0
FIRE_COOLDOWN = 15.0

def load_ai(path):
    """加载 PPO 模型，提取推理函数"""
    from train_ppo import ActorCritic
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActorCritic().to(device)
    with open(path, 'rb') as f:
        W1, B1, W2, B2, W3, B3 = pickle.load(f)
    
    with torch.no_grad():
        model.backbone[0].weight.copy_(torch.FloatTensor(W1.T))
        model.backbone[0].bias.copy_(torch.FloatTensor(B1))
        model.backbone[2].weight.copy_(torch.FloatTensor(W2.T))
        model.backbone[2].bias.copy_(torch.FloatTensor(B2))
        model.mv_head.weight.copy_(torch.FloatTensor(W3[:,0:1].T))
        model.mv_head.bias.copy_(torch.FloatTensor([B3[0]]))
        model.rt_head.weight.copy_(torch.FloatTensor(W3[:,1:2].T))
        model.rt_head.bias.copy_(torch.FloatTensor([B3[1]]))
        model.fire_head.weight.copy_(torch.FloatTensor(W3[:,2:3].T))
        model.fire_head.bias.copy_(torch.FloatTensor([B3[2]]))
    model.eval()
    return model

def build_obs(p_self, p_enemy):
    """
    根据 JSON 中的数据计算 13 维状态观测向量 (与 game_env.py 一致)
    p_self / p_enemy: {"x": 50, "y": 50, "a": 45, "hp": 100}
    """
    dx = p_enemy["x"] - p_self["x"]
    dy = p_enemy["y"] - p_self["y"]
    dist = math.hypot(dx, dy)

    etx = math.cos(math.radians(p_self["a"]))
    ety = math.sin(math.radians(p_self["a"]))
    
    dot_val = max(-1.0, min(1.0, (dx*etx + dy*ety) / max(dist, 1e-6)))
    angle_diff = math.degrees(math.acos(dot_val))
    
    # 护盾盲区暴露度计算
    etm  = math.degrees(math.atan2(-dy, -dx)) % 360
    ea   = abs((etm - p_enemy["a"] + 180) % 360 - 180)
    dc   = (math.degrees(math.asin(min(1.0, PLAYER_RADIUS/dist)))
            if dist > PLAYER_RADIUS else 90.0)
    exposure = max(0.0, 1.0 - ea / max(dc, 1e-6))

    # 距墙距离
    mw = min(p_self["x"], ARENA_SIZE - p_self["x"],
             p_self["y"], ARENA_SIZE - p_self["y"])
    wall = mw / (ARENA_SIZE / 2.0)
    
    rs = math.radians(p_self["a"])
    re = math.radians(p_enemy["a"])
    
    # 收包格式中没有 CD，假设 CD 始终为 0 (可随时开火)
    cd = 0.0

    obs = np.array([
        dist / 707.0,
        angle_diff / 180.0,
        dx / ARENA_SIZE, dy / ARENA_SIZE,
        p_self["hp"] / 100.0,
        p_enemy["hp"] / 100.0,
        exposure, wall,
        math.sin(rs), math.cos(rs),
        math.sin(re), math.cos(re),
        cd,
    ], dtype=np.float32)
    return obs

def discretize_action(raw_action):
    """
    将连续动作转换为离散指令 (-1, 0, 1)
    raw_action: [mv(-1~1), rt(-1~1), fire(0或1)]
    """
    mv_raw, rt_raw, fire_raw = raw_action
    
    # 移动阈值判断
    if mv_raw > 0.3:
        mv = 1      # 前进
    elif mv_raw < -0.3:
        mv = -1     # 后退
    else:
        mv = 0      # 不动
        
    # 旋转阈值判断
    if rt_raw > 0.3:
        rt = 1      # 顺时针
    elif rt_raw < -0.3:
        rt = -1     # 逆时针
    else:
        rt = 0      # 不转
        
    # 开火判断
    fr = 1 if fire_raw > 0.5 else 0
    
    return {"mv": mv, "rt": rt, "fr": fr}

def process_packet(ai_model, json_str):
    """处理单条 JSON 请求并返回 JSON 响应"""
    try:
        data = json.loads(json_str)
        p1 = data["p1"] # AI
        p2 = data["p2"] # 对手
        
        # 1. 组装状态向量
        obs = build_obs(p1, p2)
        
        # 2. 神经网络推理
        raw_action = ai_model.act_deterministic(obs)
        
        # 3. 动作离散化处理
        cmd = discretize_action(raw_action)
        
        return json.dumps(cmd)
    except Exception as e:
        # 如果解析错误，返回全 0 安全动作
        return json.dumps({"mv": 0, "rt": 0, "fr": 0, "error": str(e)})

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"找不到模型文件: {MODEL_PATH}")
        sys.exit(1)
        
    print(f"正在加载模型...")
    ai_model = load_ai(MODEL_PATH)
    print("加载完成！正在监听输入...\n")
    print('请输入 JSON 格式包 (例如: {"p1":{"x":50,"y":50,"a":45,"hp":100},"p2":{"x":150,"y":150,"a":225,"hp":100}})')
    print('按 Ctrl+C 退出。')

    # ========== 标准输入输出模式 ==========
    # 适用于串口透传、子进程调用或手动测试
    try:
        while True:
            # 阻塞读取一行
            line = sys.stdin.readline()
            if not line:
                break
                
            line = line.strip()
            if not line:
                continue
                
            # 处理并返回
            response = process_packet(ai_model, line)
            print(response)
            sys.stdout.flush()  # 确保立即发送
            
    except KeyboardInterrupt:
        print("\n退出。")

    # ========== 串口通信模式 (备用代码) ==========
    """
    import serial
    ser = serial.Serial('COM3', 115200, timeout=0.1)
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line:
            resp = process_packet(ai_model, line)
            ser.write((resp + "\\n").encode('utf-8'))
    """

    # ========== UDP 通信模式 (备用代码) ==========
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 8888))
    while True:
        data, addr = sock.recvfrom(1024)
        line = data.decode('utf-8').strip()
        resp = process_packet(ai_model, line)
        sock.sendto((resp + "\\n").encode('utf-8'), addr)
    """

if __name__ == "__main__":
    main()
