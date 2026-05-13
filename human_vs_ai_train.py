"""
human_vs_ai_train.py - 实时人工对战与训练
允许玩家亲自下场与 AI 对战，并且 AI 会在对战过程中实时进行 PPO 在线学习。
"""
import pygame, sys, math, os, pickle, time
import numpy as np
import torch
import torch.optim as optim
from game_env import LinkCombatEnv
from train_ppo import ActorCritic, compute_gae, ppo_update, RunningStats, MV_MAP, RT_MAP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── 显示设置 ────────────────────────────────────────────────────
ARENA         = 500
DISPLAY_SIZE  = 600
PANEL_WIDTH   = 300
WIN_W         = DISPLAY_SIZE + PANEL_WIDTH
WIN_H         = DISPLAY_SIZE
SCALE         = DISPLAY_SIZE / ARENA

# ─── 颜色 ────────────────────────────────────────────────────────
BG          = (15,  20,  35)
GRID        = (25,  35,  55)
P1_COLOR    = (64,  180, 255)   # AI
P2_COLOR    = (255,  80,  80)   # 人类
P1_DARK     = (30,  100, 180)
P2_DARK     = (180,  30,  30)
BULLET_C    = (255, 220,  80)
SHIELD_C    = (100, 255, 150)
PANEL_BG    = (20,  25,  40)
TEXT_C      = (200, 210, 230)
ACCENT      = (100, 200, 255)
GREEN       = (80,  220, 120)
RED         = (255,  80,  80)
YELLOW      = (255, 200,  60)
WHITE       = (255, 255, 255)
GRAY        = (100, 110, 130)

# ─── 渲染辅助函数 ─────────────────────────────────────────────────
from human_vs_ai import draw_arena, draw_player, draw_bullets, draw_hp_bar

def human_action(keys):
    mv = 0.0; rt = 0.0; fire = 0.0
    if keys[pygame.K_w]: mv =  1.0
    if keys[pygame.K_s]: mv = -1.0
    if keys[pygame.K_a]: rt = -1.0
    if keys[pygame.K_d]: rt =  1.0
    if keys[pygame.K_SPACE] or keys[pygame.K_f]: fire = 1.0
    return np.array([mv, rt, fire], dtype=np.float32)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Combat AI — Realtime Human-in-the-loop Training")
    clock  = pygame.time.Clock()
    font   = pygame.font.Font(None, 22)
    font_s = pygame.font.Font(None, 17)

    # 模型加载或初始化
    MODEL_PATH = "E:\\Test_FIre\\best_model_ppo_dynamic_attacker.pkl"
    policy = ActorCritic().to(device)
    if os.path.exists(MODEL_PATH):
        print("加载现有模型进行实时对抗训练...")
        with open(MODEL_PATH, 'rb') as f:
            weights = pickle.load(f)
        with torch.no_grad():
            if len(weights) == 6:
                W1,B1,W2,B2,W3,B3 = weights
                policy.backbone[0].weight.copy_(torch.FloatTensor(W1.T))
                policy.backbone[0].bias.copy_(  torch.FloatTensor(B1))
                policy.backbone[2].weight.copy_(torch.FloatTensor(W2.T))
                policy.backbone[2].bias.copy_(  torch.FloatTensor(B2))
                policy.mv_head.weight.copy_(    torch.FloatTensor(W3[:,0:1].T))
                policy.mv_head.bias.copy_(      torch.FloatTensor([B3[0]]))
                policy.rt_head.weight.copy_(    torch.FloatTensor(W3[:,1:2].T))
                policy.rt_head.bias.copy_(      torch.FloatTensor([B3[1]]))
                policy.fire_head.weight.copy_(  torch.FloatTensor(W3[:,2:3].T))
                policy.fire_head.bias.copy_(    torch.FloatTensor([B3[2]]))
            else:
                W1,B1,W2,B2,W3,B3,W4,B4 = weights
                policy.backbone[0].weight.copy_(torch.FloatTensor(W1.T))
                policy.backbone[0].bias.copy_(  torch.FloatTensor(B1))
                policy.backbone[2].weight.copy_(torch.FloatTensor(W2.T))
                policy.backbone[2].bias.copy_(  torch.FloatTensor(B2))
                policy.backbone[4].weight.copy_(torch.FloatTensor(W3.T))
                policy.backbone[4].bias.copy_(  torch.FloatTensor(B3))
                
                # 自动检测 W4 格式：v3.0 离散(32,7) vs 旧版连续(32,3)
                if W4.shape[1] == 7:
                    # v3.0 离散动作版：mv(3) + rt(3) + fire(1)
                    policy.mv_head.weight.copy_(    torch.FloatTensor(W4[:,0:3].T))
                    policy.mv_head.bias.copy_(      torch.FloatTensor(B4[0:3]))
                    policy.rt_head.weight.copy_(    torch.FloatTensor(W4[:,3:6].T))
                    policy.rt_head.bias.copy_(      torch.FloatTensor(B4[3:6]))
                    policy.fire_head.weight.copy_(  torch.FloatTensor(W4[:,6:7].T))
                    policy.fire_head.bias.copy_(    torch.FloatTensor(B4[6:7]))
                else:
                    # 旧版连续动作：mv(1) + rt(1) + fire(1) = (32,3)
                    policy.mv_head.weight.copy_(    torch.FloatTensor(W4[:,0:1].T))
                    policy.mv_head.bias.copy_(      torch.FloatTensor([B4[0]]))
                    policy.rt_head.weight.copy_(    torch.FloatTensor(W4[:,1:2].T))
                    policy.rt_head.bias.copy_(      torch.FloatTensor([B4[1]]))
                    policy.fire_head.weight.copy_(  torch.FloatTensor(W4[:,2:3].T))
                    policy.fire_head.bias.copy_(    torch.FloatTensor([B4[2]]))
    else:
        print("未找到现有模型，从头开始训练...")

    # 学习率适当放低进行人工对战微调，防止动作突变
    optimizer = optim.Adam(policy.parameters(), lr=1e-4) 
    reward_stats = RunningStats()

    # 经验回放缓冲区 (针对单个环境)
    IN_DIM = 16  # 与 game_env 和 train_ppo 保持一致（16维：含vel_perp预判射击特征）
    N_STEPS = 1024 # 每积累 1024 步进行一次 PPO 更新 (约17秒游戏时间)
    
    buf_obs    = np.zeros((N_STEPS, 1, IN_DIM), dtype=np.float32)
    buf_mv_idx = np.zeros((N_STEPS, 1), dtype=np.float32)   # 离散索引 {0,1,2}
    buf_rt_idx = np.zeros((N_STEPS, 1), dtype=np.float32)   # 离散索引 {0,1,2}
    buf_fire   = np.zeros((N_STEPS, 1), dtype=np.float32)
    buf_logp   = np.zeros((N_STEPS, 1), dtype=np.float32)
    buf_val    = np.zeros((N_STEPS, 1), dtype=np.float32)
    buf_rew    = np.zeros((N_STEPS, 1), dtype=np.float32)
    buf_done   = np.zeros((N_STEPS, 1), dtype=np.float32)

    step_idx = 0
    update_count = 0

    env = LinkCombatEnv()
    obs = env.reset(fixed_start=False)
    
    stats = dict(p1_hp=100, p2_hp=100, step=0, wins=0, losses=0, updates=0, last_loss=0.0)

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        keys = pygame.key.get_pressed()
        
        # PPO 采样动作 (带有一点探索性)
        policy.eval()
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            mv, rt, fire, mv_idx, rt_idx, logp, value = policy.act(obs_tensor)
        
        mv_np      = mv.cpu().numpy()[0]
        rt_np      = rt.cpu().numpy()[0]
        fire_np    = fire.cpu().numpy()[0]
        mv_idx_np  = mv_idx.cpu().numpy()[0]
        rt_idx_np  = rt_idx.cpu().numpy()[0]
        logp_np    = logp.cpu().numpy()[0]
        val_np     = value.cpu().numpy()[0]
        
        action1 = np.array([mv_np, rt_np, fire_np], dtype=np.float32)
        action2 = human_action(keys)
        
        next_obs, reward, done = env.step(action1, action2)
        
        # 记录经验（存储离散索引供 evaluate 使用）
        buf_obs[step_idx, 0]    = obs
        buf_mv_idx[step_idx, 0] = mv_idx_np
        buf_rt_idx[step_idx, 0] = rt_idx_np
        buf_fire[step_idx, 0]   = fire_np
        buf_logp[step_idx, 0]   = logp_np
        buf_val[step_idx, 0]    = val_np
        buf_rew[step_idx, 0]    = reward
        buf_done[step_idx, 0]   = float(done)
        
        step_idx += 1
        obs = next_obs

        stats['p1_hp'] = env.p1[3]
        stats['p2_hp'] = env.p2[3]
        stats['step']  = env.time
        
        if done:
            if env.p2[3] <= 0: stats['wins'] += 1
            elif env.p1[3] <= 0: stats['losses'] += 1
            obs = env.reset(fixed_start=False)

        # 触发 PPO 实时更新
        if step_idx >= N_STEPS:
            print(f"积累达到 {N_STEPS} 步，触发实时 PPO 更新...")
            # 奖励归一化
            buf_rew_norm = reward_stats.update_and_normalize(buf_rew.copy())
            
            # 计算最后的 value
            policy.eval()
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                last_val = policy.get_value(obs_tensor).cpu().numpy()[0]
                
            advantages, returns = compute_gae(buf_rew_norm, buf_val, buf_done, np.array([last_val]))
            
            batch = (
                buf_obs.reshape(-1, IN_DIM),
                buf_mv_idx.flatten(),
                buf_rt_idx.flatten(),
                buf_fire.flatten(),
                buf_logp.flatten(),
                advantages,
                returns
            )
            
            policy.train()
            import train_ppo
            train_ppo.BATCH_SIZE = 128  # 人机对战单次样本较少，缩小 batch
            train_ppo.N_EPOCHS = 4      # 控制更新轮数以免过度拟合单局
            loss, pg, vf, ent = ppo_update(policy, optimizer, batch)
            print(f"第 {stats['updates']+1} 次更新完成 - Loss: {loss:.3f} | Value_loss: {vf:.3f}")
            
            stats['updates'] += 1
            stats['last_loss'] = loss
            step_idx = 0
            
            # 自动保存最新模型，供下次对局或离线继续训练使用
            weights = policy.export_weights()
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(weights, f)

        # ── 渲染 ──────────────────────────────────────────────
        screen.fill(BG)
        draw_arena(screen)
        draw_bullets(screen, env.bullets)
        draw_player(screen, env.p1, P1_COLOR, P1_DARK, "AI (Learning)")
        draw_player(screen, env.p2, P2_COLOR, P2_DARK, "Human")
        
        # 绘制简易信息 (位于屏幕外侧或顶部)
        panel_rect = pygame.Rect(DISPLAY_SIZE, 0, PANEL_WIDTH, WIN_H)
        pygame.draw.rect(screen, PANEL_BG, panel_rect)
        
        y = 20
        def text(s, f, c=TEXT_C):
            nonlocal y
            t = f.render(s, True, c)
            screen.blit(t, (DISPLAY_SIZE + 15, y))
            y += t.get_height() + 10
            
        text("Real-time Training", font, ACCENT)
        text("------------------", font, GRAY)
        text(f"AI HP: {env.p1[3]:.0f}", font, P1_COLOR)
        text(f"Human HP: {env.p2[3]:.0f}", font, P2_COLOR)
        text("------------------", font, GRAY)
        text(f"Updates: {stats['updates']}", font, WHITE)
        text(f"Last Loss: {stats['last_loss']:.3f}", font, WHITE)
        text(f"Buffer: {step_idx} / {N_STEPS}", font, YELLOW)
        text(f"AI Wins: {stats['wins']}", font, GREEN)
        text(f"Human Wins: {stats['losses']}", font, RED)
        
        text("------------------", font, GRAY)
        text("CONTROLS:", font, WHITE)
        text("W/S: Move", font_s, GRAY)
        text("A/D: Rotate", font_s, GRAY)
        text("Space/F: Fire", font_s, GRAY)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
