"""
human_vs_ai.py — 可视化测试工具
观看训练好的 AI 对战，或与 AI 对战

操作说明：
  SPACE  — 暂停 / 继续
  R      — 重置 (新一局)
  +/-    — 加速 / 减速
  ESC/Q  — 退出

对战模式（顶部菜单）：
  AI vs AI       — 两个 PPO AI 互打（默认）
  AI vs Random   — AI 对抗随机策略
  AI vs Dummy    — AI 对抗静止假人
"""
import pygame, sys, math, os, pickle, time
import numpy as np
import torch
from game_env import LinkCombatEnv

# 操作说明
# 键盘控制 (人工模式 P2):  W/S=前后  A/D=旋转  F/Enter=开火

# ─── 路径 ────────────────────────────────────────────────────────
PPO_PATH = "E:\\Test_FIre\\best_model_ppo.pkl"
GA_PATH  = "E:\\Test_FIre\\best_model.pkl"

# ─── 显示设置 ────────────────────────────────────────────────────
ARENA         = 500          # 游戏逻辑尺寸
DISPLAY_SIZE  = 600          # 渲染尺寸
PANEL_WIDTH   = 300          # 右侧信息面板
WIN_W         = DISPLAY_SIZE + PANEL_WIDTH
WIN_H         = DISPLAY_SIZE
SCALE         = DISPLAY_SIZE / ARENA

# ─── 颜色 ────────────────────────────────────────────────────────
BG          = (15,  20,  35)
GRID        = (25,  35,  55)
P1_COLOR    = (64,  180, 255)   # 蓝色 - AI
P2_COLOR    = (255,  80,  80)   # 红色 - 对手
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


# ════════════════════════════════════════════════════════════════
# 加载模型
# ════════════════════════════════════════════════════════════════
def load_ppo_model(path):
    """加载 PPO 模型并返回推理函数"""
    from train_ppo import ActorCritic
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActorCritic().to(device)
    with open(path, 'rb') as f:
        W1,B1,W2,B2,W3,B3 = pickle.load(f)
    # 反向还原权重
    with torch.no_grad():
        model.backbone[0].weight.copy_(torch.FloatTensor(W1.T))
        model.backbone[0].bias.copy_(  torch.FloatTensor(B1))
        model.backbone[2].weight.copy_(torch.FloatTensor(W2.T))
        model.backbone[2].bias.copy_(  torch.FloatTensor(B2))
        model.mv_head.weight.copy_(    torch.FloatTensor(W3[:,0:1].T))
        model.mv_head.bias.copy_(      torch.FloatTensor([B3[0]]))
        model.rt_head.weight.copy_(    torch.FloatTensor(W3[:,1:2].T))
        model.rt_head.bias.copy_(      torch.FloatTensor([B3[1]]))
        model.fire_head.weight.copy_(  torch.FloatTensor(W3[:,2:3].T))
        model.fire_head.bias.copy_(    torch.FloatTensor([B3[2]]))
    model.eval()
    def act(obs_np):
        return model.act_deterministic(obs_np)
    return act

def load_ga_model(path):
    """加载 GA 模型并返回推理函数"""
    IN=13; H1=48; H2=32
    with open(path, 'rb') as f:
        W1,B1,W2,B2,W3,B3 = pickle.load(f)
    def act(obs_np):
        h1 = np.tanh(obs_np @ W1 + B1)
        h2 = np.tanh(h1 @ W2 + B2)
        out = h2 @ W3 + B3
        return np.array([np.tanh(out[0]), np.tanh(out[1]),
                         1.0 if out[2]>0 else 0.0], dtype=np.float32)
    return act

def random_agent(obs):
    return np.array([np.random.uniform(-1,1), np.random.uniform(-1,1),
                     float(np.random.rand()>0.8)], dtype=np.float32)

def dummy_agent(obs):
    return np.array([0.0, 0.0, 0.0], dtype=np.float32)

def human_action(keys):
    """
    坦克式控制:
      W / S — 沿着枪口方向前进 / 后退
      A / D — 左转 / 右转
      空格/F — 开火
    """
    mv   = 0.0
    rt   = 0.0
    fire = 0.0
    
    if keys[pygame.K_w]: mv =  1.0
    if keys[pygame.K_s]: mv = -1.0
    if keys[pygame.K_a]: rt = -1.0
    if keys[pygame.K_d]: rt =  1.0
    
    # 开火
    if keys[pygame.K_SPACE] or keys[pygame.K_f]:
        fire = 1.0

    return np.array([mv, rt, fire], dtype=np.float32)

# 人工玩家朝向缓存（由 main 游戏循环更新）
_human_facing = 0.0


# ════════════════════════════════════════════════════════════════
# 渲染工具
# ════════════════════════════════════════════════════════════════
def gs(x, y):
    """游戏坐标 → 屏幕坐标"""
    return int(x * SCALE), int(y * SCALE)

def draw_arena(surf):
    surf.fill(BG)
    # 网格
    step = int(100 * SCALE)
    for i in range(0, DISPLAY_SIZE+1, step):
        pygame.draw.line(surf, GRID, (i,0), (i,DISPLAY_SIZE))
        pygame.draw.line(surf, GRID, (0,i), (DISPLAY_SIZE,i))
    # 边界
    pygame.draw.rect(surf, ACCENT, (0,0,DISPLAY_SIZE,DISPLAY_SIZE), 2)
    # 中心十字
    c = DISPLAY_SIZE//2
    pygame.draw.line(surf, GRID, (c,0), (c,DISPLAY_SIZE))
    pygame.draw.line(surf, GRID, (0,c), (DISPLAY_SIZE,c))

def draw_player(surf, p, color, dark, label):
    sx, sy = gs(p[0], p[1])
    r = int(20 * SCALE)
    rad = math.radians(p[2])

    # 阴影
    pygame.draw.circle(surf, (0,0,0), (sx+3, sy+3), r)
    # 主体
    pygame.draw.circle(surf, dark, (sx, sy), r)
    pygame.draw.circle(surf, color, (sx, sy), r-3)

    # 炮管
    ex = sx + int(math.cos(rad) * r * 1.5)
    ey = sy + int(math.sin(rad) * r * 1.5)
    pygame.draw.line(surf, WHITE, (sx, sy), (ex, ey), 4)

    # 护盾弧（背面±45°）
    # 我们的物理中 y 朝下为正，所以角度 p[2] 顺时针旋转。
    # Pygame 的 draw.arc 默认 0 为右，90 为上（逆时针）。
    # 所以屏幕角度 screen_angle = -p[2]
    # 护盾在屁股后面，所以中心是 screen_angle + 180
    rear_angle = -p[2] + 180
    shield_start = math.radians(rear_angle - 45)
    shield_end   = math.radians(rear_angle + 45)
    pygame.draw.arc(surf, SHIELD_C,
                    (sx-r-4, sy-r-4, (r+4)*2, (r+4)*2),
                    shield_start,
                    shield_end, 3)

    # 标签
    _f = pygame.font.Font(None, 17)
    txt = _f.render(label, True, WHITE)
    surf.blit(txt, (sx - txt.get_width()//2, sy - r - 18))

def draw_bullets(surf, bullets):
    for b in bullets:
        sx, sy = gs(b[0], b[1])
        clr = P1_COLOR if b[4]==1 else P2_COLOR
        pygame.draw.circle(surf, BULLET_C, (sx,sy), 5)
        pygame.draw.circle(surf, clr,      (sx,sy), 3)

def draw_hp_bar(surf, x, y, w, hp, color):
    """HP 条"""
    pct = max(0, hp) / 100.0
    bg_r = pygame.Rect(x, y, w, 18)
    fg_r = pygame.Rect(x, y, int(w*pct), 18)
    pygame.draw.rect(surf, (40,40,60), bg_r, border_radius=4)
    bar_c = GREEN if pct>0.5 else (YELLOW if pct>0.25 else RED)
    pygame.draw.rect(surf, bar_c, fg_r, border_radius=4)
    pygame.draw.rect(surf, WHITE, bg_r, 1, border_radius=4)

def draw_panel(surf, font, font_s, stats, fps, speed, mode_name, paused):
    px = DISPLAY_SIZE
    panel = pygame.Surface((PANEL_WIDTH, WIN_H))
    panel.fill(PANEL_BG)
    y = 12

    def text(s, f, c=TEXT_C, bold=False):
        nonlocal y
        t = f.render(s, True, c)
        panel.blit(t, (12, y))
        y += t.get_height() + 5

    # 标题
    text("COMBAT AI MONITOR", font, ACCENT)
    pygame.draw.line(panel, ACCENT, (12,y), (PANEL_WIDTH-12,y)); y+=8

    # 模式
    text(f"Mode: {mode_name}", font_s, YELLOW)
    text(f"Speed: x{speed}  FPS:{fps:.0f}", font_s)
    if paused:
        text("[ PAUSED ]", font, RED)
    y += 5
    pygame.draw.line(panel, GRAY, (12,y), (PANEL_WIDTH-12,y)); y+=8

    # 当前局 HP
    text("HP", font, WHITE)
    text(f"AI   (P1): {stats['p1_hp']:>3d}", font_s, P1_COLOR)
    draw_hp_bar(panel, 12, y, PANEL_WIDTH-24, stats['p1_hp'], P1_COLOR)
    y += 24
    text(f"OPP  (P2): {stats['p2_hp']:>3d}", font_s, P2_COLOR)
    draw_hp_bar(panel, 12, y, PANEL_WIDTH-24, stats['p2_hp'], P2_COLOR)
    y += 24
    y += 5
    pygame.draw.line(panel, GRAY, (12,y), (PANEL_WIDTH-12,y)); y+=8

    # 步数 & 时间
    elapsed = stats['step'] / (30*60) * 30
    text(f"Time:  {elapsed:5.1f}s / 30.0s", font_s)
    text(f"Step:  {stats['step']:>5d} / 1800", font_s)
    text(f"Dist:  {stats['dist']:>5.0f}", font_s)
    y += 5
    pygame.draw.line(panel, GRAY, (12,y), (PANEL_WIDTH-12,y)); y+=8

    # 战绩统计
    text("MATCH RECORD", font, WHITE)
    w = stats['wins']; l = stats['losses']; d = stats['draws']
    total = w+l+d
    wr = w/total*100 if total>0 else 0
    text(f"Win  : {w:>4d}  ({wr:.0f}%)", font_s, GREEN)
    text(f"Loss : {l:>4d}", font_s, RED)
    text(f"Draw : {d:>4d}", font_s, GRAY)
    text(f"Total: {total:>4d}", font_s)
    y += 5
    pygame.draw.line(panel, GRAY, (12,y), (PANEL_WIDTH-12,y)); y+=8

    # 控制提示
    text("CONTROLS", font, WHITE)
    for s in ["SPACE  Pause/Resume", "R      New game",
              "+/-    Speed up/down", "ESC/Q  Quit"]:
        text(s, font_s, GRAY)

    surf.blit(panel, (px, 0))
    pygame.draw.line(surf, ACCENT, (px,0),(px,WIN_H), 2)


# ════════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════════
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Combat AI — Visual Tester")
    clock  = pygame.time.Clock()
    font   = pygame.font.Font(None, 22)   # 内置字体，绕过 Anaconda SysFont bug
    font_s = pygame.font.Font(None, 17)
    font_l = pygame.font.Font(None, 64)

    # ── 加载 AI ────────────────────────────────────────────────
    if os.path.exists(PPO_PATH):
        ai_act = load_ppo_model(PPO_PATH)
        model_src = "PPO"
    elif os.path.exists(GA_PATH):
        ai_act = load_ga_model(GA_PATH)
        model_src = "GA"
    else:
        ai_act = random_agent
        model_src = "Random(no model)"

    # 对手模式循环：AI vs AI → AI vs Random → AI vs Dummy → Human vs AI
    MODES   = ["AI vs AI", "AI vs Random", "AI vs Dummy", "Human vs AI"]
    mode_i  = 0
    opp_fns = [ai_act, random_agent, dummy_agent, None]  # None = 人工控制

    env     = LinkCombatEnv()
    obs     = env.reset(fixed_start=True)
    paused  = False
    speed   = 1       # 游戏速度倍数（1=正常，5=最快）
    MAX_SP  = 10

    stats = dict(p1_hp=100, p2_hp=100, step=0, dist=0,
                 wins=0, losses=0, draws=0)

    result_msg  = ""
    result_timer = 0

    print(f"[Visual Tester] 模型来源: {model_src}")
    print("启动可视化测试界面...")

    def reset_game():
        nonlocal obs, result_msg, result_timer
        obs = env.reset(fixed_start=True)
        result_msg   = ""
        result_timer = 0

    running = True
    while running:
        # ── 事件 ──────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_r:
                    reset_game()
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed = min(MAX_SP, speed+1)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(1, speed-1)
                elif ev.key == pygame.K_m:
                    mode_i = (mode_i+1) % len(MODES)
                    reset_game()

        # ── 游戏步进 ──────────────────────────────────────────
        keys = pygame.key.get_pressed()
        if not paused and not result_timer:
            for _ in range(speed):
                obs2    = env.get_relative_obs(env.p2, env.p1)
                action1 = ai_act(obs)
                # 人工模式：P2 由键盘控制
                if opp_fns[mode_i] is None:
                    import human_vs_ai as _hm; _hm._human_facing = env.p2[2]
                    action2 = human_action(keys)
                else:
                    action2 = opp_fns[mode_i](obs2)
                obs, reward, done = env.step(action1, action2)

                dx = env.p2[0]-env.p1[0]; dy = env.p2[1]-env.p1[1]
                stats['p1_hp'] = max(0, int(env.p1[3]))
                stats['p2_hp'] = max(0, int(env.p2[3]))
                stats['step']  = env.time
                stats['dist']  = math.hypot(dx,dy)

                if done:
                    if env.p2[3] <= 0:
                        result_msg = "AI WINS!"; stats['wins'] += 1
                    elif env.p1[3] <= 0:
                        result_msg = "AI LOSES"; stats['losses'] += 1
                    else:
                        if env.p1[3] > env.p2[3]:
                            result_msg = "AI WINS (HP)"; stats['wins'] += 1
                        elif env.p1[3] < env.p2[3]:
                            result_msg = "AI LOSES (HP)"; stats['losses'] += 1
                        else:
                            result_msg = "DRAW"; stats['draws'] += 1
                    result_timer = 90
                    break

        if result_timer > 0:
            result_timer -= 1
            if result_timer == 0:
                reset_game()

        # ── 渲染 ──────────────────────────────────────────────
        draw_arena(screen)
        draw_bullets(screen, env.bullets)
        draw_player(screen, env.p1, P1_COLOR, P1_DARK, f"AI({model_src})")
        draw_player(screen, env.p2, P2_COLOR, P2_DARK, MODES[mode_i].split(" vs ")[1])

        # 结果弹幕
        if result_msg:
            c = GREEN if "WIN" in result_msg else (RED if "LOSE" in result_msg else YELLOW)
            txt = font_l.render(result_msg, True, c)
            sx = (DISPLAY_SIZE - txt.get_width())//2
            screen.blit(txt, (sx, WIN_H//2 - 30))

        fps = clock.get_fps()
        mode_name = f"{MODES[mode_i]}  [M:switch]"
        draw_panel(screen, font, font_s, stats, fps, speed, mode_name, paused)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
