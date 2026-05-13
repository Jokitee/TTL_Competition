import numpy as np
import math
import random
from reward_config import RewardConfig
import analytics

# ============================================================
# Numba JIT 加速：若安装了 numba 则自动启用（约 10~30x 加速）
# 若未安装则回退到纯 Python
# ============================================================
try:
    from numba import njit
    NUMBA = True
except ImportError:
    def njit(*a, **kw):          # 空装饰器，透传原函数
        return lambda f: f
    NUMBA = False


@njit(cache=True)
def _step_geometry(p1x, p1y, p1a, p2x, p2y, p2a, player_radius):
    """纯数值几何计算，JIT 加速最有效的部分"""
    dx   = p2x - p1x
    dy   = p2y - p1y
    dist = math.sqrt(dx*dx + dy*dy)
    tgt  = math.degrees(math.atan2(dy, dx)) % 360.0
    angle_diff = abs((tgt - p1a + 180.0) % 360.0 - 180.0)

    etm  = math.degrees(math.atan2(-dy, -dx)) % 360.0
    ea   = abs((etm - p2a + 180.0) % 360.0 - 180.0)
    dc   = (math.degrees(math.asin(min(1.0, player_radius/dist)))
            if dist > player_radius else 90.0)
    exposure = max(0.0, 1.0 - ea / max(dc, 1e-6))

    # 护盾盲区
    esc  = (p2a + 180.0) % 360.0
    sgap = abs((etm - esc + 180.0) % 360.0 - 180.0)
    blind = sgap > 45.0

    return dist, angle_diff, exposure, blind, dx, dy, ea


class LinkCombatEnv:
    def __init__(self, max_steps=900):
        self.arena_size       = 500
        self.player_radius    = 20
        self.player_speed     = 3.0
        self.player_rot_speed = 3.0
        self.bullet_speed     = 15.0
        self.fire_cooldown    = 15
        self.shield_half_angle = 45.0
        self.max_hp           = 100
        self.game_duration    = max_steps   # 可按阶段调节
        self.reset()

    def reset(self, fixed_start=True):
        if fixed_start:
            self.p1 = [100.0, 100.0,  45.0, 100.0, 0.0]   # 左上，朝向右下 (45°)
            self.p2 = [400.0, 400.0, 225.0, 100.0, 0.0]   # 右下，朝向左上 (225°)
        else:
            margin = 50
            lo, hi = margin, self.arena_size - margin
            for _ in range(200):
                x1 = random.uniform(lo, hi); y1 = random.uniform(lo, hi)
                x2 = random.uniform(lo, hi); y2 = random.uniform(lo, hi)
                if 200 < math.hypot(x2-x1, y2-y1) < 550:
                    break
            self.p1 = [x1, y1, random.uniform(0,360), 100.0, 0.0]
            self.p2 = [x2, y2, random.uniform(0,360), 100.0, 0.0]

        self.bullets  = []
        self.time     = 0
        self.p1_dist  = 0.0
        self.p2_dist  = 0.0
        self.prev_dist = math.hypot(self.p2[0]-self.p1[0],
                                    self.p2[1]-self.p1[1])
        return self._get_obs()

    def get_relative_obs(self, p_self, p_enemy):
        dx   = p_enemy[0] - p_self[0]
        dy   = p_enemy[1] - p_self[1]
        dist = math.hypot(dx, dy)

        tgt  = math.degrees(math.atan2(dy, dx)) % 360
        angle_diff = (tgt - p_self[2] + 180) % 360 - 180

        etm  = math.degrees(math.atan2(-dy, -dx)) % 360
        ea   = abs((etm - p_enemy[2] + 180) % 360 - 180)
        dc   = (math.degrees(math.asin(min(1.0, self.player_radius/dist)))
                if dist > self.player_radius else 90.0)
        exposure = max(0.0, 1.0 - ea / max(dc, 1e-6))

        mw   = min(p_self[0], self.arena_size-p_self[0],
                   p_self[1], self.arena_size-p_self[1])
        wall = mw / (self.arena_size / 2.0)
        rs   = math.radians(p_self[2]); re = math.radians(p_enemy[2])
        cd   = p_self[4] / self.fire_cooldown

        return np.array([
            dist / 707.0,
            angle_diff / 180.0,
            dx / 500.0, dy / 500.0,
            p_self[3]  / 100.0,
            p_enemy[3] / 100.0,
            exposure, wall,
            math.sin(rs), math.cos(rs),
            math.sin(re), math.cos(re),
            cd,
        ], dtype=np.float32)

    def _get_obs(self):
        return self.get_relative_obs(self.p1, self.p2)

    def step(self, action1, action2):
        self.time += 1
        prev_dist = self.prev_dist

        self._update_player(self.p1, action1, is_p1=True)
        self._update_player(self.p2, action2, is_p1=False)
        self._resolve_collision()   # 碰撞推开，防止穿透

        step_reward = 0.0

        # 加速几何计算
        dist, angle_diff, exposure, blind, dx, dy, ea = _step_geometry(
            self.p1[0], self.p1[1], self.p1[2],
            self.p2[0], self.p2[1], self.p2[2],
            self.player_radius)
        self.prev_dist = dist
        hp_diff = self.p1[3] - self.p2[3]

        mw = min(self.p1[0], self.arena_size-self.p1[0],
                 self.p1[1], self.arena_size-self.p1[1])

        # ── 开火判定 ──────────────────────────────────────
        if action1[2] > 0.5 and self.p1[4] <= 0:
            self._fire(self.p1, 1)
            self.p1[4] = self.fire_cooldown
            
            bounce_threshold = getattr(RewardConfig, "BOUNCE_WALL_THRESHOLD", 60.0)
            if mw < bounce_threshold:
                # 靠墙反弹战术：允许背身开火（取正向和背向偏差中较小的一个）
                aim_err = min(angle_diff, abs(180.0 - angle_diff))
            else:
                aim_err = angle_diff
            
            # 动态距离判断：强化远距离射击的精准度与稳定性
            if dist > getattr(RewardConfig, "DISTANCE_THRESHOLD", 200.0):
                # 远程抽射模式：对瞄准要求极高，奖励更丰厚
                if aim_err <= 2.0:  
                    step_reward += RewardConfig.FIRE_PERFECT_AIM * 1.5
                    if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Aiming", RewardConfig.FIRE_PERFECT_AIM * 1.5)
                    if blind: 
                        step_reward += RewardConfig.FIRE_PERFECT_AIM_BLIND * 1.5
                        if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Aiming", RewardConfig.FIRE_PERFECT_AIM_BLIND * 1.5)
                elif aim_err <= 5.0:
                    r = (5.0-aim_err)/5.0 * RewardConfig.FIRE_GOOD_AIM_MAX * 1.5
                    step_reward += r
                    if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Aiming", r)
                else:
                    step_reward += RewardConfig.FIRE_BAD_AIM_PENALTY * 2.0
                    if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Aiming", RewardConfig.FIRE_BAD_AIM_PENALTY * 2.0)
            else:
                # 贴近近战模式：容错率稍高
                if aim_err <= 5.0:  
                    step_reward += RewardConfig.FIRE_PERFECT_AIM
                    if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Aiming", RewardConfig.FIRE_PERFECT_AIM)
                    if blind: 
                        step_reward += RewardConfig.FIRE_PERFECT_AIM_BLIND
                        if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Aiming", RewardConfig.FIRE_PERFECT_AIM_BLIND)
                elif aim_err <= 12.0:
                    r = (12.0-aim_err)/12.0 * RewardConfig.FIRE_GOOD_AIM_MAX
                    step_reward += r
                    if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Aiming", r)
                else:
                    step_reward += RewardConfig.FIRE_BAD_AIM_PENALTY
                    if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Aiming", RewardConfig.FIRE_BAD_AIM_PENALTY)

        if action2[2] > 0.5 and self.p2[4] <= 0:
            self._fire(self.p2, 2)
            self.p2[4] = self.fire_cooldown

        self.p1[4] = max(0, self.p1[4]-1)
        self.p2[4] = max(0, self.p2[4]-1)

        # ── 子弹物理 + 命中判定 ───────────────────────────
        new_bullets = []
        for b in self.bullets:
            b[0]+=b[2]; b[1]+=b[3]
            if b[0]<=0 or b[0]>=500: b[2]*=-1; b[5]+=1; b[0]=max(1,min(499,b[0]))
            if b[1]<=0 or b[1]>=500: b[3]*=-1; b[5]+=1; b[1]=max(1,min(499,b[1]))
            if b[5]>1: continue
            hit = False
            for p, pid in [(self.p1,1),(self.p2,2)]:
                if b[4]!=pid:
                    if math.hypot(b[0]-p[0],b[1]-p[1]) < self.player_radius:
                        atb = math.degrees(math.atan2(b[1]-p[1],b[0]-p[0]))%360
                        sc  = (p[2]+180)%360
                        df  = abs((atb-sc+180)%360-180)
                        if df > self.shield_half_angle:
                            p[3] -= 10
                            if b[4]==1 and pid==2: 
                                # 命中率提升：根据不同距离给予差异化额外奖励
                                current_dist = math.hypot(self.p2[0]-self.p1[0], self.p2[1]-self.p1[1])
                                hit_reward = RewardConfig.HIT_ENEMY
                                if current_dist > getattr(RewardConfig, "DISTANCE_THRESHOLD", 200.0):
                                    hit_reward += getattr(RewardConfig, "HIT_RATE_BONUS_LONG_RANGE", 150.0)
                                else:
                                    hit_reward += getattr(RewardConfig, "HIT_RATE_BONUS_CLOSE_RANGE", 50.0)
                                step_reward += hit_reward
                                if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Damage", hit_reward)
                            elif b[4]==2 and pid==1: 
                                step_reward += RewardConfig.GOT_HIT_PENALTY # 智能规避：重罚被击中
                                if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Damage", RewardConfig.GOT_HIT_PENALTY)
                        else:
                            # 护盾成功挡下子弹！
                            if b[4]==2 and pid==1: 
                                step_reward += RewardConfig.SHIELD_BLOCK  # 智能规避：用护盾接子弹重赏
                                if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Dodging", RewardConfig.SHIELD_BLOCK)
                        hit=True; break
            if not hit: new_bullets.append(b)
        self.bullets = new_bullets

        done = self.p1[3]<=0 or self.p2[3]<=0 or self.time>=self.game_duration

        # ── 每步塑形奖励 ──────────────────────────────────
        bounce_threshold = getattr(RewardConfig, "BOUNCE_WALL_THRESHOLD", 60.0)
        if mw < bounce_threshold:
            # 靠墙反弹战术：允许背身瞄准
            aim_err = min(angle_diff, abs(180.0 - angle_diff))
        else:
            aim_err = angle_diff
        
        # 积极瞄准策略核心：缩紧瞄准容差，强化"死盯"对手的奖励
        aim_f  = max(0.0, 1.0 - aim_err/30.0)  # 攻击准确：有效瞄准区缩小到30度
        
        # ==== 动态距离感知与策略切换 ====
        dist_threshold = getattr(RewardConfig, "DISTANCE_THRESHOLD", 200.0)
        if dist > dist_threshold:
            # 远程抽射攻击模式
            target_optimal_dist = getattr(RewardConfig, "LONG_RANGE_OPTIMAL_DIST", 350.0)
            dist_maintain_base  = getattr(RewardConfig, "LONG_RANGE_MAINTAIN_BASE", 2.0)
            too_close_penalty   = getattr(RewardConfig, "LONG_RANGE_TOO_CLOSE_PENALTY", -3.0)
            closing_in_pen      = getattr(RewardConfig, "LONG_RANGE_CLOSING_IN_PENALTY", -0.1)
        else:
            # 贴近近战攻击模式
            target_optimal_dist = getattr(RewardConfig, "CLOSE_COMBAT_OPTIMAL_DIST", 50.0)
            dist_maintain_base  = getattr(RewardConfig, "CLOSE_COMBAT_MAINTAIN_BASE", 1.0)
            too_close_penalty   = getattr(RewardConfig, "CLOSE_COMBAT_TOO_CLOSE_PENALTY", 0.0)
            closing_in_pen      = getattr(RewardConfig, "CLOSE_COMBAT_CLOSING_IN_REWARD", 0.5)

        dist_err = abs(dist - target_optimal_dist)
        dist_f = max(0.0, 1.0 - dist_err/200.0) # 距离当前模式的最佳距离越近，dist_f 越大
        sm     = 1.5 if blind else 1.0

        r_track = aim_f**2 * dist_f * RewardConfig.TRACK_AIM_WEIGHT * sm  # 诱导在最佳距离上强力追踪进攻
        step_reward += r_track
        if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Tracking", r_track)
        
        # 无视距离的绝对瞄准激励：只要准星对准（误差10度内），就给额外持续奖励
        if aim_err < 10.0:
            r_abs = (10.0 - aim_err) / 10.0 * RewardConfig.TRACK_ABSOLUTE_AIM
            step_reward += r_abs
            if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Tracking", r_abs)

        # 保持距离的基础奖励
        r_dist_maintain = dist_f * dist_maintain_base
        step_reward += r_dist_maintain
        if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Tracking", r_dist_maintain)
        
        # 太近惩罚：防止在远程模式下紧贴对手
        if dist < 150.0 and too_close_penalty < 0:
            step_reward += too_close_penalty
            if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Tracking", too_close_penalty)
        
        # 劣势时更远风筝
        if hp_diff < -20:
            d_err_disadv = abs(dist - 400.0) / 200.0
            r_dis = max(0.0, 1.0 - d_err_disadv) * RewardConfig.DISADVANTAGE_KITE
            step_reward += r_dis  # 劣势拉得更远
            if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Tracking", r_dis)

        # 接近/远离对手的惩罚或奖励
        closing = prev_dist - dist
        if closing > 0:
            r_close = closing * closing_in_pen
            step_reward += r_close
            if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Tracking", r_close)
        
        # ── 墙壁交互与反弹战术奖励/惩罚 ──
        if mw < bounce_threshold: 
            # 如果背对对手（角度差>135），视为正在执行墙壁反弹战术，给予奖励
            if angle_diff > 135.0:
                r_wall_bounce = ((bounce_threshold - mw) / bounce_threshold) * getattr(RewardConfig, "WALL_BOUNCE_REWARD", 2.0)
                step_reward += r_wall_bounce
                if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Tracking", r_wall_bounce)
            else:
                # 否则给与常规靠墙惩罚
                r_wall = - ((bounce_threshold - mw) / bounce_threshold) * RewardConfig.WALL_PENALTY
                step_reward += r_wall
                if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Tracking", r_wall)

        # ── 走位规避子弹奖励 ────────────────────────────────
        enemy_bullet_close = False
        incoming_b = None
        for b in self.bullets:
            if b[4] == 2:  # 敌方子弹
                if math.hypot(b[0]-self.p1[0], b[1]-self.p1[1]) < 150.0:
                    enemy_bullet_close = True
                    incoming_b = b
                    break
        if enemy_bullet_close:
            # 智能规避：面临攻击时保持强力机动
            if abs(action1[0]) > 0.5 or abs(action1[1]) > 0.5:
                step_reward += RewardConfig.EVASION_MOVE
                if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Dodging", RewardConfig.EVASION_MOVE)
            
            # 智能规避：将护盾对准袭来的子弹
            if incoming_b:
                atb = math.degrees(math.atan2(incoming_b[1]-self.p1[1], incoming_b[0]-self.p1[0])) % 360
                sc  = (self.p1[2] + 180) % 360
                df  = abs((atb - sc + 180) % 360 - 180)
                if df < self.shield_half_angle:
                    step_reward += RewardConfig.EVASION_SHIELD  # 重奖护盾防御姿态
                    if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Dodging", RewardConfig.EVASION_SHIELD)

        # 规避奖励机制：对方朝向不指向自己(ea > 30)，同时自己指向对方(angle_diff <= 10)
        if ea > 30.0 and angle_diff <= 10.0:
            step_reward += RewardConfig.TACTICAL_FLANK
            if analytics.ENABLE_ANALYTICS: analytics.analyzer.add_reward("Dodging", RewardConfig.TACTICAL_FLANK)

        # ── 终局结算 ──────────────────────────────────────
        if done:
            if analytics.ENABLE_ANALYTICS: analytics.analyzer.save_log()
            if self.p2[3] <= 0:
                tr = self.time / self.game_duration
                step_reward += RewardConfig.WIN_BASE * (1.0+(1.0-tr))
            elif self.p1[3] <= 0:
                step_reward += RewardConfig.LOSE_PENALTY
            elif self.time >= self.game_duration:
                hdf = self.p1[3]-self.p2[3]
                if hdf > 0:
                    step_reward += RewardConfig.TIME_OUT_WIN_BASE + hdf*RewardConfig.TIME_OUT_WIN_PER_HP     # 优势超时，只给小奖励（鼓励早点击杀）
                elif hdf < 0:
                    step_reward += RewardConfig.TIME_OUT_LOSE             # 劣势超时，视为失败
                else:
                    step_reward += RewardConfig.TIME_OUT_DRAW             # 完全平局，惩罚懈怠行为！

        return self._get_obs(), step_reward, done

    def _update_player(self, p, action, is_p1=False):
        mv,rt = float(action[0]),float(action[1])
        p[2]  = (p[2]+rt*self.player_rot_speed)%360
        rad   = math.radians(p[2])
        nx    = p[0]+math.cos(rad)*mv*self.player_speed
        ny    = p[1]+math.sin(rad)*mv*self.player_speed
        cx    = max(self.player_radius, min(500-self.player_radius, nx))
        cy    = max(self.player_radius, min(500-self.player_radius, ny))
        sd    = math.hypot(cx-p[0],cy-p[1])
        if is_p1: self.p1_dist+=sd
        else:     self.p2_dist+=sd
        p[0]=cx; p[1]=cy

    def _fire(self,p,owner):
        rad = math.radians(p[2])
        self.bullets.append([
            p[0], p[1],  # 解决穿模：子弹严格从球中心生成，不再向外偏移
            math.cos(rad)*self.bullet_speed,
            math.sin(rad)*self.bullet_speed, owner, 0
        ])

    def _resolve_collision(self):
        """碰撞检测与分离：两个玩家重叠时沿法线方向推开"""
        dx   = self.p2[0] - self.p1[0]
        dy   = self.p2[1] - self.p1[1]
        dist = math.hypot(dx, dy)
        min_dist = self.player_radius * 2   # 两圆半径之和

        if 0 < dist < min_dist:
            overlap = min_dist - dist
            nx = dx / dist          # 法线方向（p1→p2）
            ny = dy / dist
            push = overlap / 2.0   # 各推一半

            self.p1[0] -= nx * push
            self.p1[1] -= ny * push
            self.p2[0] += nx * push
            self.p2[1] += ny * push

            # 夹回场地边界
            r = self.player_radius
            for p in (self.p1, self.p2):
                p[0] = max(r, min(self.arena_size - r, p[0]))
                p[1] = max(r, min(self.arena_size - r, p[1]))
        elif dist == 0:
            # 完全重叠时随机分离
            self.p1[0] -= self.player_radius
            self.p2[0] += self.player_radius
