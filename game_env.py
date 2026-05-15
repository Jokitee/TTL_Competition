import numpy as np
import math
import random
from reward_config import RewardConfig
import analytics

# ============================================================
# Numba JIT 加速
# ============================================================
try:
    from numba import njit
    NUMBA = True
except ImportError:
    def njit(*a, **kw): return lambda f: f
    NUMBA = False

@njit(cache=True)
def _step_geometry(p1x, p1y, p1a, p2x, p2y, p2a, player_radius):
    dx, dy = p2x - p1x, p2y - p1y
    dist = math.sqrt(dx*dx + dy*dy)
    tgt  = math.degrees(math.atan2(dy, dx)) % 360.0
    angle_diff = abs((tgt - p1a + 180.0) % 360.0 - 180.0)
    etm  = math.degrees(math.atan2(-dy, -dx)) % 360.0
    ea   = abs((etm - p2a + 180.0) % 360.0 - 180.0)
    dc   = (math.degrees(math.asin(min(1.0, player_radius/dist))) if dist > player_radius else 90.0)
    exposure = max(0.0, 1.0 - ea / max(dc, 1e-6))
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
        self.game_duration    = max_steps
        self.gallery_mode     = False
        self.fixed_ai         = False
        self.current_stage    = 0   # 随训练阶段逐步增大开火惩罚
        self.reset()

    def set_mode(self, gallery=False, fixed=False):
        self.gallery_mode = gallery
        self.fixed_ai = fixed

    def set_stage(self, stage: int):
        """by training stage to scale fire penalty progressively"""
        self.current_stage = stage

    def reset(self, stage=3):
        if self.fixed_ai:
            self.p1 = [250.0, 250.0, random.uniform(0,360), 100.0, 0.0]
        else:
            self.p1 = [random.uniform(50,450), random.uniform(50,450), random.uniform(0,360), 100.0, 0.0]

        ranges = [(50, 150), (150, 300), (300, 500), (50, 550)]
        lo, hi = ranges[min(stage, 3)]
        self._respawn_target(lo, hi)
        self.bullets = []
        self.time = 0
        self.prev_dist = math.hypot(self.p2[0]-self.p1[0], self.p2[1]-self.p1[1])
        self.prev_obs_dist = self.prev_dist
        self.prev_aim_err  = 180.0   # 用于计算准心误差变化率
        self.tracking_frames = 0     # 连续精确追踪帧计数
        return self._get_obs()

    def _respawn_target(self, lo=50, hi=550):
        for _ in range(100):
            dist = random.uniform(lo, hi)
            angle = random.uniform(0, 2*math.pi)
            tx, ty = self.p1[0] + math.cos(angle)*dist, self.p1[1] + math.sin(angle)*dist
            if 20 <= tx <= 480 and 20 <= ty <= 480:
                self.p2 = [tx, ty, random.uniform(0, 360), 100.0, 0.0]
                break
        else: self.p2 = [400.0, 400.0, random.uniform(0,360), 100.0, 0.0]

    def get_relative_obs(self, p_self, p_enemy):
        dx, dy = p_enemy[0] - p_self[0], p_enemy[1] - p_self[1]
        dist = math.hypot(dx, dy)
        tgt  = math.degrees(math.atan2(dy, dx)) % 360
        angle_diff = (tgt - p_self[2] + 180) % 360 - 180
        etm  = math.degrees(math.atan2(-dy, -dx)) % 360
        ea   = abs((etm - p_enemy[2] + 180) % 360 - 180)
        dc   = (math.degrees(math.asin(min(1.0, self.player_radius/dist))) if dist > self.player_radius else 90.0)
        exposure = max(0.0, 1.0 - ea / max(dc, 1e-6))
        mw   = min(p_self[0], self.arena_size-p_self[0], p_self[1], self.arena_size-p_self[1])
        wall = mw / 250.0
        rs, re = math.radians(p_self[2]), math.radians(p_enemy[2])
        cd   = p_self[4] / self.fire_cooldown
        enemy_rad = math.radians(p_enemy[2])
        evx, evy = math.cos(enemy_rad) * self.player_speed, math.sin(enemy_rad) * self.player_speed
        if dist > 1e-6:
            vel_along = (evx * dx + evy * dy) / dist
            vel_perp  = (-evx * dy + evy * dx) / dist
        else: vel_along = vel_perp = 0.0
        dist_rate    = (dist - self.prev_obs_dist) / 6.0
        aim_err_abs  = abs(angle_diff)
        # 准心误差变化率：正值=在偏离，负值=在收敛（对精确锁定至关重要）
        aim_err_delta = (aim_err_abs - self.prev_aim_err) / 90.0
        self.prev_obs_dist = dist
        self.prev_aim_err  = aim_err_abs
        return np.array([
            dist / 707.0, angle_diff / 180.0, dx / 500.0, dy / 500.0,
            p_self[3]  / 100.0, p_enemy[3] / 100.0, exposure, wall,
            math.sin(rs), math.cos(rs), math.sin(re), math.cos(re),
            cd, vel_along / 3.0, dist_rate, vel_perp / 3.0
        ], dtype=np.float32)

    def _get_obs(self): return self.get_relative_obs(self.p1, self.p2)

    def step(self, action1, action2):
        self.time += 1
        prev_dist = self.prev_dist
        self._update_player(self.p1, action1, is_p1=True)
        if not self.gallery_mode: self._update_player(self.p2, action2, is_p1=False)
        self._resolve_collision()

        step_reward = 0.0
        dist, angle_diff, exposure, blind, dx, dy, ea = _step_geometry(
            self.p1[0], self.p1[1], self.p1[2], self.p2[0], self.p2[1], self.p2[2], self.player_radius)
        self.prev_dist = dist
        hp_diff = self.p1[3] - self.p2[3]
        mw = min(self.p1[0], self.arena_size-self.p1[0], self.p1[1], self.arena_size-self.p1[1])
        
        # 综合瞄准误差计算 (考虑墙壁反弹战术)
        bounce_threshold = getattr(RewardConfig, "BOUNCE_WALL_THRESHOLD", 60.0)
        aim_err = min(angle_diff, abs(180.0 - angle_diff)) if mw < bounce_threshold else angle_diff

        # 计算预判瞄准误差 (Lead Aim)
        enemy_rad = math.radians(self.p2[2])
        evx, evy = math.cos(enemy_rad) * self.player_speed, math.sin(enemy_rad) * self.player_speed
        if dist > 1e-6:
            t_int = dist / self.bullet_speed
            pred_x, pred_y = self.p2[0] + evx * t_int, self.p2[1] + evy * t_int
            lead_tgt = math.degrees(math.atan2(pred_y - self.p1[1], pred_x - self.p1[0])) % 360.0
            lead_err = abs((lead_tgt - self.p1[2] + 180.0) % 360.0 - 180.0)
            vp_abs = abs(-evx * dy + evy * dx) / dist
            lead_w = min(1.0, vp_abs / self.player_speed) * min(1.0, dist / 300.0)
        else:
            lead_err, lead_w = aim_err, 0.0
        
        effective_aim_err = (1.0 - lead_w) * aim_err + lead_w * lead_err

        if action1[2] > 0.5 and self.p1[4] <= 0:
            self._fire(self.p1, 1)
            self.p1[4] = self.fire_cooldown
            if analytics.ENABLE_ANALYTICS: analytics.analyzer.log_shot(dist, hit=False)
            if not blind: step_reward += RewardConfig.FIRE_ON_SHIELD_PENALTY
            
            # 【指数型精准开火奖励】误差越小奖励越巨额，形成强烈的收敛压力
            if effective_aim_err <= 3.0:
                # 完美准心！顶级奖励
                step_reward += RewardConfig.FIRE_PERFECT_AIM * (1.5 if blind else 1.0)
                self.tracking_frames = 0  # 开火后重置连续计数
            elif effective_aim_err <= 10.0:
                step_reward += RewardConfig.FIRE_GOOD_AIM_MAX
                self.tracking_frames = 0
            elif effective_aim_err <= 20.0:
                step_reward += RewardConfig.FIRE_MEDIOCRE_AIM
            else:
                # 阶段性惩罚缩放：早期鼓励尝试，后期强制精准纪律
                # Stage 0=无惩罚, Stage 1=25%, Stage 2=60%, Stage 3+=100%
                stage_penalty_scale = [0.0, 0.25, 0.6, 1.0]
                scale = stage_penalty_scale[min(self.current_stage, 3)]
                step_reward += RewardConfig.FIRE_BAD_AIM_PENALTY * scale

        if not self.gallery_mode and action2[2] > 0.5 and self.p2[4] <= 0:
            self._fire(self.p2, 2); self.p2[4] = self.fire_cooldown

        self.p1[4] = max(0, self.p1[4]-1); self.p2[4] = max(0, self.p2[4]-1)

        new_bullets = []
        for b in self.bullets:
            b[0]+=b[2]; b[1]+=b[3]
            if b[0]<=0 or b[0]>=500 or b[1]<=0 or b[1]>=500:
                if b[0]<=0 or b[0]>=500: b[2]*=-1; b[0]=max(1,min(499,b[0]))
                if b[1]<=0 or b[1]>=500: b[3]*=-1; b[1]=max(1,min(499,b[1]))
                b[5]+=1
            if b[5]>1: continue
            hit = False
            for p, pid in [(self.p1,1),(self.p2,2)]:
                if b[4]!=pid and math.hypot(b[0]-p[0],b[1]-p[1]) < self.player_radius:
                    atb = math.degrees(math.atan2(b[1]-p[1],b[0]-p[0]))%360
                    df  = abs((atb - (p[2]+180)%360 + 180)%360-180)
                    if df > self.shield_half_angle:
                        p[3] -= 10
                        if b[4]==1:
                            # 命中对方肉身——最核心的进攻奖励
                            step_reward += RewardConfig.HIT_ENEMY + (RewardConfig.HIT_BLIND_SIDE_BONUS if blind else 0.0)
                            if analytics.ENABLE_ANALYTICS: analytics.analyzer.log_shot(dist, hit=True)
                            if self.gallery_mode: self._respawn_target()
                        # 被击中：GOT_HIT_PENALTY=0，猎手不在乎被打
                    hit=True; break
                else:
                    hit=True; break
            if not hit: new_bullets.append(b)
        self.bullets = new_bullets

        # ── 猎手模式：纯进攻奖励 ───────────────────────────────────

        # 1. 持续追踪奖励（枪口始终对准目标）
        if aim_err <= 3.0:
            self.tracking_frames += 1
            step_reward += RewardConfig.TRACK_PERFECT_REWARD * min(self.tracking_frames, 30)
        elif aim_err <= 10.0:
            self.tracking_frames = max(0, self.tracking_frames - 1)
            step_reward += RewardConfig.TRACK_GOOD_REWARD
        else:
            self.tracking_frames = 0

        # 2. 瞄准权重叠加分
        aim_f = max(0.0, 1.0 - aim_err / 30.0)
        step_reward += aim_f**2 * RewardConfig.TRACK_AIM_WEIGHT * (1.2 if blind else 1.0)

        # 3. 主动接近奖励（冲向对手，进入有效射程）
        optimal = RewardConfig.APPROACH_OPTIMAL_DIST
        stop    = RewardConfig.APPROACH_STOP_DIST
        if dist > stop:
            dist_change = prev_dist - dist   # 正值=正在靠近
            if dist_change > 0:
                proximity_bonus = max(0.0, 1.0 - abs(dist - optimal) / optimal)
                step_reward += dist_change * RewardConfig.APPROACH_REWARD_SCALE * (1.0 + proximity_bonus)

        done = self.p1[3] <= 0 or self.p2[3] <= 0 or self.time >= self.game_duration

        # 胜负结算（快速击杀有额外奖励）
        if done and not self.gallery_mode:
            if self.p2[3] <= 0:
                time_ratio = 1.0 - self.time / self.game_duration
                step_reward += RewardConfig.WIN_BASE * (1.0 + time_ratio)
            elif self.p1[3] <= 0:
                step_reward += RewardConfig.LOSE_PENALTY  # 死亡惩罚很小，猎手不怕死

        return self._get_obs(), step_reward, done
        
    def _update_player(self, p, action, is_p1=False):
        if is_p1 and self.fixed_ai: return
        mv, rt = float(action[0]), float(action[1])
        p[2] = (p[2] + rt * self.player_rot_speed) % 360
        rad = math.radians(p[2])
        nx, ny = p[0] + math.cos(rad)*mv*self.player_speed, p[1] + math.sin(rad)*mv*self.player_speed
        p[0], p[1] = max(20, min(480, nx)), max(20, min(480, ny))

    def _fire(self, p, owner):
        rad = math.radians(p[2])
        self.bullets.append([p[0], p[1], math.cos(rad)*self.bullet_speed, math.sin(rad)*self.bullet_speed, owner, 0])

    def _resolve_collision(self):
        dx, dy = self.p2[0]-self.p1[0], self.p2[1]-self.p1[1]; dist = math.hypot(dx, dy)
        if 0 < dist < 40:
            nx, ny = dx/dist, dy/dist; push = (40 - dist)/2.0
            self.p1[0]-=nx*push; self.p1[1]-=ny*push
            self.p2[0]+=nx*push; self.p2[1]+=ny*push
