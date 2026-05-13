"""
train_ppo.py  —  PPO + 自博弈训练
架构: 13-48-32-3 (Actor-Critic 共享骨干网络)
导出权重与 nn_inference.c 完全兼容
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal, Bernoulli
from game_env import LinkCombatEnv
import copy, pickle, time, math

# ─── 设备自动检测 ───────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", end='')
if device.type == 'cuda':
    print(f"  ({torch.cuda.get_device_name(0)})")
else:
    print("  (安装 CUDA 版 PyTorch 可启用 GPU 加速)")

# ─── 超参数 ──────────────────────────────────────────────────────
IN_DIM  = 13
H1, H2  = 48, 32

LR          = 3e-4
GAMMA       = 0.99
GAE_LAM     = 0.95
CLIP_EPS    = 0.2
VF_COEF     = 0.5
ENT_COEF    = 0.001   # ↓ 从0.01降到0.001，防止策略变得过于随机
MAX_GRAD    = 0.5

N_ENVS      = 16        # 并行环境数量
N_STEPS     = 512       # 每次收集步数（总经验 = N_ENVS × N_STEPS = 8192）
N_EPOCHS    = 8         # 每批经验的 PPO 更新轮数
BATCH_SIZE  = 256       # minibatch 大小

TOTAL_STEPS         = 10_000_000   # 总训练步数
OPP_UPDATE_STEPS    = 100_000      # 每隔 N 步将当前策略加入对手池
OPP_POOL_MAX        = 20           # 对手池最大容量

LOG_EVERY   = 10    # 每 N 次更新打印日志
SAVE_EVERY  = 50    # 每 N 次更新保存模型

SAVE_PATH   = "E:\\Test_FIre\\best_model_ppo.pkl"


# ════════════════════════════════════════════════════════════════
# Actor-Critic 网络
# 骨干: Linear(13→48)→Tanh→Linear(48→32)→Tanh
# Actor头: mv_mean, rt_mean(连续), fire_logit(二值)
# Critic头: value(标量)
# 导出兼容性: W1/B1/W2/B2/W3/B3 与 nn_inference.c 完全对应
# ════════════════════════════════════════════════════════════════
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # 共享骨干
        self.backbone = nn.Sequential(
            nn.Linear(IN_DIM, H1), nn.Tanh(),
            nn.Linear(H1, H2),    nn.Tanh(),
        )
        # Actor 头（连续动作）
        self.mv_head   = nn.Linear(H2, 1)
        self.rt_head   = nn.Linear(H2, 1)
        # Actor 头（离散开火）
        self.fire_head = nn.Linear(H2, 1)
        # 可学习的动作标准差（log）
        self.log_std   = nn.Parameter(torch.tensor([-0.5, -0.5]))
        # log_std 约束在 [-2, 0]，即 std 在 [0.13, 1.0]
        # 防止策略变得过度随机（std 爆炸问题）
        # Critic 头
        self.value_head = nn.Linear(H2, 1)

        # 正交初始化（PPO 经典初始化方案）
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        for head in [self.mv_head, self.rt_head, self.fire_head]:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def _features(self, obs):
        return self.backbone(obs)

    def get_value(self, obs):
        return self.value_head(self._features(obs)).squeeze(-1)

    def _distributions(self, feats):
        # 约束 std 范围：[-3, -1] → std ∈ [0.05, 0.36]
        # 防止 Std 炸到 1.0 导致模型变成只会乱转的无头苍蝇（进而不敢开火）
        std = self.log_std.clamp(-3.0, -1.0).exp()
        mv_dist   = Normal(torch.tanh(self.mv_head(feats)),   std[0])
        rt_dist   = Normal(torch.tanh(self.rt_head(feats)),   std[1])
        fire_dist = Bernoulli(logits=self.fire_head(feats))
        return mv_dist, rt_dist, fire_dist

    def act(self, obs):
        """采样动作（训练时使用）"""
        feats = self._features(obs)
        mv_d, rt_d, fire_d = self._distributions(feats)
        mv   = mv_d.sample().clamp(-1, 1)
        rt   = rt_d.sample().clamp(-1, 1)
        fire = fire_d.sample()
        logp = (mv_d.log_prob(mv) +
                rt_d.log_prob(rt) +
                fire_d.log_prob(fire)).squeeze(-1)
        value = self.value_head(feats).squeeze(-1)
        return mv.squeeze(-1), rt.squeeze(-1), fire.squeeze(-1), logp, value

    def evaluate(self, obs, mv, rt, fire):
        """计算已存储动作的 log_prob、entropy、value（更新时使用）"""
        feats = self._features(obs)
        mv_d, rt_d, fire_d = self._distributions(feats)
        logp = (mv_d.log_prob(mv.unsqueeze(-1)) +
                rt_d.log_prob(rt.unsqueeze(-1)) +
                fire_d.log_prob(fire.unsqueeze(-1))).squeeze(-1)
        entropy = (mv_d.entropy() +
                   rt_d.entropy() +
                   fire_d.entropy()).squeeze(-1)
        value = self.value_head(feats).squeeze(-1)
        return logp, entropy, value

    def act_deterministic(self, obs_np):
        """确定性动作（用于对手推理 / 最终部署）"""
        with torch.no_grad():
            obs = torch.FloatTensor(obs_np).unsqueeze(0).to(device)
            feats = self._features(obs)
            mv   = torch.tanh(self.mv_head(feats)).item()
            rt   = torch.tanh(self.rt_head(feats)).item()
            fire = 1.0 if self.fire_head(feats).item() > 0 else 0.0
        return np.array([mv, rt, fire], dtype=np.float32)

    def export_weights(self):
        """
        导出兼容 nn_inference.c 格式的权重字典:
        W1(13×48) B1(48) W2(48×32) B2(32) W3(32×3) B3(3)
        W3 由三个 Actor 头拼接而成
        """
        W1 = self.backbone[0].weight.detach().cpu().numpy().T    # (13,48)
        B1 = self.backbone[0].bias.detach().cpu().numpy()        # (48,)
        W2 = self.backbone[2].weight.detach().cpu().numpy().T    # (48,32)
        B2 = self.backbone[2].bias.detach().cpu().numpy()        # (32,)
        # 拼接三个头 → W3(32,3) B3(3)
        W3 = np.concatenate([
            self.mv_head.weight.detach().cpu().numpy().T,        # (32,1)
            self.rt_head.weight.detach().cpu().numpy().T,        # (32,1)
            self.fire_head.weight.detach().cpu().numpy().T,      # (32,1)
        ], axis=1)                                               # (32,3)
        B3 = np.array([
            self.mv_head.bias.item(),
            self.rt_head.bias.item(),
            self.fire_head.bias.item(),
        ])
        return [W1, B1, W2, B2, W3, B3]


# ════════════════════════════════════════════════════════════════
# 奖励归一化（Running Mean/Std）
# 将奖励缩放到均值≈0、方差≈1，解决 vf loss 过高问题
# ════════════════════════════════════════════════════════════════
class RunningStats:
    """Welford 在线算法，跟踪奖励的均值和方差"""
    def __init__(self):
        self.n    = 0
        self.mean = 0.0
        self.M2   = 1.0   # 初始方差估计

    def update_and_normalize(self, rewards: np.ndarray) -> np.ndarray:
        for r in rewards.flatten():
            self.n += 1
            delta      = r - self.mean
            self.mean += delta / self.n
            delta2     = r - self.mean
            self.M2   += delta * delta2
        std = math.sqrt(self.M2 / max(self.n, 1)) + 1e-8
        return rewards / std


# 全局奖励归一化实例（跨 rollout 持续更新）
reward_stats = RunningStats()


# ════════════════════════════════════════════════════════════════
# 广义优势估计（GAE）
# ════════════════════════════════════════════════════════════════
def compute_gae(rewards, values, dones, last_value, gamma=GAMMA, lam=GAE_LAM):
    """
    rewards: (T, N_ENVS)
    values:  (T, N_ENVS)
    dones:   (T, N_ENVS)
    last_value: (N_ENVS,)
    returns: advantages (T×N_ENVS,), returns (T×N_ENVS,)
    """
    T, N = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(N)
    for t in reversed(range(T)):
        next_val = last_value if t == T-1 else values[t+1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages.flatten(), returns.flatten()


# ════════════════════════════════════════════════════════════════
# PPO 更新
# ════════════════════════════════════════════════════════════════
def ppo_update(policy, optimizer, batch):
    obs_b, mv_b, rt_b, fire_b, logp_old_b, adv_b, ret_b = [
        torch.FloatTensor(x).to(device) for x in batch
    ]
    # 标准化优势（降低方差）
    adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

    total_loss = pg_loss = vf_loss = ent_loss = 0.0
    n_batches = 0
    idx = np.arange(len(obs_b))

    for _ in range(N_EPOCHS):
        np.random.shuffle(idx)
        for start in range(0, len(idx), BATCH_SIZE):
            mb = idx[start:start+BATCH_SIZE]
            logp, entropy, value = policy.evaluate(
                obs_b[mb], mv_b[mb], rt_b[mb], fire_b[mb])

            ratio = (logp - logp_old_b[mb]).exp()
            pg1   = ratio * adv_b[mb]
            pg2   = ratio.clamp(1-CLIP_EPS, 1+CLIP_EPS) * adv_b[mb]
            l_pg  = -torch.min(pg1, pg2).mean()
            l_vf  =  0.5 * (value - ret_b[mb]).pow(2).mean()
            l_ent = -entropy.mean()

            loss = l_pg + VF_COEF * l_vf + ENT_COEF * l_ent
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD)
            optimizer.step()

            total_loss += loss.item()
            pg_loss    += l_pg.item()
            vf_loss    += l_vf.item()
            ent_loss   += l_ent.item()
            n_batches  += 1

    n = max(n_batches, 1)
    return total_loss/n, pg_loss/n, vf_loss/n, ent_loss/n


# ════════════════════════════════════════════════════════════════
# 经验收集（N_ENVS 并行环境）
# ════════════════════════════════════════════════════════════════
def collect_rollouts(policy, envs, obs_list, opponent_fn):
    """
    采集 N_STEPS × N_ENVS 步经验。
    opponent_fn(obs_np) → action_np，使用对手策略。
    """
    N = len(envs)
    T = N_STEPS

    all_obs   = np.zeros((T, N, IN_DIM), dtype=np.float32)
    all_mv    = np.zeros((T, N), dtype=np.float32)
    all_rt    = np.zeros((T, N), dtype=np.float32)
    all_fire  = np.zeros((T, N), dtype=np.float32)
    all_logp  = np.zeros((T, N), dtype=np.float32)
    all_val   = np.zeros((T, N), dtype=np.float32)
    all_rew   = np.zeros((T, N), dtype=np.float32)
    all_done  = np.zeros((T, N), dtype=np.float32)

    episode_rewards = []
    ep_buf = np.zeros(N, dtype=np.float32)

    for t in range(T):
        obs_tensor = torch.FloatTensor(np.stack(obs_list)).to(device)
        with torch.no_grad():
            mv, rt, fire, logp, value = policy.act(obs_tensor)

        mv_np   = mv.cpu().numpy()
        rt_np   = rt.cpu().numpy()
        fire_np = fire.cpu().numpy()
        logp_np = logp.cpu().numpy()
        val_np  = value.cpu().numpy()

        for i, env in enumerate(envs):
            obs2    = env.get_relative_obs(env.p2, env.p1)
            action2 = opponent_fn(obs2)
            action1 = np.array([mv_np[i], rt_np[i], fire_np[i]], dtype=np.float32)
            next_obs, reward, done = env.step(action1, action2)

            all_obs[t, i]  = obs_list[i]
            all_mv[t, i]   = mv_np[i]
            all_rt[t, i]   = rt_np[i]
            all_fire[t, i] = fire_np[i]
            all_logp[t, i] = logp_np[i]
            all_val[t, i]  = val_np[i]
            all_rew[t, i]  = reward
            all_done[t, i] = float(done)
            ep_buf[i]      += reward

            if done:
                episode_rewards.append(ep_buf[i])
                ep_buf[i] = 0.0
                obs_list[i] = env.reset()
            else:
                obs_list[i] = next_obs

    # 奖励归一化（解决 vf loss 过高问题，将奖励缩放到合理范围）
    all_rew = reward_stats.update_and_normalize(all_rew)

    # 最后一步的 bootstrap value
    obs_tensor = torch.FloatTensor(np.stack(obs_list)).to(device)
    with torch.no_grad():
        last_val = policy.get_value(obs_tensor).cpu().numpy()

    advantages, returns = compute_gae(all_rew, all_val, all_done, last_val)

    batch = (
        all_obs.reshape(-1, IN_DIM),
        all_mv.flatten(),
        all_rt.flatten(),
        all_fire.flatten(),
        all_logp.flatten(),
        advantages,
        returns,
    )
    mean_ep_rew = np.mean(episode_rewards) if episode_rewards else 0.0
    return batch, obs_list, mean_ep_rew


# ════════════════════════════════════════════════════════════════
# 主训练循环
# ════════════════════════════════════════════════════════════════
def train(mode_name="default", max_steps=4000000):
    policy    = ActorCritic().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LR, eps=1e-5)

    # 线性学习率衰减
    total_updates = max_steps // (N_ENVS * N_STEPS)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1,
        total_iters=total_updates)

    # 对手池：初始为当前策略的拷贝
    opponent_pool = [copy.deepcopy(policy)]
    opponent_pool[-1].eval()

    def sample_opponent_fn():
        """从对手池随机选一个，近期权重更高"""
        n = len(opponent_pool)
        weights = np.linspace(1.0, 3.0, n); weights /= weights.sum()
        opp = opponent_pool[np.random.choice(n, p=weights)]
        def fn(obs_np):
            return opp.act_deterministic(obs_np)
        return fn

    # 初始化环境
    envs     = [LinkCombatEnv() for _ in range(N_ENVS)]
    obs_list = [env.reset() for env in envs]

    total_steps  = 0
    update_count = 0
    best_reward  = -float('inf')
    t0 = time.time()

    n_params = sum(p.numel() for p in policy.parameters())
    print("=" * 65)
    print(f"PPO 超强人机训练 v1.0")
    print(f"  网络: {IN_DIM}-{H1}-{H2}-3  参数量: {n_params}")
    print(f"  N_ENVS={N_ENVS}  N_STEPS={N_STEPS}  总批量={N_ENVS*N_STEPS}/次更新")
    print(f"  模式: {mode_name} | 目标步数: {max_steps:,}")
    print("=" * 65)

    opp_fn = sample_opponent_fn()

    while total_steps < max_steps:
        policy.train()
        batch, obs_list, ep_rew = collect_rollouts(policy, envs, obs_list, opp_fn)
        policy.train()

        loss, pg, vf, ent = ppo_update(policy, optimizer, batch)
        scheduler.step()

        total_steps  += N_ENVS * N_STEPS
        update_count += 1

        # 更新对手池
        if total_steps % OPP_UPDATE_STEPS < N_ENVS * N_STEPS:
            snap = copy.deepcopy(policy); snap.eval()
            opponent_pool.append(snap)
            if len(opponent_pool) > OPP_POOL_MAX:
                opponent_pool.pop(0)
            opp_fn = sample_opponent_fn()   # 重新采样

        # 日志
        if update_count % LOG_EVERY == 0:
            elapsed  = time.time() - t0
            sps      = total_steps / elapsed
            lr_now   = optimizer.param_groups[0]['lr']
            std_now  = policy.log_std.exp().detach().cpu().numpy()
            print(f"Steps {total_steps:>8,} | "
                  f"Reward {ep_rew:>8.1f} | "
                  f"Loss {loss:>6.3f} (pg={pg:.3f} vf={vf:.3f} ent={ent:.3f}) | "
                  f"Std [{std_now[0]:.2f},{std_now[1]:.2f}] | "
                  f"LR {lr_now:.2e} | "
                  f"Pool {len(opponent_pool)} | "
                  f"{sps:.0f}sps")

        # 保存
        if update_count % SAVE_EVERY == 0 or ep_rew > best_reward:
            if ep_rew > best_reward:
                best_reward = ep_rew
            weights = policy.export_weights()
            save_path = f"E:\\Test_FIre\\best_model_ppo_{mode_name}.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(weights, f)

    # 最终保存
    weights = policy.export_weights()
    save_path = f"E:\\Test_FIre\\best_model_ppo_{mode_name}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(weights, f)
    print(f"\n[{mode_name}] 训练完成！最佳奖励: {best_reward:.1f}")
    print(f"权重已保存至 {save_path}")
    print(f"运行 export_to_c.py 生成 model_weights.h（替换 SAVE_PATH 为 {save_path}）")


if __name__ == "__main__":
    from reward_config import RewardConfig
    
    print(f"\n\n{'='*20} 开始训练: 动态距离感知攻击专用模型 {'='*20}")
    
    # 动态修改奖励配置(如果需要，确保BOUNCE_EXPLOIT关闭)
    setattr(RewardConfig, "BOUNCE_EXPLOIT", False)
    
    # 重置全局奖励归一化
    reward_stats = RunningStats()
    
    # 将模型精简为单一的动态攻击专用模型
    train(mode_name="dynamic_attacker", max_steps=10_000_000)
