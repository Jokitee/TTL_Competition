"""
train_ppo.py  —  PPO + 课程学习打靶特训 v4.0 (极简高效网络版)
架构优化: 16-64-64-7 (摒弃冗余的 48-32 层，使用强化学习标准双层宽网络，加速收敛)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical, Bernoulli
from game_env import LinkCombatEnv
import copy, pickle, time, math, os, random
import analytics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

IN_DIM  = 16
H1, H2  = 64, 64  # 优化网络：去除多余的 H3，使用经典的双层 64
LR          = 3e-4
GAMMA       = 0.99
GAE_LAM     = 0.95
CLIP_EPS    = 0.2
VF_COEF     = 0.5
ENT_COEF    = 0.05  # 提高探索率，防止 Stage 0/1 陷入「永远不开枪」局部最优
MAX_GRAD    = 0.5
N_ENVS      = 16
N_STEPS     = 512
BATCH_SIZE  = 256
N_EPOCHS    = 10

MV_MAP = torch.tensor([0.0, 1.0, -1.0])
RT_MAP = torch.tensor([0.0, 1.0, -1.0])

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 共享感知层 (Shared Perception): 提取环境的通用几何与物理特征
        self.shared = nn.Sequential(
            nn.Linear(IN_DIM, 64), nn.Tanh()
        )
        
        # 2. 进攻火控分支 (Offensive Branch): 专精炮塔控制与击杀时机判断
        self.offense_branch = nn.Sequential(
            nn.Linear(64, 48), nn.Tanh()
        )
        self.rt_head   = nn.Linear(48, 3)  # 旋转炮塔
        self.fire_head = nn.Linear(48, 1)  # 开火决策
        
        # 3. 战术走位分支 (Tactical Branch): 专精底盘机动、绕后侧翼与规避
        self.tactical_branch = nn.Sequential(
            nn.Linear(64, 48), nn.Tanh()
        )
        self.mv_head   = nn.Linear(48, 3)  # 移动控制
        
        # 4. 局势评估分支 (Critic Branch): 预测当前状态的胜率期望
        self.critic_branch = nn.Sequential(
            nn.Linear(64, 48), nn.Tanh(),
            nn.Linear(48, 1)
        )

    def _distributions(self, obs):
        s_feat = self.shared(obs)
        o_feat = self.offense_branch(s_feat)
        t_feat = self.tactical_branch(s_feat)
        
        mv_dist   = Categorical(logits=self.mv_head(t_feat))
        rt_dist   = Categorical(logits=self.rt_head(o_feat))
        fire_dist = Bernoulli(logits=self.fire_head(o_feat).squeeze(-1))
        return mv_dist, rt_dist, fire_dist, s_feat

    def get_value(self, obs): 
        return self.critic_branch(self.shared(obs)).squeeze(-1)

    def act(self, obs):
        mv_d, rt_d, fire_d, s_feat = self._distributions(obs)
        mv_idx = mv_d.sample().squeeze(-1)
        rt_idx = rt_d.sample().squeeze(-1)
        fire   = fire_d.sample().squeeze(-1)
        logp = (mv_d.log_prob(mv_idx) + rt_d.log_prob(rt_idx) + fire_d.log_prob(fire))
        value = self.critic_branch(s_feat).squeeze(-1)
        _mv_map, _rt_map = MV_MAP.to(obs.device), RT_MAP.to(obs.device)
        return _mv_map[mv_idx], _rt_map[rt_idx], fire, mv_idx, rt_idx, logp, value

    def evaluate(self, obs, mv_idx, rt_idx, fire):
        mv_d, rt_d, fire_d, s_feat = self._distributions(obs)
        logp = (mv_d.log_prob(mv_idx) + rt_d.log_prob(rt_idx) + fire_d.log_prob(fire))
        entropy = (mv_d.entropy() + rt_d.entropy() + fire_d.entropy())
        value = self.critic_branch(s_feat).squeeze(-1)
        return logp, entropy, value

    def act_deterministic(self, obs_np):
        with torch.no_grad():
            obs = torch.FloatTensor(obs_np).unsqueeze(0).to(device)
            s_feat = self.shared(obs)
            o_feat = self.offense_branch(s_feat)
            t_feat = self.tactical_branch(s_feat)
            
            mv_idx = self.mv_head(t_feat).argmax(dim=-1).item()
            rt_idx = self.rt_head(o_feat).argmax(dim=-1).item()
            fire   = 1.0 if self.fire_head(o_feat).item() > 0 else 0.0
        return np.array([MV_MAP[mv_idx].item(), RT_MAP[rt_idx].item(), fire], dtype=np.float32)

    def export_weights(self):
        # 导出为扁平化列表，方便 C 语言加载
        weights = []
        for seq in [self.shared, self.offense_branch, self.tactical_branch]:
            weights.append(seq[0].weight.detach().cpu().numpy().T)
            weights.append(seq[0].bias.detach().cpu().numpy())
        
        # 导出输出头
        weights.append(self.mv_head.weight.detach().cpu().numpy().T)
        weights.append(self.mv_head.bias.detach().cpu().numpy())
        weights.append(self.rt_head.weight.detach().cpu().numpy().T)
        weights.append(self.rt_head.bias.detach().cpu().numpy())
        weights.append(self.fire_head.weight.detach().cpu().numpy().T)
        weights.append(self.fire_head.bias.detach().cpu().numpy())
        return weights

def compute_gae(rewards, values, dones, last_value):
    T, N = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(N)
    for t in reversed(range(T)):
        next_val = last_value if t == T-1 else values[t+1]
        delta = rewards[t] + 0.99 * next_val * (1.0 - dones[t]) - values[t]
        last_gae = delta + 0.99 * 0.95 * (1.0 - dones[t]) * last_gae
        advantages[t] = last_gae
    return advantages.flatten(), (advantages + values).flatten()

def collect_rollouts(policy, envs, obs_list, opponent_fn, stage=3, turret_only=False):
    """turret_only=True 时强制 mv=0，只训练炮塔旋转和开火，消除走位梯度干扰"""
    T, N = N_STEPS, len(envs)
    buf = {k: np.zeros((T, N, IN_DIM) if k=='obs' else (T, N)) for k in ['obs','mv','rt','fire','logp','val','rew','done']}
    ep_rews = []
    ep_buf = np.zeros(N)

    for t in range(T):
        obs_tensor = torch.FloatTensor(np.stack(obs_list)).to(device)
        with torch.no_grad():
            mv, rt, fire, mv_idx, rt_idx, logp, value = policy.act(obs_tensor)
        
        mv_np, rt_np, fire_np = mv.cpu().numpy(), rt.cpu().numpy(), fire.cpu().numpy()
        mv_idx_np, rt_idx_np, logp_np, val_np = mv_idx.cpu().numpy(), rt_idx.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()

        for i, env in enumerate(envs):
            obs2 = env.get_relative_obs(env.p2, env.p1)
            action2 = opponent_fn(obs2)
            # 炮塑锁定模式：底盘不动，只训练炮塔（Stage 0 关键优化）
            mv_val = 0.0 if turret_only else mv_np[i]
            action1 = np.array([mv_val, rt_np[i], fire_np[i]])
            # 记录的 mv_idx 保持原样，使策略梯度不被截断
            next_obs, reward, done = env.step(action1, action2)

            buf['obs'][t,i], buf['mv'][t,i], buf['rt'][t,i], buf['fire'][t,i] = obs_list[i], mv_idx_np[i], rt_idx_np[i], fire_np[i]
            buf['logp'][t,i], buf['val'][t,i], buf['rew'][t,i], buf['done'][t,i] = logp_np[i], val_np[i], reward, float(done)
            ep_buf[i] += reward
            if done:
                ep_rews.append(ep_buf[i]); ep_buf[i] = 0.0
                obs_list[i] = env.reset(stage=stage)
            else: obs_list[i] = next_obs

    with torch.no_grad():
        last_val = policy.get_value(torch.FloatTensor(np.stack(obs_list)).to(device)).cpu().numpy()
    adv, ret = compute_gae(buf['rew'], buf['val'], buf['done'], last_val)
    batch = (buf['obs'].reshape(-1, IN_DIM), buf['mv'].flatten(), buf['rt'].flatten(), buf['fire'].flatten(), buf['logp'].flatten(), adv, ret)
    return batch, obs_list, np.mean(ep_rews) if ep_rews else 0.0

class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 1.0

    def update_and_normalize(self, rewards: np.ndarray) -> np.ndarray:
        for r in rewards.flatten():
            self.n += 1
            delta = r - self.mean
            self.mean += delta / self.n
            delta2 = r - self.mean
            self.M2 += delta * delta2
        std = math.sqrt(self.M2 / max(self.n, 1)) + 1e-8
        return rewards / std

def ppo_update(policy, optimizer, batch):
    obs_b, mv_b, rt_b, fire_b, lp_b, adv_b, ret_b = [torch.FloatTensor(x).to(device) for x in batch]
    adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
    
    total_loss = pg_loss = vf_loss = ent_loss = 0.0
    n_batches = 0
    
    for _ in range(N_EPOCHS):
        idx = np.random.permutation(len(obs_b))
        for s in range(0, len(idx), BATCH_SIZE):
            mb = idx[s:s+BATCH_SIZE]
            lp, ent, v = policy.evaluate(obs_b[mb], mv_b[mb], rt_b[mb], fire_b[mb])
            ratio = (lp - lp_b[mb]).exp()
            surr1 = ratio * adv_b[mb]
            surr2 = ratio.clamp(1-CLIP_EPS, 1+CLIP_EPS) * adv_b[mb]
            l_pg = -torch.min(surr1, surr2).mean()
            l_vf = VF_COEF * (v - ret_b[mb]).pow(2).mean()
            l_ent = -ENT_COEF * ent.mean()
            
            loss = l_pg + l_vf + l_ent
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD)
            optimizer.step()
            
            total_loss += loss.item()
            pg_loss += l_pg.item()
            vf_loss += l_vf.item()
            ent_loss += l_ent.item()
            n_batches += 1
            
    n = max(n_batches, 1)
    return total_loss/n, pg_loss/n, vf_loss/n, ent_loss/n

def train(max_steps=15_000_000):
    policy = ActorCritic().to(device)
    # 因为架构全面升级为分支网络，使用 v5 命名
    save_path = "E:\\Test_FIre\\best_model_v5_branched.pkl"
    if os.path.exists(save_path):
        try:
            with open(save_path, "rb") as f:
                w = pickle.load(f)
            with torch.no_grad():
                # 按照 export_weights 的顺序反向加载
                policy.shared[0].weight.copy_(torch.FloatTensor(w[0].T))
                policy.shared[0].bias.copy_(torch.FloatTensor(w[1]))
                policy.offense_branch[0].weight.copy_(torch.FloatTensor(w[2].T))
                policy.offense_branch[0].bias.copy_(torch.FloatTensor(w[3]))
                policy.tactical_branch[0].weight.copy_(torch.FloatTensor(w[4].T))
                policy.tactical_branch[0].bias.copy_(torch.FloatTensor(w[5]))
                
                policy.mv_head.weight.copy_(torch.FloatTensor(w[6].T))
                policy.mv_head.bias.copy_(torch.FloatTensor(w[7]))
                policy.rt_head.weight.copy_(torch.FloatTensor(w[8].T))
                policy.rt_head.bias.copy_(torch.FloatTensor(w[9]))
                policy.fire_head.weight.copy_(torch.FloatTensor(w[10].T))
                policy.fire_head.bias.copy_(torch.FloatTensor(w[11]))
            print(f"Loaded {save_path}")
        except Exception as e:
            print(f"Start from scratch. Reason: {e}")
    
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    envs = [LinkCombatEnv() for _ in range(N_ENVS)]
    obs_list = [env.reset(stage=0) for env in envs]
