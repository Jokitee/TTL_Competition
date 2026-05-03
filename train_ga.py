import numpy as np
from game_env import LinkCombatEnv
import pickle
import concurrent.futures

# 各阶段配置（按课程难度动态调整，平衡速度与质量）
# Phase1: 假人不动，15s游戏足够；Phase3: 全时长30s确保策略正确性
PHASE_CFG = {
    1: dict(episodes=3, max_steps=900),   # 假人：3局×15s，速度最快
    2: dict(episodes=5, max_steps=1200),  # 随机：5局×20s
    3: dict(episodes=4, max_steps=1800),  # HoF：4局×30s，全时长
}

# ============================================================
# 架构：13-48-32-3
# 参数：2339  Flash：9.1KB  (STM32F103C8T6 64KB完全适配)
# ============================================================
IN=13; H1=48; H2=32; OUT=3

class SimpleBrain:
    def __init__(self, weights=None):
        if weights is None:
            self.w1 = np.random.randn(IN,H1)*0.5;  self.b1 = np.zeros(H1)
            self.w2 = np.random.randn(H1,H2)*0.5;  self.b2 = np.zeros(H2)
            self.w3 = np.random.randn(H2,OUT)*0.5; self.b3 = np.zeros(OUT)
        else:
            self.w1,self.b1,self.w2,self.b2,self.w3,self.b3 = weights

    def forward(self, obs):
        h1  = np.tanh(obs @ self.w1 + self.b1)
        h2  = np.tanh(h1  @ self.w2 + self.b2)
        out = h2 @ self.w3 + self.b3
        return np.array([np.tanh(out[0]), np.tanh(out[1]),
                         1.0 if out[2]>0 else 0.0], dtype=np.float32)

    def get_weights(self):
        return [self.w1,self.b1,self.w2,self.b2,self.w3,self.b3]


# ============================================================
# 评估（随机起始位置，对抗指定对手）
# ============================================================
def evaluate(brain, opponent, episodes=5, max_steps=1800, add_noise=True):
    env   = LinkCombatEnv(max_steps=max_steps)
    total = 0.0
    for _ in range(episodes):
        obs  = env.reset()
        done = False
        while not done:
            obs2 = env.get_relative_obs(env.p2, env.p1)
            a1   = brain.forward(obs)
            a2   = opponent.forward(obs2)
            if add_noise:
                a1[0] += np.random.randn()*0.05
                a1[1] += np.random.randn()*0.05
                a1     = np.clip(a1,-1,1)
            obs, r, done = env.step(a1, a2)
            total += r
    return total / episodes

def evaluate_worker(args):
    """
    args = (brain, opponents, phase)
    opponents: 单个 Brain 或 [Brain, Brain, Brain]
    phase: 用于查 PHASE_CFG，决定 episodes 和 max_steps
    """
    brain, opponents, phase = args
    cfg = PHASE_CFG.get(phase, PHASE_CFG[3])
    ep  = cfg['episodes']; ms = cfg['max_steps']
    if isinstance(opponents, list):
        scores = [evaluate(brain, opp, episodes=ep, max_steps=ms, add_noise=True)
                  for opp in opponents]
        return float(np.mean(scores))
    return evaluate(brain, opponents, episodes=ep, max_steps=ms, add_noise=True)


# ============================================================
# 遗传算子
# ============================================================
def mutate(brain, rate):
    return SimpleBrain([l + np.random.randn(*l.shape)*rate
                        for l in brain.get_weights()])

def crossover(a, b):
    new_w = []
    for la, lb in zip(a.get_weights(), b.get_weights()):
        m = np.random.rand(*la.shape) > 0.5
        new_w.append(np.where(m, la, lb))
    return SimpleBrain(new_w)

def tournament(pop, scores, k=5):
    idx = np.random.choice(len(pop), k, replace=False)
    return pop[idx[np.argmax([scores[i] for i in idx])]]


# ============================================================
# 主训练循环 — 超强人机版
# ============================================================
def train():
    pop_size    = 100          # 更大种群 → 更丰富基因多样性
    elite_size  = 10
    generations = 200
    mut_rate    = 0.15

    # 课程阶段
    PHASE2 = 30    # 0~29:  静止假人（学追踪瞄准）
    PHASE3 = 60    # 30~59: 随机池（学追击移动目标）
                   # 60+:   Hall of Fame 自博弈（学全面对抗）

    # HoF 设置
    hall_of_fame  = []
    HOF_MAX       = 30
    HOF_INTERVAL  = 4     # 每4代存一次（更高频，HoF更新更快）

    # Phase2 对手池（多个随机个体，防止对单一对手过拟合）
    POOL2_SIZE    = 8

    # 初始种群
    population = [SimpleBrain() for _ in range(pop_size)]

    # 静止假人
    dummy_w = [np.zeros((IN,H1)), np.zeros(H1),
               np.zeros((H1,H2)), np.zeros(H2),
               np.zeros((H2,OUT)),np.zeros(OUT)]
    dummy   = SimpleBrain(weights=dummy_w)
    pool2   = [dummy]     # 初始Pool2 = 假人

    best_ever = -float('inf')
    no_imp    = 0
    total_p   = IN*H1+H1 + H1*H2+H2 + H2*OUT+OUT

    print("="*65)
    print(f"超强人机训练 v7.0: {IN}-{H1}-{H2}-{OUT}  "
          f"({total_p}params, {total_p*4}B Flash)")
    print(f"  Pop={pop_size}  Elite={elite_size}  Gens={generations}")
    print(f"  随机初始位置 | 护盾感知 | 三阶课程 | HoF自博弈")
    print("="*65)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for gen in range(generations):

            # ── 课程切换 ────────────────────────────────────
            if gen == PHASE2:
                print(f"\n>>> [Ph2] 随机对手池({POOL2_SIZE}个) <<<\n")
                pool2   = [SimpleBrain() for _ in range(POOL2_SIZE)]
                best_ever = -float('inf'); no_imp = 0; mut_rate = 0.12
            elif gen == PHASE3:
                print(f"\n>>> [Ph3] Hall of Fame 自博弈 <<<\n")
                best_ever = -float('inf'); no_imp = 0; mut_rate = 0.12

            # ── 构建对战任务（phase 决定局数和游戏时长）──────
            phase = 1 if gen<PHASE2 else (2 if gen<PHASE3 else 3)
            if gen >= PHASE3 and len(hall_of_fame) >= 3:
                n_hof   = len(hall_of_fame)
                weights = np.linspace(1.0, 3.0, n_hof); weights /= weights.sum()
                def sample_opps():
                    idx = np.random.choice(n_hof, size=3, replace=False, p=weights)
                    return [hall_of_fame[i] for i in idx]
                tasks = [(b, sample_opps(), phase) for b in population]
            elif gen >= PHASE3 and len(hall_of_fame) > 0:
                tasks = [(b, hall_of_fame[np.random.randint(len(hall_of_fame))], phase)
                         for b in population]
            elif gen >= PHASE2:
                tasks = [(b, pool2[np.random.randint(len(pool2))], phase)
                         for b in population]

            else:
                tasks = [(b, dummy, phase) for b in population]

            scores = list(executor.map(evaluate_worker, tasks))

            # ── 精英保留 ────────────────────────────────────
            ranked     = np.argsort(scores)[::-1]
            elite      = [population[i] for i in ranked[:elite_size]]
            best_score = scores[ranked[0]]

            # ── 自适应变异率 ────────────────────────────────
            if best_score > best_ever:
                best_ever = best_score; no_imp = 0
                mut_rate  = max(0.06, mut_rate * 0.97)
            else:
                no_imp += 1
                if no_imp >= 15:
                    mut_rate = min(0.40, mut_rate * 1.30)
                    no_imp   = 0
                    print(f"  [Stuck] Mut→{mut_rate:.3f}")

            phase = 1 if gen<PHASE2 else (2 if gen<PHASE3 else 3)
            print(f"Gen {gen:>3} [Ph{phase}]: "
                  f"Best={best_score:.1f}  Ever={best_ever:.1f}  "
                  f"Mut={mut_rate:.3f}  HoF={len(hall_of_fame)}")

            # ── 繁衍下一代 ──────────────────────────────────
            new_pop = elite[:]
            while len(new_pop) < pop_size:
                r = np.random.rand()
                if r < 0.35:
                    # 锦标赛变异
                    p = tournament(population, scores, k=6)
                    new_pop.append(mutate(p, mut_rate))
                elif r < 0.60:
                    # 交叉 + 微变异
                    a = tournament(population, scores, k=5)
                    b = tournament(population, scores, k=5)
                    new_pop.append(mutate(crossover(a,b), mut_rate*0.4))
                elif r < 0.80:
                    # 重度变异（跳出局部最优）
                    p = tournament(population, scores, k=3)
                    new_pop.append(mutate(p, mut_rate*4.0))
                elif r < 0.90:
                    # 精英微调（细化精英策略）
                    p = elite[np.random.randint(elite_size)]
                    new_pop.append(mutate(p, mut_rate*0.1))
                else:
                    # 全新随机（保持基因多样性）
                    new_pop.append(SimpleBrain())
            population = new_pop

            # ── 更新各种对手池 ───────────────────────────────
            if gen % HOF_INTERVAL == 0:
                hall_of_fame.append(elite[0])
                if len(hall_of_fame) > HOF_MAX:
                    hall_of_fame.pop(0)

            # Phase2：每8代刷新池（引入当代最强，淘汰旧的）
            if PHASE2 <= gen < PHASE3 and gen % 8 == 0:
                pool2.append(elite[0])
                if len(pool2) > POOL2_SIZE:
                    pool2.pop(0)

            # ── 保存最强模型 ─────────────────────────────────
            with open("E:\\Test_FIre\\best_model.pkl","wb") as f:
                pickle.dump(elite[0].get_weights(), f)

    print(f"\n训练完成！历史最高: {best_ever:.1f}")
    print("模型已保存至 E:\\Test_FIre\\best_model.pkl")

if __name__ == "__main__":
    train()
