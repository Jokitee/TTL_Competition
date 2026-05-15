class NetConfig:
    """神经网络架构配置 (v3.0 离散版)"""
    IN = 16   # dist, angle_diff, dx, dy, hp, hp_enemy, exposure, wall, sin_r, cos_r, sin_e, cos_e, cd, vel_along, dist_rate, vel_perp
    H1 = 64
    H2 = 48
    H3 = 32
    OUT = 7   # mv(3) + rt(3) + fire(1)

class GATrainConfig:
    """遗传算法(GA)训练固定参数配置"""
    POP_SIZE = 100          # 种群大小
    ELITE_SIZE = 10         # 精英保留数量
    GENERATIONS = 1000      # 总训练代数 (提高代数以拟合更复杂的网络)
    MUT_RATE_INIT = 0.15    # 初始变异率

    # 课程阶段切分点
    PHASE2_START_GEN = 150  # 阶段2：对抗随机池 开始代数
    PHASE3_START_GEN = 300  # 阶段3：Hall of Fame 自博弈 开始代数

    # Hall of Fame 设置
    HOF_MAX = 50            # 名人堂最大容量
    HOF_INTERVAL = 5        # 每N代存一次名人堂

    # Phase2 对手池大小
    POOL2_SIZE = 10          # 随机对手池容量

    # 是否启用实时监控 (另启 realtime_plot.py)
    REALTIME_MONITOR = False

    # 各阶段对局评估参数
    PHASE_CFG = {
        1: dict(episodes=3, max_steps=900),   # 假人阶段：3局×15s，速度最快
        2: dict(episodes=5, max_steps=1200),  # 随机阶段：5局×20s
        3: dict(episodes=4, max_steps=1800),  # HoF阶段：4局×30s，全时长
    }


class PPOTrainConfig:
    """强化学习(PPO)训练固定参数配置"""
    LR = 3e-4               # 学习率
    GAMMA = 0.99            # 折扣因子
    GAE_LAM = 0.95          # GAE lambda
    CLIP_EPS = 0.2          # PPO clip parameter
    VF_COEF = 0.5           # 价值函数损失系数
    ENT_COEF = 0.005        # 增加一点熵正则化，提升前期探索能力，防止过早收敛
    MAX_GRAD = 0.5          # 梯度裁剪阈值

    N_ENVS = 16             # 并行环境数量
    N_STEPS = 512           # 每次收集步数
    N_EPOCHS = 8            # 每次更新轮数
    BATCH_SIZE = 256        # Minibatch大小

    TOTAL_STEPS = 15_000_000  # 总训练步数提升到 1500万，充分训练新加深的网络
    OPP_UPDATE_STEPS = 100_000  # 每隔N步将当前策略加入对手池
    OPP_POOL_MAX = 20         # 对手池最大容量

    # 是否启用实时监控 (另启 realtime_plot.py)
    REALTIME_MONITOR = False
