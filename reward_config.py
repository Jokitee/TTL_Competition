class RewardConfig:
    """
    纯进攻型奖励配置 v5.2 — 猎手模式 (Hunter Mode)
    设计原则：所有奖励只服务于一个目标——让对方掉血直至击杀。
    删除全部防御性奖励，AI 不惧被击中，只求击倒对手。
    """

    # ══════════════════════════════════════════════════════════
    # 核心进攻奖励（这里是 AI 的全部动力）
    # ══════════════════════════════════════════════════════════

    # 命中造成伤害 —— 最直接的进攻反馈
    HIT_ENEMY              = 3000.0  # 每次成功命中对方肉身
    HIT_BLIND_SIDE_BONUS   = 1000.0  # 打到护盾盲区额外加成

    # 击杀奖励 —— AI 的终极目标
    WIN_BASE               = 30000.0  # 基础击杀奖励
    # 快速击杀加成：剩余时间越多，额外奖励越高（鼓励速攻）
    # 实际加成在 game_env.py 中计算: WIN_BASE * (1 + time_remaining_ratio)

    # ──────────────────────────────────────────────────────────
    # 精准开火奖励（让 AI 学会瞄好再打）
    # ──────────────────────────────────────────────────────────
    FIRE_PERFECT_AIM       = 150.0   # 误差 ≤3°  完美准心
    FIRE_PERFECT_AIM_BLIND = 60.0    # 盲区额外加成（此处复用字段名）
    FIRE_GOOD_AIM_MAX      = 50.0    # 误差 3°~10° 良好
    FIRE_MEDIOCRE_AIM      = 8.0     # 误差 10°~20° 勉强
    FIRE_BAD_AIM_PENALTY   = -30.0   # 误差 >20° 的乱开枪惩罚（阶段性缩放）
    FIRE_ANY_ATTEMPT       = 0.0     # 不给通用开火小奖，只奖励精准

    # ──────────────────────────────────────────────────────────
    # 持续追踪奖励（让 AI 时刻保持枪口指向对手）
    # ──────────────────────────────────────────────────────────
    TRACK_PERFECT_REWARD   = 3.0     # 误差 ≤3°  每帧持续奖励（可累积30帧）
    TRACK_GOOD_REWARD      = 1.0     # 误差 3°~10° 每帧小奖
    TRACK_AIM_WEIGHT       = 0.8     # 瞄准追踪的权重系数

    # ──────────────────────────────────────────────────────────
    # 主动接近奖励（让 AI 冲向对手进入有效射程）
    # ──────────────────────────────────────────────────────────
    APPROACH_REWARD_SCALE  = 0.5     # 靠近对手每单位距离奖励
    APPROACH_OPTIMAL_DIST  = 120.0   # 最优战斗距离（近距离命中率最高）
    APPROACH_STOP_DIST     = 60.0    # 过近时停止接近奖励（防止贴脸死循环）

    # ══════════════════════════════════════════════════════════
    # 防御性奖励（全部清零——猎手不需要防守）
    # ══════════════════════════════════════════════════════════
    GOT_HIT_PENALTY        = 0.0     # 被打不扣分！AI 不惧怕死亡
    SHIELD_BLOCK           = 0.0     # 不奖励防御格挡
    DISADVANTAGE_KITE      = 0.0     # 不逃跑
    EVASION_MOVE           = 0.0     # 不规避子弹
    EXPOSURE_REWARD_SCALE  = 0.0     # 不绕后（直面进攻）
    FIRE_ON_SHIELD_PENALTY = -5.0    # 对准护盾开枪的小惩罚（引导打盲区）

    # ══════════════════════════════════════════════════════════
    # 失败惩罚（小额，不影响进攻欲望）
    # ══════════════════════════════════════════════════════════
    LOSE_PENALTY           = -2000.0  # 死亡惩罚很小，猎手不怕死
    TIME_OUT_WIN_BASE      = 5000.0
    TIME_OUT_WIN_PER_HP    = 30.0     # 超时时血量优势折算分
    TIME_OUT_LOSE          = -3000.0
    TIME_OUT_DRAW          = -1000.0

    # ══════════════════════════════════════════════════════════
    # 保留字段（兼容代码引用，但实际值为 0）
    # ══════════════════════════════════════════════════════════
    TRACK_ABSOLUTE_AIM     = 0.0
    LONG_RANGE_OPTIMAL_DIST      = 300.0
    LONG_RANGE_MAINTAIN_BASE     = 0.0
    LONG_RANGE_TOO_CLOSE_PENALTY = 0.0
    LONG_RANGE_CLOSING_IN_PENALTY= 0.0
    LONG_RANGE_PERFECT_THRESHOLD = 1.0
    LONG_RANGE_GOOD_THRESHOLD    = 5.0
    LONG_RANGE_TRACK_BONUS       = 0.0
    CLOSE_COMBAT_OPTIMAL_DIST    = 80.0
    CLOSE_COMBAT_MAINTAIN_BASE   = 0.0
    CLOSE_COMBAT_TOO_CLOSE_PENALTY   = 0.0
    CLOSE_COMBAT_CLOSING_IN_REWARD   = 0.0
    HIT_RATE_BONUS_LONG_RANGE    = 0.0
    HIT_RATE_BONUS_CLOSE_RANGE   = 0.0
    TACTICAL_FLANK         = 0.0
    LEAD_AIM_REWARD        = 0.0
    BOUNCE_WALL_THRESHOLD  = 60.0
    WALL_BOUNCE_REWARD     = 0.0
    WALL_PENALTY           = 0.0
    DISTANCE_THRESHOLD     = 200.0
