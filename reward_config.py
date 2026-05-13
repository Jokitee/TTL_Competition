class RewardConfig:
    """得分奖励的结构体化配置"""
    
    # 攻击准确 (Aiming & Firing)
    FIRE_PERFECT_AIM = 30.0         # 完美开火奖励
    FIRE_PERFECT_AIM_BLIND = 10.0
    FIRE_GOOD_AIM_MAX = 15.0
    FIRE_BAD_AIM_PENALTY = -1.0     # 降低惩罚，防止AI完全不敢开火

    # 伤害与命中 (Damage)
    HIT_ENEMY = 800.0               # 大幅增加精确击中基础权重
    GOT_HIT_PENALTY = -150.0        # 惩罚被击中
    SHIELD_BLOCK = 100.0            # 鼓励用护盾格挡

    # 追踪与瞄准 (Tracking)
    TRACK_AIM_WEIGHT = 12.0         # 连续瞄准的权重
    TRACK_ABSOLUTE_AIM = 10.0        # 绝对精准奖励
    
    # 动态距离感知策略 (Dynamic Distance Strategy)
    DISTANCE_THRESHOLD = 200.0      # 距离阈值判断逻辑：>200远程，<=200近战

    # 远程抽射模式参数 (Long-range mode)
    LONG_RANGE_OPTIMAL_DIST = 350.0
    LONG_RANGE_MAINTAIN_BASE = 2.0
    LONG_RANGE_TOO_CLOSE_PENALTY = -3.0
    LONG_RANGE_CLOSING_IN_PENALTY = -0.1

    # 近战突击模式参数 (Close combat mode)
    CLOSE_COMBAT_OPTIMAL_DIST = 50.0
    CLOSE_COMBAT_MAINTAIN_BASE = 1.0
    CLOSE_COMBAT_TOO_CLOSE_PENALTY = 0.0
    CLOSE_COMBAT_CLOSING_IN_REWARD = 0.5

    # 专项优化：命中率提升奖励
    HIT_RATE_BONUS_LONG_RANGE = 400.0   # 远距离抽射精确命中额外奖励大幅提高
    HIT_RATE_BONUS_CLOSE_RANGE = 150.0  # 近距离精确命中额外奖励
    
    # 墙壁反弹战术 (Wall Bounce Exploit)
    BOUNCE_WALL_THRESHOLD = 60.0    # 激活靠墙反弹战术的距离阈值
    WALL_BOUNCE_REWARD = 2.0        # 成功执行背身靠墙反弹时的奖励
    
    DISADVANTAGE_KITE = 1.5         # 劣势时风筝拉扯
    WALL_PENALTY = 0.5              # 远离墙壁的惩罚（适当调低，容忍靠墙）

    # 规避与战术 (Evasion & Tactics)
    EVASION_MOVE = 3.0              # 降低无脑乱动的奖励
    EVASION_SHIELD = 12.0           # 护盾朝向威胁的奖励
    TACTICAL_FLANK = 5.0            # 绕后/侧翼打击奖励 (对方未指向自己，且自己指向对方)

    # 终局结算 (Game Over)
    WIN_BASE = 5000.0
    LOSE_PENALTY = -5000.0
    TIME_OUT_WIN_BASE = 500.0
    TIME_OUT_WIN_PER_HP = 10.0
    TIME_OUT_LOSE = -5000.0
    TIME_OUT_DRAW = -5000.0
