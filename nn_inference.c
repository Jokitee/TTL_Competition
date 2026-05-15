#include <stdio.h>
#include <math.h>
#include "model_weights.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ============================================================
// 神经网络推理 v5.0 — 分支架构 (Branched Architecture)
// 架构: 16 -> Shared(64) -> [Offense(48), Tactical(48)]
//                         -> [rt(3), fire(1)] , [mv(3)]
// 参数量约: 7248，适配 STM32F103C8T6 (64KB Flash)
//
// 设计理念:
//   共享感知层提取通用几何特征，然后分叉为：
//   - 进攻火控分支 (Offense): 专精炮塔旋转和开火时机 (含预判)
//   - 战术走位分支 (Tactical): 专精底盘机动、绕后和残血规避
//
// 输出:
//   actions[0] mv   底盘移动  {-1(后退), 0(停止), +1(前进)}
//   actions[1] rt   炮塔旋转  {-1(右转), 0(不转), +1(左转)}
//   actions[2] fire 开火      {0(不开火), 1(开火)}
//
// 观测输入（16维，与 Python get_relative_obs 严格一致）:
//  obs[0]  dist         相对距离归一化  (0~1)
//  obs[1]  angle_diff   我方枪口误差    (-1~1)
//  obs[2]  dx           X轴相对差       (-1~1)
//  obs[3]  dy           Y轴相对差       (-1~1)
//  obs[4]  self_hp      自身血量        (0~1)
//  obs[5]  enemy_hp     敌人血量        (0~1)
//  obs[6]  exposure     我方暴露系数    (0~1)
//  obs[7]  wall_dist    最近墙距        (0~1)
//  obs[8]  sin(self_a)  自身朝向sin
//  obs[9]  cos(self_a)  自身朝向cos
//  obs[10] sin(enemy_a) 敌人朝向sin
//  obs[11] cos(enemy_a) 敌人朝向cos
//  obs[12] self_cd      冷却状态        (0=可开火)
//  obs[13] vel_along    敌方沿视线速度  (-1~1)
//  obs[14] dist_rate    距离变化率      (-1~1)
//  obs[15] vel_perp     敌方垂直视线速度(-1~1，预判关键)
// ============================================================

static const float MV_MAP[3] = {0.0f, 1.0f, -1.0f};
static const float RT_MAP[3] = {0.0f, 1.0f, -1.0f};

/**
 * @brief  argmax — 从 n 个 logit 中返回最大值的索引
 */
static int argmax(const float *logits, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

/**
 * @brief  神经网络推理 v5.0 — 分支架构
 * @param  obs     16维归一化观测向量
 * @param  actions 3维输出 (mv, rt, fire)
 */
void nn_inference(const float obs[16], float actions[3]) {
    // ── Layer 1: 共享感知层 (16 -> 64) ──────────────────────────
    float shared[64];
    for (int j = 0; j < 64; j++) {
        shared[j] = B_Shared[j];
        for (int i = 0; i < 16; i++)
            shared[j] += obs[i] * W_Shared[i][j];
        shared[j] = tanhf(shared[j]);
    }

    // ── Layer 2a: 进攻火控分支 (64 -> 48) ───────────────────────
    // 专精炮塔旋转速度预判与开火时机
    float offense[48];
    for (int j = 0; j < 48; j++) {
        offense[j] = B_Offense[j];
        for (int i = 0; i < 64; i++)
            offense[j] += shared[i] * W_Offense[i][j];
        offense[j] = tanhf(offense[j]);
    }

    // ── Layer 2b: 战术走位分支 (64 -> 48) ───────────────────────
    // 专精底盘机动、绕后侧翼与残血规避
    float tactical[48];
    for (int j = 0; j < 48; j++) {
        tactical[j] = B_Tactical[j];
        for (int i = 0; i < 64; i++)
            tactical[j] += shared[i] * W_Tactical[i][j];
        tactical[j] = tanhf(tactical[j]);
    }

    // ── Output: 炮塔旋转头 rt (Offense -> 3 logits) ─────────────
    float rt_logits[3];
    for (int j = 0; j < 3; j++) {
        rt_logits[j] = B_Rt_Head[j];
        for (int i = 0; i < 48; i++)
            rt_logits[j] += offense[i] * W_Rt_Head[i][j];
    }

    // ── Output: 开火决策头 fire (Offense -> 1 logit) ─────────────
    float fire_logit = B_Fire_Head[0];
    for (int i = 0; i < 48; i++)
        fire_logit += offense[i] * W_Fire_Head[i][0];

    // ── Output: 底盘移动头 mv (Tactical -> 3 logits) ─────────────
    float mv_logits[3];
    for (int j = 0; j < 3; j++) {
        mv_logits[j] = B_Mv_Head[j];
        for (int i = 0; i < 48; i++)
            mv_logits[j] += tactical[i] * W_Mv_Head[i][j];
    }

    // ── 离散动作解码 ──────────────────────────────────────────────
    actions[0] = MV_MAP[argmax(mv_logits, 3)];           // 底盘移动
    actions[1] = RT_MAP[argmax(rt_logits, 3)];           // 炮塔旋转
    actions[2] = (fire_logit > 0.0f) ? 1.0f : 0.0f;    // 开火
}

// ============================================================
// 特征工程：绝对坐标 → 16维相对特征
// 须与 Python get_relative_obs() 保持严格一致
// ============================================================
void process_game_data(float self_x,  float self_y,  float self_a,  float self_hp,
                       float enemy_x, float enemy_y, float enemy_a, float enemy_hp,
                       float self_cd, float fire_cooldown, float prev_dist) {

    const float PLAYER_RADIUS = 20.0f;
    const float ARENA_SIZE    = 500.0f;
    const float PLAYER_SPEED  = 3.0f;

    float dx   = enemy_x - self_x;
    float dy   = enemy_y - self_y;
    float dist = sqrtf(dx*dx + dy*dy);

    // 1. 我方枪口误差角（有符号）
    float tgt_deg = atan2f(dy, dx) * 180.0f / M_PI;
    if (tgt_deg < 0.0f) tgt_deg += 360.0f;
    float angle_diff = fmodf((tgt_deg - self_a + 180.0f), 360.0f) - 180.0f;

    // 2. 暴露系数（敌方护盾覆盖判断）
    float etm_deg   = fmodf(tgt_deg + 180.0f, 360.0f);
    float enemy_err = fabsf(fmodf((etm_deg - enemy_a + 180.0f), 360.0f) - 180.0f);
    float danger_cone;
    if (dist > PLAYER_RADIUS) {
        float r = PLAYER_RADIUS / dist;
        if (r > 1.0f) r = 1.0f;
        danger_cone = asinf(r) * 180.0f / M_PI;
    } else {
        danger_cone = 90.0f;
    }
    float exposure = 1.0f - enemy_err / (danger_cone > 0.001f ? danger_cone : 0.001f);
    if (exposure < 0.0f) exposure = 0.0f;

    // 3. 最近墙壁距离（归一化）
    float walls[4] = {self_x, ARENA_SIZE - self_x, self_y, ARENA_SIZE - self_y};
    float min_wall = walls[0];
    for (int i = 1; i < 4; i++) if (walls[i] < min_wall) min_wall = walls[i];
    float wall_dist = min_wall / (ARENA_SIZE / 2.0f);

    // 4. 朝向 sin/cos 编码
    float rad_self  = self_a  * M_PI / 180.0f;
    float rad_enemy = enemy_a * M_PI / 180.0f;

    // 5. 冷却归一化
    float cd_norm = self_cd / (fire_cooldown > 0.0f ? fire_cooldown : 1.0f);
    if (cd_norm > 1.0f) cd_norm = 1.0f;
    if (cd_norm < 0.0f) cd_norm = 0.0f;

    // 6. 敌方速度分量（预判射击核心特征）
    float evx = cosf(rad_enemy) * PLAYER_SPEED;
    float evy = sinf(rad_enemy) * PLAYER_SPEED;
    float vel_along = 0.0f, vel_perp = 0.0f;
    if (dist > 0.001f) {
        vel_along = (evx * dx + evy * dy) / dist / PLAYER_SPEED;
        vel_perp  = (-evx * dy + evy * dx) / dist / PLAYER_SPEED;
    }

    // 7. 距离变化率
    float dist_rate = (prev_dist >= 0.0f) ? (dist - prev_dist) / (PLAYER_SPEED * 2.0f) : 0.0f;

    // 8. 组装 16 维观测
    float obs[16] = {
        dist / 707.0f,
        angle_diff / 180.0f,
        dx / ARENA_SIZE,
        dy / ARENA_SIZE,
        self_hp  / 100.0f,
        enemy_hp / 100.0f,
        exposure,
        wall_dist,
        sinf(rad_self),
        cosf(rad_self),
        sinf(rad_enemy),
        cosf(rad_enemy),
        cd_norm,
        vel_along,
        dist_rate,
        vel_perp
    };

    float actions[3];
    nn_inference(obs, actions);

    // actions[0] = mv   ∈ {-1, 0, +1}
    // actions[1] = rt   ∈ {-1, 0, +1}
    // actions[2] = fire ∈ {0, 1}
    // send_to_uart(actions[0], actions[1], actions[2]);
}