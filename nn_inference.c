#include <stdio.h>
#include <math.h>
#include "model_weights.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ============================================================
// 神经网络推理：13输入 -> 48隐藏1 -> 32隐藏2 -> 3输出
// 参数量: 2339，Flash占用约 9.4KB，适配 STM32F103C8T6 (64KB Flash)
//
// 观测输入（13维，与 Python get_relative_obs 严格一致）:
//  obs[0]  dist         相对距离        (0~1)
//  obs[1]  angle_diff   我方枪口误差    (-1~1，有符号)
//  obs[2]  dx           X轴相对差       (-1~1)
//  obs[3]  dy           Y轴相对差       (-1~1)
//  obs[4]  self_hp      自身血量        (0~1)
//  obs[5]  enemy_hp     敌人血量        (0~1)
//  obs[6]  exposure     我方暴露系数    (0~1)
//  obs[7]  wall_dist    最近墙距        (0~1)
//  obs[8]  sin(self_a)  自身朝向sin     (-1~1)
//  obs[9]  cos(self_a)  自身朝向cos     (-1~1)
//  obs[10] sin(enemy_a) 敌人朝向sin     (-1~1)
//  obs[11] cos(enemy_a) 敌人朝向cos     (-1~1)
//  obs[12] self_cd      自身冷却状态    (0=可开火, 1=刚射击)
// ============================================================
void nn_inference(const float obs[13], float actions[3]) {
    float h1[48], h2[32];

    // 1. 输入层 → 隐藏层1 (13 → 48)
    for (int j = 0; j < 48; j++) {
        h1[j] = B1[j];
        for (int i = 0; i < 13; i++) h1[j] += obs[i] * W1[i][j];
        h1[j] = tanhf(h1[j]);
    }

    // 2. 隐藏层1 → 隐藏层2 (48 → 32)
    for (int j = 0; j < 32; j++) {
        h2[j] = B2[j];
        for (int i = 0; i < 48; i++) h2[j] += h1[i] * W2[i][j];
        h2[j] = tanhf(h2[j]);
    }

    // 3. 隐藏层2 → 输出层 (32 → 3)
    for (int j = 0; j < 3; j++) {
        actions[j] = B3[j];
        for (int i = 0; i < 32; i++) actions[j] += h2[i] * W3[i][j];
    }

    actions[0] = tanhf(actions[0]);  // mv: 移动量  [-1, 1]
    actions[1] = tanhf(actions[1]);  // rt: 旋转量  [-1, 1]
    // actions[2] > 0 开火
}

// ============================================================
// 特征工程：绝对坐标 → 13维相对特征
// 须与 Python get_relative_obs() 保持严格一致
//
// 参数:
//   self_x/y/a/hp  — 自身 X,Y坐标,朝向角(度),血量(0~100)
//   enemy_x/y/a/hp — 敌方 X,Y坐标,朝向角(度),血量(0~100)
//   self_cd        — 自身冷却计数(0=可立即开火, fire_cooldown=刚射击)
//   fire_cooldown  — 最大冷却计数(与训练时保持一致，通常为15)
// ============================================================
void process_game_data(float self_x,  float self_y,  float self_a,  float self_hp,
                       float enemy_x, float enemy_y, float enemy_a, float enemy_hp,
                       float self_cd, float fire_cooldown) {

    const float PLAYER_RADIUS = 20.0f;
    const float ARENA_SIZE    = 500.0f;

    float dx   = enemy_x - self_x;
    float dy   = enemy_y - self_y;
    float dist = sqrtf(dx*dx + dy*dy);

    // 1. 我方枪口误差角（有符号，区分偏左/偏右）
    float tgt_deg = atan2f(dy, dx) * 180.0f / M_PI;
    if (tgt_deg < 0.0f) tgt_deg += 360.0f;
    float angle_diff = fmodf((tgt_deg - self_a + 180.0f), 360.0f) - 180.0f;

    // 2. 敌方枪口误差角 → 暴露系数
    float etm_deg    = fmodf(tgt_deg + 180.0f, 360.0f);
    float enemy_err  = fabsf(fmodf((etm_deg - enemy_a + 180.0f), 360.0f) - 180.0f);
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

    // 4. 朝向 sin/cos 编码（消除360→0°跳变）
    float rad_self  = self_a  * M_PI / 180.0f;
    float rad_enemy = enemy_a * M_PI / 180.0f;

    // 5. 冷却归一化
    float cd_norm = self_cd / (fire_cooldown > 0.0f ? fire_cooldown : 1.0f);
    if (cd_norm > 1.0f) cd_norm = 1.0f;
    if (cd_norm < 0.0f) cd_norm = 0.0f;

    // 6. 组装13维观测向量
    float obs[13] = {
        dist / 707.0f,          // 0.  相对距离
        angle_diff / 180.0f,    // 1.  我方枪口误差（有符号）
        dx / ARENA_SIZE,        // 2.  dx
        dy / ARENA_SIZE,        // 3.  dy
        self_hp  / 100.0f,      // 4.  自身血量
        enemy_hp / 100.0f,      // 5.  敌人血量
        exposure,               // 6.  暴露系数
        wall_dist,              // 7.  最近墙距
        sinf(rad_self),         // 8.  自身朝向 sin
        cosf(rad_self),         // 9.  自身朝向 cos
        sinf(rad_enemy),        // 10. 敌人朝向 sin
        cosf(rad_enemy),        // 11. 敌人朝向 cos
        cd_norm                 // 12. 冷却状态（0=可开火）
    };

    float actions[3];
    nn_inference(obs, actions);

    // 将 actions[0](mv), actions[1](rt), actions[2]>0(fire) 输出
    // send_to_uart(actions[0], actions[1], actions[2] > 0.0f ? 1 : 0);
}
