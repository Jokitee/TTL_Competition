/**
 * uart_ai_handler.c
 * 
 * STM32 UART AI 推理接入代码 v5.0 — 分支架构版
 * 协议：接收游戏状态 -> 推理 -> 发送控制指令
 * 
 * 输入帧 (上位机→STM32): {"p1":{"x":50,"y":50,"a":45,"hp":100},"p2":{"x":50,"y":50,"a":45,"hp":100}}\n
 * 输出帧 (STM32→上位机): {"mv":0,"rt":0,"fr":0}\n
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "main.h"
#include "nn_inference.h"
#include "uart_ai_handler.h"

// ============================================================
// UART 配置（与 CubeMX 中 UART5 的句柄名一致）
// ============================================================
extern UART_HandleTypeDef huart1;

// ============================================================
// 比赛时通过修改此宏决定我方是 P1 还是 P2
// 0 = 我方是 P1 (默认), 1 = 我方是 P2
// ============================================================
#define AI_PLAY_AS_P2  1

#define RX_BUF_SIZE 128
#define TX_BUF_SIZE 64

static char rx_buf[RX_BUF_SIZE];
static uint16_t rx_idx = 0;
static char tx_buf[TX_BUF_SIZE];

// ============================================================
// 发送字符串到 UART5
// ============================================================
static void uart_send(const char *str) {
    HAL_UART_Transmit(&huart1, (uint8_t *)str, strlen(str), 100);
}

// ============================================================
// 解析输入帧并执行推理，发送结果
// 格式: {"p1":{"x":50,"y":50,"a":45,"hp":100},"p2":{"x":50,"y":50,"a":45,"hp":100}}
// ============================================================
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

static void process_uart_frame(char *frame) {
    float self_x, self_y, self_a, self_hp;
    float enemy_x, enemy_y, enemy_a, enemy_hp;

    // 解析 JSON：先将 p1/p2 分别存入临时变量
    float p1_x, p1_y, p1_a, p1_hp;
    float p2_x, p2_y, p2_a, p2_hp;

    int parsed = sscanf(frame, "{\"p1\":{\"x\":%f,\"y\":%f,\"a\":%f,\"hp\":%f},\"p2\":{\"x\":%f,\"y\":%f,\"a\":%f,\"hp\":%f}}",
                        &p1_x, &p1_y, &p1_a, &p1_hp,
                        &p2_x, &p2_y, &p2_a, &p2_hp);

    if (parsed != 8) {
        // 兼容带空格的 JSON
        parsed = sscanf(frame, "{ \"p1\" : { \"x\" : %f , \"y\" : %f , \"a\" : %f , \"hp\" : %f } , \"p2\" : { \"x\" : %f , \"y\" : %f , \"a\" : %f , \"hp\" : %f } }",
                        &p1_x, &p1_y, &p1_a, &p1_hp,
                        &p2_x, &p2_y, &p2_a, &p2_hp);
        if (parsed != 8) {
            uart_send("{\"error\":\"parse failed\"}\n");
            return;
        }
    }

    // 根据编译宏决定我方阵营，交换 self/enemy
#if AI_PLAY_AS_P2
    // 我方是 P2
    self_x  = p2_x;  self_y  = p2_y;  self_a  = p2_a;  self_hp  = p2_hp;
    enemy_x = p1_x;  enemy_y = p1_y;  enemy_a = p1_a;  enemy_hp = p1_hp;
#else
    // 我方是 P1（默认）
    self_x  = p1_x;  self_y  = p1_y;  self_a  = p1_a;  self_hp  = p1_hp;
    enemy_x = p2_x;  enemy_y = p2_y;  enemy_a = p2_a;  enemy_hp = p2_hp;
#endif

    // ================== 特征工程 (16维，与 get_relative_obs 严格一致) ==================
    const float PLAYER_RADIUS = 20.0f;
    const float ARENA_SIZE    = 500.0f;
    const float PLAYER_SPEED  = 3.0f;

    float dx   = enemy_x - self_x;
    float dy   = enemy_y - self_y;
    float dist = sqrtf(dx*dx + dy*dy);

    // 1. 我方枪口误差角（有符号，区分偏左/偏右）
    float tgt_deg = atan2f(dy, dx) * 180.0f / M_PI;
    if (tgt_deg < 0.0f) tgt_deg += 360.0f;
    float angle_diff = fmodf((tgt_deg - self_a + 180.0f), 360.0f) - 180.0f;

    // 2. 敌方枪口误差角 → 暴露系数
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

    // 5. 敌方沿视线方向速度分量（远程索敌关键特征）
    float enemy_rad = rad_enemy;
    float evx = cosf(enemy_rad) * PLAYER_SPEED;
    float evy = sinf(enemy_rad) * PLAYER_SPEED;
    float vel_along = 0.0f;
    float vel_perp  = 0.0f;
    if (dist > 0.001f) {
        vel_along = (evx * dx + evy * dy) / dist;
        vel_perp  = (-evx * dy + evy * dx) / dist;  // 切向速度（预判射击关键）
    }
    vel_along /= PLAYER_SPEED;  // 归一化到 [-1, 1]
    vel_perp  /= PLAYER_SPEED;  // 归一化到 [-1, 1]

    // 5. 内部冷却计数器（跨帧保持）
    // 训练时 fire_cooldown=15，每帧减 1，开火后重置为 15
    static int  cd_counter  = 0;
    static float prev_dist  = -1.0f;
    
    if (cd_counter > 0) cd_counter--;
    float cd_norm = (float)cd_counter / 15.0f;
    
    // 6. 数据组装 — 16 维观测向量（与 Python get_relative_obs() 严格对齐）
    float obs[16] = {
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
        cd_norm,                // 12. 冷却状态（0=可开火）
        vel_along,              // 13. 敌方沿视线方向速度（-1~1）
        dist_rate,              // 14. 距离变化率（-1~1）
        vel_perp                // 15. 敌方垂直视线速度（-1~1，预判射击关键）
    };

    // ================== 执行推理 ==================
    float actions[3];
    nn_inference(obs, actions);

    // ================== 动作输出（分支架构离散动作）==================
    // actions[0] = mv ∈ {-1, 0, +1}  底盘移动
    // actions[1] = rt ∈ {-1, 0, +1}  炮塔旋转
    // actions[2] = fire ∈ {0, 1}      开火决策
    int mv = (int)actions[0];
    int rt = (int)actions[1];
    int fr = (actions[2] > 0.5f) ? 1 : 0;
    
    // 开火后重置冷却
    if (fr) cd_counter = 15;

    snprintf(tx_buf, TX_BUF_SIZE, "{\"mv\":%d,\"rt\":%d,\"fr\":%d}\n", mv, rt, fr);
    uart_send(tx_buf);
}

// ============================================================
// UART5 中断回调中调用此函数
// 将此函数加入 HAL_UART_RxCpltCallback
// ============================================================
static uint8_t rx_byte;

void AI_UART_Init(void) {
    // 启动 UART5 字节接收（中断模式）
    HAL_UART_Receive_IT(&huart1, &rx_byte, 1);
}

void AI_UART_RxCallback(void) {
    char c = (char)rx_byte;

    if (c == '\n' || c == '\r') {
        if (rx_idx > 0) {
            rx_buf[rx_idx] = '\0';
            process_uart_frame(rx_buf);
            rx_idx = 0;
        }
    } else if (rx_idx < RX_BUF_SIZE - 1) {
        rx_buf[rx_idx++] = c;
    }

    // 继续监听下一个字节
    HAL_UART_Receive_IT(&huart1, &rx_byte, 1);
}