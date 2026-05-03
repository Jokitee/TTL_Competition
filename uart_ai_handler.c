/**
 * uart_ai_handler.c
 * 
 * STM32 UART1 AI 推理接入代码
 * 协议：接收游戏状态 -> 推理 -> 发送控制指令
 * 
 * 输入帧 (上位机→STM32): {"p1":{"x":50,"y":50,"a":45,"hp":100},"p2":{"x":50,"y":50,"a":45,"hp":100}}\n
 * 输出帧 (STM32→上位机): {"mv":0,"rt":0,"fr":0}\n
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "stm32f1xx_hal.h"  // 根据你的HAL库路径调整
#include "nn_inference.c"   // 包含推理引擎

// ============================================================
// UART 配置（需与 CubeMX 中 UART1 的句柄名一致）
// ============================================================
extern UART_HandleTypeDef huart1;

#define RX_BUF_SIZE 128
#define TX_BUF_SIZE 64

static char rx_buf[RX_BUF_SIZE];
static uint16_t rx_idx = 0;
static char tx_buf[TX_BUF_SIZE];

// ============================================================
// 发送字符串到 UART1
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

    // 解析 JSON (简易匹配)
    int parsed = sscanf(frame, "{\"p1\":{\"x\":%f,\"y\":%f,\"a\":%f,\"hp\":%f},\"p2\":{\"x\":%f,\"y\":%f,\"a\":%f,\"hp\":%f}}",
                        &self_x, &self_y, &self_a, &self_hp,
                        &enemy_x, &enemy_y, &enemy_a, &enemy_hp);

    if (parsed != 8) {
        // 兼容带空格的 JSON
        parsed = sscanf(frame, "{ \"p1\" : { \"x\" : %f , \"y\" : %f , \"a\" : %f , \"hp\" : %f } , \"p2\" : { \"x\" : %f , \"y\" : %f , \"a\" : %f , \"hp\" : %f } }",
                        &self_x, &self_y, &self_a, &self_hp,
                        &enemy_x, &enemy_y, &enemy_a, &enemy_hp);
        if (parsed != 8) {
            uart_send("{\"error\":\"parse failed\"}\n");
            return;
        }
    }

    // ================== 特征工程 (13维) ==================
    const float PLAYER_RADIUS = 20.0f;
    const float ARENA_SIZE    = 500.0f;

    float dx = enemy_x - self_x;
    float dy = enemy_y - self_y;
    float dist = sqrtf(dx*dx + dy*dy);

    float etx = cosf(self_a * M_PI / 180.0f);
    float ety = sinf(self_a * M_PI / 180.0f);
    float dot_val = (dx*etx + dy*ety) / fmaxf(dist, 1e-6f);
    if (dot_val > 1.0f) dot_val = 1.0f;
    if (dot_val < -1.0f) dot_val = -1.0f;
    float angle_diff = acosf(dot_val) * 180.0f / M_PI;

    // 护盾盲区暴露度
    float etm = atan2f(-dy, -dx) * 180.0f / M_PI;
    if (etm < 0.0f) etm += 360.0f;
    float ea = fmodf(etm - enemy_a + 180.0f, 360.0f);
    if (ea < 0.0f) ea += 360.0f;
    ea = fabsf(ea - 180.0f);
    
    float dc = (dist > PLAYER_RADIUS) ? asinf(fminf(1.0f, PLAYER_RADIUS / dist)) * 180.0f / M_PI : 90.0f;
    float exposure = fmaxf(0.0f, 1.0f - ea / fmaxf(dc, 1e-6f));

    // 距墙距离
    float walls[4] = {self_x, ARENA_SIZE-self_x, self_y, ARENA_SIZE-self_y};
    float min_wall = walls[0];
    for (int i = 1; i < 4; i++) if (walls[i] < min_wall) min_wall = walls[i];
    float wall_dist = min_wall / (ARENA_SIZE / 2.0f);

    float rs = self_a * M_PI / 180.0f;
    float re = enemy_a * M_PI / 180.0f;

    float obs[13] = {
        dist / 707.0f,
        angle_diff / 180.0f,
        dx / ARENA_SIZE,
        dy / ARENA_SIZE,
        self_hp / 100.0f,
        enemy_hp / 100.0f,
        exposure,
        wall_dist,
        sinf(rs),
        cosf(rs),
        sinf(re),
        cosf(re),
        0.0f    // CD 设为0
    };

    // ================== 执行推理 ==================
    float actions[3];
    nn_inference(obs, actions);

    // ================== 动作离散化 ==================
    int mv = 0, rt = 0, fr = 0;
    
    if (actions[0] > 0.3f) mv = 1;
    else if (actions[0] < -0.3f) mv = -1;

    if (actions[1] > 0.3f) rt = 1;
    else if (actions[1] < -0.3f) rt = -1;

    if (actions[2] > 0.5f) fr = 1;

    // 输出帧: {"mv":0,"rt":0,"fr":0}\n
    snprintf(tx_buf, TX_BUF_SIZE, "{\"mv\":%d,\"rt\":%d,\"fr\":%d}\n", mv, rt, fr);
    uart_send(tx_buf);
}

// ============================================================
// 在 UART1 中断回调中调用此函数
// 将此函数加入 HAL_UART_RxCpltCallback
// ============================================================
static uint8_t rx_byte;

void AI_UART_Init(void) {
    // 启动 UART1 字节接收（中断模式）
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
