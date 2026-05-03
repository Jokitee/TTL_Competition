#include <stdio.h>
#include <string.h>
#include <math.h>

/* 
 * Link-Combat 串口通信协议 & 游戏规则定义 (C8T6 适配版)
 * 
 * 1. 发包格式 (Web -> STM32): 每 100ms 发送一次 JSON 字符串，以 \n 结尾
 *    {"t":120,"p1":{"x":50,"y":50,"a":45,"hp":100},"p2":{"x":450,"y":450,"a":225,"hp":100}}
 *    - t: 剩余时间 (秒)
 *    - p1/p2: 坐标 (x,y), 角度 (a), 血量 (hp)
 * 
 * 2. 收包格式 (STM32 -> Web): 以 \n 结尾的 JSON 字符串
 *    {"mv":1,"rt":0.5,"fr":1}
 *    - mv: 移动 (-1 到 1)
 *    - rt: 转向 (-1 到 1)
 *    - fr: 开火 (0 或 1)
 * 
 * 3. 核心规则:
 *    - 地图大小: 500x500
 *    - 护盾: 位于背后 (角度 + 180°), 宽度 120°, 击中不扣血
 *    - 伤害: 身体被击中扣 10 HP
 *    - 冷却: 开火后有约 0.25s 冷却
 */

typedef struct {
    float x, y, a;
    int hp;
} Player;

typedef struct {
    float mv; // -1 to 1
    float rt; // -1 to 1
    int fr;   // 0 or 1
} Command;

// 极简 JSON 解析 (针对 C8T6 建议使用此方法或 cJSON)
void parse_game_state(const char* json, float* time, Player* self, Player* enemy, int is_p1) {
    // 实际项目中建议使用 sscanf 或专用解析器
    // 示例解析 P1 和 P2 数据
    if (is_p1) {
        sscanf(json, "{\"t\":%f,\"p1\":{\"x\":%f,\"y\":%f,\"a\":%f,\"hp\":%d},\"p2\":{\"x\":%f,\"y\":%f,\"a\":%f,\"hp\":%d}}",
               time, &self->x, &self->y, &self->a, &self->hp, &enemy->x, &enemy->y, &enemy->a, &enemy->hp);
    } else {
        sscanf(json, "{\"t\":%f,\"p1\":{\"x\":%f,\"y\":%f,\"a\":%f,\"hp\":%d},\"p2\":{\"x\":%f,\"y\":%f,\"a\":%f,\"hp\":%d}}",
               time, &enemy->x, &enemy->y, &enemy->a, &enemy->hp, &self->x, &self->y, &self->a, &self->hp);
    }
}

// 核心逻辑: 简单的“面朝敌人攻击，背朝敌人防御”策略
void update_logic(Player* self, Player* enemy, Command* cmd) {
    float dx = enemy->x - self->x;
    float dy = enemy->y - self->y;
    float dist = sqrtf(dx*dx + dy*dy);
    
    // 目标角度 (面朝敌人)
    float target_angle = atan2f(dy, dx) * 180.0f / M_PI;
    if (target_angle < 0) target_angle += 360.0f;
    
    // 角度差计算
    float angle_diff = target_angle - self->a;
    while (angle_diff > 180) angle_diff -= 360;
    while (angle_diff < -180) angle_diff += 360;

    // --- 策略逻辑 ---
    
    // 1. 转向逻辑: 尽量对准敌人
    if (fabs(angle_diff) > 5.0f) {
        cmd->rt = (angle_diff > 0) ? 1.0f : -1.0f;
    } else {
        cmd->rt = 0;
    }

    // 2. 移动逻辑: 保持距离 (约 200)
    if (dist > 220) cmd->mv = 1.0f;
    else if (dist < 180) cmd->mv = -1.0f;
    else cmd->mv = 0;

    // 3. 防御逻辑扩展: 如果血量低且敌人在瞄准我，可以考虑转身 180 度用背后护盾挡子弹
    // (此处仅为示例，未实现完整防御姿态切换)

    // 4. 开火逻辑: 角度基本对准且有一定距离时开火
    if (fabs(angle_diff) < 15.0f && dist < 400) {
        cmd->fr = 1;
    } else {
        cmd->fr = 0;
    }
}

// 串口发送函数
void send_command(Command* cmd) {
    char buffer[64];
    sprintf(buffer, "{\"mv\":%.2f,\"rt\":%.2f,\"fr\":%d}\n", cmd->mv, cmd->rt, cmd->fr);
    // UART_SendString(buffer); 
    printf("%s", buffer);
}

int main() {
    // 模拟运行循环
    char mock_input[] = "{\"t\":115.5,\"p1\":{\"x\":100,\"y\":100,\"a\":0,\"hp\":100},\"p2\":{\"x\":300,\"y\":300,\"a\":180,\"hp\":100}}";
    float time;
    Player p1, p2;
    Command cmd = {0};

    parse_game_state(mock_input, &time, &p1, &p2, 1);
    update_logic(&p1, &p2, &cmd);
    send_command(&cmd);

    return 0;
}
