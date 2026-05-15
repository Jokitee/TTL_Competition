#ifndef __NN_INFERENCE_H
#define __NN_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file   nn_inference.h
 * @brief  战斗 AI 神经网络推理接口 v5.0 — 分支架构 (Branched Architecture)
 *
 * 网络结构:
 *   16 -> Shared(64) -> [Offense(48), Tactical(48)]
 *                    ->  [rt(3)/fire(1)]  ,  [mv(3)]
 *
 * 进攻火控分支 (Offense Branch):
 *   专精炮塔旋转控制与开火时机判断，利用 vel_perp 实现移动目标预判射击。
 *
 * 战术走位分支 (Tactical Branch):
 *   专精底盘机动、侧翼绕后与残血规避，不受火控梯度干扰。
 */

/**
 * @brief  神经网络推理主函数
 * @param  obs     16维归一化观测向量（须与 Python get_relative_obs() 严格对齐）
 *   [0]  dist         相对距离       (归一化 0~1)
 *   [1]  angle_diff   枪口误差角     (-1~1，有符号)
 *   [2]  dx           X轴差          (-1~1)
 *   [3]  dy           Y轴差          (-1~1)
 *   [4]  self_hp      自身血量       (0~1)
 *   [5]  enemy_hp     敌人血量       (0~1)
 *   [6]  exposure     敌方暴露系数   (0~1，>0.5表示肉身暴露在护盾外)
 *   [7]  wall_dist    最近墙距       (0~1)
 *   [8]  sin(self_a)  自身朝向sin
 *   [9]  cos(self_a)  自身朝向cos
 *   [10] sin(enemy_a) 敌人朝向sin
 *   [11] cos(enemy_a) 敌人朝向cos
 *   [12] self_cd      冷却状态       (0=可立即开火)
 *   [13] vel_along    敌方沿视线速度 (-1~1)
 *   [14] dist_rate    距离变化率     (-1~1)
 *   [15] vel_perp     敌方垂直视线速度(-1~1，预判射击核心特征)
 *
 * @param  actions 3维离散动作输出:
 *   actions[0] mv   底盘移动  {-1(后退), 0(停止), +1(前进)}
 *   actions[1] rt   炮塔旋转  {-1(右转), 0(不转), +1(左转)}
 *   actions[2] fire 开火决策  {0(不开火), 1(开火)}
 */
void nn_inference(const float obs[16], float actions[3]);

/**
 * @brief  特征工程 + 推理完整流程（硬件直接调用入口）
 *
 * @param  self_x/y/a/hp  自身 X,Y坐标(0~500), 朝向(度), 血量(0~100)
 * @param  enemy_x/y/a/hp 敌方 X,Y坐标(0~500), 朝向(度), 血量(0~100)
 * @param  self_cd        当前冷却计数 (0=可立即开火)
 * @param  fire_cooldown  最大冷却帧数 (训练时固定为 15)
 * @param  prev_dist      上一帧与敌方的距离（首次调用传入 -1.0f）
 *
 * @note   内部将自动完成特征归一化并调用 nn_inference()，
 *         输出 actions[] 后通过 UART 或 PWM 发送给运动控制模块。
 */
void process_game_data(float self_x,  float self_y,  float self_a,  float self_hp,
                       float enemy_x, float enemy_y, float enemy_a, float enemy_hp,
                       float self_cd, float fire_cooldown, float prev_dist);

#ifdef __cplusplus
}
#endif

#endif /* __NN_INFERENCE_H */