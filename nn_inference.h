#ifndef __NN_INFERENCE_H
#define __NN_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  神经网络推理：13输入 -> 48隐藏1 -> 32隐藏2 -> 3输出
 * @param  obs     13维归一化观测向量
 * @param  actions 3维输出 (mv, rt, fire_logit)
 */
void nn_inference(const float obs[13], float actions[3]);

/**
 * @brief  特征工程 + 推理（完整流程）
 * @param  self_x/y/a/hp  自身坐标、朝向(度)、血量(0~100)
 * @param  enemy_x/y/a/hp 敌方坐标、朝向(度)、血量(0~100)
 * @param  self_cd        冷却计数
 * @param  fire_cooldown  最大冷却
 */
void process_game_data(float self_x,  float self_y,  float self_a,  float self_hp,
                       float enemy_x, float enemy_y, float enemy_a, float enemy_hp,
                       float self_cd, float fire_cooldown);

#ifdef __cplusplus
}
#endif

#endif /* __NN_INFERENCE_H */
											 