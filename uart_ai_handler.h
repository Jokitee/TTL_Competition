#ifndef __UART_AI_HANDLER_H
#define __UART_AI_HANDLER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  初始化AI UART接收（启动中断接收）
 */
void AI_UART_Init(void);

/**
 * @brief  AI UART接收完成回调
 * @note   需在 HAL_UART_RxCpltCallback 中调用
 */
void AI_UART_RxCallback(void);

#ifdef __cplusplus
}
#endif

#endif /* __UART_AI_HANDLER_H */
