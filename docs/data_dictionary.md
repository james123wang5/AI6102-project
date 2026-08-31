# Optiver 数据字典

| 字段 | 含义 | 第一阶段处理 |
|---|---|---|
| `stock_id` | 股票编号 | 分组、统计与类别特征 |
| `date_id` | 交易日编号 | 时间切分和日内重置 |
| `seconds_in_bucket` | 收盘前十分钟中的秒数 | 排序与收盘阶段特征 |
| `time_id` | 交易时刻编号 | 对齐测试和提交 |
| `row_id` | 提交行标识 | 仅对齐，不作为模型输入 |
| `imbalance_size` | 竞价中未匹配订单量 | 构造竞价失衡特征 |
| `imbalance_buy_sell_flag` | 未匹配订单方向 | 与未匹配量组合 |
| `reference_price` | 竞价参考价格 | 构造价格偏离特征 |
| `matched_size` | 已匹配订单量 | 计算未匹配比例 |
| `far_price` / `near_price` | 竞价指示价格 | 先审计缺失，再构造价格关系 |
| `bid_price` / `ask_price` | 最优买价 / 最优卖价 | 构造 spread、mid price |
| `bid_size` / `ask_size` | 最优买盘量 / 最优卖盘量 | 构造流动性失衡 |
| `wap` | 加权平均价格 | 构造短期变化与偏离特征 |
| `target` | 未来 60 秒相对合成指数变化 | 训练标签；测试不可见 |
| `currently_scored` | 示例测试计分标识 | 仅见于示例测试流，不作输入 |
| `revealed_target` | 后验揭示的旧标签 | 第一阶段不作特征，防止泄漏 |

`far_price` 和 `near_price` 在部分时刻缺失可能是正常竞价状态。先报告缺失比例与位置，再决定后续处理。
