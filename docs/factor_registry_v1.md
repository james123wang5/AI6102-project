# Factor Registry · V1

此表登记第三阶段的候选因子。它们尚未经过 IC、相关性或样本外 MAE 筛选；名称中的 `s` 表示精确秒级历史连接，`rows` 表示同一股票、同一交易日内的记录行窗口。

| 组别 | 因子/规则 | 含义与约束 |
|---|---|---|
| 微观结构 | `price_spread`、`mid_price`、`wap_mid_diff`、`reference_wap_diff` | 最优报价、WAP 与竞价参考价格的关系。 |
| 微观结构 | `liquidity_imbalance` | `(bid_size - ask_size) / (bid_size + ask_size)`；分母无效时为 NaN。 |
| 微观结构 | `auction_imbalance_ratio`、`signed_imbalance` | 竞价未匹配压力及其方向。 |
| 微观结构 | `market_urgency` | `price_spread × liquidity_imbalance`，是待验证的交互信号。 |
| 缺失状态 | `near_price_is_missing`、`far_price_is_missing` | 明确保留竞价指示价格缺失状态。 |
| 精确时间 | `{wap, price_spread, liquidity_imbalance, signed_imbalance}_{lag,diff}_{10,30,60}s` | 按同一股票、同一日期精确连接过去时刻；不会把缺失时间桶当成连续记录。 |
| 精确时间 | `wap_return_{10,30,60}s` | `(wap - lagged_wap) / lagged_wap`；无有效历史时为 NaN。 |
| 行窗口 | `wap_rolling_rows_{3,6,12}_{mean,std}` | 因果窗口，包含当前行；日切重置。 |
| 行窗口 | `liquidity_imbalance_rolling_rows_{3,6,12}_mean` | 因果行窗口；不宣称为精确秒窗口。 |
| 横截面 | `{price_spread, liquidity_imbalance, market_urgency, wap_return_10s}_{cs_rank,cs_mean_diff,cs_zscore}` | 仅使用同一 `(date_id, seconds_in_bucket)` 已到达的股票。 |
| Alpha-inspired | `alpha_wap_return_10s_ts_rank_rows_6` | 当前 10 秒 WAP 回报在自身近期行窗口中的百分位。 |
| Alpha-inspired | `alpha_liquidity_imbalance_decay_rows_3` | 当前与精确 10/30 秒过去失衡的线性加权组合。 |
| Alpha-inspired | `alpha_wap_momentum_vol_adjusted_rows_6` | 30 秒 WAP 变化相对近期 WAP 波动；分母无效时为 NaN。 |
| Alpha-inspired | `alpha_spread_imbalance_corr_rows_6` | spread 与流动性失衡在自身近期行窗口的相关性，最少 3 个有效观测。 |
| Alpha-inspired | `alpha_signed_imbalance_mean_rows_6` | 有方向竞价失衡的近期均值。 |

`Alpha-inspired` 表示借鉴 Alpha101 的算子结构（rank、delay、rolling correlation、decay），不是直接复用日频公式，也不代表已经有效。
