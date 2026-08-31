# 阶段 03：特征与因子研究

本阶段的候选因子代码已经实现，但尚未在完整训练集生成，也没有训练模型或筛选因子。

实现位置：

- `src/ai6102/features/optiver.py`：订单簿与集合竞价的可解释特征。
- `src/ai6102/features/temporal.py`：同一股票、同一交易日内的精确时间 lag 与行窗口特征。
- `src/ai6102/features/cross_section.py`：同一时刻股票横截面 rank、偏离和 z-score。
- `src/ai6102/features/alpha_adapted.py`：少量 Alpha101-inspired 高频候选。
- `src/ai6102/features/pipeline.py`：统一、因果且保持输入行顺序的构建入口。

因子公式、窗口单位和缺失规则见 [Factor Registry](../../docs/factor_registry_v1.md)。默认参数见 `configs/features_v1.json`。

关键约束：

- 不读取 `target`，因此不能把标签泄漏进特征。
- 精确秒级 lag 使用键连接；缺失历史时保留 NaN。
- `rolling_rows_N` 是记录行窗口，名称不会伪装成秒级窗口。
- 日内时序特征在 `date_id` 边界重置。
- 横截面特征仅按同一 `(date_id, seconds_in_bucket)` 计算。
- 近端/远期指示价格缺失被保留，并提供缺失指示特征。

阶段 04 才会在冻结的时间验证折上生成完整特征、做 IC/相关性诊断和消融训练。
