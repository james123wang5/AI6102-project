# 阶段 01：数据审计

本阶段不训练模型。目标是确认字段、数据规模、缺失模式和时间结构，为后续时间验证与特征工程提供依据。

在项目根目录运行：

```bash
.venv/bin/python scripts/run_data_audit.py
```

输出写入 `reports/01_data_audit/`：

- `data_audit.json`：供后续程序读取的摘要。
- `data_audit.md`：便于阅读的审计报告。

脚本使用流式 CSV 读取，不会将 5 百万行数据一次载入内存。完成本轮审计后，再实现按日期/股票的深入 EDA、键唯一性和标签边界检查。
