# 阶段 02：基线模型

本阶段冻结时间验证协议，并比较三类基线：

- Zero：所有预测为 0。
- Median：预测该折训练标签中位数。
- Raw LightGBM：使用原始可用字段；`stock_id` 是类别，`date_id` 只负责切分与报告。

开发折固定为：

| 折 | 训练日期 | 验证日期 |
|---|---|---|
| fold_1 | 0～239 | 240～289 |
| fold_2 | 0～289 | 290～339 |
| fold_3 | 0～339 | 340～399 |

日期 400～480 是最终留出集，本阶段不使用它。每一折从训练末 20 天决定 LightGBM 的早停轮数，再使用整段训练日期重训并预测该折验证日期。

运行：

```bash
.venv/bin/python scripts/run_baselines.py
```

如需先单独运行一折、确认运行时间和内存：

```bash
.venv/bin/python scripts/run_baselines.py --fold fold_1
```

每完成一折，脚本都会立即写入当前结果；中断后已完成折不会丢失。

输出在 `reports/02_baselines/`：汇总 MAE、逐日 MAE、各折样本外预测及运行配置。模型文件放入不纳入 Git 的 `artifacts/02_baselines/`。
