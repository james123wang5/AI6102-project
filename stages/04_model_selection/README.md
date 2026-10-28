# 阶段 04：模型选择与优化主线

主模型是 LightGBM（CPU，`regression_l1`）；XGBoost（GPU，`reg:absoluteerror`）只作为"换一种梯度提升树"的对照实验，不替代主线。两者用相同的开发折、特征、早停协议和同样大小的调参网格。

三批依次运行，每批读取上一批的结果，选择规则写死在代码里；最终留出集只在第三批最后评估一次。

| 批次 | 脚本 | 内容 |
|---|---|---|
| 1 | `scripts/run_stage04_batch1.py` | 三折 Zero/Median；LightGBM 3 组参数调参；用最优参数重跑 raw→alpha101 累加消融；XGBoost GPU 3 组参数对照；只用训练日期拟合合成指数权重并报告 R² |
| 2 | `scripts/run_stage04_batch2.py` | （LightGBM）累加全市场汇总、合成指数相对特征、股票级统计；中性化（不做／等权／指数权重）× 收缩系数 |
| 3 | `scripts/run_stage04_batch3.py` | 近期加权对比、多种子平均、重新选后处理、按天块 bootstrap、误差拆分；XGBoost 与融合对照（只报告不选择）；冻结配置后的留出评估 |

提交方式见 `slurm/README.md`。结果写入 `reports/04_xgb_gpu/`，每批有 `summary.md`。

新增模块：

- `src/ai6102/index_model.py`：由训练标签反推指数收益，非负最小二乘拟合 200 只股票的静态权重；中性化后处理。
- `src/ai6102/fold_features.py`：需要在每折训练日期上拟合的特征（指数相对特征、股票级统计）。
- `src/ai6102/features/market.py`：同一时刻的全市场汇总特征，不需要拟合。
- `src/ai6102/stage04.py`：三批共用的数据缓存、LightGBM/XGBoost 训练、运行登记与评估工具。
