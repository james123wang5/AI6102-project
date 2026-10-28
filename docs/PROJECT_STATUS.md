# 项目说明：当前进度、结果与后续方向

更新日期：2026-10-28。本文档说明到目前为止做了什么、结果如何、怎样复现、还缺什么，以及推荐的下一步。所有数字都来自 `reports/` 下的实际输出文件，括号里给出出处。

## 0. 一页摘要

- **任务**：Optiver "Trading at the Close"（已结束的 Kaggle 比赛）。预测每只股票未来 60 秒相对合成指数的收益（基点），指标 MAE。课程选项为"Completed Kaggle Competition"，报告 11 月 20 日交给 yuan.zhou@ntu.edu.sg。
- **验证协议**：日期 0–399 内做三折按时间前推的开发验证（dev）；日期 400–480 作为留出集（holdout），所有选择冻结后只评估一次。这个留出区间与第 1 名公开方案的验证区间完全相同，可直接对比。
- **主线**：LightGBM（CPU，L1 目标）+ 调参 + 累加特征组 + 由训练标签反推的合成指数权重（R² = 0.99996）+ 指数特征 + 指数加权中性化与收缩 + 近期加权 + 三种子平均。XGBoost（GPU）只作模型族对照。
- **结果（holdout 400–480，全零预测 MAE 5.9936）**：

  | 模型 | MAE | 相对全零 |
  |---|---|---|
  | 原始 13 个字段 LightGBM | 5.9104 | −1.39% |
  | V1 全特征（阶段 03 最优） | 5.8711 | −2.04% |
  | **最终主线** | **5.8641** | **−2.16%** |
  | [对照] XGBoost + 同样后处理 | 5.8605 | −2.22% |
  | [对照] LightGBM/XGBoost 50/50 | 5.8599 | −2.23% |

  最终主线相对 V1 的改善为 −0.0070，按天 bootstrap 95% CI [−0.0110, −0.0036]，81 天重采样中全部为负（更好）。
- **差距**：第 1 名单个 CatBoost 在同一区间为 5.8240（−2.83%），我们落后约 0.040。
- **主要缺口**：报告要求的排行榜截图尚未获得；Optiver 是否在课件第 5 页的候选比赛名单中尚未确认（该页为图片）。

## 1. 问题与数据

- 数据：481 个交易日 × 约 200 只股票 × 每天收盘前 10 分钟每 10 秒一行（`seconds_in_bucket` 0–540，共 55 个时点），约 524 万行。原始字段为订单簿（买卖价、买卖量、WAP）和集合竞价信息（参考价、匹配量、不平衡量及方向、near/far price）。
- 标签：`target = 10000 × (个股 WAP 60 秒收益 − 合成指数 60 秒收益)`。合成指数是 200 只股票的加权组合，官方未公布权重。
- 难点：信噪比极低（好模型只比全零预测低 2–3%）；时间序列非平稳（不同时间段全零 MAE 从 5.99 到 6.44 不等，因此不同区间的 MAE 不能直接比较，只能比"相对全零的降幅"）；必须严格防止未来信息泄露。

## 2. 验证协议

| 名称 | 训练日期 | 评估日期 | 评估行数 |
|---|---|---|---|
| fold_1 | 0–239 | 240–289 | 547,250 |
| fold_2 | 0–289 | 290–339 | 549,669 |
| fold_3 | 0–339 | 340–399 | 659,945 |
| holdout | 0–399 | 400–480 | 890,999 |

- `pooled`：三折按评估行数加权的 MAE，是 dev 上所有选择的依据。
- 每次训练：训练集最后 20 天作内部早停集确定最佳轮数，然后在全部训练日期上按该轮数重训（refit）。
- 需要"拟合"的东西（指数权重、个股统计）只用该折训练日期拟合。
- holdout 只在 `frozen_config.json` 写定后评估一次（`reports/04_xgb_gpu/batch3/`）。**之后如果再改方案并重新看 holdout，报告里必须说明这是第二次使用，并且新方案的每个选择都必须先在 dev 上决定。**

## 3. 已完成的阶段与结果

### 阶段 01–02：数据审计与基线

- 数据审计：`reports/01_data_audit/`。
- fold_1 上：Zero 6.0173，Median 6.0178，原始字段 LightGBM 5.9612（`reports/02_baselines/baseline_summary.csv`）。

### 阶段 03：特征组累加实验（LightGBM，默认参数）

三折简单平均 MAE（`reports/03_feature_experiments/mae_summary.csv`）：

| 特征组（累加） | 平均 MAE |
|---|---|
| raw 原始字段 | 6.2209 |
| + optiver 微观结构 | 6.1979 |
| + temporal 日内时序 | 6.1822 |
| + cross_section 横截面 | 6.1805 |
| + alpha101 Alpha101-inspired | 6.1802 |

### 阶段 04 第一批：基线、调参、消融、指数权重（`reports/04_xgb_gpu/batch1/summary.md`）

- 三折 pooled 基线：Zero 6.30152，Median 6.30146。
- LightGBM 调参（V1 全特征）：l63 6.18561，l127 6.18422，**l255 6.18322（选中）**。
- XGBoost GPU 对照：d6 6.18247，d8 6.18036，d10 6.17899。三折均略优于 LightGBM（约 0.004）。
- 用 l255 重跑累加消融（pooled）：raw 6.22110 → optiver 6.20001 → temporal 6.18623 → cross_section 6.18170 → alpha101 6.18322。**Alpha101 组未带来收益（+0.0015）**。
- 合成指数拟合：由 `r_i − target/1e4` 反推每个时点的指数收益，用 NNLS 对个股 60 秒收益回归。四次拟合（三折 + holdout 训练段）R² 均为 0.99996，权重相关性 ≥ 0.998，隐含指数收益的跨股票离散度中位数 0.005 bp，说明标签定义被精确复现。

### 阶段 04 第二批：新特征组与后处理（`reports/04_xgb_gpu/batch2/summary.md`）

累加特征组（pooled）：

| 特征组 | 特征数 | pooled | 变化 |
|---|---|---|---|
| E5 V1 全特征 | 78 | 6.18322 | — |
| N1 + 全市场汇总（等权） | 85 | 6.18291 | −0.0003 |
| N2 + 合成指数特征 | 96 | 6.17912 | **−0.0038** |
| N3 + 个股统计 | 103 | 6.17993 | +0.0008 |

后处理（每个时点减去预测的横截面均值，再乘收缩系数 α）：

- 在 4 个特征集上，`index`（指数权重加权均值）都优于 `equal`（等权）和 `none`。原因是标签按定义满足"指数加权和为零"。
- 最优 α 在 0.9–0.95。
- 按事先规则选中 s_stock + index + α 0.9，pooled 6.17614。s_index + index + α 0.9 为 6.17619，两者在噪声范围内打平。真正起作用的是指数特征，个股统计特征单独看没有收益。

### 阶段 04 第三批：最终配置与留出评估（`reports/04_xgb_gpu/batch3/summary.md`）

dev（pooled）：

| 配置 | pooled | 相对全零 |
|---|---|---|
| E1 原始字段（调参后） | 6.22110 | −1.28% |
| E5 V1 全特征 | 6.18322 | −1.88% |
| s_stock 单种子 | 6.17993 | −1.93% |
| + 近期加权（半衰期 120 天）+ 三种子平均 | 6.17478 | −2.01% |
| **+ 指数中性化，α = 0.95（最终）** | **6.17140** | **−2.06%** |
| [对照] XGBoost d10 + 同样后处理 | 6.16982 | −2.09% |
| [对照] 50/50 融合 + 后处理 | 6.16779 | −2.12% |

- 按天 bootstrap（160 天）：最终 − V1 = −0.01182，95% CI [−0.01501, −0.00872]；最终 − raw = −0.04971。
- 误差拆分：改善集中在收盘后段（420–540 秒相对 V1 −0.43%，0–120 秒仅 −0.06%）和高流动性股票（Q4/Q5 约 −0.3%，Q1 约 −0.1%）。
- 冻结配置：`{"model": "lgb", "params_name": "l255", "feature_set": "s_stock", "recency": true, "seeds": [2026, 2027, 2028], "mode": "index", "alpha": 0.95}`。
- holdout 结果见第 0 节。dev 改善 0.0118，holdout 改善 0.0070，方向一致。

运行耗时（集群，10 CPU + 1×A40）：三批共 57 次训练，训练时间合计约 3.7 小时。LightGBM 每次 1–10 分钟，XGBoost GPU 每次约 1 分钟。

## 4. 与公开方案的对比（只作参照）

| 方案 | 区间 | 全零 | MAE | 相对全零 |
|---|---|---|---|---|
| 第 1 名单个 CatBoost，无/有后处理 | 400–480 | 5.9936 | 5.8287 / 5.8240 | −2.75% / −2.83% |
| 第 1 名单个 GRU / Transformer | 400–480 | 5.9936 | 5.8481 / 5.8619 | −2.43% / −2.20% |
| 第 1 名 CatBoost + GRU + Transformer | 400–480 | 5.9936 | 5.8117 | −3.03% |
| 第 9 名单个 XGBoost（157 特征） | 390–480 | 6.0309 | 5.8705 | −2.66% |
| **我们的最终主线** | 400–480 | 5.9936 | **5.8641** | **−2.16%** |

全零 MAE 由本地 `train.csv` 计算。第 1 名私榜 5.4030，前 10 名在 5.40–5.44（私榜是比赛后的新行情，不能和本地区间直接比）。

来源：
- 第 1 名：https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution
- 第 9 名：https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/adam-9th-place-solution ，代码 https://github.com/ChunhanLi/9th-kaggle-optiver-trading-close
- 第 5、14 名关于后处理的讨论：https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/485985

公开方案中与我们一致的做法：指数加权去均值后处理（第 1、5、9、14 名都用了）、个股中位数缩放的 size 特征、近期样本加权（第 9 名最后 45 天 1.5 倍）。我们自己完成、他们没有展示的部分：从数据反推指数权重并按折拟合、三折前推验证 + 一次性 holdout、逐组消融、bootstrap 置信区间、误差拆分。

## 5. 代码结构与复现

```text
src/ai6102/
  features/        阶段 03 特征：optiver.py, temporal.py, cross_section.py, alpha_adapted.py, pipeline.py
  features/market.py   同一时点的全市场汇总特征（等权）
  index_model.py   由标签反推指数收益，NNLS 拟合权重；neutralize() 中性化
  fold_features.py 需要按折拟合的特征：指数相对特征、个股统计
  stage04.py       三批共用：特征缓存、LightGBM/XGBoost 训练、运行登记 (runs.csv + 预测 pkl)、评估工具
scripts/
  run_baselines.py, run_feature_experiments.py   阶段 02/03
  run_stage04_batch{1,2,3}.py                     阶段 04 三批
configs/
  baseline.json    数据路径、折定义
  stage04.json     阶段 04 参数网格、种子、收缩系数等
slurm/
  stage04.sbatch   集群作业：sbatch slurm/stage04.sbatch <1|2|3>
reports/           所有实验输出（已提交）
artifacts/         特征缓存与预测文件（不提交，见 .gitignore）
```

集群（NTU TC2，分区 MGPU-TC2）复现：

```bash
module load anaconda/25.5.1 && eval "$(conda shell.bash hook)" && conda activate optiver
pip install "xgboost>=2.1" scipy
cd ~/AI6102-project
sbatch slurm/stage04.sbatch 1      # 依次运行 1、2、3，每批读取上一批结果
tail -f slurm_stage04_<job_id>.out
```

- 每次训练有唯一 key（模型/特征集/参数/近期加权/种子/折），结果写入 `reports/04_xgb_gpu/runs.csv`，预测写入 `artifacts/04_xgb_gpu/preds/`。作业中断后重新提交会跳过已完成的训练。
- 本地冒烟测试：加 `--smoke-stocks 12 --smoke-rounds 40 --xgb-device cpu`，输出写到 `reports/04_smoke/`。
- 特征缓存 `artifacts/04_xgb_gpu/features_v2_full.pkl` 约 3 GB，首次构建 15–30 分钟。

## 6. 已知问题

1. **集群共享盘曾写满**：第三批第一次运行因 `/home/msai`（10 TB 共享盘）满而报 `No space left on device`，重新提交后完成。后续代码建议把缓存改为 float32，并在开跑前检查剩余空间。
2. **stock_id 增益过高**：在 s_stock 模型中，`stock_id`（作为类别特征）的 gain 约为第二名特征的 7 倍，可能在记忆个股特性，存在过拟合风险，尚未做对照实验。
3. **Alpha101 组无效**：消融显示它没有带来收益，报告中应如实写出。
4. **s_stock 与 s_index 打平**：最终配置选中 s_stock 是按事先规则，但差异在噪声内。
5. **本地没有跑单元测试**：本地虚拟环境未安装 pytest，`tests/` 未在本轮重新运行。三批脚本都通过了本地冒烟测试，全量结果来自集群。

## 7. 还缺什么

| 项目 | 状态 | 说明 |
|---|---|---|
| 排行榜截图 | **缺** | 课件要求"交叉验证结果和排行榜截图"。比赛是 API 代码赛，迟交需要写 Kaggle 推理 notebook。做不到的话，用"与第 1 名同区间 holdout"对比并在报告中说明 |
| 选题是否在候选名单 | **待确认** | 课件第 5 页候选比赛列表是图片，需人工查看；不在名单内应尽快邮件问老师 |
| 报告 | 未开始 | 建议结构见第 8.3 节 |
| 树模型补强（第四批） | 未开始 | 见第 8.1 节 |
| 深度学习探索（第五批） | 未开始 | 见第 8.2 节 |

## 8. 推荐方向

### 8.1 第四批：树模型补强（优先）

目标：holdout 从 5.864 推进到 5.83–5.85。每一项都先在 dev 三折上验证，有效才保留。思路受第 9 名启发，但用我们自己的实现、按折防泄露，并逐项消融，报告中注明来源。

按预期收益排序：

1. **跨天历史 target 特征**：对同一 (stock_id, seconds_in_bucket)，取此前 1/2/3/5/10/…/60 天 target 的滚动均值和标准差。前一天的 target 次日已知，不构成泄露（比赛 API 也会提供）。
2. **当天已实现 target 的滚动统计**：利用精确还原的指数权重，算出 10/30/60 秒前已实现的相对收益（我们已有 `rel_return_*` 当前值，排在重要性前列），再在当天内做 24/48 步滚动均值。
3. **stock_id 处理对照**：类别 / 数值 / 去掉，三选一。
4. **订单簿组合特征**：价格与量的 `(a−b)/(a+b)` 不平衡度全组合、集合竞价买卖量、`market_urgency` 变体、near_price 的 MACD、不平衡量的 shift/diff；全市场汇总改为指数加权。
5. **低学习率精调**：学习率 0.01–0.02、`feature_fraction` 约 0.5，特征定下来之后做一次（每次训练约 15–25 分钟）。
6. **滚动重训**：在 holdout 期间每 20 天左右把已揭晓的日期并入训练集重训。先在 fold_3 上模拟验证。第 1 名报告这一项让 CatBoost 私榜从 5.4523 降到 5.4165，是他所有改进里最大的一项。

### 8.2 第五批：深度学习探索（可选，用 A40）

参照第 1 名公开结构，用与 LightGBM 相同的特征：

- **GRU（时间轴）**：每个 (stock, day) 一条长度 55 的序列，约 9.6 万条；4 层 GRU，逐时点输出。
- **Transformer（股票轴）**：每个 (day, seconds) 把约 200 只股票作为一组，约 2.6 万组；4 层 encoder；输出减去**指数加权**均值（利用我们反推的权重，第二批已证明加权优于等权）。
- 资源：数据 float32 约 2 GB，A40 48 GB 显存足够。预计每次训练 10–25 分钟，3 折 + holdout × 2 个模型 + 少量调参，合计约 4–7 小时，分两个作业跑。以上为估计，首次运行后校准。
- 合理预期：第 1 名的单个 GRU/Transformer 都不如他的单个 CatBoost，深度学习的收益主要来自与树模型融合（约 0.01）。第 1 名试过无效的：1D-CNN/MLP 融合、多天输入的 GRU、更大的 Transformer。
- 建议放在第四批之后，因为特征定下来再训练神经网络更划算。

### 8.3 报告建议结构

1. 问题陈述（相对指数收益、MAE、信噪比低、非平稳）
2. 难点（泄露风险、时间漂移、未知指数权重）
3. 方法：验证协议 → 特征组 → 指数权重反推 → 指数特征 → 加权中性化与收缩 → 近期加权与多种子
4. 实验：消融表（阶段 03 + 第一批）、后处理表（第二批）、dev 对比与 bootstrap（第三批）、误差拆分、模型族对照、holdout 结果与公开方案对比
5. 结论与局限（Alpha101 无效、stock_id 风险、与头部的差距、深度学习的未来工作）

## 9. 结果文件索引

| 文件 | 内容 |
|---|---|
| `reports/04_xgb_gpu/runs.csv` | 每次训练的 key、MAE、最佳轮数、耗时、前 10 重要特征 |
| `reports/04_xgb_gpu/batch1/` | baselines、tuning、ablation、index_fit、index_weights、best_params、importance |
| `reports/04_xgb_gpu/batch2/` | feature_groups、postprocess_grid、postprocess_summary、choice |
| `reports/04_xgb_gpu/batch3/` | dev_comparison、seeds、bootstrap、error_by_phase/liquidity/day、frozen_config、holdout、holdout_bootstrap |
