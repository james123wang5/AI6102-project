[English](README.md) | **中文**

# AI6102 课程项目 — Optiver: Trading at the Close

使用已结束的 Kaggle 比赛 [Optiver – Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close) 的数据，预测纳斯达克收盘集合竞价中每只股票未来 60 秒相对合成指数的收益。

本仓库包含 AI6102 课程报告背后的全部代码和实验输出。下文所有数字都来自 [`reports/`](reports/) 下的实际文件。

## 最终结果

离线留出集（日期 400–480，890,999 行；全零预测 MAE 5.9936）：

| 模型 | 留出 MAE | 相对全零 |
|---|---|---|
| LightGBM，13 个原始字段 | 5.9104 | −1.39% |
| LightGBM，V1 全特征 | 5.8711 | −2.04% |
| 第三批主线（指数特征、中性化、近期加权、三种子） | 5.8641 | −2.16% |
| 第四批 LightGBM（+ 跨天历史 target，stock_id 改为数值） | 5.8447 | −2.48% |
| GRU（时间轴）/ Transformer（股票轴）单模型 | 5.8648 / 5.8572 | −2.15% / −2.28% |
| **最终融合：LightGBM 0.55 + GRU 0.20 + Transformer 0.25** | **5.8393** | **−2.57%** |

融合减 LightGBM，按天配对 bootstrap：dev −0.00568（95% CI [−0.00698, −0.00431]）；留出 −0.00541（95% CI [−0.00978, −0.00261]）。

**排行榜。** 本比赛已关闭迟交（Kaggle 页面上 "Late Submission" 按钮为灰色），无法取得官方排行榜分数。我们的留出集与第 1 名公开方案的离线验证区间完全相同（日期 400–480），可以直接对照：

| 方案（日期 400–480） | MAE | 相对全零 |
|---|---|---|
| 第 1 名单个 CatBoost（含后处理） | 5.8240 | −2.83% |
| 第 1 名单个 GRU / Transformer | 5.8481 / 5.8619 | −2.43% / −2.20% |
| 第 1 名 CatBoost + GRU + Transformer | 5.8117 | −3.03% |
| **我们的最终融合** | **5.8393** | **−2.57%** |

来源：[第 1 名方案说明](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution)。最终私榜（第 1 名 5.4030）用的是比赛结束后的新行情，不能与离线 MAE 直接比较。

## 任务

股票 *i* 在时点 *t* 的标签：

```text
target = 10,000 × [ WAP_i(t+60s) / WAP_i(t) − Index(t+60s) / Index(t) ]   （基点）
```

其中指数是 200 只股票的加权组合，官方未公布权重。评价指标为 MAE。数据为 481 个交易日 × 约 200 只股票 × 每天 55 个 10 秒时点，共 5,237,980 行订单簿与集合竞价字段。

主要难点：信噪比极低（好模型只比全零预测低 2–3%）；非平稳（不同折的全零 MAE 从 6.02 到 6.44）；必须严格防止未来信息泄露；指数权重未知。

## 验证协议

| 名称 | 训练日期 | 评估日期 | 行数 |
|---|---|---|---|
| fold_1 | 0–239 | 240–289 | 547,250 |
| fold_2 | 0–289 | 290–339 | 549,669 |
| fold_3 | 0–339 | 340–399 | 659,945 |
| holdout | 0–399 | 400–480 | 890,999 |

- 所有设计选择都依据三折前推开发验证按行数加权的 **pooled** MAE。
- 每次训练用训练集最后 20 天早停，再在全部训练日期上按最佳轮数重训。
- 需要拟合的量（指数权重、个股统计、神经网络标准化参数）只在各自划分的训练日期内拟合。
- 留出集在第三、四、五批之后各评估一次（共三次）。每批都先在 dev 上冻结配置再看留出集，报告中如实说明。

## 方法

1. **特征组（LightGBM，L1 目标）**：原始字段 → 微观结构（价差、盘口与竞价不平衡）→ 日内历史（lag、diff、滚动统计）→ 同时刻横截面 → Alpha101-inspired 算子。
2. **反推合成指数**：由 `r_i − target/1e4` 得到隐含指数收益，跨股票取中位数，再用 NNLS 对个股收益回归。每个训练窗口 R² 均为 0.99996，不同窗口的权重相关性 ≥ 0.998。
3. **指数特征与指数加权中性化**：构造相对指数的个股收益特征；每个时点减去按指数权重加权的预测均值，再乘以在 dev 上选定的系数 α。
4. **近期加权**（半衰期 120 天）与**三种子平均**。
5. **跨天历史 target 特征**：同一（股票，时点）此前各天 target 的滚动统计，第 d 天只用 d−1 天及以前（比赛 API 次日会公布前一天 target）。stock_id 由类别特征改为数值特征。
6. **神经网络与融合**：GRU 沿每只股票当天 55 个时点读取；Transformer 在每个时点让所有股票互相注意（不加位置编码，缺失股票做掩码，输出做指数中性化）；3 层、宽度 128，训练协议与树模型相同（L1、近期加权、早停后重训）。融合权重与 α 在 dev 上网格搜索（步长 0.05）。

## 主要实验（pooled dev MAE；全零 = 6.30152）

**累加特征消融**（调参后 LightGBM，[`batch1/ablation.csv`](reports/04_xgb_gpu/batch1/ablation.csv)）：

| 输入 | 特征数 | Dev MAE |
|---|---|---|
| 原始字段 | 13 | 6.22110 |
| + 微观结构 | 25 | 6.20001 |
| + 日内历史 | 61 | 6.18623 |
| + 横截面 | 73 | 6.18170 |
| + Alpha101-inspired | 78 | 6.18322（无收益） |

**指数结构**（[`batch2/`](reports/04_xgb_gpu/batch2/)）：加入合成指数特征 6.18291 → 6.17912。固定特征、α = 1 时的中性化方式：不中性化 6.17912，等权 6.17944，**指数加权 6.17705**。

**流程逐步**（[`batch3/dev_comparison.csv`](reports/04_xgb_gpu/batch3/)）：单种子 6.17993 → + 近期加权与三种子 6.17478 → + 指数中性化（α = 0.95）**6.17140**。XGBoost（GPU）加同样后处理：6.16982（模型族对照）。

**历史 target 与 stock_id**（[`batch4/steps.csv`](reports/04_xgb_gpu/batch4/steps.csv)）：参照 6.17910 → + 历史 target 6.17571（三折全部变好）→ stock_id 改数值 6.16459 / 去掉 6.16473（三折全部变好）。当天相对收益滚动特征 6.17927（淘汰）。

**误差拆分**（[`batch3/`](reports/04_xgb_gpu/batch3/)）：改善集中在收盘后段（420–540 秒 −0.43%，0–120 秒仅 −0.06%）和高流动性股票（流动性最高两档约 −0.3%，最低一档 −0.11%）。

**最终模型逐折结果**（[`batch5/`](reports/04_xgb_gpu/batch5/)）：

| 模型 | Fold 1 | Fold 2 | Fold 3 | Dev pooled | 留出 |
|---|---|---|---|---|---|
| 全零 | 6.01730 | 6.43783 | 6.42367 | 6.30152 | 5.99364 |
| LightGBM | 5.89613 | 6.28155 | 6.26602 | 6.15566 | 5.84474 |
| GRU | 5.91285 | 6.29780 | 6.28100 | 6.17158 | 5.86482 |
| Transformer | 5.90909 | 6.28713 | 6.27793 | 6.16592 | 5.85717 |
| **融合** | **5.89383** | **6.27442** | **6.25874** | **6.14998** | **5.83933** |

## 仓库结构

```text
src/ai6102/
  features/            特征组：optiver、temporal、cross_section、alpha_adapted、market
  index_model.py       指数权重反推（NNLS）与指数加权中性化
  fold_features.py     按折拟合的特征（相对指数特征、个股统计）
  history_features.py  跨天历史 target 特征（第四批）
  deep.py              GRU 与 Transformer（第五批）
  stage04.py           共用的训练、缓存、运行登记与评估
scripts/               run_baselines.py、run_feature_experiments.py、run_stage04_batch{1..5}.py
configs/               数据路径、折定义、参数网格
slurm/                 集群作业脚本（NTU TC2）
reports/               全部实验输出（每批 CSV + summary.md）
docs/                  详细中文文档：PROJECT_STATUS（进度与结果）、BATCH5_DL_REPORT、数据字典、因子登记表、最初的研究计划 RESEARCH_PLAN.zh-CN
```

## 复现

```bash
# 数据：从 Kaggle 下载 train.csv 放到 optiver-trading-at-the-close/（未提交到仓库）
pip install -e .
python scripts/run_data_audit.py
python scripts/run_baselines.py
python scripts/run_feature_experiments.py
# 阶段 04 在 Slurm 集群上运行（每批读取上一批的输出）
sbatch slurm/stage04.sbatch 1   # 依次 2、3、4
sbatch slurm/stage05.sbatch     # 第五批：GRU / Transformer / 融合（1×A40）
```

本地冒烟测试：在阶段 04 脚本后加 `--smoke-stocks 12 --smoke-rounds 40 --xgb-device cpu`。特征缓存与预测文件（`artifacts/`，约 3 GB）未提交。

## 局限

- 留出集使用了三次（第三、四、五批之后），每次的选择都先在 dev 上冻结。
- 历史 target 特征假设预测时已知前一天的标签，与比赛 API 一致。
- Alpha101-inspired 特征在调参后没有收益；神经网络结构未调参。
- 比赛已关闭迟交，没有官方排行榜分数。
- MAE 只衡量预测精度，没有评估交易收益或成本。
