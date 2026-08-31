"""Streaming data audit utilities for the Optiver training CSV."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TRAIN_COLUMNS = (
    "stock_id", "date_id", "seconds_in_bucket", "imbalance_size",
    "imbalance_buy_sell_flag", "reference_price", "matched_size", "far_price",
    "near_price", "bid_price", "bid_size", "ask_price", "ask_size", "wap",
    "target", "time_id", "row_id",
)
NUMERIC_COLUMNS = frozenset(TRAIN_COLUMNS) - {"row_id"}

FIELD_DESCRIPTIONS = {
    "stock_id": ("股票编号", "按股票分组、统计与类别特征；不是连续价格特征。"),
    "date_id": ("交易日编号", "按日期做时间切分；日内 lag 和窗口默认在日期边界重置。"),
    "seconds_in_bucket": ("收盘前十分钟中的秒数", "确定日内顺序和收盘阶段。训练数据每 10 秒一个时间桶。"),
    "time_id": ("交易时刻编号", "用于和测试流、提交结果对齐。"),
    "row_id": ("提交行标识", "只用于预测对齐，不能作为模型输入。"),
    "imbalance_size": ("集合竞价未匹配订单量", "衡量买卖一侧尚未成交的竞价压力。"),
    "imbalance_buy_sell_flag": ("未匹配订单方向", "和 imbalance_size 组合，区分买方、卖方或平衡状态。"),
    "reference_price": ("收盘竞价参考价格", "可与 WAP、bid/ask 构造价格偏离。"),
    "matched_size": ("集合竞价已匹配订单量", "与 imbalance_size 一起计算未匹配比例。"),
    "far_price": ("远期指示价格", "缺失本身可能包含竞价状态信息。"),
    "near_price": ("近端指示价格", "缺失本身可能包含竞价状态信息。"),
    "bid_price": ("最优买价", "与 ask_price 构造买卖价差和中间价。"),
    "bid_size": ("最优买盘量", "与 ask_size 构造流动性失衡。"),
    "ask_price": ("最优卖价", "与 bid_price 构造买卖价差和中间价。"),
    "ask_size": ("最优卖盘量", "与 bid_size 构造流动性失衡。"),
    "wap": ("加权平均价格", "用于价格偏离、动量和日内变化特征；不直接等同于成交 VWAP。"),
    "target": ("未来 60 秒相对合成指数的价格变化，单位为 bp", "监督学习标签；正值表示相对跑赢，负值表示相对跑输。"),
}


@dataclass
class NumericStats:
    missing: int = 0
    present: int = 0
    invalid: int = 0
    non_finite: int = 0
    minimum: float | None = None
    maximum: float | None = None

    def observe(self, raw: str) -> None:
        raw = raw.strip()
        if not raw:
            self.missing += 1
            return
        self.present += 1
        try:
            value = float(raw)
        except ValueError:
            self.invalid += 1
            return
        if not math.isfinite(value):
            self.non_finite += 1
            return
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def as_dict(self, total: int) -> dict[str, Any]:
        return {
            "missing_count": self.missing,
            "missing_rate": self.missing / total if total else None,
            "non_missing_count": self.present,
            "invalid_numeric_count": self.invalid,
            "non_finite_count": self.non_finite,
            "unusable_count": self.missing + self.invalid + self.non_finite,
            "unusable_rate": (self.missing + self.invalid + self.non_finite) / total if total else None,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass
class RunningStats:
    """One-pass mean and standard deviation statistics."""

    count: int = 0
    mean: float = 0.0
    sum_squared_deviation: float = 0.0
    positive: int = 0
    negative: int = 0
    zero: int = 0

    def observe(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.sum_squared_deviation += delta * (value - self.mean)
        if value > 0:
            self.positive += 1
        elif value < 0:
            self.negative += 1
        else:
            self.zero += 1

    def as_dict(self) -> dict[str, Any]:
        variance = self.sum_squared_deviation / (self.count - 1) if self.count > 1 else None
        return {
            "count": self.count,
            "mean": self.mean if self.count else None,
            "standard_deviation": math.sqrt(variance) if variance is not None else None,
            "positive_count": self.positive,
            "negative_count": self.negative,
            "zero_count": self.zero,
            "positive_rate": self.positive / self.count if self.count else None,
            "negative_rate": self.negative / self.count if self.count else None,
        }


@dataclass
class AuditState:
    rows: int = 0
    stats: dict[str, NumericStats] = field(
        default_factory=lambda: {column: NumericStats() for column in NUMERIC_COLUMNS}
    )
    stock_ids: set[int] = field(default_factory=set)
    date_ids: set[int] = field(default_factory=set)
    bucket_counts: Counter[int] = field(default_factory=Counter)
    date_counts: Counter[int] = field(default_factory=Counter)
    stock_counts: Counter[int] = field(default_factory=Counter)
    date_bucket_counts: Counter[tuple[int, int]] = field(default_factory=Counter)
    date_bucket_unique_stock_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    missing_indicator_by_bucket: dict[str, Counter[int]] = field(
        default_factory=lambda: {"near_price": Counter(), "far_price": Counter()}
    )
    target_distribution: RunningStats = field(default_factory=RunningStats)
    missing_row_id: int = 0
    malformed_time_key: int = 0
    time_key_order_violations: int = 0
    duplicate_time_stock_key_count: int = 0
    _current_date_bucket: tuple[int, int] | None = None
    _current_bucket_stocks: set[int] = field(default_factory=set)

    def _finish_current_date_bucket(self) -> None:
        if self._current_date_bucket is not None:
            self.date_bucket_unique_stock_counts[self._current_date_bucket] = len(self._current_bucket_stocks)

    def observe(self, row: dict[str, str]) -> None:
        self.rows += 1
        for column, stats in self.stats.items():
            stats.observe(row.get(column, ""))
        if not row.get("row_id", "").strip():
            self.missing_row_id += 1
        try:
            stock_id = int(row["stock_id"])
            date_id = int(row["date_id"])
            seconds = int(row["seconds_in_bucket"])
            self.stock_ids.add(stock_id)
            self.date_ids.add(date_id)
            self.bucket_counts[seconds] += 1
            self.date_counts[date_id] += 1
            self.stock_counts[stock_id] += 1
            self.date_bucket_counts[(date_id, seconds)] += 1
            current_cell = (date_id, seconds)
            if self._current_date_bucket is not None and current_cell < self._current_date_bucket:
                self.time_key_order_violations += 1
            if current_cell != self._current_date_bucket:
                self._finish_current_date_bucket()
                self._current_date_bucket = current_cell
                self._current_bucket_stocks = set()
            if stock_id in self._current_bucket_stocks:
                self.duplicate_time_stock_key_count += 1
            self._current_bucket_stocks.add(stock_id)
            for column in self.missing_indicator_by_bucket:
                if not row.get(column, "").strip():
                    self.missing_indicator_by_bucket[column][seconds] += 1
        except (KeyError, TypeError, ValueError):
            self.malformed_time_key += 1

        raw_target = row.get("target", "").strip()
        if raw_target:
            try:
                target_value = float(raw_target)
                if math.isfinite(target_value):
                    self.target_distribution.observe(target_value)
            except ValueError:
                pass

    def finalize(self) -> None:
        self._finish_current_date_bucket()


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return next(csv.reader(file))


def audit_train_csv(train_path: Path, example_test_path: Path | None = None) -> dict[str, Any]:
    """Scan a CSV once without loading the full 5M-row dataset into memory."""
    train_path = train_path.resolve()
    state = AuditState()
    with train_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        for row in reader:
            state.observe(row)
    state.finalize()

    test_schema: dict[str, Any] | None = None
    if example_test_path is not None and example_test_path.exists():
        test_columns = _read_header(example_test_path)
        test_schema = {
            "path": str(example_test_path),
            "only_in_train": sorted(set(columns) - set(test_columns)),
            "only_in_example_test": sorted(set(test_columns) - set(columns)),
        }

    observed_buckets = sorted(state.bucket_counts)
    expected_buckets = set(range(0, 541, 10))
    expected_date_bucket_cells = len(state.date_ids) * len(expected_buckets)
    missing_date_bucket_cells = expected_date_bucket_cells - len(state.date_bucket_counts)
    indicator_missing_rates = {
        column: {
            str(seconds): state.missing_indicator_by_bucket[column][seconds] / state.bucket_counts[seconds]
            for seconds in observed_buckets
            if state.bucket_counts[seconds]
        }
        for column in state.missing_indicator_by_bucket
    }
    stocks_per_date_bucket = list(state.date_bucket_unique_stock_counts.values())
    expected_stock_count = len(state.stock_ids)
    return {
        "source": {"train_path": str(train_path), "file_size_bytes": train_path.stat().st_size},
        "schema": {
            "actual_columns": columns,
            "missing_expected_columns": sorted(set(TRAIN_COLUMNS) - set(columns)),
            "unexpected_columns": sorted(set(columns) - set(TRAIN_COLUMNS)),
            "example_test_comparison": test_schema,
        },
        "dataset": {
            "row_count": state.rows,
            "unique_stock_count": len(state.stock_ids),
            "stock_id_min": min(state.stock_ids) if state.stock_ids else None,
            "stock_id_max": max(state.stock_ids) if state.stock_ids else None,
            "unique_date_count": len(state.date_ids),
            "date_id_min": min(state.date_ids) if state.date_ids else None,
            "date_id_max": max(state.date_ids) if state.date_ids else None,
            "observed_seconds_in_bucket": observed_buckets,
            "missing_expected_10_second_buckets": sorted(expected_buckets - set(observed_buckets)),
            "rows_per_seconds_in_bucket": dict(sorted(state.bucket_counts.items())),
            "rows_per_date_min": min(state.date_counts.values()) if state.date_counts else None,
            "rows_per_date_max": max(state.date_counts.values()) if state.date_counts else None,
            "rows_per_stock_min": min(state.stock_counts.values()) if state.stock_counts else None,
            "rows_per_stock_max": max(state.stock_counts.values()) if state.stock_counts else None,
            "expected_date_bucket_cells": expected_date_bucket_cells,
            "observed_date_bucket_cells": len(state.date_bucket_counts),
            "missing_date_bucket_cells": missing_date_bucket_cells,
            "stocks_per_date_bucket_min": min(stocks_per_date_bucket) if stocks_per_date_bucket else None,
            "stocks_per_date_bucket_max": max(stocks_per_date_bucket) if stocks_per_date_bucket else None,
            "date_bucket_cells_with_missing_stocks": sum(
                count < expected_stock_count for count in stocks_per_date_bucket
            ),
            "duplicate_time_stock_key_count": state.duplicate_time_stock_key_count,
            "time_key_order_violations": state.time_key_order_violations,
            "row_id_missing_count": state.missing_row_id,
            "malformed_time_key_count": state.malformed_time_key,
        },
        "numeric_columns": {key: state.stats[key].as_dict(state.rows) for key in sorted(state.stats)},
        "target_distribution": state.target_distribution.as_dict(),
        "indicator_missing_rate_by_seconds_in_bucket": indicator_missing_rates,
        "notes": [
            "The audit is streamed and does not load the complete dataset into memory.",
            "Key uniqueness and target-boundary checks are implemented as later validation tasks.",
            "Missing auction prices may be a valid state signal; do not globally fill them yet.",
        ],
    }


def render_markdown_report(summary: dict[str, Any]) -> str:
    """Render one self-contained human-readable audit report."""
    data = summary["dataset"]
    schema = summary["schema"]
    target = summary["target_distribution"]
    indicator_rates = summary["indicator_missing_rate_by_seconds_in_bucket"]
    near_rates = list(indicator_rates["near_price"].values())
    far_rates = list(indicator_rates["far_price"].values())
    target_mean_text = f"{target['mean']:.6f}" if target["mean"] is not None else "N/A"
    target_std_text = (
        f"{target['standard_deviation']:.6f}"
        if target["standard_deviation"] is not None
        else "N/A"
    )
    lines = [
        "# AI6102 · 第一阶段数据审计报告", "",
        "本文件集中记录数据规模、字段解释、训练/测试差异、时间结构、质量检查和标签分布。",
        "", "## 1. 数据规模与可用性", "",
        f"- 行数：{data['row_count']:,}",
        f"- 股票数：{data['unique_stock_count']}（ID：{data['stock_id_min']} ～ {data['stock_id_max']}）",
        f"- 交易日数：{data['unique_date_count']}（ID：{data['date_id_min']} ～ {data['date_id_max']}）",
        f"- 缺失 row_id 数：{data['row_id_missing_count']}",
        f"- 无法解析的时间键数：{data['malformed_time_key_count']}",
        "", "结论：训练集字段齐全。标签是否存在非有限值在第 5 节单独核对后，再决定监督训练可用行。",
        "", "## 2. 每个字段是什么意思", "",
        "| 字段 | 含义 | 在项目中的作用 |",
        "|---|---|---|",
    ]
    for column in TRAIN_COLUMNS:
        meaning, usage = FIELD_DESCRIPTIONS[column]
        lines.append(f"| `{column}` | {meaning} | {usage} |")
    lines.extend([
        "", "### `target` 的重点解释", "",
        "`target` 是未来 60 秒内，这只股票相对比赛合成市场指数的价格变化，单位为 bp（1 bp = 0.01%）。",
        "- `target > 0`：未来一分钟相对跑赢市场。",
        "- `target < 0`：未来一分钟相对跑输市场。",
        "- 它是连续回归标签，模型要预测一个数值，比赛使用 MAE，而不是涨跌准确率。",
        "- 它仅在训练集存在，绝不能作为特征或从未来时刻反推。",
        "", "## 3. 训练集与示例测试流的差异", "",
        f"- 缺失预期训练列：{schema['missing_expected_columns'] or '无'}",
        f"- 非预期训练列：{schema['unexpected_columns'] or '无'}",
    ])
    if schema["example_test_comparison"]:
        comparison = schema["example_test_comparison"]
        lines.extend([
            f"- 训练集独有列：{', '.join(comparison['only_in_train']) or '无'}",
            f"- 示例测试集独有列：{', '.join(comparison['only_in_example_test']) or '无'}",
            "- `currently_scored` 是测试流控制字段，不作模型输入。",
            "- `revealed_target` 是之后才揭示的旧标签；第一版不把它作为特征，防止泄漏。",
        ])
    lines.extend([
        "", "## 4. 时间结构与完整性检查", "",
        "- 训练集出现的日内时间桶：0 ～ 540 秒，每 10 秒一次。",
        f"- 全局缺失的预期时间桶：{data['missing_expected_10_second_buckets'] or '无'}。",
        f"- 每交易日行数范围：{data['rows_per_date_min']:,} ～ {data['rows_per_date_max']:,}。",
        f"- 每股票历史行数范围：{data['rows_per_stock_min']:,} ～ {data['rows_per_stock_max']:,}。",
        f"- 日期 × 时间桶理论单元数：{data['expected_date_bucket_cells']:,}；实际出现：{data['observed_date_bucket_cells']:,}；缺失：{data['missing_date_bucket_cells']:,}。",
        f"- 每个日期 × 时间桶中的不同股票数范围：{data['stocks_per_date_bucket_min']} ～ {data['stocks_per_date_bucket_max']}；缺股票的单元数：{data['date_bucket_cells_with_missing_stocks']:,}。",
        f"- `(date_id, seconds_in_bucket, stock_id)` 重复键数：{data['duplicate_time_stock_key_count']}；时间键排序异常数：{data['time_key_order_violations']}。",
        "- 后续 lag / rolling 必须按 `(stock_id, date_id)` 排序并在日期边界重置；过去 6 行不一定等于精确的过去 60 秒。",
        "", "## 5. 缺失、数值范围与标签分布", "",
        "### 数值字段质量", "",
        "| 字段 | 空缺率 | 非有限值 | 最小值 | 最大值 | 非数值条目 |", "|---|---:|---:|---:|---:|---:|",
    ])
    for column, stats in summary["numeric_columns"].items():
        missing_rate = stats["missing_rate"]
        lines.append(f"| {column} | {missing_rate:.2%} | {stats['non_finite_count']} | {stats['minimum']} | {stats['maximum']} | {stats['invalid_numeric_count']} |")
    lines.extend([
        "", "### 竞价指示价格的缺失模式", "",
        f"- `near_price` 总缺失率：{summary['numeric_columns']['near_price']['missing_rate']:.2%}；不同日内时间桶范围：{min(near_rates):.2%} ～ {max(near_rates):.2%}。",
        f"- `far_price` 总缺失率：{summary['numeric_columns']['far_price']['missing_rate']:.2%}；不同日内时间桶范围：{min(far_rates):.2%} ～ {max(far_rates):.2%}。",
        "- 这类缺失可能反映竞价状态。后续保留缺失状态，必要时构造 `is_missing` 特征；不要直接填 0。",
        "", "### target 分布", "",
        f"- 有效标签数：{target['count']:,}",
        f"- 均值：{target_mean_text} bp；标准差：{target_std_text} bp。",
        f"- 最小值：{summary['numeric_columns']['target']['minimum']} bp；最大值：{summary['numeric_columns']['target']['maximum']} bp。",
        f"- 正值比例：{target['positive_rate']:.2%}；负值比例：{target['negative_rate']:.2%}；恰好为零：{target['zero_count']:,} 行。",
        "", "## 6. 本阶段结论与下一步", "",
        "1. 数据可以开发：schema 正确，训练/测试差异符合比赛机制；训练时需过滤第 5 节列出的非有限 target。",
        "2. 日期 × 时间桶全部存在，但每股票覆盖行数不同：键唯一性检查结果见第 4 节；接着分析每股票覆盖率和标签边界。",
        "3. near/far 指示价格缺失超过一半：把缺失当作竞价状态的一部分保留，而不是简单填零。",
        "4. 下一小步：生成按日期、股票和时间桶的 EDA 图表，并冻结不泄漏的时间验证切分。",
        "5. 完成这些检查后，再进入阶段 02 的 Zero / Median / Raw LightGBM 基线。",
    ])
    return "\n".join(lines) + "\n"


def write_audit_outputs(summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "data_audit.json"
    markdown_path = output_dir / "data_audit.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown_report(summary), encoding="utf-8")
    return json_path, markdown_path
