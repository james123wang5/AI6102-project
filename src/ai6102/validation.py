"""Chronological split definitions used throughout the project."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DateFold:
    name: str
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int

    def __post_init__(self) -> None:
        if not (self.train_start <= self.train_end < self.validation_start <= self.validation_end):
            raise ValueError(f"Fold {self.name} is not chronologically ordered: {self}")

    def train_mask(self, dates: pd.Series) -> pd.Series:
        return dates.between(self.train_start, self.train_end)

    def validation_mask(self, dates: pd.Series) -> pd.Series:
        return dates.between(self.validation_start, self.validation_end)

    def inner_validation_start(self, days: int) -> int:
        start = self.train_end - days + 1
        if start <= self.train_start:
            raise ValueError(f"Fold {self.name} has insufficient training days for {days} early-stopping days.")
        return start
