"""High-precision execution timer for measuring latency in computer vision pipelines."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional


@dataclass
class TimingReport:
    measurements: Dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0

    def add(self, stage: str, duration_ms: float) -> None:
        self.measurements[stage] = duration_ms
        self.total_ms = sum(self.measurements.values())

    def get(self, stage: str, default: float = 0.0) -> float:
        return self.measurements.get(stage, default)


class ExecutionTimer:
    """Multi-stage timing tracker using time.perf_counter()."""

    def __init__(self) -> None:
        self.report = TimingReport()

    @contextmanager
    def track(self, stage_name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.report.add(stage_name, elapsed_ms)

    def summary(self) -> TimingReport:
        return self.report
