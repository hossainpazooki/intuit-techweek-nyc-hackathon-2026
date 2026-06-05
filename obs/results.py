"""Shared result types for observability checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class Severity(str, Enum):
    ERROR = "error"  # blocks: a broken invariant
    WARN = "warn"    # informs: a trend worth a human look
    INFO = "info"    # context only


@dataclass
class CheckResult:
    section: str           # integrity | quality | selection | intervention
    name: str              # short check id
    passed: bool
    severity: Severity     # severity *if it fails*
    message: str
    n_offending: int = 0   # count of offending rows / cells / features
    details: dict = field(default_factory=dict)  # tables / per-feature payloads

    @property
    def blocking(self) -> bool:
        return (not self.passed) and self.severity is Severity.ERROR

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class CheckReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, r: CheckResult) -> CheckResult:
        self.results.append(r)
        return r

    def extend(self, rs: list[CheckResult]) -> None:
        self.results.extend(rs)

    @property
    def n_error(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity is Severity.ERROR)

    @property
    def n_warn(self) -> int:
        return sum(1 for r in self.results if not r.passed and r.severity is Severity.WARN)

    @property
    def passed(self) -> bool:
        """True iff no blocking (error) check failed."""
        return not any(r.blocking for r in self.results)

    def section(self, name: str) -> list[CheckResult]:
        return [r for r in self.results if r.section == name]
