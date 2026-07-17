"""可续跑的工作清单（generated_manifest.jsonl）。

设计要点:**崩溃安全优先于优雅**。
  - 状态只追加(append-only),不原地改写 —— 改写时断电会丢整个文件。
  - 每行一个状态事件;同一 unit 的最后一条事件即当前状态。
  - 读取时重放全部事件 → 天然幂等,重复运行不会重算已 done 的 unit。
  - running 状态在恢复时视为**未完成**(可能是上次跑到一半被杀),重新入队。

status: pending -> running -> done | failed
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
VALID_STATUS = (STATUS_PENDING, STATUS_RUNNING, STATUS_DONE, STATUS_FAILED)


@dataclass
class WorkManifest:
    """append-only 的 unit 状态日志。

    path 是事件日志;current_status() 通过重放得到当前态。
    """

    path: Path
    _events: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.exists():
            self._events = self._read_events()

    def _read_events(self) -> list[dict]:
        out: list[dict] = []
        with open(self.path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    # 断电可能留下半行。截断点之后的内容不可信,但之前的有效 —— 丢弃尾部即可,
                    # 这正是 append-only 的好处:损坏永远只在末尾。
                    break
                if "unit_id" in ev and ev.get("status") in VALID_STATUS:
                    out.append(ev)
        return out

    # ------------------------------------------------------------------ 写
    def _append(self, ev: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())     # 断电也要落盘:没有 fsync,resume 就是骗人的
        self._events.append(ev)

    def mark(self, unit_id: str, status: str, **extra: Any) -> None:
        if status not in VALID_STATUS:
            raise ValueError(f"invalid status {status!r}; expected {VALID_STATUS}")
        self._append({"unit_id": unit_id, "status": status, **extra})

    def register(self, unit_ids: Iterable[str]) -> int:
        """把未见过的 unit 登记为 pending。已见过的不重复登记（幂等）。"""
        known = self.current_status()
        n = 0
        for uid in unit_ids:
            if uid not in known:
                self.mark(uid, STATUS_PENDING)
                n += 1
        return n

    # ------------------------------------------------------------------ 读
    def current_status(self) -> dict[str, str]:
        """重放事件 → 每个 unit 的当前状态。"""
        out: dict[str, str] = {}
        for ev in self._events:
            out[ev["unit_id"]] = ev["status"]
        return out

    def last_event(self, unit_id: str) -> dict | None:
        for ev in reversed(self._events):
            if ev["unit_id"] == unit_id:
                return ev
        return None

    def pending_units(self, all_units: Iterable[str]) -> list[str]:
        """待跑 = 未 done 的。

        **running 视为未完成**:上次跑到一半被杀会留下 running,不重跑就会丢数据。
        failed 也重新入队(可能是瞬时错误);持续失败由调用方按 attempts 决定是否放弃。
        """
        st = self.current_status()
        return [u for u in all_units if st.get(u) != STATUS_DONE]

    def counts(self) -> dict[str, int]:
        st = self.current_status()
        out = {s: 0 for s in VALID_STATUS}
        for v in st.values():
            out[v] = out.get(v, 0) + 1
        return out

    def attempts(self, unit_id: str) -> int:
        return sum(1 for ev in self._events
                   if ev["unit_id"] == unit_id and ev["status"] == STATUS_RUNNING)

    def is_complete(self, all_units: Iterable[str]) -> bool:
        st = self.current_status()
        return all(st.get(u) == STATUS_DONE for u in all_units)

    def summary(self, all_units: Iterable[str]) -> dict:
        units = list(all_units)
        st = self.current_status()
        done = [u for u in units if st.get(u) == STATUS_DONE]
        failed = [u for u in units if st.get(u) == STATUS_FAILED]
        return {
            "n_units": len(units), "n_done": len(done), "n_failed": len(failed),
            "n_remaining": len(units) - len(done),
            "complete": len(done) == len(units),
            "failed_units": sorted(failed)[:20],
            "n_events": len(self._events),
        }


def write_atomic(path: Path, payload: str) -> None:
    """整文件写:先写临时文件再 rename（同目录内 rename 是原子的）。

    用于最终产物(records / npz 索引),避免读者看到写了一半的文件。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
