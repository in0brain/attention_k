r"""哨兵校验：串行队列**不能只判文件存在**。

起因（2026-09-19）：followup 队列的第 0 步只做 `if exist "%SENTINEL%"`。
这次哨兵确实是本次运行产出、verdict 也确实是 PASS，所以没有误放行；
但这纯属运气 —— 换成下面任一情形都会错误续跑：

    · 上一轮遗留的旧文件还在（根本不是本次运行的产物）
    · 文件正在写、只落了一半（JSON 解析不了）
    · 文件完整，但 verdict 是 FAIL（前置闸没过，后面不该开工）

所以这里逐项查：**能解析 / schema 对得上 / verdict 符合要求 / 比基准文件新**。
退出码 0 才算放行；任何一项不过都返回非 0，并在 stderr 说清是哪一项。

用法（在 .cmd 里替代 `if exist`）：
    python scripts\check_sentinel.py <哨兵路径> --require-verdict PASS ^
        --newer-than <基准文件> --schema posthoc_reference_reproduction_threadtest_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sentinel", type=Path)
    ap.add_argument("--require-verdict", default=None,
                    help="要求 JSON 里的 verdict 等于这个值（大小写敏感）")
    ap.add_argument("--schema", default=None, help="要求 schema 字段等于这个值")
    ap.add_argument("--newer-than", type=Path, default=None,
                    help="要求哨兵的 mtime **严格晚于**这个基准文件 —— "
                         "用来排除上一轮遗留的旧文件")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    def fail(why: str) -> int:
        print(f"[sentinel] 不放行：{why}", file=sys.stderr)
        return 1

    if not a.sentinel.is_file():
        return fail(f"文件不存在 {a.sentinel}")

    if a.newer_than is not None:
        if not a.newer_than.exists():
            return fail(f"基准文件不存在 {a.newer_than} —— 无法判断哨兵是否属于本次运行")
        st, bt = a.sentinel.stat().st_mtime, a.newer_than.stat().st_mtime
        if st <= bt:
            return fail(f"哨兵不比基准新（哨兵 {st:.0f} <= 基准 {bt:.0f}）"
                        f" —— 很可能是上一轮遗留的旧文件")

    try:
        text = a.sentinel.read_text(encoding="utf-8")
    except OSError as exc:
        return fail(f"读不了：{type(exc).__name__}: {exc}")
    try:
        d = json.loads(text)
    except json.JSONDecodeError as exc:
        # 正在写、只落了一半就是这个症状
        return fail(f"JSON 解析失败（可能还在写）：{exc}")
    if not isinstance(d, dict):
        return fail(f"顶层不是对象，而是 {type(d).__name__}")

    if a.schema is not None and d.get("schema") != a.schema:
        return fail(f"schema 不符：期望 {a.schema!r}，实际 {d.get('schema')!r}")
    if a.require_verdict is not None:
        got = d.get("verdict")
        if got != a.require_verdict:
            return fail(f"verdict 不符：期望 {a.require_verdict!r}，实际 {got!r}"
                        f" —— 前置闸没过，后续不开工")

    if not a.quiet:
        print(f"[sentinel] 放行 {a.sentinel.name}"
              f"  schema={d.get('schema')}  verdict={d.get('verdict')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
