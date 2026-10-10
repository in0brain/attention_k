"""启动 gate G2（v3）：preregistration_v3 的冻结校验。

**强制程度的如实陈述（preregistration_v3 §12.1）**：
    本模块提供**可调用的校验函数**与 lock 文件解析。
    Stage 1 的 preflight / sample-builder / generation 入口**尚未实现**，
    因此目前**没有任何生产路径在调用 G2**。
    ⇒ 现状 = 文件级冻结 + 可调用的校验函数，**不是**端到端强制。
    ❌ 不得声称"v3 的冻结已由代码端到端强制"。
    下一 sprint 必须先完成 Stage 1 入口的 G2/G5 接线，之后才能物化正式名单。

与 Stage 0 的关系：`conditional_increment.check_preregistration_frozen` 是 Stage 0 的
G2 实现，本次**一行未改**。本模块是 v3 侧的独立实现，理由有二：
  1. Stage 0 的版本只做 `split("sha256:")[1]`，对 lock 里出现第二个 `sha256:`
     （v2.4 的 lock 就有"历史版本"段）依赖首次出现的位置，语义脆弱；
     v3 的 lock 解析要求**行首键值**并拒绝重复键。
  2. v3 的 lock 还要校验 `file:` 与 `version:` 指向正确，避免拿错文件刷绿 gate。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

V3_LOCK_VERSION = "v3.1"
V3_PREREG_RELPATH = "docs/paper/preregistration_v3.md"

#: 唯一支持的哈希口径。**不设 raw 回退** —— 回退会让可移植性修复形同虚设。
HASH_MODE_TEXT_LF = "text-lf-utf8"
SUPPORTED_HASH_MODES = (HASH_MODE_TEXT_LF,)

_LOCK_KEYS = ("file", "version", "hash_mode", "sha256", "frozen_on")
_KEY_RE = re.compile(r"^(file|version|hash_mode|sha256|frozen_on):\s*(\S+)\s*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UTF8_BOM = b"\xef\xbb\xbf"


@dataclass(frozen=True)
class V3FreezeStatus:
    """G2 的结果。`match` 为假 → 调用方必须停机，不得继续任何阶段。"""

    current_sha256: str
    locked_sha256: str
    match: bool
    locked_file: str
    locked_version: str
    hash_mode: str
    frozen_on: str

    def as_dict(self) -> dict:
        return {
            "gate": "G2",
            "prereg": "preregistration_v3",
            "current_sha256": self.current_sha256,
            "locked_sha256": self.locked_sha256,
            "match": self.match,
            "locked_file": self.locked_file,
            "locked_version": self.locked_version,
            "hash_mode": self.hash_mode,
            "frozen_on": self.frozen_on,
            "enforcement": "file-level only; Stage 1 entrypoints not wired yet (§12.1)",
        }


def sha256_of(path: str | Path) -> str:
    """**原始字节**的 sha256。保留作对照与留痕，**不再**用于 G2。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text_bytes(raw: bytes, *, source: str = "<bytes>") -> bytes:
    """G2 的文本规范化：**只统一行尾，别的一概不动**。

    ```
    UTF-8 strict decode → CRLF 与裸 CR 统一为 LF → UTF-8 encode
    ```

    ⚠️ 尾随空格、缩进、空行增删、最末换行的有无**仍然改变哈希** —— 这是有意的：
    规范化只解决"同一内容在不同平台 checkout 出不同字节"这一个问题，
    多规范化一样东西，就多一种能掩盖真实改动的途径。

    BOM **不静默剥离**：检测到即抛具名错误。BOM 出现意味着文件被另一个编辑器重存过，
    那正是需要人工确认的时刻，而不是自动抹平的时刻。
    """
    if raw.startswith(_UTF8_BOM):
        raise ValueError(
            f"{source} 以 UTF-8 BOM 开头。G2 不静默剥离 BOM —— "
            "BOM 出现说明文件被另一个编辑器重存过，请人工确认内容未变后去掉 BOM。")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source} 不是合法的 UTF-8（strict）: {exc}") from exc
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def sha256_text_lf(path: str | Path) -> str:
    """`hash_mode = text-lf-utf8` 下的指纹：行尾规范化后再取 sha256。"""
    source = str(path)
    return hashlib.sha256(
        normalize_text_bytes(Path(path).read_bytes(), source=source)
    ).hexdigest()


_HASH_FUNCS = {HASH_MODE_TEXT_LF: sha256_text_lf}


def parse_v3_lock(lock_path: str | Path) -> dict[str, str]:
    """解析 lock 的**行首键值**。重复键、缺键、非法 sha256 一律抛错，不静默取第一个。"""
    fields: dict[str, str] = {}
    for line_no, line in enumerate(Path(lock_path).read_text(encoding="utf-8").splitlines(), 1):
        match = _KEY_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        if key in fields:
            raise ValueError(
                f"{lock_path}:{line_no} 重复的 lock 键 {key!r} —— "
                f"歧义的 lock 不得用于 gate（已有值 {fields[key]!r}）")
        fields[key] = value

    missing = [k for k in _LOCK_KEYS if k not in fields]
    if missing:
        raise ValueError(f"{lock_path} 缺少 lock 键: {missing}")
    if not _SHA256_RE.match(fields["sha256"]):
        raise ValueError(f"{lock_path} 的 sha256 不是 64 位小写 hex: {fields['sha256']!r}")
    if fields["hash_mode"] not in SUPPORTED_HASH_MODES:
        raise ValueError(
            f"{lock_path} 的 hash_mode={fields['hash_mode']!r} 不受支持"
            f"（支持 {list(SUPPORTED_HASH_MODES)}）。**不回退到原始字节口径** —— "
            "静默回退会让跨平台修复形同虚设。")
    return fields


def check_v3_frozen(
    prereg_path: str | Path = V3_PREREG_RELPATH,
    lock_path: str | Path = "docs/paper/preregistration_v3.lock",
    *,
    expected_version: str = V3_LOCK_VERSION,
) -> V3FreezeStatus:
    """重算 preregistration_v3 的 sha256 并与 lock 比对。

    额外校验 `file:` 与 `version:` —— 只比 hash 的话，拿一份别的文件配一份别的 lock
    也能刷绿。三者都对才算 G2 通过。
    """
    fields = parse_v3_lock(lock_path)
    if not str(fields["file"]).endswith("preregistration_v3.md"):
        raise ValueError(f"lock 的 file 指向 {fields['file']!r}，不是 preregistration_v3.md")
    if fields["version"] != expected_version:
        raise ValueError(
            f"lock 的 version={fields['version']!r} != 期望的 {expected_version!r}；"
            f"改设计须 bump version 并重算 hash")

    current = _HASH_FUNCS[fields["hash_mode"]](prereg_path)
    return V3FreezeStatus(
        current_sha256=current,
        locked_sha256=fields["sha256"],
        match=current == fields["sha256"],
        locked_file=fields["file"],
        locked_version=fields["version"],
        hash_mode=fields["hash_mode"],
        frozen_on=fields["frozen_on"],
    )


def require_v3_frozen(
    prereg_path: str | Path = V3_PREREG_RELPATH,
    lock_path: str | Path = "docs/paper/preregistration_v3.lock",
) -> V3FreezeStatus:
    """G2 的**停机**语义：不一致即抛错，调用方不得继续（对应 Stage 0 的 SystemExit）。"""
    status = check_v3_frozen(prereg_path, lock_path)
    if not status.match:
        raise ValueError(
            "G2 失败：preregistration_v3 的内容与 lock 不一致 —— "
            f"当前 {status.current_sha256}，lock {status.locked_sha256}。"
            "冻结后不得改设计；若确需修改，先 bump version、重算 hash、更新 lock。")
    return status
