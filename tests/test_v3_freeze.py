"""G2（v3 冻结）的回归测试 —— 核心是**篡改即拒**。

两类测试刻意分开：
  1. **合成 lock 上的行为测试**：篡改文件、篡改 lock、坏 lock 结构 → 必须拒绝。
     这类不依赖仓库里的真实文件，改一个字也不会让它们变红。
  2. **真实 lock 的一致性测试**：仓库里的 `preregistration_v3.md` 与
     `preregistration_v3.lock` 必须对上。这一条**故意**会在有人改了预注册却没走
     bump version + 重算 hash 流程时变红 —— 那正是它存在的意义。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from recover_attention.evaluation import v3_freeze  # noqa: E402

REAL_PREREG = _REPO_ROOT / "docs" / "paper" / "preregistration_v3.md"
REAL_LOCK = _REPO_ROOT / "docs" / "paper" / "preregistration_v3.lock"


def _write_pair(tmp_path: Path, body: str, *, sha: str | None = None,
                version: str = v3_freeze.V3_LOCK_VERSION, file_line: str = "docs/paper/preregistration_v3.md",
                hash_mode: str = "text-lf-utf8", extra: str = "",
                raw: bytes | None = None) -> tuple[Path, Path]:
    prereg = tmp_path / "preregistration_v3.md"
    if raw is not None:
        prereg.write_bytes(raw)
    else:
        prereg.write_bytes(body.encode("utf-8"))       # 不经 newline 翻译，行尾由调用方决定
    digest = sha if sha is not None else v3_freeze.sha256_text_lf(prereg)
    lock = tmp_path / "preregistration_v3.lock"
    lock.write_text(
        "# lock\n\n"
        f"file:      {file_line}\n"
        f"version:   {version}\n"
        f"hash_mode: {hash_mode}\n"
        f"sha256:    {digest}\n"
        "frozen_on: 2026-08-08\n" + extra,
        encoding="utf-8",
    )
    return prereg, lock


# ── 1. 合成 lock 上的行为 ────────────────────────────────────────────────────
def test_matching_pair_passes(tmp_path):
    prereg, lock = _write_pair(tmp_path, "# design\nR = 3\n")
    status = v3_freeze.check_v3_frozen(prereg, lock)
    assert status.match is True
    assert status.locked_version == v3_freeze.V3_LOCK_VERSION
    assert v3_freeze.require_v3_frozen(prereg, lock).match is True


def test_tampering_with_the_design_is_rejected(tmp_path):
    """冻结后改设计 —— G2 必须拒。这是本文件存在的唯一理由。"""
    prereg, lock = _write_pair(tmp_path, "# design\nR = 3\n")
    prereg.write_text("# design\nR = 5\n", encoding="utf-8")   # 偷改一个超参
    assert v3_freeze.check_v3_frozen(prereg, lock).match is False
    with pytest.raises(ValueError, match="G2 失败"):
        v3_freeze.require_v3_frozen(prereg, lock)


@pytest.mark.parametrize("tamper", [
    "# design\nR = 3\n\n",      # 多一个空行
    "# design \nR = 3\n",       # 行尾多一个空格
    "# design\nR = 3",          # 少了最末换行
    "# design\n\nR = 3\n",      # 正文中间插一个换行
])
def test_even_whitespace_only_edits_are_rejected(tmp_path, tamper):
    """行尾规范化**只管行尾**：空格、空行增删、最末换行的有无仍然算改动。"""
    prereg, lock = _write_pair(tmp_path, "# design\nR = 3\n")
    prereg.write_bytes(tamper.encode("utf-8"))
    assert v3_freeze.check_v3_frozen(prereg, lock).match is False


# ── hash_mode：跨平台可移植性 ────────────────────────────────────────────────
@pytest.mark.parametrize("first,second", [
    ("# design\r\nR = 3\r\n", "# design\nR = 3\n"),      # CRLF → LF
    ("# design\nR = 3\n", "# design\r\nR = 3\r\n"),      # LF → CRLF
    ("# design\rR = 3\r", "# design\nR = 3\n"),          # 裸 CR → LF
])
def test_line_ending_conversion_still_matches(tmp_path, first, second):
    """Linux checkout 把 CRLF 变成 LF —— 内容未变，G2 不得报警。"""
    prereg, lock = _write_pair(tmp_path, first)
    assert v3_freeze.check_v3_frozen(prereg, lock).match is True
    prereg.write_bytes(second.encode("utf-8"))
    assert v3_freeze.check_v3_frozen(prereg, lock).match is True, (
        "行尾转换必须仍然匹配 —— 否则跨平台修复没有生效")


def test_line_ending_conversion_changes_the_raw_digest(tmp_path):
    """反证：若仍用原始字节口径，同一内容会被判成改动。这就是本次修复的理由。"""
    prereg, _ = _write_pair(tmp_path, "# design\r\nR = 3\r\n")
    raw_crlf = v3_freeze.sha256_of(prereg)
    prereg.write_bytes(b"# design\nR = 3\n")
    assert v3_freeze.sha256_of(prereg) != raw_crlf
    assert v3_freeze.sha256_text_lf(prereg) == v3_freeze.sha256_text_lf(prereg)


def test_bom_raises_a_named_error_and_is_not_stripped(tmp_path):
    """BOM 不静默剥离：它意味着文件被别的编辑器重存过，需要人工确认。"""
    prereg, lock = _write_pair(tmp_path, "# design\n")
    prereg.write_bytes(b"\xef\xbb\xbf# design\n")
    with pytest.raises(ValueError, match="BOM"):
        v3_freeze.check_v3_frozen(prereg, lock)


def test_invalid_utf8_is_rejected_by_strict_decode(tmp_path):
    prereg, lock = _write_pair(tmp_path, "# design\n")
    prereg.write_bytes(b"# design \xff\xfe\n")
    with pytest.raises(ValueError, match="UTF-8"):
        v3_freeze.check_v3_frozen(prereg, lock)


def test_missing_hash_mode_is_rejected(tmp_path):
    prereg = tmp_path / "preregistration_v3.md"
    prereg.write_bytes(b"# design\n")
    lock = tmp_path / "l.lock"
    lock.write_text(
        "file:      docs/paper/preregistration_v3.md\nversion:   v3.0\n"
        f"sha256:    {v3_freeze.sha256_text_lf(prereg)}\nfrozen_on: 2026-08-08\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="缺少 lock 键"):
        v3_freeze.check_v3_frozen(prereg, lock)


def test_unknown_hash_mode_does_not_fall_back_to_raw_bytes(tmp_path):
    """静默回退到原始字节口径会让修复形同虚设，故必须报错。"""
    prereg, lock = _write_pair(tmp_path, "# design\n", hash_mode="raw-bytes")
    with pytest.raises(ValueError, match="hash_mode"):
        v3_freeze.check_v3_frozen(prereg, lock)


def test_tampering_with_the_lock_is_rejected(tmp_path):
    """改 lock 去迁就被改过的文件，同样拒 —— 除非同时 bump version（下一条）。"""
    prereg, lock = _write_pair(tmp_path, "# design\n", sha="0" * 64)
    assert v3_freeze.check_v3_frozen(prereg, lock).match is False


def test_bumping_version_without_updating_expectation_is_rejected(tmp_path):
    """lock 的 version 与代码期望的不一致即拒 —— 用一个绝不会成为真实版本的值。"""
    prereg, lock = _write_pair(tmp_path, "# design\n", version="v9.9")
    with pytest.raises(ValueError, match="bump version"):
        v3_freeze.check_v3_frozen(prereg, lock)
    # 显式声明新版本才放行
    assert v3_freeze.check_v3_frozen(prereg, lock, expected_version="v9.9").match is True


def test_v3_0_lock_no_longer_passes_after_the_v3_1_bump(tmp_path):
    """v3.1 之后，一份还写着 v3.0 的 lock 不得再刷绿 —— 否则 bump 形同虚设。"""
    prereg, lock = _write_pair(tmp_path, "# design\n", version="v3.0")
    with pytest.raises(ValueError, match="bump version"):
        v3_freeze.check_v3_frozen(prereg, lock)


def test_lock_pointing_at_another_file_is_rejected(tmp_path):
    """只比 hash 的话，拿别的文件配别的 lock 也能刷绿。"""
    prereg, lock = _write_pair(tmp_path, "# design\n", file_line="docs/paper/preregistration.md")
    with pytest.raises(ValueError, match="不是 preregistration_v3.md"):
        v3_freeze.check_v3_frozen(prereg, lock)


def test_duplicate_lock_key_is_rejected_not_silently_first_wins(tmp_path):
    """v2.4 的 lock 里就有第二个 sha256（历史版本段）；取第一个是脆弱语义。"""
    prereg, lock = _write_pair(tmp_path, "# design\n", extra="sha256:    " + "1" * 64 + "\n")
    with pytest.raises(ValueError, match="重复的 lock 键"):
        v3_freeze.check_v3_frozen(prereg, lock)


def test_missing_lock_key_is_rejected(tmp_path):
    prereg = tmp_path / "preregistration_v3.md"
    prereg.write_text("# design\n", encoding="utf-8")
    lock = tmp_path / "l.lock"
    lock.write_text("file:      docs/paper/preregistration_v3.md\nversion:   v3.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="缺少 lock 键"):
        v3_freeze.check_v3_frozen(prereg, lock)


def test_malformed_sha256_is_rejected(tmp_path):
    prereg, lock = _write_pair(tmp_path, "# design\n", sha="NOTAHASH")
    with pytest.raises(ValueError, match="不是 64 位小写 hex"):
        v3_freeze.check_v3_frozen(prereg, lock)


def test_indented_key_is_not_parsed_as_a_lock_field(tmp_path):
    """散文里缩进提到 `sha256: ...` 不得被当成 lock 键（否则正文一提就撞重复键）。"""
    prereg, lock = _write_pair(tmp_path, "# design\n", extra="    digest 见 history\n")
    assert v3_freeze.check_v3_frozen(prereg, lock).match is True


# ── 2. 仓库里的真实 lock ─────────────────────────────────────────────────────
def test_repository_preregistration_v3_matches_its_lock():
    status = v3_freeze.check_v3_frozen(REAL_PREREG, REAL_LOCK)
    assert status.match is True, (
        "preregistration_v3.md 与其 lock 不一致。冻结后不得改设计；"
        "若确需修改，先 bump version、重算 sha256、更新 lock，并在文件顶部记录变更。"
    )
    assert status.locked_version == v3_freeze.V3_LOCK_VERSION
    assert status.hash_mode == v3_freeze.HASH_MODE_TEXT_LF
    assert status.frozen_on == "2026-08-08"


def test_repository_lock_survives_a_line_ending_flip(tmp_path):
    """把仓库里的 prereg 转成 LF（模拟 Linux checkout），G2 仍须通过。"""
    import shutil
    copy = tmp_path / "preregistration_v3.md"
    copy.write_bytes(REAL_PREREG.read_bytes().replace(b"\r\n", b"\n"))
    lock_copy = tmp_path / "preregistration_v3.lock"
    shutil.copyfile(REAL_LOCK, lock_copy)
    assert v3_freeze.check_v3_frozen(copy, lock_copy).match is True


def test_enforcement_status_is_reported_honestly():
    """§12.1：lock 已建立，正式入口尚未接线 —— 产物里必须这么写，不得夸大。"""
    status = v3_freeze.check_v3_frozen(REAL_PREREG, REAL_LOCK)
    assert "not wired yet" in status.as_dict()["enforcement"]


def test_stage0_lock_is_untouched_and_cannot_be_used_as_the_v3_gate():
    """两份设计权威并存：v3 的冻结不得动 v2.4 的 lock，也不得拿它当 v3 的 gate。"""
    stage0_lock = _REPO_ROOT / "docs" / "paper" / "preregistration.lock"
    text = stage0_lock.read_text(encoding="utf-8")
    assert "preregistration.md" in text and "v2.4" in text

    # v2.4 的 lock 没有 hash_mode 键（它是原始字节口径），故 v3 解析器直接拒绝它。
    # 这比先前靠 file: 校验更早拦下 —— 两份 lock 的格式契约不同，本就不该互用。
    with pytest.raises(ValueError, match=r"缺少 lock 键.*hash_mode"):
        v3_freeze.parse_v3_lock(stage0_lock)
    with pytest.raises(ValueError):
        v3_freeze.check_v3_frozen(REAL_PREREG, stage0_lock)
