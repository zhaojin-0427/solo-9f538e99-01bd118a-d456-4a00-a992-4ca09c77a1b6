"""Month fact intake: one CSV, validated strictly, frozen atomically."""

from __future__ import annotations

import csv
import hashlib
import io
import re
import sqlite3
from datetime import timedelta

from .db import transaction
from .errors import ConflictError, ValidationError
from .timeutil import datetime_to_us, format_utc, month_of, now_str, parse_utc, validate_month

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_ROWS = 50_000
MAX_ERRORS = 40
CLOCK_OFFSET_LIMIT = 3600
_OFFSET_RE = re.compile(r"^[+-]?[0-9]+$")

FIELD_ALIASES = {
    "sequence_no": (
        "流水号",
        "出库流水号",
        "sequence_no",
        "sequenceno",
        "sequence",
        "seq",
        "单号",
    ),
    "sku": ("sku", "商品编码", "商品编号", "sku编码", "货号", "物料编码"),
    "occurred_at": (
        "出库时刻",
        "出库时间",
        "时间",
        "occurred_at",
        "occurredat",
        "shipped_at",
        "shipment_time",
        "timestamp",
    ),
    "boxes": ("箱数", "boxes", "boxcount", "box_count", "数量", "件数"),
}

FIELD_LABELS = {
    "sequence_no": "流水号",
    "sku": "SKU",
    "occurred_at": "出库时刻",
    "boxes": "箱数",
}

_SKU_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,64}$")
_SEQ_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,128}$")
_BOXES_RE = re.compile(r"^[0-9]{1,9}$")


def parse_offset_range(offset_min=None, offset_max=None):
    """Validate the whole-file clock-offset range (integer seconds).

    Both bounds must be given together, each an integer in
    ``[-3600, 3600]``, with ``min <= max``. Two empty inputs mean the
    range was not filled in and default to zero seconds (this also covers
    every month frozen before the feature existed).
    """
    min_text = "" if offset_min is None else str(offset_min).strip()
    max_text = "" if offset_max is None else str(offset_max).strip()
    if not min_text and not max_text:
        return 0, 0
    if not min_text or not max_text:
        raise ValidationError("钟差偏移上下限必须同时填写，或同时留空（按 0 秒处理）")
    if not (_OFFSET_RE.match(min_text) and _OFFSET_RE.match(max_text)):
        raise ValidationError("钟差偏移上下限必须是整数秒")
    low, high = int(min_text), int(max_text)
    if not (-CLOCK_OFFSET_LIMIT <= low <= CLOCK_OFFSET_LIMIT) or not (
        -CLOCK_OFFSET_LIMIT <= high <= CLOCK_OFFSET_LIMIT
    ):
        raise ValidationError(f"钟差偏移上下限必须在 -{CLOCK_OFFSET_LIMIT} 到 {CLOCK_OFFSET_LIMIT} 秒之间")
    if low > high:
        raise ValidationError("钟差偏移下限不能大于上限")
    return low, high


def _find_month_crossing_rows(rows, offset_min, offset_max):
    """Return sequence numbers pushed into another UTC month by some allowed offset.

    Only the two extreme offsets have to be tried: an instant's month can
    only change at a month boundary, and whether the boundary lies inside
    ``[t + min, t + max]`` is fully determined by the endpoints.
    """
    offenders = []
    extremes = {offset_min}
    if offset_max != offset_min:
        extremes.add(offset_max)
    for row in rows:
        original = row["occurred_at_dt"]
        original_month = month_of(original)
        for seconds in extremes:
            if month_of(original + timedelta(seconds=seconds)) != original_month:
                offenders.append(
                    (
                        row["source_row"],
                        row["sequence_no"],
                        seconds,
                        month_of(original + timedelta(seconds=seconds)),
                    )
                )
                break
    return offenders


def _norm_header(value):
    text = (value or "").strip().lstrip("﻿").lower()
    text = text.replace(" ", "").replace("_", "").replace("-", "")
    return text


def _decode(raw):
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValidationError("CSV 无法按 UTF-8 或 GB18030 解码，请另存为 UTF-8 后重试")


def _header_map(fieldnames):
    normalized = {}
    for index, name in enumerate(fieldnames):
        key = _norm_header(name)
        if key in normalized:
            raise ValidationError(f"CSV 表头重复：{name!r}")
        normalized[key] = (index, name)

    mapping = {}
    used = {}
    for field, aliases in FIELD_ALIASES.items():
        alias_set = {_norm_header(alias) for alias in aliases}
        matches = [(idx, name) for key, (idx, name) in normalized.items() if key in alias_set]
        if not matches:
            raise ValidationError(
                f"CSV 缺少必需列「{FIELD_LABELS[field]}」（可用列名：{' / '.join(aliases)}）"
            )
        if len(matches) > 1:
            names = "、".join(str(name) for _, name in matches)
            raise ValidationError(f"CSV 有多列都对应「{FIELD_LABELS[field]}」：{names}")
        idx, name = matches[0]
        if idx in used:
            raise ValidationError(f"表头列「{name}」被重复用于不同字段")
        used[idx] = field
        mapping[field] = idx

    if len(mapping) != 4:
        raise ValidationError("CSV 必须恰好包含流水号、SKU、出库时刻、箱数四列")
    return mapping


def parse_csv(raw, expected_month=None, offset_min=0, offset_max=0):
    """Parse and validate the entire file. Returns (month, rows, sha256).

    Nothing is written here; any error rejects the whole file so a month can
    never become partially frozen. The shared clock-offset range is checked
    here too: if either extreme offset would move any shipment into another
    UTC month the whole file is rejected and every offending sequence number
    is reported.
    """
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not raw:
        raise ValidationError("上传文件为空")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValidationError("CSV 超过 8 MiB 上限")
    sha256 = hashlib.sha256(raw).hexdigest()
    text = _decode(raw)

    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration:
        raise ValidationError("CSV 为空，缺少表头") from None
    if not any(str(cell).strip() for cell in header):
        raise ValidationError("CSV 首行表头为空")

    mapping = _header_map(header)
    width = len(header)
    errors = []
    rows = []
    seen_sequences = {}
    months = set()

    for line_no, cells in enumerate(reader, start=2):
        if not any(str(cell).strip() for cell in cells):
            continue
        if len(cells) > width and any(str(c).strip() for c in cells[width:]):
            errors.append(f"第 {line_no} 行：数据列多于表头")
            if len(errors) >= MAX_ERRORS:
                break
            continue
        if len(cells) < width:
            cells = cells + [""] * (width - len(cells))

        def cell(field):
            return str(cells[mapping[field]]).strip()

        row_ok = True
        sequence_no = cell("sequence_no")
        sku = cell("sku")
        occurred_text = cell("occurred_at")
        boxes_text = cell("boxes")

        if not _SEQ_RE.match(sequence_no):
            errors.append(f"第 {line_no} 行：流水号为空或含控制字符或超过 128 字符")
            row_ok = False
        if not _SKU_RE.match(sku):
            errors.append(f"第 {line_no} 行：SKU 为空或含控制字符或超过 64 字符")
            row_ok = False
        if not _BOXES_RE.match(boxes_text):
            errors.append(f"第 {line_no} 行：箱数必须为正整数")
            row_ok = False
        elif int(boxes_text) <= 0:
            errors.append(f"第 {line_no} 行：箱数必须为正整数")
            row_ok = False

        occurred = None
        if not occurred_text:
            errors.append(f"第 {line_no} 行：出库时刻为空")
            row_ok = False
        else:
            try:
                occurred = parse_utc(occurred_text, field="出库时刻")
            except ValueError as exc:
                errors.append(f"第 {line_no} 行：{exc}")
                row_ok = False

        if row_ok:
            if sequence_no in seen_sequences:
                errors.append(
                    f"第 {line_no} 行：流水号 {sequence_no} 与第 {seen_sequences[sequence_no]} 行重复"
                )
                row_ok = False
            else:
                seen_sequences[sequence_no] = line_no
                months.add(month_of(occurred))

        if not row_ok:
            if len(errors) >= MAX_ERRORS:
                break
            continue

        rows.append(
            {
                "sequence_no": sequence_no,
                "sku": sku,
                "occurred_at": format_utc(occurred),
                "occurred_at_dt": occurred,
                "boxes": int(boxes_text),
                "source_row": line_no,
            }
        )
        if len(rows) > MAX_ROWS:
            raise ValidationError(f"单次上传最多 {MAX_ROWS} 行")

    if errors:
        more = "（仅显示前若干条）" if len(errors) >= MAX_ERRORS else ""
        raise ValidationError("CSV 校验未通过，未写入任何数据", details=errors + ([more] if more else []))
    if not rows:
        raise ValidationError("CSV 没有数据行")
    if len(months) != 1:
        raise ValidationError(
            "CSV 含多个 UTC 月份的出库记录：" + "、".join(sorted(months)) + "；请单月上传"
        )
    month = months.pop()
    if expected_month:
        try:
            expected = validate_month(expected_month)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        if expected != month:
            raise ValidationError(
                f"指定月份 {expected} 与 CSV 中出库时刻所属 UTC 月份 {month} 不一致"
            )
    crossing = _find_month_crossing_rows(rows, offset_min, offset_max)
    if crossing:
        details = [
            f"第 {line_no} 行：流水号 {sequence_no} 在偏移 {seconds:+d} 秒时落到 UTC 月 {other_month}"
            for line_no, sequence_no, seconds, other_month in crossing[:MAX_ERRORS]
        ]
        if len(crossing) > MAX_ERRORS:
            details.append(f"……另有 {len(crossing) - MAX_ERRORS} 笔，仅显示前若干条")
        raise ValidationError(
            "存在允许偏移会把出库时刻移到其他 UTC 月的流水，整份拒绝入库",
            details=details,
        )
    return month, rows, sha256


def ingest_month(conn, raw, filename, expected_month=None, offset_min=None, offset_max=None):
    """Validate then freeze one month. All-or-nothing."""
    offset_min, offset_max = parse_offset_range(offset_min, offset_max)
    month, rows, sha256 = parse_csv(
        raw,
        expected_month=expected_month,
        offset_min=offset_min,
        offset_max=offset_max,
    )
    safe_name = (filename or "upload.csv").strip()[:255] or "upload.csv"
    total_boxes = sum(row["boxes"] for row in rows)

    with transaction(conn):
        exists = conn.execute(
            "SELECT month, frozen_at FROM months WHERE month = ?", (month,)
        ).fetchone()
        if exists:
            raise ConflictError(
                f"{month} 的事实已于 {exists['frozen_at']} 固定，不能重复上传或改写"
            )
        seqs = [row["sequence_no"] for row in rows]
        for start in range(0, len(seqs), 400):
            chunk = seqs[start : start + 400]
            placeholders = ",".join("?" for _ in chunk)
            clash = conn.execute(
                f"SELECT sequence_no FROM shipments WHERE sequence_no IN ({placeholders}) LIMIT 1",
                chunk,
            ).fetchone()
            if clash:
                raise ConflictError(
                    f"流水号 {clash['sequence_no']} 已存在于其他月份，流水号必须全局唯一"
                )
        frozen_at = now_str()
        try:
            conn.execute(
                "INSERT INTO months(month, source_filename, source_sha256, row_count, total_boxes,"
                " frozen_at, clock_offset_min, clock_offset_max) VALUES(?,?,?,?,?,?,?,?)",
                (
                    month,
                    safe_name,
                    sha256,
                    len(rows),
                    total_boxes,
                    frozen_at,
                    offset_min,
                    offset_max,
                ),
            )
            conn.executemany(
                "INSERT INTO shipments(month, sequence_no, sku, occurred_at, occurred_at_us, boxes, source_row)"
                " VALUES(?,?,?,?,?,?,?)",
                [
                    (
                        month,
                        row["sequence_no"],
                        row["sku"],
                        row["occurred_at"],
                        datetime_to_us(row["occurred_at_dt"]),
                        row["boxes"],
                        row["source_row"],
                    )
                    for row in rows
                ],
            )
        except sqlite3.IntegrityError as exc:
            raise ConflictError(f"写入被拒绝：{exc}（未产生任何数据）") from exc
        range_text = (
            f"钟差偏移范围 [{offset_min:+d}, {offset_max:+d}] 秒"
            if offset_min or offset_max
            else "未填钟差偏移范围（按 0 秒）"
        )
        conn.execute(
            "INSERT INTO audit_log(at, action, month, version_no, summary) VALUES(?,?,?,?,?)",
            (
                frozen_at,
                "freeze_month",
                month,
                None,
                f"固定 {month} 共 {len(rows)} 行 / {total_boxes} 箱；{range_text}",
            ),
        )
    return {
        "month": month,
        "row_count": len(rows),
        "total_boxes": total_boxes,
        "frozen_at": frozen_at,
        "sha256": sha256,
        "clock_offset_min": offset_min,
        "clock_offset_max": offset_max,
    }


def list_months(conn):
    return conn.execute(
        "SELECT m.month, m.row_count, m.total_boxes, m.frozen_at, m.source_filename,"
        " m.clock_offset_min, m.clock_offset_max"
        " FROM months m ORDER BY m.month DESC"
    ).fetchall()
