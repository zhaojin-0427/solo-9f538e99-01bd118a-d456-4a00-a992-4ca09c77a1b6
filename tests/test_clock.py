"""Clock-offset range intake and sensitivity-analysis tests."""

from __future__ import annotations

import io
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime, timedelta, timezone

from coldchain import clock, facts, knowledge
from coldchain.app import create_app
from coldchain.db import connect, init_db
from coldchain.errors import ValidationError
from coldchain.timeutil import datetime_to_us

UTC = timezone.utc
US = 1_000_000


def csv_bytes(rows, header="流水号,SKU,出库时刻,箱数"):
    lines = [header]
    lines.extend(",".join(str(cell) for cell in row) for row in rows)
    return ("\n".join(lines) + "\n").encode("utf-8")


def sample_rows():
    return [
        ("OUT-001", "SKU-A", "2026-03-01T01:00:00Z", 12),
        ("OUT-002", "SKU-B", "2026-03-05T23:30:00Z", 7),
        ("OUT-003", "SKU-A", "2026-03-10T00:00:00Z", 5),
        ("OUT-004", "SKU-C", "2026-03-31T23:59:59Z", 2),
    ]


class OffsetRangeValidationTests(unittest.TestCase):
    def test_blank_range_defaults_to_zero(self):
        self.assertEqual(facts.parse_offset_range(None, None), (0, 0))
        self.assertEqual(facts.parse_offset_range("", "  "), (0, 0))

    def test_both_bounds_required(self):
        with self.assertRaises(ValidationError):
            facts.parse_offset_range("-5", "")
        with self.assertRaises(ValidationError):
            facts.parse_offset_range("", "5")

    def test_must_be_integers(self):
        for bad in ("1.5", "abc", "1s", "--1"):
            with self.assertRaises(ValidationError, msg=bad):
                facts.parse_offset_range(bad, "5")

    def test_signs_and_whitespace_accepted(self):
        self.assertEqual(facts.parse_offset_range("  -5 ", " +10 "), (-5, 10))
        self.assertEqual(facts.parse_offset_range("-0", "+0"), (0, 0))
        self.assertEqual(facts.parse_offset_range("0", "0"), (0, 0))

    def test_bounds_must_be_within_limit(self):
        with self.assertRaises(ValidationError):
            facts.parse_offset_range("-3601", "0")
        with self.assertRaises(ValidationError):
            facts.parse_offset_range("0", "3601")
        # Boundary values are accepted.
        self.assertEqual(facts.parse_offset_range("-3600", "3600"), (-3600, 3600))

    def test_min_cannot_exceed_max(self):
        with self.assertRaises(ValidationError):
            facts.parse_offset_range("10", "9")


class MonthCrossingRejectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "test.sqlite3")
        self.app = create_app({"TESTING": True, "DB_PATH": self.db_path})
        self.client = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def upload(self, rows, offset_min=None, offset_max=None, month=None):
        data = {"file": (io.BytesIO(csv_bytes(rows)), "m.csv")}
        if offset_min is not None:
            data["clock_offset_min"] = str(offset_min)
        if offset_max is not None:
            data["clock_offset_max"] = str(offset_max)
        if month:
            data["month"] = month
        return self.client.post(
            "/api/months", data=data, content_type="multipart/form-data"
        )

    def test_negative_offset_crossing_into_previous_month_rejected(self):
        # 03-01 00:20 UTC minus 1800 s lands on 02-28 23:50.
        rows = [("OUT-1", "SKU-A", "2026-03-01T00:20:00Z", 4)]
        response = self.upload(rows, offset_min=-1800, offset_max=1800)
        self.assertEqual(response.status_code, 400)
        body = response.get_json()
        self.assertIn("OUT-1", str(body["details"]))
        # Nothing was written.
        self.assertEqual(self.client.get("/api/months").get_json()["months"], [])

    def test_positive_offset_crossing_into_next_month_rejected(self):
        # 03-31 23:40 plus 1800 s lands on 04-01 00:10.
        rows = [("OUT-9", "SKU-A", "2026-03-31T23:40:00Z", 4)]
        response = self.upload(rows, offset_min=-1800, offset_max=1800)
        self.assertEqual(response.status_code, 400)
        self.assertIn("OUT-9", str(response.get_json()["details"]))

    def test_safe_range_is_accepted_and_persisted(self):
        # One hour away from the boundary: +/- 3600 s stays inside March.
        rows = [("OUT-1", "SKU-A", "2026-03-01T01:00:00Z", 4)]
        response = self.upload(rows, offset_min=-3600, offset_max=3600)
        self.assertEqual(response.status_code, 201)
        body = response.get_json()
        self.assertEqual(body["clock_offset_min"], -3600)
        self.assertEqual(body["clock_offset_max"], 3600)
        months = self.client.get("/api/months").get_json()["months"]
        self.assertEqual(months[0]["clock_offset_min"], -3600)
        self.assertEqual(months[0]["clock_offset_max"], 3600)

    def test_zero_range_on_boundary_shipment_accepted(self):
        # No range filled in: defaults to zero, so a boundary shipment is fine.
        rows = [("OUT-1", "SKU-A", "2026-03-01T00:00:00Z", 4)]
        response = self.upload(rows)
        self.assertEqual(response.status_code, 201)
        body = response.get_json()
        self.assertEqual(body["clock_offset_min"], 0)
        self.assertEqual(body["clock_offset_max"], 0)

    def test_range_that_does_not_reach_boundary_accepted(self):
        # 03-01 00:30 with offsets only in [-900, 900]: stays inside March.
        rows = [("OUT-1", "SKU-A", "2026-03-01T00:30:00Z", 4)]
        response = self.upload(rows, offset_min=-900, offset_max=900)
        self.assertEqual(response.status_code, 201)

    def test_multiple_offenders_are_all_listed(self):
        rows = [
            ("OUT-A", "SKU-A", "2026-03-01T00:00:30Z", 1),
            ("OUT-B", "SKU-A", "2026-03-31T23:59:30Z", 1),
            ("OUT-C", "SKU-A", "2026-03-15T12:00:00Z", 1),
        ]
        response = self.upload(rows, offset_min=-3600, offset_max=3600)
        self.assertEqual(response.status_code, 400)
        details = str(response.get_json()["details"])
        self.assertIn("OUT-A", details)
        self.assertIn("OUT-B", details)
        self.assertNotIn("OUT-C", details)

    def test_rejected_file_leaves_clean_database(self):
        rows = [("OUT-1", "SKU-A", "2026-03-01T00:10:00Z", 4)]
        self.assertEqual(self.upload(rows, -3600, 3600).status_code, 400)
        conn = connect(self.db_path)
        try:
            self.assertEqual(conn.execute("SELECT COUNT(*) AS c FROM months").fetchone()["c"], 0)
            self.assertEqual(
                conn.execute("SELECT COUNT(*) AS c FROM shipments").fetchone()["c"], 0
            )
        finally:
            conn.close()
        # The same file with a safe range then succeeds.
        self.assertEqual(self.upload(rows, 0, 0).status_code, 201)


class ClockAnalysisEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "test.sqlite3")
        self.app = create_app({"TESTING": True, "DB_PATH": self.db_path})
        self.client = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    # Timestamps stay more than one hour away from any UTC month boundary so
    # the full [-3600, 3600] range cannot push a row into another month.
    def _rows(self):
        return [
            # OUT-001 sits deep inside 冷冻 (which ends 03-08).
            ("OUT-001", "SKU-A", "2026-03-01T01:00:00Z", 12),
            # OUT-003 is 30 minutes into 冷藏: -3600 s pushes it back into 冷冻.
            ("OUT-003", "SKU-A", "2026-03-08T00:30:00Z", 5),
            # SKUs without knowledge stay unclassified for every offset.
            ("OUT-002", "SKU-B", "2026-03-05T23:30:00Z", 7),
            ("OUT-004", "SKU-C", "2026-03-30T12:00:00Z", 2),
        ]

    def _upload_and_version(self, offset_min, offset_max):
        data = {
            "file": (io.BytesIO(csv_bytes(self._rows())), "m.csv"),
            "clock_offset_min": str(offset_min),
            "clock_offset_max": str(offset_max),
        }
        response = self.client.post(
            "/api/months", data=data, content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        # SKU-A: 冷冻 until 03-08, 冷藏 afterwards.
        response = self.client.post(
            "/api/versions",
            json={
                "sku": "SKU-A",
                "reason": "v1",
                "base_version": None,
                "intervals": [
                    {"zone": "冷冻", "valid_from": "2026-03-01T00:00:00Z",
                     "valid_to": "2026-03-08T00:00:00Z"},
                    {"zone": "冷藏", "valid_from": "2026-03-08T00:00:00Z", "valid_to": ""},
                ],
            },
        )
        self.assertEqual(response.status_code, 201)

    def test_report_contains_clock_analysis_with_extreme_offsets(self):
        self._upload_and_version(-3600, 3600)
        body = self.client.get("/api/report?month=2026-03").get_json()
        analysis = body["clock_analysis"]
        self.assertEqual(analysis["offset_min"], -3600)
        self.assertEqual(analysis["offset_max"], 3600)
        by_zone = {("" if r["unclassified"] else r["zone"]): r for r in analysis["rows"]}
        # 冷藏 only ever holds OUT-003's 5 boxes; at -3600 s it crosses back
        # into 冷冻, so 冷藏 can reach zero.
        self.assertEqual(by_zone["冷藏"]["min_boxes"], 0)
        self.assertEqual(by_zone["冷藏"]["max_boxes"], 5)
        # 冷冻 always keeps OUT-001 (12) and can additionally gain OUT-003 (5).
        self.assertEqual(by_zone["冷冻"]["min_boxes"], 12)
        self.assertEqual(by_zone["冷冻"]["max_boxes"], 17)
        # OUT-003 crosses at -1800 s and stays 冷冻 for every smaller offset,
        # so the earliest offset attaining the maximum is the range's lower bound.
        self.assertEqual(by_zone["冷冻"]["max_offset"], -3600)
        self.assertEqual(by_zone["冷藏"]["min_offset"], -3600)
        # Every advertised offset lies inside the allowed range.
        for row in analysis["rows"]:
            for offset_field in ("min_offset", "max_offset"):
                self.assertBetween(row[offset_field], -3600, 3600)
        # Unclassified covers SKU-B (7) and SKU-C (2) = 9 boxes, unmovable.
        self.assertEqual(by_zone[""]["min_boxes"], 9)
        self.assertEqual(by_zone[""]["max_boxes"], 9)

    def assertBetween(self, value, low, high):
        self.assertTrue(low <= value <= high, (value, low, high))

    def test_original_report_still_uses_unshifted_instants(self):
        self._upload_and_version(-3600, 3600)
        body = self.client.get("/api/report?month=2026-03").get_json()
        # Exactly the same numbers as the zero-offset baseline report.
        self.assertEqual({z["zone"]: z["boxes"] for z in body["zones"]},
                         {"冷冻": 12, "冷藏": 5})
        self.assertEqual(body["unclassified"]["boxes"], 9)

    def test_zero_range_collapses_analysis_to_baseline(self):
        self._upload_and_version(0, 0)
        body = self.client.get("/api/report?month=2026-03").get_json()
        for row in body["clock_analysis"]["rows"]:
            self.assertEqual(row["min_boxes"], row["max_boxes"])
            self.assertEqual(row["min_offset"], 0)
            self.assertEqual(row["max_offset"], 0)

    def test_dedicated_clock_analysis_endpoint(self):
        self._upload_and_version(-600, 600)
        response = self.client.get("/api/clock-analysis?month=2026-03&version=1")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["month_fact"]["clock_offset_min"], -600)
        self.assertEqual(payload["clock_analysis"]["offset_max"], 600)

    def test_old_knowledge_version_keeps_its_own_analysis(self):
        self._upload_and_version(-3600, 3600)
        # v2 reclassifies SKU-A entirely as 常温.
        response = self.client.post(
            "/api/versions",
            json={
                "sku": "SKU-A",
                "reason": "v2",
                "base_version": 1,
                "intervals": [
                    {"zone": "常温", "valid_from": "2026-03-01T00:00:00Z", "valid_to": ""},
                ],
            },
        )
        self.assertEqual(response.status_code, 201)
        v1 = self.client.get("/api/report?month=2026-03&version=1").get_json()
        v2 = self.client.get("/api/report?month=2026-03&version=2").get_json()
        v1_zones = {r["zone"] for r in v1["clock_analysis"]["rows"] if not r["unclassified"]}
        v2_zones = {r["zone"] for r in v2["clock_analysis"]["rows"] if not r["unclassified"]}
        self.assertEqual(v1_zones, {"冷冻", "冷藏"})
        self.assertEqual(v2_zones, {"常温"})


class SweepAlgorithmTests(unittest.TestCase):
    """Direct algorithm tests, including a brute-force envelope check."""

    @staticmethod
    def _interval(zone, start, end=None):
        return {
            "zone": zone,
            "valid_from_us": datetime_to_us(start),
            "valid_to_us": datetime_to_us(end) if end else None,
        }

    def test_no_knowledge_means_everything_unclassified(self):
        shipments = [
            {"sku": "X", "occurred_at_us": datetime_to_us(datetime(2026, 3, 2, tzinfo=UTC)),
             "boxes": 3},
        ]
        result = clock.build_clock_analysis(shipments, {}, -100, 100)
        self.assertEqual(len(result["rows"]), 1)
        row = result["rows"][0]
        self.assertTrue(row["unclassified"])
        self.assertEqual(row["min_boxes"], 3)
        self.assertEqual(row["max_boxes"], 3)

    def test_single_offset_range(self):
        start = datetime(2026, 3, 1, tzinfo=UTC)
        grouped = {"X": [self._interval("冷藏", start)]}
        shipments = [
            {"sku": "X", "occurred_at_us": datetime_to_us(start + timedelta(seconds=10)),
             "boxes": 2},
        ]
        result = clock.build_clock_analysis(shipments, grouped, 30, 30)
        by_key = {("" if r["unclassified"] else r["zone"]): r for r in result["rows"]}
        self.assertEqual(set(by_key), {"", "冷藏"})
        row = by_key["冷藏"]
        self.assertEqual((row["min_boxes"], row["max_boxes"]), (2, 2))
        self.assertEqual((row["min_offset"], row["max_offset"]), (30, 30))
        self.assertEqual(by_key[""]["min_boxes"], 0)

    def test_matches_brute_force_on_fixed_scenario(self):
        # Two SKUs, gaps, adjacent zones and a shipment crossing a boundary
        # inside the range.
        base = datetime(2026, 3, 1, tzinfo=UTC)
        grouped = {
            "A": [
                self._interval("冷冻", base, base + timedelta(hours=10)),
                self._interval("冷藏", base + timedelta(hours=10), base + timedelta(hours=20)),
            ],
            "B": [self._interval("常温", base + timedelta(hours=5))],
        }
        shipments = [
            {"sku": "A", "occurred_at_us": datetime_to_us(base + timedelta(hours=9, minutes=59)),
             "boxes": 4},
            {"sku": "A", "occurred_at_us": datetime_to_us(base + timedelta(hours=15)),
             "boxes": 6},
            {"sku": "B", "occurred_at_us": datetime_to_us(base + timedelta(hours=4, minutes=59)),
             "boxes": 8},
            {"sku": "C", "occurred_at_us": datetime_to_us(base + timedelta(hours=2)),
             "boxes": 1},
        ]
        lo, hi = -3600, 3600
        result = clock.build_clock_analysis(shipments, grouped, lo, hi)

        # Brute-force envelope across every allowed offset.
        snapshots = []
        all_keys = {""}
        for offset in range(lo, hi + 1):
            totals = {}
            for s in shipments:
                iv = knowledge.classify(
                    s["occurred_at_us"] + offset * US, grouped.get(s["sku"], ())
                )
                key = iv["zone"] if iv else ""
                totals[key] = totals.get(key, 0) + s["boxes"]
            all_keys.update(totals)
            snapshots.append((offset, totals))
        envelope = {}
        for key in all_keys:
            values = [(off, totals.get(key, 0)) for off, totals in snapshots]
            min_off, min_val = min(values, key=lambda pair: (pair[1], pair[0]))
            max_off, max_val = max(values, key=lambda pair: (pair[1], -pair[0]))
            envelope[key] = {
                "min": min_val, "min_off": min_off,
                "max": max_val, "max_off": max_off,
            }

        by_key = {("" if r["unclassified"] else r["zone"]): r for r in result["rows"]}
        self.assertEqual(set(by_key), set(envelope))
        for key, tracked in envelope.items():
            row = by_key[key]
            self.assertEqual(row["min_boxes"], tracked["min"], key)
            self.assertEqual(row["max_boxes"], tracked["max"], key)
            self.assertIn(row["min_offset"], range(lo, hi + 1))
            self.assertIn(row["max_offset"], range(lo, hi + 1))
            # Advertised offsets must really attain the advertised totals.
            for offset_field, value_field in (
                ("min_offset", "min_boxes"),
                ("max_offset", "max_boxes"),
            ):
                offset = row[offset_field]
                totals = {}
                for s in shipments:
                    iv = knowledge.classify(
                        s["occurred_at_us"] + offset * US, grouped.get(s["sku"], ())
                    )
                    k2 = iv["zone"] if iv else ""
                    totals[k2] = totals.get(k2, 0) + s["boxes"]
                self.assertEqual(totals.get(key, 0), row[value_field])

    def test_fifty_thousand_rows_performance_and_event_bound(self):
        # The sweep must stay cheap even for 50k rows and the full 7201s
        # range: runtime is driven by events, not by the offset span.
        base = datetime(2026, 3, 1, tzinfo=UTC)
        grouped = {
            "A": [
                self._interval("冷冻", base, base + timedelta(days=15)),
                self._interval("冷藏", base + timedelta(days=15)),
            ],
        }
        shipments = []
        for i in range(50_000):
            shipments.append(
                {
                    "sku": "A",
                    "occurred_at_us": datetime_to_us(
                        base + timedelta(seconds=i * 50)
                    ),
                    "boxes": 1,
                }
            )
        import time

        started = time.perf_counter()
        result = clock.build_clock_analysis(shipments, grouped, -3600, 3600)
        elapsed = time.perf_counter() - started
        # 50k shipments over a one-hour range must finish in well under a
        # second; a per-offset recompute would do ~360 million classifications.
        self.assertLess(elapsed, 2.0)
        self.assertEqual(result["total_boxes"], 50_000)
        self.assertLessEqual(
            sum(1 for r in result["rows"]), 3
        )


class SchemaMigrationTests(unittest.TestCase):
    def test_legacy_database_gets_zero_offset_columns(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = str(Path(tmp.name) / "legacy.sqlite3")
        # Build the old schema without the new columns.
        conn = sqlite3.connect(path)
        try:
            conn.execute(
                "CREATE TABLE months (month TEXT PRIMARY KEY, source_filename TEXT NOT NULL,"
                " source_sha256 TEXT NOT NULL, row_count INTEGER NOT NULL,"
                " total_boxes INTEGER NOT NULL, frozen_at TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT INTO months VALUES ('2026-03', 'old.csv', 'x', 1, 4, '2026-03-31T00:00:00.000000Z')"
            )
            conn.commit()
        finally:
            conn.close()

        init_db(path)
        conn = connect(path)
        try:
            row = conn.execute(
                "SELECT clock_offset_min, clock_offset_max FROM months WHERE month = '2026-03'"
            ).fetchone()
            self.assertEqual((row["clock_offset_min"], row["clock_offset_max"]), (0, 0))
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
