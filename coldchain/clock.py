"""Clock-offset sensitivity analysis.

The outbound device's clock bias is only known as an integer-second range
``[offset_min, offset_max]`` shared by every shipment of one month. For each
temperature zone (and for unclassified shipments) we need the minimum and
maximum box total reachable by *some* offset in that range, together with an
offset attaining each extreme.

Naively this means re-classifying the whole month once per allowed offset,
which is ``O(rows * (max - min))`` — up to 50,000 * 7,201 classifications.
That is unnecessary. A shipment's zone can only change when the shifted
instant crosses one of its SKU's interval boundaries, so each shipment
contributes at most a handful of *events* ``(offset, zone_before, zone_after)``
inside the range. We then sweep every event once in offset order and keep
running per-zone totals. The whole analysis is

    O(rows * log(intervals_per_sku) + events * log events)

and never re-reads or re-classifies the month per offset.
"""

from __future__ import annotations

import bisect

UNCLASSIFIED = ""
US_PER_SECOND = 1_000_000


def _classify(ts_us, intervals):
    # Imported lazily: knowledge.py imports this module at load time, so a
    # module-level import would be circular.
    from .knowledge import classify

    return classify(ts_us, intervals)


def _ceil_div(value, divisor):
    """Ceiling of integer division that also works for negative values."""
    return -((-value) // divisor)


def _build_sku_boundaries(grouped):
    """Per-SKU sorted boundary microseconds and the zone entered there.

    An interval ``[start, end)`` creates two boundaries: at ``start`` the
    interval's zone begins, at ``end`` it ends. Adjacent intervals naturally
    share a boundary timestamp; the corresponding entries are both emitted
    and applied in order during the sweep.
    """
    boundaries_by_sku = {}
    for sku, intervals in grouped.items():
        events = []
        for item in intervals:
            events.append((item["valid_from_us"], item["zone"]))
            if item["valid_to_us"] is not None:
                events.append((item["valid_to_us"], None))
        events.sort(key=lambda pair: pair[0])
        boundaries_by_sku[sku] = events
    return boundaries_by_sku


def shipment_events(shipment, intervals, boundary_pairs, offset_min, offset_max):
    """Yield the shipment's zone-change events over the offset range.

    The first yielded tuple is ``(offset_min, None, zone_at_min)`` and seeds
    the sweep; every later tuple ``(offset, zone_before, zone_after)`` marks
    an integer offset inside the range where the shifted instant enters a
    different zone.

    The shifted instant is ``occurred_at_us + offset * 1_000_000``. Crossing
    a boundary ``b`` happens at the smallest integer offset with
    ``t + offset >= b``, i.e. ``offset = ceil((b - t) / 1_000_000)``.
    Boundaries on either side of ``t`` matter: negative offsets cross
    boundaries below the original instant and positive offsets cross ones
    above it, so the candidate window is the whole interval
    ``[t + offset_min, t + offset_max]``.
    """
    t_us = shipment["occurred_at_us"]
    if not boundary_pairs:
        return

    window_low = t_us + offset_min * US_PER_SECOND
    window_high = t_us + offset_max * US_PER_SECOND

    # State at the smallest allowed offset, classified with the same
    # half-open semantics as the ordinary report (start <= t < end).
    start_interval = _classify(window_low, intervals)
    current_zone = start_interval["zone"] if start_interval else None
    yield offset_min, None, current_zone

    # Only boundaries the shifted instant can actually reach while the
    # offset stays in the allowed range can change the zone.
    first_index = bisect.bisect_right(boundary_pairs, window_low, key=lambda pair: pair[0])
    last_index = bisect.bisect_right(boundary_pairs, window_high, key=lambda pair: pair[0])
    for boundary_us, _entered_zone in boundary_pairs[first_index:last_index]:
        offset_s = _ceil_div(boundary_us - t_us, US_PER_SECOND)
        shifted = t_us + offset_s * US_PER_SECOND
        # Classify the shifted instant directly so adjacent intervals
        # sharing a boundary timestamp, gaps and half-open endpoints are all
        # handled by the same code path as the ordinary report.
        interval = _classify(shifted, intervals)
        new_zone = interval["zone"] if interval else None
        if new_zone == current_zone:
            continue
        yield offset_s, current_zone, new_zone
        current_zone = new_zone


def build_clock_analysis(shipments, grouped, offset_min, offset_max):
    """Compute per-zone reachable box bounds over the offset range.

    Parameters
    ----------
    shipments:
        Iterable of rows with at least ``sku``, ``occurred_at_us`` and
        ``boxes`` (the same rows the ordinary report classifies).
    grouped:
        Mapping ``sku -> sorted non-overlapping intervals`` from the
        selected knowledge version (``knowledge.ranges_by_sku`` shape).
    offset_min, offset_max:
        Inclusive integer-second range stored with the month fact.

    Returns a dict with the range, total boxes and one row per zone that is
    reachable at all (plus an always-present unclassified row). Every row
    carries ``min_boxes``/``min_offset`` and ``max_boxes``/``max_offset``.
    """
    if offset_min > offset_max:
        raise ValueError("offset_min must not exceed offset_max")

    boundaries_by_sku = _build_sku_boundaries(grouped)

    # ``current_key[shipment_id]`` is the bucket that shipment's boxes are
    # counted in at the offset currently being swept. Events only carry the
    # destination bucket; the source always comes from this state map, so a
    # shipment can never be subtracted from a bucket it no longer occupies.
    current_key = {}
    totals = {}
    events = []
    total_boxes = 0

    for shipment_id, shipment in enumerate(shipments):
        boxes = shipment["boxes"]
        total_boxes += boxes
        sku = shipment["sku"]
        intervals = grouped.get(sku, ())
        if not intervals:
            # The SKU has no knowledge in this version: unclassified for
            # every offset, no events can ever change that.
            current_key[shipment_id] = UNCLASSIFIED
            totals[UNCLASSIFIED] = totals.get(UNCLASSIFIED, 0) + boxes
            continue

        boundary_pairs = boundaries_by_sku[sku]
        started = False
        for offset_s, _zone_before, zone_after in shipment_events(
            shipment, intervals, boundary_pairs, offset_min, offset_max
        ):
            key_after = zone_after or UNCLASSIFIED
            if not started:
                # Seed event at offset_min describing the starting state.
                current_key[shipment_id] = key_after
                totals[key_after] = totals.get(key_after, 0) + boxes
                started = True
            else:
                events.append((offset_s, shipment_id, key_after, boxes))
        if not started:
            # No boundary inside the range at all: the original
            # classification holds for every allowed offset.
            interval = _classify(shipment["occurred_at_us"], intervals)
            key = interval["zone"] if interval else UNCLASSIFIED
            current_key[shipment_id] = key
            totals[key] = totals.get(key, 0) + boxes

    # Sweep once through all zone-change events in offset order. Keeping the
    # running totals means the month is never reclassified per offset.
    events.sort(key=lambda item: item[0])

    # Every bucket that can be reached at any offset is tracked from the
    # start, including buckets whose initial total is zero: a zero-valued
    # bucket never appears in ``totals`` until its first shipment arrives,
    # so pre-registering it here is the only way to record its later maximum.
    reachable_keys = set(totals)
    for _offset_s, _shipment_id, key_after, _boxes in events:
        reachable_keys.add(key_after)
    reachable_keys.add(UNCLASSIFIED)
    extremes = {
        key: {
            "min_boxes": None,
            "min_offset": offset_min,
            "max_boxes": 0,
            "max_offset": offset_min,
        }
        for key in reachable_keys
    }

    def snapshot(offsets_seen):
        for key in reachable_keys:
            value = totals.get(key, 0)
            tracked = extremes[key]
            # Strict comparisons only: snapshots run in ascending offset
            # order, so keeping the first hit naturally records the smallest
            # offset that attains each extreme.
            if tracked["min_boxes"] is None or value < tracked["min_boxes"]:
                tracked["min_boxes"] = value
                tracked["min_offset"] = offsets_seen
            if value > tracked["max_boxes"]:
                tracked["max_boxes"] = value
                tracked["max_offset"] = offsets_seen

    snapshot(offset_min)
    index = 0
    while index < len(events):
        offset_s = events[index][0]
        moved = False
        while index < len(events) and events[index][0] == offset_s:
            _, shipment_id, key_after, boxes = events[index]
            index += 1
            key_before = current_key[shipment_id]
            if key_before == key_after:
                continue
            totals[key_before] -= boxes
            totals[key_after] = totals.get(key_after, 0) + boxes
            current_key[shipment_id] = key_after
            moved = True
        if moved:
            snapshot(offset_s)

    rows = []
    for key in sorted(extremes):
        tracked = extremes[key]
        rows.append(
            {
                "zone": None if key == UNCLASSIFIED else key,
                "unclassified": key == UNCLASSIFIED,
                "min_boxes": tracked["min_boxes"],
                "min_offset": tracked["min_offset"],
                "max_boxes": tracked["max_boxes"],
                "max_offset": tracked["max_offset"],
            }
        )

    # Note: the per-bucket extrema need not share an offset, so their sums
    # are not expected to equal the month total. Conservation is instead
    # enforced structurally: every sweep move subtracts and adds the same
    # box count, and the seeding step already assigned every box once.
    if __debug__:
        assert sum(totals.values()) == total_boxes
        assert all(row["min_boxes"] is not None for row in rows)

    return {
        "offset_min": offset_min,
        "offset_max": offset_max,
        "total_boxes": total_boxes,
        "rows": rows,
    }
