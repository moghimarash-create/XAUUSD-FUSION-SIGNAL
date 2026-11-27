#!/usr/bin/env python3
"""
Authentic Google Sheets Gold Elemental Archive Loader & Sync Engine (`src/sheet_archive_loader.py`).

Bundles & syncs the authentic 4,705-row Gold `12-Col` elemental archive (`2012–2027`):
- `HISTORY_CSV`: cols `0=DAY`, `1=DATE`, `2=DAILY PRIC`, `4..7=ELE-P-1..4`, `9..12=ELE-FINAL-1..4`, `14..17=ELE-PRIM-1..4`, `18=GEO-FIG`
- `SIGNALS_CSV`: live/upcoming `2025–2027` search sheet (`556` weekday rows)
- Cached locally in `data/gold_sheet_history.json` & `data/gold_sheet_signals.json` for zero runtime latency,
  with `POST /api/refresh` able to re-sync live over HTTPS.
"""
from __future__ import annotations

import csv
import datetime
import io
import json
import os
from typing import Any, Final

import requests

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")

HISTORY_CSV_URL: Final[str] = (
    "https://docs.google.com/spreadsheets/d/1AUQrY_jZFrMEcb90BE4X6231YGM8lg2og1WGRgOxsMo/export?format=csv&gid=0"
)
SIGNALS_CSV_URL: Final[str] = (
    "https://docs.google.com/spreadsheets/d/1ma2rSL9EHXseKpT2QKxGEDAGwb6XrtRuHeOsxHriqrc/export?format=csv&gid=0"
)

GOLD_SHEET_HISTORY_JSON: Final[str] = os.path.join(DATA_DIR, "gold_sheet_history.json")
GOLD_SHEET_SIGNALS_JSON: Final[str] = os.path.join(DATA_DIR, "gold_sheet_signals.json")

VALID_WEEKDAYS: Final[set[str]] = {"MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY"}

_SHEET_MEM_CACHE: dict[str, Any] = {
    "history": None,
    "signals": None,
    "merged_by_date": None,
}


def _parse_elem_int(cell: str) -> int:
    s = str(cell or "").strip()
    if not s:
        return 0
    tok = s.split()[0]
    return int(float(tok))


def _parse_sheet_date(raw_dt: str) -> str | None:
    s = str(raw_dt or "").strip()
    if not s:
        return None
    if "-" in s and len(s) >= 10:
        return s[:10]
    parts = s.split("/")
    if len(parts) != 3:
        return None
    m, d, y = int(parts[0]), int(parts[1]), int(parts[2])
    if y < 100:
        y += 2000
    return f"{y:04d}-{m:02d}-{d:02d}"


def parse_gold_sheet_csv_text(csv_text: str, source_url: str) -> dict[str, Any]:
    """Parse raw Google Sheet CSV text into canonical zero-latency JSON archive structure."""
    reader = csv.reader(io.StringIO(csv_text))
    all_rows = list(reader)
    raw_row_count = len(all_rows)
    if raw_row_count < 2:
        raise ValueError(f"Empty or invalid Google Sheet CSV from {source_url}")

    parsed_rows: list[dict[str, Any]] = []
    weekly_count = 0

    for r in all_rows[1:]:
        if not r or len(r) < 18:
            continue
        day = str(r[0] or "").strip().upper()
        if day == "WEEKLY":
            weekly_count += 1
            continue
        if day not in VALID_WEEKDAYS:
            continue
        iso = _parse_sheet_date(r[1])
        if not iso:
            continue

        pr_raw = str(r[2] or "").strip().replace(",", "")
        price: float | None = None
        if pr_raw not in ("", "-", "N/A", "#N/A"):
            try:
                price = round(float(pr_raw), 4)
            except ValueError:
                price = None

        elep = [_parse_elem_int(r[4 + k]) for k in range(4)]
        fin = [_parse_elem_int(r[9 + k]) for k in range(4)]
        prim = [_parse_elem_int(r[14 + k]) for k in range(4)]
        # Canonical 4-bit Geomantic parity of ELE-PRIM (`=ELEM_TO_CODE(O,P,Q,R)`)
        geo_fig = "".join(str(p % 2) for p in prim)

        parsed_rows.append(
            {
                "date": iso,
                "day": day,
                "price": price,
                "elep": elep,
                "final": fin,
                "prim": prim,
                "geo_fig": geo_fig,
            }
        )

    parsed_rows.sort(key=lambda x: x["date"])
    # Deduplicate by date keeping latest non-null price if duplicated
    dedup: dict[str, dict[str, Any]] = {}
    for item in parsed_rows:
        dt = item["date"]
        if dt not in dedup or (dedup[dt]["price"] is None and item["price"] is not None):
            dedup[dt] = item
    final_rows = [dedup[k] for k in sorted(dedup.keys())]

    priced_count = sum(1 for x in final_rows if x["price"] is not None)
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "sourceUrl": source_url,
        "syncedUtc": now_iso,
        "rawRowCount": raw_row_count,
        "weekdayRowCount": len(final_rows),
        "weeklyRowCount": weekly_count,
        "pricedRowCount": priced_count,
        "minDate": final_rows[0]["date"] if final_rows else "",
        "maxDate": final_rows[-1]["date"] if final_rows else "",
        "rows": final_rows,
    }


def _atomic_save_json(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def sync_gold_sheets_from_remote(
    timeout: tuple[float, float] = (3.0, 8.0),
    local_csv_fallbacks: tuple[str | None, str | None] = (None, None),
) -> tuple[bool, str]:
    """
    Fetch `HISTORY_CSV_URL` and `SIGNALS_CSV_URL` (or local CSV fallbacks during initial bootstrap),
    save `data/gold_sheet_history.json` & `data/gold_sheet_signals.json`, and invalidate in-memory cache.
    Returns `(updated_ok, status_label)`.
    """
    hist_csv_fb, sig_csv_fb = local_csv_fallbacks
    headers = {"User-Agent": "Mozilla/5.0 XAUUSD-Fusion-Signal/5.0"}

    hist_text: str | None = None
    sig_text: str | None = None
    source_mode = "GOOGLE_SHEETS_LIVE_CSV"

    try:
        r_h = requests.get(HISTORY_CSV_URL, headers=headers, timeout=timeout)
        r_s = requests.get(SIGNALS_CSV_URL, headers=headers, timeout=timeout)
        if r_h.status_code == 200 and r_s.status_code == 200 and "ELE-PRIM-1" in r_h.text:
            hist_text = r_h.text
            sig_text = r_s.text
    except Exception:
        hist_text = None
        sig_text = None

    if (hist_text is None or sig_text is None) and hist_csv_fb and sig_csv_fb:
        if os.path.exists(hist_csv_fb) and os.path.exists(sig_csv_fb):
            with open(hist_csv_fb, encoding="utf-8") as fh:
                hist_text = fh.read()
            with open(sig_csv_fb, encoding="utf-8") as fs:
                sig_text = fs.read()
            source_mode = "LOCAL_CSV_BOOTSTRAP"

    if hist_text is None or sig_text is None:
        return False, "CACHED_LOCAL_SHEET_JSON"

    hist_bundle = parse_gold_sheet_csv_text(hist_text, HISTORY_CSV_URL)
    sig_bundle = parse_gold_sheet_csv_text(sig_text, SIGNALS_CSV_URL)
    if hist_bundle["weekdayRowCount"] < 3500 or sig_bundle["weekdayRowCount"] < 400:
        return False, "INVALID_REMOTE_SHEET_COUNT"

    _atomic_save_json(GOLD_SHEET_HISTORY_JSON, hist_bundle)
    _atomic_save_json(GOLD_SHEET_SIGNALS_JSON, sig_bundle)
    _SHEET_MEM_CACHE["history"] = hist_bundle
    _SHEET_MEM_CACHE["signals"] = sig_bundle
    _SHEET_MEM_CACHE["merged_by_date"] = None
    return True, source_mode


def load_gold_sheet_bundles(auto_sync_if_missing: bool = True) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """
    Load `data/gold_sheet_history.json` & `data/gold_sheet_signals.json` in O(1) from memory or local disk.
    Returns `(history_bundle, signals_bundle, merged_by_date)`.
    """
    if (
        _SHEET_MEM_CACHE["history"] is not None
        and _SHEET_MEM_CACHE["signals"] is not None
        and _SHEET_MEM_CACHE["merged_by_date"] is not None
    ):
        return (
            _SHEET_MEM_CACHE["history"],
            _SHEET_MEM_CACHE["signals"],
            _SHEET_MEM_CACHE["merged_by_date"],
        )

    if not (os.path.exists(GOLD_SHEET_HISTORY_JSON) and os.path.exists(GOLD_SHEET_SIGNALS_JSON)):
        if auto_sync_if_missing:
            sync_gold_sheets_from_remote(
                timeout=(4.0, 12.0),
                local_csv_fallbacks=("/tmp/gold_sheet_history.csv", "/tmp/gold_sheet_signals.csv"),
            )

    with open(GOLD_SHEET_HISTORY_JSON, encoding="utf-8") as fh:
        hist_bundle = json.load(fh)
    with open(GOLD_SHEET_SIGNALS_JSON, encoding="utf-8") as fs:
        sig_bundle = json.load(fs)

    merged: dict[str, dict[str, Any]] = {}
    for r in hist_bundle.get("rows", []):
        merged[r["date"]] = r
    for r in sig_bundle.get("rows", []):
        dt = r["date"]
        if dt not in merged:
            merged[dt] = r
        elif merged[dt].get("price") is None and r.get("price") is not None:
            merged[dt] = {**merged[dt], "price": r["price"]}

    _SHEET_MEM_CACHE["history"] = hist_bundle
    _SHEET_MEM_CACHE["signals"] = sig_bundle
    _SHEET_MEM_CACHE["merged_by_date"] = merged
    return hist_bundle, sig_bundle, merged


if __name__ == "__main__":
    ok, mode = sync_gold_sheets_from_remote(
        timeout=(5.0, 15.0),
        local_csv_fallbacks=("/tmp/gold_sheet_history.csv", "/tmp/gold_sheet_signals.csv"),
    )
    hb, sb, mp = load_gold_sheet_bundles(False)
    print(
        f"Synced={ok} ({mode}) | History: raw={hb['rawRowCount']}, weekdays={hb['weekdayRowCount']}, "
        f"priced={hb['pricedRowCount']} ({hb['minDate']}..{hb['maxDate']}) | "
        f"Signals: raw={sb['rawRowCount']}, weekdays={sb['weekdayRowCount']} | Merged dates={len(mp)}"
    )
