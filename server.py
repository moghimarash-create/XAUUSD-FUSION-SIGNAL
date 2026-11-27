#!/usr/bin/env python3
"""
Production FastAPI Microservice for `XAUUSD-FUSION-SIGNAL` (`v5.0.0`) on Port `8080`.

- Bound to `0.0.0.0:${PORT:-8080}` with `RSS < 150 MiB` (serves precomputed compact JSON shards,
  local Google Sheet archive `data/gold_sheet_history.json` & `data/gold_sheet_signals.json`,
  and O(1) indexed Past/Present/Future Date Engine).
- Endpoints:
    * `GET  /api/health`
    * `GET  /api/signals/today`
    * `GET  /api/signals/by-date?date=YYYY-MM-DD`  (Past `2012..latest`, Present, & Future `> latest`)
    * `GET  /api/signals/history?limit=120&tier=ALL&day=ALL&q=&split=VAL_OOS`
    * `GET  /api/stats/summary`
    * `POST /api/refresh`  (Re-syncs Google Sheets CSVs + live spot quote)
    * `GET  /` -> Full Persian RTL Trading Radar, 3-Method Comparison Bar & Date Time-Machine (`static/index.html`)
"""
from __future__ import annotations

import datetime
import gc
import json
import os
import pickle
import resource
import threading
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
SHARDS_DIR = os.path.join(ROOT_DIR, "shards")
STATIC_DIR = os.path.join(ROOT_DIR, "static")

LATEST_SHARD_PATH = os.path.join(SHARDS_DIR, "latest_xauusd_signals.json")
SUMMARY_PATH = os.path.join(DATA_DIR, "xauusd_backtest_summary.json")
HISTORY_SHARD_PATH = os.path.join(DATA_DIR, "xauusd_history_shard.json")
ARCHIVE_PKL_PATH = os.path.join(DATA_DIR, "xauusd_twin_archive.pkl")

SERVICE_VERSION = "5.0.0"

_STATE_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {
    "today": None,
    "summary": None,
    "history": [],
    "history_by_date": {},
    "last_updated_utc": None,
    "engine_bundle": None,
}


def get_rss_mb() -> float:
    """Read current process resident set size (VmRSS) in MiB from Linux `/proc/self/status`."""
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = float(line.split()[1])
                    return round(kb / 1024.0, 2)
    except Exception:
        pass
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(float(ru) / 1024.0, 2)


def _atomic_write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _load_shards_into_memory() -> None:
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    today_data = None
    summary_data = None
    history_data: list[dict[str, Any]] = []

    if os.path.exists(LATEST_SHARD_PATH):
        with open(LATEST_SHARD_PATH, encoding="utf-8") as f:
            today_data = json.load(f)
            today_data["version"] = SERVICE_VERSION
            now_iso = today_data.get("lastUpdatedUtc", now_iso)

    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary_data = json.load(f)
            summary_data["version"] = SERVICE_VERSION

    if os.path.exists(HISTORY_SHARD_PATH):
        with open(HISTORY_SHARD_PATH, encoding="utf-8") as f:
            history_data = json.load(f)

    history_by_date = {str(r["date"]): r for r in history_data if "date" in r}

    if today_data and "liveSession" in today_data:
        live_dt = str(today_data["liveSession"].get("sessionDate", ""))
        next_dt = str(today_data.get("nextSessionPreview", {}).get("sessionDate", ""))
        sorted_h_dates = sorted(history_by_date.keys())
        prev_dt = sorted_h_dates[-2] if len(sorted_h_dates) >= 2 else live_dt
        h_live = history_by_date.get(live_dt, {})

        live_extra = {
            "sessionType": "LIVE",
            "isHistorical": True,
            "isFuture": False,
            "isLiveSession": True,
            "latestBarDate": live_dt,
            "prevSessionDate": prev_dt,
            "nextSessionDate": next_dt,
            "modeBadgeFa": "سشن زنده فعلی (Live Session)",
            "actualMove": h_live.get("actualMove"),
            "actualDir": h_live.get("actualDir"),
            "hitOutcome": h_live.get("outcomeBadge"),
            "dailyTargetWin": h_live.get("dailyTargetWin"),
            "dailyTargetPnl": h_live.get("dailyTargetPnl"),
            "execPnl": h_live.get("execPnl"),
            "execWin": h_live.get("execWin"),
            "rawClosePnl": h_live.get("rawClosePnl"),
            "rawCloseWin": h_live.get("rawCloseWin"),
        }
        today_data["liveSession"].update(live_extra)
        today_data.update(live_extra)

        if "nextSessionPreview" in today_data:
            nxt_iso = next_dt
            nxt_d = datetime.date.fromisoformat(nxt_iso) + datetime.timedelta(days=1)
            while nxt_d.weekday() >= 5:
                nxt_d += datetime.timedelta(days=1)
            today_data["nextSessionPreview"].update(
                {
                    "sessionType": "FUTURE",
                    "isHistorical": False,
                    "isFuture": True,
                    "isLiveSession": False,
                    "latestBarDate": live_dt,
                    "prevSessionDate": live_dt,
                    "nextSessionDate": nxt_d.strftime("%Y-%m-%d"),
                    "modeBadgeFa": "پیشبینی تاریخ آتی",
                    "futureProjectionNoteFa": (
                        f"پیشبینی تاریخ آتی ({nxt_iso}) — ماتریس ۱۲ ستونی شیت رسمی طلا + انتخابگر هوشمند ۳ روشه "
                        f"(LOCK3 / GEO / STEP) و سطوح ۸×۸ بر پایه آخرین اسپات لنگر شده‌اند."
                    ),
                    "actualMove": None,
                    "hitOutcome": "پیشبینی تاریخ آتی",
                    "dailyTargetWin": None,
                    "dailyTargetPnl": None,
                    "execPnl": None,
                }
            )

    with _STATE_LOCK:
        _CACHE["today"] = today_data
        _CACHE["summary"] = summary_data
        _CACHE["history"] = history_data
        _CACHE["history_by_date"] = history_by_date
        _CACHE["last_updated_utc"] = now_iso


def _ensure_engine_bundle() -> dict[str, Any]:
    """Lazily load and cache the compact twin archive + daily records index for `/api/signals/by-date`."""
    with _STATE_LOCK:
        bundle = _CACHE.get("engine_bundle")
        summary = _CACHE.get("summary")
    if bundle is not None:
        return bundle

    from src.xauusd_gold_cba_engine import FrozenGoldCbaConfig

    if summary is None:
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary = json.load(f)
    frozen_cfg = FrozenGoldCbaConfig(**summary["frozen_cfg"])

    latest_rec = None
    archive_full = None
    records_by_date = None

    if os.path.exists(ARCHIVE_PKL_PATH):
        try:
            with open(ARCHIVE_PKL_PATH, "rb") as pf:
                pk = pickle.load(pf)
            if pk.get("version") == SERVICE_VERSION:
                latest_rec = pk.get("latest_rec")
                archive_full = pk.get("archive_full")
                records_by_date = pk.get("records_by_date")
        except Exception:
            latest_rec = None

    if records_by_date is None or latest_rec is None or archive_full is None:
        from src.run_pipeline import refresh_live_signal_shard_only

        refresh_live_signal_shard_only(try_network_fetch=False)
        if os.path.exists(ARCHIVE_PKL_PATH):
            with open(ARCHIVE_PKL_PATH, "rb") as pf:
                pk = pickle.load(pf)
            archive_full = pk["archive_full"]
            records_by_date = pk.get("records_by_date")
            latest_rec = pk.get("latest_rec")

    if records_by_date is None or latest_rec is None:
        from src.data_loader import build_zero_leakage_daily_records, fetch_and_cache_market_data

        df_daily, _df_1h, df_sess = fetch_and_cache_market_data(force_refresh=False)
        recs = build_zero_leakage_daily_records(df_daily, df_sess)
        records_by_date = {r["date"]: r for r in recs if r["date"] >= "2012-01-01"}
        latest_rec = recs[-1]
        del df_daily, _df_1h, df_sess, recs
        gc.collect()

    sorted_dates = sorted(records_by_date.keys())
    bundle = {
        "records_by_date": records_by_date,
        "sorted_dates": sorted_dates,
        "archive_full": archive_full,
        "latest_rec": latest_rec,
        "frozen_cfg": frozen_cfg,
    }
    with _STATE_LOCK:
        _CACHE["engine_bundle"] = bundle
    return bundle


app = FastAPI(
    title="XAUUSD-FUSION-SIGNAL v5.0 Microservice",
    version=SERVICE_VERSION,
    description="True Sheet-Backed 12-Col Archive + 4-Method Smart Twin Selector (LOCK3/GEO/STEP/LOCK2) + 8v8 S/R Roster Fusion",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _on_startup() -> None:
    _load_shards_into_memory()
    gc.collect()


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    with _STATE_LOCK:
        last_upd = _CACHE["last_updated_utc"] or datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    return {
        "ok": True,
        "service": "XAUUSD-FUSION-SIGNAL",
        "version": SERVICE_VERSION,
        "anchor": "ANCHOR_DUAL_POLE_1800",
        "rssMb": get_rss_mb(),
        "lastUpdatedUtc": last_upd,
    }


@app.get("/api/signals/today")
def api_signals_today() -> dict[str, Any]:
    with _STATE_LOCK:
        payload = _CACHE["today"]
    if payload is None:
        raise HTTPException(status_code=503, detail="Signal shard not initialized yet.")
    return {
        **payload,
        "version": SERVICE_VERSION,
        "rssMb": get_rss_mb(),
    }


@app.get("/api/signals/by-date")
def api_signals_by_date(
    date: str = Query(..., description="Target date in YYYY-MM-DD format (Past 2012..latest, Present, or Future)"),
) -> dict[str, Any]:
    """
    Dynamically evaluate ANY requested date (`2012-01-01` .. `2035-12-31`) with strict `hist_date < sessionDate`.
    """
    from src.astro_ephemeris_engine import _DATE_CTX_CACHE, _SNAP_CACHE
    from src.price_roster_engine import _POLE_RES_CACHE
    from src.run_pipeline import build_signal_by_date_payload

    bundle = _ensure_engine_bundle()
    with _STATE_LOCK:
        history_by_date = _CACHE["history_by_date"]
        today_cached = _CACHE["today"] or {}
        live_spot = today_cached.get("spotPrice")

    try:
        res = build_signal_by_date_payload(
            requested_date_str=date,
            records_by_date=bundle["records_by_date"],
            sorted_dates=bundle["sorted_dates"],
            history_by_date=history_by_date,
            archive_full=bundle["archive_full"],
            latest_rec=bundle["latest_rec"],
            frozen_cfg=bundle["frozen_cfg"],
            live_spot=float(live_spot) if live_spot is not None else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if len(_SNAP_CACHE) > 48:
            _SNAP_CACHE.clear()
        if len(_DATE_CTX_CACHE) > 128:
            _DATE_CTX_CACHE.clear()
        if len(_POLE_RES_CACHE) > 512:
            _POLE_RES_CACHE.clear()

    res["rssMb"] = get_rss_mb()
    return res


@app.get("/api/signals/history")
def api_signals_history(
    limit: int = Query(default=120, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
    tier: str = Query(default="ALL"),
    day: str = Query(default="ALL"),
    split: str = Query(default="VAL_OOS"),
    q: str = Query(default=""),
) -> dict[str, Any]:
    with _STATE_LOCK:
        rows: list[dict[str, Any]] = list(_CACHE["history"] or [])

    tier_u = tier.strip().upper()
    day_u = day.strip().upper()
    split_u = split.strip().upper()
    q_s = q.strip().lower()

    filtered: list[dict[str, Any]] = []
    for r in rows:
        sp = str(r.get("split", "")).upper()
        if split_u == "VAL_OOS" and sp not in ("VAL", "OOS"):
            continue
        elif split_u in ("TRAIN", "VAL", "OOS") and sp != split_u:
            continue

        if tier_u == "C" and not r.get("isC"):
            continue
        elif tier_u == "B" and not r.get("isB"):
            continue
        elif tier_u == "A" and not r.get("isA"):
            continue
        elif tier_u in ("HOLD", "VETO", "VETOED") and r.get("isA"):
            continue

        if day_u not in ("", "ALL") and str(r.get("weekday", "")).upper() != day_u:
            continue

        if q_s:
            hay = (
                f"{r.get('date', '')} {r.get('weekday', '')} {r.get('direction', '')} "
                f"{r.get('outcomeBadge', '')} {r.get('selectedMethod', '')} {r.get('consensusBadgeFa', '')}"
            ).lower()
            if q_s not in hay:
                continue

        filtered.append(r)

    active_trades = [
        r
        for r in filtered
        if (r.get("isA") if tier_u not in ("HOLD", "VETO", "VETOED") else True)
        and r.get("actualDir") in ("XAU", "USD")
    ]
    n_trd = len(active_trades)
    if n_trd > 0:
        daily_wins = sum(1 for r in active_trades if r.get("dailyTargetWin", r.get("rawCloseWin")))
        exec_wins = sum(1 for r in active_trades if r.get("execWin"))
        raw_wins = sum(1 for r in active_trades if r.get("rawCloseWin"))
        tp1_hits = sum(1 for r in active_trades if r.get("hitTp1"))
        tp2_hits = sum(1 for r in active_trades if r.get("hitTp2"))
        tp3_hits = sum(1 for r in active_trades if r.get("hitTp3"))

        dpnls = [float(r.get("dailyTargetPnl", r.get("rawClosePnl", 0.0))) for r in active_trades]
        dgw = sum(p for p in dpnls if p > 0)
        dgl = abs(sum(p for p in dpnls if p < 0))
        dpf = round(dgw / dgl, 2) if dgl > 1e-6 else (99.99 if dgw > 0 else 0.0)

        pnls = [float(r.get("execPnl", 0.0)) for r in active_trades]
        gw = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p < 0))
        pf = round(gw / gl, 2) if gl > 1e-6 else (99.99 if gw > 0 else 0.0)
        exp_usd = round(sum(pnls) / n_trd, 2)
        kpi = {
            "trades": n_trd,
            "dailyWins": daily_wins,
            "dailyLosses": n_trd - daily_wins,
            "dailyWinRatePct": round(daily_wins * 100.0 / n_trd, 2),
            "dailyProfitFactor": dpf,
            "dailyExpectancyUsd": round(sum(dpnls) / n_trd, 2),
            "execWins": exec_wins,
            "execLosses": n_trd - exec_wins,
            "execWinRatePct": round(exec_wins * 100.0 / n_trd, 2),
            "rawWinRatePct": round(raw_wins * 100.0 / n_trd, 2),
            "tp1HitRatePct": round(tp1_hits * 100.0 / n_trd, 1),
            "tp2HitRatePct": round(tp2_hits * 100.0 / n_trd, 1),
            "tp3HitRatePct": round(tp3_hits * 100.0 / n_trd, 1),
            "profitFactor": pf,
            "expectancyUsd": exp_usd,
            "netPnlUsd": round(sum(pnls), 2),
        }
    else:
        kpi = {
            "trades": 0,
            "dailyWins": 0,
            "dailyLosses": 0,
            "dailyWinRatePct": 0.0,
            "dailyProfitFactor": 0.0,
            "dailyExpectancyUsd": 0.0,
            "execWins": 0,
            "execLosses": 0,
            "execWinRatePct": 0.0,
            "rawWinRatePct": 0.0,
            "tp1HitRatePct": 0.0,
            "tp2HitRatePct": 0.0,
            "tp3HitRatePct": 0.0,
            "profitFactor": 0.0,
            "expectancyUsd": 0.0,
            "netPnlUsd": 0.0,
        }

    sliced = filtered[offset : offset + limit]
    return {
        "ok": True,
        "version": SERVICE_VERSION,
        "totalMatched": len(filtered),
        "offset": offset,
        "limit": limit,
        "filters": {"tier": tier_u, "day": day_u, "split": split_u, "q": q},
        "kpi": kpi,
        "items": sliced,
    }


@app.get("/api/stats/summary")
def api_stats_summary() -> dict[str, Any]:
    with _STATE_LOCK:
        summary = _CACHE["summary"]
    if summary is None:
        raise HTTPException(status_code=503, detail="Summary stats shard not initialized yet.")
    return {
        "ok": True,
        "rssMb": get_rss_mb(),
        **summary,
        "version": SERVICE_VERSION,
    }


@app.post("/api/refresh")
def api_refresh() -> dict[str, Any]:
    """
    Re-sync Google Sheets CSVs (`HISTORY_CSV` & `SIGNALS_CSV`), pull latest live `XAUUSD` quote,
    atomically refresh `shards/latest_xauusd_signals.json`, and purge transient caches (`RSS < 150 MiB`).
    """
    from src.astro_ephemeris_engine import _DATE_CTX_CACHE, _SNAP_CACHE
    from src.price_roster_engine import _POLE_RES_CACHE
    from src.run_pipeline import refresh_live_signal_shard_only

    try:
        fresh_shard = refresh_live_signal_shard_only(try_network_fetch=True)
        _atomic_write_json(LATEST_SHARD_PATH, fresh_shard)
        with _STATE_LOCK:
            if fresh_shard.get("sheetsSynced"):
                _CACHE["engine_bundle"] = None
        _load_shards_into_memory()
        with _STATE_LOCK:
            updated_today = _CACHE["today"] or fresh_shard
    finally:
        _SNAP_CACHE.clear()
        _DATE_CTX_CACHE.clear()
        _POLE_RES_CACHE.clear()
        gc.collect()

    return {
        "ok": True,
        "refreshed": True,
        "rssMb": get_rss_mb(),
        **updated_today,
    }


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_dashboard_root() -> Any:
    idx_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(idx_path):
        return FileResponse(idx_path, media_type="text/html; charset=utf-8")
    return JSONResponse({"ok": True, "service": "XAUUSD-FUSION-SIGNAL", "version": SERVICE_VERSION})


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, workers=1)
