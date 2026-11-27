#!/usr/bin/env python3
"""
Production FastAPI Microservice for `XAUUSD-FUSION-SIGNAL` (`v4.0.0`) on Port `8080`.

- Bound to `0.0.0.0:${PORT:-8080}` with `RSS < 150 MiB` (serves precomputed compact JSON shards
  in O(1) memory and clears transient ephemeris caches after `POST /api/refresh`).
- Endpoints:
    * `GET  /api/health`
    * `GET  /api/signals/today`
    * `GET  /api/signals/history?limit=120&tier=ALL&day=ALL&q=&split=VAL_OOS`
    * `GET  /api/stats/summary`
    * `POST /api/refresh`
    * `GET  /` -> Full Persian RTL Trading Radar & 12-Col Twin Explorer (`static/index.html`)
"""
from __future__ import annotations

import datetime
import gc
import json
import os
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

_STATE_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {
    "today": None,
    "summary": None,
    "history": [],
    "last_updated_utc": None,
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
            now_iso = today_data.get("lastUpdatedUtc", now_iso)

    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, encoding="utf-8") as f:
            summary_data = json.load(f)

    if os.path.exists(HISTORY_SHARD_PATH):
        with open(HISTORY_SHARD_PATH, encoding="utf-8") as f:
            history_data = json.load(f)

    with _STATE_LOCK:
        _CACHE["today"] = today_data
        _CACHE["summary"] = summary_data
        _CACHE["history"] = history_data
        _CACHE["last_updated_utc"] = now_iso


app = FastAPI(
    title="XAUUSD-FUSION-SIGNAL v4 Microservice",
    version="4.0.0",
    description="Non-Parametric Twin-First Astro-Elemental-Geomantic Spot Gold Signal Engine",
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
        "version": "4.0.0",
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
        "rssMb": get_rss_mb(),
    }


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
            hay = f"{r.get('date', '')} {r.get('weekday', '')} {r.get('direction', '')} {r.get('outcomeBadge', '')}".lower()
            if q_s not in hay:
                continue

        filtered.append(r)

    # Compute dynamic KPI summary across active filtered trades (or all filtered rows if viewing VETOED)
    active_trades = [
        r for r in filtered
        if (r.get("isA") if tier_u not in ("HOLD", "VETO", "VETOED") else True)
        and r.get("actualDir") in ("XAU", "USD")
    ]
    n_trd = len(active_trades)
    if n_trd > 0:
        exec_wins = sum(1 for r in active_trades if r.get("execWin"))
        raw_wins = sum(1 for r in active_trades if r.get("rawCloseWin"))
        tp1_hits = sum(1 for r in active_trades if r.get("hitTp1"))
        tp2_hits = sum(1 for r in active_trades if r.get("hitTp2"))
        tp3_hits = sum(1 for r in active_trades if r.get("hitTp3"))
        pnls = [float(r.get("execPnl", 0.0)) for r in active_trades]
        gw = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p < 0))
        pf = round(gw / gl, 2) if gl > 1e-6 else (99.99 if gw > 0 else 0.0)
        exp_usd = round(sum(pnls) / n_trd, 2)
        kpi = {
            "trades": n_trd,
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
    }


@app.post("/api/refresh")
def api_refresh() -> dict[str, Any]:
    """
    Pull latest live `XAUUSD` & `DXY` bars (with graceful offline/cache fallback),
    atomically refresh `shards/latest_xauusd_signals.json`, purge transient caches,
    and update in-memory state while keeping `RSS < 150 MiB`.
    """
    from src.astro_ephemeris_engine import _DATE_CTX_CACHE, _SNAP_CACHE
    from src.price_roster_engine import _POLE_RES_CACHE
    from src.run_pipeline import refresh_live_signal_shard_only

    try:
        fresh_shard = refresh_live_signal_shard_only(try_network_fetch=True)
        _atomic_write_json(LATEST_SHARD_PATH, fresh_shard)
        with _STATE_LOCK:
            _CACHE["today"] = fresh_shard
            _CACHE["last_updated_utc"] = fresh_shard["lastUpdatedUtc"]
    finally:
        # Release transient ephemeris lookup tables so web worker RSS stays < 95 MiB
        _SNAP_CACHE.clear()
        _DATE_CTX_CACHE.clear()
        _POLE_RES_CACHE.clear()
        gc.collect()

    return {
        "ok": True,
        "refreshed": True,
        "rssMb": get_rss_mb(),
        **fresh_shard,
    }


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_dashboard_root() -> Any:
    idx_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(idx_path):
        return FileResponse(idx_path, media_type="text/html; charset=utf-8")
    return JSONResponse({"ok": True, "service": "XAUUSD-FUSION-SIGNAL", "ui": "pending"})


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, workers=1)
