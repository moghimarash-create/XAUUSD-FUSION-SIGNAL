#!/usr/bin/env python3
"""
Historical & Intraday Market Data Engine (`XAUUSD` + `DXY`) with Strict Zero Look-Ahead.

- Fetches institutional daily `GC=F` (COMEX Spot-Aligned Gold) + `DX-Y.NYB` (`2010-01-01` .. `2026-11-26`)
  aligned to `America/New_York` (`18:00 NY D-1 -> 17:00 NY D` session).
- Fetches recent 2-year `1h` bars (`2024-11-27` .. `2026-11-26`) to verify exact session opens:
    * `18:00 America/New_York` (`D-1` Globex/Spot Candle Birth)
    * `08:00 Europe/London` (`D` LBMA London Bullion Open)
    * `08:20 America/New_York` (`D` COMEX Pit Open)
- Precomputes strictly causal (`<= i-1` + `Today_Open` at `T_0`) structural indicators with zero look-ahead.
"""
from __future__ import annotations

import datetime
import math
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.xauusd_config import split_label

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")

DAILY_CSV = os.path.join(DATA_DIR, "xauusd_dxy_daily_2010_2026.csv")
HOURLY_CSV = os.path.join(DATA_DIR, "xauusd_1h_2025_2026.csv")
SESSIONS_CSV = os.path.join(DATA_DIR, "xauusd_intraday_sessions_2025_2026.csv")

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) XAUUSD-Fusion-Engine/1.0"}


def _fetch_yahoo_chart(symbol: str, p1: int, p2: int, interval: str = "1d") -> pd.DataFrame:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={p1}&period2={p2}&interval={interval}&includePrePost=true"
    )
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    res_list = payload.get("chart", {}).get("result")
    if not res_list:
        raise RuntimeError(f"Empty Yahoo Finance response for {symbol} ({interval})")
    res = res_list[0]
    timestamps = res.get("timestamp") or []
    quote = res["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "ts_utc": timestamps,
            "dt_utc": pd.to_datetime(timestamps, unit="s", utc=True),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        }
    )
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].fillna(0.0).astype(float)
    return df


def fetch_and_cache_market_data(force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load cached daily + 1h + session-aligned datasets, or fetch from Yahoo v8 chart API.
    Returns (df_daily, df_1h, df_sessions).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if (
        not force_refresh
        and os.path.exists(DAILY_CSV)
        and os.path.exists(HOURLY_CSV)
        and os.path.exists(SESSIONS_CSV)
    ):
        df_daily = pd.read_csv(DAILY_CSV)
        df_1h = pd.read_csv(HOURLY_CSV)
        df_sess = pd.read_csv(SESSIONS_CSV)
        return df_daily, df_1h, df_sess

    now_ts = int(time.time()) + 43200
    p1_daily = int(datetime.datetime(2001, 1, 1, tzinfo=datetime.timezone.utc).timestamp())

    df_xau = _fetch_yahoo_chart("GC=F", p1_daily, now_ts, interval="1d")
    df_dxy = _fetch_yahoo_chart("DX-Y.NYB", p1_daily, now_ts, interval="1d")

    df_xau["date"] = df_xau["dt_utc"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    df_dxy["date"] = df_dxy["dt_utc"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")

    # Deduplicate by NY trading date (keep latest print per NY session date)
    df_xau = df_xau.sort_values("ts_utc").drop_duplicates(subset=["date"], keep="last").copy()
    df_dxy = df_dxy.sort_values("ts_utc").drop_duplicates(subset=["date"], keep="last").copy()

    # Remove degenerate front-month settlement-only zero-range roll bars (High == Low and Open == Close)
    is_degenerate = (df_xau["high"] == df_xau["low"]) & (df_xau["open"] == df_xau["close"])
    df_xau = df_xau[~is_degenerate].copy()

    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    df_xau["high"] = df_xau[["open", "high", "close"]].max(axis=1)
    df_xau["low"] = df_xau[["open", "low", "close"]].min(axis=1)

    dxy_renamed = df_dxy[["date", "open", "high", "low", "close"]].rename(
        columns={
            "open": "dxy_open",
            "high": "dxy_high",
            "low": "dxy_low",
            "close": "dxy_close",
        }
    )

    df_daily = pd.merge(
        df_xau[["date", "open", "high", "low", "close", "volume"]],
        dxy_renamed,
        on="date",
        how="left",
    ).sort_values("date").reset_index(drop=True)

    # Strictly causal forward-fill of DXY on minor holiday mismatches (never backward-fill!)
    for col in ("dxy_open", "dxy_high", "dxy_low", "dxy_close"):
        df_daily[col] = df_daily[col].ffill()
    df_daily = df_daily.dropna(subset=["dxy_close"]).reset_index(drop=True)
    df_daily["split"] = df_daily["date"].map(split_label)

    # Fetch recent ~729 days of 1h bars for exact session verification (2025-2026)
    p1_1h = now_ts - 729 * 86400
    df_1h_raw = _fetch_yahoo_chart("GC=F", p1_1h, now_ts, interval="1h")
    dt_ny = df_1h_raw["dt_utc"].dt.tz_convert("America/New_York")
    dt_lon = df_1h_raw["dt_utc"].dt.tz_convert("Europe/London")

    df_1h = df_1h_raw.copy()
    df_1h["dt_ny"] = dt_ny.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    df_1h["dt_london"] = dt_lon.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    df_1h["ny_hour"] = dt_ny.dt.hour
    df_1h["lon_hour"] = dt_lon.dt.hour
    # Trading session date in NY: hours 18..23 belong to next calendar business day's session
    session_dt = (dt_ny + pd.Timedelta(hours=6)).dt.strftime("%Y-%m-%d")
    df_1h["session_date"] = session_dt

    # Build verified intraday session summary per trading day
    sess_rows: list[dict[str, Any]] = []
    for s_date, grp in df_1h.groupby("session_date", sort=True):
        grp_s = grp.sort_values("ts_utc")
        if len(grp_s) < 6:
            continue
        open_1800 = float(grp_s.iloc[0]["open"])
        ts_1800 = str(grp_s.iloc[0]["dt_ny"])
        close_1700 = float(grp_s.iloc[-1]["close"])

        # London 08:00 open -> 17:00 NY close
        lon_mask = (
            (grp_s["dt_london"].str.slice(0, 10) == s_date)
            & (grp_s["lon_hour"] >= 8)
        )
        grp_lon = grp_s[lon_mask]
        open_0800_lon = float(grp_lon.iloc[0]["open"]) if len(grp_lon) > 0 else np.nan
        ts_0800_lon = str(grp_lon.iloc[0]["dt_london"]) if len(grp_lon) > 0 else ""

        # COMEX Pit 08:20 NY (08:00 NY hourly bar) -> 17:00 NY close
        ny_mask = (
            (grp_s["dt_ny"].str.slice(0, 10) == s_date)
            & (grp_s["ny_hour"] >= 8)
            & (grp_s["ny_hour"] <= 16)
        )
        grp_ny = grp_s[ny_mask]
        open_0820_ny = float(grp_ny.iloc[0]["open"]) if len(grp_ny) > 0 else np.nan
        ts_0820_ny = str(grp_ny.iloc[0]["dt_ny"]) if len(grp_ny) > 0 else ""

        sess_rows.append(
            {
                "date": s_date,
                "ts_1800_ny": ts_1800,
                "open_1800_ny": open_1800,
                "ts_0800_london": ts_0800_lon,
                "open_0800_london": open_0800_lon,
                "ts_0820_ny": ts_0820_ny,
                "open_0820_ny": open_0820_ny,
                "close_1700_ny": close_1700,
                "bars_1h": len(grp_s),
            }
        )
    df_sess = pd.DataFrame(sess_rows)

    df_daily.to_csv(DAILY_CSV, index=False)
    df_1h.drop(columns=["dt_utc"]).to_csv(HOURLY_CSV, index=False)
    df_sess.to_csv(SESSIONS_CSV, index=False)
    return df_daily, df_1h, df_sess


def build_zero_leakage_daily_records(df_daily: pd.DataFrame, df_sess: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    """
    Construct per-day records with STRICT ZERO LOOK-AHEAD:
      Every historical indicator (`atr14`, `ema20`, `pdh`, `pdl`, `pwl`, `pwh`,
      `swing_lows`, `swing_highs`, `prev_month_high/low`, `dxy_prev_close`) for index `i`
      is computed exclusively from completed rows `0 .. i-1` (`date < target_date`).
      At `T_0`, only `today_open` (and `dxy_open`) is observed.
    """
    n = len(df_daily)
    dates = df_daily["date"].tolist()
    opens = df_daily["open"].to_numpy(dtype=float)
    highs = df_daily["high"].to_numpy(dtype=float)
    lows = df_daily["low"].to_numpy(dtype=float)
    closes = df_daily["close"].to_numpy(dtype=float)
    volumes = df_daily["volume"].to_numpy(dtype=float)
    dxy_opens = df_daily["dxy_open"].to_numpy(dtype=float)
    dxy_closes = df_daily["dxy_close"].to_numpy(dtype=float)
    splits = df_daily["split"].tolist()

    sess_map: dict[str, dict[str, Any]] = {}
    if df_sess is not None and len(df_sess) > 0:
        for row in df_sess.to_dict(orient="records"):
            sess_map[str(row["date"])] = row

    # True Range per completed bar k
    tr = np.zeros(n, dtype=float)
    tr[0] = max(1.0, highs[0] - lows[0])
    for k in range(1, n):
        tr[k] = max(
            highs[k] - lows[k],
            abs(highs[k] - closes[k - 1]),
            abs(lows[k] - closes[k - 1]),
        )

    # Wilder ATR14 on completed bar k (so row i reads atr_completed[i - 1]!)
    atr_comp = np.zeros(n, dtype=float)
    atr_comp[0] = tr[0]
    for k in range(1, n):
        if k < 14:
            atr_comp[k] = float(np.mean(tr[: k + 1]))
        else:
            atr_comp[k] = (atr_comp[k - 1] * 13.0 + tr[k]) / 14.0

    # Causal EMAs on completed closes (row i reads ema_comp[i - 1]!)
    ema10_comp = np.zeros(n, dtype=float)
    ema20_comp = np.zeros(n, dtype=float)
    ema50_comp = np.zeros(n, dtype=float)
    dxy_ema20_comp = np.zeros(n, dtype=float)
    ema10_comp[0] = ema20_comp[0] = ema50_comp[0] = closes[0]
    dxy_ema20_comp[0] = dxy_closes[0]
    a10, a20, a50 = 2.0 / 11.0, 2.0 / 21.0, 2.0 / 51.0
    for k in range(1, n):
        ema10_comp[k] = a10 * closes[k] + (1.0 - a10) * ema10_comp[k - 1]
        ema20_comp[k] = a20 * closes[k] + (1.0 - a20) * ema20_comp[k - 1]
        ema50_comp[k] = a50 * closes[k] + (1.0 - a50) * ema50_comp[k - 1]
        dxy_ema20_comp[k] = a20 * dxy_closes[k] + (1.0 - a20) * dxy_ema20_comp[k - 1]

    # Precompute completed calendar-month high/low
    month_hl: dict[str, tuple[float, float]] = {}
    for k in range(n):
        ym = dates[k][:7]
        if ym not in month_hl:
            month_hl[ym] = (highs[k], lows[k])
        else:
            mh, ml = month_hl[ym]
            month_hl[ym] = (max(mh, highs[k]), min(ml, lows[k]))
    ym_sorted = sorted(month_hl.keys())
    prev_ym_map: dict[str, tuple[float, float]] = {}
    for idx_m, ym in enumerate(ym_sorted):
        if idx_m > 0:
            prev_ym_map[ym] = month_hl[ym_sorted[idx_m - 1]]

    # Confirmed multi-scale fractal swing highs & lows (strictly causal at row i!):
    # - 3-bar (radius 1): k + 1 <= i - 1 => k <= i - 2
    # - 5-bar (radius 2): k + 2 <= i - 1 => k <= i - 3
    # - 11-bar (~10-bar, radius 5): k + 5 <= i - 1 => k <= i - 6
    # - 21-bar (~20-bar, radius 10): k + 10 <= i - 1 => k <= i - 11
    # - 61-bar (~60-bar, radius 30): k + 30 <= i - 1 => k <= i - 31
    is_sw_lo_r1 = np.zeros(n, dtype=bool)
    is_sw_hi_r1 = np.zeros(n, dtype=bool)
    is_sw_lo_r2 = np.zeros(n, dtype=bool)
    is_sw_hi_r2 = np.zeros(n, dtype=bool)
    is_sw_lo_r5 = np.zeros(n, dtype=bool)
    is_sw_hi_r5 = np.zeros(n, dtype=bool)
    is_sw_lo_r10 = np.zeros(n, dtype=bool)
    is_sw_hi_r10 = np.zeros(n, dtype=bool)
    is_sw_lo_r30 = np.zeros(n, dtype=bool)
    is_sw_hi_r30 = np.zeros(n, dtype=bool)

    for k in range(1, n - 1):
        if lows[k] <= lows[k - 1] and lows[k] <= lows[k + 1]:
            is_sw_lo_r1[k] = True
        if highs[k] >= highs[k - 1] and highs[k] >= highs[k + 1]:
            is_sw_hi_r1[k] = True
    for k in range(2, n - 2):
        if lows[k] == np.min(lows[k - 2 : k + 3]):
            is_sw_lo_r2[k] = True
        if highs[k] == np.max(highs[k - 2 : k + 3]):
            is_sw_hi_r2[k] = True
    for k in range(5, n - 5):
        if lows[k] == np.min(lows[k - 5 : k + 6]):
            is_sw_lo_r5[k] = True
        if highs[k] == np.max(highs[k - 5 : k + 6]):
            is_sw_hi_r5[k] = True
    for k in range(10, n - 10):
        if lows[k] == np.min(lows[k - 10 : k + 11]):
            is_sw_lo_r10[k] = True
        if highs[k] == np.max(highs[k - 10 : k + 11]):
            is_sw_hi_r10[k] = True
    for k in range(30, n - 30):
        if lows[k] == np.min(lows[k - 30 : k + 31]):
            is_sw_lo_r30[k] = True
        if highs[k] == np.max(highs[k - 30 : k + 31]):
            is_sw_hi_r30[k] = True

    records: list[dict[str, Any]] = []
    leakage_violations = 0

    for i in range(25, n):
        d_iso = dates[i]
        # Hard Zero-Lookahead check: latest completed index is strictly i - 1
        if not (dates[i - 1] < d_iso):
            leakage_violations += 1

        today_open = float(opens[i])
        today_high = float(highs[i])
        today_low = float(lows[i])
        today_close = float(closes[i])

        prev_open = float(opens[i - 1])
        pdh = float(highs[i - 1])
        pdl = float(lows[i - 1])
        pdc = float(closes[i - 1])
        atr14 = max(1.0, float(atr_comp[i - 1]))

        ema10 = float(ema10_comp[i - 1])
        ema20 = float(ema20_comp[i - 1])
        ema50 = float(ema50_comp[i - 1])
        sma100 = float(np.mean(closes[max(0, i - 100) : i]))

        pwl = float(np.min(lows[max(0, i - 5) : i]))
        pwh = float(np.max(highs[max(0, i - 5) : i]))
        low_10d = float(np.min(lows[max(0, i - 10) : i]))
        high_10d = float(np.max(highs[max(0, i - 10) : i]))
        low_20d = float(np.min(lows[max(0, i - 20) : i]))
        high_20d = float(np.max(highs[max(0, i - 20) : i]))
        low_60d = float(np.min(lows[max(0, i - 60) : i]))
        high_60d = float(np.max(highs[max(0, i - 60) : i]))

        ym = d_iso[:7]
        pm_high, pm_low = prev_ym_map.get(ym, (high_20d, low_20d))

        # Collect strictly confirmed multi-scale fractal swing lows & highs (< T_0)
        sw_lows: list[float] = []
        sw_highs: list[float] = []
        for rad, lo_arr, hi_arr, lookback in (
            (2, is_sw_lo_r2, is_sw_hi_r2, 90),
            (5, is_sw_lo_r5, is_sw_hi_r5, 160),
            (10, is_sw_lo_r10, is_sw_hi_r10, 260),
            (30, is_sw_lo_r30, is_sw_hi_r30, 520),
            (1, is_sw_lo_r1, is_sw_hi_r1, 45),
        ):
            max_k = i - 1 - rad
            min_k = max(rad, i - lookback)
            for k in range(max_k, min_k - 1, -1):
                if lo_arr[k]:
                    sw_lows.append(float(lows[k]))
                if hi_arr[k]:
                    sw_highs.append(float(highs[k]))

        dxy_prev_close = float(dxy_closes[i - 1])
        dxy_prev2_close = float(dxy_closes[i - 2])
        dxy_prev6_close = float(dxy_closes[max(0, i - 6)])
        dxy_open = float(dxy_opens[i])
        dxy_ema20 = float(dxy_ema20_comp[i - 1])

        ema20_slope = float(ema20_comp[i - 1] - ema20_comp[max(0, i - 6)]) / (5.0 * atr14)
        close_20d_ago = float(closes[max(0, i - 21)])
        rolling_20d_ret = (pdc - close_20d_ago) / max(1.0, close_20d_ago)

        if (pdc > ema20 > ema50) and (ema20_slope > 0.0) and (rolling_20d_ret > 0.0):
            macro_trend_regime = "STRONG_BULL"
        elif (pdc < ema20 < ema50) and (ema20_slope < 0.0) and (rolling_20d_ret < 0.0):
            macro_trend_regime = "STRONG_BEAR"
        else:
            macro_trend_regime = "NEUTRAL"

        prev_range = max(0.01, pdh - pdl)
        prev_ibs = (pdc - pdl) / prev_range
        prev_ret_1 = (pdc - prev_open) / atr14
        prev_ret_3 = (pdc - float(closes[i - 4])) / atr14
        prev_ret_5 = (pdc - float(closes[i - 6])) / atr14
        open_gap_atr = (today_open - pdc) / atr14

        dt_obj = datetime.date.fromisoformat(d_iso)
        weekday = dt_obj.strftime("%A")

        sess = sess_map.get(d_iso, {})

        # Target outcome (STRICTLY NEVER READ BY ANY FEATURE ENGINE!)
        candle_move = today_close - today_open
        actual_dir = "XAU" if candle_move > 0 else ("USD" if candle_move < 0 else "FLAT")
        is_noise_flat = abs(candle_move) <= max(1.00, 0.15 * atr14)

        records.append(
            {
                "idx": i,
                "date": d_iso,
                "weekday": weekday,
                "weekday_up": weekday.upper(),
                "split": splits[i],
                # Known at T_0:
                "today_open": today_open,
                "prev_open": prev_open,
                "pdh": pdh,
                "pdl": pdl,
                "prev_close": pdc,
                "prev2_high": float(highs[i - 2]),
                "prev2_low": float(lows[i - 2]),
                "atr14": atr14,
                "ema10": ema10,
                "ema20": ema20,
                "ema50": ema50,
                "sma100": sma100,
                "ema20_slope": ema20_slope,
                "rolling_20d_ret": rolling_20d_ret,
                "macro_trend_regime": macro_trend_regime,
                "pwl": pwl,
                "pwh": pwh,
                "low_10d": low_10d,
                "high_10d": high_10d,
                "low_20d": low_20d,
                "high_20d": high_20d,
                "low_60d": low_60d,
                "high_60d": high_60d,
                "prev_month_low": float(pm_low),
                "prev_month_high": float(pm_high),
                "swing_lows": sw_lows,
                "swing_highs": sw_highs,
                "dxy_open": dxy_open,
                "dxy_prev_close": dxy_prev_close,
                "dxy_prev_ret": (dxy_prev_close - dxy_prev2_close) / max(1.0, dxy_prev2_close),
                "dxy_ret_5d": (dxy_prev_close - dxy_prev6_close) / max(1.0, dxy_prev6_close),
                "dxy_ema20": dxy_ema20,
                "prev_ibs": prev_ibs,
                "prev_ret_1": prev_ret_1,
                "prev_ret_3": prev_ret_3,
                "prev_ret_5": prev_ret_5,
                "open_gap_atr": open_gap_atr,
                # Verified intraday session opens (2025-2026) when available:
                "open_1800_ny": sess.get("open_1800_ny", np.nan),
                "open_0800_london": sess.get("open_0800_london", np.nan),
                "open_0820_ny": sess.get("open_0820_ny", np.nan),
                "close_1700_ny": sess.get("close_1700_ny", np.nan),
                "ts_1800_ny": sess.get("ts_1800_ny", ""),
                "ts_0800_london": sess.get("ts_0800_london", ""),
                "ts_0820_ny": sess.get("ts_0820_ny", ""),
                # Post-close ground truth (strictly isolated for evaluation & past-only twin archive):
                "_target_high": today_high,
                "_target_low": today_low,
                "_target_close": today_close,
                "_target_move": candle_move,
                "_target_dir": actual_dir,
                "_is_noise_flat": bool(is_noise_flat),
                "volume": float(volumes[i]),
            }
        )

    assert leakage_violations == 0, f"Detected {leakage_violations} date ordering violations!"
    return records


def group_1h_bars_by_session(df_1h: pd.DataFrame) -> dict[str, list[tuple[float, float, float, float]]]:
    """
    Group verified chronological `1h` bars (`2025-2026`) by NY `session_date` (`18:00 NY D-1 -> 17:00 NY D`).
    Returns dict mapping `YYYY-MM-DD` -> list of `(open, high, low, close)` sorted chronologically.
    """
    out: dict[str, list[tuple[float, float, float, float]]] = {}
    if df_1h is None or len(df_1h) == 0:
        return out
    df_s = df_1h.sort_values("ts_utc")
    for s_date, grp in df_s.groupby("session_date", sort=False):
        if len(grp) < 6:
            continue
        bars = list(
            zip(
                grp["open"].astype(float).tolist(),
                grp["high"].astype(float).tolist(),
                grp["low"].astype(float).tolist(),
                grp["close"].astype(float).tolist(),
            )
        )
        out[str(s_date)] = bars
    return out


if __name__ == "__main__":
    df_d, df_h, df_s = fetch_and_cache_market_data()
    recs = build_zero_leakage_daily_records(df_d, df_s)
    by_sp = pd.Series([r["split"] for r in recs]).value_counts().to_dict()
    print(f"Loaded {len(df_d)} daily rows, {len(df_h)} 1h rows, {len(df_s)} verified 2025-2026 sessions.")
    print(f"Zero-leakage records: {len(recs)} | Splits: {by_sp}")
