#!/usr/bin/env python3
"""
Zero-Leakage Intraday `1h` & Conservative Daily `OHLC` Trade Path Simulator
(`src/trade_path_simulator.py`).

Evaluates TWO side-by-side metrics with ZERO optimistic bias:
1. **Raw Candle Close Win %**:
   - `LONG XAU`: `Close > Open` (`pnl_close = Close - Open`)
   - `SHORT XAU`: `Close < Open` (`pnl_close = Open - Close`)
2. **Executable `TP1-Before-SL` Win %**:
   - Adaptive causal targets fixed strictly at `T_0`:
       * `TP1_dist = max(0.35 * ATR14, 0.65 * twin_p50_move)`
       * `TP2_dist = max(TP1_dist + 0.12 * ATR14, twin_p75_move)`
       * `TP3_dist = max(TP2_dist + 0.18 * ATR14, twin_p90_move)`
       * `SL_dist  = min(1.10 * ATR14, max(0.45 * ATR14, dist_to_S2_or_R2_invalidation))`
   - On `2025–2026` sessions with verified chronological `1h` bars (basis-aligned to `Today_Open`):
       Steps chronologically bar-by-bar (`18:00 NY D-1 -> 17:00 NY D`).
       If BOTH `TP1` and `SL` are touched within the same `1h` bar -> **CONSERVATIVE `SL` LOSS**.
   - On `2012–2024` daily `OHLC` bars (or sessions without `1h` bars):
       If BOTH `TP1` and `SL` are touched within the daily bar -> **CONSERVATIVE `SL` LOSS**!
       If neither `TP1` nor `SL` is touched intraday -> exits at `17:00 NY` Close (`Close - Open` signed).
"""
from __future__ import annotations

from typing import Any

import numpy as np


def compute_adaptive_causal_levels(
    direction: str,
    entry_open: float,
    atr14: float,
    s2_level: float,
    r2_level: float,
    twin_p50_atr: float,
    twin_p75_atr: float,
    twin_p90_atr: float,
) -> dict[str, float]:
    """
    Derive adaptive causal `Entry / SL / TP1 / TP2 / TP3` strictly at `T_0` (zero look-ahead):
      - `TP1 = max(0.35 * ATR14, 0.65 * twin_p50_move)`
      - `TP2 = twin_p75_move` (floored above `TP1`)
      - `TP3 = twin_p90_move` (floored above `TP2`)
      - `SL  = min(1.10 * ATR14, distance_to_S2_or_R2_invalidation)`
    """
    atr = max(1.0, float(atr14))
    p50_move = float(twin_p50_atr) * atr
    p75_move = float(twin_p75_atr) * atr
    p90_move = float(twin_p90_atr) * atr

    tp1_dist = max(0.35 * atr, 0.65 * p50_move)
    tp2_dist = max(tp1_dist + 0.12 * atr, p75_move)
    tp3_dist = max(tp2_dist + 0.18 * atr, p90_move)

    is_long = direction == "XAU"
    if is_long:
        raw_inv = max(0.45 * atr, (entry_open - float(s2_level)) + 0.08 * atr)
        sl_dist = min(1.10 * atr, raw_inv)
        return {
            "entry": round(entry_open, 2),
            "sl": round(entry_open - sl_dist, 2),
            "tp1": round(entry_open + tp1_dist, 2),
            "tp2": round(entry_open + tp2_dist, 2),
            "tp3": round(entry_open + tp3_dist, 2),
            "sl_dist": round(sl_dist, 2),
            "tp1_dist": round(tp1_dist, 2),
        }
    else:
        raw_inv = max(0.45 * atr, (float(r2_level) - entry_open) + 0.08 * atr)
        sl_dist = min(1.10 * atr, raw_inv)
        return {
            "entry": round(entry_open, 2),
            "sl": round(entry_open + sl_dist, 2),
            "tp1": round(entry_open - tp1_dist, 2),
            "tp2": round(entry_open - tp2_dist, 2),
            "tp3": round(entry_open - tp3_dist, 2),
            "sl_dist": round(sl_dist, 2),
            "tp1_dist": round(tp1_dist, 2),
        }


def simulate_session_execution(
    direction: str,
    entry: float,
    sl: float,
    tp1: float,
    tp2: float,
    atr14: float,
    daily_high: float,
    daily_low: float,
    daily_close: float,
    bars_1h: list[tuple[float, float, float, float]] | None = None,
    tp3: float | None = None,
) -> dict[str, Any]:
    """
    Simulate trade path with ZERO optimistic bias:
      - Uses chronological `1h` bars when available & contract-aligned (`|open_1h - entry| <= 0.25 * ATR14`).
      - Otherwise uses conservative worst-case daily `OHLC` (both `TP1` and `SL` touched -> **`SL` LOSS**).
    """
    is_long = direction == "XAU"
    raw_close_pnl = (daily_close - entry) if is_long else (entry - daily_close)
    raw_close_win = raw_close_pnl > 0.0

    tp1_gain = abs(tp1 - entry)
    sl_risk = abs(entry - sl)
    tp3_val = float(tp3) if tp3 is not None else (tp2 + 0.25 * atr14 if is_long else tp2 - 0.25 * atr14)

    # Check if chronological 1h bars are available and on the same active contract basis
    use_1h = False
    basis_shift = 0.0
    if bars_1h and len(bars_1h) >= 6:
        first_open = float(bars_1h[0][0])
        if abs(first_open - entry) <= 0.25 * max(1.0, atr14):
            use_1h = True
            basis_shift = entry - first_open

    if use_1h and bars_1h is not None:
        outcome = "TIME_CLOSE"
        exec_pnl = raw_close_pnl
        hit_tp1 = False
        hit_tp2 = False
        hit_tp3 = False
        hit_sl = False

        for _bo, bh_raw, bl_raw, _bc in bars_1h:
            bh = float(bh_raw) + basis_shift
            bl = float(bl_raw) + basis_shift
            if is_long:
                b_tp1 = bh >= tp1
                b_sl = bl <= sl
                b_tp2 = bh >= tp2
                b_tp3 = bh >= tp3_val
            else:
                b_tp1 = bl <= tp1
                b_sl = bh >= sl
                b_tp2 = bl <= tp2
                b_tp3 = bl <= tp3_val

            if not hit_tp1 and not hit_sl:
                # CONSERVATIVE INTRA-BAR RULE: simultaneous touch in same 1h candle -> SL_LOSS
                if b_tp1 and b_sl:
                    hit_sl = True
                    outcome = "AMBIGUOUS_1H_SL_LOSS"
                    exec_pnl = -sl_risk
                    break
                elif b_sl:
                    hit_sl = True
                    outcome = "SL_LOSS_1H"
                    exec_pnl = -sl_risk
                    break
                elif b_tp1:
                    hit_tp1 = True
                    outcome = "TP1_WIN_1H"
                    exec_pnl = +tp1_gain
                    if b_tp2:
                        hit_tp2 = True
                    if b_tp3:
                        hit_tp3 = True
            elif hit_tp1 and not hit_sl:
                if b_sl:
                    break
                if b_tp2:
                    hit_tp2 = True
                if b_tp3:
                    hit_tp3 = True

        exec_win = exec_pnl > 0.0
        return {
            "path_source": "1H_CHRONOLOGICAL",
            "raw_close_win": bool(raw_close_win),
            "raw_close_pnl": round(raw_close_pnl, 2),
            "exec_win": bool(exec_win),
            "exec_pnl": round(exec_pnl, 2),
            "hit_tp1_clean": bool(hit_tp1),
            "hit_tp2_clean": bool(hit_tp2),
            "hit_tp3_clean": bool(hit_tp3),
            "outcome": outcome,
        }

    # Conservative Worst-Case Daily OHLC Simulator
    if is_long:
        d_tp1 = daily_high >= tp1
        d_tp2 = daily_high >= tp2
        d_tp3 = daily_high >= tp3_val
        d_sl = daily_low <= sl
    else:
        d_tp1 = daily_low <= tp1
        d_tp2 = daily_low <= tp2
        d_tp3 = daily_low <= tp3_val
        d_sl = daily_high >= sl

    if d_tp1 and d_sl:
        # STRICT CONSERVATIVE RULE: both touched on same daily bar -> count as SL_LOSS!
        outcome = "AMBIGUOUS_DAILY_SL_LOSS"
        exec_pnl = -sl_risk
        hit_tp1 = False
        hit_tp2 = False
        hit_tp3 = False
    elif d_sl:
        outcome = "SL_LOSS_DAILY"
        exec_pnl = -sl_risk
        hit_tp1 = False
        hit_tp2 = False
        hit_tp3 = False
    elif d_tp1:
        outcome = "TP1_WIN_DAILY"
        exec_pnl = +tp1_gain
        hit_tp1 = True
        hit_tp2 = bool(d_tp2)
        hit_tp3 = bool(d_tp3)
    else:
        outcome = "TIME_CLOSE_DAILY"
        exec_pnl = raw_close_pnl
        hit_tp1 = False
        hit_tp2 = False
        hit_tp3 = False

    return {
        "path_source": "DAILY_CONSERVATIVE",
        "raw_close_win": bool(raw_close_win),
        "raw_close_pnl": round(raw_close_pnl, 2),
        "exec_win": bool(exec_pnl > 0.0),
        "exec_pnl": round(exec_pnl, 2),
        "hit_tp1_clean": bool(hit_tp1),
        "hit_tp2_clean": bool(hit_tp2),
        "hit_tp3_clean": bool(hit_tp3),
        "outcome": outcome,
    }


def summarize_dual_metrics(rows: list[dict[str, Any]], mask_fn) -> dict[str, Any]:
    """Aggregate both Raw Candle Close metrics and Executable `TP1-Before-SL` metrics."""
    sub = [r for r in rows if mask_fn(r) and r["actual_dir"] in ("XAU", "USD")]
    n = len(sub)
    if n == 0:
        return {
            "n": 0,
            "lock3_rate": 0.0,
            "raw_wins": 0,
            "raw_losses": 0,
            "raw_win_rate": 0.0,
            "raw_pf": 0.0,
            "raw_exp_usd": 0.0,
            "exec_wins": 0,
            "exec_losses": 0,
            "exec_win_rate": 0.0,
            "exec_pf": 0.0,
            "exec_exp_usd": 0.0,
            "tp1_clean_rate": 0.0,
            "tp2_clean_rate": 0.0,
            "tp3_clean_rate": 0.0,
        }

    lock3_cnt = sum(1 for r in sub if r.get("has_lock3"))

    raw_pnls = [float(r["raw_close_pnl"]) for r in sub]
    raw_wins = sum(1 for p in raw_pnls if p > 0.0)
    raw_losses = n - raw_wins
    raw_gw = sum(p for p in raw_pnls if p > 0.0)
    raw_gl = abs(sum(p for p in raw_pnls if p < 0.0))
    raw_pf = (raw_gw / raw_gl) if raw_gl > 1e-6 else (99.99 if raw_gw > 0 else 0.0)

    exec_pnls = [float(r["exec_pnl"]) for r in sub]
    exec_wins = sum(1 for r in sub if r["exec_win"])
    exec_losses = n - exec_wins
    exec_gw = sum(p for p in exec_pnls if p > 0.0)
    exec_gl = abs(sum(p for p in exec_pnls if p < 0.0))
    exec_pf = (exec_gw / exec_gl) if exec_gl > 1e-6 else (99.99 if exec_gw > 0 else 0.0)

    tp1_c = sum(1 for r in sub if r.get("hit_tp1_clean"))
    tp2_c = sum(1 for r in sub if r.get("hit_tp2_clean"))
    tp3_c = sum(1 for r in sub if r.get("hit_tp3_clean"))

    return {
        "n": n,
        "lock3_rate": round(lock3_cnt * 100.0 / n, 1),
        "raw_wins": raw_wins,
        "raw_losses": raw_losses,
        "raw_win_rate": round(raw_wins * 100.0 / n, 2),
        "raw_pf": round(raw_pf, 2),
        "raw_exp_usd": round(float(np.mean(raw_pnls)), 2),
        "exec_wins": exec_wins,
        "exec_losses": exec_losses,
        "exec_win_rate": round(exec_wins * 100.0 / n, 2),
        "exec_pf": round(exec_pf, 2),
        "exec_exp_usd": round(float(np.mean(exec_pnls)), 2),
        "tp1_clean_rate": round(tp1_c * 100.0 / n, 1),
        "tp2_clean_rate": round(tp2_c * 100.0 / n, 1),
        "tp3_clean_rate": round(tp3_c * 100.0 / n, 1),
    }
