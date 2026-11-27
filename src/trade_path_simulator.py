#!/usr/bin/env python3
"""
Zero-Leakage Intraday `1h` & Conservative Daily `OHLC` Trade Path Simulator
(`src/trade_path_simulator.py` — `v4.2.0` Symmetric `SL` & 2-Tranche Scale-Out).

Evaluates TWO side-by-side metrics with ZERO optimistic bias:
1. **Raw Candle Close Win %**:
   - `LONG XAU`: `Close > Open` (`pnl_close = Close - Open`)
   - `SHORT XAU`: `Close < Open` (`pnl_close = Open - Close`)
2. **Executable `TP1-Before-SL` + 2-Tranche Scale-Out (`50% TP1 + BE Runner -> TP2/TP3`)**:
   - Adaptive causal targets fixed strictly at `T_0`:
       * `TP1_dist = max(0.35 * ATR14, 0.65 * twin_p50_move)`
       * `TP2_dist = max(TP1_dist + 0.12 * ATR14, twin_p75_move)`
       * `TP3_dist = max(TP2_dist + 0.18 * ATR14, twin_p90_move)`
       * `SL_dist  = min(0.72 * ATR14, max(0.45 * ATR14, structural_S1_R1_invalidation))`
         so `TP1 / SL` R:R is balanced (`~0.75R–1.05R`).
   - Institutional 2-Tranche Scale-Out:
       * When `TP1` is hit cleanly first: bank `50%` at `+TP1`, move `SL` on remaining `50%`
         strictly to `Entry (Breakeven $0.00)`, and scale `25%` at `TP2` + `25%` at `TP3`
         (or exit remaining runner at `Breakeven $0.00` if stopped, or `17:00 NY` Close).
   - Strict Conservative Intra-Bar Ordering:
       * If BOTH `TP1` and `SL` are touched within the exact same candle (`1h` or `Daily`),
         it is ALWAYS counted as a full `-100% SL` loss (`AMBIGUOUS_*_SL_LOSS`).
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
    s1_level: float | None = None,
    r1_level: float | None = None,
) -> dict[str, float]:
    """
    Derive symmetric adaptive causal `Entry / SL / TP1 / TP2 / TP3` strictly at `T_0` (zero look-ahead):
      - `TP1 = max(0.35 * ATR14, 0.65 * twin_p50_move)`
      - `TP2 = max(TP1 + 0.12 * ATR14, twin_p75_move)`
      - `TP3 = max(TP2 + 0.18 * ATR14, twin_p90_move)`
      - `SL  = min(0.72 * ATR14, max(0.45 * ATR14, structural_S1_R1_invalidation))`
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
        sup_anchor = float(s1_level) if s1_level is not None else float(s2_level)
        raw_inv = max(0.45 * atr, (entry_open - sup_anchor) + 0.08 * atr)
        sl_dist = min(0.72 * atr, raw_inv)
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
        res_anchor = float(r1_level) if r1_level is not None else float(r2_level)
        raw_inv = max(0.45 * atr, (res_anchor - entry_open) + 0.08 * atr)
        sl_dist = min(0.72 * atr, raw_inv)
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
    Simulate 2-Tranche Scale-Out (`50% TP1 + Breakeven Runner -> 25% TP2 + 25% TP3`) with ZERO optimistic bias:
      - Uses chronological `1h` bars when available & contract-aligned (`|open_1h - entry| <= 0.25 * ATR14`).
      - Otherwise uses conservative worst-case daily `OHLC`.
      - Same-bar `TP1` + `SL` touch ALWAYS resolves as a full `-100% SL` loss.
    """
    is_long = direction == "XAU"
    raw_close_pnl = (daily_close - entry) if is_long else (entry - daily_close)
    raw_close_win = raw_close_pnl > 0.0

    tp1_gain = abs(tp1 - entry)
    tp2_gain = abs(tp2 - entry)
    tp3_val = float(tp3) if tp3 is not None else (tp2 + 0.25 * atr14 if is_long else tp2 - 0.25 * atr14)
    tp3_gain = abs(tp3_val - entry)
    sl_risk = abs(entry - sl)

    use_1h = False
    basis_shift = 0.0
    if bars_1h and len(bars_1h) >= 6:
        first_open = float(bars_1h[0][0])
        if abs(first_open - entry) <= 0.25 * max(1.0, atr14):
            use_1h = True
            basis_shift = entry - first_open

    if use_1h and bars_1h is not None:
        outcome = "TIME_CLOSE"
        hit_tp1 = False
        hit_tp2 = False
        hit_tp3 = False
        hit_sl = False
        rem_qty = 1.0
        realized_pnl = 0.0

        for _bo, bh_raw, bl_raw, _bc in bars_1h:
            bh = float(bh_raw) + basis_shift
            bl = float(bl_raw) + basis_shift
            if is_long:
                b_tp1 = bh >= tp1
                b_tp2 = bh >= tp2
                b_tp3 = bh >= tp3_val
                b_sl = bl <= sl
                b_be = bl <= entry
            else:
                b_tp1 = bl <= tp1
                b_tp2 = bl <= tp2
                b_tp3 = bl <= tp3_val
                b_sl = bh >= sl
                b_be = bh >= entry

            if not hit_tp1 and not hit_sl:
                # STRICT CONSERVATIVE INTRA-BAR RULE: simultaneous TP1 & SL in same 1h bar -> full SL_LOSS
                if b_tp1 and b_sl:
                    hit_sl = True
                    rem_qty = 0.0
                    realized_pnl = -sl_risk
                    outcome = "AMBIGUOUS_1H_SL_LOSS"
                    break
                elif b_sl:
                    hit_sl = True
                    rem_qty = 0.0
                    realized_pnl = -sl_risk
                    outcome = "SL_LOSS_1H"
                    break
                elif b_tp1:
                    hit_tp1 = True
                    realized_pnl += 0.50 * tp1_gain
                    rem_qty = 0.50
                    outcome = "TP1_WIN_1H"
                    # Only award same-bar TP2/TP3 on the breakout candle if price stayed strictly clear of Entry (BE)
                    if not b_be:
                        if b_tp2:
                            hit_tp2 = True
                            realized_pnl += 0.25 * tp2_gain
                            rem_qty = 0.25
                            outcome = "TP2_WIN_1H"
                        if b_tp3:
                            hit_tp3 = True
                            realized_pnl += 0.25 * tp3_gain
                            rem_qty = 0.0
                            outcome = "TP3_WIN_1H"
                            break
            elif hit_tp1 and rem_qty > 0.0:
                # Runner phase: stop is strictly at Entry (Breakeven $0.00).
                # Conservative intra-bar rule: if BE is touched in the same 1h bar as TP2/TP3, runner stops at BE ($0.00)!
                if b_be:
                    rem_qty = 0.0
                    outcome = "TP2_BE_1H" if hit_tp2 else "TP1_BE_1H"
                    break
                if (not hit_tp2) and b_tp2:
                    hit_tp2 = True
                    realized_pnl += 0.25 * tp2_gain
                    rem_qty = 0.25
                    outcome = "TP2_WIN_1H"
                if (not hit_tp3) and b_tp3:
                    hit_tp3 = True
                    realized_pnl += 0.25 * tp3_gain
                    rem_qty = 0.0
                    outcome = "TP3_WIN_1H"
                    break

        if not hit_tp1 and not hit_sl:
            exec_pnl = raw_close_pnl
        else:
            if rem_qty > 0.0:
                realized_pnl += rem_qty * max(0.0, raw_close_pnl)
            exec_pnl = realized_pnl

        return {
            "path_source": "1H_CHRONOLOGICAL",
            "raw_close_win": bool(raw_close_win),
            "raw_close_pnl": round(raw_close_pnl, 2),
            "exec_win": bool(exec_pnl > 0.0),
            "exec_pnl": round(exec_pnl, 2),
            "hit_tp1_clean": bool(hit_tp1),
            "hit_tp2_clean": bool(hit_tp2),
            "hit_tp3_clean": bool(hit_tp3),
            "outcome": outcome,
        }

    # Conservative Worst-Case Daily OHLC Simulator (`2012-2024`)
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
        # STRICT CONSERVATIVE RULE: both touched on same daily bar -> full SL_LOSS!
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
        hit_tp1 = True
        if raw_close_pnl <= 0.0:
            # Price reversed across Entry after hitting TP1 -> runner stopped out at Breakeven ($0.00)
            hit_tp2 = False
            hit_tp3 = False
            exec_pnl = 0.50 * tp1_gain
            outcome = "TP1_BE_DAILY"
        elif d_tp3 and raw_close_pnl >= tp1_gain:
            hit_tp2 = True
            hit_tp3 = True
            exec_pnl = 0.50 * tp1_gain + 0.25 * tp2_gain + 0.25 * tp3_gain
            outcome = "TP3_WIN_DAILY"
        elif d_tp2 and raw_close_pnl >= 0.50 * tp1_gain:
            hit_tp2 = True
            hit_tp3 = False
            exec_pnl = 0.50 * tp1_gain + 0.25 * tp2_gain + 0.25 * raw_close_pnl
            outcome = "TP2_WIN_DAILY"
        else:
            hit_tp2 = False
            hit_tp3 = False
            exec_pnl = 0.50 * tp1_gain + 0.50 * raw_close_pnl
            outcome = "TP1_WIN_DAILY"
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
    """Aggregate both Raw Candle Close metrics and Executable `TP1-Before-SL` 2-Tranche metrics."""
    sub = [r for r in rows if mask_fn(r) and r["actual_dir"] in ("XAU", "USD")]
    n = len(sub)
    if n == 0:
        return {
            "n": 0,
            "lock3_rate": 0.0,
            "flip_count": 0,
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
    flip_cnt = sum(1 for r in sub if r.get("polarity_flipped"))

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
        "flip_count": flip_cnt,
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
