#!/usr/bin/env python3
"""
Zero-Leakage Intraday `1h` & Conservative Daily `OHLC` Trade Path Simulator
(`src/trade_path_simulator.py` — `v5.1.0` Runner `PnL` & `TP1_D` Fix + Dual Win Metrics).

Evaluates TWO primary transparent win metrics side-by-side (plus Raw Close Win%):
1. **`برد روزانه بدون استاپ تنگ` (`Daily Target / Direction Win %`, `slEnabled=False`)**:
   - Session hits `TP1 = Q25` (`[0.25 * ATR14, 0.45 * ATR14]`) intraday OR closes in the predicted
     direction (`Close > Open` for `LONG XAU`, `Close < Open` for `SHORT XAU`) — matching the
     `KXGOLDPERP` / `explore.tsx` Gold Twin benchmark!
2. **`برد اجرایی با استاپ پله‌ای` (`Stepped SL = 0.85 ATR + True TP1/TP2/TP3/EOD Runner`)**:
   - Exact `v5.1.0` Stepped Runner Rules:
       1. Before `TP1` is touched: if `SL` is touched first (or both `SL` and `TP1` are inside the
          exact same entry bar), full `100%` exits at `-SL_dist` (strict conservative rule).
       2. Once `TP1` is touched: lock `+0.50 * TP1_dist` immediately and move stop on remaining `50%`
          to `Entry ($0.00)`.
       3. On that bar (if high/low clearly extends past `TP2`/`TP3` without violating `Entry` after `TP1`)
          and all subsequent chronological bars:
          - If `TP2` is reached before returning to `Entry`, lock `+0.25 * TP2_dist` (and move remaining
            `25%` stop to `TP1`).
          - If `TP3` is reached, lock `+0.25 * max(TP3_dist, EOD_Close_dist)`.
          - Only zero out the remaining runner tranche at `$0.00` if a **subsequent** bar after the `TP1`
            bar retraces back through `Entry` before `TP2` is reached; otherwise settle any open runner
            tranche at `EOD Close`!
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
    twin_p50_atr: float = 0.30,
    twin_p75_atr: float = 0.58,
    twin_p90_atr: float = 0.88,
    s1_level: float | None = None,
    r1_level: float | None = None,
    *,
    twin_q25_atr: float | None = None,
    twin_q50_atr: float | None = None,
    twin_q75_atr: float | None = None,
) -> dict[str, float]:
    """
    Derive `KXGOLDPERP`-standard causal `Entry / SL / TP1(Q25) / TP2(Q50) / TP3(Q75)` strictly at `T_0`:
      - `TP1 = clip(Q25_atr, 0.25, 0.45) * ATR14`
      - `TP2 = max(TP1 + 0.32 * ATR14, clip(Q50_atr * 1.45, 0.68, 1.45) * ATR14)`
      - `TP3 = max(TP2 + 0.42 * ATR14, clip(Q75_atr * 1.60, 1.15, 2.05) * ATR14)`
      - `SL  = min(0.85 * ATR14, max(0.55 * ATR14, structural_S1_R1_invalidation))`
    """
    atr = max(1.0, float(atr14))
    q25_in = float(twin_q25_atr if twin_q25_atr is not None else twin_p50_atr)
    q50_in = float(twin_q50_atr if twin_q50_atr is not None else twin_p75_atr)
    q75_in = float(twin_q75_atr if twin_q75_atr is not None else twin_p90_atr)

    q25_clamped = float(np.clip(q25_in, 0.25, 0.45))
    tp1_dist = q25_clamped * atr
    tp2_dist = max(tp1_dist + 0.24 * atr, float(np.clip(q50_in * 1.30, 0.58, 1.20)) * atr)
    tp3_dist = max(tp2_dist + 0.34 * atr, float(np.clip(q75_in * 1.40, 0.95, 1.80)) * atr)

    is_long = direction == "XAU"
    if is_long:
        sup_anchor = float(s1_level) if s1_level is not None else float(s2_level)
        raw_inv = max(0.55 * atr, (entry_open - sup_anchor) + 0.04 * atr)
        sl_dist = min(0.85 * atr, max(0.55 * atr, raw_inv))
        return {
            "entry": round(entry_open, 2),
            "sl": round(entry_open - sl_dist, 2),
            "tp1": round(entry_open + tp1_dist, 2),
            "tp2": round(entry_open + tp2_dist, 2),
            "tp3": round(entry_open + tp3_dist, 2),
            "sl_dist": round(sl_dist, 2),
            "tp1_dist": round(tp1_dist, 2),
            "tp2_dist": round(tp2_dist, 2),
            "tp3_dist": round(tp3_dist, 2),
        }
    else:
        res_anchor = float(r1_level) if r1_level is not None else float(r2_level)
        raw_inv = max(0.55 * atr, (res_anchor - entry_open) + 0.04 * atr)
        sl_dist = min(0.85 * atr, max(0.55 * atr, raw_inv))
        return {
            "entry": round(entry_open, 2),
            "sl": round(entry_open + sl_dist, 2),
            "tp1": round(entry_open - tp1_dist, 2),
            "tp2": round(entry_open - tp2_dist, 2),
            "tp3": round(entry_open - tp3_dist, 2),
            "sl_dist": round(sl_dist, 2),
            "tp1_dist": round(tp1_dist, 2),
            "tp2_dist": round(tp2_dist, 2),
            "tp3_dist": round(tp3_dist, 2),
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
    Simulate BOTH (`v5.1.0` — Zero-Premature-BE & Unshifted `1h` Fix):
      1. `daily_target_win` / `daily_target_pnl` (`slEnabled=False` `KXGOLDPERP` standard:
         hits `TP1` intraday OR closes in predicted direction), AND
      2. `exec_win` / `exec_pnl` (`Stepped SL = 0.85 ATR + True TP1/TP2/TP3/EOD Runner`).
    """
    is_long = direction == "XAU"
    raw_close_pnl = (daily_close - entry) if is_long else (entry - daily_close)
    raw_close_win = raw_close_pnl > 0.0

    tp1_gain = abs(tp1 - entry)
    tp2_gain = abs(tp2 - entry)
    tp3_val = float(tp3) if tp3 is not None else (tp2 + 0.25 * atr14 if is_long else tp2 - 0.25 * atr14)
    tp3_gain = abs(tp3_val - entry)
    sl_risk = abs(entry - sl)
    atr_eff = max(1.0, float(atr14))

    use_1h = False
    sess_max_high = float(daily_high)
    sess_min_low = float(daily_low)

    if bars_1h and len(bars_1h) >= 6:
        first_open = float(bars_1h[0][0])
        # `xauusd_1h_2024_2026.csv` is already on the exact same unshifted price scale as `daily_high/low`
        # (`median |max(1h H) - daily H| == 0.0`); never apply a corrupt `entry - first_open` shift!
        if abs(first_open - entry) <= 0.55 * atr_eff:
            use_1h = True
            sess_max_high = max(sess_max_high, max(float(b[1]) for b in bars_1h))
            sess_min_low = min(sess_min_low, min(float(b[2]) for b in bars_1h))

    # Metric 1: Daily Target / Direction Win (`slEnabled=False` — `KXGOLDPERP بدون SL تنگ`)
    hit_tp1_any = (sess_max_high >= tp1) if is_long else (sess_min_low <= tp1)
    daily_target_win = bool(hit_tp1_any or raw_close_win)
    if hit_tp1_any:
        if raw_close_pnl > tp1_gain:
            daily_target_pnl = 0.50 * tp1_gain + 0.50 * max(tp3_gain if raw_close_pnl >= tp3_gain else 0.0, raw_close_pnl)
        else:
            daily_target_pnl = tp1_gain
    else:
        daily_target_pnl = raw_close_pnl

    tp3_runner_pay = max(tp3_gain, raw_close_pnl)

    # Metric 2: Chronological `1h` Stepped SL (`2025-2026`)
    if use_1h and bars_1h is not None:
        outcome = "TIME_CLOSE"
        hit_tp1 = False
        hit_tp2 = False
        hit_tp3 = False
        hit_sl = False
        rem_qty = 1.0
        realized_pnl = 0.0
        be_tol = 0.10 * atr_eff

        for _bo, bh_raw, bl_raw, bc_raw in bars_1h:
            bh = float(bh_raw)
            bl = float(bl_raw)
            bc = float(bc_raw)
            if is_long:
                b_tp1 = bh >= tp1
                b_tp2 = bh >= tp2
                b_tp3 = bh >= tp3_val
                b_sl = bl <= sl
                b_be_samebar = (bc <= entry) and (bl < entry - 0.05 * atr_eff)
                b_be_subseq = (bc <= entry) or (bl <= entry - be_tol)
                b_step1 = (bc <= tp1) or (bl <= tp1 - be_tol)
            else:
                b_tp1 = bl <= tp1
                b_tp2 = bl <= tp2
                b_tp3 = bl <= tp3_val
                b_sl = bh >= sl
                b_be_samebar = (bc >= entry) and (bh > entry + 0.05 * atr_eff)
                b_be_subseq = (bc >= entry) or (bh >= entry + be_tol)
                b_step1 = (bc >= tp1) or (bh >= tp1 + be_tol)

            if not hit_tp1 and not hit_sl:
                # Rule 1: Before TP1 is touched, if SL is touched first OR both SL & TP1 in same bar -> full -SL_dist
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
                    # Rule 2: Lock +0.50 * TP1_dist immediately & step runner stop to Entry ($0.00)
                    hit_tp1 = True
                    realized_pnl += 0.50 * tp1_gain
                    rem_qty = 0.50
                    outcome = "TP1_WIN_1H"
                    if not b_be_samebar:
                        if b_tp2:
                            hit_tp2 = True
                            realized_pnl += 0.25 * tp2_gain
                            rem_qty = 0.25
                            outcome = "TP2_WIN_1H"
                        if b_tp3:
                            hit_tp3 = True
                            realized_pnl += 0.25 * tp3_runner_pay
                            rem_qty = 0.0
                            outcome = "TP3_WIN_1H"
                            break
            elif hit_tp1 and rem_qty > 0.0:
                # Rule 3: Subsequent chronological bars after TP1 bar
                if hit_tp2:
                    if (not hit_tp3) and b_tp3 and (not b_step1):
                        hit_tp3 = True
                        realized_pnl += 0.25 * tp3_runner_pay
                        rem_qty = 0.0
                        outcome = "TP3_WIN_1H"
                        break
                    if b_step1:
                        realized_pnl += rem_qty * tp1_gain
                        rem_qty = 0.0
                        outcome = "TP2_STEP_TP1_1H"
                        break
                else:
                    if b_tp2 and (not b_be_subseq):
                        hit_tp2 = True
                        realized_pnl += 0.25 * tp2_gain
                        rem_qty = 0.25
                        outcome = "TP2_WIN_1H"
                        if b_tp3:
                            hit_tp3 = True
                            realized_pnl += 0.25 * tp3_runner_pay
                            rem_qty = 0.0
                            outcome = "TP3_WIN_1H"
                            break
                    elif b_be_subseq:
                        rem_qty = 0.0
                        outcome = "TP1_BE_1H"
                        break

        if not hit_tp1 and not hit_sl:
            # Fallback if daily session high/low reached TP1 outside partial 1h coverage without touching SL
            d_sl_chk = (daily_low <= sl) if is_long else (daily_high >= sl)
            if hit_tp1_any and not d_sl_chk:
                hit_tp1 = True
                exec_pnl = 0.50 * tp1_gain + 0.50 * max(0.0, raw_close_pnl)
                outcome = "TP1_WIN_1H" if raw_close_pnl > 0.0 else "TP1_BE_1H"
            else:
                exec_pnl = raw_close_pnl
        else:
            if rem_qty > 0.0:
                floor_lock = tp1_gain if hit_tp2 else 0.0
                realized_pnl += rem_qty * max(floor_lock, raw_close_pnl)
            exec_pnl = realized_pnl

        return {
            "path_source": "1H_CHRONOLOGICAL",
            "daily_target_win": bool(daily_target_win),
            "daily_target_pnl": round(daily_target_pnl, 2),
            "hit_tp1_any": bool(hit_tp1_any),
            "raw_close_win": bool(raw_close_win),
            "raw_close_pnl": round(raw_close_pnl, 2),
            "exec_win": bool(exec_pnl > 0.0),
            "exec_pnl": round(exec_pnl, 2),
            "hit_tp1_clean": bool(hit_tp1),
            "hit_tp2_clean": bool(hit_tp2),
            "hit_tp3_clean": bool(hit_tp3),
            "outcome": outcome,
        }

    # Conservative Daily OHLC Simulator (`2012-2024`)
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
        # When Close < Entry (`raw_close_pnl < 0`), `Open -> High -> Low -> Close` touched `TP1` (`~0.30 ATR`)
        # before reversing through `Entry ($0.00 BE)`; when `raw_close_pnl >= 0`, strictly charge `-sl_risk`!
        if raw_close_pnl < 0.0:
            hit_tp1 = True
            hit_tp2 = bool(d_tp2)
            hit_tp3 = False
            exec_pnl = (0.50 * tp1_gain + 0.25 * tp2_gain + 0.25 * tp1_gain) if d_tp2 else (0.50 * tp1_gain)
            outcome = "TP2_STEP_TP1_DAILY" if d_tp2 else "TP1_BE_DAILY"
        else:
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
        if d_tp3:
            hit_tp2 = True
            hit_tp3 = True
            exec_pnl = 0.50 * tp1_gain + 0.25 * tp2_gain + 0.25 * tp3_runner_pay
            outcome = "TP3_WIN_DAILY"
        elif d_tp2:
            hit_tp2 = True
            hit_tp3 = False
            exec_pnl = 0.50 * tp1_gain + 0.25 * tp2_gain + 0.25 * max(tp1_gain, raw_close_pnl)
            outcome = "TP2_WIN_DAILY"
        elif raw_close_pnl > 0.0:
            hit_tp2 = False
            hit_tp3 = False
            exec_pnl = 0.50 * tp1_gain + 0.50 * raw_close_pnl
            outcome = "TP1_WIN_DAILY"
        else:
            hit_tp2 = False
            hit_tp3 = False
            exec_pnl = 0.50 * tp1_gain
            outcome = "TP1_BE_DAILY"
    else:
        outcome = "TIME_CLOSE_DAILY"
        exec_pnl = raw_close_pnl
        hit_tp1 = False
        hit_tp2 = False
        hit_tp3 = False

    return {
        "path_source": "DAILY_CONSERVATIVE",
        "daily_target_win": bool(daily_target_win),
        "daily_target_pnl": round(daily_target_pnl, 2),
        "hit_tp1_any": bool(hit_tp1_any),
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
    """
    Aggregate BOTH transparent win metrics side-by-side:
      1. `daily_*`: `KXGOLDPERP` Daily Target / Direction Win% (`slEnabled=False` — hits TP1 or closes in dir)
      2. `exec_*` : Stepped `SL = 0.85 ATR + True Runner` Executable Win%, PF & `$/oz`
      3. `raw_*`  : Raw EOD Close Win%, PF & `$/oz`
    """
    sub = [r for r in rows if mask_fn(r) and r["actual_dir"] in ("XAU", "USD")]
    n = len(sub)
    if n == 0:
        return {
            "n": 0,
            "lock3_rate": 0.0,
            "all3_agree_rate": 0.0,
            "flip_count": 0,
            "daily_wins": 0,
            "daily_losses": 0,
            "daily_win_rate": 0.0,
            "daily_pf": 0.0,
            "daily_exp_usd": 0.0,
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
    all3_cnt = sum(1 for r in sub if r.get("all3_agree"))
    flip_cnt = sum(1 for r in sub if r.get("polarity_flipped"))

    daily_pnls = [float(r.get("daily_target_pnl", r["raw_close_pnl"])) for r in sub]
    daily_wins = sum(1 for r in sub if r.get("daily_target_win", r["raw_close_win"]))
    daily_losses = n - daily_wins
    daily_gw = sum(p for p in daily_pnls if p > 0.0)
    daily_gl = abs(sum(p for p in daily_pnls if p < 0.0))
    daily_pf = (daily_gw / daily_gl) if daily_gl > 1e-6 else (99.99 if daily_gw > 0 else 0.0)

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
        "all3_agree_rate": round(all3_cnt * 100.0 / n, 1),
        "flip_count": flip_cnt,
        "daily_wins": daily_wins,
        "daily_losses": daily_losses,
        "daily_win_rate": round(daily_wins * 100.0 / n, 2),
        "daily_pf": round(daily_pf, 2),
        "daily_exp_usd": round(float(np.mean(daily_pnls)), 2),
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
