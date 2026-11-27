#!/usr/bin/env python3
"""
8v8 Support/Resistance "Player Roster" Engine (`src/price_roster_engine.py`).
Exact `NBA-PLAYER-SIGNAL` Isomorphic Analog for Spot Gold (`XAUUSD`):

- Models every trading session as HOME = `XAU` (`77° GEMINI`, Bulls / 8 Support Defenders <= Open)
  vs AWAY = `USD` (`70° GEMINI`, Bears / 8 Resistance Cappers >= Open).
- Strictly causal at `T_0` (`Today_Open` + completed prior candles `<= i-1`): ZERO look-ahead.
- Dynamic Proximity Multiplier ("Minutes Played"):
    `prox_mult = exp(-abs(Today_Open - Level) / (1.5 * ATR14))`
    `Room_To_Run = (R1 - Today_Open) - (Today_Open - S1)`
"""
from __future__ import annotations

import math
from typing import Any

from src.astro_ephemeris_engine import eval_astro_entity_v82, score_degree_24_aspects
from src.xauusd_config import (
    ARC_ELEMENT_IDX,
    ASPECT_HARMONIC_WEIGHT,
    AWAY_USD_DEG,
    DEG1_GOLD,
    DEG2_XAUUSD,
    ELEMENT_INDEX,
    ELEMENT_ORDER,
    ELEMENT_VECTORS,
    HOME_XAU_DEG,
    best_aspect,
    decan36_info,
    deg_to_decan36,
    mod_arcanum,
    price_to_deg,
    zodiac_from_deg,
)


def _draft_support_slot(
    pref_cands: list[tuple[str, float]],
    master_pool: list[tuple[str, float]],
    today_open: float,
    min_gap: float,
    drafted: list[float],
    drafted_rounds: set[int],
    default_lbl: str,
) -> tuple[str, float]:
    """Pick highest-priority support candidate `<= today_open` separated by `>= min_gap` from all drafted teammates."""
    for lbl, val in pref_cands:
        if val <= today_open - 0.01 and int(round(val)) not in drafted_rounds:
            if all(abs(val - d) >= min_gap for d in drafted):
                drafted.append(val)
                drafted_rounds.add(int(round(val)))
                return lbl, round(val, 2)

    for lbl, val in master_pool:
        if val <= today_open - 0.01 and int(round(val)) not in drafted_rounds:
            if all(abs(val - d) >= min_gap for d in drafted):
                drafted.append(val)
                drafted_rounds.add(int(round(val)))
                return lbl, round(val, 2)

    # Ultimate psychological $25 step fallback strictly below lowest drafted support
    lowest = min(drafted) if drafted else today_open
    step_psych = math.floor((lowest - min_gap) / 25.0) * 25.0
    while int(round(step_psych)) in drafted_rounds or any(abs(step_psych - d) < min_gap for d in drafted):
        step_psych -= 25.0
    drafted.append(step_psych)
    drafted_rounds.add(int(round(step_psych)))
    return default_lbl, round(step_psych, 2)


def _draft_resistance_slot(
    pref_cands: list[tuple[str, float]],
    master_pool: list[tuple[str, float]],
    today_open: float,
    min_gap: float,
    drafted: list[float],
    drafted_rounds: set[int],
    default_lbl: str,
) -> tuple[str, float]:
    """Pick highest-priority resistance candidate `>= today_open` separated by `>= min_gap` from all drafted teammates."""
    for lbl, val in pref_cands:
        if val >= today_open + 0.01 and int(round(val)) not in drafted_rounds:
            if all(abs(val - d) >= min_gap for d in drafted):
                drafted.append(val)
                drafted_rounds.add(int(round(val)))
                return lbl, round(val, 2)

    for lbl, val in master_pool:
        if val >= today_open + 0.01 and int(round(val)) not in drafted_rounds:
            if all(abs(val - d) >= min_gap for d in drafted):
                drafted.append(val)
                drafted_rounds.add(int(round(val)))
                return lbl, round(val, 2)

    highest = max(drafted) if drafted else today_open
    step_psych = math.ceil((highest + min_gap) / 25.0) * 25.0
    while int(round(step_psych)) in drafted_rounds or any(abs(step_psych - d) < min_gap for d in drafted):
        step_psych += 25.0
    drafted.append(step_psych)
    drafted_rounds.add(int(round(step_psych)))
    return default_lbl, round(step_psych, 2)


def _select_distinct_supports(
    rec: dict[str, Any],
    today_open: float,
) -> list[tuple[str, str, float, float, bool]]:
    """
    Build the 8 strictly distinct HOME (`XAU`) Support Roster slots (`min_gap = max(2.0, 0.20 * ATR14)`):
      PG -> SG -> SF -> PF -> C -> B6 -> B7 -> B8.
    """
    atr14 = float(rec["atr14"])
    pdh, pdl, pdc = float(rec["pdh"]), float(rec["pdl"]), float(rec["prev_close"])
    prev2_low = float(rec["prev2_low"])
    pwl = float(rec["pwl"])
    low_10d = float(rec.get("low_10d", pwl))
    low_20d = float(rec["low_20d"])
    low_60d = float(rec.get("low_60d", low_20d))
    ema20 = float(rec["ema20"])
    ema50 = float(rec["ema50"])
    sma100 = float(rec.get("sma100", ema50))
    pm_low = float(rec["prev_month_low"])
    min_gap = max(2.0, 0.20 * atr14)
    swing_lows = [
        float(x) for x in rec["swing_lows"]
        if (today_open - 4.5 * atr14) <= float(x) < today_open
    ]
    rng = max(0.50, pdh - pdl)
    pp = (pdh + pdl + pdc) / 3.0

    # Floor, Fibonacci & Camarilla Supports (all strictly causal from i-1)
    piv_s1 = 2.0 * pp - pdh
    piv_s2 = pp - rng
    piv_s3 = pdl - 2.0 * (pdh - pp)
    piv_s4 = pdl - 3.0 * (pdh - pp)

    fib_s1 = pp - 0.382 * rng
    fib_s2 = pp - 0.618 * rng
    fib_s3 = pp - 1.000 * rng
    fib_s4 = pp - 1.618 * rng

    cam_s1 = pdc - 1.1 * rng / 12.0
    cam_s2 = pdc - 1.1 * rng / 6.0
    cam_s3 = pdc - 1.1 * rng / 4.0
    cam_s4 = pdc - 1.1 * rng / 2.0

    # Psychological $100 / $50 / $25 floors strictly below Today_Open
    p50_top = math.floor((today_open - 0.5) / 50.0) * 50.0
    p25_top = math.floor((today_open - 0.5) / 25.0) * 25.0
    psych_cands: list[tuple[str, float]] = []
    for k in range(12):
        v50 = p50_top - 50.0 * k
        lbl50 = "Psych_100_Sup" if int(v50) % 100 == 0 else "Psych_Floor_50"
        psych_cands.append((lbl50, v50))
    for k in range(20):
        v25 = p25_top - 25.0 * k
        lbl25 = "Psych_100_Sup" if int(v25) % 100 == 0 else ("Psych_Floor_50" if int(v25) % 50 == 0 else "Psych_25_Sup")
        psych_cands.append((lbl25, v25))

    sw_sorted_desc = sorted(set(round(x, 2) for x in swing_lows), reverse=True)
    sw_cands = [(f"Swing_Sup_{idx+1}", v) for idx, v in enumerate(sw_sorted_desc)]

    master_pool_raw: list[tuple[str, float]] = (
        [
            ("Pivot_S1", piv_s1), ("Fib_S1", fib_s1), ("Cam_S3", cam_s3), ("Cam_S4", cam_s4),
            ("PDL", pdl), ("Prev2_Low", prev2_low),
            ("Pivot_S2", piv_s2), ("Fib_S2", fib_s2), ("Fib_S3", fib_s3),
            ("PWL_5D", pwl), ("Low_10D", low_10d),
            ("Pivot_S3", piv_s3), ("Pivot_S4", piv_s4), ("Fib_S4", fib_s4),
            ("EMA20_Sup", ema20), ("EMA50_Sup", ema50), ("SMA100_Sup", sma100),
            ("PrevMo_Low", pm_low), ("Low_20D", low_20d), ("Low_60D", low_60d),
            ("Cam_S2", cam_s2), ("Cam_S1", cam_s1),
        ]
        + sw_cands
        + psych_cands
    )
    # Sort master pool descending (closest to Today_Open first) while keeping first-seen label per rounded price
    seen_r: set[int] = set()
    master_pool: list[tuple[str, float]] = []
    for lbl, val in sorted(master_pool_raw, key=lambda x: x[1], reverse=True):
        if val <= today_open - 0.05:
            rv = int(round(val))
            if rv not in seen_r:
                seen_r.add(rv)
                master_pool.append((lbl, val))

    # 1. PG: Today_Open
    s_pg = round(today_open, 2)
    drafted: list[float] = [s_pg]
    drafted_rounds: set[int] = {int(round(s_pg))}

    # 2. SG: Nearest structural S1 (Pivot/Camarilla/Fib S1 or closest fractal swing)
    sg_pref = [
        ("S1_Pivot", piv_s1), ("S1_Fib382", fib_s1), ("S1_Cam3", cam_s3),
    ] + [("S1_Swing", v) for v in sw_sorted_desc[:4]] + [("S1_Fib618", fib_s2)]
    sg_lbl, sg_lvl = _draft_support_slot(sg_pref, master_pool, today_open, min_gap, drafted, drafted_rounds, "S1_Support")

    # 3. SF: PDL (Prev_Day_Low -> Prev2_Low -> Cam_S4 -> Fib_S2)
    sf_pref = [("PDL", pdl), ("Prev2_Low", prev2_low), ("Cam_S4", cam_s4), ("Fib_S2", fib_s2)]
    sf_lbl, sf_lvl = _draft_support_slot(sf_pref, master_pool, today_open, min_gap, drafted, drafted_rounds, "PDL_Support")

    # 4. PF: S2 (Pivot S2 / Fib S2 / Fib S3 / nearby structural swing strictly within 2.5*ATR14)
    pf_cands = [
        ("S2_Pivot", piv_s2), ("S2_Fib618", fib_s2), ("S2_Fib100", fib_s3), ("Cam_S4", cam_s4),
    ] + [("S2_Swing", v) for v in sw_sorted_desc if (today_open - 2.5 * atr14) <= v < sg_lvl]
    pf_pref = sorted(pf_cands, key=lambda x: abs((today_open - 0.85 * atr14) - x[1]))
    pf_lbl, pf_lvl = _draft_support_slot(pf_pref, master_pool, today_open, min_gap, drafted, drafted_rounds, "S2_Struct")

    # 5. C: PWL (Rolling 5D Low -> 10D Low -> PrevMo Low -> 20D Low)
    c_pref = [("PWL_5D", pwl), ("Low_10D", low_10d), ("PrevMo_Low", pm_low), ("Low_20D", low_20d)]
    c_lbl, c_lvl = _draft_support_slot(c_pref, master_pool, today_open, min_gap, drafted, drafted_rounds, "PWL_Support")

    # 6. B6: S3 (Deep swing / Pivot S3 / Fib S4 / 60D Low)
    b6_pref = (
        [("S3_Pivot", piv_s3), ("S3_Fib1618", fib_s4), ("S4_Pivot", piv_s4)]
        + [("S3_DeepSw", v) for v in sw_sorted_desc if v < pf_lvl]
        + [("Low_60D", low_60d)]
    )
    b6_lbl, b6_lvl = _draft_support_slot(b6_pref, master_pool, today_open, min_gap, drafted, drafted_rounds, "S3_DeepSup")

    # 7. B7: Dynamic Moving Average / Structural Monthly Floor (EMA20 -> EMA50 -> SMA100 -> PrevMo_Low -> Low_20D)
    b7_pref = [
        ("EMA20_Sup", ema20), ("EMA50_Sup", ema50), ("SMA100_Sup", sma100),
        ("PrevMo_Low", pm_low), ("Low_20D", low_20d), ("Low_60D", low_60d),
    ]
    b7_lbl, b7_lvl = _draft_support_slot(b7_pref, master_pool, today_open, min_gap, drafted, drafted_rounds, "MA_Floor")

    # 8. B8: Psychological $50 / $100 / $25 Whole-Dollar Support
    b8_lbl, b8_lvl = _draft_support_slot(psych_cands, master_pool, today_open, min_gap, drafted, drafted_rounds, "Psych_Floor_50")

    return [
        ("PG", "Today_Open", s_pg,   1.30, True),
        ("SG", sg_lbl,       sg_lvl, 1.25, True),
        ("SF", sf_lbl,       sf_lvl, 1.15, True),
        ("PF", pf_lbl,       pf_lvl, 1.05, True),
        ("C",  c_lbl,        c_lvl,  1.00, True),
        ("B6", b6_lbl,       b6_lvl, 0.80, False),
        ("B7", b7_lbl,       b7_lvl, 0.75, False),
        ("B8", b8_lbl,       b8_lvl, 0.70, False),
    ]


def _select_distinct_resistances(
    rec: dict[str, Any],
    today_open: float,
) -> list[tuple[str, str, float, float, bool]]:
    """
    Build the 8 strictly distinct AWAY (`USD`) Resistance Roster slots (`min_gap = max(2.0, 0.20 * ATR14)`):
      PG (Prev_Close anchor) -> SG -> SF -> PF -> C -> B6 -> B7 -> B8.
    """
    atr14 = float(rec["atr14"])
    pdh, pdl, pdc = float(rec["pdh"]), float(rec["pdl"]), float(rec["prev_close"])
    prev2_high = float(rec["prev2_high"])
    pwh = float(rec["pwh"])
    high_10d = float(rec.get("high_10d", pwh))
    high_20d = float(rec["high_20d"])
    high_60d = float(rec.get("high_60d", high_20d))
    ema20 = float(rec["ema20"])
    ema50 = float(rec["ema50"])
    sma100 = float(rec.get("sma100", ema50))
    pm_high = float(rec["prev_month_high"])
    min_gap = max(2.0, 0.20 * atr14)
    swing_highs = [
        float(x) for x in rec["swing_highs"]
        if today_open < float(x) <= (today_open + 4.5 * atr14)
    ]
    rng = max(0.50, pdh - pdl)
    pp = (pdh + pdl + pdc) / 3.0

    piv_r1 = 2.0 * pp - pdl
    piv_r2 = pp + rng
    piv_r3 = pdh + 2.0 * (pp - pdl)
    piv_r4 = pdh + 3.0 * (pp - pdl)

    fib_r1 = pp + 0.382 * rng
    fib_r2 = pp + 0.618 * rng
    fib_r3 = pp + 1.000 * rng
    fib_r4 = pp + 1.618 * rng

    cam_r1 = pdc + 1.1 * rng / 12.0
    cam_r2 = pdc + 1.1 * rng / 6.0
    cam_r3 = pdc + 1.1 * rng / 4.0
    cam_r4 = pdc + 1.1 * rng / 2.0

    p50_bot = math.ceil((today_open + 0.5) / 50.0) * 50.0
    p25_bot = math.ceil((today_open + 0.5) / 25.0) * 25.0
    psych_cands: list[tuple[str, float]] = []
    for k in range(12):
        v50 = p50_bot + 50.0 * k
        lbl50 = "Psych_100_Cap" if int(v50) % 100 == 0 else "Psych_Cap_50"
        psych_cands.append((lbl50, v50))
    for k in range(20):
        v25 = p25_bot + 25.0 * k
        lbl25 = "Psych_100_Cap" if int(v25) % 100 == 0 else ("Psych_Cap_50" if int(v25) % 50 == 0 else "Psych_25_Cap")
        psych_cands.append((lbl25, v25))

    sw_sorted_asc = sorted(set(round(x, 2) for x in swing_highs))
    sw_cands = [(f"Swing_Res_{idx+1}", v) for idx, v in enumerate(sw_sorted_asc)]

    master_pool_raw: list[tuple[str, float]] = (
        [
            ("Pivot_R1", piv_r1), ("Fib_R1", fib_r1), ("Cam_R3", cam_r3), ("Cam_R4", cam_r4),
            ("PDH", pdh), ("Prev2_High", prev2_high),
            ("Pivot_R2", piv_r2), ("Fib_R2", fib_r2), ("Fib_R3", fib_r3),
            ("PWH_5D", pwh), ("High_10D", high_10d),
            ("Pivot_R3", piv_r3), ("Pivot_R4", piv_r4), ("Fib_R4", fib_r4),
            ("High_20D", high_20d), ("PrevMo_High", pm_high), ("High_60D", high_60d),
            ("EMA20_Cap", ema20), ("EMA50_Cap", ema50), ("SMA100_Cap", sma100),
            ("Cam_R2", cam_r2), ("Cam_R1", cam_r1),
        ]
        + sw_cands
        + psych_cands
    )
    seen_r: set[int] = set()
    master_pool: list[tuple[str, float]] = []
    for lbl, val in sorted(master_pool_raw, key=lambda x: x[1]):
        if val >= today_open + 0.05:
            rv = int(round(val))
            if rv not in seen_r:
                seen_r.add(rv)
                master_pool.append((lbl, val))

    # 1. PG: Prev_Close anchor
    r_pg = round(pdc, 2)
    drafted: list[float] = [r_pg]
    drafted_rounds: set[int] = {int(round(r_pg))}

    # 2. SG: Nearest structural R1 (Pivot/Camarilla/Fib R1 or closest fractal swing)
    sg_pref = [
        ("R1_Pivot", piv_r1), ("R1_Fib382", fib_r1), ("R1_Cam3", cam_r3),
    ] + [("R1_Swing", v) for v in sw_sorted_asc[:4]] + [("R1_Fib618", fib_r2)]
    sg_lbl, sg_lvl = _draft_resistance_slot(sg_pref, master_pool, today_open, min_gap, drafted, drafted_rounds, "R1_Resist")

    # 3. SF: PDH (Prev_Day_High -> Prev2_High -> Cam_R4 -> Fib_R2)
    sf_pref = [("PDH", pdh), ("Prev2_High", prev2_high), ("Cam_R4", cam_r4), ("Fib_R2", fib_r2)]
    sf_lbl, sf_lvl = _draft_resistance_slot(sf_pref, master_pool, today_open, min_gap, drafted, drafted_rounds, "PDH_Resist")

    # 4. PF: R2 (Pivot R2 / Fib R2 / Fib R3 / nearby structural swing strictly within 2.5*ATR14)
    pf_cands = [
        ("R2_Pivot", piv_r2), ("R2_Fib618", fib_r2), ("R2_Fib100", fib_r3), ("Cam_R4", cam_r4),
    ] + [("R2_Swing", v) for v in sw_sorted_asc if sg_lvl < v <= (today_open + 2.5 * atr14)]
    pf_pref = sorted(pf_cands, key=lambda x: abs((today_open + 0.85 * atr14) - x[1]))
    pf_lbl, pf_lvl = _draft_resistance_slot(pf_pref, master_pool, today_open, min_gap, drafted, drafted_rounds, "R2_Struct")

    # 5. C: PWH (Rolling 5D High -> 10D High -> PrevMo High -> 20D High)
    c_pref = [("PWH_5D", pwh), ("High_10D", high_10d), ("PrevMo_High", pm_high), ("High_20D", high_20d)]
    c_lbl, c_lvl = _draft_resistance_slot(c_pref, master_pool, today_open, min_gap, drafted, drafted_rounds, "PWH_Resist")

    # 6. B6: R3 (Deep swing / Pivot R3 / Fib R4 / 60D High)
    b6_pref = (
        [("R3_Pivot", piv_r3), ("R3_Fib1618", fib_r4), ("R4_Pivot", piv_r4)]
        + [("R3_DeepSw", v) for v in sw_sorted_asc if v > pf_lvl]
        + [("High_60D", high_60d)]
    )
    b6_lbl, b6_lvl = _draft_resistance_slot(b6_pref, master_pool, today_open, min_gap, drafted, drafted_rounds, "R3_DeepRes")

    # 7. B7: Rolling 20D High / PrevMo High / MA Cap
    b7_pref = [
        ("High_20D", high_20d), ("PrevMo_High", pm_high), ("High_60D", high_60d),
        ("EMA20_Cap", ema20), ("EMA50_Cap", ema50), ("SMA100_Cap", sma100),
    ]
    b7_lbl, b7_lvl = _draft_resistance_slot(b7_pref, master_pool, today_open, min_gap, drafted, drafted_rounds, "High20D_Cap")

    # 8. B8: Psychological $50 / $100 / $25 Whole-Dollar Resistance
    b8_lbl, b8_lvl = _draft_resistance_slot(psych_cands, master_pool, today_open, min_gap, drafted, drafted_rounds, "Psych_Cap_50")

    return [
        ("PG", "Prev_Close", r_pg,   1.30, True),
        ("SG", sg_lbl,       sg_lvl, 1.25, True),
        ("SF", sf_lbl,       sf_lvl, 1.15, True),
        ("PF", pf_lbl,       pf_lvl, 1.05, True),
        ("C",  c_lbl,        c_lvl,  1.00, True),
        ("B6", b6_lbl,       b6_lvl, 0.80, False),
        ("B7", b7_lbl,       b7_lvl, 0.75, False),
        ("B8", b8_lbl,       b8_lvl, 0.70, False),
    ]


_POLE_RES_CACHE: dict[tuple[int, str, int], float] = {}


def _score_pole_resonance(
    p_deg: int,
    team: str,
    d_ctx: dict[str, Any],
    need_hits: bool = False,
) -> tuple[float, list[str]]:
    """
    24-Aspect resonance of player degree `p_deg` against foundational poles
    (`DEG1=107°`, `DEG2=147°`, `HOME=77°`, `AWAY=70°` and daily `F1, F2, F3_HOME, F3_AWAY`).
    """
    fh_mod = d_ctx["f_helper"] % 360
    ck = (p_deg, team, fh_mod)
    if not need_hits and ck in _POLE_RES_CACHE:
        return _POLE_RES_CACHE[ck], []

    own_deg = HOME_XAU_DEG if team == "XAU" else AWAY_USD_DEG
    opp_deg = AWAY_USD_DEG if team == "XAU" else HOME_XAU_DEG
    own_f3 = (d_ctx["F3_HOME"] % 360) if team == "XAU" else (d_ctx["F3_AWAY"] % 360)

    poles = (
        ("GOLD107", DEG1_GOLD, +0.90),
        ("PAIR147", DEG2_XAUUSD, +0.55),
        ("OWN_POLE", own_deg, +1.15),
        ("OPP_POLE", opp_deg, -0.55),
        ("F1", d_ctx["F1"] % 360, +0.65),
        ("F3_OWN", own_f3, +0.85),
    )
    tot = 0.0
    tags: list[str] = []
    for pname, tdeg, mult in poles:
        sym, ex = best_aspect(p_deg, tdeg, "")
        if sym is not None:
            hw = ASPECT_HARMONIC_WEIGHT.get(sym, 0.0)
            tot += hw * mult * ex
            if need_hits and ex >= 0.50:
                tags.append(f"{pname}:{sym}")
    res = round(tot, 4)
    if not need_hits:
        _POLE_RES_CACHE[ck] = res
    return res, tags


def _evaluate_player_card(
    slot_idx: int,
    team: str,
    pos: str,
    role: str,
    level: float,
    base_w: float,
    is_starter: bool,
    today_open: float,
    atr14: float,
    d_ctx: dict[str, Any],
    snap: dict[str, Any],
    rounding_mode: str,
    need_hits: bool = False,
) -> dict[str, Any]:
    """Evaluate a single S/R Roster Player (`int(Level) % 360` or `round(Level) % 360`)."""
    prox_mult = math.exp(-abs(today_open - level) / max(0.5, 1.5 * atr14))
    eff_w = base_w * prox_mult

    p_deg = price_to_deg(level, rounding_mode)
    z_label, z_elem, _ = zodiac_from_deg(p_deg)
    hc_vec, md_vec = ELEMENT_VECTORS[z_elem]

    dec_num = deg_to_decan36(p_deg)
    dec_meta = decan36_info(dec_num)
    arc_num = mod_arcanum(int(math.floor(level)) if rounding_mode == "floor" else int(round(level)))

    # Player intrinsic elemental 4-vector (Zodiac + Chaldean Ruler + Arcanum)
    p_fawe = [0, 0, 0, 0]
    p_fawe[ELEMENT_INDEX[z_elem]] += 1
    p_fawe[ELEMENT_INDEX[dec_meta["ruler_element"]]] += 1
    p_fawe[ARC_ELEMENT_IDX[arc_num]] += 1

    team_pole = HOME_XAU_DEG if team == "XAU" else AWAY_USD_DEG
    deg3_coupled = (team_pole + dec_num + slot_idx) % 360
    v82 = eval_astro_entity_v82(d_ctx, DEG1_GOLD, p_deg, deg3_coupled, tuple(p_fawe), snap["deg_tbl"])

    transit_asp, transit_hits = score_degree_24_aspects(p_deg, snap, need_hits=need_hits)
    pole_asp, pole_hits = _score_pole_resonance(p_deg, team, d_ctx, need_hits=need_hits)

    # Ruler thermal/moisture alignment & Planetary Hour / Pole elemental harmony
    r_hc, r_md = ELEMENT_VECTORS[dec_meta["ruler_element"]]
    net_hc = hc_vec + 0.65 * r_hc
    net_md = md_vec + 0.65 * r_md

    ph_elem = str(snap.get("planetary_hour_element", "FIRE"))
    ph_match = 0.30 if (z_elem == ph_elem or dec_meta["ruler_element"] == ph_elem) else -0.08
    pole_elem_match = (
        0.22 if (team == "XAU" and z_elem in ("WATER", "FIRE"))
        else (0.22 if (team == "USD" and z_elem in ("AIR", "EARTH")) else -0.05)
    )

    # Composite single-player resonance strength (weighted by eff_w, normalized at team level so proximity alone never fades trends)
    v82_anchor = (-v82["gf_fin"]) * 0.16 + (-v82["pwr"]) * 0.04
    unit_resonance = 0.50 * transit_asp + 0.30 * pole_asp + 0.14 * v82_anchor + 0.18 * (ph_match + pole_elem_match)
    player_resonance = unit_resonance * eff_w

    return {
        "slot": slot_idx + 1,
        "team": team,
        "pos": pos,
        "role": role,
        "starter": is_starter,
        "level": level,
        "dist_atr": round(abs(today_open - level) / max(0.5, atr14), 3),
        "base_w": base_w,
        "prox_mult": round(prox_mult, 4),
        "eff_w": round(eff_w, 4),
        "deg": p_deg,
        "zodiac": z_label,
        "element": z_elem,
        "decan": dec_num,
        "decan_label": dec_meta["label"],
        "decan_ruler": dec_meta["ruler"],
        "ruler_element": dec_meta["ruler_element"],
        "arcanum": arc_num,
        "net_hc": round(net_hc, 3),
        "net_md": round(net_md, 3),
        "transit_asp": transit_asp,
        "pole_asp": pole_asp,
        "aspect_hits": (transit_hits[:3] + pole_hits[:2])[:4],
        "ep": v82["ep"],
        "ef": v82["ef"],
        "ed": v82["ed"],
        "gf": v82["gf"],
        "gf_fin": v82["gf_fin"],
        "pwr": v82["pwr"],
        "unit_resonance": round(unit_resonance, 4),
        "resonance": round(player_resonance, 4),
    }


def _summarize_team_roster(players: list[dict[str, Any]]) -> dict[str, Any]:
    w_fawe = [0.0, 0.0, 0.0, 0.0]
    decan_fawe_int = [0, 0, 0, 0]
    zod_fawe_int = [0, 0, 0, 0]
    ef_tot = [0, 0, 0, 0]

    tot_eff_w = 0.0
    starter_eff_w = 0.0
    net_hc_w = 0.0
    net_md_w = 0.0
    res_sum = 0.0
    starter_res_sum = 0.0
    gf_fin_w = 0.0
    warm_1xxx_w = 0.0
    cold_0xxx_w = 0.0
    bipolar_0011_w = 0.0
    cnt_1xxx = 0

    for p in players:
        ew = p["eff_w"]
        tot_eff_w += ew
        if p["starter"]:
            starter_eff_w += ew
        el_idx = ELEMENT_INDEX[p["element"]]
        rul_idx = ELEMENT_INDEX[p["ruler_element"]]

        w_fawe[el_idx] += ew
        w_fawe[rul_idx] += 0.5 * ew

        if p["starter"]:
            zod_fawe_int[el_idx] += 1
            decan_fawe_int[rul_idx] += 1

        for j in range(4):
            ef_tot[j] += p["ef"][j]

        net_hc_w += ew * p["net_hc"]
        net_md_w += ew * p["net_md"]
        res_sum += p["resonance"]
        if p["starter"]:
            starter_res_sum += p["resonance"]

        gf_fin_w += ew * p["gf_fin"]
        if p["gf"].startswith("1"):
            warm_1xxx_w += ew * p["pwr"]
            cnt_1xxx += 1
        else:
            cold_0xxx_w += ew * p["pwr"]
        if p["gf"] == "0011":
            bipolar_0011_w += ew * p["pwr"]

    mx_w = max(w_fawe)
    dom_idx = w_fawe.index(mx_w)
    dom_elem = ELEMENT_ORDER[dom_idx]

    denom_all = max(0.35, tot_eff_w)
    denom_st = max(0.35, starter_eff_w)

    return {
        "players": players,
        "tot_eff_w": round(tot_eff_w, 4),
        "starter_eff_w": round(starter_eff_w, 4),
        "w_fawe": [round(x, 3) for x in w_fawe],
        "zod_fawe_int": tuple(zod_fawe_int),
        "decan_fawe_int": tuple(decan_fawe_int),
        "ef_tot": tuple(ef_tot),
        "dom_element": dom_elem,
        "net_hc": round(net_hc_w / denom_all, 4),
        "net_md": round(net_md_w / denom_all, 4),
        "resonance_sum": round(res_sum, 4),
        "starter_resonance_sum": round(starter_res_sum, 4),
        "resonance_norm": round(res_sum / denom_all, 4),
        "starter_resonance_norm": round(starter_res_sum / denom_st, 4),
        "gf_fin_w": round(gf_fin_w / denom_all, 4),
        "warm_1xxx_w": round(warm_1xxx_w / denom_all, 4),
        "cold_0xxx_w": round(cold_0xxx_w / denom_all, 4),
        "bipolar_0011_w": round(bipolar_0011_w / denom_all, 4),
        "cnt_1xxx": cnt_1xxx,
    }


def evaluate_price_roster(
    rec: dict[str, Any],
    eng_a: dict[str, Any],
    rounding_mode: str = "floor",
    override_open: float | None = None,
    need_hits: bool = False,
) -> dict[str, Any]:
    """
    Build & evaluate Breakout-Aware 8v8 Support/Resistance Roster (`HOME = XAU` vs `AWAY = USD`)
    strictly at `T_0` (`override_open` if session open B/C, else `rec["today_open"]`).
    """
    today_open = float(override_open if override_open is not None else rec["today_open"])
    atr14 = float(rec["atr14"])
    d_ctx = eng_a["d_ctx"]
    snap_xau = eng_a["snap_xau"]
    snap_usd = eng_a["snap_usd"]

    xau_slots = _select_distinct_supports(rec=rec, today_open=today_open)
    usd_slots = _select_distinct_resistances(rec=rec, today_open=today_open)

    xau_cards = [
        _evaluate_player_card(i, "XAU", pos, role, lvl, bw, st, today_open, atr14, d_ctx, snap_xau, rounding_mode, need_hits=need_hits)
        for i, (pos, role, lvl, bw, st) in enumerate(xau_slots)
    ]
    usd_cards = [
        _evaluate_player_card(i, "USD", pos, role, lvl, bw, st, today_open, atr14, d_ctx, snap_usd, rounding_mode, need_hits=need_hits)
        for i, (pos, role, lvl, bw, st) in enumerate(usd_slots)
    ]

    # Auxiliary DXY*100 captain resonance for AWAY USD
    dxy_cap_deg = price_to_deg(rec["dxy_prev_close"] * 100.0, rounding_mode)
    dxy_asp, _ = score_degree_24_aspects(dxy_cap_deg, snap_usd, need_hits=False)

    xau_sum = _summarize_team_roster(xau_cards)
    usd_sum = _summarize_team_roster(usd_cards)

    s1_lvl = xau_cards[1]["level"]
    r1_lvl = usd_cards[1]["level"]
    s2_lvl = xau_cards[3]["level"]
    r2_lvl = usd_cards[3]["level"]

    dist_to_s1 = max(0.0, today_open - s1_lvl)
    dist_to_r1 = max(0.0, r1_lvl - today_open)
    dist_s1_atr = dist_to_s1 / max(0.5, atr14)
    dist_r1_atr = dist_to_r1 / max(0.5, atr14)

    # Major structural Room_To_Run (`dist_to_major_R2 - dist_to_major_S2`):
    # Positive when solid structural support S2 sits close beneath Open while R2 leaves wide upside air pocket!
    dist_to_s2 = max(0.0, today_open - s2_lvl)
    dist_to_r2 = max(0.0, r2_lvl - today_open)
    room_to_run = dist_to_r2 - dist_to_s2
    room_norm = room_to_run / max(0.5, atr14)

    # 3-of-4 Element Lead check across 8-player rosters
    xau_elem_leads = sum(1 for j in range(4) if xau_sum["ef_tot"][j] > usd_sum["ef_tot"][j])
    usd_elem_leads = sum(1 for j in range(4) if usd_sum["ef_tot"][j] > xau_sum["ef_tot"][j])

    # Proximity-normalized astrological resonance & elemental harmony (no blind R1/S1 proximity fade!)
    res_edge = (xau_sum["resonance_norm"] - (usd_sum["resonance_norm"] + 0.05 * dxy_asp)) * 1.45
    st_res_edge = (xau_sum["starter_resonance_norm"] - usd_sum["starter_resonance_norm"]) * 0.95
    thermal_edge = (xau_sum["net_hc"] - usd_sum["net_hc"]) * 0.25 + (xau_sum["net_md"] - usd_sum["net_md"]) * 0.18
    cold_0xxx_edge = -(usd_sum["cold_0xxx_w"] - xau_sum["cold_0xxx_w"]) * 0.08
    warm_1xxx_edge = (usd_sum["warm_1xxx_w"] - xau_sum["warm_1xxx_w"]) * 0.06
    lead_edge = 0.18 if xau_elem_leads >= 3 else (-0.18 if usd_elem_leads >= 3 else 0.0)

    # Supported Breakout + Macro Trend Launchpad (`EMA20`, `EMA50`, `5D Weekly Candle State`)
    regime = str(rec.get("macro_trend_regime", "NEUTRAL"))
    ema20_slope = float(rec.get("ema20_slope", 0.0))
    dxy_ret_5d = float(rec.get("dxy_ret_5d", 0.0))
    ret_5d = float(rec["prev_ret_5"])

    sup_launchpad_res = sum(max(0.0, float(c["unit_resonance"])) for c in xau_cards[1:4]) / 3.0
    res_cap_res = sum(max(0.0, float(c["unit_resonance"])) for c in usd_cards[1:4]) / 3.0
    if regime == "STRONG_BULL":
        macro_launchpad_bonus = +0.26 + 0.16 * sup_launchpad_res + (0.12 if ret_5d > 0.0 else 0.0)
    elif regime == "STRONG_BEAR":
        macro_launchpad_bonus = -0.26 - 0.16 * res_cap_res - (0.12 if ret_5d < 0.0 else 0.0)
    else:
        macro_launchpad_bonus = 0.10 * math.tanh(ret_5d * 0.45)

    room_breakout_edge = +0.16 * math.tanh(room_norm * 0.65)
    trend_momentum = +0.16 * math.tanh(ema20_slope * 1.6) + macro_launchpad_bonus
    chop_damping = (-0.12 * math.tanh(rec["prev_ret_1"] * 0.75)) if regime == "NEUTRAL" else 0.0
    dxy_alignment = +0.14 * math.tanh(rec["dxy_prev_ret"] * 160.0) - 0.14 * math.tanh(dxy_ret_5d * 55.0)

    score_b = (
        res_edge
        + st_res_edge
        + thermal_edge
        + cold_0xxx_edge
        + warm_1xxx_edge
        + lead_edge
        + room_breakout_edge
        + trend_momentum
        + chop_damping
        + dxy_alignment
    )

    # Boxed S/R Compression diagnostics for Veto Layer 5 (LONG breakouts in Strong Bull are NOT penalized):
    boxed_against_r1 = (
        (regime != "STRONG_BULL")
        and (dist_r1_atr < 0.15)
        and (usd_cards[1]["eff_w"] >= 1.10)
        and (usd_cards[1]["unit_resonance"] >= xau_cards[1]["unit_resonance"])
    )
    boxed_against_s1 = (
        (regime != "STRONG_BEAR")
        and (dist_s1_atr < 0.15)
        and (xau_cards[1]["eff_w"] >= 1.10)
        and (xau_cards[1]["unit_resonance"] >= usd_cards[1]["unit_resonance"])
    )

    return {
        "entry_open": round(today_open, 2),
        "s1": s1_lvl,
        "s2": s2_lvl,
        "r1": r1_lvl,
        "r2": r2_lvl,
        "dist_s1_atr": round(dist_s1_atr, 4),
        "dist_r1_atr": round(dist_r1_atr, 4),
        "room_to_run": round(room_to_run, 2),
        "room_norm": round(room_norm, 4),
        "macro_trend_regime": regime,
        "boxed_against_r1": bool(boxed_against_r1),
        "boxed_against_s1": bool(boxed_against_s1),
        "xau_roster": xau_sum,
        "usd_roster": usd_sum,
        "xau_elem_leads": xau_elem_leads,
        "usd_elem_leads": usd_elem_leads,
        "dxy_cap_deg": dxy_cap_deg,
        "dxy_asp": round(dxy_asp, 4),
        "score_b": round(score_b, 4),
        "dir_b": "XAU" if score_b >= 0 else "USD",
    }
