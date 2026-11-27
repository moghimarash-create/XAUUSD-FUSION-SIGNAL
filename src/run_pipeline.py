#!/usr/bin/env python3
"""
Master Zero-Leakage Walk-Forward Pipeline, Shard Exporter & Benchmark Runner
for `XAUUSD-FUSION-SIGNAL` (`v4.0.0` Twin-First Production Engine).

Executes:
1. Locked Phase 1 Winner (`ANCHOR_DUAL_POLE_1800` + `round(P) % 360`) Zero-Leakage Walk-Forward:
   - LBMA London (`51.5142, -0.0885`) for `HOME XAU` vs COMEX New York (`40.7143, -74.0139`) for `AWAY USD`
     at `18:00:00 America/New_York` on `D-1`.
2. 100% Price-Free Pure Astro-Arcana-Decan `12-Col Signature` + ATR-Normalized Recency-Weighted
   (`exp(-years_ago / 6.0)`) Non-Parametric Twin Consensus + 6-Layer Veto Shield (`V6_COUNTER_MACRO_TREND`).
3. Strict Train-Only (`2012–2024`) Calibration & Freeze (`Val 2025` and `Blind OOS 2026` untouched).
4. Side-by-Side `v1 vs v3 vs v4` Comparison across `Train`, `Val`, and `Blind OOS` (Raw Close & `1h` `TP1-Before-SL`).
5. Exports production shards:
   - `shards/latest_xauusd_signals.json`
   - `data/xauusd_history_shard.json`
   - `data/xauusd_backtest_summary.json`
   - `data/latest_xauusd_signal.json`
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import datetime
import json
import math
import os
import pickle
import time
from typing import Any

import requests

from src.astro_ephemeris_engine import evaluate_engine_a
from src.data_loader import (
    DATA_DIR,
    build_zero_leakage_daily_records,
    fetch_and_cache_market_data,
    group_1h_bars_by_session,
)
from src.price_roster_engine import evaluate_price_roster
from src.trade_path_simulator import simulate_session_execution, summarize_dual_metrics
from src.twin_lock_engine import HistoricalTwinArchive, build_day_12col_signature
from src.xauusd_config import (
    ANCHOR_CANDIDATES,
    AWAY_USD_DEG,
    DEG1_GOLD,
    DEG2_XAUUSD,
    HOME_XAU_DEG,
    zodiac_from_deg,
)
from src.xauusd_gold_cba_engine import (
    FrozenGoldCbaConfig,
    evaluate_xauusd_gold_cba,
    fit_gold_cba_on_train,
    verify_subset_invariant,
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHARDS_DIR = os.path.join(ROOT_DIR, "shards")

LOCKED_ANCHOR_KEY = "ANCHOR_DUAL_POLE_1800"
LOCKED_ROUNDING_MODE = "round"


def _roster_is_distinct_8(roster: dict[str, Any]) -> bool:
    players = roster.get("players", [])
    return len(players) == 8 and len({int(round(float(p["level"]))) for p in players}) == 8


def _enrich_legacy_keys(dual_m: dict[str, Any]) -> dict[str, Any]:
    return {
        **dual_m,
        "wins": dual_m["raw_wins"],
        "losses": dual_m["raw_losses"],
        "win_rate": dual_m["raw_win_rate"],
        "pf": dual_m["raw_pf"],
        "expectancy_usd": dual_m["raw_exp_usd"],
    }


def _verify_price_free_twin_signature(sample_date: str = "2026-11-26") -> bool:
    """Verify that changing Gold price from $1,150 to $4,350 produces bit-for-bit identical 12-col signature."""
    ea_1 = evaluate_engine_a(sample_date, 1150.25, 1142.80, 92.40, LOCKED_ANCHOR_KEY, LOCKED_ROUNDING_MODE)
    ea_2 = evaluate_engine_a(sample_date, 4350.90, 4295.10, 108.75, LOCKED_ANCHOR_KEY, LOCKED_ROUNDING_MODE)
    sig_1 = build_day_12col_signature(ea_1, None)
    sig_2 = build_day_12col_signature(ea_2, None)
    return bool(sig_1["price_free"] and sig_1 == sig_2)


def _outcome_badge(row: dict[str, Any]) -> str:
    if not row.get("is_a"):
        return "🛡️ VETO / HOLD"
    if row.get("hit_tp3_clean"):
        return "✓ TP3"
    if row.get("hit_tp2_clean"):
        return "✓ TP2"
    if row.get("hit_tp1_clean"):
        return "✓ TP1"
    out_s = str(row.get("outcome", ""))
    if "SL_LOSS" in out_s:
        return "✗ SL"
    return "✓ CLOSE+" if row.get("exec_win") else "✗ CLOSE-"


def _build_compact_history_row(r: dict[str, Any]) -> dict[str, Any]:
    out_s = str(r.get("outcome", ""))
    hit_sl = "SL_LOSS" in out_s
    return {
        "date": r["date"],
        "weekday": r["weekday"],
        "split": r["split"],
        "macroRegime": r["macro_trend_regime"],
        "tier": r["tier"] or "HOLD",
        "tierBadge": r["tier_badge"],
        "isC": bool(r["is_c"]),
        "isB": bool(r["is_b"]),
        "isA": bool(r["is_a"]),
        "direction": r["action"],
        "biasDirection": r["pred_dir"],
        "winProbPct": r["win_prob_pct"],
        "kellyMultiplier": r["kelly_multiplier"],
        "entryPrice": r["entry"],
        "sl": r["sl"],
        "tp1": r["tp1"],
        "tp2": r["tp2"],
        "tp3": r["tp3"],
        "rrRatio": r["rr_ratio"],
        "actualHigh": r["actual_high"],
        "actualLow": r["actual_low"],
        "actualClose": r["actual_close"],
        "actualDir": r["actual_dir"],
        "actualMove": r["actual_move"],
        "execWin": bool(r["exec_win"]),
        "execPnl": r["exec_pnl"],
        "rawCloseWin": bool(r["raw_close_win"]),
        "rawClosePnl": r["raw_close_pnl"],
        "hitTp1": bool(r.get("hit_tp1_clean", False)),
        "hitTp2": bool(r.get("hit_tp2_clean", False)),
        "hitTp3": bool(r.get("hit_tp3_clean", False)),
        "hitSl": hit_sl,
        "outcome": out_s,
        "outcomeBadge": _outcome_badge(r),
        "pathSource": r.get("path_source", "DAILY_CONSERVATIVE"),
        "vetoReasons": r["veto_flags"],
        "twPct": r["tw_pct"],
        "wDispPct": r.get("w_disp_pct", r["tw_pct"]),
        "twinVotes": f"{r['up_twins']} XAU - {r['down_twins']} USD",
        "lock3Peak": bool(r["lock3_peak"]),
        "hasLock3": bool(r["has_lock3"]),
        "geoFig": f"{r['geo_fig']} ({r['geo_name']})",
        "decGeo": r["dec_geo"],
        "rulerGeo": r["ruler_geo"],
    }


def run_walk_forward_for_config(
    records: list[dict[str, Any]],
    anchor_key: str = LOCKED_ANCHOR_KEY,
    rounding_mode: str = LOCKED_ROUNDING_MODE,
    bars_1h_map: dict[str, list[tuple[float, float, float, float]]] | None = None,
    use_intraday_session_open: bool = False,
) -> dict[str, Any]:
    """
    Execute a complete zero-leakage walk-forward run (`v4 Twin-First Production Engine`):
      Pass 1 (Causal Feature Pass `2001..2026`):
        - At each day `i`, compute Engines A, B, C strictly from `< date_i` + `T_0` open,
          THEN append day `i`'s completed candle into `HistoricalTwinArchive`.
      Pass 2 (Train-Only Calibration & Freeze):
        - Fit `FrozenGoldCbaConfig` exclusively on `TRAIN (2012–2024)` rows (`date <= 2024-12-31`).
        - Evaluate frozen `Gold CBA` + `6-Layer Veto Shield` + Dual Execution Simulator
          across `TRAIN`, `VAL (2025)`, and `OOS (2026)`.
    """
    t0 = time.time()
    archive = HistoricalTwinArchive()
    archive_before_latest: HistoricalTwinArchive | None = None
    day_states: list[dict[str, Any]] = []

    distinct_both_ok = 0
    total_roster_days = 0
    price_free_ok = _verify_price_free_twin_signature()
    max_sig_sums = [0, 0, 0]  # [ELE-P, FINAL, PRIM]

    n_recs = len(records)
    for idx_rec, rec in enumerate(records):
        if idx_rec == n_recs - 1:
            archive_before_latest = copy.deepcopy(archive)

        entry_open = float(rec["today_open"])
        target_close = float(rec["_target_close"])
        if use_intraday_session_open and rec["split"] in ("VAL", "OOS"):
            if anchor_key == "ANCHOR_B_LONDON_0800" and not math.isnan(float(rec.get("open_0800_london", math.nan))):
                entry_open = float(rec["open_0800_london"])
                if not math.isnan(float(rec.get("close_1700_ny", math.nan))):
                    target_close = float(rec["close_1700_ny"])
            elif anchor_key == "ANCHOR_C_NY_0820" and not math.isnan(float(rec.get("open_0820_ny", math.nan))):
                entry_open = float(rec["open_0820_ny"])
                if not math.isnan(float(rec.get("close_1700_ny", math.nan))):
                    target_close = float(rec["close_1700_ny"])

        ea = evaluate_engine_a(
            d_iso=rec["date"],
            today_open=entry_open,
            prev_close=rec["prev_close"],
            dxy_prev_close=rec["dxy_prev_close"],
            anchor_key=anchor_key,
            rounding_mode=rounding_mode,
            need_hits=False,
        )
        eb = evaluate_price_roster(rec, ea, rounding_mode=rounding_mode, override_open=entry_open, need_hits=False)
        sig = build_day_12col_signature(ea)
        price_free_ok = price_free_ok and bool(sig.get("price_free", False))

        total_roster_days += 1
        if _roster_is_distinct_8(eb["xau_roster"]) and _roster_is_distinct_8(eb["usd_roster"]):
            distinct_both_ok += 1

        max_sig_sums[0] = max(max_sig_sums[0], sum(sig["elep"]))
        max_sig_sums[1] = max(max_sig_sums[1], sum(sig["final"]))
        max_sig_sums[2] = max(max_sig_sums[2], sum(sig["prim"]))

        ec = archive.evaluate_twin_and_gaussian(
            target_date=rec["date"],
            weekday_up=rec["weekday_up"],
            sig=sig,
            entry_open=entry_open,
            atr14=rec["atr14"],
            s1=eb["s1"],
            r1=eb["r1"],
            fallback_dir=eb["dir_b"],
            s2=eb["s2"],
            r2=eb["r2"],
        )

        sess_move = target_close - entry_open
        sess_dir = "XAU" if sess_move > 0 else ("USD" if sess_move < 0 else "FLAT")
        act_high = max(entry_open, float(rec["_target_high"]))
        act_low = min(entry_open, float(rec["_target_low"]))

        if rec["split"] in ("TRAIN", "VAL", "OOS"):
            day_states.append(
                {
                    "rec": rec,
                    "ea": ea,
                    "eb": eb,
                    "sig": sig,
                    "ec": ec,
                    "date": rec["date"],
                    "split": rec["split"],
                    "score_a": ea["score_a"],
                    "score_b": eb["score_b"],
                    "score_c": ec["score_c"],
                    "actual_dir": sess_dir,
                    "actual_move": sess_move,
                    "actual_high": act_high,
                    "actual_low": act_low,
                    "actual_close": target_close,
                }
            )

        # Ingest completed historical daily bar strictly AFTER day_i prediction is emitted!
        d_sgn = 1 if rec["_target_dir"] == "XAU" else (-1 if rec["_target_dir"] == "USD" else 0)
        open_safe = max(1.0, float(rec["today_open"]))
        fav_up_atr = max(0.0, act_high - open_safe) / float(rec["atr14"])
        fav_dn_atr = max(0.0, open_safe - act_low) / float(rec["atr14"])
        fav_up_pct = max(0.0, act_high - open_safe) / open_safe
        fav_dn_pct = max(0.0, open_safe - act_low) / open_safe
        archive.add_completed_day(
            date_iso=rec["date"],
            weekday_up=rec["weekday_up"],
            sig=sig,
            direction_sign=d_sgn,
            move_dollars=rec["_target_move"],
            fav_up_atr=fav_up_atr,
            fav_dn_atr=fav_dn_atr,
            is_noise_flat=rec["_is_noise_flat"],
            fav_up_pct=fav_up_pct,
            fav_dn_pct=fav_dn_pct,
            hist_atr14=float(rec["atr14"]),
        )

    # STRICT TRAIN-ONLY CALIBRATION (`2012-01-01` .. `2024-12-31`)
    train_only = [s for s in day_states if s["split"] == "TRAIN" and s["actual_dir"] in ("XAU", "USD")]
    frozen_cfg: FrozenGoldCbaConfig = fit_gold_cba_on_train(train_only)

    subset_violations = 0
    evaluated_rows: list[dict[str, Any]] = []
    veto_counts: dict[str, int] = {}

    for st in day_states:
        cba = evaluate_xauusd_gold_cba(st["rec"], st["ea"], st["eb"], st["sig"], st["ec"], frozen_cfg)
        subset_violations += verify_subset_invariant(cba)
        for vf in cba["veto_flags"]:
            veto_counts[vf] = veto_counts.get(vf, 0) + 1

        pred_dir = cba["pick"] if cba["pick"] else cba["raw_dir"]
        bars_1h = bars_1h_map.get(st["date"]) if bars_1h_map is not None else None
        sim = simulate_session_execution(
            direction=pred_dir,
            entry=float(cba["entry"]),
            sl=float(cba["sl"]),
            tp1=float(cba["tp1"]),
            tp2=float(cba["tp2"]),
            atr14=float(st["rec"]["atr14"]),
            daily_high=float(st["actual_high"]),
            daily_low=float(st["actual_low"]),
            daily_close=float(st["actual_close"]),
            bars_1h=bars_1h,
            tp3=float(cba["tp3"]),
        )

        evaluated_rows.append(
            {
                **cba,
                "pred_dir": pred_dir,
                "actual_dir": st["actual_dir"],
                "actual_move": round(st["actual_move"], 2),
                "actual_high": round(st["actual_high"], 2),
                "actual_low": round(st["actual_low"], 2),
                "actual_close": round(st["actual_close"], 2),
                "score_a": st["score_a"],
                "score_b": st["score_b"],
                "score_c": st["score_c"],
                "tw_pct": st["ec"]["tw_pct"],
                "w_disp_pct": st["ec"]["w_disp_pct"],
                "twin_count": st["ec"]["twin_count"],
                "up_twins": st["ec"]["up_twins"],
                "down_twins": st["ec"]["down_twins"],
                "in_golden_zone": st["ec"]["in_golden_zone"],
                "has_lock3": st["ec"]["has_lock3"],
                "lock3_peak": st["ec"]["lock3_peak"],
                "geo_fig": st["sig"]["geo_fig"],
                "geo_name": st["sig"]["geo_name"],
                "dec_geo": st["sig"]["dec_geo"],
                "ruler_geo": st["sig"]["ruler_geo"],
                "zero_elep_cnt": st["sig"]["zero_elep_cnt"],
                **sim,
            }
        )

    splits_summary: dict[str, dict[str, Any]] = {}
    for sp in ("TRAIN", "VAL", "OOS"):
        sp_rows = [r for r in evaluated_rows if r["split"] == sp]
        splits_summary[sp] = {
            "ALL_RAW": _enrich_legacy_keys(summarize_dual_metrics(sp_rows, lambda r: True)),
            "TIER_A": _enrich_legacy_keys(summarize_dual_metrics(sp_rows, lambda r: r["is_a"])),
            "TIER_B": _enrich_legacy_keys(summarize_dual_metrics(sp_rows, lambda r: r["is_b"])),
            "TIER_C": _enrich_legacy_keys(summarize_dual_metrics(sp_rows, lambda r: r["is_c"])),
        }

    distinct_rate = round(distinct_both_ok * 100.0 / max(1, total_roster_days), 2)
    lock3_all_rate = round(
        sum(1 for r in evaluated_rows if r["has_lock3"]) * 100.0 / max(1, len(evaluated_rows)), 2
    )
    lock3_peak_all_rate = round(
        sum(1 for r in evaluated_rows if r["lock3_peak"]) * 100.0 / max(1, len(evaluated_rows)), 2
    )

    return {
        "anchor_key": anchor_key,
        "anchor_label": ANCHOR_CANDIDATES[anchor_key]["label"],
        "rounding_mode": rounding_mode,
        "use_intraday_session_open": use_intraday_session_open,
        "elapsed_sec": round(time.time() - t0, 2),
        "frozen_cfg": asdict(frozen_cfg),
        "subset_violations": subset_violations,
        "lookahead_leakage": archive.lookahead_violations,
        "total_twin_queries": archive.total_queries,
        "distinct_8v8_rate_pct": distinct_rate,
        "price_free_twin_signature": bool(price_free_ok),
        "max_sig_sums": {"ELE_P": max_sig_sums[0], "FINAL": max_sig_sums[1], "PRIM": max_sig_sums[2]},
        "lock3_rate_pct": lock3_all_rate,
        "lock3_peak_rate_pct": lock3_peak_all_rate,
        "veto_counts": veto_counts,
        "splits": splits_summary,
        "rows": evaluated_rows,
        "archive_full": archive,
        "archive_before_latest": archive_before_latest,
    }


def build_live_signal_card(
    latest_rec: dict[str, Any],
    archive: HistoricalTwinArchive,
    frozen_cfg: FrozenGoldCbaConfig,
    anchor_key: str = LOCKED_ANCHOR_KEY,
    rounding_mode: str = LOCKED_ROUNDING_MODE,
    spot_override: float | None = None,
) -> dict[str, Any]:
    """
    Generate full diagnostic & REST-ready Live Signal Card (`need_hits=True` for full aspect breakdown).
    `archive` MUST contain strictly `< latest_rec["date"]` rows!
    """
    ea = evaluate_engine_a(
        d_iso=latest_rec["date"],
        today_open=latest_rec["today_open"],
        prev_close=latest_rec["prev_close"],
        dxy_prev_close=latest_rec["dxy_prev_close"],
        anchor_key=anchor_key,
        rounding_mode=rounding_mode,
        need_hits=True,
    )
    eb = evaluate_price_roster(latest_rec, ea, rounding_mode=rounding_mode, need_hits=True)
    sig = build_day_12col_signature(ea)
    ec = archive.evaluate_twin_and_gaussian(
        target_date=latest_rec["date"],
        weekday_up=latest_rec["weekday_up"],
        sig=sig,
        entry_open=eb["entry_open"],
        atr14=latest_rec["atr14"],
        s1=eb["s1"],
        r1=eb["r1"],
        fallback_dir=eb["dir_b"],
        s2=eb["s2"],
        r2=eb["r2"],
    )
    cba = evaluate_xauusd_gold_cba(latest_rec, ea, eb, sig, ec, frozen_cfg)

    spot_price = round(
        float(spot_override if spot_override is not None else latest_rec.get("_target_close", eb["entry_open"])),
        2,
    )
    bias_dir = cba["pick"] if cba["pick"] else cba["raw_dir"]

    return {
        # Production v4 REST schema fields (`GET /api/signals/today`)
        "sessionDate": latest_rec["date"],
        "weekday": latest_rec["weekday"],
        "candleOpenUtc": ea["utc_iso"],
        "spotPrice": spot_price,
        "prevClose": round(float(latest_rec["prev_close"]), 2),
        "atr14": round(float(latest_rec["atr14"]), 2),
        "ema20": round(float(latest_rec["ema20"]), 2),
        "ema50": round(float(latest_rec["ema50"]), 2),
        "dxyPrevClose": round(float(latest_rec["dxy_prev_close"]), 3),
        "roomToRunUsd": eb["room_to_run"],
        "macroRegime": latest_rec["macro_trend_regime"],
        "tier": cba["tier"] or "HOLD",
        "tierBadge": cba["tier_badge"],
        "isC": cba["is_c"],
        "isB": cba["is_b"],
        "isA": cba["is_a"],
        "direction": cba["action"],
        "biasDirection": bias_dir,
        "winProbPct": cba["win_prob_pct"],
        "kellyMultiplier": cba["kelly_multiplier"],
        "entryPrice": cba["entry"],
        "sl": cba["sl"],
        "tp1": cba["tp1"],
        "tp2": cba["tp2"],
        "tp3": cba["tp3"],
        "rrRatio": cba["rr_ratio"],
        "vetoReasons": cba["veto_flags"],
        "vetoShield6": cba["veto_shield_6"],
        "kalshiBracket": cba["kalshi_bracket"],
        "signature12Col": {
            "priceFree": sig["price_free"],
            "PRIM_FAWE": list(sig["prim"]),
            "FINAL_FAWE": list(sig["final"]),
            "ELEP_FAWE": list(sig["elep"]),
            "zeroElepCnt": sig["zero_elep_cnt"],
            "GEO_FIG": f"{sig['geo_fig']} ({sig['geo_name']})",
            "geoCode": sig["geo_fig"],
            "geoName": sig["geo_name"],
            "DEC_GEO": sig["dec_geo"],
            "RULER_GEO": sig["ruler_geo"],
            "step3Hits": sig["step3_hits"],
            "celestialDecanTags": sig["celestial_decan_tags"],
        },
        "homeXauRoster": eb["xau_roster"]["players"],
        "awayUsdRoster": eb["usd_roster"]["players"],
        "homeTeamMeta": {
            "pole": "HOME XAU 77° (17° GEMINI • AIR)",
            "domElement": eb["xau_roster"]["dom_element"],
            "wFawe": eb["xau_roster"]["w_fawe"],
            "efTot": list(eb["xau_roster"]["ef_tot"]),
            "resonanceNorm": eb["xau_roster"]["resonance_norm"],
            "starterResonanceNorm": eb["xau_roster"]["starter_resonance_norm"],
            "elemLeads": eb["xau_elem_leads"],
        },
        "awayTeamMeta": {
            "pole": "AWAY USD 70° (10° GEMINI • AIR)",
            "domElement": eb["usd_roster"]["dom_element"],
            "wFawe": eb["usd_roster"]["w_fawe"],
            "efTot": list(eb["usd_roster"]["ef_tot"]),
            "resonanceNorm": eb["usd_roster"]["resonance_norm"],
            "starterResonanceNorm": eb["usd_roster"]["starter_resonance_norm"],
            "elemLeads": eb["usd_elem_leads"],
        },
        "matchedTwins": ec["matched_twins"],
        "twinConsensus": {
            "twinCount": ec["twin_count"],
            "upTwins": ec["up_twins"],
            "downTwins": ec["down_twins"],
            "votes": f"{ec['up_twins']} XAU - {ec['down_twins']} USD ({ec['tw_pct']}%)",
            "dispPct": ec["tw_pct"],
            "weightedDispPct": ec["w_disp_pct"],
            "twinWinProbPct": round(ec["twin_win_prob"] * 100.0, 1),
            "strongAvgDominance": ec["strong_avg_dominance"],
            "majAvgAtrMove": ec["maj_avg_atr_move"],
            "minAvgAtrMove": ec["min_avg_atr_move"],
            "inGoldenZone": ec["in_golden_zone"],
            "hasLock3": ec["has_lock3"],
            "lock3Cols": ec["lock3_cols"],
            "lock3Peak": ec["lock3_peak"],
            "highPurityLock3Peak": ec["high_purity_lock3_peak"],
            "activeLockElem": ec["active_lock_elem"],
            "decan11": ec["decan_11"],
            "decan00": ec["decan_00"],
            "fp1_4": list(ec["fp"]),
            "ff1_4": list(ec["ff"]),
            "gaussScore": ec["gauss_score"],
            "tpAtrs": list((ec["tp1_atr"], ec["tp2_atr"], ec["tp3_atr"])),
        },
        # Backwards-compatible keys preserved for CLI report & invariant unit tests
        "date": latest_rec["date"],
        "anchor_key": anchor_key,
        "anchor_label": ANCHOR_CANDIDATES[anchor_key]["label"],
        "utc_timestamp": ea["utc_iso"],
        "rounding_mode": rounding_mode,
        "distinct_8v8_verified": bool(
            _roster_is_distinct_8(eb["xau_roster"]) and _roster_is_distinct_8(eb["usd_roster"])
        ),
        "price_free_twin_signature": bool(sig["price_free"]),
        "entity_locks": {
            "DEG1_GOLD": f"{DEG1_GOLD}° ({zodiac_from_deg(DEG1_GOLD)[0]} • WATER)",
            "DEG2_XAUUSD": f"{DEG2_XAUUSD}° ({zodiac_from_deg(DEG2_XAUUSD)[0]} • FIRE)",
            "HOME_XAU_DEG": f"{HOME_XAU_DEG}° ({zodiac_from_deg(HOME_XAU_DEG)[0]} • AIR)",
            "AWAY_USD_DEG": f"{AWAY_USD_DEG}° ({zodiac_from_deg(AWAY_USD_DEG)[0]} • AIR)",
            "F1": ea["d_ctx"]["F1"],
            "F2": ea["d_ctx"]["F2"],
            "F3_HOME": ea["d_ctx"]["F3_HOME"],
            "F3_AWAY": ea["d_ctx"]["F3_AWAY"],
        },
        "ephemeris": {
            "asc": f"{ea['asc_deg']}° ({zodiac_from_deg(ea['asc_deg'])[0]})",
            "mc": f"{ea['mc_deg']}° ({zodiac_from_deg(ea['mc_deg'])[0]})",
            "pof": f"{ea['pof_deg']}° ({zodiac_from_deg(ea['pof_deg'])[0]})",
            "planetary_hour_ruler": ea["planetary_hour_ruler"],
            "planetary_hour_element": ea["planetary_hour_element"],
            "hits_gold_107": ea["hits_gold"],
            "hits_xau_77": ea["hits_xau"],
            "hits_usd_70": ea["hits_usd"],
            "hits_open": ea["hits_open"],
        },
        "market_context": {
            "today_open": latest_rec["today_open"],
            "prev_close": latest_rec["prev_close"],
            "pdh": latest_rec["pdh"],
            "pdl": latest_rec["pdl"],
            "atr14": round(latest_rec["atr14"], 2),
            "ema20": round(latest_rec["ema20"], 2),
            "ema50": round(latest_rec["ema50"], 2),
            "macro_trend_regime": latest_rec["macro_trend_regime"],
            "dxy_prev_close": round(latest_rec["dxy_prev_close"], 3),
            "room_to_run_r2s2": eb["room_to_run"],
        },
        "home_xau_roster": eb["xau_roster"],
        "away_usd_roster": eb["usd_roster"],
        "signature_12col": {
            "price_free": sig["price_free"],
            "PRIM_FAWE": list(sig["prim"]),
            "FINAL_FAWE": list(sig["final"]),
            "ELEP_FAWE": list(sig["elep"]),
            "zeroElepCnt": sig["zero_elep_cnt"],
            "GEO_FIG": f"{sig['geo_fig']} ({sig['geo_name']})",
            "DEC_GEO": sig["dec_geo"],
            "RULER_GEO": sig["ruler_geo"],
            "step3_hits": sig["step3_hits"],
            "celestial_decan_tags": sig["celestial_decan_tags"],
        },
        "twins_and_gaussian": {
            "twin_count": ec["twin_count"],
            "votes": f"{ec['up_twins']} XAU - {ec['down_twins']} USD ({ec['tw_pct']}%)",
            "w_disp_pct": ec["w_disp_pct"],
            "strong_avg_dominance": ec["strong_avg_dominance"],
            "twin_dates": ec["twin_dates"],
            "in_golden_zone": ec["in_golden_zone"],
            "has_lock3": ec["has_lock3"],
            "lock3_cols": ec["lock3_cols"],
            "lock3_peak": ec["lock3_peak"],
            "high_purity_lock3_peak": ec["high_purity_lock3_peak"],
            "decan_11": ec["decan_11"],
            "decan_00": ec["decan_00"],
            "fp1_4": list(ec["fp"]),
            "ff1_4": list(ec["ff"]),
            "gauss_score": ec["gauss_score"],
            "tp_atrs": list((ec["tp1_atr"], ec["tp2_atr"], ec["tp3_atr"])),
        },
        "signal": cba,
    }


def _build_next_session_preview_rec(latest_rec: dict[str, Any], live_spot: float | None = None) -> dict[str, Any]:
    """
    Construct a causal rollover preview record for the next trading weekday after `latest_rec["date"]`
    using `latest_rec`'s completed OHLC as `Prev_Day` and `live_spot` (or `Prev_Close`) as indicative open.
    """
    y, m, d = map(int, latest_rec["date"].split("-"))
    nxt_dt = datetime.date(y, m, d) + datetime.timedelta(days=1)
    while nxt_dt.weekday() >= 5:
        nxt_dt += datetime.timedelta(days=1)

    roll_open = round(float(live_spot if live_spot is not None else latest_rec["_target_close"]), 2)
    prev_h = max(float(latest_rec["today_open"]), float(latest_rec["_target_high"]), roll_open)
    prev_l = min(float(latest_rec["today_open"]), float(latest_rec["_target_low"]), roll_open)
    prev_c = roll_open
    atr = float(latest_rec["atr14"])

    a20, a50 = 2.0 / 21.0, 2.0 / 51.0
    new_ema20 = a20 * prev_c + (1.0 - a20) * float(latest_rec["ema20"])
    new_ema50 = a50 * prev_c + (1.0 - a50) * float(latest_rec["ema50"])
    new_slope = float(latest_rec["ema20_slope"]) * 0.8 + (new_ema20 - float(latest_rec["ema20"])) / max(1.0, atr)
    new_ret20 = float(latest_rec["rolling_20d_ret"]) + (prev_c - float(latest_rec["prev_close"])) / max(1.0, float(latest_rec["prev_close"]))
    if (prev_c > new_ema20 > new_ema50) and (new_slope > 0.0) and (new_ret20 > 0.0):
        new_regime = "STRONG_BULL"
    elif (prev_c < new_ema20 < new_ema50) and (new_slope < 0.0) and (new_ret20 < 0.0):
        new_regime = "STRONG_BEAR"
    else:
        new_regime = "NEUTRAL"

    nxt = dict(latest_rec)
    nxt.update(
        {
            "date": nxt_dt.strftime("%Y-%m-%d"),
            "weekday": nxt_dt.strftime("%A"),
            "weekday_up": nxt_dt.strftime("%A").upper(),
            "split": "LIVE_PREVIEW",
            "today_open": roll_open,
            "prev_open": float(latest_rec["today_open"]),
            "pdh": prev_h,
            "pdl": prev_l,
            "prev_close": prev_c,
            "_target_close": roll_open,
            "ema20": new_ema20,
            "ema50": new_ema50,
            "ema20_slope": new_slope,
            "rolling_20d_ret": new_ret20,
            "macro_trend_regime": new_regime,
            "prev2_high": float(latest_rec["pdh"]),
            "prev2_low": float(latest_rec["pdl"]),
            "pwh": max(float(latest_rec["pwh"]), prev_h),
            "pwl": min(float(latest_rec["pwl"]), prev_l),
            "high_10d": max(float(latest_rec["high_10d"]), prev_h),
            "low_10d": min(float(latest_rec["low_10d"]), prev_l),
            "high_20d": max(float(latest_rec["high_20d"]), prev_h),
            "low_20d": min(float(latest_rec["low_20d"]), prev_l),
            "high_60d": max(float(latest_rec["high_60d"]), prev_h),
            "low_60d": min(float(latest_rec["low_60d"]), prev_l),
            "prev_ret_1": (roll_open - float(latest_rec["today_open"])) / max(1.0, atr),
            "open_gap_atr": 0.0,
        }
    )
    return nxt


def _fetch_live_yahoo_spot_quote() -> tuple[float | None, str]:
    """Attempt quick non-blocking live quote fetch for Gold (`GC=F`) with safe fallback."""
    url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=1h&range=2d"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=(1.2, 1.5))
        if r.status_code == 200:
            meta = r.json().get("chart", {}).get("result", [{}])[0].get("meta", {})
            rmp = meta.get("regularMarketPrice")
            if rmp is not None and float(rmp) > 500.0:
                return round(float(rmp), 2), "YAHOO_LIVE_QUOTE"
    except Exception:
        pass
    return None, "CACHED_VERIFIED_FEED"


def refresh_live_signal_shard_only(try_network_fetch: bool = True) -> dict[str, Any]:
    """
    Fast O(1) live refresh used by `POST /api/refresh`:
    loads pre-indexed `xauusd_twin_archive.pkl` (rebuilding once if absent) + optional live spot quote,
    and recomputes `shards/latest_xauusd_signals.json` in `< 0.2s` while keeping `RSS < 90 MiB`.
    """
    live_spot = None
    feed_source = "CACHED_VERIFIED_FEED"
    if try_network_fetch:
        live_spot, feed_source = _fetch_live_yahoo_spot_quote()

    summary_path = os.path.join(DATA_DIR, "xauusd_backtest_summary.json")
    with open(summary_path, encoding="utf-8") as f:
        frozen_cfg = FrozenGoldCbaConfig(**json.load(f)["frozen_cfg"])

    pkl_path = os.path.join(DATA_DIR, "xauusd_twin_archive.pkl")
    latest_rec: dict[str, Any] | None = None
    archive_before_latest: HistoricalTwinArchive | None = None
    archive: HistoricalTwinArchive | None = None

    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as pf:
                cached_bundle = pickle.load(pf)
            latest_rec = cached_bundle["latest_rec"]
            archive_before_latest = cached_bundle["archive_before_latest"]
            archive = cached_bundle["archive_full"]
        except Exception:
            latest_rec = None

    if latest_rec is None or archive_before_latest is None or archive is None:
        df_daily, _df_1h, df_sess = fetch_and_cache_market_data(force_refresh=False)
        records = build_zero_leakage_daily_records(df_daily, df_sess)
        archive = HistoricalTwinArchive()
        n_recs = len(records)
        for idx_r, rec in enumerate(records):
            if idx_r == n_recs - 1:
                archive_before_latest = copy.deepcopy(archive)
            entry_open = float(rec["today_open"])
            ea = evaluate_engine_a(
                rec["date"], entry_open, rec["prev_close"], rec["dxy_prev_close"],
                LOCKED_ANCHOR_KEY, LOCKED_ROUNDING_MODE, need_hits=False,
            )
            sig = build_day_12col_signature(ea)
            d_sgn = 1 if rec["_target_dir"] == "XAU" else (-1 if rec["_target_dir"] == "USD" else 0)
            act_h = max(entry_open, float(rec["_target_high"]))
            act_l = min(entry_open, float(rec["_target_low"]))
            archive.add_completed_day(
                date_iso=rec["date"],
                weekday_up=rec["weekday_up"],
                sig=sig,
                direction_sign=d_sgn,
                move_dollars=rec["_target_move"],
                fav_up_atr=max(0.0, act_h - entry_open) / float(rec["atr14"]),
                fav_dn_atr=max(0.0, entry_open - act_l) / float(rec["atr14"]),
                is_noise_flat=rec["_is_noise_flat"],
                fav_up_pct=max(0.0, act_h - entry_open) / max(1.0, entry_open),
                fav_dn_pct=max(0.0, entry_open - act_l) / max(1.0, entry_open),
                hist_atr14=float(rec["atr14"]),
            )
        latest_rec = records[-1]
        assert archive_before_latest is not None
        with open(pkl_path, "wb") as pf:
            pickle.dump(
                {
                    "latest_rec": latest_rec,
                    "archive_before_latest": archive_before_latest,
                    "archive_full": archive,
                },
                pf,
            )

    live_card = build_live_signal_card(
        latest_rec=latest_rec,
        archive=archive_before_latest,
        frozen_cfg=frozen_cfg,
        spot_override=live_spot,
    )
    next_rec = _build_next_session_preview_rec(latest_rec, live_spot=live_spot)
    next_card = build_live_signal_card(
        latest_rec=next_rec,
        archive=archive,
        frozen_cfg=frozen_cfg,
        spot_override=live_spot,
    )
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "service": "XAUUSD-FUSION-SIGNAL",
        "version": "4.0.0",
        "anchor": LOCKED_ANCHOR_KEY,
        "feedSource": feed_source,
        "lastUpdatedUtc": now_iso,
        **live_card,
        "liveSession": live_card,
        "nextSessionPreview": next_card,
        "latest_session": live_card,
        "next_session_preview": next_card,
    }


def _load_prior_versions_for_comparison() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    v1_splits = None
    v3_splits = None
    p1_path = os.path.join(DATA_DIR, "phase1_anchor_benchmark.json")
    v3_path = os.path.join(DATA_DIR, "v3_backtest_summary.json")
    if os.path.exists(p1_path):
        with open(p1_path, encoding="utf-8") as f:
            p1_list = json.load(f)
        for item in p1_list:
            if item.get("anchor_key") == LOCKED_ANCHOR_KEY and item.get("rounding_mode") == LOCKED_ROUNDING_MODE:
                v1_splits = item.get("splits")
                break
    if os.path.exists(v3_path):
        with open(v3_path, encoding="utf-8") as f:
            v3_splits = json.load(f).get("splits")
    return v1_splits, v3_splits


def _print_signal_card_block(title_prefix: str, card: dict[str, Any]) -> None:
    s = card["signal"]
    mc = card["market_context"]
    ep = card["ephemeris"]
    sg = card["signature_12col"]
    tw = card["twins_and_gaussian"]

    print("\n" + "=" * 126)
    print(
        f"  {title_prefix} • {card['date']} ({card['weekday']}) • {card['utc_timestamp']} • "
        f"Distinct8v8={card['distinct_8v8_verified']} • PriceFreeSig={card['price_free_twin_signature']}"
    )
    print("=" * 126)
    print(
        f"  Entity Locks : DEG1={DEG1_GOLD}° (17° CANCER) | DEG2={DEG2_XAUUSD}° (27° LEO) | "
        f"HOME XAU={HOME_XAU_DEG}° (17° GEMINI) | AWAY USD={AWAY_USD_DEG}° (10° GEMINI)"
    )
    print(
        f"  Topocentric  : ASC={ep['asc']} | MC={ep['mc']} | POF={ep['pof']} | "
        f"Planetary Hour={ep['planetary_hour_ruler']} ({ep['planetary_hour_element']})"
    )
    print(
        f"  Market @ T_0 : Open=${mc['today_open']:.2f} | PrevClose=${mc['prev_close']:.2f} | "
        f"EMA20=${mc['ema20']:.2f} | EMA50=${mc['ema50']:.2f} | MacroRegime={mc['macro_trend_regime']} | "
        f"ATR14=${mc['atr14']:.2f} | Room(R2-S2)=${mc['room_to_run_r2s2']:+.2f}"
    )

    print("\n  [HOME TEAM: XAU 8-Player Breakout-Aware Support Roster (Strictly Distinct Levels <= Open)]")
    print(f"  {'Pos':4s} | {'Role':15s} | {'Level ($)':10s} | {'eff_w':6s} | {'Deg & Zodiac':21s} | {'Elem':5s} | {'Decan & Ruler':27s} | {'GEO':4s} | {'uRes':7s} | {'Aspects'}")
    print("  " + "-" * 122)
    for p in card["home_xau_roster"]["players"]:
        d_z = f"{p['deg']:3d}° ({p['zodiac']})"
        dec_r = f"D{p['decan']:02d} {p['decan_label']} ({p['decan_ruler']})"
        asp_s = ",".join(p["aspect_hits"]) if p["aspect_hits"] else "—"
        print(f"  {p['pos']:4s} | {p['role']:15s} | ${p['level']:9.2f} | {p['eff_w']:6.3f} | {d_z:21s} | {p['element']:5s} | {dec_r:27s} | {p['gf']:4s} | {p['unit_resonance']:+7.3f} | {asp_s}")

    print("\n  [AWAY TEAM: USD 8-Player Breakout-Aware Resistance Roster (Strictly Distinct Levels >= Open)]")
    print(f"  {'Pos':4s} | {'Role':15s} | {'Level ($)':10s} | {'eff_w':6s} | {'Deg & Zodiac':21s} | {'Elem':5s} | {'Decan & Ruler':27s} | {'GEO':4s} | {'uRes':7s} | {'Aspects'}")
    print("  " + "-" * 122)
    for p in card["away_usd_roster"]["players"]:
        d_z = f"{p['deg']:3d}° ({p['zodiac']})"
        dec_r = f"D{p['decan']:02d} {p['decan_label']} ({p['decan_ruler']})"
        asp_s = ",".join(p["aspect_hits"]) if p["aspect_hits"] else "—"
        print(f"  {p['pos']:4s} | {p['role']:15s} | ${p['level']:9.2f} | {p['eff_w']:6.3f} | {d_z:21s} | {p['element']:5s} | {dec_r:27s} | {p['gf']:4s} | {p['unit_resonance']:+7.3f} | {asp_s}")

    print("\n  [100% Price-Free Pure Celestial 12-Col Signature, Recency-ATR Twins & 6-Layer Veto Shield]")
    print(
        f"  12-Col FAWE  : PRIM={sg['PRIM_FAWE']} | FINAL={sg['FINAL_FAWE']} | "
        f"ELE-P={sg['ELEP_FAWE']} (zeroElepCnt={sg['zeroElepCnt']})"
    )
    print(f"  Sky Anchors  : {', '.join(sg['celestial_decan_tags'])} | Hits: {', '.join(sg['step3_hits'])}")
    print(f"  Parity Codes : GEO-FIG={sg['GEO_FIG']} | DEC-GEO={sg['DEC_GEO']} | RULER-GEO={sg['RULER_GEO']}")
    print(
        f"  Shared Twins : {tw['votes']} (Weighted={tw['w_disp_pct']}%) | Dom={tw['strong_avg_dominance']} | "
        f"LOCK3={tw['lock3_cols']} | LOCK3_PEAK={tw['lock3_peak']} | Dec1/1={tw['decan_11']}"
    )
    print(f"  Twin Dates   : {', '.join(tw['twin_dates']) if tw['twin_dates'] else 'None'}")
    print(f"  6-Lyr Veto   : {'CLEAN (0/6 Vetoes Triggered)' if not s['veto_flags'] else ', '.join(s['veto_flags'])}")
    print("  " + "-" * 122)
    dir_disp = "LONG XAU" if (s["pick"] or s["raw_dir"]) == "XAU" else "SHORT XAU"
    print(
        f"  VERDICT      : {s['tier_badge']} ({s['kelly_multiplier']:.2f}x Kelly) | Action: {s['action']} "
        f"(Bias: {dir_disp} • WinProb {s['win_prob_pct']}%) | RR={s['rr_ratio']:.2f} | "
        f"Kalshi: {s['kalshi_bracket']['recommendedContract']}"
    )
    print(
        f"  LEVELS ($)   : Entry = ${s['entry']:.2f} | SL = ${s['sl']:.2f} | "
        f"TP1 = ${s['tp1']:.2f} | TP2 = ${s['tp2']:.2f} | TP3 = ${s['tp3']:.2f}"
    )
    print("=" * 126)


def print_v4_report(
    winner_run: dict[str, Any],
    live_card: dict[str, Any],
    next_card: dict[str, Any],
    v1_splits: dict[str, Any] | None,
    v3_splits: dict[str, Any] | None,
) -> None:
    print("\n" + "=" * 126)
    print(
        f"  XAUUSD-FUSION-SIGNAL v4.0.0 TWIN-FIRST PRODUCTION ENGINE • {winner_run['anchor_label']} | "
        f"Rounding: round(P) % 360 ({winner_run['rounding_mode']})"
    )
    print(
        f"  INVARIANT AUDIT: subsetViolations == {winner_run['subset_violations']} | "
        f"lookAheadLeakage == {winner_run['lookahead_leakage']} | "
        f"distinct8v8Roster == {winner_run['distinct_8v8_rate_pct']:.2f}% | "
        f"priceFreeTwinSignature == {winner_run['price_free_twin_signature']}"
    )
    print(
        f"  PURE CELESTIAL TWIN LOCKS: LOCK3 == {winner_run['lock3_rate_pct']:.1f}% | "
        f"LOCK3_PEAK == {winner_run['lock3_peak_rate_pct']:.1f}%"
    )
    print("=" * 126)

    for sp_code, sp_title, yrs in [
        ("TRAIN", "TRAIN (2012-01-01 .. 2024-12-31 • Conservative Daily OHLC)", 13.0),
        ("VAL",   "VALIDATION (2025-01-01 .. 2025-12-31 • Chronological 1h Path)", 1.0),
        ("OOS",   "BLIND OUT-OF-SAMPLE (2026-01-01 .. 2026-11-26 YTD • Chronological 1h Path)", 0.90),
    ]:
        print(f"\n  [{sp_title}]")
        print(
            f"  {'Tier':18s} | {'Trd/Yr':7s} | {'LOCK3%':7s} | "
            f"{'Raw Close W-L':14s} | {'Raw Win%':8s} | {'Raw PF':6s} | {'Raw Exp$':9s} || "
            f"{'TP1-vs-SL W-L':14s} | {'Exec Win%':9s} | {'Exec PF':7s} | {'Exec Exp$':9s}"
        )
        print("  " + "-" * 124)
        sp_d = winner_run["splits"][sp_code]
        for tk, tlabel in [
            ("TIER_C", "👑 Tier C (1.20x)"),
            ("TIER_B", "⭐ Tier B (1.00x)"),
            ("TIER_A", "🟢 Tier A (0.80x)"),
            ("ALL_RAW", "⚪ All Sessions"),
        ]:
            m = sp_d[tk]
            trd_yr = m["n"] / yrs
            rw_wl = f"{m['n']}({m['raw_wins']}-{m['raw_losses']})"
            ex_wl = f"{m['n']}({m['exec_wins']}-{m['exec_losses']})"
            print(
                f"  {tlabel:18s} | {trd_yr:6.1f}  | {m['lock3_rate']:5.1f}%  | "
                f"{rw_wl:14s} | {m['raw_win_rate']:6.2f}%  | {m['raw_pf']:6.2f} | ${m['raw_exp_usd']:+7.2f} || "
                f"{ex_wl:14s} | {m['exec_win_rate']:7.2f}%  | {m['exec_pf']:7.2f} | ${m['exec_exp_usd']:+7.2f}"
            )

    if v1_splits and v3_splits:
        print("\n" + "=" * 126)
        print("  SIDE-BY-SIDE EVOLUTION BENCHMARK: v1 (Phase 1) vs v3 (OLS Hybrid Lock) vs v4 (Non-Parametric Twin-First Consensus)")
        print("=" * 126)
        for sp_code, sp_lbl, yrs in [("TRAIN", "TRAIN 12-24", 13.0), ("VAL", "VAL 2025", 1.0), ("OOS", "OOS 2026 YTD", 0.90)]:
            print(f"\n  [{sp_lbl}]")
            print(f"  {'Tier':12s} | {'v1 RawWin% (Trd/Yr, PF, Exp$)':35s} | {'v3 Raw|Exec Win% (Trd/Yr, ExecPF, Exp$)':42s} | {'v4 Raw|Exec Win% (Trd/Yr, ExecPF, Exp$)':42s}")
            print("  " + "-" * 126)
            for tk, tshort in [("TIER_C", "👑 Tier C"), ("TIER_B", "⭐ Tier B"), ("TIER_A", "🟢 Tier A")]:
                m1 = v1_splits[sp_code][tk]
                m3 = v3_splits[sp_code][tk]
                m4 = winner_run["splits"][sp_code][tk]
                s1 = f"{m1['win_rate']:5.1f}% ({m1['n']/yrs:4.1f}/y, PF {m1['pf']:4.2f}, ${m1['expectancy_usd']:+6.2f})"
                s3 = f"{m3['raw_win_rate']:5.1f}%|{m3['exec_win_rate']:5.1f}% ({m3['n']/yrs:4.1f}/y, PF {m3['exec_pf']:4.2f}, ${m3['exec_exp_usd']:+6.2f})"
                s4 = f"{m4['raw_win_rate']:5.1f}%|{m4['exec_win_rate']:5.1f}% ({m4['n']/yrs:4.1f}/y, PF {m4['exec_pf']:4.2f}, ${m4['exec_exp_usd']:+6.2f})"
                print(f"  {tshort:12s} | {s1:35s} | {s3:42s} | {s4:42s}")

    print("\n  [6-Layer Veto Shield Trap & Counter-Macro Rejections Across 2012-2026]")
    for vk, vc in sorted(winner_run["veto_counts"].items()):
        print(f"    - {vk:30s}: {vc:4d} days shielded from Tier B / Tier C")

    _print_signal_card_block("v4 LIVE LATEST COMPLETED SESSION DIAGNOSTIC CARD", live_card)
    _print_signal_card_block("v4 UPCOMING ROLLOVER SESSION PREVIEW CARD (INDICATIVE OPEN = PREV CLOSE)", next_card)


def run_full_pipeline() -> dict[str, Any]:
    df_daily, df_1h, df_sess = fetch_and_cache_market_data(force_refresh=False)
    records = build_zero_leakage_daily_records(df_daily, df_sess)
    bars_1h_map = group_1h_bars_by_session(df_1h)

    winner_run = run_walk_forward_for_config(
        records=records,
        anchor_key=LOCKED_ANCHOR_KEY,
        rounding_mode=LOCKED_ROUNDING_MODE,
        bars_1h_map=bars_1h_map,
        use_intraday_session_open=False,
    )

    frozen_cfg = FrozenGoldCbaConfig(**winner_run["frozen_cfg"])
    latest_rec = records[-1]
    assert winner_run["archive_before_latest"] is not None

    live_card = build_live_signal_card(
        latest_rec=latest_rec,
        archive=winner_run["archive_before_latest"],
        frozen_cfg=frozen_cfg,
        anchor_key=LOCKED_ANCHOR_KEY,
        rounding_mode=LOCKED_ROUNDING_MODE,
    )

    next_rec = _build_next_session_preview_rec(latest_rec)
    next_card = build_live_signal_card(
        latest_rec=next_rec,
        archive=winner_run["archive_full"],
        frozen_cfg=frozen_cfg,
        anchor_key=LOCKED_ANCHOR_KEY,
        rounding_mode=LOCKED_ROUNDING_MODE,
    )

    v1_splits, v3_splits = _load_prior_versions_for_comparison()
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SHARDS_DIR, exist_ok=True)

    summary_export = {
        "service": "XAUUSD-FUSION-SIGNAL",
        "version": "4.0.0",
        "versionTag": "v4",
        "lastUpdatedUtc": now_iso,
        "locked_anchor": winner_run["anchor_key"],
        "locked_anchor_label": winner_run["anchor_label"],
        "locked_rounding": winner_run["rounding_mode"],
        "frozen_cfg": winner_run["frozen_cfg"],
        "subsetViolations": winner_run["subset_violations"],
        "lookAheadLeakage": winner_run["lookahead_leakage"],
        "distinct8v8RosterPct": winner_run["distinct_8v8_rate_pct"],
        "priceFreeTwinSignature": winner_run["price_free_twin_signature"],
        "subset_violations": winner_run["subset_violations"],
        "lookahead_leakage": winner_run["lookahead_leakage"],
        "total_twin_queries": winner_run["total_twin_queries"],
        "distinct_8v8_rate_pct": winner_run["distinct_8v8_rate_pct"],
        "price_free_twin_signature": winner_run["price_free_twin_signature"],
        "max_sig_sums": winner_run["max_sig_sums"],
        "lock3_rate_pct": winner_run["lock3_rate_pct"],
        "lock3_peak_rate_pct": winner_run["lock3_peak_rate_pct"],
        "veto_counts": winner_run["veto_counts"],
        "splits": winner_run["splits"],
        "versionComparison": {
            "v1": v1_splits,
            "v3": v3_splits,
            "v4": winner_run["splits"],
        },
    }
    with open(os.path.join(DATA_DIR, "xauusd_backtest_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_export, f, ensure_ascii=False, indent=2)

    with open(os.path.join(DATA_DIR, "xauusd_predictions_2012_2026.json"), "w", encoding="utf-8") as f:
        json.dump(winner_run["rows"], f, ensure_ascii=False, indent=2)

    # Export compact newest-first history shard for `/api/signals/history` (keeps web worker < 75 MiB RSS)
    compact_history = [_build_compact_history_row(r) for r in reversed(winner_run["rows"])]
    with open(os.path.join(DATA_DIR, "xauusd_history_shard.json"), "w", encoding="utf-8") as f:
        json.dump(compact_history, f, ensure_ascii=False)

    shard_payload = {
        "service": "XAUUSD-FUSION-SIGNAL",
        "version": "4.0.0",
        "anchor": LOCKED_ANCHOR_KEY,
        "feedSource": "CACHED_VERIFIED_FEED",
        "lastUpdatedUtc": now_iso,
        **live_card,
        "liveSession": live_card,
        "nextSessionPreview": next_card,
        "latest_session": live_card,
        "next_session_preview": next_card,
    }
    with open(os.path.join(DATA_DIR, "xauusd_twin_archive.pkl"), "wb") as pf:
        pickle.dump(
            {
                "latest_rec": latest_rec,
                "archive_before_latest": winner_run["archive_before_latest"],
                "archive_full": winner_run["archive_full"],
            },
            pf,
        )

    with open(os.path.join(SHARDS_DIR, "latest_xauusd_signals.json"), "w", encoding="utf-8") as f:
        json.dump(shard_payload, f, ensure_ascii=False, indent=2)

    with open(os.path.join(DATA_DIR, "latest_xauusd_signal.json"), "w", encoding="utf-8") as f:
        json.dump(shard_payload, f, ensure_ascii=False, indent=2)

    print_v4_report(winner_run, live_card, next_card, v1_splits, v3_splits)
    return {
        "summary": summary_export,
        "live_card": live_card,
        "next_card": next_card,
    }


if __name__ == "__main__":
    run_full_pipeline()
