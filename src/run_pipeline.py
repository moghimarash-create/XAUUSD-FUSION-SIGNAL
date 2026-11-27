#!/usr/bin/env python3
"""
Master Zero-Leakage Walk-Forward Pipeline, Sheet Sync, Shard Exporter & Benchmark Runner
for `XAUUSD-FUSION-SIGNAL` (`v5.0.0` True Sheet-Backed 4-Method Smart Twin Selector + `8v8` Roster Fusion).

Executes:
1. Authentic 4,705-row Google Sheet Gold Archive (`data/gold_sheet_history.json` & `data/gold_sheet_signals.json`)
   + Multi-Planet (`12 planets × 24 aspects` + `107°/147°/77°/70°` + `36-Decan`) fallback beyond `2027-02-26`.
2. Locked Dual-Pole Anchor (`ANCHOR_DUAL_POLE_1800` + `round(P) % 360`) Zero-Leakage Walk-Forward:
   - LBMA London (`51.5142, -0.0885`) for `HOME XAU` vs COMEX New York (`40.7143, -74.0139`) for `AWAY USD`
     at `18:00:00 America/New_York` on `D-1`.
3. Exact 4-Method Smart Twin Selector (`LOCK3`, `GEO`, `STEP`, `LOCK2` + `pickBestSmartNarrowing`)
   strictly on `hist_date < target_date`, fused with `8v8` Distinct S/R Price Roster + `Q25/Q50/Q75` Twin Targets.
4. Strict Train-Only (`2012–2024`) Calibration & Freeze (`Val 2025` and `Blind OOS 2026` untouched).
5. Exports production shards (`v5.0.0`):
   - `shards/latest_xauusd_signals.json`
   - `data/xauusd_history_shard.json`
   - `data/xauusd_backtest_summary.json`
   - `data/latest_xauusd_signal.json`
"""
from __future__ import annotations

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
from src.price_roster_engine import evaluate_price_roster
from src.sheet_archive_loader import load_gold_sheet_bundles, sync_gold_sheets_from_remote
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
DATA_DIR = os.path.join(ROOT_DIR, "data")
SHARDS_DIR = os.path.join(ROOT_DIR, "shards")

PIPELINE_VERSION = "5.1.0"
LOCKED_ANCHOR_KEY = "ANCHOR_DUAL_POLE_1800"
LOCKED_ROUNDING_MODE = "round"


def _roster_is_distinct_8(roster: dict[str, Any]) -> bool:
    players = roster.get("players", [])
    return len(players) == 8 and len({int(round(float(p["level"]))) for p in players}) == 8


def _enrich_legacy_keys(dual_m: dict[str, Any]) -> dict[str, Any]:
    return {
        **dual_m,
        "wins": dual_m["daily_wins"],
        "losses": dual_m["daily_losses"],
        "win_rate": dual_m["daily_win_rate"],
        "pf": dual_m["daily_pf"],
        "expectancy_usd": dual_m["daily_exp_usd"],
    }


def _verify_price_free_twin_signature() -> bool:
    """Verify `$1,150` vs `$4,350` produces bit-for-bit identical 12-col signature on both Sheet and Fallback dates."""
    for sample_date in ("2026-11-26", "2028-05-10"):
        ea_1 = evaluate_engine_a(sample_date, 1150.25, 1142.80, 92.40, LOCKED_ANCHOR_KEY, LOCKED_ROUNDING_MODE)
        ea_2 = evaluate_engine_a(sample_date, 4350.90, 4295.10, 108.75, LOCKED_ANCHOR_KEY, LOCKED_ROUNDING_MODE)
        sig_1 = build_day_12col_signature(ea_1, {"dummy_price": 1150.25})
        sig_2 = build_day_12col_signature(ea_2, {"dummy_price": 4350.90})
        if not (sig_1["price_free"] and sig_1 == sig_2):
            return False
    return True


def _outcome_badge(row: dict[str, Any]) -> str:
    if not row.get("is_a"):
        return "🛡️ VETO / HOLD"
    out_s = str(row.get("outcome", ""))
    if "SL_LOSS" in out_s:
        return "✗ SL"
    if row.get("hit_tp3_clean"):
        return "✓ TP3"
    if row.get("hit_tp2_clean"):
        return "✓ TP2"
    if row.get("hit_tp1_clean"):
        return "✓ TP1"
    if row.get("hit_tp1_any") and row.get("exec_win"):
        return "✓ TP1"
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
        "polarityFlipped": bool(r.get("polarity_flipped", False)),
        "polarityFlipCode": r.get("polarity_flip_code", ""),
        "polarityFlipBadgeFa": r.get("polarity_flip_badge_fa", ""),
        "polarityFlipReasonFa": r.get("polarity_flip_reason_fa", ""),
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
        "dailyTargetWin": bool(r.get("daily_target_win", r["raw_close_win"])),
        "dailyTargetPnl": r.get("daily_target_pnl", r["raw_close_pnl"]),
        "execWin": bool(r["exec_win"]),
        "execPnl": r["exec_pnl"],
        "rawCloseWin": bool(r["raw_close_win"]),
        "rawClosePnl": r["raw_close_pnl"],
        "hitTp1": bool(r.get("hit_tp1_clean", False) or r.get("hit_tp1_any", False)),
        "hitTp1Clean": bool(r.get("hit_tp1_clean", False)),
        "hitTp2": bool(r.get("hit_tp2_clean", False)),
        "hitTp3": bool(r.get("hit_tp3_clean", False)),
        "hitSl": hit_sl,
        "outcome": out_s,
        "outcomeBadge": _outcome_badge(r),
        "pathSource": r.get("path_source", "DAILY_CONSERVATIVE"),
        "vetoReasons": r["veto_flags"],
        "selectedMethod": r.get("selected_method", "LOCK3"),
        "selectedMethodFa": r.get("selected_method_fa", "🔒 قفل سه‌لایه"),
        "all3Agree": bool(r.get("all3_agree", False)),
        "consensusMode": int(r.get("consensus_mode", 3)),
        "consensusBadgeFa": r.get("consensus_badge_fa", ""),
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
    Execute complete zero-leakage walk-forward run (`v5.0.0`):
      - Seeds `HistoricalTwinArchive` with the authentic Google Sheet Gold archive + completed bars,
        where every `evaluate_twin_and_gaussian(target_date)` strictly slices `hist_date < target_date`!
      - Calibrates & freezes `FrozenGoldCbaConfig` strictly on `TRAIN (2012–2024)`.
    """
    t0 = time.time()
    records_by_date_all = {r["date"]: r for r in records}

    archive_full = HistoricalTwinArchive()
    archive_full.seed_from_sheet_and_records(records_by_date_all)

    latest_date = records[-1]["date"]
    archive_before_latest = HistoricalTwinArchive()
    archive_before_latest.seed_from_sheet_and_records(records_by_date_all, max_date_exclusive=latest_date)

    day_states: list[dict[str, Any]] = []
    distinct_both_ok = 0
    total_roster_days = 0
    price_free_ok = _verify_price_free_twin_signature()
    max_sig_sums = [0, 0, 0]

    for rec in records:
        if rec["split"] not in ("TRAIN", "VAL", "OOS"):
            continue

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

        ec = archive_full.evaluate_twin_and_gaussian(
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

        # Also ingest fallback-computed signature if `rec["date"]` wasn't in Google Sheet
        if rec["date"] not in archive_full._seen_dates:
            d_sgn = 1 if rec["_target_dir"] == "XAU" else (-1 if rec["_target_dir"] == "USD" else 0)
            act_h = max(entry_open, float(rec["_target_high"]))
            act_l = min(entry_open, float(rec["_target_low"]))
            archive_full.add_completed_day(
                date_iso=rec["date"],
                weekday_up=rec["weekday_up"],
                sig=sig,
                direction_sign=d_sgn,
                move_dollars=rec["_target_move"],
                fav_up_atr=max(0.0, act_h - entry_open) / float(rec["atr14"]),
                fav_dn_atr=max(0.0, entry_open - act_l) / float(rec["atr14"]),
                is_noise_flat=rec["_is_noise_flat"],
                hist_atr14=float(rec["atr14"]),
            )

        sess_move = target_close - entry_open
        sess_dir = "XAU" if sess_move > 0 else ("USD" if sess_move < 0 else "FLAT")
        act_high = max(entry_open, float(rec["_target_high"]))
        act_low = min(entry_open, float(rec["_target_low"]))

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
                "selected_method": st["ec"]["selected_method"],
                "selected_method_fa": st["ec"]["selected_method_fa"],
                "all3_agree": st["ec"]["all3_agree"],
                "consensus_mode": st["ec"]["consensus_mode"],
                "consensus_badge_fa": st["ec"]["consensus_badge_fa"],
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
    all3_all_rate = round(
        sum(1 for r in evaluated_rows if r["all3_agree"]) * 100.0 / max(1, len(evaluated_rows)), 2
    )

    return {
        "anchor_key": anchor_key,
        "anchor_label": ANCHOR_CANDIDATES[anchor_key]["label"],
        "rounding_mode": rounding_mode,
        "use_intraday_session_open": use_intraday_session_open,
        "elapsed_sec": round(time.time() - t0, 2),
        "frozen_cfg": asdict(frozen_cfg),
        "subset_violations": subset_violations,
        "lookahead_leakage": archive_full.lookahead_violations,
        "total_twin_queries": archive_full.total_queries,
        "distinct_8v8_rate_pct": distinct_rate,
        "price_free_twin_signature": bool(price_free_ok),
        "max_sig_sums": {"ELE_P": max_sig_sums[0], "FINAL": max_sig_sums[1], "PRIM": max_sig_sums[2]},
        "lock3_rate_pct": lock3_all_rate,
        "lock3_peak_rate_pct": lock3_peak_all_rate,
        "all3_agree_rate_pct": all3_all_rate,
        "veto_counts": veto_counts,
        "splits": splits_summary,
        "rows": evaluated_rows,
        "archive_full": archive_full,
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
    Generate full diagnostic & REST-ready Live Signal Card (`v5.0.0`).
    `archive.evaluate_twin_and_gaussian` strictly filters `hist_date < latest_rec["date"]`!
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
        "polarityFlipped": bool(cba.get("polarity_flipped", False)),
        "polarityFlipCode": cba.get("polarity_flip_code", ""),
        "polarityFlipBadgeFa": cba.get("polarity_flip_badge_fa", ""),
        "polarityFlipReasonFa": cba.get("polarity_flip_reason_fa", ""),
        "f1ArcanumElement": cba.get("f1_arcanum_element", ""),
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
        "threeMethodComparison": ec["three_method_comparison"],
        "signature12Col": {
            "priceFree": sig["price_free"],
            "source": sig.get("source", "GOOGLE_SHEET"),
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
            "selectedMethod": ec["selected_method"],
            "selectedMethodFa": ec["selected_method_fa"],
            "all3Agree": ec["all3_agree"],
            "consensusMode": ec["consensus_mode"],
            "consensusBadgeFa": ec["consensus_badge_fa"],
            "ruleReason": ec["rule_reason"],
            "ruleReasonFa": ec["rule_reason_fa"],
            "threeMethodComparison": ec["three_method_comparison"],
            "twinCount": ec["twin_count"],
            "upTwins": ec["up_twins"],
            "downTwins": ec["down_twins"],
            "margin": ec["margin"],
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
            "posLayers": ec["pos_layers"],
            "decan11": ec["decan_11"],
            "decan00": ec["decan_00"],
            "fp1_4": list(ec["fp"]),
            "ff1_4": list(ec["ff"]),
            "gaussScore": ec["gauss_score"],
            "tpAtrs": list((ec["tp1_atr"], ec["tp2_atr"], ec["tp3_atr"])),
        },
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
            "source": sig.get("source", "GOOGLE_SHEET"),
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
            "selected_method": ec["selected_method"],
            "all3_agree": ec["all3_agree"],
            "consensus_mode": ec["consensus_mode"],
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


def _build_projected_session_rec(
    latest_rec: dict[str, Any],
    target_dt: datetime.date,
    live_spot: float | None = None,
    split_tag: str = "FUTURE_PROJECTION",
) -> dict[str, Any]:
    """Construct causal projected session record for any future weekday `target_dt > latest_rec["date"]`."""
    roll_open = round(float(live_spot if live_spot is not None else latest_rec["_target_close"]), 2)
    prev_h = max(float(latest_rec["today_open"]), float(latest_rec["_target_high"]), roll_open)
    prev_l = min(float(latest_rec["today_open"]), float(latest_rec["_target_low"]), roll_open)
    prev_c = roll_open
    atr = float(latest_rec["atr14"])

    a20, a50 = 2.0 / 21.0, 2.0 / 51.0
    new_ema20 = a20 * prev_c + (1.0 - a20) * float(latest_rec["ema20"])
    new_ema50 = a50 * prev_c + (1.0 - a50) * float(latest_rec["ema50"])
    new_slope = float(latest_rec["ema20_slope"]) * 0.8 + (new_ema20 - float(latest_rec["ema20"])) / max(1.0, atr)
    new_ret20 = float(latest_rec["rolling_20d_ret"]) + (prev_c - float(latest_rec["prev_close"])) / max(
        1.0, float(latest_rec["prev_close"])
    )
    if (prev_c > new_ema20 > new_ema50) and (new_slope > 0.0) and (new_ret20 > 0.0):
        new_regime = "STRONG_BULL"
    elif (prev_c < new_ema20 < new_ema50) and (new_slope < 0.0) and (new_ret20 < 0.0):
        new_regime = "STRONG_BEAR"
    else:
        new_regime = "NEUTRAL"

    nxt = dict(latest_rec)
    nxt.update(
        {
            "date": target_dt.strftime("%Y-%m-%d"),
            "weekday": target_dt.strftime("%A"),
            "weekday_up": target_dt.strftime("%A").upper(),
            "split": split_tag,
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


def _build_next_session_preview_rec(latest_rec: dict[str, Any], live_spot: float | None = None) -> dict[str, Any]:
    y, m, d = map(int, latest_rec["date"].split("-"))
    nxt_dt = datetime.date(y, m, d) + datetime.timedelta(days=1)
    while nxt_dt.weekday() >= 5:
        nxt_dt += datetime.timedelta(days=1)
    return _build_projected_session_rec(latest_rec, nxt_dt, live_spot=live_spot, split_tag="LIVE_PREVIEW")


def _fetch_live_yahoo_spot_quote() -> tuple[float | None, str]:
    url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=1h&range=2d"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=(1.2, 1.8))
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
      - Re-syncs Google Sheets CSVs (`HISTORY_CSV` & `SIGNALS_CSV`) when `try_network_fetch=True`
        with graceful local fallback,
      - Fetches optional live `GC=F` spot quote,
      - Recomputes `shards/latest_xauusd_signals.json` (`v5.0.0`) while keeping `RSS < 150 MiB`.
    """
    live_spot = None
    feed_source = "CACHED_VERIFIED_FEED"
    sheets_synced = False
    sheet_sync_mode = "CACHED_LOCAL_SHEET_JSON"

    if try_network_fetch:
        sheets_synced, sheet_sync_mode = sync_gold_sheets_from_remote(timeout=(3.5, 9.5))
        live_spot, feed_source = _fetch_live_yahoo_spot_quote()

    summary_path = os.path.join(DATA_DIR, "xauusd_backtest_summary.json")
    with open(summary_path, encoding="utf-8") as f:
        frozen_cfg = FrozenGoldCbaConfig(**json.load(f)["frozen_cfg"])

    pkl_path = os.path.join(DATA_DIR, "xauusd_twin_archive.pkl")
    latest_rec: dict[str, Any] | None = None
    archive_before_latest: HistoricalTwinArchive | None = None
    archive: HistoricalTwinArchive | None = None
    records_by_date: dict[str, dict[str, Any]] | None = None

    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, "rb") as pf:
                cached_bundle = pickle.load(pf)
            if cached_bundle.get("version") == PIPELINE_VERSION:
                latest_rec = cached_bundle["latest_rec"]
                records_by_date = cached_bundle.get("records_by_date")
                if not sheets_synced:
                    archive_before_latest = cached_bundle["archive_before_latest"]
                    archive = cached_bundle["archive_full"]
        except Exception:
            latest_rec = None

    if latest_rec is None or records_by_date is None:
        from src.data_loader import build_zero_leakage_daily_records, fetch_and_cache_market_data

        df_daily, _df_1h, df_sess = fetch_and_cache_market_data(force_refresh=False)
        records = build_zero_leakage_daily_records(df_daily, df_sess)
        records_by_date = {r["date"]: r for r in records if r["date"] >= "2012-01-01"}
        latest_rec = records[-1]

    if archive is None or archive_before_latest is None:
        archive = HistoricalTwinArchive()
        archive.seed_from_sheet_and_records(records_by_date)
        archive_before_latest = HistoricalTwinArchive()
        archive_before_latest.seed_from_sheet_and_records(records_by_date, max_date_exclusive=latest_rec["date"])

        with open(pkl_path, "wb") as pf:
            pickle.dump(
                {
                    "version": PIPELINE_VERSION,
                    "latest_rec": latest_rec,
                    "archive_before_latest": archive_before_latest,
                    "archive_full": archive,
                    "records_by_date": records_by_date,
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
        "version": PIPELINE_VERSION,
        "anchor": LOCKED_ANCHOR_KEY,
        "feedSource": feed_source,
        "sheetSyncMode": sheet_sync_mode,
        "sheetsSynced": sheets_synced,
        "lastUpdatedUtc": now_iso,
        **live_card,
        "liveSession": live_card,
        "nextSessionPreview": next_card,
        "latest_session": live_card,
        "next_session_preview": next_card,
    }


def build_signal_by_date_payload(
    requested_date_str: str,
    records_by_date: dict[str, dict[str, Any]],
    sorted_dates: list[str],
    history_by_date: dict[str, dict[str, Any]],
    archive_full: HistoricalTwinArchive,
    latest_rec: dict[str, Any],
    frozen_cfg: FrozenGoldCbaConfig,
    live_spot: float | None = None,
) -> dict[str, Any]:
    """
    Evaluate ANY requested date (`Past 2012..latest_bar`, `Present latest_bar`, or `Future > latest_bar` up to `2035-12-31`):
      - Auto-snaps weekend / holiday dates to nearest trading weekday.
      - Strictly enforces `matchedTwins < sessionDate` (`lookAheadLeakage == 0`).
      - Uses Google Sheet signature up to `2027-02-26` and multi-planet `v82` fallback beyond `2027-02-26`.
    """
    raw_s = requested_date_str.strip()
    try:
        req_dt = datetime.date.fromisoformat(raw_s)
    except ValueError as exc:
        raise ValueError(f"Invalid date format '{raw_s}'. Expected YYYY-MM-DD.") from exc

    if req_dt < datetime.date(2012, 1, 1) or req_dt > datetime.date(2035, 12, 31):
        raise ValueError(f"Date '{raw_s}' out of supported range (2012-01-01 .. 2035-12-31).")

    req_iso = req_dt.strftime("%Y-%m-%d")
    latest_bar_iso = str(latest_rec["date"])
    latest_bar_dt = datetime.date.fromisoformat(latest_bar_iso)

    snapped = False
    snapped_note_fa = ""

    if req_iso <= latest_bar_iso:
        if req_iso in records_by_date:
            eval_iso = req_iso
        else:
            eval_iso = min(
                sorted_dates,
                key=lambda d: (abs((datetime.date.fromisoformat(d) - req_dt).days), 0 if d >= req_iso else 1),
            )
            snapped = True
            snapped_note_fa = f"تاریخ {req_iso} تعطیل بازار بود؛ نزدیک‌ترین سشن معاملاتی ({eval_iso}) بارگذاری شد."

        rec = records_by_date[eval_iso]
        is_live = eval_iso == latest_bar_iso
        spot_ref = (
            live_spot
            if (is_live and live_spot is not None)
            else float(rec.get("_target_close", rec["today_open"]))
        )
        card = build_live_signal_card(
            latest_rec=rec,
            archive=archive_full,
            frozen_cfg=frozen_cfg,
            spot_override=spot_ref,
        )
        for tw in card["matchedTwins"]:
            assert tw["date"] < eval_iso, f"Look-ahead violation in by-date: {tw['date']} >= {eval_iso}"

        idx = sorted_dates.index(eval_iso)
        prev_date = sorted_dates[idx - 1] if idx > 0 else eval_iso
        if idx + 1 < len(sorted_dates):
            next_date = sorted_dates[idx + 1]
        else:
            nxt_dt = latest_bar_dt + datetime.timedelta(days=1)
            while nxt_dt.weekday() >= 5:
                nxt_dt += datetime.timedelta(days=1)
            next_date = nxt_dt.strftime("%Y-%m-%d")

        hrow = history_by_date.get(eval_iso, {})
        act_move = float(hrow.get("actualMove", round(float(rec.get("_target_move", 0.0)), 2)))
        act_dir = str(hrow.get("actualDir", rec.get("_target_dir", "FLAT")))
        hit_out = str(hrow.get("outcomeBadge", "—"))
        exec_pnl = float(hrow.get("execPnl", 0.0))
        exec_win = bool(hrow.get("execWin", False))
        raw_pnl = float(hrow.get("rawClosePnl", act_move if card["biasDirection"] == "XAU" else -act_move))
        raw_win = bool(hrow.get("rawCloseWin", raw_pnl > 0))
        daily_win = bool(hrow.get("dailyTargetWin", raw_win))
        daily_pnl = float(hrow.get("dailyTargetPnl", raw_pnl))

        session_type = "LIVE" if is_live else "HISTORICAL"
        mode_badge_fa = "سشن زنده فعلی (Live Session)" if is_live else "بازبینی سشن تاریخی"

        return {
            "ok": True,
            "service": "XAUUSD-FUSION-SIGNAL",
            "version": PIPELINE_VERSION,
            "requestedDate": req_iso,
            "snapped": snapped,
            "snappedNoteFa": snapped_note_fa,
            "sessionType": session_type,
            "isHistorical": True,
            "isFuture": False,
            "isLiveSession": is_live,
            "latestBarDate": latest_bar_iso,
            "prevSessionDate": prev_date,
            "nextSessionDate": next_date,
            "modeBadgeFa": mode_badge_fa,
            **card,
            "actualMove": act_move,
            "actualDir": act_dir,
            "actualOpen": round(float(rec["today_open"]), 2),
            "actualHigh": float(hrow.get("actualHigh", round(float(rec.get("_target_high", rec["today_open"])), 2))),
            "actualLow": float(hrow.get("actualLow", round(float(rec.get("_target_low", rec["today_open"])), 2))),
            "actualClose": float(hrow.get("actualClose", round(float(rec.get("_target_close", rec["today_open"])), 2))),
            "hitOutcome": hit_out,
            "outcomeCode": str(hrow.get("outcome", "")),
            "dailyTargetWin": daily_win,
            "dailyTargetPnl": round(daily_pnl, 2),
            "execPnl": exec_pnl,
            "execWin": exec_win,
            "rawClosePnl": round(raw_pnl, 2),
            "rawCloseWin": raw_win,
            "hitTp1": bool(hrow.get("hitTp1", False)),
            "hitTp2": bool(hrow.get("hitTp2", False)),
            "hitTp3": bool(hrow.get("hitTp3", False)),
            "hitSl": bool(hrow.get("hitSl", False)),
        }

    # Future Date Branch (`req_iso > latest_bar_iso`)
    eval_dt = req_dt
    if eval_dt.weekday() == 5:
        fri_dt = eval_dt - datetime.timedelta(days=1)
        eval_dt = fri_dt if fri_dt > latest_bar_dt else (eval_dt + datetime.timedelta(days=2))
        snapped = True
    elif eval_dt.weekday() == 6:
        eval_dt = eval_dt + datetime.timedelta(days=1)
        snapped = True

    eval_iso = eval_dt.strftime("%Y-%m-%d")
    if snapped:
        snapped_note_fa = f"تاریخ {req_iso} آخر هفته بود؛ سشن معاملاتی آتی ({eval_iso}) بارگذاری شد."

    fut_rec = _build_projected_session_rec(
        latest_rec=latest_rec,
        target_dt=eval_dt,
        live_spot=live_spot,
        split_tag="FUTURE_PROJECTION",
    )
    card = build_live_signal_card(
        latest_rec=fut_rec,
        archive=archive_full,
        frozen_cfg=frozen_cfg,
        spot_override=live_spot,
    )
    for tw in card["matchedTwins"]:
        assert tw["date"] <= latest_bar_iso and tw["date"] < eval_iso

    prv_dt = eval_dt - datetime.timedelta(days=1)
    while prv_dt.weekday() >= 5:
        prv_dt -= datetime.timedelta(days=1)
    prev_date = latest_bar_iso if prv_dt <= latest_bar_dt else prv_dt.strftime("%Y-%m-%d")

    nxt_dt = eval_dt + datetime.timedelta(days=1)
    while nxt_dt.weekday() >= 5:
        nxt_dt += datetime.timedelta(days=1)
    next_date = nxt_dt.strftime("%Y-%m-%d")

    sig_src_fa = (
        "شیت رسمی طلا (۲۰۱۲–۲۰۲۷)"
        if card["signature12Col"]["source"] == "GOOGLE_SHEET"
        else "محاسبه نجومی ۱۲ سیاره و ۲۴ زاویه (فراتر از ۲۰۲۷)"
    )

    return {
        "ok": True,
        "service": "XAUUSD-FUSION-SIGNAL",
        "version": PIPELINE_VERSION,
        "requestedDate": req_iso,
        "snapped": snapped,
        "snappedNoteFa": snapped_note_fa,
        "sessionType": "FUTURE",
        "isHistorical": False,
        "isFuture": True,
        "isLiveSession": False,
        "latestBarDate": latest_bar_iso,
        "prevSessionDate": prev_date,
        "nextSessionDate": next_date,
        "modeBadgeFa": "پیشبینی تاریخ آتی",
        "futureProjectionNoteFa": (
            f"پیشبینی تاریخ آتی ({eval_iso}) — منبع ماتریس ۱۲ ستونی: {sig_src_fa} • "
            f"انتخابگر هوشمند ۳ روشه ({card['twinConsensus']['selectedMethodFa']}) و سطوح ۸×۸ بر پایه آخرین اسپات (${card['spotPrice']:,.2f})."
        ),
        **card,
        "actualMove": None,
        "actualDir": None,
        "hitOutcome": "پیشبینی تاریخ آتی",
        "dailyTargetWin": None,
        "dailyTargetPnl": None,
        "execPnl": None,
        "execWin": None,
    }


def _load_prior_versions_for_comparison() -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    v1_splits = None
    v3_splits = None
    v4_splits = None
    p1_path = os.path.join(DATA_DIR, "phase1_anchor_benchmark.json")
    v3_path = os.path.join(DATA_DIR, "v3_backtest_summary.json")
    v4_path = os.path.join(DATA_DIR, "v41_backtest_summary.json")
    if os.path.exists(p1_path):
        with open(p1_path, encoding="utf-8") as f:
            for item in json.load(f):
                if item.get("anchor_key") == LOCKED_ANCHOR_KEY and item.get("rounding_mode") == LOCKED_ROUNDING_MODE:
                    v1_splits = item.get("splits")
                    break
    if os.path.exists(v3_path):
        with open(v3_path, encoding="utf-8") as f:
            v3_splits = json.load(f).get("splits")
    if os.path.exists(v4_path):
        with open(v4_path, encoding="utf-8") as f:
            v4_splits = json.load(f).get("splits")
    return v1_splits, v3_splits, v4_splits


def print_v5_report(
    winner_run: dict[str, Any],
    live_card: dict[str, Any],
    next_card: dict[str, Any],
) -> None:
    print("\n" + "=" * 128)
    print(
        f"  XAUUSD-FUSION-SIGNAL v{PIPELINE_VERSION} • SHEET-BACKED 4-METHOD SMART TWIN SELECTOR + 8v8 ROSTER FUSION"
    )
    print(
        f"  INVARIANT AUDIT: subsetViolations == {winner_run['subset_violations']} | "
        f"lookAheadLeakage == {winner_run['lookahead_leakage']} | "
        f"distinct8v8Roster == {winner_run['distinct_8v8_rate_pct']:.2f}% | "
        f"maxSigSums == {winner_run['max_sig_sums']}"
    )
    print("=" * 128)

    for sp_code, sp_title, yrs in [
        ("TRAIN", "TRAIN (2012-01-01 .. 2024-12-31 • Calibrated & Frozen)", 13.0),
        ("VAL",   "VALIDATION (2025-01-01 .. 2025-12-31 • Chronological 1h Path)", 1.0),
        ("OOS",   "BLIND OUT-OF-SAMPLE (2026-01-01 .. 2026-11-26 YTD • Chronological 1h Path)", 0.90),
    ]:
        print(f"\n  [{sp_title}]")
        print(
            f"  {'Tier':18s} | {'Trd/Yr':7s} | {'All3%':6s} | "
            f"{'Daily Target Win% (KXGOLDPERP)':32s} || "
            f"{'Stepped SL=0.85ATR Win% / PF / Exp$':36s}"
        )
        print("  " + "-" * 116)
        sp_d = winner_run["splits"][sp_code]
        for tk, tlabel in [
            ("TIER_C", "👑 Tier C (1.20x)"),
            ("TIER_B", "⭐ Tier B (1.00x)"),
            ("TIER_A", "🟢 Tier A (0.80x)"),
            ("ALL_RAW", "⚪ All Sessions"),
        ]:
            m = sp_d[tk]
            trd_yr = m["n"] / yrs
            dw_s = f"{m['daily_win_rate']:6.2f}% ({m['daily_wins']}-{m['daily_losses']}, PF {m['daily_pf']:4.2f})"
            sw_s = f"{m['exec_win_rate']:6.2f}% ({m['exec_wins']}-{m['exec_losses']}, PF {m['exec_pf']:4.2f}, ${m['exec_exp_usd']:+6.2f})"
            print(f"  {tlabel:18s} | {trd_yr:6.1f}  | {m['all3_agree_rate']:5.1f}% | {dw_s:32s} || {sw_s:36s}")

    print(
        f"\n  LIVE SESSION ({live_card['sessionDate']}): {live_card['tierBadge']} {live_card['direction']} | "
        f"Method={live_card['twinConsensus']['selectedMethod']} ({live_card['twinConsensus']['votes']}) | "
        f"Mode={live_card['twinConsensus']['consensusMode']}"
    )
    print(
        f"  NEXT PREVIEW ({next_card['sessionDate']}): {next_card['tierBadge']} {next_card['direction']} | "
        f"Method={next_card['twinConsensus']['selectedMethod']} ({next_card['twinConsensus']['votes']}) | "
        f"Mode={next_card['twinConsensus']['consensusMode']}"
    )


def run_full_pipeline() -> dict[str, Any]:
    from src.data_loader import (
        build_zero_leakage_daily_records,
        fetch_and_cache_market_data,
        group_1h_bars_by_session,
    )

    hist_bundle, sig_bundle, _ = load_gold_sheet_bundles(auto_sync_if_missing=True)
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

    v1_splits, v3_splits, v4_splits = _load_prior_versions_for_comparison()
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SHARDS_DIR, exist_ok=True)

    v50_splits = None
    v50_path = os.path.join(DATA_DIR, "v50_backtest_summary.json")
    if os.path.exists(v50_path):
        with open(v50_path, encoding="utf-8") as f:
            v50_splits = json.load(f).get("splits")

    summary_export = {
        "service": "XAUUSD-FUSION-SIGNAL",
        "version": PIPELINE_VERSION,
        "versionTag": "v5.1",
        "lastUpdatedUtc": now_iso,
        "locked_anchor": winner_run["anchor_key"],
        "locked_anchor_label": winner_run["anchor_label"],
        "locked_rounding": winner_run["rounding_mode"],
        "sheetArchiveMeta": {
            "historyRawRows": hist_bundle["rawRowCount"],
            "historyWeekdayRows": hist_bundle["weekdayRowCount"],
            "historyPricedRows": hist_bundle["pricedRowCount"],
            "signalsRawRows": sig_bundle["rawRowCount"],
            "signalsWeekdayRows": sig_bundle["weekdayRowCount"],
            "minDate": hist_bundle["minDate"],
            "maxDate": hist_bundle["maxDate"],
        },
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
        "all3_agree_rate_pct": winner_run["all3_agree_rate_pct"],
        "veto_counts": winner_run["veto_counts"],
        "splits": winner_run["splits"],
        "versionComparison": {
            "v1": v1_splits,
            "v3": v3_splits,
            "v4": v4_splits,
            "v50": v50_splits,
            "v51": winner_run["splits"],
        },
    }
    with open(os.path.join(DATA_DIR, "xauusd_backtest_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_export, f, ensure_ascii=False, indent=2)

    with open(os.path.join(DATA_DIR, "xauusd_predictions_2012_2026.json"), "w", encoding="utf-8") as f:
        json.dump(winner_run["rows"], f, ensure_ascii=False, indent=2)

    compact_history = [_build_compact_history_row(r) for r in reversed(winner_run["rows"])]
    with open(os.path.join(DATA_DIR, "xauusd_history_shard.json"), "w", encoding="utf-8") as f:
        json.dump(compact_history, f, ensure_ascii=False)

    shard_payload = {
        "service": "XAUUSD-FUSION-SIGNAL",
        "version": PIPELINE_VERSION,
        "anchor": LOCKED_ANCHOR_KEY,
        "feedSource": "CACHED_VERIFIED_FEED",
        "lastUpdatedUtc": now_iso,
        **live_card,
        "liveSession": live_card,
        "nextSessionPreview": next_card,
        "latest_session": live_card,
        "next_session_preview": next_card,
    }
    records_by_date = {r["date"]: r for r in records if r["date"] >= "2012-01-01"}
    with open(os.path.join(DATA_DIR, "xauusd_twin_archive.pkl"), "wb") as pf:
        pickle.dump(
            {
                "version": PIPELINE_VERSION,
                "latest_rec": latest_rec,
                "archive_before_latest": winner_run["archive_before_latest"],
                "archive_full": winner_run["archive_full"],
                "records_by_date": records_by_date,
            },
            pf,
        )

    with open(os.path.join(SHARDS_DIR, "latest_xauusd_signals.json"), "w", encoding="utf-8") as f:
        json.dump(shard_payload, f, ensure_ascii=False, indent=2)

    with open(os.path.join(DATA_DIR, "latest_xauusd_signal.json"), "w", encoding="utf-8") as f:
        json.dump(shard_payload, f, ensure_ascii=False, indent=2)

    print_v5_report(winner_run, live_card, next_card)
    return {
        "summary": summary_export,
        "live_card": live_card,
        "next_card": next_card,
    }


if __name__ == "__main__":
    run_full_pipeline()
