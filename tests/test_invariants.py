#!/usr/bin/env python3
"""
Automated Invariant, Zero-Leakage & Shard Verification Suite for `XAUUSD-FUSION-SIGNAL` (`v4.0.0`).

Verifies:
1. Hard Astrological & Entity Locks (`107° CANCER`, `147° LEO`, `77° GEMINI`, `70° GEMINI`).
2. 22-Arcana, 36-Chaldean Decan, and 24-Aspect mathematical parity.
3. `priceFreeTwinSignature == True`: 100% price-invariant pure celestial `12-Col Signature`
   (`$1,150` vs `$4,350` on the same date yields bit-for-bit identical twin keys) + compact bounds (`<= 4, 8, 12`).
4. Strictly Distinct `8v8` Support & Resistance Rosters (`100%` unique rounded levels).
5. 6-Layer Veto Shield (`V6_COUNTER_MACRO_TREND` blocking counter-macro signals unless Twin `LOCK3_PEAK >= 78%`
   + `Engine A` + `Decan 1/1` unanimously confirm exhaustion).
6. Conservative `TP1-Before-SL` Trade Path Simulator (`TP1 & SL` same-bar touch -> strict `SL_LOSS` + `TP3` tracking).
7. `subsetViolations == 0` (`👑 Tier C ⊂ ⭐ Tier B ⊂ 🟢 Tier A`), `lookAheadLeakage == 0`, and production shard schema.
"""
from __future__ import annotations

import json
import os
import unittest

from src.astro_ephemeris_engine import evaluate_engine_a
from src.data_loader import build_zero_leakage_daily_records, fetch_and_cache_market_data
from src.price_roster_engine import (
    _select_distinct_resistances,
    _select_distinct_supports,
)
from src.trade_path_simulator import compute_adaptive_causal_levels, simulate_session_execution
from src.twin_lock_engine import build_day_12col_signature
from src.xauusd_config import (
    ARCANUM_22_MAP,
    AWAY_USD_DEG,
    CHALDEAN_RULERS,
    DEG1_GOLD,
    DEG2_XAUUSD,
    HOME_XAU_DEG,
    RULER_ELEMENT,
    compute_f_helper,
    daily_shifted_degrees,
    deg_to_decan36,
    digital_sum,
    mod_arcanum,
    price_to_deg,
    zodiac_from_deg,
)
from src.xauusd_gold_cba_engine import check_6layer_veto_shield, verify_subset_invariant

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
SHARDS_DIR = os.path.join(ROOT_DIR, "shards")


class TestXauusdFusionInvariants(unittest.TestCase):
    def test_hard_entity_locks(self) -> None:
        self.assertEqual(DEG1_GOLD, 107)
        self.assertEqual(zodiac_from_deg(DEG1_GOLD), ("17° CANCER", "WATER", 3))

        self.assertEqual(DEG2_XAUUSD, 147)
        self.assertEqual(zodiac_from_deg(DEG2_XAUUSD), ("27° LEO", "FIRE", 4))

        self.assertEqual(HOME_XAU_DEG, 77)
        self.assertEqual(zodiac_from_deg(HOME_XAU_DEG), ("17° GEMINI", "AIR", 2))

        self.assertEqual(AWAY_USD_DEG, 70)
        self.assertEqual(zodiac_from_deg(AWAY_USD_DEG), ("10° GEMINI", "AIR", 2))

    def test_price_to_deg_and_helpers(self) -> None:
        self.assertEqual(price_to_deg(3583.75, "floor"), 343)
        self.assertEqual(price_to_deg(3583.75, "round"), 344)
        self.assertEqual(zodiac_from_deg(343), ("13° PISCES", "WATER", 11))
        self.assertEqual(digital_sum("2026-11-26"), 20)
        self.assertEqual(compute_f_helper("2026-11-26"), 24)
        sh = daily_shifted_degrees("2026-11-26")
        self.assertEqual(sh["F1"], 131)
        self.assertEqual(sh["F2"], 171)
        self.assertEqual(sh["F3_HOME"], 101)
        self.assertEqual(sh["F3_AWAY"], 94)

    def test_arcanum_and_decan36(self) -> None:
        self.assertEqual(len(ARCANUM_22_MAP), 22)
        self.assertEqual(mod_arcanum(1), 1)
        self.assertEqual(mod_arcanum(22), 22)
        self.assertEqual(mod_arcanum(23), 1)
        self.assertEqual(deg_to_decan36(0.0), 1)
        self.assertEqual(deg_to_decan36(359.9), 36)
        self.assertEqual(CHALDEAN_RULERS[0], "Mars")
        self.assertEqual(RULER_ELEMENT["Jupiter"], "Air")

    def test_price_free_twin_signature(self) -> None:
        for dt_s in ("2015-08-13", "2020-08-06", "2025-04-22", "2026-11-26"):
            ea_a = evaluate_engine_a(dt_s, 1185.40, 1179.10, 94.20, "ANCHOR_DUAL_POLE_1800", "round")
            ea_b = evaluate_engine_a(dt_s, 4195.80, 4110.50, 107.90, "ANCHOR_DUAL_POLE_1800", "round")
            sig_a = build_day_12col_signature(ea_a, {"dummy_price": 1185.40})
            sig_b = build_day_12col_signature(ea_b, {"dummy_price": 4195.80})
            self.assertTrue(sig_a["price_free"])
            self.assertEqual(sig_a, sig_b)
            self.assertLessEqual(sum(sig_a["elep"]), 4)
            self.assertLessEqual(sum(sig_a["final"]), 8)
            self.assertLessEqual(sum(sig_a["prim"]), 12)

    def test_macro_trend_veto_shield_v6(self) -> None:
        sig = {"zero_elep_cnt": 0}
        eb = {"dir_b": "USD", "boxed_against_r1": False, "boxed_against_s1": False}
        ec_weak = {
            "trap_100": False,
            "dir_c": "USD",
            "tw_pct": 72.0,
            "decan_00": False,
            "decan_11": False,
            "lock3_peak": False,
            "high_purity_lock3_peak": False,
            "twin_dir": "USD",
        }
        # Counter-trend SHORT XAU in STRONG_BULL without LOCK3_PEAK >= 78% + Engine A + Decan 1/1 MUST trigger V6
        v_blocked = check_6layer_veto_shield("USD", -0.80, -0.90, -0.50, 3, "STRONG_BULL", sig, eb, ec_weak)
        self.assertIn("V6_COUNTER_MACRO_TREND", v_blocked)

        # LOCK3_PEAK >= 78% + Engine A aligned (z_a < 0 for USD) + Decan 1/1 overrides V6 cleanly
        ec_strong = {**ec_weak, "lock3_peak": True, "high_purity_lock3_peak": True, "tw_pct": 80.0, "decan_11": True}
        v_override = check_6layer_veto_shield("USD", -0.95, -0.90, -0.80, 3, "STRONG_BULL", sig, eb, ec_strong)
        self.assertNotIn("V6_COUNTER_MACRO_TREND", v_override)

    def test_distinct_8v8_rosters(self) -> None:
        df_daily, _df_1h, df_sess = fetch_and_cache_market_data(force_refresh=False)
        records = build_zero_leakage_daily_records(df_daily, df_sess)
        eval_recs = [r for r in records if r["split"] in ("TRAIN", "VAL", "OOS")]
        self.assertGreater(len(eval_recs), 3500)

        distinct_ok = 0
        for r in eval_recs:
            sup = _select_distinct_supports(r, float(r["today_open"]))
            res = _select_distinct_resistances(r, float(r["today_open"]))
            if len({int(round(x[2])) for x in sup}) == 8 and len({int(round(x[2])) for x in res}) == 8:
                distinct_ok += 1

        self.assertEqual(distinct_ok, len(eval_recs))

    def test_conservative_path_simulator_zero_optimism(self) -> None:
        lvls = compute_adaptive_causal_levels("XAU", 3600.0, 50.0, 3560.0, 3640.0, 0.65, 0.95, 1.35)
        sim_d = simulate_session_execution(
            "XAU", 3600.0, lvls["sl"], lvls["tp1"], lvls["tp2"], 50.0,
            daily_high=lvls["tp2"] + 10.0, daily_low=lvls["sl"] - 5.0, daily_close=3620.0,
            bars_1h=None, tp3=lvls["tp3"],
        )
        self.assertFalse(sim_d["exec_win"])
        self.assertFalse(sim_d["hit_tp3_clean"])
        self.assertEqual(sim_d["outcome"], "AMBIGUOUS_DAILY_SL_LOSS")

        ambig_1h = [(3600.0, lvls["tp1"] + 5.0, lvls["sl"] - 2.0, 3610.0)] * 8
        sim_h = simulate_session_execution(
            "XAU", 3600.0, lvls["sl"], lvls["tp1"], lvls["tp2"], 50.0,
            daily_high=3630.0, daily_low=3540.0, daily_close=3615.0,
            bars_1h=ambig_1h, tp3=lvls["tp3"],
        )
        self.assertFalse(sim_h["exec_win"])
        self.assertEqual(sim_h["outcome"], "AMBIGUOUS_1H_SL_LOSS")

    def test_pipeline_outputs_and_production_shards(self) -> None:
        summary_path = os.path.join(DATA_DIR, "xauusd_backtest_summary.json")
        preds_path = os.path.join(DATA_DIR, "xauusd_predictions_2012_2026.json")
        shard_path = os.path.join(SHARDS_DIR, "latest_xauusd_signals.json")
        hist_shard_path = os.path.join(DATA_DIR, "xauusd_history_shard.json")

        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        self.assertEqual(summary["version"], "4.1.0")
        self.assertEqual(summary["subset_violations"], 0)
        self.assertEqual(summary["lookahead_leakage"], 0)
        self.assertEqual(summary["distinct_8v8_rate_pct"], 100.0)
        self.assertTrue(summary["price_free_twin_signature"])
        self.assertLessEqual(summary["max_sig_sums"]["ELE_P"], 4)
        self.assertLessEqual(summary["max_sig_sums"]["FINAL"], 8)
        self.assertLessEqual(summary["max_sig_sums"]["PRIM"], 12)

        with open(hist_shard_path, encoding="utf-8") as f:
            hrows = json.load(f)
        self.assertGreater(len(hrows), 3500)
        for hr in hrows:
            self.assertFalse((hr["isC"] and not hr["isB"]) or (hr["isB"] and not hr["isA"]))
            if hr["isB"] or hr["isC"]:
                self.assertEqual(len(hr["vetoReasons"]), 0)

        if os.path.exists(preds_path):
            with open(preds_path, encoding="utf-8") as f:
                preds = json.load(f)
            self.assertEqual(len(hrows), len(preds))
            for row in preds:
                self.assertEqual(verify_subset_invariant(row), 0)

        with open(shard_path, encoding="utf-8") as f:
            shard = json.load(f)
        self.assertEqual(shard["service"], "XAUUSD-FUSION-SIGNAL")
        self.assertEqual(shard["version"], "4.1.0")
        self.assertIn("sessionDate", shard)
        self.assertIn("kellyMultiplier", shard)
        self.assertIn("kalshiBracket", shard)
        self.assertEqual(len({int(round(p["level"])) for p in shard["homeXauRoster"]}), 8)
        self.assertEqual(len({int(round(p["level"])) for p in shard["awayUsdRoster"]}), 8)
        for tw in shard["matchedTwins"]:
            self.assertLess(tw["date"], shard["sessionDate"])

    def test_by_date_past_weekend_and_future(self) -> None:
        import pickle
        from src.run_pipeline import build_signal_by_date_payload
        from src.xauusd_gold_cba_engine import FrozenGoldCbaConfig

        with open(os.path.join(DATA_DIR, "xauusd_backtest_summary.json"), encoding="utf-8") as f:
            frozen_cfg = FrozenGoldCbaConfig(**json.load(f)["frozen_cfg"])
        with open(os.path.join(DATA_DIR, "xauusd_history_shard.json"), encoding="utf-8") as f:
            hist_map = {r["date"]: r for r in json.load(f)}
        with open(os.path.join(DATA_DIR, "xauusd_twin_archive.pkl"), "rb") as pf:
            pk = pickle.load(pf)

        records_by_date = pk["records_by_date"]
        sorted_dates = sorted(records_by_date.keys())

        # 1. Past OOS date (2026-11-10)
        past = build_signal_by_date_payload(
            "2026-11-10", records_by_date, sorted_dates, hist_map, pk["archive_full"], pk["latest_rec"], frozen_cfg
        )
        self.assertEqual(past["sessionDate"], "2026-11-10")
        self.assertTrue(past["isHistorical"])
        self.assertFalse(past["isFuture"])
        self.assertIsNotNone(past["actualMove"])
        self.assertIsNotNone(past["execPnl"])
        for tw in past["matchedTwins"]:
            self.assertLess(tw["date"], "2026-11-10")

        # 2. Weekend auto-snap (Saturday 2026-11-14 -> Friday 2026-11-13)
        wknd = build_signal_by_date_payload(
            "2026-11-14", records_by_date, sorted_dates, hist_map, pk["archive_full"], pk["latest_rec"], frozen_cfg
        )
        self.assertTrue(wknd["snapped"])
        self.assertEqual(wknd["sessionDate"], "2026-11-13")

        # 3. Future date (Wednesday 2026-12-02)
        fut = build_signal_by_date_payload(
            "2026-12-02", records_by_date, sorted_dates, hist_map, pk["archive_full"], pk["latest_rec"], frozen_cfg
        )
        self.assertEqual(fut["sessionDate"], "2026-12-02")
        self.assertTrue(fut["isFuture"])
        self.assertEqual(fut["modeBadgeFa"], "پیشبینی تاریخ آتی")
        self.assertEqual(len(fut["homeXauRoster"]), 8)
        self.assertEqual(len(fut["awayUsdRoster"]), 8)
        for tw in fut["matchedTwins"]:
            self.assertLessEqual(tw["date"], pk["latest_rec"]["date"])


if __name__ == "__main__":
    unittest.main()
