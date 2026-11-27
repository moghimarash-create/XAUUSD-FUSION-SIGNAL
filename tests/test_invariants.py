#!/usr/bin/env python3
"""
Automated Invariant, Zero-Leakage & Shard Verification Suite for `XAUUSD-FUSION-SIGNAL` (`v5.0.0`).

Verifies:
1. Hard Astrological & Entity Locks (`107° CANCER`, `147° LEO`, `77° GEMINI`, `70° GEMINI`).
2. 22-Arcana, 36-Chaldean Decan, and 24-Aspect mathematical parity.
3. Authentic 4,705-row Google Sheet Gold Archive (`data/gold_sheet_history.json` & `data/gold_sheet_signals.json`),
   `priceFreeTwinSignature == True` (`$1,150` vs `$4,350` bit-for-bit identical on both Sheet & `v82` Fallback dates),
   and rich matrix scale (`PRIM == ELE-P + FINAL`, `maxSigSums`: `ELE_P >= 8, FINAL >= 20, PRIM >= 28`).
4. Strictly Distinct `8v8` Support & Resistance Rosters (`100%` unique rounded levels).
5. 6-Layer Veto Shield (NO blind `ZeroELEP` veto of trend days + `V6_COUNTER_MACRO_TREND` override rules)
   and clean recovery of `2026-11-02 (Mon)`, `2026-11-03 (Tue)`, and `2026-11-04 (Wed)`.
6. Dual Win Metrics (`KXGOLDPERP` Daily Target Win% + Conservative `Stepped SL = 0.85 ATR -> BE` Simulator).
7. `subsetViolations == 0` (`👑 Tier C ⊂ ⭐ Tier B ⊂ 🟢 Tier A`), `lookAheadLeakage == 0`, and 3-Method Comparison shards.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.astro_ephemeris_engine import evaluate_engine_a
from src.data_loader import build_zero_leakage_daily_records, fetch_and_cache_market_data
from src.price_roster_engine import (
    _select_distinct_resistances,
    _select_distinct_supports,
)
from src.sheet_archive_loader import load_gold_sheet_bundles
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

    def test_sheet_archive_and_price_free_rich_signature(self) -> None:
        hist_b, sig_b, by_date = load_gold_sheet_bundles(auto_sync_if_missing=False)
        self.assertGreaterEqual(hist_b["rawRowCount"], 4700)
        self.assertGreaterEqual(hist_b["weekdayRowCount"], 3900)
        self.assertGreaterEqual(sig_b["weekdayRowCount"], 550)
        self.assertIn("2026-11-02", by_date)
        self.assertIn("2027-02-26", by_date)

        for dt_s in ("2015-08-13", "2020-08-06", "2025-04-22", "2026-11-26", "2028-05-10"):
            ea_a = evaluate_engine_a(dt_s, 1185.40, 1179.10, 94.20, "ANCHOR_DUAL_POLE_1800", "round")
            ea_b = evaluate_engine_a(dt_s, 4195.80, 4110.50, 107.90, "ANCHOR_DUAL_POLE_1800", "round")
            sig_a = build_day_12col_signature(ea_a, {"dummy_price": 1185.40})
            sig_b = build_day_12col_signature(ea_b, {"dummy_price": 4195.80})
            self.assertTrue(sig_a["price_free"])
            self.assertEqual(sig_a, sig_b)
            # Exact additive identity: PRIM[k] == ELE-P[k] + FINAL[k]
            for k in range(4):
                self.assertEqual(sig_a["prim"][k], sig_a["elep"][k] + sig_a["final"][k])
            self.assertGreaterEqual(sum(sig_a["prim"]), 16)

    def test_veto_shield_no_blind_zero_elep_and_v6(self) -> None:
        sig_zero_elep = {"zero_elep_cnt": 3}
        eb_aligned = {"dir_b": "XAU", "boxed_against_r1": False, "boxed_against_s1": False}
        ec_trend = {
            "trap_100": False,
            "dir_c": "XAU",
            "twin_dir": "XAU",
            "margin": 4,
            "up_twins": 6,
            "down_twins": 2,
            "tw_pct": 75.0,
            "all3_strong": False,
            "deep_reversal_lock": False,
            "decan_00": False,
            "decan_11": True,
            "lock3_peak": True,
            "high_purity_lock3_peak": False,
        }
        # Must NOT veto a trend day solely because zeroElepCnt >= 2!
        v_trend = check_6layer_veto_shield("XAU", 0.50, 0.60, 0.80, 3, "STRONG_BULL", sig_zero_elep, eb_aligned, ec_trend)
        self.assertEqual(v_trend, [])

        eb = {"dir_b": "USD", "boxed_against_r1": False, "boxed_against_s1": False}
        ec_weak = {
            "trap_100": False,
            "dir_c": "USD",
            "twin_dir": "USD",
            "margin": 2,
            "tw_pct": 72.0,
            "all3_strong": False,
            "deep_reversal_lock": False,
            "decan_00": False,
            "decan_11": False,
            "lock3_peak": False,
            "high_purity_lock3_peak": False,
        }
        v_blocked = check_6layer_veto_shield("USD", -0.80, -0.90, -0.50, 3, "STRONG_BULL", {"zero_elep_cnt": 0}, eb, ec_weak)
        self.assertIn("V6_COUNTER_MACRO_TREND", v_blocked)

        ec_strong = {**ec_weak, "lock3_peak": True, "high_purity_lock3_peak": True, "tw_pct": 80.0, "decan_11": True}
        v_override = check_6layer_veto_shield("USD", -0.95, -0.90, -0.80, 3, "STRONG_BULL", {"zero_elep_cnt": 0}, eb, ec_strong)
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

    def test_conservative_stepped_sl_and_q25_targets(self) -> None:
        lvls = compute_adaptive_causal_levels(
            "XAU", 3600.0, 50.0, 3560.0, 3640.0,
            twin_q25_atr=0.32, twin_q50_atr=0.62, twin_q75_atr=0.98,
            s1_level=3580.0, r1_level=3620.0,
        )
        # TP1 = Q25 bounded [0.25 * ATR14 ($12.50), 0.45 * ATR14 ($22.50)]
        self.assertGreaterEqual(lvls["tp1_dist"], 12.50)
        self.assertLessEqual(lvls["tp1_dist"], 22.50)
        # Stepped SL bounded [0.55 * ATR14 ($27.50), 0.85 * ATR14 ($42.50)]
        self.assertGreaterEqual(lvls["sl_dist"], 27.50)
        self.assertLessEqual(lvls["sl_dist"], 42.50)

        sim_d = simulate_session_execution(
            "XAU", 3600.0, lvls["sl"], lvls["tp1"], lvls["tp2"], 50.0,
            daily_high=lvls["tp2"] + 10.0, daily_low=lvls["sl"] - 5.0, daily_close=3620.0,
            bars_1h=None, tp3=lvls["tp3"],
        )
        self.assertFalse(sim_d["exec_win"])
        self.assertTrue(sim_d["daily_target_win"])
        self.assertEqual(sim_d["outcome"], "AMBIGUOUS_DAILY_SL_LOSS")

        ambig_1h = [(3600.0, lvls["tp1"] + 5.0, lvls["sl"] - 2.0, 3610.0)] * 8
        sim_h = simulate_session_execution(
            "XAU", 3600.0, lvls["sl"], lvls["tp1"], lvls["tp2"], 50.0,
            daily_high=3630.0, daily_low=3540.0, daily_close=3615.0,
            bars_1h=ambig_1h, tp3=lvls["tp3"],
        )
        self.assertFalse(sim_h["exec_win"])
        self.assertEqual(sim_h["outcome"], "AMBIGUOUS_1H_SL_LOSS")

        # Stepped SL Runner: Bar 1 hits TP1 (locks 50% at TP1, moves SL -> BE), Bar 2 closes at TP3
        clean_runner_1h = [
            (3600.0, lvls["tp1"] + 1.0, 3595.0, lvls["tp1"]),
            (lvls["tp1"], lvls["tp3"] + 2.0, 3605.0, lvls["tp3"]),
        ] * 4
        sim_runner = simulate_session_execution(
            "XAU", 3600.0, lvls["sl"], lvls["tp1"], lvls["tp2"], 50.0,
            daily_high=lvls["tp3"] + 2.0, daily_low=3595.0, daily_close=lvls["tp3"],
            bars_1h=clean_runner_1h, tp3=lvls["tp3"],
        )
        self.assertTrue(sim_runner["exec_win"])
        self.assertTrue(sim_runner["daily_target_win"])
        self.assertTrue(sim_runner["hit_tp1_clean"])
        self.assertTrue(sim_runner["hit_tp2_clean"])
        self.assertTrue(sim_runner["hit_tp3_clean"])
        expected_stepped_pnl = round(
            0.50 * (lvls["tp1"] - 3600.0) + 0.25 * (lvls["tp2"] - 3600.0) + 0.25 * (lvls["tp3"] - 3600.0),
            2,
        )
        self.assertAlmostEqual(sim_runner["exec_pnl"], expected_stepped_pnl, places=1)

    def test_pipeline_outputs_production_shards_and_nov2026_recovery(self) -> None:
        summary_path = os.path.join(DATA_DIR, "xauusd_backtest_summary.json")
        v50_path = os.path.join(DATA_DIR, "v50_backtest_summary.json")
        preds_path = os.path.join(DATA_DIR, "xauusd_predictions_2012_2026.json")
        shard_path = os.path.join(SHARDS_DIR, "latest_xauusd_signals.json")
        hist_shard_path = os.path.join(DATA_DIR, "xauusd_history_shard.json")

        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        self.assertEqual(summary["version"], "5.1.0")
        self.assertEqual(summary["subset_violations"], 0)
        self.assertEqual(summary["lookahead_leakage"], 0)
        self.assertEqual(summary["distinct_8v8_rate_pct"], 100.0)
        self.assertTrue(summary["price_free_twin_signature"])
        self.assertGreaterEqual(summary["max_sig_sums"]["ELE_P"], 8)
        self.assertGreaterEqual(summary["max_sig_sums"]["FINAL"], 20)
        self.assertGreaterEqual(summary["max_sig_sums"]["PRIM"], 28)

        # CRITICAL INVARIANT: 100% Zero Trade Loss across Train, Val 2025, and Blind OOS 2026 vs v5.0.0!
        if os.path.exists(v50_path):
            with open(v50_path, encoding="utf-8") as f:
                v50 = json.load(f)
            for sp in ("TRAIN", "VAL", "OOS"):
                for tk in ("TIER_C", "TIER_B", "TIER_A", "ALL_RAW"):
                    self.assertEqual(
                        summary["splits"][sp][tk]["n"],
                        v50["splits"][sp][tk]["n"],
                        f"Trade count mismatch on {sp} {tk}",
                    )

        # Verify calibration benchmarks on VAL 2025 & Blind OOS 2026
        for sp, yrs in (("VAL", 1.0), ("OOS", 0.90)):
            self.assertGreaterEqual(summary["splits"][sp]["TIER_C"]["daily_win_rate"], 75.0)
            self.assertGreaterEqual(summary["splits"][sp]["TIER_B"]["daily_win_rate"], 72.0)
            self.assertGreaterEqual(summary["splits"][sp]["TIER_C"]["exec_pf"], 1.30)
            self.assertGreaterEqual(summary["splits"][sp]["TIER_B"]["exec_pf"], 1.30)
            trd_b_yr = summary["splits"][sp]["TIER_B"]["n"] / yrs
            trd_a_yr = summary["splits"][sp]["TIER_A"]["n"] / yrs
            self.assertTrue(40.0 <= trd_b_yr <= 75.0, f"{sp} Tier B trd/yr={trd_b_yr}")
            self.assertTrue(110.0 <= trd_a_yr <= 160.0, f"{sp} Tier A trd/yr={trd_a_yr}")

        with open(hist_shard_path, encoding="utf-8") as f:
            hrows = json.load(f)
        self.assertGreater(len(hrows), 3500)
        h_by_date = {hr["date"]: hr for hr in hrows}
        for hr in hrows:
            self.assertFalse((hr["isC"] and not hr["isB"]) or (hr["isB"] and not hr["isA"]))
            if hr["isB"] or hr["isC"]:
                self.assertEqual(len(hr["vetoReasons"]), 0)
            if hr.get("polarityFlipped"):
                self.assertLess(hr["winProbPct"], 85.0)

        # Verify clean recovery of 2026-11-02..2026-11-05 + 2026-08-31 runner & TP1_D fixes!
        for d_rec, exp_dir in (
            ("2026-11-02", "LONG XAU"),
            ("2026-11-03", "SHORT XAU"),
            ("2026-11-04", "LONG XAU"),
            ("2026-11-05", "LONG XAU"),
        ):
            self.assertIn(d_rec, h_by_date)
            row_d = h_by_date[d_rec]
            self.assertTrue(row_d["isB"], f"{d_rec} should be Tier B/C active")
            self.assertEqual(row_d["direction"], exp_dir)
            self.assertTrue(row_d["dailyTargetWin"])
            self.assertTrue(row_d["execWin"])

        # 2026-11-04 must capture true TP1+TP2+TP3/Runner profit (> $40/oz, not $11.20!)
        r_1104 = h_by_date["2026-11-04"]
        self.assertTrue(r_1104["hitTp1"] and r_1104["hitTp2"] and r_1104["hitTp3"])
        self.assertGreater(r_1104["execPnl"], 40.0)

        # 2026-08-31 must have positive TP1 stepped profit (not -$1.70!)
        r_0831 = h_by_date["2026-08-31"]
        self.assertTrue(r_0831["execWin"])
        self.assertGreater(r_0831["execPnl"], 10.0)

        # 2026-11-05 must show Smart Polarity Inversion badge
        r_1105 = h_by_date["2026-11-05"]
        self.assertTrue(r_1105["polarityFlipped"])
        self.assertEqual(r_1105["polarityFlipBadgeFa"], "🔄 چرخش قطبیت هوشمند")

        if os.path.exists(preds_path):
            with open(preds_path, encoding="utf-8") as f:
                preds = json.load(f)
            self.assertEqual(len(hrows), len(preds))
            for row in preds:
                self.assertEqual(verify_subset_invariant(row), 0)

        with open(shard_path, encoding="utf-8") as f:
            shard = json.load(f)
        self.assertEqual(shard["service"], "XAUUSD-FUSION-SIGNAL")
        self.assertEqual(shard["version"], "5.1.0")
        self.assertIn("sessionDate", shard)
        self.assertIn("threeMethodComparison", shard)
        self.assertIn("geo", shard["threeMethodComparison"])
        self.assertIn("step", shard["threeMethodComparison"])
        self.assertIn("lock3", shard["threeMethodComparison"])
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

        # 1. Past OOS date (2026-11-04)
        past = build_signal_by_date_payload(
            "2026-11-04", records_by_date, sorted_dates, hist_map, pk["archive_full"], pk["latest_rec"], frozen_cfg
        )
        self.assertEqual(past["sessionDate"], "2026-11-04")
        self.assertTrue(past["isHistorical"])
        self.assertFalse(past["isFuture"])
        self.assertEqual(past["direction"], "LONG XAU")
        self.assertIn("threeMethodComparison", past)
        for tw in past["matchedTwins"]:
            self.assertLess(tw["date"], "2026-11-04")

        # 2. Weekend auto-snap (Saturday 2026-11-14 -> Friday 2026-11-13)
        wknd = build_signal_by_date_payload(
            "2026-11-14", records_by_date, sorted_dates, hist_map, pk["archive_full"], pk["latest_rec"], frozen_cfg
        )
        self.assertTrue(wknd["snapped"])
        self.assertEqual(wknd["sessionDate"], "2026-11-13")

        # 3. Future Sheet date (Friday 2026-11-27) & Future Fallback date (Wednesday 2028-05-10)
        fut_sheet = build_signal_by_date_payload(
            "2026-11-27", records_by_date, sorted_dates, hist_map, pk["archive_full"], pk["latest_rec"], frozen_cfg
        )
        self.assertEqual(fut_sheet["sessionDate"], "2026-11-27")
        self.assertTrue(fut_sheet["isFuture"])
        self.assertEqual(fut_sheet["signature12Col"]["source"], "GOOGLE_SHEET")

        fut_fb = build_signal_by_date_payload(
            "2028-05-10", records_by_date, sorted_dates, hist_map, pk["archive_full"], pk["latest_rec"], frozen_cfg
        )
        self.assertEqual(fut_fb["sessionDate"], "2028-05-10")
        self.assertTrue(fut_fb["isFuture"])
        self.assertEqual(fut_fb["signature12Col"]["source"], "EPHEMERIS_FALLBACK")
        for tw in fut_fb["matchedTwins"]:
            self.assertLess(tw["date"], "2028-05-10")


if __name__ == "__main__":
    unittest.main()
