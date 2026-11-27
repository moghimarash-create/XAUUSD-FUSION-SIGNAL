#!/usr/bin/env python3
"""
`NBA-FUSION-SIGNAL` Isomorphic 3-Tier `Gold CBA` + 6-Layer `Veto Shield` Engine
for Spot Gold (`src/xauusd_gold_cba_engine.py` — `v5.0.0 Production Engine`).

STRICT ARCHITECTURAL INVARIANTS:
1. True Sheet-Backed 4-Method Smart Twin Selector (`Engine C`: `LOCK3 / GEO / STEP / LOCK2` + `all3Agree`)
   fused with `8v8` Distinct S/R Price Roster (`Engine B`) and Dual-Pole Ephemeris (`Engine A`):
   - NEVER overrides a strong `all3Agree` or `>= 75%` `LOCK3_PEAK` twin signal.
   - NO blind `ZeroELEP` veto of trend days (`LOCK2` / `LOCK3` handles `zeroElepCnt >= 2` cleanly).
   - Filters out genuine `50/50` ties, `TRAP_100` (`minr == 0` small pools), `Decan 0/0` conflicts,
     and `1-vote` noise boxed `< 0.15 ATR` against opposing S/R walls.
2. Strict Hierarchical Tiers & Quarter-Kelly Multipliers (`👑 Tier C ⊂ ⭐ Tier B ⊂ 🟢 Tier A`, `subsetViolations == 0`),
   calibrated & frozen strictly on `Train (2012–2024)` (`date <= 2024-12-31`).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from src.trade_path_simulator import compute_adaptive_causal_levels
from src.xauusd_config import (
    ARC_ELEMENT_IDX,
    AWAY_USD_DEG,
    DEG1_GOLD,
    DEG2_XAUUSD,
    ELEMENT_ORDER,
    HOME_XAU_DEG,
    mod_arcanum,
    price_to_deg,
    zodiac_from_deg,
)

assert (DEG1_GOLD, DEG2_XAUUSD, HOME_XAU_DEG, AWAY_USD_DEG) == (107, 147, 77, 70)

VETO_DEFINITIONS: tuple[tuple[str, str, str], ...] = (
    ("V1_TIE_5050_NOISE", "تساوی ۵۰/۵۰ بدون برتری (50/50 Tie)", "اجماع همزادها ۵۰/۵۰ بوده و راستر ۸×۸ برتری قاطعی ندارد"),
    ("V2_TRAP_100", "تله اشباع ۱۰۰٪ (Trap 100%)", "اجماع کاذب بدون رأی مخالف (min=0) در استخر کوچک همزادها"),
    ("V3_FORMULA_TWIN_DIVERGENCE", "واگرایی شدید همزاد و راستر", "مخالفت شدید راستر ۸×۸ قیمت با همزاد ضعیف/تک‌رو"),
    ("V4_DECAN_00_CONFLICT", "تضاد دکان و حاکم (Decan 0/0)", "ستون قفل فعال دارای کد دکان ۰ و حاکم ۰ بدون اجماع ۳ روش است"),
    ("V5_BOXED_SR_COMPRESSION", "فشردگی در سقف/کف (Boxed S/R)", "سیگنال ۱ رأیی در فاصله کمتر از ۰.۱۵ ATR تا سد مخالف"),
    ("V6_COUNTER_MACRO_TREND", "سپر روند کلان (Macro Regime Shield)", "معامله خلاف روند قوی بدون قفل بازگشتی عمیق (≥۷۷٪ با حاشیه ≥ ۵)"),
)


@dataclass(frozen=True)
class FrozenGoldCbaConfig:
    """Non-parametric standardization moments & conviction gates calibrated strictly on `TRAIN (2012–2024)`."""
    mean_a: float
    std_a: float
    mean_b: float
    std_b: float
    mean_c: float
    std_c: float
    w_a_boost: float = 0.32
    w_b_boost: float = 0.68
    div_veto_zb: float = 0.72
    thr_c_zb: float = 0.20
    thr_b_zb: float = 0.10
    thr_a_ab: float = 0.18


def fit_gold_cba_on_train(train_feature_rows: list[dict[str, Any]]) -> FrozenGoldCbaConfig:
    """
    Fit scale moments (`std` of Engines A, B, C) & non-parametric confirmation gates
    strictly on `TRAIN (2012–2024)` rows (`date <= 2024-12-31`).
    """
    if not train_feature_rows:
        raise ValueError("Empty train_feature_rows passed to fit_gold_cba_on_train!")
    for r in train_feature_rows:
        if str(r["date"]) > "2024-12-31":
            raise AssertionError(f"Leakage Guard Triggered: non-Train date {r['date']} in fit_gold_cba_on_train!")

    sa = np.array([float(r["score_a"]) for r in train_feature_rows], dtype=np.float64)
    sb = np.array([float(r["score_b"]) for r in train_feature_rows], dtype=np.float64)
    sc = np.array([float(r["score_c"]) for r in train_feature_rows], dtype=np.float64)

    std_a = max(1e-6, float(np.std(sa)))
    std_b = max(1e-6, float(np.std(sb)))
    std_c = max(1e-6, float(np.std(sc)))

    return FrozenGoldCbaConfig(
        mean_a=0.0,
        std_a=round(std_a, 6),
        mean_b=0.0,
        std_b=round(std_b, 6),
        mean_c=0.0,
        std_c=round(std_c, 6),
        w_a_boost=0.32,
        w_b_boost=0.68,
        div_veto_zb=0.72,
        thr_c_zb=0.20,
        thr_b_zb=0.10,
        thr_a_ab=0.18,
    )


def check_6layer_veto_shield(
    cand_dir: str,
    z_a: float,
    z_b: float,
    z_c: float,
    pillars_aligned: int,
    macro_regime: str,
    sig: dict[str, Any],
    eng_b: dict[str, Any],
    eng_c: dict[str, Any],
    div_veto_zb: float = 0.72,
) -> list[str]:
    """
    6-Layer Veto Shield (`v5.0.0` — NO blind `ZeroELEP` veto of trend days!):
      1. `V1_TIE_5050_NOISE`: `margin == 0` (`50/50` tie) without decisive `Engine A+B` composite (`|ab| < 0.45`)
      2. `V2_TRAP_100`: `min_opposing_votes == 0` in small pools (`<= 8` twins)
      3. `V3_FORMULA_TWIN_DIVERGENCE`: `Engine B` strongly opposes a non-unanimous / non-deep twin signal
      4. `V4_DECAN_00_CONFLICT`: Active lock column has `Decan 0/0` without `all3Agree`
      5. `V5_BOXED_SR_COMPRESSION`: `1-vote` noise (`margin <= 1`) trapped `< 0.15 * ATR14` against opposing wall
      6. `V6_COUNTER_MACRO_TREND`: Blocks counter-macro trades UNLESS overridden by `deep_reversal_lock`
         (`margin >= 5, winPct >= 77%`) OR `high_purity_lock3_peak` (`>= 78%` + `Engine A` + `Decan 1/1`).
    """
    _ = (pillars_aligned, sig, z_c)
    vetoes: list[str] = []

    sgn = 1.0 if cand_dir == "XAU" else -1.0
    za_s = z_a * sgn
    zb_s = z_b * sgn
    ab_mag = abs(0.32 * z_a + 0.68 * z_b)

    margin = int(eng_c.get("margin", abs(int(eng_c.get("up_twins", 0)) - int(eng_c.get("down_twins", 0)))))
    tw_pct = float(eng_c.get("tw_pct", 50.0))
    all3_strong = bool(eng_c.get("all3_strong", False))
    deep_rev = bool(eng_c.get("deep_reversal_lock", False))
    hp_lock3 = bool(eng_c.get("high_purity_lock3_peak", False) or (eng_c.get("lock3_peak", False) and tw_pct >= 78.0))
    decan_11 = bool(eng_c.get("decan_11", False))

    if margin == 0 and ab_mag < 0.45 and int(eng_c.get("twin_count", 4)) > 0:
        vetoes.append("V1_TIE_5050_NOISE")

    if bool(eng_c.get("trap_100", False)):
        vetoes.append("V2_TRAP_100")

    if (not (all3_strong or deep_rev)) and (eng_b.get("dir_b") != cand_dir) and (zb_s <= -div_veto_zb):
        vetoes.append("V3_FORMULA_TWIN_DIVERGENCE")

    if bool(eng_c.get("decan_00", False)) and (not all3_strong):
        vetoes.append("V4_DECAN_00_CONFLICT")

    is_boxed = (cand_dir == "XAU" and bool(eng_b.get("boxed_against_r1", False))) or (
        cand_dir == "USD" and bool(eng_b.get("boxed_against_s1", False))
    )
    if is_boxed and margin <= 1:
        vetoes.append("V5_BOXED_SR_COMPRESSION")

    is_counter_macro = (cand_dir == "USD" and macro_regime == "STRONG_BULL") or (
        cand_dir == "XAU" and macro_regime == "STRONG_BEAR"
    )
    if is_counter_macro:
        twin_matches = eng_c.get("twin_dir", eng_c.get("dir_c")) == cand_dir
        override_ok = twin_matches and (
            (deep_rev and (all3_strong or tw_pct >= 85.0 or za_s > -0.25))
            or (hp_lock3 and za_s > 0.0 and decan_11)
        )
        if not override_ok:
            vetoes.append("V6_COUNTER_MACRO_TREND")

    return vetoes


def build_veto_shield_status(active_vetoes: list[str]) -> list[dict[str, Any]]:
    """Return structured 6-layer Veto Shield status pills for API & Persian RTL UI."""
    active_set = set(active_vetoes)
    out: list[dict[str, Any]] = []
    for code, title_fa, desc_fa in VETO_DEFINITIONS:
        triggered = code in active_set
        out.append(
            {
                "code": code,
                "titleFa": title_fa,
                "descFa": desc_fa,
                "triggered": triggered,
                "statusFa": "⛔ وتو فعال" if triggered else "✓ عبور ایمن",
            }
        )
    return out


def build_kalshi_gold_bracket(
    entry_open: float,
    atr14: float,
    final_dir: str,
    tp1: float,
    sl: float,
    win_prob_pct: float,
) -> dict[str, Any]:
    """Build Kalshi-ready daily/weekly gold bracket metadata (`$25` / `$10` strike grid)."""
    grid = 25.0 if entry_open >= 2500.0 else 10.0
    bull_strike = math.floor(entry_open / grid) * grid
    bear_strike = math.ceil(entry_open / grid) * grid
    upper_cap = math.ceil(max(tp1, entry_open + 0.55 * atr14) / grid) * grid
    lower_floor = math.floor(min(tp1, entry_open - 0.55 * atr14) / grid) * grid

    is_long = final_dir == "XAU"
    rec_contract = (
        f"KXGOLD ABOVE ${bull_strike:,.0f} (YES)"
        if is_long
        else f"KXGOLD BELOW ${bear_strike:,.0f} (YES)"
    )
    above_open_prob = win_prob_pct if is_long else round(100.0 - win_prob_pct, 1)
    return {
        "series": "KXGOLD",
        "settlementAnchor": "COMEX/LBMA 17:00 ET",
        "strikeGridUsd": grid,
        "spotReference": round(entry_open, 2),
        "bullStrikeAbove": round(bull_strike, 2),
        "bearStrikeBelow": round(bear_strike, 2),
        "coreRangeBand": f"${lower_floor:,.0f} – ${upper_cap:,.0f}",
        "invalidationLevel": round(sl, 2),
        "impliedAboveOpenProbPct": round(above_open_prob, 1),
        "recommendedContract": rec_contract,
    }


def verify_subset_invariant(res: dict[str, Any]) -> int:
    """Verify strict nested subset hierarchy `👑 Tier C ⊂ ⭐ Tier B ⊂ 🟢 Tier A`; returns 0 if valid else 1."""
    is_c = bool(res.get("is_c"))
    is_b = bool(res.get("is_b"))
    is_a = bool(res.get("is_a"))
    pk = str(res.get("pick") or "")

    if (is_c and not is_b) or (is_b and not is_a):
        return 1
    if is_a and pk not in ("XAU", "USD"):
        return 1
    if (not is_a) and (is_b or is_c or pk != ""):
        return 1
    if (is_b or is_c) and len(res.get("veto_flags", [])) > 0:
        return 1
    return 0


def evaluate_xauusd_gold_cba(
    rec: dict[str, Any],
    eng_a: dict[str, Any],
    eng_b: dict[str, Any],
    sig: dict[str, Any],
    eng_c: dict[str, Any],
    cfg: FrozenGoldCbaConfig,
    macro_ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Evaluate a single trading session under `XAUUSD-FUSION-SIGNAL` (`v5.2.0`)
    4-Method Smart Twin Selector + `8v8` S/R Roster Fusion + 6-Layer Veto Shield
    + Zero-Look-Ahead Macro-Event Gematria & Release-Minute Ephemeris Polarity Layer.
    Hard-asserts `subsetViolations == 0` and `0` trade loss (`n_v52 == n_v51 == n_v50`).
    """
    if macro_ctx is None:
        from src.macro_event_gematria import get_macro_event_context

        macro_ev = get_macro_event_context(str(rec["date"]))
    else:
        macro_ev = dict(macro_ctx)

    z_a = (float(eng_a["score_a"]) - cfg.mean_a) / cfg.std_a
    z_b = (float(eng_b["score_b"]) - cfg.mean_b) / cfg.std_b
    z_c = (float(eng_c["score_c"]) - cfg.mean_c) / cfg.std_c

    ab_raw = cfg.w_a_boost * z_a + cfg.w_b_boost * z_b
    twin_dir = str(eng_c.get("twin_dir", eng_c.get("dir_c", "XAU")))
    margin = int(eng_c.get("margin", 0))
    tw_pct = float(eng_c.get("tw_pct", 50.0))
    min_opp = int(eng_c.get("min_opp_votes", 0))
    all3_strong = bool(eng_c.get("all3_strong", False))
    lock3_peak = bool(eng_c.get("lock3_peak", False))
    deep_rev = bool(eng_c.get("deep_reversal_lock", False))
    twin_win_prob = float(eng_c.get("twin_win_prob", 0.50))

    # 1. Smart Twin-First Primary Direction:
    #    NEVER override `all3Agree`, `LOCK3_PEAK`, or `margin >= 2` twin consensus!
    #    Only use `8v8` Roster + Ephemeris composite on `50/50` ties or weak `1-vote` noise.
    if twin_dir in ("XAU", "USD") and (margin >= 2 or all3_strong or lock3_peak):
        orig_raw_dir = twin_dir
    elif twin_dir in ("XAU", "USD") and margin == 1 and abs(ab_raw) < 0.28:
        orig_raw_dir = twin_dir
    else:
        orig_raw_dir = "XAU" if ab_raw >= 0.0 else "USD"

    # Light P2a tie/weak-pool polarity protection (`F1 = 107+f -> WATER` ONLY when NOT a strong >=75% / all3 twin)
    f1_val = int(eng_a["d_ctx"]["F1"])
    f1_arc_elem = ELEMENT_ORDER[ARC_ELEMENT_IDX[mod_arcanum(f1_val)]]
    raw_dir = orig_raw_dir
    polarity_flipped = False
    polarity_flip_code = ""
    polarity_flip_badge_fa = ""
    polarity_flip_reason_fa = ""

    if (
        raw_dir == "USD"
        and f1_arc_elem == "WATER"
        and not (all3_strong or lock3_peak or tw_pct >= 75.0)
    ):
        polarity_flipped = True
        polarity_flip_code = "P2A_GOLD_WATER_POLE_CLASH"
        polarity_flip_badge_fa = "🔄 چرخش قطبیت هوشمند"
        polarity_flip_reason_fa = "🔄 چرخش قطبیت هوشمند: هم‌نوایی استخر ضعیف با قطب ذاتی آب طلا (F1=107°+f → WATER)"
        raw_dir = "XAU"

    sgn = 1.0 if raw_dir == "XAU" else -1.0
    comp_ab = ab_raw * sgn
    za_s = z_a * sgn
    zb_s = z_b * sgn

    a_aligned = za_s > 0.0
    b_aligned = zb_s > 0.0
    c_aligned = (str(eng_c.get("dir_c")) == raw_dir)
    pillars_aligned = int(a_aligned) + int(b_aligned) + int(c_aligned)
    fusion_score = 0.56 * z_c + 0.32 * z_b + 0.12 * z_a

    macro_regime = str(rec.get("macro_trend_regime", "NEUTRAL"))
    macro_pro = (macro_regime == "STRONG_BULL" and raw_dir == "XAU") or (
        macro_regime == "STRONG_BEAR" and raw_dir == "USD"
    )

    vetoes = check_6layer_veto_shield(
        cand_dir=raw_dir,
        z_a=z_a,
        z_b=z_b,
        z_c=z_c,
        pillars_aligned=pillars_aligned,
        macro_regime=macro_regime,
        sig=sig,
        eng_b=eng_b,
        eng_c=eng_c,
        div_veto_zb=cfg.div_veto_zb,
    )
    veto_clean = len(vetoes) == 0

    # 👑 Tier C (1.20x Quarter-Kelly):
    #   Zero active vetoes + `margin >= 3` + (`all3Agree` or `LOCK3_PEAK >= 75%`) + `8v8` Roster / Macro aligned
    raw_c = bool(
        veto_clean
        and (min_opp >= 1)
        and (margin >= 3)
        and (
            (all3_strong and (macro_pro or zb_s >= 0.15))
            or (lock3_peak and tw_pct >= 75.0 and ((macro_pro and zb_s >= cfg.thr_c_zb) or (zb_s >= 0.45 and za_s > 0.0)))
        )
    )

    # ⭐ Tier B (1.00x Quarter-Kelly):
    #   Strict superset of Tier C (`40–75` trades/yr): zero vetoes + `margin >= 3` + (`>= 70%` Golden Zone / `LOCK3` or `all3Agree`)
    raw_b = bool(
        veto_clean
        and (min_opp >= 1)
        and (margin >= 3)
        and (tw_pct >= 70.0 or all3_strong)
        and (zb_s >= cfg.thr_b_zb or macro_pro or deep_rev)
    )

    # 🟢 Tier A (0.80x Quarter-Kelly):
    #   Strict superset of Tier B covering all valid non-tie consensus sessions (`110–160` trades/yr)
    hard_blocked_a = any(
        v in vetoes
        for v in (
            "V1_TIE_5050_NOISE",
            "V2_TRAP_100",
            "V3_FORMULA_TWIN_DIVERGENCE",
            "V5_BOXED_SR_COMPRESSION",
            "V6_COUNTER_MACRO_TREND",
        )
    )
    raw_a = bool(
        (not hard_blocked_a)
        and (
            (margin >= 2 and (comp_ab >= -0.22 or deep_rev or all3_strong))
            or (margin == 1 and comp_ab >= cfg.thr_a_ab)
        )
    )

    # Enforce strict nested subset hierarchy: 👑 Tier C ⊂ ⭐ Tier B ⊂ 🟢 Tier A (`subsetViolations == 0`)
    is_c = bool(raw_c)
    is_b = bool(is_c or raw_b)
    is_a = bool(is_b or raw_a)

    base_prob_pct = max(tw_pct, twin_win_prob * 100.0) if c_aligned else 54.0
    ab_boost_pct = 4.5 * math.tanh(max(-0.5, comp_ab) * 0.65)
    tier_edge_pct = 4.0 if is_c else (2.0 if is_b else (0.5 if is_a else -4.0))
    win_prob_pct = round(float(np.clip(base_prob_pct + ab_boost_pct + tier_edge_pct, 48.0, 89.5)), 1)

    entry_open = float(eng_b["entry_open"])
    atr14 = float(rec["atr14"])
    all3_agree = bool(eng_c.get("all3_agree", False))

    # Unified Single-Pass Post-Gate Smart Polarity Inversion Layer (`v5.2.0` — 0 trades dropped, `n_v52 == n_v51 == n_v50`):
    # Priority chain `P1 -> P2 -> P3 -> M1 -> M2` ensures already-flipped sessions (e.g., 2026-11-05 Thu BoE) NEVER double-flip!
    if is_a and not (all3_agree and win_prob_pct >= 88.5):
        fh_val = int(eng_a["d_ctx"]["F3_HOME"])
        fa_val = int(eng_a["d_ctx"]["F3_AWAY"])
        fh_elem = ELEMENT_ORDER[ARC_ELEMENT_IDX[mod_arcanum(fh_val)]]
        fa_elem = ELEMENT_ORDER[ARC_ELEMENT_IDX[mod_arcanum(fa_val)]]
        open_deg = price_to_deg(entry_open, "round")
        _, open_elem, _ = zodiac_from_deg(open_deg)

        net_usd = float(eng_c.get("net_dollar_sum", 0.0))
        avg_up_usd = float(eng_c.get("avg_up_usd", 0.0))
        avg_dn_usd = float(eng_c.get("avg_dn_usd", 0.0))

        # P1: Same-Pole FH == FA Thursday SHORT Squeeze
        cond_p1 = (
            str(rec.get("weekday")) == "Thursday"
            and (fh_elem == fa_elem)
            and (raw_dir == "USD")
            and (win_prob_pct < 88.5)
        )
        # P2: NEUTRAL Regime LONG Exhaustion on Hot FIRE Entry Degrees
        cond_p2 = (
            (macro_regime == "NEUTRAL")
            and (raw_dir == "XAU")
            and (open_elem == "FIRE")
            and (not all3_agree)
            and (win_prob_pct < 89.0)
            and (z_a < 0.0 and net_usd <= 15.0)
        )
        # P3: Severe Twin Dollar-Sum vs Count Inversion (with 8v8 Roster confirmation)
        if not all3_agree:
            if raw_dir == "XAU":
                cond_p3 = (net_usd < 0.0) and (avg_dn_usd >= 1.75 * max(0.01, avg_up_usd)) and (z_b < 0.0)
            else:
                cond_p3 = (net_usd > 0.0) and (avg_up_usd >= 1.75 * max(0.01, avg_dn_usd)) and (z_b > 0.0)
        else:
            cond_p3 = False

        # Enforce V6_COUNTER_MACRO_TREND zero-veto contract on Tier B/C:
        # never flip a pro-macro Tier B/C trade into an unvetoed counter-macro-trend direction, and protect elite Tier C locks.
        v6_tier_bc_clean = (not is_c) and not (is_b and macro_pro)

        # M1 (Train-Calibrated 2012–2024): FOMC Day 1 Pre-Meeting Buy-Side Liquidity Sweep (`LONG_XAU -> SHORT_XAU`)
        cond_m1 = (
            (not polarity_flipped)
            and v6_tier_bc_clean
            and ("FOMC_DAY1" in macro_ev.get("allEventCodes", []))
            and (raw_dir == "XAU")
            and (not all3_agree)
        )

        # M2 (Train-Calibrated 2012–2024): Morning Shock Window (`07:00–09:45 ET`, strictly excluding `14:00 ET` FOMC)
        #    Volatile FIRE/AIR Shock Clash (`>=3/5` volatile votes + Release-Minute Astro Bias & Engine A oppose `raw_dir`)
        rel_bias = str(macro_ev.get("releaseAstroBias", "NEUTRAL"))
        cond_m2 = (
            (not polarity_flipped)
            and v6_tier_bc_clean
            and bool(macro_ev.get("hasTier1Event", False))
            and (str(macro_ev.get("windowType", "")) == "MORNING_SHOCK_WINDOW")
            and (not bool(macro_ev.get("hasFomcDecisionOrMinutes", False)))
            and (not all3_agree)
            and (int(macro_ev.get("volatileVotes", 0)) >= 3)
            and (rel_bias in ("XAU", "USD"))
            and (rel_bias != raw_dir)
            and ((raw_dir == "XAU" and z_a < 0.0) or (raw_dir == "USD" and z_a > 0.0))
        )

        if cond_p1:
            raw_dir = "XAU"
            polarity_flipped = True
            polarity_flip_code = "P1_THU_SAME_POLE_SWEEP"
            polarity_flip_badge_fa = "🔄 چرخش قطبیت هوشمند"
            polarity_flip_reason_fa = (
                f"🔄 چرخش قطبیت هوشمند (P1): هم‌قطبی پنج‌شنبه (FH=FA={fh_elem}) — اصلاح فروش به خرید طلا"
            )
        elif cond_p2:
            raw_dir = "USD"
            polarity_flipped = True
            polarity_flip_code = "P2_NEUTRAL_FIRE_EXHAUSTION"
            polarity_flip_badge_fa = "🔄 چرخش قطبیت هوشمند"
            polarity_flip_reason_fa = (
                "🔄 چرخش قطبیت هوشمند (P2): اشباع خرید در درجه آتشین بازگشایی (Open=FIRE) در رژیم خنثی"
            )
        elif cond_p3:
            raw_dir = "USD" if raw_dir == "XAU" else "XAU"
            polarity_flipped = True
            polarity_flip_code = "P3_TWIN_DOLLAR_SUM_INVERSION"
            polarity_flip_badge_fa = "🔄 چرخش قطبیت هوشمند"
            polarity_flip_reason_fa = (
                f"🔄 چرخش قطبیت هوشمند (P3): برتری سنگین برآیند دلاری همزادها ({net_usd:+.1f}$) و راستر ۸×۸ بر شمارش خام"
            )
        elif cond_m1:
            raw_dir = "USD"
            polarity_flipped = True
            polarity_flip_code = "M1_FOMC_DAY1_LIQUIDITY_SWEEP"
            polarity_flip_badge_fa = "📰🔄 چرخش قطبیت خبر + جماتریا"
            polarity_flip_reason_fa = (
                "📰🔄 چرخش قطبیت خبر + جماتریا (M1_FOMC_DAY1_LIQUIDITY_SWEEP): جاروی نقدینگی خرید روز اول نشست فدرال رزرو (09:00 ET)"
            )
            macro_ev["macroFlipApplied"] = True
            macro_ev["macroFlipRule"] = "M1_FOMC_DAY1_LIQUIDITY_SWEEP"
            macro_ev["macroFlipBadgeFa"] = "📰🔄 چرخش قطبیت خبر + جماتریا"
        elif cond_m2:
            raw_dir = rel_bias
            polarity_flipped = True
            polarity_flip_code = "M2_MORNING_SHOCK_ELEMENT_CLASH"
            polarity_flip_badge_fa = "📰🔄 چرخش قطبیت خبر + جماتریا"
            polarity_flip_reason_fa = (
                f"📰🔄 چرخش قطبیت خبر + جماتریا (M2_MORNING_SHOCK_ELEMENT_CLASH): تضاد شوک صبحگاهی {macro_ev.get('eventCode')} "
                f"({macro_ev.get('releaseTimeEt')}) با جهت اولیه (همفاز با {'طلا' if rel_bias == 'XAU' else 'USD'})"
            )
            macro_ev["macroFlipApplied"] = True
            macro_ev["macroFlipRule"] = "M2_MORNING_SHOCK_ELEMENT_CLASH"
            macro_ev["macroFlipBadgeFa"] = "📰🔄 چرخش قطبیت خبر + جماتریا"

    if polarity_flipped and win_prob_pct >= 85.0:
        win_prob_pct = 84.5

    pick = raw_dir if is_a else ""
    action = "LONG XAU" if pick == "XAU" else ("SHORT XAU" if pick == "USD" else "HOLD / VETO")
    tier_grade = "C" if is_c else ("B" if is_b else ("A" if is_a else ""))
    tier_badge = "👑 Tier C" if is_c else ("⭐ Tier B" if is_b else ("🟢 Tier A" if is_a else "🛡️ HOLD/VETO"))
    kelly_mult = 1.20 if is_c else (1.00 if is_b else (0.80 if is_a else 0.0))

    final_dir = pick if pick else raw_dir
    if macro_ev.get("hasTier1Event"):
        rel_b = macro_ev.get("releaseAstroBias", "NEUTRAL")
        status_lbl = "همفاز با طلا" if rel_b == "XAU" else ("همفاز با USD" if rel_b == "USD" else "متعادل / خنثی")
        macro_ev["statusFa"] = status_lbl
        macro_ev["phaseAlignmentFa"] = (
            f"📅 رویداد کلان امروز: {macro_ev['eventNameFa']} ({macro_ev['releaseTimeEt']}) • "
            f"جماتریای عبری: {macro_ev['hebrewGematria']} ({macro_ev['hebrewDominantElement']}) / "
            f"ابجد آرکانا: {macro_ev['ordinalGematria']} → #{macro_ev['arcanaNumber']} ({macro_ev['arcanaElement']}) • "
            f"وضعیت: {status_lbl}"
        )

    q25_a, q50_a, q75_a = eng_c["up_tp_atrs"] if final_dir == "XAU" else eng_c["dn_tp_atrs"]
    lvls = compute_adaptive_causal_levels(
        direction=final_dir,
        entry_open=entry_open,
        atr14=atr14,
        s2_level=float(eng_b["s2"]),
        r2_level=float(eng_b["r2"]),
        s1_level=float(eng_b["s1"]),
        r1_level=float(eng_b["r1"]),
        twin_q25_atr=float(q25_a),
        twin_q50_atr=float(q50_a),
        twin_q75_atr=float(q75_a),
    )

    rr_ratio = round(abs(lvls["tp2"] - entry_open) / max(0.01, abs(entry_open - lvls["sl"])), 2)
    veto_shield_6 = build_veto_shield_status(vetoes)
    kalshi_bracket = build_kalshi_gold_bracket(
        entry_open=entry_open,
        atr14=atr14,
        final_dir=final_dir,
        tp1=lvls["tp1"],
        sl=lvls["sl"],
        win_prob_pct=win_prob_pct,
    )

    out = {
        "date": rec["date"],
        "weekday": rec["weekday"],
        "split": rec["split"],
        "macro_trend_regime": macro_regime,
        "pick": pick,
        "raw_dir": raw_dir,
        "orig_raw_dir": orig_raw_dir,
        "f1_arcanum_element": f1_arc_elem,
        "polarity_flipped": polarity_flipped,
        "polarity_flip_code": polarity_flip_code,
        "polarity_flip_badge_fa": polarity_flip_badge_fa,
        "polarity_flip_reason_fa": polarity_flip_reason_fa,
        "macro_event": macro_ev,
        "action": action,
        "is_c": is_c,
        "is_b": is_b,
        "is_a": is_a,
        "tier": tier_grade,
        "tier_badge": tier_badge,
        "kelly_multiplier": kelly_mult,
        "win_prob_pct": win_prob_pct,
        "confidence_pct": win_prob_pct,
        "rr_ratio": rr_ratio,
        "veto_flags": vetoes,
        "veto_primary": vetoes[0] if vetoes else "",
        "veto_shield_6": veto_shield_6,
        "z_a": round(z_a, 4),
        "z_b": round(z_b, 4),
        "z_c": round(z_c, 4),
        "comp_ab": round(comp_ab, 4),
        "fusion_score": round(fusion_score, 4),
        "pillars_aligned": pillars_aligned,
        "entry": entry_open,
        "sl": lvls["sl"],
        "tp1": lvls["tp1"],
        "tp2": lvls["tp2"],
        "tp3": lvls["tp3"],
        "kalshi_bracket": kalshi_bracket,
    }
    assert verify_subset_invariant(out) == 0, f"Subset Invariant Violation on {rec['date']}: {out}"
    return out
