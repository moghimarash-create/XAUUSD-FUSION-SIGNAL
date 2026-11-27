#!/usr/bin/env python3
"""
100% Price-Free Pure Astro-Arcana-Decan `12-Col Twin Signature`, ATR-Normalized &
Recency-Weighted (`exp(-years_ago / 6.0)`) Non-Parametric Twin Consensus Engine
(`src/twin_lock_engine.py` — `v4 Production Engine`).

STRICT ARCHITECTURAL INVARIANTS (`lookAheadLeakage == 0`, `priceFreeTwinSignature == True`):
1. Twin pool for `target_date` ONLY searches completed historical days with `hist_date < target_date`,
   excluding flat noise days (`|Close - Open| <= max($1.00, 0.15 * ATR14)`).
2. 100% Price-Invariant Celestial `12-Column Elemental Signature`:
   - `ELE-P (0..4)`: Tightest 12-planet aspect (`Sun..Chiron`, `+1°` Sun/Moon bonus) on `F1, F2, F3_HOME, F3_AWAY`.
   - `FINAL (0..8)`: `ELE-P` + 4 Pure Celestial Angles/Luminaries (`LBMA_ASC`, `COMEX_MC`, `MOON`, `LBMA_POF`) Decans.
   - `PRIM  (0..12)`: `FINAL` + 4 Base Shifted Arcana Cards `mod_arcanum(F_i)`.
3. ATR-Dominance & Recency-Weighted Twin Consensus (`4..10` non-flat twins):
   - `w_i = exp(-years_ago / 6.0) * atr_norm_move_i * decan_bit_similarity_i`
   - `strongAvgDominance = (maj_avg_atr_move >= 1.35 * min_avg_atr_move)`
   - Parity adjustments: `isDecBoth1 -> +6.5%`, `isDecBoth0 -> -6.5%`, `zeroElepCnt == 0 -> +4.0%`.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.trade_path_simulator import compute_adaptive_causal_levels
from src.xauusd_config import (
    ARC_ELEMENT_IDX,
    ASPECTS_24,
    ELEMENT_INDEX,
    ELEMENT_ORDER,
    GEOMANCY_FIGURES,
    SUN_MOON_BODIES,
    decan36_info,
    deg_to_decan36,
    mod_arcanum,
)

PLANET_NAMES_12: tuple[str, ...] = (
    "Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter",
    "Saturn", "Uranus", "Neptune", "Pluto", "Node", "Chiron",
)

_POPCOUNT_16 = np.array([bin(x).count("1") for x in range(16)], dtype=np.int8)


def _iso_to_year_float(d_iso: str) -> float:
    y, m, d = map(int, d_iso.split("-"))
    return float(y) + (float(m) - 1.0) / 12.0 + (float(d) - 1.0) / 365.25


def _tightest_planet_arcanum_elem(label_val: int, bodies: dict[str, dict[str, Any]]) -> tuple[int | None, str]:
    ldeg_mod = float(label_val) % 360.0
    best_err: float | None = None
    best_plon: float | None = None
    best_tag = ""

    for pname in PLANET_NAMES_12:
        b = bodies.get(pname)
        if b is None:
            continue
        plon = float(b["lon"]) % 360.0
        orb_bonus = 1.0 if pname in SUN_MOON_BODIES else 0.0
        diff = abs(ldeg_mod - plon)
        if diff > 180.0:
            diff = 360.0 - diff
        for sym, ang, orb in ASPECTS_24:
            err = abs(diff - ang)
            if err <= orb + orb_bonus:
                if best_err is None or err < best_err:
                    best_err = err
                    best_plon = plon
                    best_tag = f"{pname}{sym}"

    if best_plon is None:
        return None, ""
    res_deg = int(round(best_plon + ldeg_mod))
    arc = mod_arcanum(res_deg)
    return ARC_ELEMENT_IDX[arc], best_tag


def build_day_12col_signature(
    eng_a: dict[str, Any],
    eng_b: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute 100% Price-Free Pure Astro-Arcana-Decan `12-Col Signature`
    (`sum(ELE-P) <= 4`, `sum(FINAL) <= 8`, `sum(PRIM) <= 12`).
    """
    _ = eng_b  # Explicitly unused: twin lookup signature is 100% price-invariant!
    d_ctx = eng_a["d_ctx"]
    f1 = int(d_ctx["F1"])
    f2 = int(d_ctx["F2"])
    f3_h = int(d_ctx["F3_HOME"])
    f3_a = int(d_ctx["F3_AWAY"])

    snap_xau = eng_a["snap_xau"]
    snap_usd = eng_a["snap_usd"]

    # 1. ELE-P (0..4)
    elep_list = [0, 0, 0, 0]
    step3_hits: list[str] = []
    for lbl_name, lbl_val, sn in (
        ("F1", f1, snap_xau),
        ("F2", f2, snap_usd),
        ("F3_H", f3_h, snap_xau),
        ("F3_A", f3_a, snap_usd),
    ):
        el_tbl = sn.get("step3_el_tbl")
        if el_tbl is not None:
            d_mod = lbl_val % 360
            el_idx_raw = el_tbl[d_mod]
            if el_idx_raw >= 0:
                pi = sn["step3_pa_tbl"][0][d_mod]
                ai = sn["step3_pa_tbl"][1][d_mod]
                elep_list[el_idx_raw] += 1
                step3_hits.append(f"{lbl_name}:{PLANET_NAMES_12[pi]}{ASPECTS_24[ai][0]}->{ELEMENT_ORDER[el_idx_raw][0]}")
        else:
            el_idx, hit_s = _tightest_planet_arcanum_elem(lbl_val, sn["bodies"])
            if el_idx is not None:
                elep_list[el_idx] += 1
                step3_hits.append(f"{lbl_name}:{hit_s}->{ELEMENT_ORDER[el_idx][0]}")

    elep = tuple(elep_list)
    zero_elep_cnt = sum(1 for x in elep if x == 0)

    # 2. FINAL (0..8): ELE-P + 4 Pure Topocentric Angles & Luminary/POF Decan Rulers (ZERO Dollar Prices!)
    lbma_asc = float(snap_xau["asc"]) % 360.0
    comex_mc = float(snap_usd["mc"]) % 360.0
    moon_lon = float(snap_xau["bodies"]["Moon"]["lon"]) % 360.0
    lbma_pof = float(snap_xau["pof"]) % 360.0

    celestial_decan_4 = [0, 0, 0, 0]
    celestial_tags: list[str] = []
    for cname, cdeg in (
        ("LBMA_ASC", lbma_asc),
        ("COMEX_MC", comex_mc),
        ("MOON", moon_lon),
        ("POF", lbma_pof),
    ):
        dec_info = decan36_info(deg_to_decan36(cdeg))
        r_elem = dec_info["ruler_element"]
        celestial_decan_4[ELEMENT_INDEX[r_elem]] += 1
        celestial_tags.append(f"{cname}:D{dec_info['decan']:02d}({dec_info['ruler']}->{r_elem[0]})")

    final_4 = tuple(int(elep[k] + celestial_decan_4[k]) for k in range(4))

    # 3. PRIM (0..12): FINAL + 4 Base Shifted Arcana Cards (F1, F2, F3_HOME, F3_AWAY)
    base_arc_4 = [0, 0, 0, 0]
    for fv in (f1, f2, f3_h, f3_a):
        base_arc_4[ARC_ELEMENT_IDX[mod_arcanum(fv)]] += 1

    prim_4 = tuple(int(final_4[k] + base_arc_4[k]) for k in range(4))

    geo_fig = "".join(str(prim_4[k] % 2) for k in range(4))
    geo_name = GEOMANCY_FIGURES.get(geo_fig, "Populus")

    d_bits = tuple((final_4[k] + elep[(k + 1) % 4] + 1) % 2 for k in range(4))
    r_bits = tuple((prim_4[k] + final_4[(k + 2) % 4]) % 2 for k in range(4))
    dec_geo = "".join(str(b) for b in d_bits)
    ruler_geo = "".join(str(b) for b in r_bits)

    col12 = prim_4 + final_4 + elep

    return {
        "price_free": True,
        "elep": elep,
        "final": final_4,
        "prim": prim_4,
        "col12": col12,
        "zero_elep_cnt": zero_elep_cnt,
        "geo_fig": geo_fig,
        "geo_name": geo_name,
        "dec_geo": dec_geo,
        "ruler_geo": ruler_geo,
        "step3_hits": step3_hits,
        "celestial_decan_tags": celestial_tags,
    }


class HistoricalTwinArchive:
    """
    Strictly causal walk-forward archive of historical trading days (`date < target_date`),
    bucketed by `weekday_up` for O(1) vectorized `LOCK3_PEAK` progressive narrowing (`4..10` twins),
    ATR-Dominance & Recency-Weighted (`exp(-years_ago / 6.0)`) Consensus, and Regime-Normalized Targets.
    """

    def __init__(self) -> None:
        self.by_weekday: dict[str, dict[str, list]] = {}
        self._np_cache: dict[str, dict[str, Any]] = {}
        self._dirty: set[str] = set()
        self.lookahead_violations: int = 0
        self.total_queries: int = 0

    def add_completed_day(
        self,
        date_iso: str,
        weekday_up: str,
        sig: dict[str, Any],
        direction_sign: int,
        move_dollars: float,
        fav_up_atr: float,
        fav_dn_atr: float,
        is_noise_flat: bool,
        fav_up_pct: float | None = None,
        fav_dn_pct: float | None = None,
        hist_atr14: float | None = None,
    ) -> None:
        """Ingest a COMPLETED historical session AFTER `target_date` prediction has already been emitted."""
        if is_noise_flat or direction_sign == 0:
            return
        b = self.by_weekday.get(weekday_up)
        if b is None:
            b = {
                "dates": [],
                "year_float": [],
                "col12": [],
                "geo_fig": [],
                "dec_geo": [],
                "ruler_geo": [],
                "dir": [],
                "move": [],
                "body_atr": [],
                "fav_up_atr": [],
                "fav_dn_atr": [],
                "fav_up_pct": [],
                "fav_dn_pct": [],
            }
            self.by_weekday[weekday_up] = b

        atr_est = float(hist_atr14) if (hist_atr14 is not None and hist_atr14 > 0.5) else max(1.0, abs(move_dollars) / max(0.25, max(fav_up_atr, fav_dn_atr)))
        body_atr = max(0.08, abs(float(move_dollars)) / atr_est)

        b["dates"].append(date_iso)
        b["year_float"].append(_iso_to_year_float(date_iso))
        b["col12"].append(sig["col12"])
        b["geo_fig"].append(int(sig["geo_fig"], 2))
        b["dec_geo"].append(int(sig["dec_geo"], 2))
        b["ruler_geo"].append(int(sig["ruler_geo"], 2))
        b["dir"].append(int(direction_sign))
        b["move"].append(float(move_dollars))
        b["body_atr"].append(float(body_atr))
        b["fav_up_atr"].append(float(fav_up_atr))
        b["fav_dn_atr"].append(float(fav_dn_atr))
        b["fav_up_pct"].append(float(fav_up_pct if fav_up_pct is not None else (fav_up_atr * 0.012)))
        b["fav_dn_pct"].append(float(fav_dn_pct if fav_dn_pct is not None else (fav_dn_atr * 0.012)))
        self._dirty.add(weekday_up)

    def _sync_weekday(self, weekday_up: str) -> dict[str, Any] | None:
        b = self.by_weekday.get(weekday_up)
        if not b or not b["dates"]:
            return None
        if weekday_up in self._dirty or weekday_up not in self._np_cache:
            self._np_cache[weekday_up] = {
                "dates": b["dates"],
                "max_date": b["dates"][-1],
                "year_float": np.asarray(b["year_float"], dtype=np.float64),
                "col12": np.asarray(b["col12"], dtype=np.int16),
                "geo_fig": np.asarray(b["geo_fig"], dtype=np.int8),
                "dec_geo": np.asarray(b["dec_geo"], dtype=np.int8),
                "ruler_geo": np.asarray(b["ruler_geo"], dtype=np.int8),
                "dir": np.asarray(b["dir"], dtype=np.int8),
                "move": np.asarray(b["move"], dtype=np.float32),
                "body_atr": np.asarray(b["body_atr"], dtype=np.float32),
                "fav_up_atr": np.asarray(b["fav_up_atr"], dtype=np.float32),
                "fav_dn_atr": np.asarray(b["fav_dn_atr"], dtype=np.float32),
                "fav_up_pct": np.asarray(b["fav_up_pct"], dtype=np.float32),
                "fav_dn_pct": np.asarray(b["fav_dn_pct"], dtype=np.float32),
            }
            self._dirty.discard(weekday_up)
        return self._np_cache[weekday_up]

    @staticmethod
    def _detect_lock3(sub_col12: np.ndarray, target_col12: np.ndarray, zero_elep_cnt: int) -> list[int]:
        if len(sub_col12) == 0:
            return []
        eq_all = np.all(sub_col12 == target_col12, axis=0)
        locked_elems: list[int] = []
        for k in range(4):
            if zero_elep_cnt >= 1 and target_col12[8 + k] == 0:
                continue
            if bool(eq_all[k] and eq_all[4 + k] and eq_all[8 + k]):
                locked_elems.append(k)
        return locked_elems

    @staticmethod
    def _narrow_pool_largest_count(
        start_idx: np.ndarray,
        col12_all: np.ndarray,
        dec_all: np.ndarray,
        rul_all: np.ndarray,
        t_col12: np.ndarray,
        t_dec: int,
        t_rul: int,
        eligible_cols: np.ndarray,
        preset_used: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Outcome-blind largest-count progressive narrowing (`automaticSelection`) into `4..10` twins.
        """
        idx = start_idx.copy()
        used = preset_used.copy() if preset_used is not None else np.zeros(12, dtype=bool)

        while len(idx) > 10:
            cur_n = len(idx)
            avail = eligible_cols & (~used)
            if not np.any(avail):
                break
            eq_mat = (col12_all[idx] == t_col12)
            cnts = np.sum(eq_mat, axis=0)
            valid = avail & (cnts >= 4) & (cnts < cur_n)
            if not np.any(valid):
                break
            masked = np.where(valid, cnts, -1)
            bc = int(np.argmax(masked))
            used[bc] = True
            idx = idx[eq_mat[:, bc]]
            if 4 <= len(idx) <= 10:
                break

        if len(idx) > 10:
            sub_dec = idx[dec_all[idx] == t_dec]
            if len(sub_dec) >= 4:
                idx = sub_dec
            if len(idx) > 10:
                sub_rul = idx[rul_all[idx] == t_rul]
                if len(sub_rul) >= 4:
                    idx = sub_rul
            if len(idx) > 10:
                idx = idx[-10:]
        return idx

    def evaluate_twin_and_gaussian(
        self,
        target_date: str,
        weekday_up: str,
        sig: dict[str, Any],
        entry_open: float,
        atr14: float,
        s1: float,
        r1: float,
        fallback_dir: str = "XAU",
        s2: float | None = None,
        r2: float | None = None,
    ) -> dict[str, Any]:
        """
        Execute strictly causal (`hist_date < target_date`) Non-Parametric Twin-First Consensus:
          - `w_i = exp(-years_ago / 6.0) * atr_norm_move_i * decan_bit_similarity_i`
          - `strongAvgDominance = (maj_avg_atr_move >= 1.35 * min_avg_atr_move)`
          - Parity adjustments: `isDecBoth1 -> +6.5%`, `isDecBoth0 -> -6.5%`, `zeroElepCnt == 0 -> +4.0%`.
        """
        self.total_queries += 1
        s2_eff = float(s2 if s2 is not None else (s1 - 0.50 * atr14))
        r2_eff = float(r2 if r2 is not None else (r1 + 0.50 * atr14))

        arr = self._sync_weekday(weekday_up)
        if arr is None or len(arr["dates"]) < 6:
            lvls = compute_adaptive_causal_levels(fallback_dir, entry_open, atr14, s2_eff, r2_eff, 0.65, 0.95, 1.35)
            return {
                "has_twins": False,
                "twin_count": 0,
                "up_twins": 0,
                "down_twins": 0,
                "max_votes": 0,
                "min_opp_votes": 0,
                "tw_pct": 50.0,
                "w_disp_pct": 50.0,
                "twin_win_prob": 0.50,
                "twin_dir": fallback_dir,
                "strong_avg_dominance": False,
                "maj_avg_atr_move": 0.0,
                "min_avg_atr_move": 0.0,
                "in_golden_zone": False,
                "lock3_cols": [],
                "has_lock3": False,
                "lock3_peak": False,
                "high_purity_lock3_peak": False,
                "active_lock_col": 0,
                "active_lock_elem": ELEMENT_ORDER[0],
                "decan_11": False,
                "decan_00": False,
                "trap_100": False,
                "fp": (0.0, 0.0, 0.0, 0.0),
                "ff": (0.0, 0.0, 0.0, 0.0),
                "gauss_score": 0.0,
                "score_c": 0.0,
                "dir_c": fallback_dir,
                "twin_dates": [],
                "matched_twins": [],
                "up_tp_atrs": (0.65, 0.95, 1.35),
                "dn_tp_atrs": (0.65, 0.95, 1.35),
                "tp1_atr": 0.65,
                "tp2_atr": 0.95,
                "tp3_atr": 1.35,
                "tp1": lvls["tp1"],
                "tp2": lvls["tp2"],
                "tp3": lvls["tp3"],
                "sl": lvls["sl"],
            }

        # HARD ZERO-LOOKAHEAD VERIFICATION
        if arr["max_date"] >= target_date:
            valid_n = int(np.searchsorted(arr["dates"], target_date, side="left"))
            if any(d >= target_date for d in arr["dates"][:valid_n]):
                self.lookahead_violations += 1
                raise AssertionError(f"Look-ahead violation detected on {target_date}")
        else:
            valid_n = len(arr["dates"])

        col12_all = arr["col12"][:valid_n]
        geo_all = arr["geo_fig"][:valid_n]
        dec_all = arr["dec_geo"][:valid_n]
        rul_all = arr["ruler_geo"][:valid_n]
        dir_all = arr["dir"][:valid_n].astype(np.float64)
        move_all = arr["move"][:valid_n]
        body_atr_all = arr["body_atr"][:valid_n].astype(np.float64)
        fav_up_all = arr["fav_up_atr"][:valid_n]
        fav_dn_all = arr["fav_dn_atr"][:valid_n]
        pct_up_all = arr["fav_up_pct"][:valid_n]
        pct_dn_all = arr["fav_dn_pct"][:valid_n]
        year_all = arr["year_float"][:valid_n]
        dates_all = arr["dates"][:valid_n]

        if valid_n > 0 and not (dates_all[valid_n - 1] < target_date):
            self.lookahead_violations += 1
            raise AssertionError(f"Look-ahead leakage: {dates_all[valid_n - 1]} >= {target_date}")

        target_yf = _iso_to_year_float(target_date)
        years_ago_all = np.maximum(0.0, target_yf - year_all)
        recency_all = np.exp(-years_ago_all / 6.0)

        t_col12 = np.asarray(sig["col12"], dtype=np.int16)
        t_geo = int(sig["geo_fig"], 2)
        t_dec = int(sig["dec_geo"], 2)
        t_rul = int(sig["ruler_geo"], 2)
        zero_elep_cnt = int(sig["zero_elep_cnt"])

        # -------------------------------------------------------------------
        # 1. Recency-Weighted (`exp(-years_ago / 6.0)`) Gaussian Priority Consensus
        # -------------------------------------------------------------------
        geo_match_bonus = np.where(geo_all == t_geo, 1.35, 0.75) * recency_all
        dec_match_bonus = np.where(dec_all == t_dec, 1.25, 0.85) * recency_all

        fp_list: list[float] = []
        ff_list: list[float] = []
        fp_weights: list[float] = []
        ff_weights: list[float] = []

        for k in range(4):
            diff_p = (col12_all[:, k].astype(np.float64) - float(t_col12[k])) / 1.10
            wp = np.exp(-0.5 * diff_p * diff_p) * geo_match_bonus
            swp = float(np.sum(wp))
            fp_k = float(np.sum(wp * dir_all) / swp) if swp > 1e-6 else 0.0
            fp_list.append(round(fp_k, 4))
            fp_weights.append(abs(fp_k) * min(3.0, np.sqrt(swp / 6.0)))

            diff_f = (col12_all[:, 4 + k].astype(np.float64) - float(t_col12[4 + k])) / 0.95
            wf = np.exp(-0.5 * diff_f * diff_f) * dec_match_bonus
            swf = float(np.sum(wf))
            ff_k = float(np.sum(wf * dir_all) / swf) if swf > 1e-6 else 0.0
            ff_list.append(round(ff_k, 4))
            ff_weights.append(abs(ff_k) * min(3.0, np.sqrt(swf / 6.0)))

        tot_gw = sum(fp_weights) + sum(ff_weights)
        if tot_gw > 1e-6:
            gauss_score = (
                sum(fp_list[k] * fp_weights[k] for k in range(4))
                + sum(ff_list[k] * ff_weights[k] for k in range(4))
            ) / tot_gw
        else:
            gauss_score = float(np.mean(fp_list + ff_list))

        # -------------------------------------------------------------------
        # 2. Outcome-Blind Canonical `LOCK3` + Largest-Count Narrowing (`4..10` Twins)
        # -------------------------------------------------------------------
        idx_base = np.flatnonzero(geo_all == t_geo)
        if len(idx_base) < 4:
            idx_base = np.arange(valid_n, dtype=np.int64)

        eligible = np.ones(12, dtype=bool)
        if zero_elep_cnt >= 1:
            for k in range(4):
                if t_col12[8 + k] == 0:
                    eligible[8 + k] = False

        best_k: int | None = None
        best_sub: np.ndarray | None = None
        best_key = (-1, -1, -1)
        for k in range(4):
            if zero_elep_cnt >= 1 and t_col12[8 + k] == 0:
                continue
            m_k = (
                (col12_all[idx_base, k] == t_col12[k])
                & (col12_all[idx_base, 4 + k] == t_col12[4 + k])
                & (col12_all[idx_base, 8 + k] == t_col12[8 + k])
            )
            sub_k = idx_base[m_k]
            if len(sub_k) >= 4:
                cand_key = (int(t_col12[8 + k]), len(sub_k), int(t_col12[k]))
                if cand_key > best_key:
                    best_key = cand_key
                    best_k = k
                    best_sub = sub_k

        if best_sub is not None and best_k is not None:
            p_used = np.zeros(12, dtype=bool)
            p_used[best_k] = p_used[4 + best_k] = p_used[8 + best_k] = True
            final_idx = self._narrow_pool_largest_count(
                best_sub, col12_all, dec_all, rul_all, t_col12, t_dec, t_rul, eligible, p_used
            )
        else:
            final_idx = self._narrow_pool_largest_count(
                idx_base, col12_all, dec_all, rul_all, t_col12, t_dec, t_rul, eligible, None
            )

        n_twins = len(final_idx)
        d_twins = dir_all[final_idx]
        up_twins = int(np.sum(d_twins > 0))
        dn_twins = n_twins - up_twins
        max_votes = max(up_twins, dn_twins)
        min_opp = min(up_twins, dn_twins)
        tw_pct = (max_votes * 100.0 / n_twins) if n_twins > 0 else 50.0

        # ATR-Dominance & Recency Weighting: w_i = exp(-years_ago/6.0) * atr_norm_move_i * decan_bit_similarity_i
        rec_sub = recency_all[final_idx]
        atr_sub = body_atr_all[final_idx]
        dec_diff_bits = _POPCOUNT_16[np.bitwise_xor(dec_all[final_idx], t_dec)]
        rul_diff_bits = _POPCOUNT_16[np.bitwise_xor(rul_all[final_idx], t_rul)]
        match_bits_8 = (4 - dec_diff_bits) + (4 - rul_diff_bits)
        dec_sim_sub = 0.65 + 0.0875 * match_bits_8.astype(np.float64)

        w_sub = rec_sub * atr_sub * dec_sim_sub
        up_w = float(np.sum(w_sub[d_twins > 0]))
        dn_w = float(np.sum(w_sub[d_twins < 0]))
        w_up_prob = up_w / max(1e-6, up_w + dn_w)
        raw_up_prob = up_twins / float(max(1, n_twins))

        # Primary Twin-First blend inside the 4..10 narrowed twin pool (Gaussian only breaks close ties)
        gauss_up_prob = 0.50 + 0.50 * float(np.clip(gauss_score * 1.50, -0.35, 0.35))
        p_xau_base = 0.50 * raw_up_prob + 0.40 * w_up_prob + 0.10 * gauss_up_prob

        dir_c = "XAU" if p_xau_base >= 0.50 else "USD"
        if up_twins > dn_twins:
            twin_dir = "XAU"
        elif dn_twins > up_twins:
            twin_dir = "USD"
        else:
            twin_dir = dir_c

        dir_tw_pct = round((raw_up_prob if dir_c == "XAU" else (1.0 - raw_up_prob)) * 100.0, 1)
        dir_w_pct = round((w_up_prob if dir_c == "XAU" else (1.0 - w_up_prob)) * 100.0, 1)

        # Compute strongAvgDominance: maj_avg_atr_move >= 1.35 * min_avg_atr_move
        maj_mask = (d_twins > 0) if dir_c == "XAU" else (d_twins < 0)
        min_mask = ~maj_mask
        maj_avg_atr = float(np.mean(atr_sub[maj_mask])) if np.any(maj_mask) else 0.50
        min_avg_atr = float(np.mean(atr_sub[min_mask])) if np.any(min_mask) else 0.35
        strong_avg_dom = bool(np.any(maj_mask) and (maj_avg_atr >= 1.35 * max(0.08, min_avg_atr)))

        lock3_cols = self._detect_lock3(col12_all[final_idx], t_col12, zero_elep_cnt)
        has_lock3 = len(lock3_cols) > 0
        in_golden_zone = (
            (twin_dir == dir_c)
            and (4 <= n_twins <= 10)
            and (1 <= min_opp <= 2)
            and (dir_tw_pct >= 70.0)
            and (abs(up_twins - dn_twins) >= 2)
        )
        is_lock3_peak = bool(
            has_lock3
            and (twin_dir == dir_c)
            and (4 <= n_twins <= 10)
            and (min_opp >= 1)
            and (dir_tw_pct >= 70.0)
        )
        high_purity_lock3_peak = bool(
            is_lock3_peak
            and (dir_tw_pct >= 78.0)
            and (max_votes >= 6 or (max_votes == 5 and min_opp == 1))
        )

        if lock3_cols:
            active_col = lock3_cols[0]
        else:
            active_col = int(np.argmax([abs(fp_list[k]) + abs(ff_list[k]) for k in range(4)]))

        dec_str = sig["dec_geo"]
        rul_str = sig["ruler_geo"]
        decan_11 = (dec_str[active_col] == "1") and (rul_str[active_col] == "1")
        decan_00 = (dec_str[active_col] == "0") and (rul_str[active_col] == "0")

        trap_100 = (n_twins > 0) and (n_twins <= 6) and (min_opp == 0)

        # 4-Bit Decan & Ruler Parity Adjustments (+6.5%, -6.5%, +4.0%)
        maj_base_prob = p_xau_base if dir_c == "XAU" else (1.0 - p_xau_base)
        parity_adj = (
            (0.065 if decan_11 else 0.0)
            + (-0.065 if decan_00 else 0.0)
            + (0.040 if zero_elep_cnt == 0 else 0.0)
            + (0.025 if strong_avg_dom else 0.0)
        )
        twin_win_prob = float(np.clip(maj_base_prob + parity_adj, 0.36, 0.92))
        w_disp_pct = dir_w_pct

        # Non-parametric signed score_c in [-1, +1] centered at 0.50 probability
        score_c = (twin_win_prob - 0.50) * 2.0 * (1.0 if dir_c == "XAU" else -1.0)

        # -------------------------------------------------------------------
        # 3. Regime-Normalized Twin Percentiles (% Return + ATR14 Multiples -> Today's ATR14)
        # -------------------------------------------------------------------
        atr_safe = max(1.0, float(atr14))

        def _calc_regime_norm_pcts(is_up: bool) -> tuple[float, float, float]:
            w_m = (d_twins > 0) if is_up else (d_twins < 0)
            w_idx = final_idx[w_m]
            if len(w_idx) < 3:
                all_w = np.flatnonzero((dir_all > 0) if is_up else (dir_all < 0))
                w_idx = all_w[-35:] if len(all_w) >= 3 else final_idx

            fm_atr = fav_up_all[w_idx] if is_up else fav_dn_all[w_idx]
            fm_pct = pct_up_all[w_idx] if is_up else pct_dn_all[w_idx]
            valid_m = fm_atr > 0.01
            fm_atr = fm_atr[valid_m]
            fm_pct = fm_pct[valid_m]

            if len(fm_atr) > 0:
                proj_atr_from_pct = (fm_pct.astype(np.float64) * float(entry_open)) / atr_safe
                blended_atr = 0.60 * fm_atr.astype(np.float64) + 0.40 * proj_atr_from_pct
                p50 = float(np.clip(np.percentile(blended_atr, 50), 0.38, 1.15))
                p75 = float(np.clip(np.percentile(blended_atr, 75), max(p50 + 0.16, 0.62), 1.65))
                p90 = float(np.clip(np.percentile(blended_atr, 90), max(p75 + 0.20, 0.92), 2.35))
                return p50, p75, p90
            return 0.65, 0.95, 1.35

        up_pcts = _calc_regime_norm_pcts(True)
        dn_pcts = _calc_regime_norm_pcts(False)
        is_long_c = dir_c == "XAU"
        tp1_atr, tp2_atr, tp3_atr = up_pcts if is_long_c else dn_pcts

        lvls = compute_adaptive_causal_levels(
            direction=dir_c,
            entry_open=entry_open,
            atr14=atr14,
            s2_level=s2_eff,
            r2_level=r2_eff,
            twin_p50_atr=tp1_atr,
            twin_p75_atr=tp2_atr,
            twin_p90_atr=tp3_atr,
            s1_level=float(s1),
            r1_level=float(r1),
        )

        twin_dates = [dates_all[int(i)] for i in final_idx[-10:]]
        matched_twins: list[dict[str, Any]] = []
        for j_pos, idx_m in enumerate(final_idx):
            im = int(idx_m)
            matched_twins.append(
                {
                    "date": dates_all[im],
                    "weekday": weekday_up.title(),
                    "direction": "XAU" if dir_all[im] > 0 else "USD",
                    "move_usd": round(float(move_all[im]), 2),
                    "move_atr": round(float(body_atr_all[im]), 2),
                    "col12_matches": int(np.sum(col12_all[im] == t_col12)),
                    "decan_sim": round(float(dec_sim_sub[j_pos]), 2),
                    "weight": round(float(w_sub[j_pos]), 3),
                }
            )

        return {
            "has_twins": n_twins >= 4,
            "twin_count": n_twins,
            "up_twins": up_twins,
            "down_twins": dn_twins,
            "max_votes": max_votes,
            "min_opp_votes": min_opp,
            "tw_pct": round(tw_pct, 1),
            "dir_tw_pct": dir_tw_pct,
            "w_disp_pct": w_disp_pct,
            "dir_w_pct": dir_w_pct,
            "twin_win_prob": round(twin_win_prob, 4),
            "twin_dir": twin_dir,
            "strong_avg_dominance": strong_avg_dom,
            "maj_avg_atr_move": round(maj_avg_atr, 3),
            "min_avg_atr_move": round(min_avg_atr, 3),
            "in_golden_zone": bool(in_golden_zone),
            "lock3_cols": [ELEMENT_ORDER[c] for c in lock3_cols],
            "has_lock3": bool(has_lock3),
            "lock3_peak": bool(is_lock3_peak),
            "high_purity_lock3_peak": bool(high_purity_lock3_peak),
            "active_lock_col": active_col,
            "active_lock_elem": ELEMENT_ORDER[active_col],
            "decan_11": bool(decan_11),
            "decan_00": bool(decan_00),
            "trap_100": bool(trap_100),
            "fp": tuple(fp_list),
            "ff": tuple(ff_list),
            "gauss_score": round(gauss_score, 4),
            "score_c": round(score_c, 4),
            "dir_c": dir_c,
            "twin_dates": twin_dates,
            "matched_twins": matched_twins,
            "up_tp_atrs": up_pcts,
            "dn_tp_atrs": dn_pcts,
            "tp1_atr": round(tp1_atr, 3),
            "tp2_atr": round(tp2_atr, 3),
            "tp3_atr": round(tp3_atr, 3),
            "tp1": lvls["tp1"],
            "tp2": lvls["tp2"],
            "tp3": lvls["tp3"],
            "sl": lvls["sl"],
        }
