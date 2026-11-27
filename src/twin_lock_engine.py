#!/usr/bin/env python3
"""
True Sheet-Backed Rich `12-Col` Elemental Archive + Exact 4-Method Smart Twin Selector
(`LOCK3` + `GEO` + `STEP` + `LOCK2` + `pickBestSmartNarrowing`)
(`src/twin_lock_engine.py` — `v5.0.0 Production Engine`).

STRICT ARCHITECTURAL INVARIANTS (`lookAheadLeakage == 0`, `priceFreeTwinSignature == True`):
1. Authentic 4,705-row Google Sheet Gold Archive (`data/gold_sheet_history.json` & `data/gold_sheet_signals.json`,
   `PRIM ~31, FINAL ~23, ELE-P ~8–10`, `2012–2027`) with zero-latency local lookup and multi-planet
   (`12 planets × 24 aspects` + `107°/147°/77°/70°` + `36-Decan`) fallback beyond `2027-02-26`.
2. Twin pool for `target_date` ONLY searches completed historical days with `hist_date < target_date`,
   excluding flat noise (`|price| <= 1.0`).
3. Exact 4-Method Smart Twin Selector:
   - `LOCK3` (`run3LayerLockNarrowing`) + `LOCK2` (`PRIM-k + FINAL-k` recovery when `zeroElepCnt >= 2`)
   - `GEO` (`startWithGeo=True`, `4..8` rows, `realPrimFinalSteps > 0`)
   - `STEP` (`startWithGeo=False`, `4..8` rows, `realPrimFinalSteps > 0`)
   - `pickBestSmartNarrowing` Consensus (`حالت ۱: اجماع کامل ۳ روش`, `حالت ۲: اجماع ۲ از ۳`, `حالت ۳: قفل غالب`).
4. True `KXGOLDPERP` Twin Quantile Targets (`TP1 = Q25` bounded `[0.25 * ATR14, 0.45 * ATR14]`, `TP2 = Q50`, `TP3 = Q75`).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.sheet_archive_loader import load_gold_sheet_bundles
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

METHOD_LABELS_FA: dict[str, str] = {
    "GEO": "اول GEO",
    "STEP": "گام‌به‌گام",
    "LOCK3": "🔒 قفل سه‌لایه",
    "LOCK2": "🔒 قفل دولایه",
}

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


def _compute_price_free_decan_ruler_bits(eng_a: dict[str, Any]) -> tuple[str, str]:
    """Compute 100% price-free Chaldean 36-Decan & Ruler 4-bit parity from `xau_pole` + `usd_pole`."""
    xp = eng_a["xau_pole"]
    up = eng_a["usd_pole"]
    dec_geo = "".join(str((int(xp["ele_decf"][j]) + int(up["ele_decf"][j])) % 2) for j in range(4))
    rul_geo = "".join(
        str((int(abs(xp["rul"][j % 2])) + int(abs(up["rul"][j % 2])) + j) % 2) for j in range(4)
    )
    return dec_geo, rul_geo


def build_day_12col_signature(
    eng_a: dict[str, Any],
    eng_b: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build 100% Price-Free Rich `12-Col` Elemental Signature (`PRIM`, `FINAL`, `ELE-P`):
      - Primary (`2012-01-09 .. 2027-02-26`): Authentic Google Sheet Gold matrix (`source = "GOOGLE_SHEET"`).
      - Fallback (outside sheet dates, e.g. `> 2027-02-26`): Multi-planet (`12 planets × 24 aspects` +
        `107°/147°/77°/70°` + `36-Decan` + `v82` `GEO_BITS_4`) scaled to the exact same `ELE-P / FINAL / PRIM` schema.
    """
    _ = eng_b  # Explicitly unused: twin lookup signature is 100% price-invariant!
    d_ctx = eng_a["d_ctx"]
    d_iso = str(d_ctx["date"])
    f1 = int(d_ctx["F1"])
    f2 = int(d_ctx["F2"])
    f3_h = int(d_ctx["F3_HOME"])
    f3_a = int(d_ctx["F3_AWAY"])

    snap_xau = eng_a["snap_xau"]
    snap_usd = eng_a["snap_usd"]

    # Diagnostic celestial aspect & Decan tags (100% price-free)
    step3_hits: list[str] = []
    step3_bonus = [0, 0, 0, 0]
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
                step3_bonus[el_idx_raw] += 1
                step3_hits.append(f"{lbl_name}:{PLANET_NAMES_12[pi]}{ASPECTS_24[ai][0]}->{ELEMENT_ORDER[el_idx_raw][0]}")
        else:
            el_idx, hit_s = _tightest_planet_arcanum_elem(lbl_val, sn["bodies"])
            if el_idx is not None:
                step3_bonus[el_idx] += 1
                step3_hits.append(f"{lbl_name}:{hit_s}->{ELEMENT_ORDER[el_idx][0]}")

    lbma_asc = float(snap_xau["asc"]) % 360.0
    comex_mc = float(snap_usd["mc"]) % 360.0
    moon_lon = float(snap_xau["bodies"]["Moon"]["lon"]) % 360.0
    lbma_pof = float(snap_xau["pof"]) % 360.0

    celestial_tags: list[str] = []
    for cname, cdeg in (
        ("LBMA_ASC", lbma_asc),
        ("COMEX_MC", comex_mc),
        ("MOON", moon_lon),
        ("POF", lbma_pof),
    ):
        dec_info = decan36_info(deg_to_decan36(cdeg))
        r_elem = dec_info["ruler_element"]
        celestial_tags.append(f"{cname}:D{dec_info['decan']:02d}({dec_info['ruler']}->{r_elem[0]})")

    dec_geo, ruler_geo = _compute_price_free_decan_ruler_bits(eng_a)

    # Check authentic Google Sheet archive first (`2012-01-09 .. 2027-02-26`)
    _, _, sheet_by_date = load_gold_sheet_bundles(auto_sync_if_missing=True)
    sr = sheet_by_date.get(d_iso)
    if sr is not None:
        elep = tuple(int(x) for x in sr["elep"])
        final_4 = tuple(int(x) for x in sr["final"])
        prim_4 = tuple(int(x) for x in sr["prim"])
        sig_source = "GOOGLE_SHEET"
    else:
        # Rich multi-planet (`12 planets × 24 aspects` + `107/147/77/70` + `36-Decan` + `v82` `GEO_BITS_4`) fallback
        xp = eng_a["xau_pole"]
        up = eng_a["usd_pole"]
        ed_sum = d_ctx["ele_date_sum"]
        elep_fb = [
            int(xp["ep"][j]) + (1 if (int(up["ep"][j]) >= 2 and step3_bonus[j] >= 1) else 0)
            for j in range(4)
        ]
        elep = tuple(elep_fb)
        final_4 = tuple(int(elep[j] + ed_sum[j]) for j in range(4))
        prim_4 = tuple(int(elep[j] + final_4[j]) for j in range(4))
        sig_source = "EPHEMERIS_FALLBACK"

    zero_elep_cnt = sum(1 for x in elep if x == 0)
    geo_fig = "".join(str(prim_4[k] % 2) for k in range(4))
    geo_name = GEOMANCY_FIGURES.get(geo_fig, "Populus")
    col12 = prim_4 + final_4 + elep

    return {
        "price_free": True,
        "source": sig_source,
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
    Strictly causal walk-forward archive of historical Gold sessions (`hist_date < target_date`),
    implementing the exact 4-Method Smart Twin Selector (`LOCK3`, `GEO`, `STEP`, `LOCK2`)
    and `pickBestSmartNarrowing` consensus with `Q25 / Q50 / Q75` regime-ATR targets.
    """

    def __init__(self) -> None:
        self.by_weekday: dict[str, dict[str, list]] = {}
        self._seen_dates: set[str] = set()
        self._np_cache: dict[str, dict[str, Any]] = {}
        self._dirty: set[str] = set()
        self.lookahead_violations: int = 0
        self.total_queries: int = 0

    def seed_from_sheet_and_records(
        self,
        records_by_date: dict[str, dict[str, Any]],
        max_date_exclusive: str | None = None,
    ) -> int:
        """
        Populate the historical twin archive from `data/gold_sheet_history.json` (`3,834` priced Spot Gold days)
        plus completed recent market sessions (`2026-11-09 .. latest_bar`) where sheet `price` is blank,
        strictly excluding flat noise (`|price| <= 1.0`).
        Note: Every query in `evaluate_twin_and_gaussian(target_date)` still strictly filters `hist_date < target_date`!
        """
        _, _, sheet_by_date = load_gold_sheet_bundles(auto_sync_if_missing=True)
        last_atr = 18.0
        added = 0

        all_dates = sorted(set(sheet_by_date.keys()) | set(records_by_date.keys()))
        for d_iso in all_dates:
            if max_date_exclusive is not None and d_iso >= max_date_exclusive:
                continue
            r_mkt = records_by_date.get(d_iso)
            if r_mkt is not None:
                last_atr = float(r_mkt["atr14"])

            sr = sheet_by_date.get(d_iso)
            if sr is None:
                continue

            pr = sr.get("price")
            if pr is None and r_mkt is not None:
                pr = float(r_mkt["_target_move"])
            if pr is None or abs(float(pr)) <= 1.0:
                continue

            move_val = float(pr)
            dir_sgn = 1 if move_val > 0.0 else -1
            prim_4 = tuple(int(x) for x in sr["prim"])
            final_4 = tuple(int(x) for x in sr["final"])
            elep_4 = tuple(int(x) for x in sr["elep"])
            geo_fig = "".join(str(p % 2) for p in prim_4)
            d_bits = "".join(str((final_4[k] + elep_4[(k + 1) % 4] + 1) % 2) for k in range(4))
            r_bits = "".join(str((prim_4[k] + final_4[(k + 2) % 4]) % 2) for k in range(4))

            sig_stub = {
                "col12": prim_4 + final_4 + elep_4,
                "geo_fig": geo_fig,
                "dec_geo": d_bits,
                "ruler_geo": r_bits,
            }

            if r_mkt is not None:
                op = max(1.0, float(r_mkt["today_open"]))
                hi = max(op, float(r_mkt["_target_high"]))
                lo = min(op, float(r_mkt["_target_low"]))
                fav_up_atr = max(0.0, hi - op) / max(1.0, last_atr)
                fav_dn_atr = max(0.0, op - lo) / max(1.0, last_atr)
                fav_up_pct = max(0.0, hi - op) / op
                fav_dn_pct = max(0.0, op - lo) / op
            else:
                b_atr = abs(move_val) / max(1.0, last_atr)
                fav_up_atr = b_atr if dir_sgn > 0 else 0.15
                fav_dn_atr = b_atr if dir_sgn < 0 else 0.15
                fav_up_pct = fav_up_atr * 0.01
                fav_dn_pct = fav_dn_atr * 0.01

            self.add_completed_day(
                date_iso=d_iso,
                weekday_up=str(sr["day"]).upper(),
                sig=sig_stub,
                direction_sign=dir_sgn,
                move_dollars=move_val,
                fav_up_atr=fav_up_atr,
                fav_dn_atr=fav_dn_atr,
                is_noise_flat=False,
                fav_up_pct=fav_up_pct,
                fav_dn_pct=fav_dn_pct,
                hist_atr14=last_atr,
            )
            added += 1
        return added

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
        """Ingest a COMPLETED historical session (`|move| > 1.0`), maintaining strict chronological sort."""
        if is_noise_flat or direction_sign == 0 or abs(float(move_dollars)) <= 1.0:
            return
        if date_iso in self._seen_dates:
            return
        self._seen_dates.add(date_iso)

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

        atr_est = (
            float(hist_atr14)
            if (hist_atr14 is not None and hist_atr14 > 0.5)
            else max(1.0, abs(move_dollars) / max(0.25, max(fav_up_atr, fav_dn_atr)))
        )
        body_atr = max(0.08, abs(float(move_dollars)) / atr_est)

        # Insert in sorted date order (fast append in chronological walk-forward)
        ins_pos = len(b["dates"])
        if ins_pos > 0 and b["dates"][-1] > date_iso:
            import bisect
            ins_pos = bisect.bisect_left(b["dates"], date_iso)

        b["dates"].insert(ins_pos, date_iso)
        b["year_float"].insert(ins_pos, _iso_to_year_float(date_iso))
        b["col12"].insert(ins_pos, tuple(int(x) for x in sig["col12"]))
        b["geo_fig"].insert(ins_pos, int(sig["geo_fig"], 2))
        b["dec_geo"].insert(ins_pos, int(sig["dec_geo"], 2))
        b["ruler_geo"].insert(ins_pos, int(sig["ruler_geo"], 2))
        b["dir"].insert(ins_pos, int(direction_sign))
        b["move"].insert(ins_pos, float(move_dollars))
        b["body_atr"].insert(ins_pos, float(body_atr))
        b["fav_up_atr"].insert(ins_pos, float(fav_up_atr))
        b["fav_dn_atr"].insert(ins_pos, float(fav_dn_atr))
        b["fav_up_pct"].insert(ins_pos, float(fav_up_pct if fav_up_pct is not None else (fav_up_atr * 0.012)))
        b["fav_dn_pct"].insert(ins_pos, float(fav_dn_pct if fav_dn_pct is not None else (fav_dn_atr * 0.012)))
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
    def _run_progressive_narrowing(
        idx_start: np.ndarray,
        col12_all: np.ndarray,
        gf_all: np.ndarray,
        dir_all: np.ndarray,
        t_col12: np.ndarray,
        t_gf: int,
        zero_elep_cnt: int,
        start_with_geo: bool,
        min_rows: int = 4,
        max_rows: int = 8,
        preset_used: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        """
        True largest-count progressive narrowing (`GEO` when `start_with_geo=True`, `STEP` when `False`)
        stopping in `4..8` rows (`realPrimFinalSteps > 0`, skipping `"0 "` columns when `zeroElepCnt >= 1`,
        avoiding false sign flips).
        """
        idx = idx_start.copy()
        steps: list[str] = []
        if start_with_geo:
            g_idx = idx[gf_all[idx] == t_gf]
            if len(g_idx) >= 3:
                idx = g_idx
                steps.append("GEO")

        used = np.zeros(12, dtype=bool)
        if preset_used is not None:
            for c in preset_used:
                used[c] = True

        elig = np.ones(12, dtype=bool)
        if zero_elep_cnt >= 1:
            for k in range(4):
                if t_col12[8 + k] == 0:
                    elig[8 + k] = False
                if t_col12[k] == 0:
                    elig[k] = False
                if t_col12[4 + k] == 0:
                    elig[4 + k] = False

        real_pf_steps = int(np.sum(used[:8]))

        while len(idx) > max_rows:
            cur_n = len(idx)
            cur_u = int(np.sum(dir_all[idx] > 0))
            cur_d = cur_n - cur_u
            cur_sgn = 1 if cur_u > cur_d else (-1 if cur_d > cur_u else 0)
            cur_strong = (cur_n <= 14) and (abs(cur_u - cur_d) >= 3)

            eq_mat = col12_all[idx] == t_col12
            cnts = np.sum(eq_mat, axis=0)
            avail = elig & (~used) & (cnts >= min_rows) & (cnts < cur_n)
            if not np.any(avail):
                break

            # Enforce realPrimFinalSteps > 0: prefer PRIM/FINAL (0..7) over ELE-P (8..11) whenever available
            if np.any(avail[:8]):
                avail_step = avail.copy()
                avail_step[8:] = False
            else:
                avail_step = avail

            best_key: tuple[int, int] | None = None
            best_c: int | None = None
            for c in np.flatnonzero(avail_step):
                sub_d = dir_all[idx[eq_mat[:, c]]]
                u = int(np.sum(sub_d > 0))
                d = len(sub_d) - u
                nsgn = 1 if u > d else (-1 if d > u else 0)
                no_flip = not (cur_strong and nsgn != 0 and nsgn != cur_sgn)
                key = (int(no_flip), int(cnts[c]))
                if best_key is None or key > best_key:
                    best_key = key
                    best_c = int(c)

            if best_c is None or best_key is None:
                break
            if cur_n <= 10 and best_key[0] == 0:
                break

            used[best_c] = True
            if best_c < 8:
                real_pf_steps += 1
            idx = idx[eq_mat[:, best_c]]
            steps.append(f"c{best_c}({len(idx)})")
            if min_rows <= len(idx) <= max_rows:
                break

        n = len(idx)
        ups = int(np.sum(dir_all[idx] > 0))
        dns = n - ups
        margin = abs(ups - dns)
        minr = min(ups, dns)
        win_pct = round(max(ups, dns) * 100.0 / max(1, n), 1)
        d_str = "XAU" if ups > dns else ("USD" if dns > ups else "TIE")
        method = "GEO" if start_with_geo else "STEP"
        return {
            "method": method,
            "labelFa": METHOD_LABELS_FA[method],
            "idx": idx,
            "n": n,
            "ups": ups,
            "dns": dns,
            "margin": margin,
            "minr": minr,
            "win_pct": win_pct,
            "dir": d_str,
            "real_pf_steps": real_pf_steps,
            "steps": steps,
        }

    @classmethod
    def _run_3layer_lock_narrowing(
        cls,
        idx_all: np.ndarray,
        col12_all: np.ndarray,
        gf_all: np.ndarray,
        dir_all: np.ndarray,
        t_col12: np.ndarray,
        t_gf: int,
        zero_elep_cnt: int,
        dec_geo: str,
        rul_geo: str,
    ) -> dict[str, Any]:
        """
        `run3LayerLockNarrowing` (`LOCK3` + `LOCK2` recovery):
          - Filters `GEO-FIG` first (if `>= 3` rows), testing all 4 element columns `k in 0..3`
            (`FIRE, AIR, WATER, EARTH`) for simultaneous 3-layer exact match (`PRIM-k + FINAL-k + ELE-P-k`).
          - Scores candidates by:
              * `posLayers == 3` (`+95`) vs `0-layer penalty` (`-120` on Gold, or `-20` `LOCK2` relief when `zeroElepCnt >= 2`)
              * `Decan 1/1` (`+45`) vs `Decan 0/0` (`-35`)
              * `min(ups, downs) == 0` trap (`-150`)
              * Narrowing into `4..10` rows (`>= 66%` majority, `margin >= 2`).
          - Falls back to same-weekday history without `GEO-FIG` pre-filter when `GEO-FIG` produces no valid lock.
        """
        g_idx = idx_all[gf_all[idx_all] == t_gf]
        passes: list[tuple[str, np.ndarray]] = [("GEO", g_idx)] if len(g_idx) >= 3 else []
        passes.append(("ALL", idx_all))

        best_overall: dict[str, Any] | None = None
        best_overall_sc = -1e9

        for base_tag, b_idx in passes:
            pass_best: dict[str, Any] | None = None
            pass_best_sc = -1e9

            for k in range(4):
                m_k = (
                    (col12_all[b_idx, k] == t_col12[k])
                    & (col12_all[b_idx, 4 + k] == t_col12[4 + k])
                    & (col12_all[b_idx, 8 + k] == t_col12[8 + k])
                )
                s0 = b_idx[m_k]
                if len(s0) < 4:
                    continue

                direct_lock = 4 <= len(s0) <= 10
                if len(s0) > 10:
                    pr = cls._run_progressive_narrowing(
                        s0, col12_all, gf_all, dir_all, t_col12, t_gf, zero_elep_cnt,
                        start_with_geo=False, min_rows=4, max_rows=10, preset_used=(k, 4 + k, 8 + k),
                    )
                    sn = pr["idx"]
                else:
                    sn = s0
                if len(sn) > 12:
                    sn = sn[-10:]

                n = len(sn)
                ups = int(np.sum(dir_all[sn] > 0))
                dns = n - ups
                margin = abs(ups - dns)
                minr = min(ups, dns)
                wp = max(ups, dns) * 100.0 / max(1, n)

                pos_layers = int(t_col12[k] > 0) + int(t_col12[4 + k] > 0) + int(t_col12[8 + k] > 0)
                dec11 = (dec_geo[k] == "1") and (rul_geo[k] == "1")
                dec00 = (dec_geo[k] == "0") and (rul_geo[k] == "0")

                sc = 0.0
                if pos_layers == 3:
                    sc += 95.0
                elif zero_elep_cnt >= 2 and t_col12[k] > 0 and t_col12[4 + k] > 0:
                    sc -= 20.0
                else:
                    sc -= 120.0

                if dec11:
                    sc += 45.0
                elif dec00:
                    sc -= 35.0

                if minr == 0:
                    sc -= 150.0
                if direct_lock:
                    sc += 40.0

                if wp >= 66.0 and margin >= 2:
                    sc += 65.0 + (wp - 66.0) * 1.6 + margin * 11.0
                elif margin == 0:
                    sc -= 90.0
                else:
                    sc += (wp - 50.0) * 1.0 + margin * 4.0

                is_lock2 = (pos_layers < 3) and (zero_elep_cnt >= 2) and (t_col12[k] > 0) and (t_col12[4 + k] > 0)
                m_tag = "LOCK2" if is_lock2 else "LOCK3"
                cand = {
                    "method": m_tag,
                    "labelFa": METHOD_LABELS_FA[m_tag],
                    "idx": sn,
                    "n": n,
                    "ups": ups,
                    "dns": dns,
                    "margin": margin,
                    "minr": minr,
                    "win_pct": round(wp, 1),
                    "dir": "XAU" if ups > dns else ("USD" if dns > ups else "TIE"),
                    "lock_col": k,
                    "lock_elem": ELEMENT_ORDER[k],
                    "pos_layers": pos_layers,
                    "dec11": dec11,
                    "dec00": dec00,
                    "base_tag": base_tag,
                    "direct_lock": direct_lock,
                    "score": round(sc, 1),
                }
                if sc > pass_best_sc:
                    pass_best_sc = sc
                    pass_best = cand
                if sc > best_overall_sc:
                    best_overall_sc = sc
                    best_overall = cand

            if (
                base_tag == "GEO"
                and pass_best is not None
                and pass_best["win_pct"] >= 66.0
                and pass_best["margin"] >= 2
                and pass_best["score"] > 0.0
            ):
                return pass_best

        if best_overall is not None:
            return best_overall

        fb = cls._run_progressive_narrowing(
            idx_all, col12_all, gf_all, dir_all, t_col12, t_gf, zero_elep_cnt,
            start_with_geo=True, min_rows=4, max_rows=10,
        )
        m_fb = "LOCK2" if zero_elep_cnt >= 2 else "LOCK3"
        fb.update(
            {
                "method": m_fb,
                "labelFa": METHOD_LABELS_FA[m_fb],
                "lock_col": 0,
                "lock_elem": ELEMENT_ORDER[0],
                "pos_layers": 0,
                "dec11": False,
                "dec00": False,
                "base_tag": "FB",
                "direct_lock": False,
                "score": -200.0,
            }
        )
        return fb

    @staticmethod
    def _pick_best_smart_narrowing(
        geo_r: dict[str, Any],
        step_r: dict[str, Any],
        lock_r: dict[str, Any],
    ) -> tuple[dict[str, Any], bool, int, str, str]:
        """
        `pickBestSmartNarrowing` Consensus:
          1. If `GEO` and `STEP` agree (`margin >= 2`) against `LOCK3`, pick `STEP`.
          2. If `GEO` is tie/`margin <= 1` and `STEP` has `margin >= 2` (`ms > ml` or `ml <= 2`), pick `STEP`.
          3. If `GEO` has `margin >= 2` & `winPct >= 66%` while `LOCK3` has `ml <= 1`, pick `GEO`.
          4. Otherwise pick `LOCK3` (or `LOCK2` if `zeroElepCnt >= 2` and `LOCK3` is weak).
        """
        mg, ms, ml = int(geo_r["margin"]), int(step_r["margin"]), int(lock_r["margin"])
        dg, ds, dl = str(geo_r["dir"]), str(step_r["dir"]), str(lock_r["dir"])

        g_valid = geo_r["minr"] > 0 or geo_r["n"] > 6
        s_valid = step_r["minr"] > 0 and step_r["real_pf_steps"] > 0
        all3_agree = (dg in ("XAU", "USD")) and (dg == ds == dl)

        if dg in ("XAU", "USD") and dg == ds and mg >= 2 and ms >= 2 and s_valid and dl != ds:
            chosen = step_r
            rule_code = "R1_GEO_STEP_AGREE_VS_LOCK3"
            rule_fa = "توافق GEO و گام‌به‌گام (حاشیه ≥ ۲) در برابر قفل سه‌لایه ← انتخاب گام‌به‌گام"
        elif mg <= 1 and ms >= 2 and s_valid and ds in ("XAU", "USD") and (ms > ml or ml <= 2):
            chosen = step_r
            rule_code = "R2_STEP_DOMINANT_MARGIN"
            rule_fa = "تساوی/ضعف GEO و برتری قاطع گام‌به‌گام (حاشیه ≥ ۲) ← انتخاب گام‌به‌گام"
        elif mg >= 2 and geo_r["win_pct"] >= 66.0 and g_valid and dg in ("XAU", "USD") and ml <= 1:
            chosen = geo_r
            rule_code = "R3_GEO_STRONG_LOCK3_WEAK"
            rule_fa = "برتری GEO (≥۶۶٪ و حاشیه ≥ ۲) در برابر ضعف قفل سه‌لایه ← انتخاب اول GEO"
        else:
            chosen = lock_r
            if lock_r["method"] == "LOCK2":
                rule_code = "R4_LOCK2_RECOVERY"
                rule_fa = "بازیابی هوشمند قفل دولایه (PRIM+FINAL در حضور Zero ELE-P ≥ 2)"
            else:
                rule_code = "R4_LOCK3_PRIMARY"
                rule_fa = "حاکمیت قفل سه‌لایه ستونی (LOCK3: تطابق هم‌زمان PRIM + FINAL + ELE-P)"

        valid_dirs = [d for d in (dg, ds, dl) if d in ("XAU", "USD")]
        agree_cnt = sum(1 for d in valid_dirs if d == chosen["dir"])
        consensus_mode = 1 if all3_agree else (2 if agree_cnt >= 2 else 3)
        return chosen, all3_agree, consensus_mode, rule_code, rule_fa

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
        Execute strictly causal (`hist_date < target_date`) 4-Method Smart Twin Selector
        (`LOCK3` + `GEO` + `STEP` + `LOCK2`) + `Q25 / Q50 / Q75` Winning Twin Quantiles.
        """
        self.total_queries += 1
        s2_eff = float(s2 if s2 is not None else (s1 - 0.50 * atr14))
        r2_eff = float(r2 if r2 is not None else (r1 + 0.50 * atr14))

        arr = self._sync_weekday(weekday_up)
        valid_n = 0
        if arr is not None:
            if arr["max_date"] >= target_date:
                valid_n = int(np.searchsorted(arr["dates"], target_date, side="left"))
                if any(d >= target_date for d in arr["dates"][:valid_n]):
                    self.lookahead_violations += 1
                    raise AssertionError(f"Look-ahead violation detected on {target_date}")
            else:
                valid_n = len(arr["dates"])

        if arr is None or valid_n < 8:
            lvls = compute_adaptive_causal_levels(
                fallback_dir, entry_open, atr14, s2_eff, r2_eff,
                s1_level=s1, r1_level=r1,
                twin_q25_atr=0.28, twin_q50_atr=0.56, twin_q75_atr=0.88,
            )
            empty_m = {"method": "NONE", "labelFa": "—", "n": 0, "ups": 0, "dns": 0, "margin": 0, "winPct": 50.0, "dir": "TIE", "votes": "0 - 0"}
            return {
                "has_twins": False,
                "twin_count": 0,
                "up_twins": 0,
                "down_twins": 0,
                "max_votes": 0,
                "min_opp_votes": 0,
                "margin": 0,
                "tw_pct": 50.0,
                "dir_tw_pct": 50.0,
                "w_disp_pct": 50.0,
                "dir_w_pct": 50.0,
                "twin_win_prob": 0.50,
                "twin_dir": fallback_dir,
                "selected_method": "LOCK3",
                "selected_method_fa": METHOD_LABELS_FA["LOCK3"],
                "all3_agree": False,
                "consensus_mode": 3,
                "consensus_badge_fa": "حالت ۳: داده همزاد اولیه",
                "rule_reason": "WARMUP_FALLBACK",
                "rule_reason_fa": "دوره گرم‌شدن آرشیو همزادها",
                "three_method_comparison": {
                    "geo": empty_m,
                    "step": empty_m,
                    "lock3": empty_m,
                    "selectedMethod": "LOCK3",
                    "selectedLabelFa": METHOD_LABELS_FA["LOCK3"],
                    "all3Agree": False,
                    "consensusMode": 3,
                    "consensusBadgeFa": "حالت ۳: داده همزاد اولیه",
                    "ruleReason": "WARMUP_FALLBACK",
                    "ruleReasonFa": "دوره گرم‌شدن آرشیو همزادها",
                },
                "strong_avg_dominance": False,
                "maj_avg_atr_move": 0.0,
                "min_avg_atr_move": 0.0,
                "net_dollar_sum": 0.0,
                "avg_up_usd": 0.0,
                "avg_dn_usd": 0.0,
                "in_golden_zone": False,
                "lock3_cols": [],
                "has_lock3": False,
                "lock3_peak": False,
                "high_purity_lock3_peak": False,
                "deep_reversal_lock": False,
                "active_lock_col": 0,
                "active_lock_elem": ELEMENT_ORDER[0],
                "pos_layers": 0,
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
                "up_tp_atrs": (0.28, 0.56, 0.88),
                "dn_tp_atrs": (0.28, 0.56, 0.88),
                "tp1_atr": 0.28,
                "tp2_atr": 0.56,
                "tp3_atr": 0.88,
                "tp1": lvls["tp1"],
                "tp2": lvls["tp2"],
                "tp3": lvls["tp3"],
                "sl": lvls["sl"],
            }

        dates_all = arr["dates"][:valid_n]
        if valid_n > 0 and not (dates_all[valid_n - 1] < target_date):
            self.lookahead_violations += 1
            raise AssertionError(f"Look-ahead leakage: {dates_all[valid_n - 1]} >= {target_date}")

        col12_all = arr["col12"][:valid_n]
        gf_all = arr["geo_fig"][:valid_n]
        dec_all = arr["dec_geo"][:valid_n]
        rul_all = arr["ruler_geo"][:valid_n]
        dir_all = arr["dir"][:valid_n]
        dir_f = dir_all.astype(np.float64)
        move_all = arr["move"][:valid_n]
        body_atr_all = arr["body_atr"][:valid_n].astype(np.float64)
        year_all = arr["year_float"][:valid_n]

        target_yf = _iso_to_year_float(target_date)
        years_ago_all = np.maximum(0.0, target_yf - year_all)
        recency_all = np.exp(-years_ago_all / 5.5)

        t_col12 = np.asarray(sig["col12"], dtype=np.int16)
        t_gf = int(sig["geo_fig"], 2)
        t_dec = int(sig["dec_geo"], 2)
        t_rul = int(sig["ruler_geo"], 2)
        zero_elep_cnt = int(sig["zero_elep_cnt"])
        idx_all = np.arange(valid_n, dtype=np.int64)

        # 1. Execute 3 Primary Narrowing Engines (`GEO`, `STEP`, `LOCK3/LOCK2`) + Smart Consensus
        geo_r = self._run_progressive_narrowing(
            idx_all, col12_all, gf_all, dir_all, t_col12, t_gf, zero_elep_cnt,
            start_with_geo=True, min_rows=4, max_rows=8,
        )
        step_r = self._run_progressive_narrowing(
            idx_all, col12_all, gf_all, dir_all, t_col12, t_gf, zero_elep_cnt,
            start_with_geo=False, min_rows=4, max_rows=8,
        )
        lock_r = self._run_3layer_lock_narrowing(
            idx_all, col12_all, gf_all, dir_all, t_col12, t_gf, zero_elep_cnt,
            sig["dec_geo"], sig["ruler_geo"],
        )

        chosen, all3_agree, consensus_mode, rule_code, rule_fa = self._pick_best_smart_narrowing(
            geo_r, step_r, lock_r
        )
        selected_method = str(chosen["method"])
        final_idx = chosen["idx"]

        # 2. Recency-Weighted Gaussian Priority Diagnostics across PRIM/FINAL
        geo_match_bonus = np.where(gf_all == t_gf, 1.35, 0.75) * recency_all
        dec_match_bonus = np.where(dec_all == t_dec, 1.25, 0.85) * recency_all

        fp_list: list[float] = []
        ff_list: list[float] = []
        fp_weights: list[float] = []
        ff_weights: list[float] = []

        for k in range(4):
            diff_p = (col12_all[:, k].astype(np.float64) - float(t_col12[k])) / 1.40
            wp = np.exp(-0.5 * diff_p * diff_p) * geo_match_bonus
            swp = float(np.sum(wp))
            fp_k = float(np.sum(wp * dir_f) / swp) if swp > 1e-6 else 0.0
            fp_list.append(round(fp_k, 4))
            fp_weights.append(abs(fp_k) * min(3.0, math.sqrt(swp / 6.0)))

            diff_f = (col12_all[:, 4 + k].astype(np.float64) - float(t_col12[4 + k])) / 1.15
            wf = np.exp(-0.5 * diff_f * diff_f) * dec_match_bonus
            swf = float(np.sum(wf))
            ff_k = float(np.sum(wf * dir_f) / swf) if swf > 1e-6 else 0.0
            ff_list.append(round(ff_k, 4))
            ff_weights.append(abs(ff_k) * min(3.0, math.sqrt(swf / 6.0)))

        tot_gw = sum(fp_weights) + sum(ff_weights)
        if tot_gw > 1e-6:
            gauss_score = (
                sum(fp_list[k] * fp_weights[k] for k in range(4))
                + sum(ff_list[k] * ff_weights[k] for k in range(4))
            ) / tot_gw
        else:
            gauss_score = float(np.mean(fp_list + ff_list))

        # 3. Consensus Statistics & Recency-ATR Weighting inside `final_idx`
        n_twins = int(chosen["n"])
        up_twins = int(chosen["ups"])
        dn_twins = int(chosen["dns"])
        margin = int(chosen["margin"])
        max_votes = max(up_twins, dn_twins)
        min_opp = min(up_twins, dn_twins)
        tw_pct = float(chosen["win_pct"])

        d_twins = dir_f[final_idx]
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

        gauss_up_prob = 0.50 + 0.50 * float(np.clip(gauss_score * 1.50, -0.35, 0.35))
        p_xau_base = 0.60 * raw_up_prob + 0.30 * w_up_prob + 0.10 * gauss_up_prob

        if chosen["dir"] in ("XAU", "USD"):
            dir_c = str(chosen["dir"])
            twin_dir = dir_c
        else:
            dir_c = "XAU" if p_xau_base >= 0.50 else "USD"
            twin_dir = "TIE"

        dir_tw_pct = round((raw_up_prob if dir_c == "XAU" else (1.0 - raw_up_prob)) * 100.0, 1)
        dir_w_pct = round((w_up_prob if dir_c == "XAU" else (1.0 - w_up_prob)) * 100.0, 1)

        maj_mask = (d_twins > 0) if dir_c == "XAU" else (d_twins < 0)
        maj_avg_atr = float(np.mean(atr_sub[maj_mask])) if np.any(maj_mask) else 0.50
        min_avg_atr = float(np.mean(atr_sub[~maj_mask])) if np.any(~maj_mask) else 0.35
        strong_avg_dom = bool(np.any(maj_mask) and (maj_avg_atr >= 1.35 * max(0.08, min_avg_atr)))

        active_col = int(lock_r.get("lock_col", 0))
        pos_layers = int(lock_r.get("pos_layers", 0))
        decan_11 = bool(lock_r.get("dec11", False)) if selected_method in ("LOCK3", "LOCK2") else False
        decan_00 = bool(lock_r.get("dec00", False)) if selected_method in ("LOCK3", "LOCK2") else False

        has_lock3 = bool(selected_method in ("LOCK3", "LOCK2") or lock_r["margin"] >= 2)
        lock3_cols = [ELEMENT_ORDER[active_col]] if has_lock3 else []

        in_golden_zone = bool((4 <= n_twins <= 10) and (min_opp >= 1) and (tw_pct >= 70.0) and (margin >= 2))
        is_lock3_peak = bool(
            (selected_method in ("LOCK3", "LOCK2", "GEO"))
            and (min_opp >= 1)
            and (margin >= 3)
            and (tw_pct >= 75.0)
        )
        all3_strong = bool(all3_agree and (margin >= 2) and (tw_pct >= 66.0) and (min_opp >= 1))
        high_purity_lock3_peak = bool(is_lock3_peak and (tw_pct >= 77.5) and (margin >= 4))
        deep_reversal_lock = bool(
            (min_opp >= 1)
            and (margin >= 5)
            and (tw_pct >= 77.0)
            and (selected_method in ("LOCK3", "LOCK2"))
        )
        trap_100 = bool((n_twins > 0) and (n_twins <= 8) and (min_opp == 0))

        maj_base_prob = p_xau_base if dir_c == "XAU" else (1.0 - p_xau_base)
        parity_adj = (
            (0.045 if all3_agree else 0.0)
            + (0.045 if decan_11 else 0.0)
            + (-0.055 if decan_00 else 0.0)
            + (0.025 if strong_avg_dom else 0.0)
        )
        twin_win_prob = float(np.clip(maj_base_prob + parity_adj, 0.36, 0.92))
        score_c = (twin_win_prob - 0.50) * 2.0 * (1.0 if dir_c == "XAU" else -1.0)

        # 4. True `KXGOLDPERP` `Q25 / Q50 / Q75` Winning Twin Quantiles (scaled by today's ATR14)
        def _calc_twin_quantiles(is_up: bool) -> tuple[float, float, float]:
            wm = (d_twins > 0) if is_up else (d_twins < 0)
            mv = atr_sub[wm] if int(np.sum(wm)) >= 2 else atr_sub
            if len(mv) > 0:
                q25_raw = float(np.percentile(mv, 25))
                q50_raw = float(np.percentile(mv, 50))
                q75_raw = float(np.percentile(mv, 75))
                q25 = float(np.clip(0.72 * q25_raw, 0.25, 0.45))
                q50 = float(np.clip(q50_raw, max(q25 + 0.14, 0.45), 0.95))
                q75 = float(np.clip(q75_raw, max(q50 + 0.18, 0.70), 1.50))
                return round(q25, 3), round(q50, 3), round(q75, 3)
            return 0.28, 0.56, 0.88

        up_qs = _calc_twin_quantiles(True)
        dn_qs = _calc_twin_quantiles(False)
        tp1_atr, tp2_atr, tp3_atr = up_qs if dir_c == "XAU" else dn_qs

        lvls = compute_adaptive_causal_levels(
            direction=dir_c,
            entry_open=entry_open,
            atr14=atr14,
            s2_level=s2_eff,
            r2_level=r2_eff,
            s1_level=float(s1),
            r1_level=float(r1),
            twin_q25_atr=tp1_atr,
            twin_q50_atr=tp2_atr,
            twin_q75_atr=tp3_atr,
        )

        mode_badges_fa = {
            1: "حالت ۱: اجماع کامل هر ۳ روش ✨",
            2: "حالت ۲: اجماع ۲ از ۳ روش ✓",
            3: "حالت ۳: انتخاب هوشمند قفل غالب 🔒",
        }
        consensus_badge_fa = mode_badges_fa.get(consensus_mode, mode_badges_fa[3])

        def _fmt_method_card(mr: dict[str, Any], is_selected: bool) -> dict[str, Any]:
            card = {
                "method": mr["method"],
                "labelFa": mr["labelFa"],
                "selected": is_selected,
                "n": mr["n"],
                "ups": mr["ups"],
                "dns": mr["dns"],
                "votes": f"{mr['ups']} XAU - {mr['dns']} USD",
                "winPct": mr["win_pct"],
                "margin": mr["margin"],
                "minr": mr["minr"],
                "dir": mr["dir"],
                "dirFa": "صعودی (LONG)" if mr["dir"] == "XAU" else ("نزولی (SHORT)" if mr["dir"] == "USD" else "تساوی (TIE)"),
            }
            if "lock_elem" in mr:
                card["lockElem"] = mr["lock_elem"]
                card["posLayers"] = mr["pos_layers"]
                card["dec11"] = mr["dec11"]
                card["dec00"] = mr["dec00"]
                card["baseTag"] = mr["base_tag"]
            if "real_pf_steps" in mr:
                card["realPfSteps"] = mr["real_pf_steps"]
            return card

        three_method_comparison = {
            "geo": _fmt_method_card(geo_r, selected_method == "GEO"),
            "step": _fmt_method_card(step_r, selected_method == "STEP"),
            "lock3": _fmt_method_card(lock_r, selected_method in ("LOCK3", "LOCK2")),
            "selectedMethod": selected_method,
            "selectedLabelFa": METHOD_LABELS_FA.get(selected_method, selected_method),
            "all3Agree": bool(all3_agree),
            "consensusMode": consensus_mode,
            "consensusBadgeFa": consensus_badge_fa,
            "ruleReason": rule_code,
            "ruleReasonFa": rule_fa,
        }

        twin_dates = [dates_all[int(i)] for i in final_idx[-10:]]
        matched_twins: list[dict[str, Any]] = []
        up_usd_vals: list[float] = []
        dn_usd_vals: list[float] = []
        net_dollar_sum = 0.0
        for j_pos, idx_m in enumerate(final_idx):
            im = int(idx_m)
            mv_u = float(move_all[im])
            net_dollar_sum += mv_u
            if dir_all[im] > 0:
                up_usd_vals.append(abs(mv_u))
            elif dir_all[im] < 0:
                dn_usd_vals.append(abs(mv_u))
            matched_twins.append(
                {
                    "date": dates_all[im],
                    "weekday": weekday_up.title(),
                    "direction": "XAU" if dir_all[im] > 0 else "USD",
                    "move_usd": round(mv_u, 2),
                    "move_atr": round(float(body_atr_all[im]), 2),
                    "col12_matches": int(np.sum(col12_all[im] == t_col12)),
                    "decan_sim": round(float(dec_sim_sub[j_pos]), 2),
                    "weight": round(float(w_sub[j_pos]), 3),
                }
            )
        avg_up_usd = float(np.mean(up_usd_vals)) if up_usd_vals else 0.0
        avg_dn_usd = float(np.mean(dn_usd_vals)) if dn_usd_vals else 0.0

        return {
            "has_twins": n_twins >= 4,
            "twin_count": n_twins,
            "up_twins": up_twins,
            "down_twins": dn_twins,
            "max_votes": max_votes,
            "min_opp_votes": min_opp,
            "margin": margin,
            "tw_pct": round(tw_pct, 1),
            "dir_tw_pct": dir_tw_pct,
            "w_disp_pct": dir_w_pct,
            "dir_w_pct": dir_w_pct,
            "twin_win_prob": round(twin_win_prob, 4),
            "twin_dir": twin_dir,
            "selected_method": selected_method,
            "selected_method_fa": METHOD_LABELS_FA.get(selected_method, selected_method),
            "all3_agree": bool(all3_agree),
            "all3_strong": bool(all3_strong),
            "consensus_mode": consensus_mode,
            "consensus_badge_fa": consensus_badge_fa,
            "rule_reason": rule_code,
            "rule_reason_fa": rule_fa,
            "three_method_comparison": three_method_comparison,
            "strong_avg_dominance": strong_avg_dom,
            "maj_avg_atr_move": round(maj_avg_atr, 3),
            "min_avg_atr_move": round(min_avg_atr, 3),
            "net_dollar_sum": round(net_dollar_sum, 2),
            "avg_up_usd": round(avg_up_usd, 2),
            "avg_dn_usd": round(avg_dn_usd, 2),
            "in_golden_zone": bool(in_golden_zone),
            "lock3_cols": lock3_cols,
            "has_lock3": bool(has_lock3),
            "lock3_peak": bool(is_lock3_peak),
            "high_purity_lock3_peak": bool(high_purity_lock3_peak),
            "deep_reversal_lock": bool(deep_reversal_lock),
            "active_lock_col": active_col,
            "active_lock_elem": ELEMENT_ORDER[active_col],
            "pos_layers": pos_layers,
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
            "up_tp_atrs": up_qs,
            "dn_tp_atrs": dn_qs,
            "tp1_atr": round(tp1_atr, 3),
            "tp2_atr": round(tp2_atr, 3),
            "tp3_atr": round(tp3_atr, 3),
            "tp1": lvls["tp1"],
            "tp2": lvls["tp2"],
            "tp3": lvls["tp3"],
            "sl": lvls["sl"],
        }
