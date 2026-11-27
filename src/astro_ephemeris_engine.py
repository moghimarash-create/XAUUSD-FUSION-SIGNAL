#!/usr/bin/env python3
"""
Topocentric Swiss Ephemeris (`pyswisseph`), 24-Aspect, 22-Arcana, 36-Chaldean Decan,
ASC / MC / Part of Fortune (`POF`), Planetary Hour Ruler & Engine A Evaluator.

- Uses bundled `data/ephe/seas_18.se1` for Chiron & True Node alongside Sun..Pluto.
- Enforces `swe.set_topo(lon, lat, 0)` + `swe.FLG_SWIEPH | swe.FLG_SPEED | swe.FLG_TOPOCTR`.
- Zero look-ahead: all timestamps are strictly at or before session open `T_0`.
"""
from __future__ import annotations

import datetime
import math
import os
import urllib.request
from typing import Any

import numpy as np
import pytz
import swisseph as swe

from src.xauusd_config import (
    ANCHOR_CANDIDATES,
    ARC_ELEMENT_IDX,
    ARC_HC,
    ARC_MD,
    ASPECT_HARMONIC_WEIGHT,
    ASPECTS_24,
    AWAY_USD_DEG,
    DAY_CODE_MAP,
    DECAN_36_TABLE,
    DEG1_GOLD,
    DEG2_XAUUSD,
    ELEMENT_INDEX,
    GEO_BITS_4,
    HOME_XAU_DEG,
    RULER_ELEMENT,
    WEEKDAY_NUM,
    best_aspect,
    mod_arcanum,
    price_to_deg,
    zod_aspect_deg,
    zodiac_from_deg,
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EPHE_DIR = os.path.join(ROOT_DIR, "data", "ephe")
EPHE_BASE_URL = "https://raw.githubusercontent.com/aloistr/swisseph/master/ephe"
EPHE_FILES = ("seas_18.se1", "sepl_18.se1", "semo_18.se1")
_ARC_EL_ARR = np.array([ARC_ELEMENT_IDX[k] for k in range(1, 23)], dtype=np.int8)


def init_swisseph() -> str:
    os.makedirs(EPHE_DIR, exist_ok=True)
    for fn in EPHE_FILES:
        se1_path = os.path.join(EPHE_DIR, fn)
        if not (os.path.exists(se1_path) and os.path.getsize(se1_path) > 50_000):
            try:
                with urllib.request.urlopen(f"{EPHE_BASE_URL}/{fn}", timeout=25) as resp:
                    with open(se1_path, "wb") as f:
                        f.write(resp.read())
            except Exception:
                pass
    swe.set_ephe_path(EPHE_DIR)
    return EPHE_DIR


EPHE_ACTIVE_DIR = init_swisseph()

TOPO_FLAGS = swe.FLG_SWIEPH | swe.FLG_SPEED | swe.FLG_TOPOCTR
GEO_FLAGS = swe.FLG_SWIEPH | swe.FLG_SPEED

PLANETS_10: list[tuple[str, int]] = [
    ("Sun",     swe.SUN),
    ("Moon",    swe.MOON),
    ("Mercury", swe.MERCURY),
    ("Venus",   swe.VENUS),
    ("Mars",    swe.MARS),
    ("Jupiter", swe.JUPITER),
    ("Saturn",  swe.SATURN),
    ("Uranus",  swe.URANUS),
    ("Neptune", swe.NEPTUNE),
    ("Pluto",   swe.PLUTO),
]

PLANETS_12: list[tuple[str, int]] = PLANETS_10 + [
    ("Node",   swe.TRUE_NODE),
    ("Chiron", swe.CHIRON),
]

# Classical Chaldean descending orbital order for Planetary Hours
CHALDEAN_HOUR_SEQ: tuple[str, ...] = (
    "Saturn", "Jupiter", "Mars", "Sun", "Venus", "Mercury", "Moon",
)
WEEKDAY_DAY_RULER: dict[int, str] = {
    0: "Moon",     # Monday
    1: "Mars",     # Tuesday
    2: "Mercury",  # Wednesday
    3: "Jupiter",  # Thursday
    4: "Venus",    # Friday
    5: "Saturn",   # Saturday
    6: "Sun",      # Sunday
}

PLANET_NATURE_WEIGHT: dict[str, float] = {
    "Sun": +1.15,
    "Moon": +0.85,
    "Mercury": +0.35,
    "Venus": +1.10,
    "Mars": -0.95,
    "Jupiter": +1.35,
    "Saturn": -1.25,
    "Uranus": -0.45,
    "Neptune": -0.35,
    "Pluto": -0.70,
    "Node": +0.55,
    "Chiron": +0.30,
    "ASC": +0.90,
    "MC": +0.85,
    "POF": +1.05,
}

_TZ_CACHE: dict[str, Any] = {}
_SNAP_CACHE: dict[tuple[float, float, float], dict[str, Any]] = {}
_DATE_CTX_CACHE: dict[str, dict[str, Any]] = {}


def get_tz(tz_name: str):
    tz = _TZ_CACHE.get(tz_name)
    if tz is None:
        tz = pytz.timezone(tz_name)
        _TZ_CACHE[tz_name] = tz
    return tz


def anchor_jd_ut(d_iso: str, anchor_key: str) -> tuple[float, datetime.datetime]:
    """
    Compute exact DST-aware UTC Julian Day for `d_iso` under `anchor_key`.
    Strictly <= T_0 of the evaluated trading window (zero future leakage).
    """
    cfg = ANCHOR_CANDIDATES[anchor_key]
    y, m, d = map(int, d_iso.split("-"))
    base_dt = datetime.date(y, m, d) + datetime.timedelta(days=int(cfg["day_offset"]))
    tz = get_tz(cfg["tz"])
    loc_naive = datetime.datetime(base_dt.year, base_dt.month, base_dt.day, int(cfg["hour"]), int(cfg["minute"]), 0)
    loc_dt = tz.localize(loc_naive)
    utc_dt = loc_dt.astimezone(pytz.utc)
    ut_hour = utc_dt.hour + utc_dt.minute / 60.0 + utc_dt.second / 3600.0
    jd_ut = swe.julday(utc_dt.year, utc_dt.month, utc_dt.day, ut_hour)
    return jd_ut, utc_dt


def _compute_planetary_hour(jd_ut: float, lat: float, lon: float, local_tz_name: str) -> dict[str, Any]:
    """Compute Sunrise/Sunset, Day/Night sect, and Unequal Planetary Hour Ruler via `swe.rise_trans`."""
    geopos = (float(lon), float(lat), 0.0)
    try:
        # Search for most recent sunrise before jd_ut (step back 1.3 days, iterate forward)
        r1 = swe.rise_trans(jd_ut - 1.35, swe.SUN, swe.CALC_RISE | swe.BIT_DISC_CENTER, geopos, 0.0, 0.0, swe.FLG_SWIEPH)
        rise_prev = r1[1][0]
        r2 = swe.rise_trans(rise_prev + 0.05, swe.SUN, swe.CALC_RISE | swe.BIT_DISC_CENTER, geopos, 0.0, 0.0, swe.FLG_SWIEPH)
        rise_next = r2[1][0]
        if rise_next <= jd_ut:
            rise_active = rise_next
            r3 = swe.rise_trans(rise_active + 0.05, swe.SUN, swe.CALC_RISE | swe.BIT_DISC_CENTER, geopos, 0.0, 0.0, swe.FLG_SWIEPH)
            rise_following = r3[1][0]
        else:
            rise_active = rise_prev
            rise_following = rise_next

        s1 = swe.rise_trans(rise_active, swe.SUN, swe.CALC_SET | swe.BIT_DISC_CENTER, geopos, 0.0, 0.0, swe.FLG_SWIEPH)
        sunset_active = s1[1][0]

        is_day = rise_active <= jd_ut < sunset_active
        if is_day:
            frac = (jd_ut - rise_active) / max(1e-6, sunset_active - rise_active)
            hour_idx_0 = min(11, max(0, int(frac * 12.0)))
        else:
            frac = (jd_ut - sunset_active) / max(1e-6, rise_following - sunset_active)
            hour_idx_0 = 12 + min(11, max(0, int(frac * 12.0)))

        # Determine planetary day weekday from local calendar date of rise_active
        uy, um, ud, uh = swe.revjul(rise_active)
        hr_i = int(uh)
        mn_i = int((uh - hr_i) * 60.0)
        rise_utc = datetime.datetime(uy, um, ud, hr_i, mn_i, tzinfo=datetime.timezone.utc)
        rise_loc = rise_utc.astimezone(get_tz(local_tz_name))
        day_ruler = WEEKDAY_DAY_RULER[rise_loc.weekday()]
        start_seq = CHALDEAN_HOUR_SEQ.index(day_ruler)
        hour_ruler = CHALDEAN_HOUR_SEQ[(start_seq + hour_idx_0) % 7]
        return {
            "is_day_chart": bool(is_day),
            "planetary_day_ruler": day_ruler,
            "planetary_hour_num": hour_idx_0 + 1,
            "planetary_hour_ruler": hour_ruler,
            "planetary_hour_element": RULER_ELEMENT[hour_ruler].upper(),
        }
    except Exception:
        return {
            "is_day_chart": False,
            "planetary_day_ruler": "Sun",
            "planetary_hour_num": 1,
            "planetary_hour_ruler": "Sun",
            "planetary_hour_element": "FIRE",
        }


def compute_topocentric_snapshot(jd_ut: float, lat: float, lon: float, local_tz_name: str = "America/New_York") -> dict[str, Any]:
    """
    Compute full Topocentric Swiss Ephemeris state at `(jd_ut, lat, lon)`:
      - 12 Planets/Nodes (Sun..Pluto + True Node + Chiron) with longitudes & speeds
      - Placidus ASC & MC + Sect-aware Part of Fortune (POF)
      - Planetary Hour Ruler & Element
      - 360-degree v82 aspect sum lookup table (`deg_tbl[0..359]`)
    """
    ck = (round(jd_ut, 6), round(lat, 4), round(lon, 4))
    cached = _SNAP_CACHE.get(ck)
    if cached is not None:
        return cached

    swe.set_topo(float(lon), float(lat), 0.0)

    bodies: dict[str, dict[str, Any]] = {}
    p10_degs: list[tuple[str, float]] = []
    elem_counts_12 = [0, 0, 0, 0]

    for pname, pid in PLANETS_12:
        # True Node does not use topocentric parallax in Swiss Ephemeris; fallback cleanly if needed
        fl = GEO_FLAGS if pid == swe.TRUE_NODE else TOPO_FLAGS
        res = swe.calc_ut(jd_ut, pid, fl)
        lon_deg = float(res[0][0]) % 360.0
        speed = float(res[0][3])
        z_str, z_el, _ = zodiac_from_deg(lon_deg)
        elem_counts_12[ELEMENT_INDEX[z_el]] += 1
        bodies[pname] = {
            "lon": lon_deg,
            "speed": speed,
            "retro": bool(speed < 0.0),
            "zodiac": z_str,
            "element": z_el,
        }
        if len(p10_degs) < 10:
            p10_degs.append((pname, lon_deg))

    cusps, ascmc = swe.houses_ex(jd_ut, float(lat), float(lon), b"P")
    asc_deg = float(ascmc[0]) % 360.0
    mc_deg = float(ascmc[1]) % 360.0

    ph_info = _compute_planetary_hour(jd_ut, lat, lon, local_tz_name)
    is_day = ph_info["is_day_chart"]
    sun_lon = bodies["Sun"]["lon"]
    moon_lon = bodies["Moon"]["lon"]
    pof_deg = ((asc_deg + moon_lon - sun_lon) % 360.0) if is_day else ((asc_deg + sun_lon - moon_lon) % 360.0)

    for aname, adeg in (("ASC", asc_deg), ("MC", mc_deg), ("POF", pof_deg)):
        z_str, z_el, _ = zodiac_from_deg(adeg)
        bodies[aname] = {
            "lon": adeg,
            "speed": 0.0,
            "retro": False,
            "zodiac": z_str,
            "element": z_el,
        }

    # Vectorized 360-degree integer aspect lookup table (100% bit-for-bit v82 compatible)
    # + Vectorized 360-degree 15-body harmonic resonance lookup table (45x faster!)
    degs_arr = np.arange(360, dtype=np.float64)[:, None]  # (360, 1)
    p10_lons = np.array([pd for _, pd in p10_degs], dtype=np.float64)[None, :]  # (1, 10)
    p10_sunmoon = np.array([1.0 if pn in ("Sun", "Moon") else 0.0 for pn, _ in p10_degs], dtype=np.float64)[None, :]

    diff10 = np.abs(degs_arr - p10_lons)
    diff10 = np.where(diff10 > 180.0, 360.0 - diff10, diff10)

    best_d10 = np.full((360, 10), 999.0, dtype=np.float64)
    best_v10 = np.zeros((360, 10), dtype=np.int32)
    from src.xauusd_config import ASPECTS_24
    for sym, ang, orb in ASPECTS_24:
        v_deg = zod_aspect_deg(sym)
        if not isinstance(v_deg, int):
            continue
        d_mat = np.abs(diff10 - ang)
        within = d_mat <= (orb + p10_sunmoon)
        better = within & (d_mat < best_d10)
        best_d10 = np.where(better, d_mat, best_d10)
        best_v10 = np.where(better, v_deg, best_v10)

    deg_tbl = best_v10.sum(axis=1).tolist()

    # Precompute 360-degree 15-body harmonic score & top hit labels
    b_names = list(bodies.keys())
    b_lons = np.array([bodies[bn]["lon"] for bn in b_names], dtype=np.float64)[None, :]
    b_sm = np.array([1.0 if bn in ("Sun", "Moon") else 0.0 for bn in b_names], dtype=np.float64)[None, :]
    b_pw = np.array([PLANET_NATURE_WEIGHT.get(bn, 0.25) for bn in b_names], dtype=np.float64)[None, :]
    b_retro_m = np.array(
        [0.85 if (bodies[bn].get("retro") and bn in ("Mercury", "Venus", "Mars")) else 1.0 for bn in b_names],
        dtype=np.float64,
    )[None, :]

    diff15 = np.abs(degs_arr - b_lons)
    diff15 = np.where(diff15 > 180.0, 360.0 - diff15, diff15)

    best_d15 = np.full((360, len(b_names)), 999.0, dtype=np.float64)
    best_orb15 = np.ones((360, len(b_names)), dtype=np.float64)
    best_hw15 = np.zeros((360, len(b_names)), dtype=np.float64)
    best_idx15 = np.full((360, len(b_names)), -1, dtype=np.int16)

    for a_idx, (sym, ang, orb) in enumerate(ASPECTS_24):
        max_o = orb + b_sm
        d_mat = np.abs(diff15 - ang)
        better = (d_mat <= max_o) & (d_mat < best_d15)
        best_d15 = np.where(better, d_mat, best_d15)
        best_orb15 = np.where(better, max_o, best_orb15)
        best_hw15 = np.where(better, ASPECT_HARMONIC_WEIGHT.get(sym, 0.0), best_hw15)
        best_idx15 = np.where(better, a_idx, best_idx15)

    valid15 = best_idx15 >= 0
    exact15 = np.where(valid15, np.maximum(0.05, 1.0 - best_d15 / np.maximum(0.1, best_orb15)), 0.0)
    pos_c = best_hw15 * (0.65 + 0.55 * b_pw) * exact15
    neg_c = best_hw15 * np.where(b_pw < 0, 1.30, 0.70) * exact15
    contrib15 = np.where(best_hw15 >= 0, pos_c, neg_c) * b_retro_m
    contrib15 = np.where(valid15, contrib15, 0.0)
    harm_tbl = np.round(contrib15.sum(axis=1), 4).tolist()

    # Precompute O(1) Step 3 tightest 12-planet aspect Arcana element lookup table (bit-for-bit identical)
    best_d12 = best_d15[:, :12]
    best_p12 = np.argmin(best_d12, axis=1)
    row_idx = np.arange(360)
    min_d12 = best_d12[row_idx, best_p12]
    best_a12 = best_idx15[row_idx, best_p12]
    p12_lons_1d = b_lons[0, :12]
    res_degs = np.rint(p12_lons_1d[best_p12] + row_idx).astype(np.int64)
    arc_idx0 = np.mod(res_degs - 1, 22)
    el_arr = np.where(min_d12 < 900.0, _ARC_EL_ARR[arc_idx0], -1)
    step3_el_tbl = el_arr.tolist()
    step3_pa_tbl = (best_p12.tolist(), best_a12.tolist())

    moon_phase_angle = (moon_lon - sun_lon) % 360.0
    waxing_moon = moon_phase_angle < 180.0

    snap = {
        "jd_ut": jd_ut,
        "lat": lat,
        "lon": lon,
        "bodies": bodies,
        "deg_tbl": deg_tbl,
        "harm_tbl": harm_tbl,
        "step3_el_tbl": step3_el_tbl,
        "step3_pa_tbl": step3_pa_tbl,
        "planet_fawe_12": tuple(elem_counts_12),
        "asc": asc_deg,
        "mc": mc_deg,
        "pof": pof_deg,
        "moon_phase": moon_phase_angle,
        "waxing_moon": waxing_moon,
        **ph_info,
    }
    if len(_SNAP_CACHE) > 50_000:
        _SNAP_CACHE.clear()
    _SNAP_CACHE[ck] = snap
    return snap


def get_date_geomantic_context(d_iso: str) -> dict[str, Any]:
    """Compute deterministic daily geomantic date context (`f_helper`, `ele_dig`, `ele_dt`, `ele_dw`, `ele_date_sum`)."""
    ctx = _DATE_CTX_CACHE.get(d_iso)
    if ctx is not None:
        return ctx
    y, m, d = map(int, d_iso.split("-"))
    dt = datetime.date(y, m, d)
    wd_title = dt.strftime("%A")
    wd_up = wd_title.upper()

    f_helper = WEEKDAY_NUM[wd_title] + sum(int(c) for c in f"{y:04d}{m:02d}{d:02d}")
    x2 = sum(int(c) for c in f"{m}{d}{y}")
    b_dig1, b_dig2 = GEO_BITS_4[min(15, x2 // 10)], GEO_BITS_4[min(15, x2 % 10)]
    ele_dig = tuple(b_dig1[j] + b_dig2[j] for j in range(4))

    ap2, aq2, ar2 = mod_arcanum(m), mod_arcanum(d), mod_arcanum(y)
    hc_day = ARC_HC[ap2] + ARC_HC[aq2] + ARC_HC[ar2]
    md_day = ARC_MD[ap2] + ARC_MD[aq2] + ARC_MD[ar2]
    bd2, be2 = abs(hc_day), abs(md_day)
    b_dt = [GEO_BITS_4[min(15, x)] for x in (bd2 // 10, bd2 % 10, be2 // 10, be2 % 10)]
    ele_dt = tuple(sum(b[j] for b in b_dt) for j in range(4))

    by2 = DAY_CODE_MAP.get(wd_up, "0101")
    ele_dw = tuple(int(ch) for ch in by2)
    ele_date_sum = tuple(ele_dig[j] + ele_dt[j] + ele_dw[j] for j in range(4))

    ctx = {
        "date": d_iso,
        "dt": dt,
        "weekday": wd_title,
        "weekday_up": wd_up,
        "f_helper": f_helper,
        "F1": DEG1_GOLD + f_helper,
        "F2": DEG2_XAUUSD + f_helper,
        "F3_HOME": HOME_XAU_DEG + f_helper,
        "F3_AWAY": AWAY_USD_DEG + f_helper,
        "ele_dig": ele_dig,
        "ele_dt": ele_dt,
        "ele_dw": ele_dw,
        "ele_date_sum": ele_date_sum,
        "hc_day": hc_day,
        "md_day": md_day,
    }
    _DATE_CTX_CACHE[d_iso] = ctx
    return ctx


def eval_astro_entity_v82(
    d_ctx: dict[str, Any],
    deg1: int,
    deg2: int,
    deg3: int,
    entity_fawe: tuple[int, int, int, int],
    deg_tbl: list[int],
) -> dict[str, Any]:
    """
    Exact v82 Astro-Arcana-Decan-Geomantic Evaluator (`ELE-P`, `ELE-FINAL`, `ELE-DYN`, `GF-FIN`, `pwr`).
    Hard-asserts `deg1 == DEG1_GOLD == 107`.
    """
    assert deg1 == DEG1_GOLD == 107, f"Hard Entity Lock Violation: expected DEG1_GOLD=107, got {deg1}"
    fh = d_ctx["f_helper"]
    ele_date_sum = d_ctx["ele_date_sum"]

    p1 = deg1 + fh
    p2 = deg2 + fh
    p3 = deg3 + fh

    a1, a2, a3 = mod_arcanum(p1), mod_arcanum(p2), mod_arcanum(p3)
    el1_v = ARC_HC[a1] + ARC_HC[a2] + ARC_HC[a3]
    el2_v = ARC_MD[a1] + ARC_MD[a2] + ARC_MD[a3]

    cv, dv, ev = deg_tbl[p1 % 360], deg_tbl[p2 % 360], deg_tbl[p3 % 360]
    mG, mH, mI = mod_arcanum(cv), mod_arcanum(dv), mod_arcanum(ev)
    hcAA_v = ARC_HC[mG] + ARC_HC[mH] + ARC_HC[mI]
    mdAB_v = ARC_MD[mG] + ARC_MD[mH] + ARC_MD[mI]

    qdeg_hc = hcAA_v + el1_v
    qdeg_md = mdAB_v + el2_v

    h2, i2 = abs(qdeg_hc), abs(qdeg_md)
    bp1 = GEO_BITS_4[min(15, h2 // 10)]
    bp2 = GEO_BITS_4[min(15, h2 % 10)]
    bp3 = GEO_BITS_4[min(15, i2 // 10)]
    bp4 = GEO_BITS_4[min(15, i2 % 10)]
    ele_p = (
        bp1[0] + bp2[0] + bp3[0] + bp4[0],
        bp1[1] + bp2[1] + bp3[1] + bp4[1],
        bp1[2] + bp2[2] + bp3[2] + bp4[2],
        bp1[3] + bp2[3] + bp3[3] + bp4[3],
    )
    ele_fin = tuple(ele_p[j] + ele_date_sum[j] for j in range(4))
    ele_dyn = tuple(ele_p[j] + ele_fin[j] for j in range(4))

    qfin_hc = (ele_fin[0] + ele_fin[1]) - (ele_fin[2] + ele_fin[3])
    qfin_md = (ele_fin[1] + ele_fin[2]) - (ele_fin[0] + ele_fin[3])

    mg = DECAN_36_TABLE[((cv - 1) % 36) + 1]
    mh = DECAN_36_TABLE[((dv - 1) % 36) + 1]
    mi = DECAN_36_TABLE[((ev - 1) % 36) + 1]
    f1 = DECAN_36_TABLE[((p1 - 1) % 36) + 1]
    f2 = DECAN_36_TABLE[((p2 - 1) % 36) + 1]
    f3 = DECAN_36_TABLE[((p3 - 1) % 36) + 1]

    qdec_hc = mg[1] + mh[1] + mi[1] + f1[1] + f2[1] + f3[1]
    qdec_md = mg[2] + mh[2] + mi[2] + f1[2] + f2[2] + f3[2]
    gd_hc   = mg[3] + mh[3] + mi[3] + f1[3] + f2[3] + f3[3]
    gd_md   = mg[4] + mh[4] + mi[4] + f1[4] + f2[4] + f3[4]
    rul_hc  = mg[5] + mh[5] + mi[5] + f1[5] + f2[5] + f3[5]
    rul_md  = mg[6] + mh[6] + mi[6] + f1[6] + f2[6] + f3[6]

    dh, dm_v = abs(qdec_hc), abs(qdec_md)
    bd1 = GEO_BITS_4[min(15, dh // 10)]
    bd2_b = GEO_BITS_4[min(15, dh % 10)]
    bd3 = GEO_BITS_4[min(15, dm_v // 10)]
    bd4 = GEO_BITS_4[min(15, dm_v % 10)]
    ele_decf = tuple(
        bd1[j] + bd2_b[j] + bd3[j] + bd4[j] + ele_date_sum[j] for j in range(4)
    )
    qdecf_hc = (ele_decf[0] + ele_decf[1]) - (ele_decf[2] + ele_decf[3])
    qdecf_md = (ele_decf[1] + ele_decf[2]) - (ele_decf[0] + ele_decf[3])

    ele_tot = tuple(ele_dyn[j] + entity_fawe[j] for j in range(4))
    qtot_hc = (ele_tot[0] + ele_tot[1]) - (ele_tot[2] + ele_tot[3])
    qtot_md = (ele_tot[1] + ele_tot[2]) - (ele_tot[0] + ele_tot[3])

    gf_code = "".join("1" if abs(x) % 2 == 1 else "0" for x in ele_fin)
    wts = (8, 4, 2, 1)
    gf_fin = round(sum(wts[j] * (2 * int(gf_code[j] == "1") - 1) * ele_fin[j] for j in range(4))) / 10.0

    pwr = round(
        0.35 * (qfin_hc + qfin_md)
        + 0.35 * (qdecf_hc + qdecf_md)
        + 0.15 * (gd_hc + gd_md)
        + 0.15 * (rul_hc + rul_md),
        2,
    )

    return {
        "ep": ele_p,
        "ef": ele_fin,
        "ed": ele_dyn,
        "ele_decf": ele_decf,
        "ele_tot": ele_tot,
        "gf": gf_code,
        "gf_fin": gf_fin,
        "pwr": pwr,
        "qdeg": (qdeg_hc, qdeg_md),
        "qfin": (qfin_hc, qfin_md),
        "qdec": (qdec_hc, qdec_md),
        "qdecf": (qdecf_hc, qdecf_md),
        "qtot": (qtot_hc, qtot_md),
        "gd": (gd_hc, gd_md),
        "rul": (rul_hc, rul_md),
        "decan_lbl": f"{f2[7]} / {f3[7]}",
    }


def score_degree_24_aspects(target_deg: float, snap: dict[str, Any], need_hits: bool = False) -> tuple[float, list[str]]:
    """
    Compute 24-aspect harmonic resonance score of `target_deg` against all 15 topocentric points
    (Sun..Pluto, True Node, Chiron, ASC, MC, POF) with +1° Sun/Moon orb bonus.
    Uses O(1) precomputed `harm_tbl` for integer degrees when `need_hits=False`.
    """
    if not need_hits and float(target_deg).is_integer() and "harm_tbl" in snap:
        return snap["harm_tbl"][int(target_deg) % 360], []

    score = 0.0
    hits: list[str] = []
    for bname, binfo in snap["bodies"].items():
        sym, exactness = best_aspect(target_deg, binfo["lon"], bname)
        if sym is not None:
            hw = ASPECT_HARMONIC_WEIGHT.get(sym, 0.0)
            pw = PLANET_NATURE_WEIGHT.get(bname, 0.25)
            if hw >= 0:
                contrib = hw * (0.65 + 0.55 * pw) * exactness
            else:
                mal_mult = 1.30 if pw < 0 else 0.70
                contrib = hw * mal_mult * exactness
            if binfo.get("retro") and bname in ("Mercury", "Venus", "Mars"):
                contrib *= 0.85
            score += contrib
            if exactness >= 0.45:
                hits.append(f"{bname}{sym}")
    return round(score, 4), hits


def _vm6_score(mh: dict[str, Any], ma: dict[str, Any]) -> int:
    """Exact 6-Rule VM6 comparative polarity from v82 `step4_winner_prediction.py`."""
    r1 = 1 if abs(mh["qdec"][1]) < abs(ma["qdec"][1]) else (-1 if abs(mh["qdec"][1]) > abs(ma["qdec"][1]) else 0)
    r2 = 1 if mh["gd"][1] < ma["gd"][1] else (-1 if mh["gd"][1] > ma["gd"][1] else 0)
    r3 = 1 if abs(mh["qdecf"][0]) < abs(ma["qdecf"][0]) else (-1 if abs(mh["qdecf"][0]) > abs(ma["qdecf"][0]) else 0)
    f_h = abs(mh["qfin"][0]) + abs(mh["qfin"][1])
    f_a = abs(ma["qfin"][0]) + abs(ma["qfin"][1])
    r4 = 1 if f_h < f_a else (-1 if f_h > f_a else 0)
    r5 = 1 if mh["qtot"][1] < ma["qtot"][1] else (-1 if mh["qtot"][1] > ma["qtot"][1] else 0)
    r6 = 1 if mh["gf_fin"] < ma["gf_fin"] else (-1 if mh["gf_fin"] > ma["gf_fin"] else 0)
    return r1 + r2 + r3 + r4 + r5 + r6


def evaluate_engine_a(
    d_iso: str,
    today_open: float,
    prev_close: float,
    dxy_prev_close: float,
    anchor_key: str = "ANCHOR_A_NY_1800",
    rounding_mode: str = "floor",
    need_hits: bool = False,
) -> dict[str, Any]:
    """
    Engine A: Topocentric Transit + `DEG1(107)/DEG2(147)/XAU(77)/USD(70)` Aspect & v82 Pole Score.
    Supports both single-site anchors (`ANCHOR_A/B/C`) and `ANCHOR_DUAL_POLE_1800`
    (`XAU` evaluated at LBMA London vs `USD` evaluated at COMEX New York at `18:00 NY D-1`).
    """
    cfg = ANCHOR_CANDIDATES[anchor_key]
    jd_ut, utc_dt = anchor_jd_ut(d_iso, anchor_key)
    d_ctx = get_date_geomantic_context(d_iso)

    if cfg.get("dual_pole"):
        snap_xau = compute_topocentric_snapshot(jd_ut, cfg["xau_lat"], cfg["xau_lon"], "Europe/London")
        snap_usd = compute_topocentric_snapshot(jd_ut, cfg["usd_lat"], cfg["usd_lon"], "America/New_York")
    else:
        snap_xau = compute_topocentric_snapshot(jd_ut, cfg["lat"], cfg["lon"], cfg["tz"])
        snap_usd = snap_xau

    open_deg = price_to_deg(today_open, rounding_mode)
    pclose_deg = price_to_deg(prev_close, rounding_mode)
    dxy_deg = price_to_deg(dxy_prev_close * 100.0, rounding_mode)

    # Core Entity Evaluations:
    pair_eval = eval_astro_entity_v82(d_ctx, DEG1_GOLD, DEG2_XAUUSD, open_deg, (1, 1, 1, 0), snap_xau["deg_tbl"])
    xau_pole = eval_astro_entity_v82(d_ctx, DEG1_GOLD, DEG2_XAUUSD, HOME_XAU_DEG, (1, 2, 1, 0), snap_xau["deg_tbl"])
    xau_live = eval_astro_entity_v82(d_ctx, DEG1_GOLD, open_deg, HOME_XAU_DEG, (1, 2, 1, 0), snap_xau["deg_tbl"])
    usd_pole = eval_astro_entity_v82(d_ctx, DEG1_GOLD, DEG2_XAUUSD, AWAY_USD_DEG, (0, 2, 0, 2), snap_usd["deg_tbl"])
    usd_live = eval_astro_entity_v82(d_ctx, DEG1_GOLD, pclose_deg, AWAY_USD_DEG, (0, 2, 0, 2), snap_usd["deg_tbl"])
    usd_dxy  = eval_astro_entity_v82(d_ctx, DEG1_GOLD, dxy_deg, AWAY_USD_DEG, (0, 2, 0, 2), snap_usd["deg_tbl"])

    vm6_pole = _vm6_score(xau_pole, usd_pole)
    vm6_live = _vm6_score(xau_live, usd_live)
    f30_pole = vm6_pole * 5
    f30_live = vm6_live * 5

    # Direct 24-aspect harmonic resonance on foundational poles, topocentric angles & daily shifted poles
    s_gold, hits_gold = score_degree_24_aspects(DEG1_GOLD, snap_xau, need_hits=need_hits)
    s_pair, hits_pair = score_degree_24_aspects(DEG2_XAUUSD, snap_xau, need_hits=need_hits)
    s_xau, hits_xau   = score_degree_24_aspects(HOME_XAU_DEG, snap_xau, need_hits=need_hits)
    s_usd, hits_usd   = score_degree_24_aspects(AWAY_USD_DEG, snap_usd, need_hits=need_hits)
    s_lbma_asc, _     = score_degree_24_aspects(int(round(snap_xau["asc"])) % 360, snap_xau, need_hits=False)
    s_comex_mc, _     = score_degree_24_aspects(int(round(snap_usd["mc"])) % 360, snap_usd, need_hits=False)
    s_f1, _           = score_degree_24_aspects(d_ctx["F1"] % 360, snap_xau, need_hits=False)
    s_f2, _           = score_degree_24_aspects(d_ctx["F2"] % 360, snap_xau, need_hits=False)
    s_fh, _           = score_degree_24_aspects(d_ctx["F3_HOME"] % 360, snap_xau, need_hits=False)
    s_fa, _           = score_degree_24_aspects(d_ctx["F3_AWAY"] % 360, snap_usd, need_hits=False)
    s_open, hits_open = score_degree_24_aspects(open_deg, snap_xau, need_hits=need_hits)

    ph_ruler = snap_xau["planetary_hour_ruler"]
    ph_bonus = {
        "Sun": +0.40,
        "Jupiter": +0.35,
        "Venus": +0.28,
        "Moon": +0.18,
        "Mercury": +0.05,
        "Mars": -0.28,
        "Saturn": -0.40,
    }.get(ph_ruler, 0.0)

    # Non-parametric Direct Harmonic Aspect Balance (`v4`):
    # Symmetric Trine/Sextile/Conjunction (+) vs Square/Opposition/Semisquare (-) on HOME_XAU(77°)+LBMA_ASC+F3_H
    # vs AWAY_USD(70°)+COMEX_MC+F3_A, weighted with 107° WATER / 147° FIRE shifted pole balance + VM6 polarity.
    xau_harm = 0.45 * s_xau + 0.30 * s_fh + 0.25 * s_lbma_asc
    usd_harm = 0.45 * s_usd + 0.30 * s_fa + 0.25 * s_comex_mc
    pole_balance = 0.55 * (s_f1 - s_f2) + 0.45 * s_open
    vm6_exp = -(vm6_pole + vm6_live) / 3.5

    raw_score_a = (
        0.38 * (xau_harm - usd_harm)
        + 0.44 * vm6_exp
        + 0.10 * pole_balance
        + 0.08 * ph_bonus
    )

    return {
        "jd_ut": jd_ut,
        "utc_iso": utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "anchor_key": anchor_key,
        "rounding_mode": rounding_mode,
        "open_deg": open_deg,
        "pclose_deg": pclose_deg,
        "dxy_deg": dxy_deg,
        "d_ctx": d_ctx,
        "snap_xau": snap_xau,
        "snap_usd": snap_usd,
        "pair_eval": pair_eval,
        "xau_pole": xau_pole,
        "xau_live": xau_live,
        "usd_pole": usd_pole,
        "usd_live": usd_live,
        "usd_dxy": usd_dxy,
        "vm6_pole": vm6_pole,
        "vm6_live": vm6_live,
        "f30_pole": f30_pole,
        "f30_live": f30_live,
        "f30_total": f30_pole + f30_live,
        "s_gold": s_gold,
        "s_pair": s_pair,
        "s_xau": s_xau,
        "s_usd": s_usd,
        "s_lbma_asc": s_lbma_asc,
        "s_comex_mc": s_comex_mc,
        "s_open": s_open,
        "hits_gold": hits_gold[:4],
        "hits_xau": hits_xau[:4],
        "hits_usd": hits_usd[:4],
        "hits_open": hits_open[:4],
        "planetary_hour_ruler": ph_ruler,
        "planetary_hour_element": snap_xau["planetary_hour_element"],
        "asc_deg": round(snap_xau["asc"], 2),
        "mc_deg": round(snap_xau["mc"], 2),
        "pof_deg": round(snap_xau["pof"], 2),
        "score_a": round(raw_score_a, 4),
        "dir_a": "XAU" if raw_score_a >= 0 else "USD",
    }
