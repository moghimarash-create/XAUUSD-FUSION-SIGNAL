#!/usr/bin/env python3
"""
Hard Astrological, Elemental & Geomantic Entity Locks for Spot Gold (`XAUUSD-FUSION-SIGNAL`).

STRICT ARCHITECTURAL INVARIANTS:
- `DEG1_GOLD = 107`   -> `17° CANCER` (`WATER` — Cold & Moist — Intrinsic Gold Pole)
- `DEG2_XAUUSD = 147` -> `27° LEO`    (`FIRE`  — Hot & Dry   — Pair Volatility / Regulus Pole)
- `HOME_XAU_DEG = 77` -> `17° GEMINI` (`AIR`   — Hot & Moist — HOME Team = Bulls / UP Candle: Close > Open)
- `AWAY_USD_DEG = 70` -> `10° GEMINI` (`AIR`   — Hot & Moist — AWAY Team = Bears / DOWN Candle: Close < Open)
"""
from __future__ import annotations

import datetime
import math
import re
from typing import Any, Final

# ---------------------------------------------------------------------------
# 1. HARD ASTROLOGICAL & ENTITY LOCKS
# ---------------------------------------------------------------------------
DEG1_GOLD: Final[int] = 107     # 17° CANCER (WATER — Cold & Moist — Intrinsic Gold Pole)
DEG2_XAUUSD: Final[int] = 147   # 27° LEO    (FIRE  — Hot & Dry   — Pair Volatility / Regulus Pole)
HOME_XAU_DEG: Final[int] = 77   # 17° GEMINI (AIR   — Hot & Moist — HOME Team = Bulls / UP: Close > Open)
AWAY_USD_DEG: Final[int] = 70   # 10° GEMINI (AIR   — Hot & Moist — AWAY Team = Bears / DOWN: Close < Open)

HOME_TEAM: Final[str] = "XAU"
AWAY_TEAM: Final[str] = "USD"

ZODIAC_SIGNS_12: Final[tuple[str, ...]] = (
    "ARIES", "TAURUS", "GEMINI", "CANCER", "LEO", "VIRGO",
    "LIBRA", "SCORPIO", "SAGITTARIUS", "CAPRICORN", "AQUARIUS", "PISCES",
)

SIGN_ELEMENTS: Final[dict[str, str]] = {
    "ARIES": "FIRE", "LEO": "FIRE", "SAGITTARIUS": "FIRE",
    "TAURUS": "EARTH", "VIRGO": "EARTH", "CAPRICORN": "EARTH",
    "GEMINI": "AIR", "LIBRA": "AIR", "AQUARIUS": "AIR",
    "CANCER": "WATER", "SCORPIO": "WATER", "PISCES": "WATER",
}

ELEMENT_ORDER: Final[tuple[str, str, str, str]] = ("FIRE", "AIR", "WATER", "EARTH")
ELEMENT_INDEX: Final[dict[str, int]] = {"FIRE": 0, "AIR": 1, "WATER": 2, "EARTH": 3}

ELEMENT_QUALITIES: Final[dict[str, tuple[str, str]]] = {
    "Fire": ("HOT", "DRY"),
    "Earth": ("COLD", "DRY"),
    "Air": ("HOT", "MOIST"),
    "Water": ("COLD", "MOIST"),
    "FIRE": ("HOT", "DRY"),
    "EARTH": ("COLD", "DRY"),
    "AIR": ("HOT", "MOIST"),
    "WATER": ("COLD", "MOIST"),
}

ELEMENT_VECTORS: Final[dict[str, tuple[int, int]]] = {
    "Fire": (1, -1), "FIRE": (1, -1),
    "Earth": (-1, -1), "EARTH": (-1, -1),
    "Air": (1, 1), "AIR": (1, 1),
    "Water": (-1, 1), "WATER": (-1, 1),
}


def zodiac_from_deg(deg: int | float) -> tuple[str, str, int]:
    """Convert ecliptic longitude [0..360) to ('17° CANCER', 'WATER', sign_index_0)."""
    d = int(math.floor(float(deg))) % 360
    house_idx = d // 30
    within_deg = d % 30
    sign = ZODIAC_SIGNS_12[house_idx]
    return f"{within_deg}° {sign}", SIGN_ELEMENTS[sign], house_idx


def price_to_deg(price: float, mode: str = "floor") -> int:
    """Convert XAUUSD dollar price to [0..359] degree (`floor` = int(math.floor(P)) % 360, `round` = round(P) % 360)."""
    if mode == "round":
        return int(round(float(price))) % 360
    return int(math.floor(float(price))) % 360


# Compile-time assertions on foundational entity locks
assert zodiac_from_deg(DEG1_GOLD) == ("17° CANCER", "WATER", 3)
assert zodiac_from_deg(DEG2_XAUUSD) == ("27° LEO", "FIRE", 4)
assert zodiac_from_deg(HOME_XAU_DEG) == ("17° GEMINI", "AIR", 2)
assert zodiac_from_deg(AWAY_USD_DEG) == ("10° GEMINI", "AIR", 2)
assert price_to_deg(3583.45, "floor") == 343 and zodiac_from_deg(343) == ("13° PISCES", "WATER", 11)

# ---------------------------------------------------------------------------
# 2. GEOMANTIC & ELEMENTAL HELPER CONSTANTS (PORTED VERBATIM)
# ---------------------------------------------------------------------------
WEEKDAY_NUM: Final[dict[str, int]] = {
    "Monday": 9,
    "Tuesday": 5,
    "Wednesday": 8,
    "Thursday": 4,
    "Friday": 7,
    "Saturday": 6,
    "Sunday": 3,
}

DAY_CODE_MAP: Final[dict[str, str]] = {
    "SUNDAY": "0011",
    "MONDAY": "1111",
    "TUESDAY": "1101",
    "WEDNESDAY": "0110",
    "THURSDAY": "0101",
    "FRIDAY": "1011",
    "SATURDAY": "0001",
}


def _to_date(dt: datetime.date | datetime.datetime | str) -> datetime.date:
    if isinstance(dt, datetime.datetime):
        return dt.date()
    if isinstance(dt, datetime.date):
        return dt
    y, m, d = map(int, str(dt)[:10].split("-"))
    return datetime.date(y, m, d)


def digital_sum(dt: datetime.date | datetime.datetime | str) -> int:
    """Sum of digits of YYYYMMDD."""
    d = _to_date(dt)
    return sum(int(ch) for ch in f"{d.year:04d}{d.month:02d}{d.day:02d}")


def compute_f_helper(dt: datetime.date | datetime.datetime | str) -> int:
    """f_helper = WEEKDAY_NUM[weekday] + digital_sum(YYYYMMDD)."""
    d = _to_date(dt)
    wd = d.strftime("%A")
    return WEEKDAY_NUM[wd] + digital_sum(d)


def daily_shifted_degrees(dt: datetime.date | datetime.datetime | str) -> dict[str, int]:
    """Daily shifted label degrees F1, F2, F3_HOME, F3_AWAY."""
    fh = compute_f_helper(dt)
    return {
        "f_helper": fh,
        "F1": DEG1_GOLD + fh,
        "F2": DEG2_XAUUSD + fh,
        "F3_HOME": HOME_XAU_DEG + fh,
        "F3_AWAY": AWAY_USD_DEG + fh,
    }


def mod_arcanum(n: int | float) -> int:
    """22-Arcana cyclic reduction: ((int(n) - 1) % 22) + 1."""
    return ((int(n) - 1) % 22) + 1


ARCANUM_22_MAP: Final[dict[int, str]] = {
    1: "Scorpio-Water",
    2: "Leo-Fire",
    3: "Gemini-Air",
    4: "Aries-Fire",
    5: "Gemini-Air",
    6: "Gemini-Air",
    7: "Capricorn-Earth",
    8: "Libra-Air",
    9: "Capricorn-Earth",
    10: "Gemini-Air",
    11: "Taurus-Earth",
    12: "Cancer-Water",
    13: "Gemini-Air",
    14: "Leo-Fire",
    15: "Aries-Fire",
    16: "Gemini-Air",
    17: "Taurus-Earth",
    18: "Scorpio-Water",
    19: "Capricorn-Earth",
    20: "Leo-Fire",
    21: "Aquarius-Air",
    22: "Virgo-Earth",
}

ZOD_MAP: Final[dict[int, tuple[str, str, str]]] = {}
ARC_HC: Final[list[int]] = [0] * 23
ARC_MD: Final[list[int]] = [0] * 23
ARC_ELEMENT_IDX: Final[dict[int, int]] = {}

for _arc_idx, _arc_str in ARCANUM_22_MAP.items():
    _sign_part, _elem_part = _arc_str.split("-")
    _hc_q, _md_q = ELEMENT_QUALITIES[_elem_part]
    ZOD_MAP[_arc_idx] = (f"{_sign_part} - [{_elem_part}]", _hc_q, _md_q)
    ARC_HC[_arc_idx] = _arc_idx if _hc_q == "HOT" else -_arc_idx
    ARC_MD[_arc_idx] = _arc_idx if _md_q == "MOIST" else -_arc_idx
    ARC_ELEMENT_IDX[_arc_idx] = ELEMENT_INDEX[_elem_part.upper()]

# ---------------------------------------------------------------------------
# 3. 36 CHALDEAN DECANS (1..36) & RULERS
# ---------------------------------------------------------------------------
CHALDEAN_RULERS: Final[list[str]] = [
    "Mars", "Sun", "Venus", "Mercury", "Moon", "Saturn", "Jupiter",
]

RULER_ELEMENT: Final[dict[str, str]] = {
    "Mars": "Fire",
    "Sun": "Fire",
    "Venus": "Water",
    "Mercury": "Earth",
    "Moon": "Water",
    "Saturn": "Earth",
    "Jupiter": "Air",
}

_SIGNS_36_META: Final[list[tuple[str, str, str, str]]] = [
    ("Aries", "Fire", "Cardinal", "Wands"),
    ("Taurus", "Earth", "Fixed", "Pentacles"),
    ("Gemini", "Air", "Mutable", "Swords"),
    ("Cancer", "Water", "Cardinal", "Cups"),
    ("Leo", "Fire", "Fixed", "Wands"),
    ("Virgo", "Earth", "Mutable", "Pentacles"),
    ("Libra", "Air", "Cardinal", "Swords"),
    ("Scorpio", "Water", "Fixed", "Cups"),
    ("Sagittarius", "Fire", "Mutable", "Wands"),
    ("Capricorn", "Earth", "Cardinal", "Pentacles"),
    ("Aquarius", "Air", "Fixed", "Swords"),
    ("Pisces", "Water", "Mutable", "Cups"),
]


def deg_to_decan36(deg: int | float) -> int:
    """Map ecliptic degree [0..360) to Chaldean Decan 1..36."""
    return int((float(deg) % 360.0) // 10.0) + 1


def decan36_info(d: int) -> dict[str, Any]:
    """Return full Chaldean Decan metadata for d in 1..36."""
    d_norm = ((int(d) - 1) % 36) + 1
    si, face = (d_norm - 1) // 3, ((d_norm - 1) % 3) + 1
    sign, sign_el, modality, suit = _SIGNS_36_META[si]
    card_num = (1 if modality == "Cardinal" else (4 if modality == "Fixed" else 7)) + face
    ruler = CHALDEAN_RULERS[(d_norm - 1) % 7]
    ruler_el = RULER_ELEMENT[ruler]
    s_hc, s_md = ELEMENT_VECTORS[sign_el]
    r_hc, r_md = ELEMENT_VECTORS[ruler_el]
    return {
        "decan": d_norm,
        "sign": sign,
        "face": face,
        "label": f"{sign} D{face}",
        "sign_element": sign_el.upper(),
        "modality": modality,
        "suit": suit,
        "card_num": card_num,
        "ruler": ruler,
        "ruler_element": ruler_el.upper(),
        "qdec_hc": d_norm * s_hc,
        "qdec_md": d_norm * s_md,
        "gd_hc": card_num * s_hc,
        "gd_md": card_num * s_md,
        "rul_hc": card_num * r_hc,
        "rul_md": card_num * r_md,
    }


DECAN_36_TABLE: Final[dict[int, tuple]] = {}
for _d in range(1, 37):
    _info = decan36_info(_d)
    DECAN_36_TABLE[_d] = (
        _d,
        _info["qdec_hc"], _info["qdec_md"],
        _info["gd_hc"], _info["gd_md"],
        _info["rul_hc"], _info["rul_md"],
        _info["label"],
        _info["sign_element"],
        _info["ruler"],
        _info["ruler_element"],
    )

# ---------------------------------------------------------------------------
# 4. 24-ASPECT SWISS EPHEMERIS HARMONIC TABLE
# ---------------------------------------------------------------------------
ASPECTS_24: Final[list[tuple[str, float, float]]] = [
    ("\u2609", 0.0,        9.0),   # Conjunction
    ("\u260D", 180.0,      9.0),   # Opposition
    ("\u25B3", 120.0,      9.0),   # Trine
    ("\u25A1", 90.0,       9.0),   # Square
    ("\u2217", 60.0,       6.0),   # Sextile
    ("\u26BB", 150.0,      3.0),   # Quincunx
    ("\u26BA", 30.0,       2.0),   # Semisextile
    ("\u2220", 45.0,       2.5),   # Semisquare
    ("\u26BC", 135.0,      2.5),   # Sesquisquare
    ("Q",      72.0,       2.5),   # Quintile
    ("bQ",     144.0,      2.5),   # Biquintile
    ("D",      36.0,       2.0),   # Decile
    ("D3",     108.0,      2.0),   # Tredecile
    ("S",      51.428571,  2.0),   # Septile
    ("S2",     102.857143, 2.0),   # Biseptile
    ("S3",     154.285714, 2.0),   # Triseptile
    ("N",      40.0,       2.0),   # Novile
    ("N2",     80.0,       2.0),   # Binovile
    ("N4",     160.0,      2.0),   # Quadnovile
    ("U",      32.727273,  2.0),   # Undecile
    ("U2",     65.454545,  2.0),   # Biundecile
    ("U3",     98.181818,  2.0),   # Triundecile
    ("U4",     130.909091, 2.0),   # Quadundecile
    ("U5",     163.636363, 2.0),   # Quinundecile
]

SUN_MOON_BODIES: Final[set[str]] = {"Sun", "Moon"}

ASPECT_DEG_MAP: Final[dict[str, int]] = {
    "\u2609": 0,
    "\u260D": 180,
    "\u25B3": 120,
    "\u25A1": 90,
    "\u2217": 60,
    "\u26BB": 150,
    "\u26BA": 30,
    "\u2220": 45,
    "\u26BC": 135,
    "Q": 72,
    "bQ": 144,
    "BQ": 144,
    "D": 36,
    "D3": 108,
    "S": 51,
    "S2": 103,
    "S3": 154,
    "N": 40,
    "N2": 80,
    "N4": 160,
    "U": 33,
    "U2": 65,
    "U3": 98,
    "U4": 131,
    "U5": 164,
}

# Harmonic polarity weights of the 24 aspects (+ = benefic/constructive, - = hard/tensional)
ASPECT_HARMONIC_WEIGHT: Final[dict[str, float]] = {
    "\u2609": +1.25,  # 0° Conjunction
    "\u25B3": +1.35,  # 120° Trine
    "\u2217": +1.00,  # 60° Sextile
    "Q":      +0.75,  # 72° Quintile
    "bQ":     +0.75,  # 144° Biquintile
    "D":      +0.45,  # 36° Decile
    "D3":     +0.45,  # 108° Tredecile
    "N":      +0.40,  # 40° Novile
    "N2":     +0.35,  # 80° Binovile
    "N4":     +0.30,  # 160° Quadnovile
    "\u26BA": +0.25,  # 30° Semisextile
    "\u260D": -1.30,  # 180° Opposition
    "\u25A1": -1.25,  # 90° Square
    "\u2220": -0.80,  # 45° Semisquare
    "\u26BC": -0.80,  # 135° Sesquisquare
    "\u26BB": -0.65,  # 150° Quincunx
    "S":      -0.25,  # 51.43° Septile
    "S2":     -0.25,
    "S3":     -0.25,
    "U":      +0.20,
    "U2":     +0.20,
    "U3":     +0.20,
    "U4":     +0.20,
    "U5":     +0.20,
}


def best_aspect(label_deg: float, planet_deg: float, planet_name: str = "") -> tuple[str | None, float]:
    """
    Find tightest matching aspect among the 24 aspects (+1.0° orb bonus for Sun/Moon).
    Returns (symbol_or_None, exactness_0_to_1).
    """
    orb_bonus = 1.0 if planet_name in SUN_MOON_BODIES else 0.0
    diff = abs((float(label_deg) % 360.0) - (float(planet_deg) % 360.0))
    if diff > 180.0:
        diff = 360.0 - diff
    best_sym: str | None = None
    best_d: float | None = None
    best_max_orb: float = 1.0
    for sym, ang, orb in ASPECTS_24:
        max_orb = orb + orb_bonus
        d = abs(diff - ang)
        if d <= max_orb:
            if best_d is None or d < best_d:
                best_d = d
                best_sym = sym
                best_max_orb = max_orb
    if best_sym is None or best_d is None:
        return None, 0.0
    exactness = max(0.05, 1.0 - (best_d / max(0.1, best_max_orb)))
    return best_sym, exactness


def zod_aspect_deg(aspect: Any) -> int | str:
    """Convert aspect symbol to canonical integer degree value (v82 exact mapping)."""
    if aspect is None or aspect == "":
        return ""
    raw = str(aspect).strip()
    if raw in ASPECT_DEG_MAP:
        return ASPECT_DEG_MAP[raw]
    code = re.sub(r"\s", "", raw.upper())
    if code in ASPECT_DEG_MAP:
        return ASPECT_DEG_MAP[code]
    return ""


# ---------------------------------------------------------------------------
# 5. 16 CLASSICAL GEOMANTIC FIGURES & 4-BIT PARITY MAPS
# ---------------------------------------------------------------------------
GEO_BITS_4: Final[dict[int, tuple[int, int, int, int]]] = {
    i: ((i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1) for i in range(16)
}

GEOMANCY_FIGURES: Final[dict[str, str]] = {
    "0000": "Populus",
    "0001": "Tristitia",
    "0010": "Rubeus",
    "0011": "Fortuna_Major",
    "0100": "Albus",
    "0101": "Acquisitio",
    "0110": "Conjunctio",
    "0111": "Caput_Draconis",
    "1000": "Laetitia",
    "1001": "Carcer",
    "1010": "Amissio",
    "1011": "Puer",
    "1100": "Fortuna_Minor",
    "1101": "Puella",
    "1110": "Cauda_Draconis",
    "1111": "Via",
}

BULLISH_FIGURES: Final[set[str]] = {"Albus", "Acquisitio", "Laetitia", "Fortuna_Major"}
BEARISH_FIGURES: Final[set[str]] = {"Conjunctio", "Amissio", "Tristitia", "Cauda_Draconis", "Carcer"}

# ---------------------------------------------------------------------------
# 6. SPATIO-TEMPORAL ANCHOR DEFINITIONS & ZERO-LEAKAGE SPLITS
# ---------------------------------------------------------------------------
COMEX_NY: Final[dict[str, Any]] = {
    "name": "COMEX New York",
    "lat": 40.7143,
    "lon": -74.0139,
    "alt": 0.0,
    "tz": "America/New_York",
}

LBMA_LONDON: Final[dict[str, Any]] = {
    "name": "LBMA London",
    "lat": 51.5142,
    "lon": -0.0885,
    "alt": 0.0,
    "tz": "Europe/London",
}

ANCHOR_CANDIDATES: Final[dict[str, dict[str, Any]]] = {
    "ANCHOR_A_NY_1800": {
        "label": "Anchor A (18:00 NY D-1 Globex/Spot Candle Birth @ COMEX NY)",
        "day_offset": -1,
        "hour": 18,
        "minute": 0,
        "tz": "America/New_York",
        "lat": COMEX_NY["lat"],
        "lon": COMEX_NY["lon"],
        "dual_pole": False,
    },
    "ANCHOR_B_LONDON_0800": {
        "label": "Anchor B (08:00 London D Bullion Open @ LBMA London)",
        "day_offset": 0,
        "hour": 8,
        "minute": 0,
        "tz": "Europe/London",
        "lat": LBMA_LONDON["lat"],
        "lon": LBMA_LONDON["lon"],
        "dual_pole": False,
    },
    "ANCHOR_C_NY_0820": {
        "label": "Anchor C (08:20 NY D COMEX Pit Open @ COMEX NY)",
        "day_offset": 0,
        "hour": 8,
        "minute": 20,
        "tz": "America/New_York",
        "lat": COMEX_NY["lat"],
        "lon": COMEX_NY["lon"],
        "dual_pole": False,
    },
    "ANCHOR_DUAL_POLE_1800": {
        "label": "Dual-Pole Fusion (18:00 NY D-1 Candle Birth: XAU @ LBMA London vs USD @ COMEX NY)",
        "day_offset": -1,
        "hour": 18,
        "minute": 0,
        "tz": "America/New_York",
        "lat": COMEX_NY["lat"],
        "lon": COMEX_NY["lon"],
        "xau_lat": LBMA_LONDON["lat"],
        "xau_lon": LBMA_LONDON["lon"],
        "usd_lat": COMEX_NY["lat"],
        "usd_lon": COMEX_NY["lon"],
        "dual_pole": True,
    },
}

TRAIN_START: Final[str] = "2012-01-01"
TRAIN_END: Final[str] = "2024-12-31"
VAL_START: Final[str] = "2025-01-01"
VAL_END: Final[str] = "2025-12-31"
OOS_START: Final[str] = "2026-01-01"
OOS_END: Final[str] = "2026-12-31"


def split_label(d_iso: str) -> str:
    if d_iso < TRAIN_START:
        return "WARMUP"
    if d_iso <= TRAIN_END:
        return "TRAIN"
    if d_iso <= VAL_END:
        return "VAL"
    return "OOS"
