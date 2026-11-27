# `XAUUSD-FUSION-SIGNAL` (`v4.1.0`) — Non-Parametric Twin-First Spot Gold Engine & Persian RTL Radar

Standalone `NBA-FUSION-SIGNAL` / `NBA-PLAYER-SIGNAL` isomorphic daily prediction, target-level & Kalshi bracket engine for **Spot Gold (`XAUUSD`)**, wrapped in a low-memory (`~55–68 MiB RSS`) FastAPI microservice on **Port `8080`** with a full **Persian RTL Trading Radar, Interactive Past/Present/Future Date Time-Machine & 12-Col Twin Explorer**.

## Hard Entity & Astrological Locks (`src/xauusd_config.py`)
- **`DEG1_GOLD = 107`** → `17° CANCER` (`WATER` — Cold & Moist — Intrinsic Gold Pole)
- **`DEG2_XAUUSD = 147`** → `27° LEO` (`FIRE` — Hot & Dry — Pair Volatility / Regulus Pole)
- **`HOME_XAU_DEG = 77`** → `17° GEMINI` (`AIR` — Hot & Moist — HOME Team = Bulls / `UP`: `Close > Open`)
- **`AWAY_USD_DEG = 70`** → `10° GEMINI` (`AIR` — Hot & Moist — AWAY Team = Bears / `DOWN`: `Close < Open`)
- **Locked Anchor:** `ANCHOR_DUAL_POLE_1800` (`18:00:00 America/New_York` on `D-1`, LBMA London `51.5142, -0.0885` for `HOME XAU` vs COMEX New York `40.7143, -74.0139` for `AWAY USD`, topocentric `pyswisseph`, `round(P) % 360`).

## Production Architecture (`v4.1.0`)
- `src/twin_lock_engine.py` — 100% Price-Free Pure Astro-Arcana-Decan `12-Col Signature` (`sum <= 4, 8, 12`) + Recency-ATR Weighted (`exp(-years_ago / 6.0) * atr_norm_move * decan_sim`) Non-Parametric Twin Consensus (`4..10` twins `< target_date`).
- `src/price_roster_engine.py` — Strictly Distinct (`100%` unique rounded levels) `8v8` Support/Resistance Roster + Macro Trend Launchpad (`EMA20`, `EMA50`, `5D`).
- `src/xauusd_gold_cba_engine.py` — 3-Tier `Gold CBA` (`👑 Tier C 1.20x ⊂ ⭐ Tier B 1.00x ⊂ 🟢 Tier A 0.80x`, `subsetViolations == 0`) + 6-Layer `Veto Shield` (`V1..V6`, including `V6_COUNTER_MACRO_TREND`) + Kalshi `KXGOLD` bracket metadata.
- `server.py` & `static/` — FastAPI + Uvicorn microservice on `0.0.0.0:${PORT:-8080}` (`GET /api/health`, `GET /api/signals/today`, `GET /api/signals/by-date?date=YYYY-MM-DD`, `GET /api/signals/history`, `GET /api/stats/summary`, `POST /api/refresh`) and Persian RTL Dashboard (`رادار هوشمند طلا`).

## Run Pipeline, Tests & Microservice
```bash
# 1. Run walk-forward backtest & export production JSON shards
python3 main.py

# 2. Run automated zero-leakage & subset invariant tests
python3 -m unittest discover -s tests -v

# 3. Launch Port 8080 microservice
uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
```
