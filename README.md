# `XAUUSD-FUSION-SIGNAL` (`v5.1.0`) — Sheet-Backed 4-Method Smart Twin Selector, Polarity Inversion & `8v8` Spot Gold Radar

Standalone `NBA-FUSION-SIGNAL` / `NBA-PLAYER-SIGNAL` isomorphic daily prediction, `Q25/Q50/Q75` target-level & Kalshi bracket engine for **Spot Gold (`XAUUSD`)**, wrapped in a low-memory (`~58–75 MiB RSS`) FastAPI microservice on **Port `8080`** with a full **Persian RTL Trading Radar, 3-Method Comparison Bar (`اول GEO` | `گام‌به‌گام` | `🔒 قفل سه‌لایه`), Smart Polarity Inversion (`🔄 چرخش قطبیت هوشمند`), Dual Win-Rate Ledger & Interactive Past/Present/Future Date Time-Machine**.

## Hard Entity & Astrological Locks (`src/xauusd_config.py`)
- **`DEG1_GOLD = 107`** → `17° CANCER` (`WATER` — Cold & Moist — Intrinsic Gold Pole)
- **`DEG2_XAUUSD = 147`** → `27° LEO` (`FIRE` — Hot & Dry — Pair Volatility / Regulus Pole)
- **`HOME_XAU_DEG = 77`** → `17° GEMINI` (`AIR` — Hot & Moist — HOME Team = Bulls / `UP`: `Close > Open`)
- **`AWAY_USD_DEG = 70`** → `10° GEMINI` (`AIR` — Hot & Moist — AWAY Team = Bears / `DOWN`: `Close < Open`)
- **Locked Anchor:** `ANCHOR_DUAL_POLE_1800` (`18:00:00 America/New_York` on `D-1`, LBMA London `51.5142, -0.0885` for `HOME XAU` vs COMEX New York `40.7143, -74.0139` for `AWAY USD`, topocentric `pyswisseph`, `round(P) % 360`).

## Production Architecture (`v5.1.0`)
- `src/sheet_archive_loader.py` — Bundles the authentic 4,705-row Gold elemental archive (`2012–2027`, `PRIM ~32, FINAL ~24, ELE-P ~10`) in `data/gold_sheet_history.json` & `data/gold_sheet_signals.json` for zero runtime latency, with live re-sync in `POST /api/refresh` and multi-planet (`12 planets × 24 aspects` + `36-Decan` + `v82` `GEO_BITS_4`) fallback beyond `2027-02-26`.
- `src/twin_lock_engine.py` — Strict Zero Look-Ahead (`hist_date < target_date`) 4-Method Smart Twin Selector (`LOCK3`, `GEO`, `STEP`, `LOCK2` + `pickBestSmartNarrowing` & `all3Agree`) + `KXGOLDPERP` `Q25 / Q50 / Q75` winning twin target quantiles.
- `src/price_roster_engine.py` — Strictly Distinct (`100%` unique rounded levels) `8v8` Support/Resistance Roster + Macro Trend Launchpad (`EMA20`, `EMA50`).
- `src/xauusd_gold_cba_engine.py` — Train-Frozen (`2012–2024`) 3-Tier `Gold CBA` (`👑 Tier C 1.20x ⊂ ⭐ Tier B 1.00x ⊂ 🟢 Tier A 0.80x`, `subsetViolations == 0`) + 6-Layer `Veto Shield` (no blind `ZeroELEP` veto of trend days) + Kalshi `KXGOLD` bracket metadata.
- `src/trade_path_simulator.py` — Reports side-by-side **Daily Target Win% (`KXGOLDPERP` without tight stop)** and **Stepped-SL Execution Win% (`SL = 0.85 ATR → Breakeven Runner`)**.
- `server.py` & `static/` — FastAPI + Uvicorn microservice on `0.0.0.0:${PORT:-8080}` (`GET /api/health`, `GET /api/signals/today`, `GET /api/signals/by-date?date=YYYY-MM-DD`, `GET /api/signals/history`, `GET /api/stats/summary`, `POST /api/refresh`) and Persian RTL Dashboard (`رادار هوشمند طلا`).

## Run Pipeline, Tests & Microservice
```bash
# 1. Run walk-forward backtest & export production JSON shards
python3 main.py

# 2. Run automated zero-leakage & subset invariant tests
pytest tests/test_invariants.py

# 3. Launch Port 8080 microservice
uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
```
