(() => {
  "use strict";

  const ELEM_META = {
    FIRE: { fa: "آتش", icon: "🔥", code: "FIRE" },
    AIR: { fa: "هوا", icon: "💨", code: "AIR" },
    WATER: { fa: "💧 آب".replace("💧 ", ""), icon: "💧", code: "WATER" },
    EARTH: { fa: "خاک", icon: "🌍", code: "EARTH" },
  };

  const WEEKDAY_FA = {
    Monday: "دوشنبه",
    Tuesday: "سه‌شنبه",
    Wednesday: "چهارشنبه",
    Thursday: "پنج‌شنبه",
    Friday: "جمعه",
    Saturday: "شنبه",
    Sunday: "یک‌شنبه",
  };

  const REGIME_FA = {
    STRONG_BULL: "روند صعودی قوی (STRONG_BULL)",
    STRONG_BEAR: "روند نزولی قوی (STRONG_BEAR)",
    NEUTRAL: "تعادل / رنج (NEUTRAL)",
  };

  const state = {
    todayPayload: null,
    summaryPayload: null,
    activeCard: null,
    liveDate: "",
    nextPreviewDate: "",
    prevNavDate: "",
    nextNavDate: "",
    twinsExpanded: true,
    ledgerTier: "ALL",
    ledgerSplit: "VAL_OOS",
    ledgerDay: "ALL",
    ledgerQuery: "",
    searchDebounce: null,
  };

  const $ = (id) => document.getElementById(id);

  function fmtUsd(val) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) return "$--";
    return "$" + Number(val).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function fmtSignedUsd(val) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) return "$0.00";
    const n = Number(val);
    const sign = n > 0 ? "+" : n < 0 ? "-" : "";
    return `${sign}$${Math.abs(n).toFixed(2)}`;
  }

  function showToast(msg) {
    const el = $("toast-banner");
    if (!el) return;
    el.textContent = msg;
    el.classList.remove("hidden");
    window.clearTimeout(el._timer);
    el._timer = window.setTimeout(() => el.classList.add("hidden"), 4800);
  }

  function syncHeaderSelectionUI(card) {
    if (!card) return;
    const curDate = card.sessionDate || "";
    const picker = $("session-date-picker");
    if (picker && curDate) {
      picker.value = curDate;
    }

    state.prevNavDate = card.prevSessionDate || "";
    state.nextNavDate = card.nextSessionDate || "";

    const tabLive = $("tab-live-session");
    const tabNext = $("tab-next-session");
    const isLive = curDate === state.liveDate && !card.isFuture;
    const isNextPreview = curDate === state.nextPreviewDate && Boolean(card.isFuture);

    tabLive.classList.toggle("active", isLive);
    tabLive.setAttribute("aria-selected", String(isLive));
    tabNext.classList.toggle("active", isNextPreview);
    tabNext.setAttribute("aria-selected", String(isNextPreview));

    // Highlight matching row in History Ledger if visible
    document.querySelectorAll("#tbody-history-ledger tr.clickable-ledger-row").forEach((tr) => {
      tr.classList.toggle("selected-ledger-row", tr.dataset.date === curDate);
    });
  }

  function renderBenchmarkStrip(summary) {
    if (!summary || !summary.splits) return;

    const fillCard = (containerId, splitObj) => {
      const container = $(containerId);
      if (!container || !splitObj) return;
      const tiers = [
        ["TIER_C", "👑 Tier C"],
        ["TIER_B", "⭐ Tier B"],
        ["TIER_A", "🟢 Tier A"],
      ];
      container.innerHTML = tiers
        .map(([tk, label]) => {
          const m = splitObj[tk] || {};
          const wr = Number(m.exec_win_rate ?? m.win_rate ?? 0).toFixed(1);
          const pf = Number(m.exec_pf ?? m.pf ?? 0).toFixed(2);
          const n = m.n ?? 0;
          const exp = fmtSignedUsd(m.exec_exp_usd ?? m.expectancy_usd ?? 0);
          return `
            <div class="bench-tier-cell">
              <span class="bt-lbl ltr-num">${label}</span>
              <strong class="bt-wr ltr-num">${wr}%</strong>
              <span class="bt-meta ltr-num">n=${n} • PF ${pf} • ${exp}</span>
            </div>
          `;
        })
        .join("");
    };

    fillCard("bench-oos-grid", summary.splits.OOS);
    fillCard("bench-val-grid", summary.splits.VAL);
    fillCard("bench-train-grid", summary.splits.TRAIN);

    $("inv-subset").textContent = `✓ سلسله‌مراتب C ⊂ B ⊂ A (خطا: ${summary.subsetViolations ?? 0})`;
    $("inv-leakage").textContent = `✓ نشت داده آینده: ${summary.lookAheadLeakage ?? 0} (Zero-Leakage)`;
    $("inv-distinct").textContent = `✓ تمایز راستر ۸×۸: ${summary.distinct8v8RosterPct ?? 100}٪`;
    $("inv-pricefree").textContent = summary.priceFreeTwinSignature
      ? "✓ امضای ۱۲ ستونی ۱۰۰٪ مستقل از قیمت"
      : "⚠ بررسی امضای ۱۲ ستونی";
  }

  function renderOutcomeOrFutureBanner(card) {
    const banner = $("hero-outcome-banner");
    if (!banner || !card) return;

    banner.className = "hero-outcome-banner";
    if (card.isFuture) {
      banner.classList.add("future-banner");
      const note =
        card.futureProjectionNoteFa ||
        `پیشبینی تاریخ آتی (${card.sessionDate}) — نقشه نجومی دوقطبی ۱۸:۰۰ نیویورک و همزادهای ۱۲ ستونی دقیق برای این تاریخ محاسبه شده‌اند.`;
      banner.innerHTML = `
        <div class="outcome-banner-main">
          <strong>🔮 پیشبینی تاریخ آتی</strong>
          <span>${note}</span>
        </div>
      `;
      return;
    }

    if (card.isHistorical && card.hitOutcome) {
      const isActionable = Boolean(card.isA);
      const isWin = Boolean(card.execWin);
      banner.classList.add(!isActionable ? "hold-banner" : isWin ? "win-banner" : "loss-banner");
      const chipCls = !isActionable ? "hold" : isWin ? "win" : "loss";
      const execPnlStr = isActionable && card.execPnl !== null && card.execPnl !== undefined
        ? fmtSignedUsd(card.execPnl)
        : "بدون ورود (0.00$)";
      const moveStr = card.actualMove !== null && card.actualMove !== undefined
        ? `${fmtSignedUsd(card.actualMove)} (${card.actualDir || "FLAT"})`
        : "—";
      const rangeStr =
        card.actualHigh !== undefined && card.actualLow !== undefined
          ? `High ${fmtUsd(card.actualHigh)} / Low ${fmtUsd(card.actualLow)} / Close ${fmtUsd(card.actualClose)}`
          : "";

      banner.innerHTML = `
        <div class="outcome-banner-main">
          <strong>نتیجه واقعی کندل (${card.sessionDate}):</strong>
          <span class="outcome-chip ${chipCls}">${card.hitOutcome}</span>
          <span>سود-زیان اجرایی: <strong class="ltr-num ${!isActionable ? "" : isWin ? "pos-res" : "neg-res"}">${execPnlStr}</strong></span>
        </div>
        <div class="outcome-banner-details">
          <span>حرکت واقعی بازار: <strong class="ltr-num">${moveStr}</strong></span>
          ${rangeStr ? `<span class="ltr-num">${rangeStr}</span>` : ""}
        </div>
      `;
      return;
    }

    banner.classList.add("hidden");
  }

  function renderHeroAndCard(card) {
    if (!card) return;

    const modeBadge = $("hero-session-mode-badge");
    modeBadge.className = "meta-badge";
    if (card.isFuture) {
      modeBadge.classList.add("future-mode");
      modeBadge.textContent = card.modeBadgeFa || "پیشبینی تاریخ آتی";
    } else if (card.isLiveSession) {
      modeBadge.textContent = "سشن زنده فعلی (Live Session)";
    } else {
      modeBadge.classList.add("hist-mode");
      modeBadge.textContent = card.modeBadgeFa || "بازبینی سشن تاریخی";
    }

    const wdFa = WEEKDAY_FA[card.weekday] || card.weekday;
    $("hero-session-date").textContent = `${card.sessionDate} • ${wdFa} (${card.weekday})`;
    $("hero-candle-utc").textContent = `T₀ UTC: ${card.candleOpenUtc}`;

    renderOutcomeOrFutureBanner(card);

    const dir = card.direction || "HOLD / VETO";
    const bias = card.biasDirection || "XAU";
    const vbox = $("hero-verdict-box");
    const vdir = $("hero-direction-fa");
    vbox.classList.remove("bull-glow", "bear-glow", "hold-glow");
    vdir.classList.remove("bull", "bear", "hold");

    if (dir === "LONG XAU") {
      vbox.classList.add("bull-glow");
      vdir.classList.add("bull");
      vdir.textContent = "خرید طلا — LONG XAU 🟢";
    } else if (dir === "SHORT XAU") {
      vbox.classList.add("bear-glow");
      vdir.classList.add("bear");
      vdir.textContent = "فروش طلا — SHORT XAU 🔴";
    } else {
      vbox.classList.add("hold-glow");
      vdir.classList.add("hold");
      const biasFa = bias === "XAU" ? "سوگیری خام صعودی" : "سوگیری خام نزولی";
      vdir.textContent = `عدم ورود (سپر محافظ فعال • ${biasFa})`;
    }

    $("hero-tier-badge").textContent = card.tierBadge || "🛡️ HOLD/VETO";
    $("hero-kelly-pill").textContent = `Quarter-Kelly: ${Number(card.kellyMultiplier || 0).toFixed(2)}x`;
    $("hero-direction-en").textContent = `${dir} • Bias: ${bias === "XAU" ? "LONG XAU (77°)" : "SHORT XAU / USD (70°)"}`;

    const flipEl = $("hero-polarity-flip");
    if (flipEl) {
      if (card.polarityFlipped) {
        flipEl.classList.remove("hidden");
        flipEl.textContent =
          card.polarityFlipReasonFa || "🔄 قطبیت معکوس اصلاح شد — چرخش قطبیت (Polarity Flip)";
      } else {
        flipEl.classList.add("hidden");
      }
    }

    const wp = Number(card.winProbPct || 50);
    $("hero-win-prob").textContent = `${wp.toFixed(1)}%`;
    $("hero-prob-fill").style.width = `${Math.max(15, Math.min(100, wp))}%`;

    $("hero-spot-price").textContent = fmtUsd(card.spotPrice);
    $("hero-atr14").textContent = fmtUsd(card.atr14);
    $("hero-rr").textContent = `${Number(card.rrRatio || 0).toFixed(2)} R`;
    $("hero-macro-regime").textContent = REGIME_FA[card.macroRegime] || card.macroRegime;

    // Causal Price Ladder
    const entry = Number(card.entryPrice || 0);
    const atr = Math.max(1, Number(card.atr14 || 1));
    const fmtDelta = (target) => {
      const diff = Number(target) - entry;
      const atrMult = Math.abs(diff) / atr;
      return `${fmtSignedUsd(diff)} (${atrMult.toFixed(2)} ATR)`;
    };

    $("lvl-entry").textContent = fmtUsd(entry);
    $("lvl-sl").textContent = fmtUsd(card.sl);
    $("lvl-sl-delta").textContent = fmtDelta(card.sl);
    $("lvl-tp1").textContent = fmtUsd(card.tp1);
    $("lvl-tp1-delta").textContent = fmtDelta(card.tp1);
    $("lvl-tp2").textContent = fmtUsd(card.tp2);
    $("lvl-tp2-delta").textContent = fmtDelta(card.tp2);
    $("lvl-tp3").textContent = fmtUsd(card.tp3);
    $("lvl-tp3-delta").textContent = fmtDelta(card.tp3);

    // Kalshi Bracket
    const kb = card.kalshiBracket || {};
    $("kalshi-contract").textContent = kb.recommendedContract || "--";
    $("kalshi-band").textContent = kb.coreRangeBand || "--";
    $("kalshi-prob").textContent = `${Number(kb.impliedAboveOpenProbPct ?? 50).toFixed(1)}%`;

    // Sky Ephemeris Strip
    const ep = card.ephemeris || {};
    $("sky-asc-mc").textContent = `ASC ${ep.asc || "--"} | MC ${ep.mc || "--"}`;
    $("sky-pof-ph").textContent = `POF ${ep.pof || "--"} • ${ep.planetary_hour_ruler || "--"} (${ep.planetary_hour_element || "--"})`;

    // 6-Layer Veto Shield
    const shieldList = card.vetoShield6 || [];
    const activeCount = (card.vetoReasons || []).length;
    const sumBadge = $("veto-summary-badge");
    sumBadge.className = "veto-summary-badge " + (activeCount === 0 ? "clean" : "blocked");
    sumBadge.textContent =
      activeCount === 0
        ? "✓ تمام ۶ لایه ایمن (۰ وتو فعال)"
        : `⛔ ${activeCount} لایه محافظ فعال شده است`;

    $("hero-veto-grid").innerHTML = shieldList
      .map((v) => {
        const cls = v.triggered ? "triggered" : "safe";
        return `
          <div class="veto-card ${cls}">
            <div class="veto-card-top">
              <span class="veto-title">${v.titleFa}</span>
              <span class="veto-pill ${cls}">${v.statusFa}</span>
            </div>
            <span class="veto-desc">${v.descFa}</span>
          </div>
        `;
      })
      .join("");
  }

  function renderRosters(card) {
    if (!card) return;
    const homePlayers = card.homeXauRoster || [];
    const awayPlayers = card.awayUsdRoster || [];
    const hm = card.homeTeamMeta || {};
    const am = card.awayTeamMeta || {};

    const hmDom = ELEM_META[hm.domElement] || ELEM_META.FIRE;
    const amDom = ELEM_META[am.domElement] || ELEM_META.EARTH;

    $("home-dom-summary").textContent = `عنصر غالب: ${hmDom.icon} ${hmDom.fa} • برتری ستونی: ${hm.elemLeads ?? 0}/4`;
    $("away-dom-summary").textContent = `عنصر غالب: ${amDom.icon} ${amDom.fa} • برتری ستونی: ${am.elemLeads ?? 0}/4`;
    $("home-res-pill").textContent = `Resonance: ${Number(hm.resonanceNorm ?? 0) >= 0 ? "+" : ""}${Number(hm.resonanceNorm ?? 0).toFixed(3)}`;
    $("away-res-pill").textContent = `Resonance: ${Number(am.resonanceNorm ?? 0) >= 0 ? "+" : ""}${Number(am.resonanceNorm ?? 0).toFixed(3)}`;

    const elemKeys = ["FIRE", "AIR", "WATER", "EARTH"];
    const hw = hm.wFawe || [1, 1, 1, 1];
    const aw = am.wFawe || [1, 1, 1, 1];
    $("elem-dominance-bars").innerHTML = elemKeys
      .map((ek, idx) => {
        const meta = ELEM_META[ek];
        const hv = Number(hw[idx] || 0);
        const av = Number(aw[idx] || 0);
        const tot = Math.max(0.01, hv + av);
        const hPct = ((hv / tot) * 100).toFixed(1);
        const aPct = (100 - Number(hPct)).toFixed(1);
        return `
          <div class="elem-comp-cell">
            <div class="elem-comp-top">
              <span>${meta.icon} ${meta.fa} (<span class="ltr-num">${ek}</span>)</span>
              <span class="ltr-num">${hv.toFixed(2)} vs ${av.toFixed(2)}</span>
            </div>
            <div class="dual-bar">
              <div class="home-seg" style="width:${hPct}%" title="HOME XAU: ${hv.toFixed(2)}"></div>
              <div class="away-seg" style="width:${aPct}%" title="AWAY USD: ${av.toFixed(2)}"></div>
            </div>
          </div>
        `;
      })
      .join("");

    const buildRows = (players) =>
      players
        .map((p) => {
          const el = ELEM_META[p.element] || ELEM_META.FIRE;
          const effPct = Math.min(100, Math.round((Number(p.eff_w || 0) / 1.3) * 100));
          const uRes = Number(p.unit_resonance || 0);
          const resCls = uRes >= 0 ? "pos-res" : "neg-res";
          const resTxt = `${uRes >= 0 ? "+" : ""}${uRes.toFixed(3)}`;
          const aspects = p.aspect_hits && p.aspect_hits.length > 0 ? p.aspect_hits.join(" • ") : "—";
          return `
            <tr class="${p.starter ? "starter-row" : "bench-row"}">
              <td><span class="pos-badge ${p.starter ? "starter" : ""}">${p.pos}</span></td>
              <td><span class="ltr-num">${p.role}</span></td>
              <td><strong class="ltr-num">${fmtUsd(p.level)}</strong></td>
              <td class="effw-cell">
                <span class="ltr-num">${Number(p.eff_w).toFixed(3)}</span>
                <div class="effw-bar"><div class="effw-fill" style="width:${effPct}%"></div></div>
              </td>
              <td><span class="ltr-num">${p.deg}° (${p.zodiac})</span></td>
              <td><span class="elem-badge ${el.code}">${el.icon} ${el.fa}</span></td>
              <td><span class="ltr-num">D${String(p.decan).padStart(2, "0")} ${p.decan_label} (${p.decan_ruler})</span></td>
              <td><span class="ltr-num ${resCls}">${resTxt}</span></td>
              <td><span class="ltr-num">${aspects}</span></td>
            </tr>
          `;
        })
        .join("");

    $("tbody-home-roster").innerHTML = buildRows(homePlayers);
    $("tbody-away-roster").innerHTML = buildRows(awayPlayers);
  }

  function renderTwinsExplorer(card) {
    if (!card) return;
    const sig = card.signature12Col || {};
    const tc = card.twinConsensus || {};
    const twins = card.matchedTwins || [];

    const lockCols = new Set(tc.lock3Cols || []);
    const elemOrder = ["FIRE", "AIR", "WATER", "EARTH"];

    const badges = [];
    if (tc.lock3Peak) {
      badges.push(`<span class="tw-badge gold">🔒 LOCK3_PEAK فعال (${(tc.lock3Cols || []).join(", ")})</span>`);
    } else if (tc.hasLock3) {
      badges.push(`<span class="tw-badge gold">🔒 قفل ستونی LOCK3 (${(tc.lock3Cols || []).join(", ")})</span>`);
    }
    if (tc.inGoldenZone) {
      badges.push(`<span class="tw-badge emerald">✨ ناحیه طلایی (Golden Zone)</span>`);
    }
    if (tc.decan11) {
      badges.push(`<span class="tw-badge emerald">✓ هم‌فازی دکان و حاکم (1/1)</span>`);
    }
    if (tc.decan00) {
      badges.push(`<span class="tw-badge">⚠ تضاد دکان (0/0)</span>`);
    }
    $("twin-header-badges").innerHTML = badges.join("") || `<span class="tw-badge">پالایش پیش‌رونده ۱۲ ستونی</span>`;

    $("sig-zero-elep").textContent = `zeroElepCnt: ${sig.zeroElepCnt ?? 0}`;

    const prim = sig.PRIM_FAWE || [0, 0, 0, 0];
    const fin = sig.FINAL_FAWE || [0, 0, 0, 0];
    const elep = sig.ELEP_FAWE || [0, 0, 0, 0];

    let matHtml = `<div class="m12-head">لایه / عنصر</div>`;
    elemOrder.forEach((ek) => {
      const em = ELEM_META[ek];
      const locked = lockCols.has(ek) ? "🔒 " : "";
      matHtml += `<div class="m12-head">${locked}${em.icon} ${em.fa}</div>`;
    });

    const addMatrixRow = (rowTitle, arr) => {
      matHtml += `<div class="m12-row-lbl">${rowTitle}</div>`;
      arr.forEach((val, idx) => {
        const isLocked = lockCols.has(elemOrder[idx]);
        matHtml += `<div class="m12-cell ${isLocked ? "locked-col" : ""}">${val}</div>`;
      });
    };
    addMatrixRow("PRIM (1..4)", prim);
    addMatrixRow("FINAL (1..4)", fin);
    addMatrixRow("ELE-P (1..4)", elep);
    $("matrix-12col").innerHTML = matHtml;

    const geoCode = String(sig.geoCode || "0000");
    $("geo-fig-dots").innerHTML = geoCode
      .split("")
      .map((bit) => `<span>${bit === "1" ? "●" : "●   ●"}</span>`)
      .join("");
    $("sig-geo-fig").textContent = sig.GEO_FIG || "--";
    $("sig-dec-geo").textContent = sig.DEC_GEO || "--";
    $("sig-ruler-geo").textContent = sig.RULER_GEO || "--";

    $("sig-celestial-tags").innerHTML = (sig.celestialDecanTags || [])
      .map((t) => `<span class="mini-chip">${t}</span>`)
      .join("");
    $("sig-step3-hits").innerHTML = (sig.step3Hits || [])
      .map((t) => `<span class="mini-chip">${t}</span>`)
      .join("");

    const upN = Number(tc.upTwins || 0);
    const dnN = Number(tc.downTwins || 0);
    const totN = Math.max(1, upN + dnN);
    const upBarPct = ((upN / totN) * 100).toFixed(1);
    const dnBarPct = (100 - Number(upBarPct)).toFixed(1);

    $("tw-up-count").textContent = String(upN);
    $("tw-dn-count").textContent = String(dnN);
    $("tw-raw-pct").textContent = `${Number(tc.dispPct || 50).toFixed(1)}%`;
    $("tw-bar-up").style.width = `${upBarPct}%`;
    $("tw-bar-dn").style.width = `${dnBarPct}%`;

    $("tw-weighted-pct").textContent = `${Number(tc.weightedDispPct || 50).toFixed(1)}%`;
    $("tw-final-prob").textContent = `${Number(tc.twinWinProbPct || 50).toFixed(1)}%`;
    $("tw-maj-atr").textContent = `${Number(tc.majAvgAtrMove || 0).toFixed(2)}x ATR`;
    $("tw-min-atr").textContent = `${Number(tc.minAvgAtrMove || 0).toFixed(2)}x ATR`;
    $("tw-dom-flag").textContent = tc.strongAvgDominance ? "✓ بله (قوی)" : "خیر (معمولی)";
    $("tw-gauss-score").textContent = `${Number(tc.gaussScore || 0) >= 0 ? "+" : ""}${Number(tc.gaussScore || 0).toFixed(4)}`;
    $("tw-count-inline").textContent = String(twins.length);

    $("tbody-matched-twins").innerHTML = twins
      .map((tw, i) => {
        const isUp = tw.direction === "XAU";
        const dirBadge = isUp
          ? `<span class="outcome-chip win">▲ UP (XAU)</span>`
          : `<span class="outcome-chip loss">▼ DOWN (USD)</span>`;
        const wdFa = WEEKDAY_FA[tw.weekday] || tw.weekday;
        return `
          <tr>
            <td class="ltr-num">${i + 1}</td>
            <td><strong class="ltr-num">${tw.date}</strong></td>
            <td>${wdFa} (<span class="ltr-num">${tw.weekday}</span>)</td>
            <td>${dirBadge}</td>
            <td><span class="ltr-num ${isUp ? "pos-res" : "neg-res"}">${fmtSignedUsd(tw.move_usd)}</span></td>
            <td class="ltr-num">${Number(tw.move_atr).toFixed(2)}x ATR</td>
            <td class="ltr-num">${tw.col12_matches} / 12</td>
            <td class="ltr-num">${Number(tw.decan_sim).toFixed(2)}</td>
            <td><strong class="ltr-num">${Number(tw.weight).toFixed(3)}</strong></td>
          </tr>
        `;
      })
      .join("");
  }

  function applyFullSessionCard(card) {
    if (!card) return;
    state.activeCard = card;
    if (card.rssMb) {
      $("hdr-rss").textContent = `RSS: ${Number(card.rssMb).toFixed(1)} MiB`;
    }
    renderHeroAndCard(card);
    renderRosters(card);
    renderTwinsExplorer(card);
    syncHeaderSelectionUI(card);
  }

  async function loadSessionByDate(targetDate, opts = {}) {
    if (!targetDate) return;
    try {
      const res = await fetch(`/api/signals/by-date?date=${encodeURIComponent(targetDate)}`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        showToast(`⚠ خطا در بارگذاری تاریخ: ${err.detail || res.status}`);
        return;
      }
      const data = await res.json();
      applyFullSessionCard(data);

      if (data.snapped && data.snappedNoteFa) {
        showToast(`ℹ ${data.snappedNoteFa}`);
      }
      if (opts.scrollToHero) {
        const heroEl = $("section-hero-signal");
        if (heroEl) {
          heroEl.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }
    } catch (err) {
      console.error("loadSessionByDate error:", err);
      showToast("⚠ خطا در ارتباط با سرور هنگام دریافت اطلاعات تاریخ.");
    }
  }

  async function loadHistoryLedger() {
    const params = new URLSearchParams({
      limit: "150",
      tier: state.ledgerTier,
      split: state.ledgerSplit,
      day: state.ledgerDay,
      q: state.ledgerQuery,
    });
    const res = await fetch(`/api/signals/history?${params.toString()}`);
    if (!res.ok) return;
    const data = await res.json();

    $("ledger-count-badge").textContent = `نمایش ${data.items.length} از ${data.totalMatched} سشن`;

    const kpi = data.kpi || {};
    $("lkpi-trades").textContent = `${kpi.trades ?? 0} (${kpi.execWins ?? 0}W-${kpi.execLosses ?? 0}L)`;
    $("lkpi-exec-wr").textContent = `${Number(kpi.execWinRatePct ?? 0).toFixed(2)}%`;
    $("lkpi-raw-wr").textContent = `${Number(kpi.rawWinRatePct ?? 0).toFixed(2)}%`;
    $("lkpi-pf").textContent = Number(kpi.profitFactor ?? 0).toFixed(2);
    $("lkpi-exp").textContent = fmtSignedUsd(kpi.expectancyUsd ?? 0);
    $("lkpi-tps").textContent = `${Number(kpi.tp1HitRatePct ?? 0).toFixed(0)}% / ${Number(kpi.tp2HitRatePct ?? 0).toFixed(0)}% / ${Number(kpi.tp3HitRatePct ?? 0).toFixed(0)}%`;

    const curActiveDate = state.activeCard ? state.activeCard.sessionDate : "";

    $("tbody-history-ledger").innerHTML = (data.items || [])
      .map((r) => {
        const wdFa = WEEKDAY_FA[r.weekday] || r.weekday;
        const badgeCls = !r.isA ? "hold" : r.execWin ? "win" : "loss";
        const pnlCls = !r.isA ? "" : Number(r.execPnl) >= 0 ? "pos-res" : "neg-res";
        const pnlTxt = !r.isA ? "—" : fmtSignedUsd(r.execPnl);
        const flipMini = r.polarityFlipped ? `<span class="mini-flip-tag" title="چرخش قطبیت فعال">🔄 قطبیت</span>` : "";
        const sigTxt =
          r.direction === "LONG XAU"
            ? `<span class="pos-res ltr-num">▲ LONG XAU</span> ${flipMini}`
            : r.direction === "SHORT XAU"
            ? `<span class="neg-res ltr-num">▼ SHORT XAU</span> ${flipMini}`
            : `<span class="ltr-num">🛡️ HOLD (${r.biasDirection})</span> ${flipMini}`;
        const selCls = r.date === curActiveDate ? "selected-ledger-row" : "";

        return `
          <tr class="clickable-ledger-row ${selCls}" data-date="${r.date}" title="کلیک برای بارگذاری کامل رادار تاریخ ${r.date}">
            <td><strong class="ltr-num">🔍 ${r.date}</strong></td>
            <td>${wdFa}</td>
            <td><span class="ltr-num">${r.split}</span></td>
            <td><span class="ltr-num">${r.macroRegime}</span></td>
            <td><span class="ltr-num">${r.tierBadge} (${Number(r.kellyMultiplier).toFixed(2)}x)</span></td>
            <td>${sigTxt}</td>
            <td class="ltr-num">${Number(r.winProbPct).toFixed(1)}%</td>
            <td class="ltr-num">${fmtUsd(r.entryPrice)} / ${fmtUsd(r.sl)}</td>
            <td class="ltr-num">${fmtUsd(r.tp1)} • ${fmtUsd(r.tp2)} • ${fmtUsd(r.tp3)}</td>
            <td class="ltr-num">${fmtSignedUsd(r.actualMove)} (${r.actualDir})</td>
            <td><span class="outcome-chip ${badgeCls}">${r.outcomeBadge}</span></td>
            <td><strong class="ltr-num ${pnlCls}">${pnlTxt}</strong></td>
          </tr>
        `;
      })
      .join("");
  }

  async function initialBoot() {
    try {
      const [healthRes, summaryRes, todayRes] = await Promise.all([
        fetch("/api/health"),
        fetch("/api/stats/summary"),
        fetch("/api/signals/today"),
      ]);

      if (healthRes.ok) {
        const h = await healthRes.json();
        $("hdr-version").textContent = `v${h.version}`;
        $("hdr-rss").textContent = `RSS: ${Number(h.rssMb).toFixed(1)} MiB`;
        $("hdr-updated").textContent = `UTC: ${h.lastUpdatedUtc}`;
      }

      if (summaryRes.ok) {
        state.summaryPayload = await summaryRes.json();
        renderBenchmarkStrip(state.summaryPayload);
      }

      if (todayRes.ok) {
        state.todayPayload = await todayRes.json();
        state.liveDate = state.todayPayload.liveSession?.sessionDate || state.todayPayload.sessionDate || "";
        state.nextPreviewDate = state.todayPayload.nextSessionPreview?.sessionDate || "";
        applyFullSessionCard(state.todayPayload.liveSession || state.todayPayload);
      }

      await loadHistoryLedger();
    } catch (err) {
      console.error("Dashboard boot error:", err);
    }
  }

  function bindEvents() {
    // Quick Tabs: Live Session vs Next Rollover Preview
    $("tab-live-session").addEventListener("click", () => {
      if (state.todayPayload && state.todayPayload.liveSession) {
        applyFullSessionCard(state.todayPayload.liveSession);
      } else if (state.liveDate) {
        loadSessionByDate(state.liveDate);
      }
    });

    $("tab-next-session").addEventListener("click", () => {
      if (state.todayPayload && state.todayPayload.nextSessionPreview) {
        applyFullSessionCard(state.todayPayload.nextSessionPreview);
      } else if (state.nextPreviewDate) {
        loadSessionByDate(state.nextPreviewDate);
      }
    });

    // Date Picker Input + Prev/Next Session Buttons
    $("session-date-picker").addEventListener("change", (e) => {
      const val = e.target.value;
      if (val) {
        loadSessionByDate(val);
      }
    });

    $("btn-prev-session").addEventListener("click", () => {
      if (state.prevNavDate) {
        loadSessionByDate(state.prevNavDate);
      }
    });

    $("btn-next-session").addEventListener("click", () => {
      if (state.nextNavDate) {
        loadSessionByDate(state.nextNavDate);
      }
    });

    // Delegated click handler on History Ledger rows
    $("tbody-history-ledger").addEventListener("click", (e) => {
      const row = e.target.closest("tr.clickable-ledger-row");
      if (!row || !row.dataset.date) return;
      loadSessionByDate(row.dataset.date, { scrollToHero: true });
    });

    // Live Refresh Button
    const btnRefresh = $("btn-live-refresh");
    btnRefresh.addEventListener("click", async () => {
      btnRefresh.disabled = true;
      $("refresh-icon").classList.add("spin");
      try {
        const res = await fetch("/api/refresh", { method: "POST" });
        if (res.ok) {
          const fresh = await res.json();
          state.todayPayload = fresh;
          state.liveDate = fresh.liveSession?.sessionDate || fresh.sessionDate || state.liveDate;
          state.nextPreviewDate = fresh.nextSessionPreview?.sessionDate || state.nextPreviewDate;
          $("hdr-rss").textContent = `RSS: ${Number(fresh.rssMb).toFixed(1)} MiB`;
          $("hdr-updated").textContent = `UTC: ${fresh.lastUpdatedUtc}`;
          applyFullSessionCard(fresh.liveSession || fresh);
          showToast(
            `✓ بروزرسانی زنده انجام شد (${fresh.feedSource} • Spot ${fmtUsd(fresh.spotPrice)} • RSS ${fresh.rssMb} MiB)`
          );
        }
      } catch (err) {
        console.error("Refresh failed:", err);
      } finally {
        btnRefresh.disabled = false;
        $("refresh-icon").classList.remove("spin");
      }
    });

    // Twins Drawer Toggle
    $("btn-toggle-twins").addEventListener("click", () => {
      state.twinsExpanded = !state.twinsExpanded;
      const drawer = $("matched-twins-drawer");
      drawer.classList.toggle("collapsed", !state.twinsExpanded);
      $("btn-toggle-twins").setAttribute("aria-expanded", String(state.twinsExpanded));
      $("tw-toggle-arrow").textContent = state.twinsExpanded ? "▾" : "◂";
    });

    // Ledger Tier Pills
    document.querySelectorAll(".filter-pill").forEach((pill) => {
      pill.addEventListener("click", () => {
        document.querySelectorAll(".filter-pill").forEach((p) => p.classList.remove("active"));
        pill.classList.add("active");
        state.ledgerTier = pill.dataset.tier || "ALL";
        loadHistoryLedger();
      });
    });

    // Ledger Selects & Search
    $("flt-split").addEventListener("change", (e) => {
      state.ledgerSplit = e.target.value;
      loadHistoryLedger();
    });
    $("flt-day").addEventListener("change", (e) => {
      state.ledgerDay = e.target.value;
      loadHistoryLedger();
    });
    $("flt-search").addEventListener("input", (e) => {
      window.clearTimeout(state.searchDebounce);
      state.searchDebounce = window.setTimeout(() => {
        state.ledgerQuery = e.target.value;
        loadHistoryLedger();
      }, 220);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    bindEvents();
    initialBoot();
  });
})();
