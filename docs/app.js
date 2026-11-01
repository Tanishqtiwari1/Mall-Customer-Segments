(() => {
  "use strict";

  const M = window.SEGMENTATION_MODEL;
  if (!M) {
    document.querySelector("main").insertAdjacentHTML("afterbegin",
      '<p class="container" style="padding:24px;color:#b91c1c">Model data not found. Run <code>python -m src.segmentation</code> to generate docs/data/model.js.</p>');
    return;
  }

  const SEG_VARS = {
    "Premium Spenders": "--seg-premium",
    "Cautious Affluent": "--seg-cautious",
    "Impulse Spenders": "--seg-impulse",
    "Budget Conscious": "--seg-budget",
    "Mainstream Shoppers": "--seg-mainstream",
  };
  const segById = Object.fromEntries(M.segments.map((s) => [s.id, s]));
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const colorOf = (id) => cssVar(SEG_VARS[segById[id].name]);
  const varOf = (id) => `var(${SEG_VARS[segById[id].name]})`;
  const fmt = (n, d = 0) => Number(n).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
  const pct = (x) => `${Math.round(x * 100)}%`;
  const el = (tag, attrs = {}, html) => {
    const node = document.createElementNS(tag === "svg" || attrs.__svg ? "http://www.w3.org/2000/svg" : "http://www.w3.org/1999/xhtml", tag);
    for (const [k, v] of Object.entries(attrs)) if (k !== "__svg") node.setAttribute(k, v);
    if (html !== undefined) node.innerHTML = html;
    return node;
  };
  const svgEl = (tag, attrs = {}) => el(tag, { ...attrs, __svg: true });

  /* ---------- model: same rule as KMeans.predict (nearest centre in scaled space) ---------- */
  function rankSegments(income, spending) {
    const { scaler_mean: mu, scaler_scale: sd, centroids_scaled: C } = M.model;
    const z = [(income - mu[0]) / sd[0], (spending - mu[1]) / sd[1]];
    return C.map((c, id) => ({ id, d: Math.hypot(z[0] - c[0], z[1] - c[1]) })).sort((a, b) => a.d - b.d);
  }
  const predict = (income, spending) => rankSegments(income, spending)[0].id;

  /* ---------- state ---------- */
  const state = { income: 60, spending: 50, hidden: new Set() };
  const LIMITS = { income: [15, 140], spending: [1, 100] };
  const clamp = (v, [lo, hi]) => Math.min(hi, Math.max(lo, Math.round(Number(v) || lo)));

  /* ---------- hero stats ---------- */
  function renderStats() {
    const biggest = [...M.segments].sort((a, b) => b.size - a.size)[0];
    const items = [
      ["Customers analysed", fmt(M.dataset.rows)],
      ["Segments found", M.model.k],
      ["Silhouette score", `${fmt(M.model.silhouette, 2)}<small>(−1 to 1)</small>`],
      ["Largest segment", `${pct(biggest.share)}<small>${biggest.name}</small>`],
    ];
    document.getElementById("stats").innerHTML = items
      .map(([k, v]) => `<div class="stat"><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  }

  /* ---------- finder controls ---------- */
  const inputs = {
    income: [document.getElementById("income"), document.getElementById("income-num")],
    spending: [document.getElementById("spending"), document.getElementById("spending-num")],
  };

  function setCustomer(income, spending) {
    state.income = clamp(income, LIMITS.income);
    state.spending = clamp(spending, LIMITS.spending);
    for (const key of ["income", "spending"]) for (const input of inputs[key]) input.value = state[key];
    update();
  }

  function bindControls() {
    for (const key of ["income", "spending"]) {
      const [range, num] = inputs[key];
      range.addEventListener("input", () => setCustomer(key === "income" ? range.value : state.income, key === "spending" ? range.value : state.spending));
      num.addEventListener("change", () => setCustomer(key === "income" ? num.value : state.income, key === "spending" ? num.value : state.spending));
    }
    const presets = document.querySelector(".presets");
    for (const s of M.segments) {
      const b = el("button", { type: "button", class: "chip" });
      b.innerHTML = `<span class="dot" style="background:${varOf(s.id)}"></span>${s.name.split(" ")[0]}`;
      b.title = `Example: ${s.name} (income ${Math.round(s.center.income)}k$, spending ${Math.round(s.center.spending)})`;
      b.addEventListener("click", () => setCustomer(s.center.income, s.center.spending));
      presets.append(b);
    }
  }

  function renderResult() {
    const ranked = rankSegments(state.income, state.spending);
    const best = segById[ranked[0].id];
    const runner = segById[ranked[1].id];
    // 1 = right on the centre, 0 = exactly on the boundary between two segments
    const margin = 1 - ranked[0].d / ranked[1].d;
    const label = margin > 0.5 ? "Strong" : margin > 0.2 ? "Moderate" : "Borderline";
    const color = varOf(best.id);

    document.getElementById("result").style.setProperty("--seg-color", color);
    document.getElementById("result-swatch").style.background = color;
    document.getElementById("result-name").textContent = best.name;
    document.getElementById("result-tagline").textContent = best.tagline;
    document.getElementById("result-tagline").style.color = color;
    document.getElementById("result-desc").textContent = best.description;
    document.getElementById("result-conf").textContent = label;
    const bar = document.getElementById("result-bar");
    bar.style.width = `${Math.max(6, margin * 100)}%`;
    bar.style.background = color;
    document.getElementById("result-runner").textContent = label === "Borderline"
      ? `Close to the boundary with ${runner.name}. A small change could flip the segment.`
      : `Next closest segment: ${runner.name}.`;
    document.getElementById("result-strategy").innerHTML = best.strategy.map((s) => `<li>${s}</li>`).join("");

    document.querySelectorAll(".seg-card").forEach((c) => c.classList.toggle("highlight", Number(c.dataset.id) === best.id));
  }

  /* ---------- scatter plot ---------- */
  const W = 720, H = 500, PAD = { l: 54, r: 18, t: 16, b: 48 };
  const X_DOM = [10, 140], Y_DOM = [0, 100];
  const sx = (v) => PAD.l + ((v - X_DOM[0]) / (X_DOM[1] - X_DOM[0])) * (W - PAD.l - PAD.r);
  const sy = (v) => H - PAD.b - ((v - Y_DOM[0]) / (Y_DOM[1] - Y_DOM[0])) * (H - PAD.t - PAD.b);
  const ix = (px) => X_DOM[0] + ((px - PAD.l) / (W - PAD.l - PAD.r)) * (X_DOM[1] - X_DOM[0]);
  const iy = (py) => Y_DOM[0] + ((H - PAD.b - py) / (H - PAD.t - PAD.b)) * (Y_DOM[1] - Y_DOM[0]);

  const svg = document.getElementById("scatter");
  const tooltip = document.getElementById("tooltip");
  let regionImg, pointsG, centroidG, youG;

  function buildScatter() {
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    regionImg = svgEl("image", { x: PAD.l, y: PAD.t, width: W - PAD.l - PAD.r, height: H - PAD.t - PAD.b, preserveAspectRatio: "none" });
    svg.append(regionImg);

    const grid = svgEl("g");
    for (let v = 20; v <= 140; v += 20) {
      grid.append(svgEl("line", { class: "grid-line", x1: sx(v), x2: sx(v), y1: PAD.t, y2: H - PAD.b }));
      grid.append(Object.assign(svgEl("text", { class: "axis-label", x: sx(v), y: H - PAD.b + 18, "text-anchor": "middle" }), { textContent: v }));
    }
    for (let v = 0; v <= 100; v += 20) {
      grid.append(svgEl("line", { class: "grid-line", x1: PAD.l, x2: W - PAD.r, y1: sy(v), y2: sy(v) }));
      grid.append(Object.assign(svgEl("text", { class: "axis-label", x: PAD.l - 10, y: sy(v) + 4, "text-anchor": "end" }), { textContent: v }));
    }
    grid.append(svgEl("line", { class: "axis-line", x1: PAD.l, x2: W - PAD.r, y1: H - PAD.b, y2: H - PAD.b }));
    grid.append(svgEl("line", { class: "axis-line", x1: PAD.l, x2: PAD.l, y1: PAD.t, y2: H - PAD.b }));
    grid.append(Object.assign(svgEl("text", { class: "axis-title", x: (PAD.l + W - PAD.r) / 2, y: H - 8, "text-anchor": "middle" }), { textContent: "Annual income (k$)" }));
    grid.append(Object.assign(svgEl("text", { class: "axis-title", x: -(PAD.t + H - PAD.b) / 2, y: 16, transform: "rotate(-90)", "text-anchor": "middle" }), { textContent: "Spending score (1–100)" }));
    svg.append(grid);

    pointsG = svgEl("g");
    for (const c of M.customers) {
      const p = svgEl("circle", { class: "pt", cx: sx(c.income), cy: sy(c.spending), r: 5.5, tabindex: 0,
        "data-seg": c.segment, "aria-label": `Customer ${c.id}, ${segById[c.segment].name}` });
      p.addEventListener("mouseenter", () => showTip(p, c));
      p.addEventListener("focus", () => showTip(p, c));
      p.addEventListener("mouseleave", hideTip);
      p.addEventListener("blur", hideTip);
      p.addEventListener("click", (e) => { e.stopPropagation(); setCustomer(c.income, c.spending); });
      pointsG.append(p);
    }
    svg.append(pointsG);

    centroidG = svgEl("g");
    for (const s of M.segments) {
      const g = svgEl("g", { "data-seg": s.id, transform: `translate(${sx(s.center.income)},${sy(s.center.spending)})` });
      const arm = "M-7,-7L7,7M7,-7L-7,7";
      g.append(svgEl("path", { class: "centroid", d: arm }));
      g.append(svgEl("path", { class: "centroid-inner", d: arm }));
      centroidG.append(g);
    }
    svg.append(centroidG);

    youG = svgEl("g", { "aria-hidden": "true" });
    youG.innerHTML = `<line class="link-line"/><circle class="pulse" r="9"/><circle class="you-ring" r="11"/><circle class="you-dot" r="6"/><text class="you-label" dy="-18" text-anchor="middle">You</text>`;
    svg.append(youG);

    svg.addEventListener("click", (e) => {
      const pt = svg.createSVGPoint();
      pt.x = e.clientX; pt.y = e.clientY;
      const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
      if (loc.x < PAD.l || loc.x > W - PAD.r || loc.y < PAD.t || loc.y > H - PAD.b) return;
      setCustomer(ix(loc.x), iy(loc.y));
    });
  }

  function paintColors() {
    pointsG.querySelectorAll(".pt").forEach((p) => p.setAttribute("fill", colorOf(+p.dataset.seg)));
    centroidG.querySelectorAll("g").forEach((g) => g.querySelector(".centroid-inner").setAttribute("stroke", colorOf(+g.dataset.seg)));
    drawRegions();
  }

  function drawRegions() {
    if (!document.getElementById("show-regions").checked) { regionImg.removeAttribute("href"); return; }
    const w = 260, h = Math.round(w * (H - PAD.t - PAD.b) / (W - PAD.l - PAD.r));
    const canvas = document.createElement("canvas");
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(w, h);
    const rgb = M.segments.map((s) => hexToRgb(colorOf(s.id)));
    const dark = getComputedStyle(document.documentElement).colorScheme === "dark";
    const alpha = dark ? 46 : 30;
    for (let j = 0; j < h; j++) {
      const sp = Y_DOM[1] - ((j + 0.5) / h) * (Y_DOM[1] - Y_DOM[0]);
      for (let i = 0; i < w; i++) {
        const inc = X_DOM[0] + ((i + 0.5) / w) * (X_DOM[1] - X_DOM[0]);
        const id = predict(inc, sp);
        const k = (j * w + i) * 4;
        const [r, g, b] = rgb[id];
        img.data[k] = r; img.data[k + 1] = g; img.data[k + 2] = b;
        img.data[k + 3] = state.hidden.has(id) ? 0 : alpha;
      }
    }
    ctx.putImageData(img, 0, 0);
    regionImg.setAttribute("href", canvas.toDataURL());
  }

  function hexToRgb(c) {
    const h = c.replace("#", "");
    const n = parseInt(h.length === 3 ? h.split("").map((x) => x + x).join("") : h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  function showTip(node, c) {
    const s = segById[c.segment];
    tooltip.innerHTML = `<strong>Customer #${c.id}</strong>${s.name}<br>${c.gender}, age ${c.age}<br>Income ${c.income}k$<br>Spending score ${c.spending}`;
    const box = svg.getBoundingClientRect(), wrap = svg.parentElement.getBoundingClientRect();
    const k = box.width / W;
    tooltip.style.left = `${box.left - wrap.left + node.cx.baseVal.value * k}px`;
    tooltip.style.top = `${box.top - wrap.top + node.cy.baseVal.value * k}px`;
    tooltip.hidden = false;
    node.classList.add("active");
  }
  function hideTip() {
    tooltip.hidden = true;
    svg.querySelectorAll(".pt.active").forEach((p) => p.classList.remove("active"));
  }

  function renderYou() {
    const id = predict(state.income, state.spending);
    const s = segById[id];
    const x = sx(state.income), y = sy(state.spending);
    const line = youG.querySelector(".link-line");
    line.setAttribute("x1", x); line.setAttribute("y1", y);
    line.setAttribute("x2", sx(s.center.income)); line.setAttribute("y2", sy(s.center.spending));
    for (const c of youG.querySelectorAll("circle, text")) {
      c.setAttribute(c.tagName === "text" ? "x" : "cx", x);
      c.setAttribute(c.tagName === "text" ? "y" : "cy", y);
    }
    youG.querySelector(".you-dot").style.fill = colorOf(id);
  }

  function renderLegend() {
    const legend = document.getElementById("legend");
    for (const s of M.segments) {
      const b = el("button", { type: "button", class: "chip", "aria-pressed": "true" });
      b.innerHTML = `<span class="dot" style="background:${varOf(s.id)}"></span>${s.name}`;
      b.addEventListener("click", () => {
        const on = state.hidden.has(s.id);
        on ? state.hidden.delete(s.id) : state.hidden.add(s.id);
        b.setAttribute("aria-pressed", String(on));
        pointsG.querySelectorAll(`.pt[data-seg="${s.id}"]`).forEach((p) => p.classList.toggle("dim", !on));
        drawRegions();
      });
      legend.append(b);
    }
    document.getElementById("show-regions").addEventListener("change", drawRegions);
  }

  /* ---------- segment cards ---------- */
  function renderSegments() {
    const grid = document.getElementById("segment-grid");
    const sorted = [...M.segments].sort((a, b) => b.size - a.size);
    for (const s of sorted) {
      const card = el("article", { class: "card seg-card", "data-id": s.id, style: `--seg-color:${varOf(s.id)}` });
      card.innerHTML = `
        <div class="seg-top">
          <div><h3>${s.name}</h3><p class="seg-tag">${s.tagline}</p></div>
          <div class="seg-size"><strong>${s.size}</strong><span>customers · ${pct(s.share)}</span></div>
        </div>
        <div class="share-bar" aria-hidden="true"><span style="width:${s.share * 100 / sorted[0].share}%"></span></div>
        <p class="seg-desc">${s.description}</p>
        <div class="metrics">
          <div class="metric"><span>Avg income</span><strong>${fmt(s.avg_income)}k$</strong></div>
          <div class="metric"><span>Avg spending</span><strong>${fmt(s.avg_spending)}</strong></div>
          <div class="metric"><span>Avg age</span><strong>${fmt(s.avg_age)}</strong></div>
        </div>
        <div class="gender"><span>${pct(s.female_share)} female</span><div class="split" role="img" aria-label="${pct(s.female_share)} female, ${pct(1 - s.female_share)} male"><span style="width:${s.female_share * 100}%"></span></div><span>${pct(1 - s.female_share)} male</span></div>
        <ul class="checklist">${s.strategy.map((x) => `<li>${x}</li>`).join("")}</ul>
        <button class="try-btn" type="button">Show on the map →</button>`;
      card.querySelector(".try-btn").addEventListener("click", () => {
        setCustomer(s.center.income, s.center.spending);
        document.getElementById("finder").scrollIntoView();
      });
      grid.append(card);
    }
  }

  /* ---------- method mini charts ---------- */
  function lineChart(svgId, xs, ys, bestX, fmtY, zeroBased = false) {
    const node = document.getElementById(svgId);
    const w = 480, h = 240, p = { l: 52, r: 16, t: 20, b: 34 };
    node.setAttribute("viewBox", `0 0 ${w} ${h}`);
    const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
    const lo = Math.min(...ys), hi = Math.max(...ys), padY = (hi - lo) * 0.1;
    const [y0, y1] = [zeroBased ? 0 : lo - padY, hi + padY];
    const X = (v) => p.l + ((v - x0) / (x1 - x0)) * (w - p.l - p.r);
    const Y = (v) => h - p.b - ((v - y0) / (y1 - y0)) * (h - p.t - p.b);
    let out = "";
    for (let i = 0; i <= 4; i++) {
      const v = y0 + ((y1 - y0) * i) / 4;
      out += `<line class="grid-line" x1="${p.l}" x2="${w - p.r}" y1="${Y(v)}" y2="${Y(v)}"/><text class="axis-label" x="${p.l - 8}" y="${Y(v) + 4}" text-anchor="end">${fmtY(v)}</text>`;
    }
    for (const x of xs) out += `<text class="axis-label" x="${X(x)}" y="${h - p.b + 18}" text-anchor="middle">${x}</text>`;
    out += `<text class="axis-title" x="${(p.l + w - p.r) / 2}" y="${h - 2}" text-anchor="middle">Number of clusters (k)</text>`;
    out += `<line class="grid-line" x1="${X(bestX)}" x2="${X(bestX)}" y1="${p.t}" y2="${h - p.b}" stroke-dasharray="4 4" style="stroke:var(--accent);opacity:.5"/>`;
    out += `<polyline class="line" points="${xs.map((x, i) => `${X(x)},${Y(ys[i])}`).join(" ")}"/>`;
    xs.forEach((x, i) => { out += `<circle class="line-pt${x === bestX ? " best" : ""}" cx="${X(x)}" cy="${Y(ys[i])}" r="${x === bestX ? 6 : 4}"/>`; });
    const bi = xs.indexOf(bestX);
    out += `<text class="best-label" x="${X(bestX) + 10}" y="${Y(ys[bi]) - 10}">k = ${bestX}</text>`;
    node.innerHTML = out;
  }

  function renderMethod() {
    const d = M.diagnostics;
    lineChart("elbow-chart", d.k, d.wcss, M.model.k, (v) => fmt(v), true);
    lineChart("silhouette-chart", d.silhouette_k, d.silhouette, M.model.k, (v) => fmt(v, 2));
  }

  /* ---------- theme ---------- */
  function bindTheme() {
    const btn = document.getElementById("theme-toggle");
    btn.addEventListener("click", () => {
      const dark = getComputedStyle(document.documentElement).colorScheme === "dark";
      const next = dark ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("theme", next); } catch (e) {}
      paintColors(); renderYou();
    });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { paintColors(); renderYou(); });
  }

  /* ---------- wire up ---------- */
  function update() { renderResult(); renderYou(); }

  renderStats();
  bindControls();
  buildScatter();
  renderLegend();
  renderSegments();
  renderMethod();
  bindTheme();
  paintColors();
  setCustomer(state.income, state.spending);
})();
