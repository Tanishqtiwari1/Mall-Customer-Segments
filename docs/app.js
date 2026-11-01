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

  // Soft K-Means: a Gaussian with the shared within-cluster variance around each centre.
  // Sorted by distance, so the most likely segment is always the K-Means segment.
  function segmentProbabilities(income, spending) {
    const ranked = rankSegments(income, spending);
    const logits = ranked.map((r) => -(r.d * r.d) / (2 * M.model.sigma2));
    const w = logits.map((l) => Math.exp(l - logits[0]));
    const total = w.reduce((a, b) => a + b, 0);
    return ranked.map((r, i) => ({ ...r, p: w[i] / total }));
  }
  const probText = (p) => (p > 0.995 ? ">99%" : p < 0.005 ? "<1%" : `${Math.round(p * 100)}%`);

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
    const ranked = segmentProbabilities(state.income, state.spending);
    const best = segById[ranked[0].id];
    const runner = segById[ranked[1].id];
    const p = ranked[0].p;
    const shown = Math.round(p * 100); // label from the rounded value so "90%" never reads as Moderate
    const label = shown >= 90 ? "Strong" : shown >= 60 ? "Moderate" : "Borderline";
    const color = varOf(best.id);

    document.getElementById("result").style.setProperty("--seg-color", color);
    document.getElementById("result-swatch").style.background = color;
    document.getElementById("result-name").textContent = best.name;
    document.getElementById("result-tagline").textContent = best.tagline;
    document.getElementById("result-tagline").style.color = color;
    document.getElementById("result-desc").textContent = best.description;
    document.getElementById("result-conf").textContent = `${probText(p)} · ${label}`;
    document.getElementById("prob-list").innerHTML = ranked.map((r, i) => `
      <li class="${i === 0 ? "top" : ""}">
        <span class="prob-name"><span class="dot" style="background:${varOf(r.id)}"></span>${segById[r.id].name}</span>
        <span class="prob-bar"><span style="width:${r.p * 100}%;background:${varOf(r.id)}"></span></span>
        <span class="prob-val">${probText(r.p)}</span>
        <span class="prob-dist">distance to centre d = ${r.d.toFixed(2)}</span>
      </li>`).join("");
    document.getElementById("result-runner").textContent = label === "Borderline"
      ? `Close to the boundary with ${runner.name} (${probText(ranked[1].p)}). A small change could flip the segment.`
      : `Next most likely: ${runner.name} (${probText(ranked[1].p)}).`;
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
    svg.append(axesGroup());

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

  // grid lines, tick labels and axis titles shared by the segment map and the k explorer
  function axesGroup() {
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
    return grid;
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

  /* ---------- k explorer: K-Means re-run live in the browser ---------- */
  const K_PALETTE = ["#2563eb", "#db2777", "#0d9488", "#d97706", "#7c3aed", "#16a34a", "#dc2626", "#0891b2", "#ca8a04", "#64748b"];
  const DEFAULT_SEED = 42;
  const scaled = M.customers.map((c) => [
    (c.income - M.model.scaler_mean[0]) / M.model.scaler_scale[0],
    (c.spending - M.model.scaler_mean[1]) / M.model.scaler_scale[1],
  ]);
  const sqDist = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
  const pairDist = scaled.map((a) => scaled.map((b) => Math.sqrt(sqDist(a, b))));

  function mulberry32(seed) {
    return () => {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function kmeansOnce(k, rnd) {
    // k-means++ initialisation: each new centre is picked with probability proportional to squared distance
    const centers = [scaled[Math.floor(rnd() * scaled.length)].slice()];
    const nearest = scaled.map((p) => sqDist(p, centers[0]));
    while (centers.length < k) {
      let r = rnd() * nearest.reduce((a, b) => a + b, 0), i = 0;
      for (; i < scaled.length - 1; i++) { r -= nearest[i]; if (r <= 0) break; }
      centers.push(scaled[i].slice());
      scaled.forEach((p, j) => { nearest[j] = Math.min(nearest[j], sqDist(p, scaled[i])); });
    }
    const labels = new Array(scaled.length).fill(-1);
    for (let iter = 0; iter < 300; iter++) {
      let changed = false;
      scaled.forEach((p, j) => {
        let best = 0;
        for (let c = 1; c < k; c++) if (sqDist(p, centers[c]) < sqDist(p, centers[best])) best = c;
        if (labels[j] !== best) { labels[j] = best; changed = true; }
      });
      if (!changed) break;
      for (let c = 0; c < k; c++) {
        const members = scaled.filter((_, j) => labels[j] === c);
        if (members.length) centers[c] = [0, 1].map((d) => members.reduce((s, p) => s + p[d], 0) / members.length);
      }
    }
    const inertia = scaled.reduce((s, p, j) => s + sqDist(p, centers[labels[j]]), 0);
    return { centers, labels, inertia };
  }

  function kmeans(k, seed, nInit = 25) {
    const rnd = mulberry32(seed);
    let best = null;
    for (let i = 0; i < nInit; i++) {
      const run = kmeansOnce(k, rnd);
      if (!best || run.inertia < best.inertia) best = run;
    }
    // number clusters left-to-right, bottom-to-top so colours stay stable between runs
    const order = best.centers.map((c, i) => i).sort((a, b) => best.centers[a][0] - best.centers[b][0] || best.centers[a][1] - best.centers[b][1]);
    const remap = Object.fromEntries(order.map((old, i) => [old, i]));
    return { inertia: best.inertia, centers: order.map((i) => best.centers[i]), labels: best.labels.map((l) => remap[l]) };
  }

  function silhouette(labels, k) {
    let total = 0;
    const sizes = Array.from({ length: k }, (_, c) => labels.filter((l) => l === c).length);
    labels.forEach((li, i) => {
      if (sizes[li] < 2) return; // silhouette of a singleton is 0
      const sums = new Array(k).fill(0);
      labels.forEach((lj, j) => { if (i !== j) sums[lj] += pairDist[i][j]; });
      const a = sums[li] / (sizes[li] - 1);
      const b = Math.min(...sums.map((s, c) => (c === li || !sizes[c] ? Infinity : s / sizes[c])));
      total += (b - a) / Math.max(a, b);
    });
    return total / labels.length;
  }

  const kSvg = document.getElementById("k-scatter");
  const bestK = M.diagnostics.silhouette_k[M.diagnostics.silhouette.indexOf(Math.max(...M.diagnostics.silhouette))];
  const SIL_SCALE = 0.7;
  let kPoints = [], kCentroidG, kSeed = DEFAULT_SEED;

  function buildKExplorer() {
    kSvg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    kSvg.append(axesGroup());
    const g = svgEl("g");
    kPoints = M.customers.map((c) => {
      const p = svgEl("circle", { class: "k-pt", cx: sx(c.income), cy: sy(c.spending), r: 5.5 });
      g.append(p);
      return p;
    });
    kSvg.append(g);
    kCentroidG = svgEl("g");
    kSvg.append(kCentroidG);
    document.getElementById("k-sil-best").style.left = `${(Math.max(...M.diagnostics.silhouette) / SIL_SCALE) * 100}%`;

    const slider = document.getElementById("k-slider");
    slider.addEventListener("input", () => renderK(+slider.value));
    document.getElementById("k-rerun").addEventListener("click", () => {
      kSeed = Math.floor(Math.random() * 1e9);
      renderK(+slider.value);
    });
  }

  function renderK(k) {
    const { labels, centers, inertia } = kmeans(k, kSeed);
    const sil = silhouette(labels, k);
    document.getElementById("k-value").textContent = k;
    document.getElementById("k-sil").textContent = sil.toFixed(3);
    document.getElementById("k-wcss").textContent = inertia.toFixed(1);
    document.getElementById("k-sil-bar").style.width = `${Math.max(0, sil / SIL_SCALE) * 100}%`;

    kPoints.forEach((p, i) => p.setAttribute("fill", K_PALETTE[labels[i]]));
    kCentroidG.innerHTML = centers.map((c, i) => {
      const x = sx(c[0] * M.model.scaler_scale[0] + M.model.scaler_mean[0]);
      const y = sy(c[1] * M.model.scaler_scale[1] + M.model.scaler_mean[1]);
      return `<g transform="translate(${x},${y})"><path class="centroid" d="M-7,-7L7,7M7,-7L-7,7"/><path class="centroid-inner" stroke="${K_PALETTE[i]}" d="M-7,-7L7,7M7,-7L-7,7"/></g>`;
    }).join("");

    const sizes = Array.from({ length: k }, (_, c) => labels.filter((l) => l === c).length);
    const maxSize = Math.max(...sizes);
    document.getElementById("k-sizes").innerHTML = sizes.map((n, c) => `
      <li><span class="prob-name"><span class="dot" style="background:${K_PALETTE[c]}"></span>Cluster ${c + 1}</span>
      <span class="prob-bar"><span style="width:${(n / maxSize) * 100}%;background:${K_PALETTE[c]}"></span></span><b>${n}</b></li>`).join("");

    const verdict = document.getElementById("k-verdict");
    verdict.classList.toggle("best", k === bestK);
    verdict.textContent = k === bestK
      ? `✓ k = ${k} gives the best-separated clusters. This is the model used across the site.`
      : k < bestK
        ? `With only ${k} clusters, distinct customer groups get merged, so separation is lower than at k = ${bestK}.`
        : `With ${k} clusters, natural groups start getting split apart. The extra clusters add little, and separation is lower than at k = ${bestK}.`;
    document.getElementById("k-runinfo").textContent = kSeed === DEFAULT_SEED
      ? "Best of 25 k-means++ starts. For some k the numbers can differ slightly from the Method charts, because K-Means can settle in different local optima."
      : `Best of 25 new random starts (seed ${kSeed}). The partition barely changes, which shows the result is stable.`;
  }

  /* ---------- batch scoring ---------- */
  const SAMPLE_HEADERS = ["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"];
  const MAX_ROWS = 200000;
  const PREVIEW_ROWS = 100;
  let batchOutput = null;

  function parseCSV(text) {
    text = text.replace(/^﻿/, "");
    const firstLine = text.slice(0, text.indexOf("\n") >>> 0);
    const delim = (firstLine.match(/;/g) || []).length > (firstLine.match(/,/g) || []).length ? ";" : ",";
    const rows = [];
    let row = [], field = "", quoted = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (quoted) {
        if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
        else if (ch === '"') quoted = false;
        else field += ch;
      } else if (ch === '"') quoted = true;
      else if (ch === delim) { row.push(field); field = ""; }
      else if (ch === "\n" || ch === "\r") {
        if (ch === "\r" && text[i + 1] === "\n") i++;
        row.push(field); rows.push(row); row = []; field = "";
      } else field += ch;
    }
    if (field !== "" || row.length) { row.push(field); rows.push(row); }
    return rows.filter((r) => r.some((v) => v.trim() !== ""));
  }

  const csvCell = (v) => (/[",\n\r]/.test(v) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
  const toCSV = (rows) => rows.map((r) => r.map(csvCell).join(",")).join("\n") + "\n";

  function scoreRows(rows) {
    const header = rows[0].map((h) => h.trim());
    const incomeCol = header.findIndex((h) => /income/i.test(h));
    const spendCol = header.findIndex((h) => /spend/i.test(h));
    if (incomeCol < 0 || spendCol < 0) {
      throw new Error(`Couldn't find both an income column and a spending-score column. Columns found: ${header.join(", ") || "(none)"}.`);
    }
    if (rows.length - 1 > MAX_ROWS) throw new Error(`The file has ${fmt(rows.length - 1)} rows. The limit is ${fmt(MAX_ROWS)}.`);

    const [lo, hi] = M.dataset.income_range;
    const results = rows.slice(1).map((r, idx) => {
      const line = idx + 2;
      const income = Number(String(r[incomeCol] ?? "").trim());
      const spending = Number(String(r[spendCol] ?? "").trim());
      if (r[incomeCol] === undefined || String(r[incomeCol]).trim() === "" || !Number.isFinite(income) || income < 0) {
        return { raw: r, line, error: `income "${r[incomeCol] ?? ""}" is not a valid number` };
      }
      if (String(r[spendCol] ?? "").trim() === "" || !Number.isFinite(spending) || spending < 1 || spending > 100) {
        return { raw: r, line, error: `spending score "${r[spendCol] ?? ""}" must be a number from 1 to 100` };
      }
      const probs = segmentProbabilities(income, spending);
      return { raw: r, line, income, spending, segment: probs[0].id, p: probs[0].p,
               note: income < lo || income > hi ? `income outside training range (${lo}-${hi}k$)` : "" };
    });
    return { header, results };
  }

  function showBatchError(message) {
    const err = document.getElementById("batch-error");
    err.textContent = message;
    err.hidden = false;
    document.getElementById("batch-results").hidden = true;
  }

  function runBatch(text, fileName) {
    document.getElementById("batch-error").hidden = true;
    try {
      const rows = parseCSV(text);
      if (rows.length < 2) throw new Error("The file has no data rows.");
      const { header, results } = scoreRows(rows);
      if (!results.some((r) => !r.error)) throw new Error("No rows could be scored. Check that the income and spending columns contain numbers.");
      renderBatch(header, results, fileName);
    } catch (e) {
      showBatchError(e.message);
    }
  }

  function renderBatch(header, results, fileName) {
    const scored = results.filter((r) => !r.error);
    const skipped = results.filter((r) => r.error);
    const outside = scored.filter((r) => r.note).length;
    const counts = Object.fromEntries(M.segments.map((s) => [s.id, 0]));
    scored.forEach((r) => counts[r.segment]++);
    const order = [...M.segments].sort((a, b) => counts[b.id] - counts[a.id]);

    document.getElementById("batch-file").textContent = fileName;
    document.getElementById("batch-summary").textContent = `${fmt(scored.length)} customer${scored.length === 1 ? "" : "s"} segmented`;

    const dist = document.getElementById("batch-dist");
    dist.innerHTML = order.filter((s) => counts[s.id]).map((s) =>
      `<span style="width:${(counts[s.id] / scored.length) * 100}%;background:${varOf(s.id)}" title="${s.name}: ${counts[s.id]}"></span>`).join("");
    dist.setAttribute("aria-label", order.map((s) => `${s.name} ${counts[s.id]}`).join(", "));
    document.getElementById("batch-legend").innerHTML = order.map((s) =>
      `<li><span class="dot" style="background:${varOf(s.id)}"></span>${s.name} <strong>${fmt(counts[s.id])}</strong> <span class="hint">(${pct(counts[s.id] / scored.length)})</span></li>`).join("");

    const skippedBox = document.getElementById("batch-skipped");
    skippedBox.hidden = !skipped.length;
    if (skipped.length) {
      skippedBox.querySelector("summary").textContent = `${skipped.length} row${skipped.length === 1 ? "" : "s"} skipped. They stay in the download with the reason in the "note" column.`;
      const ul = skippedBox.querySelector("ul");
      ul.replaceChildren(...skipped.slice(0, 50).map((r) => Object.assign(document.createElement("li"), { textContent: `Line ${r.line}: ${r.error}` })));
      if (skipped.length > 50) ul.append(Object.assign(document.createElement("li"), { textContent: `…and ${skipped.length - 50} more` }));
    }

    // preview table: up to 6 original columns plus the prediction
    const cols = header.slice(0, 6);
    const table = document.getElementById("batch-table");
    const headRow = document.createElement("tr");
    [...cols, "Segment", "Probability"].forEach((h) => headRow.append(Object.assign(document.createElement("th"), { textContent: h })));
    table.tHead.replaceChildren(headRow);
    table.tBodies[0].replaceChildren(...scored.slice(0, PREVIEW_ROWS).map((r) => {
      const tr = document.createElement("tr");
      cols.forEach((_, i) => tr.append(Object.assign(document.createElement("td"), { textContent: r.raw[i] ?? "" })));
      const seg = document.createElement("td");
      seg.innerHTML = `<span class="seg-pill"><span class="dot" style="background:${varOf(r.segment)}"></span></span>`;
      seg.firstChild.append(segById[r.segment].name);
      tr.append(seg, Object.assign(document.createElement("td"), { textContent: probText(r.p) }));
      return tr;
    }));
    const notes = [];
    if (scored.length > PREVIEW_ROWS) notes.push(`Showing the first ${PREVIEW_ROWS} of ${fmt(scored.length)} rows. Download the CSV for all of them.`);
    if (outside) notes.push(`${outside} customer${outside === 1 ? " has an income" : "s have incomes"} outside the range the model was trained on, so treat ${outside === 1 ? "that result" : "those results"} with care.`);
    document.getElementById("batch-table-note").textContent = notes.join(" ");

    batchOutput = {
      name: fileName.replace(/\.csv$/i, "") + "_segmented.csv",
      rows: [[...header, "segment", "segment_probability", "note"], ...results.map((r) => [
        ...header.map((_, i) => r.raw[i] ?? ""),
        r.error ? "" : segById[r.segment].name,
        r.error ? "" : r.p.toFixed(3),
        r.error ? `skipped: ${r.error}` : r.note,
      ])],
    };
    document.getElementById("batch-results").hidden = false;
  }

  function downloadText(name, text) {
    const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: name });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function bindBatch() {
    const input = document.getElementById("csv-file");
    const zone = document.getElementById("dropzone");
    const readFile = async (file) => {
      if (!file) return;
      if (!/\.csv$/i.test(file.name) && file.type !== "text/csv") {
        showBatchError(`"${file.name}" is not a CSV file. Export your sheet as .csv and try again.`);
        return;
      }
      runBatch(await file.text(), file.name);
    };
    input.addEventListener("change", () => { readFile(input.files[0]); input.value = ""; });
    document.querySelector(".link-label").addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
    });
    ["dragenter", "dragover"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.remove("drag"); }));
    zone.addEventListener("drop", (e) => readFile(e.dataTransfer.files[0]));

    document.getElementById("batch-sample").addEventListener("click", () => {
      const rows = [SAMPLE_HEADERS, ...M.customers.map((c) => [c.id, c.gender, c.age, c.income, c.spending])];
      runBatch(toCSV(rows), "Mall_Customers.csv");
    });
    document.getElementById("batch-template").addEventListener("click", () => {
      downloadText("customers_template.csv", toCSV([["customer_id", "income", "spending"], ["1", "45", "60"], ["2", "90", "20"], ["3", "25", "85"]]));
    });
    document.getElementById("batch-download").addEventListener("click", () => {
      if (batchOutput) downloadText(batchOutput.name, toCSV(batchOutput.rows));
    });
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
  buildKExplorer();
  renderK(bestK);
  bindBatch();
  bindTheme();
  paintColors();
  setCustomer(state.income, state.spending);
})();
