/* credit-default-mlops: prediction page. Sends the form to the live Render API (/health to wake
   it, then /explain) and shows the calibrated probability, the decision at the chosen cutoff and
   the SHAP contributions. Nothing is computed or faked in the browser: if the API cannot be
   reached the page says so, and precomputed results are offered only with that label. */
(function () {
  "use strict";
  var C = window.SBCharts;
  var WAKE_LIMIT_S = 120;
  var warm = false;

  function q(id) { return document.getElementById(id); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function timeout(ms) {
    if (window.AbortSignal && AbortSignal.timeout) return AbortSignal.timeout(ms);
    var c = new AbortController(); setTimeout(function () { c.abort(); }, ms); return c.signal;
  }

  function setStatus(kind, html) {
    var s = q("pf-status");
    s.className = "pf-status is-" + kind;
    s.innerHTML = html;
  }

  function wake(api) {
    var t0 = Date.now();
    return new Promise(function (resolve, reject) {
      function tick() {
        var secs = Math.round((Date.now() - t0) / 1000);
        if (secs > WAKE_LIMIT_S) { reject(new Error("the API did not wake within " + WAKE_LIMIT_S + " s")); return; }
        setStatus("waking", '<span class="sb-spinner" aria-hidden="true"></span> Waking the API on Render\'s free tier: <b>' + secs + " s</b> (usually 30–60 s on the first request, then well under a second).");
        fetch(api + "/health", { signal: timeout(15000), cache: "no-store" })
          .then(function (r) { if (r.ok) resolve(r.json()); else setTimeout(tick, 2500); })
          .catch(function () { setTimeout(tick, 2500); });
      }
      tick();
    });
  }

  function payload(form) {
    var out = {};
    Array.prototype.forEach.call(form.querySelectorAll("[data-feature]"), function (n) { out[n.name] = parseInt(n.value, 10); });
    return out;
  }
  function fill(form, values) {
    Object.keys(values).forEach(function (k) { var n = form.elements[k]; if (n) n.value = values[k]; });
  }

  function waterfall(host, res) {
    var rows = res.contributions.slice();
    var other = res.other_features_contribution || 0;
    var start = res.base_value, items = [];
    rows.forEach(function (c) { items.push({ label: c.feature + " = " + fmtVal(c.value), v: c.shap_value }); });
    if (Math.abs(other) > 1e-9) items.push({ label: "other features", v: other });
    var pos = start, segs = [];
    items.forEach(function (it) { segs.push({ label: it.label, a: pos, b: pos + it.v, v: it.v }); pos += it.v; });
    var all = segs.reduce(function (acc, s) { return acc.concat([s.a, s.b]); }, [start]);
    var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all), pad = (hi - lo) * 0.08 || 0.1;
    lo -= pad; hi += pad;
    var w = Math.max(300, host.clientWidth || 600), rowH = 28, m = { t: 10, r: 70, b: 30, l: Math.min(190, w * 0.4) };
    var h = m.t + m.b + rowH * (segs.length + 2), iw = w - m.l - m.r;
    var X = function (v) { return m.l + ((v - lo) / (hi - lo)) * iw; };
    var NS = "http://www.w3.org/2000/svg";
    var svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", "0 0 " + w + " " + h); svg.setAttribute("width", "100%"); svg.setAttribute("class", "chart");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "SHAP waterfall: base value " + start.toFixed(2) + " to raw score " + res.raw_score.toFixed(2));
    function text(x, y, s, cls, anchor) {
      var t = document.createElementNS(NS, "text"); t.setAttribute("x", x); t.setAttribute("y", y);
      t.setAttribute("class", "chart-tick " + (cls || "")); if (anchor) t.setAttribute("text-anchor", anchor);
      t.textContent = s; svg.appendChild(t);
    }
    function rect(x, y, wd, cls) {
      var r = document.createElementNS(NS, "rect"); r.setAttribute("x", x); r.setAttribute("y", y);
      r.setAttribute("width", Math.max(2, wd)); r.setAttribute("height", rowH - 10); r.setAttribute("rx", 3);
      r.setAttribute("class", "chart-bar " + cls); svg.appendChild(r);
    }
    var y = m.t;
    text(m.l - 8, y + rowH / 2 + 4, "base value", "chart-rowlabel", "end");
    rect(Math.min(X(0 > lo ? 0 : lo), X(start)), y + 5, Math.abs(X(start) - X(0 > lo ? 0 : lo)), "s-muted");
    text(X(start) + 6, y + rowH / 2 + 4, start.toFixed(3), "chart-val");
    segs.forEach(function (s, i) {
      y = m.t + rowH * (i + 1);
      text(m.l - 8, y + rowH / 2 + 4, s.label, "chart-rowlabel", "end");
      rect(X(Math.min(s.a, s.b)), y + 5, Math.abs(X(s.b) - X(s.a)), s.v > 0 ? "s-up" : "s-down");
      text(X(Math.max(s.a, s.b)) + 6, y + rowH / 2 + 4, (s.v > 0 ? "+" : "−") + Math.abs(s.v).toFixed(3), "chart-val");
    });
    y = m.t + rowH * (segs.length + 1);
    text(m.l - 8, y + rowH / 2 + 4, "raw score", "chart-rowlabel", "end");
    var line = document.createElementNS(NS, "line");
    line.setAttribute("x1", X(res.raw_score)); line.setAttribute("x2", X(res.raw_score));
    line.setAttribute("y1", m.t); line.setAttribute("y2", y + rowH - 4); line.setAttribute("class", "chart-ref");
    svg.appendChild(line);
    text(X(res.raw_score) + 6, y + rowH / 2 + 4, res.raw_score.toFixed(3), "chart-val");
    host.innerHTML = ""; host.appendChild(svg);
  }
  function fmtVal(v) { return Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : (Math.round(v * 1000) / 1000).toString(); }

  function show(res, cutoff, source, precomputed) {
    var box = q("pf-result");
    box.hidden = false;
    var p = res.probability_of_default, decline = p >= cutoff;
    q("pf-prob").textContent = (100 * p).toFixed(1) + "%";
    var meter = q("pf-meter");
    meter.style.width = Math.min(100, 100 * p).toFixed(1) + "%";
    q("pf-cut-mark").style.left = Math.min(100, 100 * cutoff).toFixed(1) + "%";
    var dec = q("pf-decision");
    dec.textContent = decline ? "Decline" : "Approve";
    dec.className = "sb-pill " + (decline ? "pill-decline" : "pill-ok");
    q("pf-cutoff").textContent = cutoff.toFixed(3) + " (" + source + ")";
    q("pf-version").textContent = (res.model_version || "?") + " · " + (res.calibration || "") + " calibration";
    q("pf-source").textContent = precomputed ? "Precomputed at site build time by the committed model (not a live call)" : "Live response from the Render API";
    var up = res.contributions.filter(function (c) { return c.shap_value > 0; }).slice(0, 3).map(function (c) { return c.feature; });
    var down = res.contributions.filter(function (c) { return c.shap_value < 0; }).slice(0, 3).map(function (c) { return c.feature; });
    var space = (res.explanation_space || "").indexOf("probability") >= 0 ? "probability" : "log-odds";
    q("pf-story").innerHTML =
      "Raised the score: " + (up.length ? up.map(esc).join(", ") : "none of the top features") +
      ". Lowered it: " + (down.length ? down.map(esc).join(", ") : "none of the top features") + ". " +
      "Base value " + res.base_value.toFixed(3) + " plus all contributions gives the raw " + space + " score " + res.raw_score.toFixed(3) +
      " (uncalibrated probability " + (100 * res.uncalibrated_probability).toFixed(1) + "%), which the " + esc(res.calibration || "") +
      " calibrator maps to " + (100 * p).toFixed(1) + "%. Contributions describe this model's arithmetic, not causes.";
    waterfall(q("pf-waterfall"), res);
    box.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "nearest" });
  }

  function init() {
    var form = q("pf");
    if (!form) return;
    var api = form.getAttribute("data-api");
    var examples = C.data("examples-data") || { examples: [] };
    var policy = parseFloat(form.getAttribute("data-policy"));
    var thr = q("pf-threshold"), thrOut = q("pf-threshold-out");
    var fnI = q("pf-fn"), fpI = q("pf-fp");
    var source = "policy";
    function syncThr() { thrOut.textContent = parseFloat(thr.value).toFixed(3); }
    thr.addEventListener("input", function () { source = "your cutoff"; syncThr(); });
    q("pf-use-costs").addEventListener("click", function () {
      var fn = parseFloat(fnI.value), fp = parseFloat(fpI.value);
      if (!(fn > 0 && fp > 0)) return;
      thr.value = (fp / (fp + fn)).toFixed(3); source = "Bayes cutoff for " + fn + ":" + fp; syncThr();
    });
    q("pf-use-policy").addEventListener("click", function () { thr.value = policy.toFixed(3); source = "policy"; syncThr(); });
    thr.value = policy.toFixed(3); syncThr();

    var pre = q("pf-precomputed");
    function useExample(i, announce) {
      var ex = examples.examples[i];
      if (!ex) return;
      fill(form, ex.features);
      if (announce) setStatus("idle", "Loaded “" + esc(ex.name) + "”. Press <b>Score with the live API</b>, or view the precomputed result.");
      pre.hidden = !ex.result;
      pre.onclick = function () { show(ex.result, parseFloat(thr.value), source, true); };
    }
    document.querySelectorAll("[data-example]").forEach(function (b) {
      b.addEventListener("click", function () { useExample(parseInt(b.getAttribute("data-example"), 10), true); });
    });
    // The form starts filled with the first example; editing any field drops its precomputed result.
    useExample(0, false);
    form.addEventListener("input", function (e) { if (e.target.hasAttribute("data-feature")) pre.hidden = true; });

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      if (!form.reportValidity()) return;
      var btn = q("pf-submit");
      btn.disabled = true;
      var cutoff = parseFloat(thr.value);
      var started = warm ? Promise.resolve() : wake(api);
      started
        .then(function () {
          warm = true;
          setStatus("busy", '<span class="sb-spinner" aria-hidden="true"></span> Scoring…');
          return fetch(api + "/explain?top_k=8&threshold=" + encodeURIComponent(cutoff), {
            method: "POST", headers: { "content-type": "application/json" },
            body: JSON.stringify(payload(form)), signal: timeout(30000),
          });
        })
        .then(function (r) {
          return r.json().then(function (body) {
            if (!r.ok) {
              var d = body && body.detail;
              var msg = Array.isArray(d) ? d.map(function (x) { return (x.loc || []).slice(-1)[0] + ": " + x.msg; }).join("; ") : d || ("HTTP " + r.status);
              throw new Error(msg);
            }
            return body;
          });
        })
        .then(function (res) {
          setStatus("ok", "Scored by the live API in model " + esc(res.model_version) + " (request " + esc(res.request_id.slice(0, 12)) + ").");
          show(res, cutoff, source, false);
        })
        .catch(function (err) {
          setStatus("error", "The live API could not score this request: " + esc(err.message || err) +
            ". The free tier may be restarting; try again in a minute, use the <a href=\"" + esc(api) + "/docs\" target=\"_blank\" rel=\"noopener\">Swagger UI</a>, or load an example and view its precomputed result.");
        })
        .finally(function () { btn.disabled = false; });
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
