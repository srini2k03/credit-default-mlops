/* credit-default-mlops: evaluation page. The decision slider recomputes the confusion matrix and
   every decision metric from the committed held-out predictions (data/predictions.json), so the
   numbers are the model's real test-split behaviour at any cutoff and cost ratio. */
(function () {
  "use strict";
  var C = window.SBCharts;

  function prep(split) {
    // Sort by probability (descending) once; then any cutoff is one binary search.
    var idx = split.p.map(function (_, i) { return i; });
    idx.sort(function (a, b) { return split.p[b] - split.p[a]; });
    var p = idx.map(function (i) { return split.p[i]; });
    var cum = [0];
    idx.forEach(function (i, k) { cum.push(cum[k] + split.y[i]); });
    return { p: p, cum: cum, n: p.length, pos: cum[cum.length - 1] };
  }
  function countAtLeast(sorted, t) {
    var lo = 0, hi = sorted.length; // first index with p < t (descending order)
    while (lo < hi) { var mid = (lo + hi) >> 1; if (sorted[mid] >= t) lo = mid + 1; else hi = mid; }
    return lo;
  }
  function metrics(s, t, fn, fp) {
    var k = countAtLeast(s.p, t);
    var TP = s.cum[k], FP = k - TP, FN = s.pos - TP, TN = s.n - s.pos - FP;
    var prec = TP + FP ? TP / (TP + FP) : 0, rec = s.pos ? TP / s.pos : 0;
    var spec = TN + FP ? TN / (TN + FP) : 0;
    return {
      tp: TP, fp: FP, tn: TN, fn: FN, n: s.n,
      cost: (FN * fn + FP * fp) / s.n, total: FN * fn + FP * fp,
      recall: rec, precision: prec, specificity: spec,
      f1: prec + rec ? (2 * prec * rec) / (prec + rec) : 0,
      fpr: 1 - spec, fnr: s.pos ? FN / s.pos : 0, approval: (TN + FN) / s.n,
    };
  }
  var GRID = []; for (var i = 1; i < 200; i++) GRID.push(i / 200);
  function optimum(s, fn, fp) {
    var best = GRID[0], bc = Infinity;
    GRID.forEach(function (t) { var c = metrics(s, t, fn, fp).cost; if (c < bc - 1e-12) { bc = c; best = t; } });
    return best;
  }
  var pct = function (v) { return (100 * v).toFixed(1) + "%"; };

  function initSlider(data) {
    var root = document.getElementById("ev-tool");
    if (!root) return;
    var splits = { test: prep(data.test), validation: prep(data.validation) };
    var q = function (id) { return document.getElementById(id); };
    var thr = q("ev-threshold"), out = q("ev-threshold-out"), fnI = q("ev-fn"), fpI = q("ev-fp"), sel = q("ev-split");
    var deployed = data.threshold;
    var chart = null;
    function costs() {
      var fn = parseFloat(fnI.value), fp = parseFloat(fpI.value);
      return { fn: isFinite(fn) && fn > 0 ? fn : 5, fp: isFinite(fp) && fp > 0 ? fp : 1 };
    }
    function render() {
      var c = costs(), t = parseFloat(thr.value), s = splits[sel.value];
      out.textContent = t.toFixed(3);
      var m = metrics(s, t, c.fn, c.fp);
      var at05 = metrics(s, 0.5, c.fn, c.fp);
      var set = function (k, v) { root.querySelectorAll('[data-ev="' + k + '"]').forEach(function (n) { n.textContent = v; }); };
      set("tp", m.tp.toLocaleString()); set("fp", m.fp.toLocaleString());
      set("tn", m.tn.toLocaleString()); set("fn", m.fn.toLocaleString());
      set("cost", m.cost.toFixed(4)); set("total", m.total.toLocaleString(undefined, { maximumFractionDigits: 1 }));
      set("recall", pct(m.recall)); set("precision", pct(m.precision)); set("approval", pct(m.approval));
      set("fpr", pct(m.fpr)); set("fnr", pct(m.fnr)); set("f1", m.f1.toFixed(3)); set("specificity", pct(m.specificity));
      set("n", s.n.toLocaleString());
      var vs = at05.cost ? m.cost / at05.cost - 1 : 0;
      set("vs05", (vs >= 0 ? "+" : "−") + Math.abs(100 * vs).toFixed(1) + "% vs the 0.5 cutoff");
      var opt = optimum(splits.validation, c.fn, c.fp);
      set("optimal", opt.toFixed(3)); set("bayes", (c.fp / (c.fp + c.fn)).toFixed(3));
      set("ratio", (c.fn / c.fp).toFixed(c.fn / c.fp % 1 ? 1 : 0));
      var curve = GRID.map(function (g) { return [g, metrics(s, g, c.fn, c.fp).cost]; });
      var ymax = Math.max.apply(null, curve.map(function (p) { return p[1]; })) * 1.05;
      var host = q("ev-chart");
      host.innerHTML = "";
      chart = C.line(host, {
        series: [{ label: "cost / applicant", cls: "s-1", points: curve }],
        xDomain: [0, 1], yDomain: [0, ymax], xLabel: "cutoff", yLabel: "cost / applicant",
        xFmt: function (v) { return v.toFixed(2); }, yFmt: function (v) { return v.toFixed(3); },
        vlines: [{ x: deployed, label: "deployed " + deployed.toFixed(3), cls: "is-muted" }],
        markers: [{ x: t, y: m.cost, cls: "s-2" }], height: 260,
        aria: "Expected cost per applicant against the cutoff on the " + sel.value + " split",
      });
    }
    [thr, fnI, fpI, sel].forEach(function (n) { n.addEventListener("input", render); n.addEventListener("change", render); });
    root.querySelectorAll("[data-ev-set]").forEach(function (b) {
      b.addEventListener("click", function () {
        var c = costs(), kind = b.getAttribute("data-ev-set");
        var t = kind === "deployed" ? deployed : kind === "bayes" ? c.fp / (c.fp + c.fn) : kind === "half" ? 0.5 : optimum(splits.validation, c.fn, c.fp);
        thr.value = (Math.round(t * 200) / 200).toFixed(3);
        render();
      });
    });
    thr.value = (Math.round(deployed * 200) / 200).toFixed(3);
    render();
    root.classList.add("is-ready");
  }

  function initCurves() {
    var ev = C.data("evaluation-data");
    if (!ev) return;
    var champ = ev.champion;
    var sel = document.getElementById("curve-family");
    function drawCurves() {
      var other = sel ? sel.value : null;
      var series = function (kind) {
        var s = [{ label: champ + " (champion)", cls: "s-1", points: ev.curves[champ][kind] }];
        if (other && other !== champ && ev.curves[other]) s.push({ label: other, cls: "s-2", points: ev.curves[other][kind] });
        return s;
      };
      var roc = document.getElementById("chart-roc"), pr = document.getElementById("chart-pr");
      if (roc) { roc.innerHTML = ""; C.line(roc, { series: series("roc"), diagonal: true, xLabel: "false-positive rate", yLabel: "recall", height: 280, aria: "ROC curves on the test split" }); }
      if (pr) { pr.innerHTML = ""; C.line(pr, { series: series("pr"), xLabel: "recall", yLabel: "precision", height: 280, aria: "Precision-recall curves on the test split" }); }
    }
    if (sel) sel.addEventListener("change", drawCurves);
    drawCurves();
    var rel = document.getElementById("chart-reliability");
    if (rel && ev.reliability) {
      var cls = { sigmoid: "s-1", isotonic: "s-2", none: "s-muted" };
      var names = Object.keys(ev.reliability);
      var maxv = 0;
      names.forEach(function (n) { ev.reliability[n].forEach(function (r) { maxv = Math.max(maxv, r.mean_predicted, r.observed_rate); }); });
      maxv = Math.min(1, Math.ceil(maxv * 10) / 10);
      C.line(rel, {
        series: names.map(function (n) {
          return { label: n, cls: cls[n] || "s-muted", points: ev.reliability[n].map(function (r) { return [r.mean_predicted, r.observed_rate]; }) };
        }),
        diagonal: true, xDomain: [0, maxv], yDomain: [0, maxv], xLabel: "mean predicted", yLabel: "observed default rate",
        height: 280, aria: "Reliability curves (deciles) on the test split",
      });
    }
    var dec = C.data("decision-data");
    var sens = document.getElementById("chart-sensitivity");
    if (dec && sens) {
      var rows = dec.sensitivity;
      var maxr = rows[rows.length - 1].ratio;
      C.line(sens, {
        series: [
          { label: "optimal cutoff (validation)", cls: "s-1", points: rows.map(function (r) { return [r.ratio, r.optimal_threshold]; }) },
          { label: "Bayes cutoff fp/(fp+fn)", cls: "s-2", dash: true, points: rows.map(function (r) { return [r.ratio, r.bayes_threshold]; }) },
        ],
        xDomain: [0, maxr], yDomain: [0, 0.6], xLabel: "FN : FP cost ratio", yLabel: "cutoff",
        xFmt: function (v) { return v.toFixed(0); }, yFmt: function (v) { return v.toFixed(2); },
        vlines: [{ x: dec.policy.ratio, label: "configured " + dec.policy.ratio + ":1", cls: "is-muted" }],
        height: 260, aria: "Optimal cutoff against the cost ratio",
      });
    }
  }

  function init() {
    initCurves();
    var root = document.getElementById("ev-tool");
    if (!root) return;
    var status = document.getElementById("ev-status");
    fetch(root.getAttribute("data-src"))
      .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
      .then(function (d) { if (status) status.textContent = "Loaded " + d.test.p.length.toLocaleString() + " test and " + d.validation.p.length.toLocaleString() + " validation predictions."; initSlider(d); })
      .catch(function (e) { if (status) { status.textContent = "Could not load the predictions (" + e.message + "). The tables below show the committed numbers."; status.classList.add("is-error"); } });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
