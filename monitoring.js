/* credit-default-mlops: monitoring page. Pick a simulated batch to see its per-feature PSI
   (bands at 0.1 and 0.2) from the committed results/monitoring/scenarios.json. */
(function () {
  "use strict";
  var C = window.SBCharts;
  function init() {
    var data = C.data("scenarios-data");
    var sel = document.getElementById("sc-select");
    var host = document.getElementById("chart-psi");
    if (!data || !sel || !host) return;
    var byName = {};
    data.scenarios.forEach(function (s) { byName[s.scenario] = s; });
    function draw() {
      var s = byName[sel.value];
      if (!s) return;
      var rows = s.feature_drift.slice().sort(function (a, b) { return b.psi - a.psi; }).slice(0, 12);
      host.innerHTML = "";
      C.hbars(host, {
        rows: rows.map(function (r) {
          return {
            label: r.feature, value: Math.min(r.psi, 1),
            cls: r.psi_band === "major" ? "s-2" : r.psi_band === "moderate" ? "s-1" : "s-muted",
            text: r.psi.toFixed(3) + (r.drifted ? " drift" : ""),
            note: r.test + " p = " + (r.p_value < 1e-4 ? r.p_value.toExponential(1) : r.p_value.toFixed(4)) + " · band " + r.psi_band,
          };
        }),
        max: Math.max(0.3, Math.min(1, rows[0] ? rows[0].psi * 1.1 : 0.3)),
        refs: [{ value: 0.1, label: "0.1" }, { value: 0.2, label: "0.2" }],
        fmt: function (v) { return v.toFixed(3); },
        aria: "Population stability index per feature for the " + s.scenario + " batch",
      });
      var d = document.getElementById("sc-detail");
      if (d) {
        d.querySelector("[data-sc=desc]").textContent = s.description;
        d.querySelector("[data-sc=free]").textContent = s.label_free_decision.action;
        d.querySelector("[data-sc=labelled]").textContent = s.decision.action;
        d.querySelector("[data-sc=why]").textContent = s.decision.reasons.join("; ") || "nothing moved";
      }
    }
    sel.addEventListener("change", draw);
    draw();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
