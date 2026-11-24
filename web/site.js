/* credit-default-mlops site: switch the embedded Evidently report between slices.
   Progressive enhancement: without JS the drifted report is embedded and both reports are linked. */
(function () {
  "use strict";
  function init() {
    var tabs = Array.prototype.slice.call(document.querySelectorAll('[role="tab"][data-src]'));
    var frame = document.getElementById("drift-frame");
    var open = document.getElementById("drift-open");
    var panel = document.getElementById("drift-panel");
    if (!tabs.length || !frame) return;
    function select(tab, focus) {
      tabs.forEach(function (t) {
        var on = t === tab;
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.tabIndex = on ? 0 : -1;
      });
      var src = tab.getAttribute("data-src");
      if (frame.getAttribute("src") !== src) frame.setAttribute("src", src);
      frame.title = "Evidently data-drift report: " + tab.getAttribute("data-label") + " slice";
      if (open) {
        open.href = src;
        open.firstChild.nodeValue = src + " ";
      }
      if (panel) panel.setAttribute("aria-labelledby", tab.id);
      if (focus) tab.focus();
    }
    tabs.forEach(function (t, i) {
      t.tabIndex = t.getAttribute("aria-selected") === "true" ? 0 : -1;
      t.addEventListener("click", function () { select(t, false); });
      t.addEventListener("keydown", function (e) {
        var k = e.key, n = tabs.length;
        if (k === "ArrowRight" || k === "ArrowLeft") {
          e.preventDefault();
          select(tabs[(i + (k === "ArrowRight" ? 1 : n - 1)) % n], true);
        }
      });
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
