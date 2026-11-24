/* credit-default-mlops: small SVG chart helpers (line charts with a hover crosshair, horizontal
   bars). Colours come from CSS classes mapped to the kit's chart tokens, so both themes work.
   Every chart has a server-rendered data table next to it; these charts are the visual layer. */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function ticks(min, max, n) {
    // "Nice" steps (1, 2, 2.5, 5 x 10^k) so labels read 0.10, 0.20 rather than 0.18, 0.36.
    var raw = (max - min) / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), out = [];
    var step = [1, 2, 2.5, 5, 10].map(function (k) { return k * mag; }).filter(function (v) { return v >= raw; })[0];
    for (var v = Math.ceil(min / step - 1e-9) * step; v <= max + 1e-9; v += step) out.push(Math.round(v / step) * step);
    return out;
  }
  function fmt(v, f) { return f ? f(v) : (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)); }

  function tooltip(host) {
    var t = host.querySelector(".chart-tip");
    if (!t) {
      t = document.createElement("div");
      t.className = "chart-tip";
      t.setAttribute("role", "status");
      t.setAttribute("aria-live", "polite");
      host.appendChild(t);
    }
    return t;
  }

  /* Line chart. opts: series [{label, cls, points [[x,y]], dash}], xDomain, yDomain, xLabel,
     yLabel, xFmt, yFmt, markers [{x, y, label, cls}], vlines [{x, label, cls}], diagonal,
     height, xTicks, yTicks. */
  function line(host, opts) {
    function draw() {
      var w = Math.max(280, host.clientWidth || 560);
      var h = opts.height || 300;
      var m = { t: opts.yLabel ? 26 : 14, r: 16, b: 40, l: 52 };
      var iw = w - m.l - m.r, ih = h - m.t - m.b;
      var xd = opts.xDomain || [0, 1], yd = opts.yDomain || [0, 1];
      var X = function (v) { return m.l + ((v - xd[0]) / (xd[1] - xd[0])) * iw; };
      var Y = function (v) { return m.t + ih - ((v - yd[0]) / (yd[1] - yd[0])) * ih; };
      var old = host.querySelector("svg");
      if (old) old.remove();
      var svg = el("svg", { viewBox: "0 0 " + w + " " + h, width: "100%", height: h, class: "chart",
        role: "img", "aria-label": opts.aria || "chart" });
      host.insertBefore(svg, host.firstChild);
      var g = el("g", { class: "chart-grid" }, svg);
      ticks(yd[0], yd[1], opts.yTicks || 4).forEach(function (v) {
        el("line", { x1: m.l, x2: m.l + iw, y1: Y(v), y2: Y(v) }, g);
        var t = el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end", class: "chart-tick" }, svg);
        t.textContent = fmt(v, opts.yFmt);
      });
      ticks(xd[0], xd[1], opts.xTicks || 5).forEach(function (v) {
        var t = el("text", { x: X(v), y: m.t + ih + 16, "text-anchor": "middle", class: "chart-tick" }, svg);
        t.textContent = fmt(v, opts.xFmt);
      });
      el("line", { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih, class: "chart-axis" }, svg);
      if (opts.xLabel) {
        var xl = el("text", { x: m.l + iw / 2, y: h - 4, "text-anchor": "middle", class: "chart-label" }, svg);
        xl.textContent = opts.xLabel;
      }
      if (opts.yLabel) {
        var yl = el("text", { x: m.l - 8, y: 11, "text-anchor": "start", class: "chart-label" }, svg);
        yl.textContent = "↑ " + opts.yLabel;
      }
      if (opts.diagonal) {
        el("line", { x1: X(xd[0]), y1: Y(yd[0]), x2: X(xd[1]), y2: Y(yd[1]), class: "chart-ref" }, svg);
      }
      (opts.vlines || []).forEach(function (v) {
        el("line", { x1: X(v.x), x2: X(v.x), y1: m.t, y2: m.t + ih, class: "chart-vline " + (v.cls || "") }, svg);
        if (v.label) {
          var vt = el("text", { x: X(v.x) + 4, y: m.t + 10, class: "chart-tick chart-vlabel" }, svg);
          vt.textContent = v.label;
        }
      });
      opts.series.forEach(function (s) {
        var d = s.points.map(function (p, i) { return (i ? "L" : "M") + X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1); }).join("");
        el("path", { d: d, class: "chart-line " + (s.cls || "s-1"), "stroke-dasharray": s.dash ? "5 4" : null }, svg);
      });
      (opts.markers || []).forEach(function (mk) {
        el("circle", { cx: X(mk.x), cy: Y(mk.y), r: 5, class: "chart-dot " + (mk.cls || "s-1") }, svg);
      });
      // hover layer
      var cross = el("line", { y1: m.t, y2: m.t + ih, class: "chart-cross", visibility: "hidden" }, svg);
      var dots = opts.series.map(function (s) {
        return el("circle", { r: 4, class: "chart-dot " + (s.cls || "s-1"), visibility: "hidden" }, svg);
      });
      var hit = el("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent", class: "chart-hit" }, svg);
      var tip = tooltip(host);
      function nearest(pts, x) {
        var lo = 0, hi = pts.length - 1;
        while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (pts[mid][0] < x) lo = mid; else hi = mid; }
        return Math.abs(pts[lo][0] - x) <= Math.abs(pts[hi][0] - x) ? pts[lo] : pts[hi];
      }
      function move(evt) {
        var r = svg.getBoundingClientRect();
        var px = ((evt.clientX - r.left) / r.width) * w;
        var xv = xd[0] + ((px - m.l) / iw) * (xd[1] - xd[0]);
        xv = Math.min(xd[1], Math.max(xd[0], xv));
        cross.setAttribute("x1", X(xv)); cross.setAttribute("x2", X(xv));
        cross.setAttribute("visibility", "visible");
        var rows = [];
        opts.series.forEach(function (s, i) {
          var p = nearest(s.points, xv);
          dots[i].setAttribute("cx", X(p[0])); dots[i].setAttribute("cy", Y(p[1]));
          dots[i].setAttribute("visibility", "visible");
          rows.push('<span><i class="sw ' + (s.cls || "s-1") + '"></i>' + s.label + ": <b>" + fmt(p[1], opts.yFmt) + "</b></span>");
        });
        tip.innerHTML = "<span>" + (opts.xLabel || "x") + " " + fmt(xv, opts.xFmt) + "</span>" + rows.join("");
        tip.style.left = Math.min(r.width - 170, Math.max(0, ((X(xv) / w) * r.width) + 12)) + "px";
        tip.classList.add("is-on");
      }
      function leave() {
        cross.setAttribute("visibility", "hidden");
        dots.forEach(function (d) { d.setAttribute("visibility", "hidden"); });
        tip.classList.remove("is-on");
      }
      hit.addEventListener("pointermove", move);
      hit.addEventListener("pointerleave", leave);
    }
    draw();
    if (window.ResizeObserver) {
      var last = host.clientWidth;
      new ResizeObserver(function () {
        if (Math.abs(host.clientWidth - last) > 8) { last = host.clientWidth; draw(); }
      }).observe(host);
    }
    return { redraw: draw };
  }

  /* Horizontal bars. opts: rows [{label, value, cls, text}], max, refs [{value, label}], fmt */
  function hbars(host, opts) {
    function draw() {
      var w = Math.max(280, host.clientWidth || 560);
      var rowH = 26, m = { t: 8, r: 64, b: 24, l: Math.min(150, w * 0.34) };
      var h = m.t + m.b + rowH * opts.rows.length;
      var iw = w - m.l - m.r;
      var max = opts.max || Math.max.apply(null, opts.rows.map(function (r) { return r.value; }).concat([1e-9]));
      var X = function (v) { return m.l + Math.max(0, Math.min(1, v / max)) * iw; };
      var old = host.querySelector("svg");
      if (old) old.remove();
      var svg = el("svg", { viewBox: "0 0 " + w + " " + h, width: "100%", height: h, class: "chart",
        role: "img", "aria-label": opts.aria || "bar chart" });
      host.insertBefore(svg, host.firstChild);
      var tip = tooltip(host);
      opts.rows.forEach(function (r, i) {
        var y = m.t + i * rowH;
        var lab = el("text", { x: m.l - 8, y: y + rowH / 2 + 4, "text-anchor": "end", class: "chart-tick chart-rowlabel" }, svg);
        lab.textContent = r.label;
        var bw = Math.max(2, X(r.value) - m.l);
        var bar = el("rect", { x: m.l, y: y + 5, width: bw, height: rowH - 10, rx: 3, class: "chart-bar " + (r.cls || "s-1") }, svg);
        var val = el("text", { x: m.l + bw + 6, y: y + rowH / 2 + 4, class: "chart-tick chart-val" }, svg);
        val.textContent = r.text || fmt(r.value, opts.fmt);
        var hitr = el("rect", { x: 0, y: y, width: w, height: rowH, fill: "transparent" }, svg);
        hitr.addEventListener("pointerenter", function () {
          tip.innerHTML = "<span>" + r.label + "</span><b>" + (r.text || fmt(r.value, opts.fmt)) + "</b>" + (r.note ? "<span>" + r.note + "</span>" : "");
          tip.style.left = Math.min(w - 180, m.l + bw / 2) / w * 100 + "%";
          tip.style.top = (y + rowH) + "px";
          tip.classList.add("is-on");
          bar.classList.add("is-hover");
        });
        hitr.addEventListener("pointerleave", function () { tip.classList.remove("is-on"); bar.classList.remove("is-hover"); });
      });
      (opts.refs || []).forEach(function (ref) {
        if (ref.value > max) return;
        el("line", { x1: X(ref.value), x2: X(ref.value), y1: m.t, y2: h - m.b + 4, class: "chart-ref" }, svg);
        var t = el("text", { x: X(ref.value), y: h - 6, "text-anchor": "middle", class: "chart-tick" }, svg);
        t.textContent = ref.label;
      });
    }
    draw();
    if (window.ResizeObserver) {
      var last = host.clientWidth;
      new ResizeObserver(function () {
        if (Math.abs(host.clientWidth - last) > 8) { last = host.clientWidth; draw(); }
      }).observe(host);
    }
    return { redraw: draw };
  }

  function data(id) {
    var n = document.getElementById(id);
    if (!n) return null;
    try { return JSON.parse(n.textContent); } catch (e) { return null; }
  }

  window.SBCharts = { line: line, hbars: hbars, data: data };
})();
