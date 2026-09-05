/**
 * NTRO-SRM — Inline SVG chart toolkit
 * Hand-rolled, dependency-free scientific charts: spectral reflectance profiles,
 * metric gauges, histograms, class proportion bars, gradient legends and grouped
 * bars. Theme-aware through the shared design tokens; no charting library.
 */
(function (window, document) {
    "use strict";

    var NTRO = (window.NTRO = window.NTRO || {});

    var SVG_NS = "http://www.w3.org/2000/svg";
    var PAD = 8;          // internal padding used by every chart
    var FS_TICK = 10;     // tick labels
    var FS_AXIS = 10;     // axis captions
    var FS_SERIES = 11;   // series / legend labels
    var FS_SUB = 9;       // secondary tick line (wavelengths, ranges)
    var seq = 0;

    // Fallback series colours; these mirror the --primary-600 / --accent-* tokens
    // so an un-coloured series still lands inside the application palette.
    var PALETTE = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#0284c7", "#dc2626", "#0891b2", "#65a30d"];

    var EM_DASH = "—";

    var RATING_COLOR = {
        excellent: "var(--accent-emerald, #059669)",
        good: "var(--accent-blue, #0284c7)",
        fair: "var(--accent-amber, #d97706)"
    };

    /* ------------------------------------------------------------------ styles */

    function injectStyles() {
        if (document.querySelector('style[data-ntro-style="charts"]')) return;
        var css = [
            ".ntro-chart{display:block;width:100%;overflow:visible;",
            "font-family:var(--font-family,-apple-system,BlinkMacSystemFont,sans-serif);",
            "animation:ntro-chart-fade .18s ease-out both}",
            ".ntro-chart:focus{outline:none}",
            ".ntro-chart:focus-visible{outline:2px solid var(--border-focus,#3b82f6);outline-offset:2px;",
            "border-radius:var(--radius-sm,6px)}",
            ".ntro-chart .c-muted{fill:var(--text-muted,#64748b)}",
            ".ntro-chart .c-sec{fill:var(--text-secondary,#475569)}",
            ".ntro-chart .c-pri{fill:var(--text-primary,#0f172a)}",
            ".ntro-chart .c-mono{font-family:var(--font-mono,SFMono-Regular,Menlo,Consolas,monospace)}",
            ".ntro-chart .c-grid{fill:none;stroke:var(--border-subtle,#e2e8f0);stroke-width:1;opacity:.7}",
            ".ntro-chart .c-rule{fill:none;stroke:var(--border-medium,#cbd5e1);stroke-width:1}",
            ".ntro-chart .c-guide{fill:none;stroke:var(--border-medium,#cbd5e1);stroke-width:1;stroke-dasharray:3 3}",
            ".ntro-chart .c-panel{fill:var(--bg-panel,#ffffff);fill-opacity:.96;",
            "stroke:var(--border-medium,#cbd5e1);stroke-width:1}",
            ".ntro-chart .c-hit{fill:transparent;stroke:none}",
            ".ntro-chart .c-tip{pointer-events:none;transition:opacity .12s ease-out}",
            ".ntro-chart .c-fill{transform-box:fill-box;transform-origin:left center;",
            "animation:ntro-chart-grow .45s cubic-bezier(.4,0,.2,1) both}",
            "@keyframes ntro-chart-fade{from{opacity:0}to{opacity:1}}",
            "@keyframes ntro-chart-grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}",
            ".ntro-chart-note{display:flex;align-items:center;gap:.45rem;padding:.7rem .75rem;",
            "font-family:var(--font-family,sans-serif);font-size:.75rem;line-height:1.35;",
            "color:var(--text-muted,#64748b);background:var(--bg-subtle,#f1f5f9);",
            "border:1px dashed var(--border-subtle,#e2e8f0);border-radius:var(--radius-sm,6px)}",
            ".ntro-chart-note svg{flex:0 0 auto;opacity:.75}",
            ".ntro-chart-note.is-error{color:var(--accent-red,#dc2626);",
            "background:var(--accent-red-bg,#fef2f2);border-style:solid;",
            "border-color:var(--accent-red-border,#fecaca)}",
            "@media (prefers-reduced-motion: reduce){",
            ".ntro-chart,.ntro-chart .c-fill{animation:none!important}",
            ".ntro-chart .c-tip{transition:none!important}}"
        ].join("");
        var style = document.createElement("style");
        style.setAttribute("data-ntro-style", "charts");
        style.textContent = css;
        (document.head || document.documentElement).appendChild(style);
    }

    /* ----------------------------------------------------------------- toolkit */

    function svgEl(tag, attrs) {
        var node = document.createElementNS(SVG_NS, tag);
        if (attrs) {
            for (var k in attrs) {
                if (!Object.prototype.hasOwnProperty.call(attrs, k)) continue;
                var v = attrs[k];
                if (v === null || v === undefined) continue;
                if (k === "text") node.textContent = String(v);
                else node.setAttribute(k, String(v));
            }
        }
        return node;
    }

    function add(parent, tag, attrs) {
        return parent.appendChild(svgEl(tag, attrs));
    }

    function isNum(v) { return typeof v === "number" && isFinite(v); }
    function clamp(v, lo, hi) { return hi < lo ? lo : v < lo ? lo : v > hi ? hi : v; }
    function r2(v) { return Math.round(v * 100) / 100; }
    function uid(prefix) { seq += 1; return "ntro-" + prefix + "-" + seq; }
    function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

    function resolveContainer(c) {
        var el = typeof c === "string" ? document.querySelector(c) : c;
        return el && el.nodeType === 1 ? el : null;
    }

    /** Coerce anything plausibly numeric to a number, else null. */
    function toNum(v) {
        if (isNum(v)) return v;
        if (typeof v === "string" && v.trim() !== "" && isFinite(+v)) return +v;
        return null;
    }

    function perChar(size, kind) {
        return size * (kind === "mono" ? 0.6 : kind === "bold" ? 0.58 : 0.54);
    }

    /** Advance-width estimate — enough to size tooltip panels and wrap legends. */
    function textW(str, size, kind) {
        return String(str == null ? "" : str).length * perChar(size, kind);
    }

    function truncate(str, maxW, size, kind) {
        var s = String(str == null ? "" : str);
        if (maxW <= 0) return "";
        if (textW(s, size, kind) <= maxW) return s;
        var keep = Math.max(1, Math.floor(maxW / perChar(size, kind)) - 1);
        return s.slice(0, keep).replace(/\s+$/, "") + "…";
    }

    function round(v, d) {
        var p = Math.pow(10, clamp(d, 0, 12));
        return Math.round(v * p) / p;
    }

    /** Decimal places implied by a 1/2/5 axis step. */
    function decimalsFor(step) {
        if (!isNum(step) || step <= 0) return 0;
        return clamp(Math.ceil(-Math.log(step) / Math.LN10), 0, 6);
    }

    /**
     * Format a number. `d` pins the decimal count (locale-aware, so counts pick up
     * thousands separators); omit it and the precision follows the magnitude.
     */
    function fmt(n, d) {
        if (!isNum(n)) return EM_DASH;
        if (typeof d !== "number") {
            var a = Math.abs(n);
            if (a !== 0 && (a >= 1e6 || a < 1e-4)) return n.toExponential(2).replace("e+", "e");
            var auto = a >= 1000 ? 0 : a >= 100 ? 1 : a >= 10 ? 2 : a >= 1 ? 3 : 4;
            var s = n.toFixed(auto);
            return s.indexOf(".") < 0 ? s : s.replace(/0+$/, "").replace(/\.$/, "");
        }
        try {
            return n.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
        } catch (e) {
            return n.toFixed(d);
        }
    }

    /**
     * Round 1/2/5 axis steps spanning [min, max] in roughly `ticks` intervals.
     * Degenerate input (equal bounds, NaN, a single sample) falls back to a padded
     * range, so a flat series still gets a readable axis instead of a divide by zero.
     */
    function niceScale(min, max, ticks) {
        var lo = isNum(min) ? min : 0;
        var hi = isNum(max) ? max : 1;
        if (hi < lo) { var t = lo; lo = hi; hi = t; }
        if (hi - lo < 1e-12) {
            var pad = Math.abs(hi) > 1e-12 ? Math.abs(hi) * 0.1 : 1;
            lo -= pad;
            hi += pad;
        }
        var want = clamp(Math.round(ticks || 5), 2, 12);
        var raw = (hi - lo) / want;
        var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
        var norm = raw / mag;
        var step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
        if (!isNum(step) || step <= 0) return { min: 0, max: 1, step: 0.5, ticks: [0, 0.5, 1] };
        var dec = decimalsFor(step);
        var base = Math.floor(lo / step) * step;
        var count = clamp(Math.round((Math.ceil(hi / step) * step - base) / step), 1, 24);
        var out = [];
        for (var i = 0; i <= count; i++) out.push(round(base + i * step, dec));
        return { min: out[0], max: out[out.length - 1], step: step, ticks: out };
    }

    /** Integer-only variant for counts; avoids "0.5 pixels" on sparse histograms. */
    function countScale(maxCount) {
        var s = niceScale(0, Math.max(1, maxCount), 4);
        if (s.step >= 1) { s.min = 0; return s; }
        var hi = Math.max(1, Math.ceil(maxCount));
        var out = [];
        for (var i = 0; i <= hi; i++) out.push(i);
        return { min: 0, max: hi, step: 1, ticks: out };
    }

    function parseColor(c) {
        if (typeof c !== "string") return null;
        var m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(c.trim());
        if (m) {
            var h = m[1];
            if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
            return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
        }
        m = /^rgba?\(([^)]+)\)$/i.exec(c.trim());
        if (m) {
            var p = m[1].split(/[,\s/]+/).map(parseFloat).filter(isNum);
            if (p.length >= 3) return [p[0], p[1], p[2]];
        }
        return null;
    }

    /** Ink colour that stays legible on top of an arbitrary segment fill. */
    function readableOn(color) {
        var rgb = parseColor(color);
        if (!rgb) return "#ffffff";
        var lum = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255;
        return lum > 0.62 ? "#0f172a" : "#ffffff";
    }

    /** Content-box width of the host — used by the fixed-height strip charts. */
    function hostWidth(el, fallback) {
        var w = 0;
        try {
            var cs = window.getComputedStyle(el);
            w = el.clientWidth - (parseFloat(cs.paddingLeft) || 0) - (parseFloat(cs.paddingRight) || 0);
        } catch (e) {
            w = el.clientWidth || 0;
        }
        return Math.round(w > 60 ? w : fallback);
    }

    /* --------------------------------------------------------- frame + chrome */

    function begin(host, o) {
        clear(host);
        var svg = svgEl("svg", {
            "class": "ntro-chart ntro-chart--" + o.cls,
            viewBox: "0 0 " + o.w + " " + o.h,
            preserveAspectRatio: "xMidYMid meet",
            role: "img",
            "aria-label": o.aria || o.title
        });
        svg.style.display = "block";
        svg.style.width = "100%";
        svg.style.height = o.fixed ? o.h + "px" : "auto";
        add(svg, "title", { text: o.title });
        host.appendChild(svg);
        return svg;
    }

    function notice(container, message, kind) {
        injectStyles();
        var host = resolveContainer(container);
        if (!host) return null;
        clear(host);
        var box = document.createElement("div");
        box.className = "ntro-chart-note" + (kind === "error" ? " is-error" : "");
        if (kind === "error") box.setAttribute("role", "status");
        var icon = svgEl("svg", {
            width: 14, height: 14, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor",
            "stroke-width": 2, "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true"
        });
        add(icon, "path", {
            d: kind === "error"
                ? "M12 9v4m0 4h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"
                : "M3 3v18h18M7 14l4-4 3 3 5-6"
        });
        box.appendChild(icon);
        var span = document.createElement("span");
        span.textContent = message || "No data available";
        box.appendChild(span);
        host.appendChild(box);
        return null;
    }

    /** Never let a rendering fault take the surrounding panel down with it. */
    function guard(name, fn) {
        return function (container, config) {
            try {
                injectStyles();
                var host = resolveContainer(container);
                if (!host) return null;
                return fn(host, config && typeof config === "object" ? config : {}, container);
            } catch (err) {
                if (window.console && window.console.warn) {
                    window.console.warn("[NTRO.charts] " + name + " failed:", err);
                }
                return notice(container, "Chart could not be rendered", "error");
            }
        };
    }

    function makeY(box, s) {
        var span = (s.max - s.min) || 1;
        return function (v) { return box.y + box.h - ((v - s.min) / span) * box.h; };
    }

    /** Widest y tick label plus gutter, so the axis never collides with the plot. */
    function leftGutter(s, hasLabel) {
        var dec = decimalsFor(s.step);
        var w = 0;
        for (var i = 0; i < s.ticks.length; i++) {
            w = Math.max(w, textW(fmt(s.ticks[i], dec), FS_TICK, "mono"));
        }
        return Math.round(PAD + (hasLabel ? 13 : 0) + Math.min(w, 48) + 7);
    }

    /**
     * The one cartesian frame every chart shares: y gridlines with tick labels, the
     * x baseline and its ticks (optionally a second, smaller caption line beneath)
     * and the two axis captions.
     */
    function drawFrame(parent, o) {
        var b = o.box, s = o.scale, i, y;
        var dec = decimalsFor(s.step);
        var g = add(parent, "g", { "class": "c-frame", "aria-hidden": "true" });
        for (i = 0; i < s.ticks.length; i++) {
            y = r2(o.y(s.ticks[i]));
            add(g, "line", {
                x1: b.x, y1: y, x2: b.x + b.w, y2: y,
                "class": "c-grid", "shape-rendering": "crispEdges"
            });
            add(g, "text", {
                x: b.x - 6, y: y + 3.4, "class": "c-muted c-mono", "font-size": FS_TICK,
                "text-anchor": "end", text: fmt(s.ticks[i], dec)
            });
        }
        var baseY = b.y + b.h;
        add(g, "line", {
            x1: b.x, y1: baseY, x2: b.x + b.w, y2: baseY,
            "class": "c-rule", "shape-rendering": "crispEdges"
        });
        var xt = o.xTicks || [];
        for (i = 0; i < xt.length; i++) {
            var at = r2(xt[i].at);
            add(g, "line", {
                x1: at, y1: baseY, x2: at, y2: baseY + 3,
                "class": "c-rule", "shape-rendering": "crispEdges"
            });
            add(g, "text", {
                x: at, y: baseY + 14, "class": xt[i].strong ? "c-sec" : "c-muted",
                "font-size": FS_TICK, "font-weight": xt[i].strong ? 600 : 400,
                "text-anchor": "middle", text: xt[i].label
            });
            if (xt[i].sub) {
                add(g, "text", {
                    x: at, y: baseY + 24, "class": "c-muted c-mono", "font-size": FS_SUB,
                    "text-anchor": "middle", text: xt[i].sub
                });
            }
        }
        if (o.yLabel) {
            add(g, "text", {
                "class": "c-muted", "font-size": FS_AXIS, "font-weight": 600, "letter-spacing": ".04em",
                "text-anchor": "middle", text: truncate(o.yLabel, b.h, FS_AXIS),
                transform: "translate(" + (PAD + 4) + "," + r2(b.y + b.h / 2) + ") rotate(-90)"
            });
        }
        if (o.xLabel) {
            add(g, "text", {
                x: b.x + b.w / 2, y: o.xLabelY, "class": "c-muted", "font-size": FS_AXIS,
                "font-weight": 600, "letter-spacing": ".04em", "text-anchor": "middle",
                text: truncate(o.xLabel, b.w, FS_AXIS)
            });
        }
        return g;
    }

    /** Normalise a series list: fixed length, coerced numbers, guaranteed colour. */
    function normSeries(list, n) {
        var out = [];
        if (!Array.isArray(list)) return out;
        for (var i = 0; i < list.length; i++) {
            var s = list[i];
            if (!s || typeof s !== "object") continue;
            var src = Array.isArray(s.values) ? s.values : [];
            var vals = [];
            for (var j = 0; j < n; j++) vals.push(toNum(src[j]));
            out.push({
                label: s.label != null && String(s.label) !== "" ? String(s.label) : "Series " + (out.length + 1),
                color: typeof s.color === "string" && s.color ? s.color : PALETTE[out.length % PALETTE.length],
                dashed: !!s.dashed,
                values: vals
            });
        }
        return out;
    }

    function finiteValues(series) {
        var vals = [];
        for (var i = 0; i < series.length; i++) {
            for (var j = 0; j < series[i].values.length; j++) {
                if (isNum(series[i].values[j])) vals.push(series[i].values[j]);
            }
        }
        return vals;
    }

    /** Flow legend chips left to right, wrapping at `maxW`. Rows are 15px. */
    function layoutLegend(items, maxW, swatchW) {
        var x = 0, row = 0, gap = 14, out = [];
        for (var i = 0; i < items.length; i++) {
            var w = swatchW + 5 + textW(items[i].label, FS_SERIES) + gap;
            if (x > 0 && x + w - gap > maxW) { x = 0; row += 1; }
            out.push({ item: items[i], x: x, row: row });
            x += w;
        }
        return { placed: out, rows: items.length ? row + 1 : 0, height: items.length ? (row + 1) * 15 : 0 };
    }

    function drawLegend(parent, layout, top, asLine) {
        var g = add(parent, "g", { "class": "c-legend", "aria-hidden": "true" });
        for (var i = 0; i < layout.placed.length; i++) {
            var p = layout.placed[i], s = p.item;
            var x = PAD + p.x;
            var cy = top + p.row * 15 + 7;
            if (asLine) {
                add(g, "line", {
                    x1: x, y1: cy, x2: x + 16, y2: cy, stroke: s.color, "stroke-width": 2,
                    "stroke-linecap": "round", "stroke-dasharray": s.dashed ? "4 3" : null
                });
                add(g, "circle", {
                    cx: x + 8, cy: cy, r: 2.5, fill: s.color,
                    stroke: "var(--bg-card, #ffffff)", "stroke-width": 1
                });
                add(g, "text", { x: x + 21, y: cy + 3.7, "class": "c-sec", "font-size": FS_SERIES, text: s.label });
            } else {
                add(g, "rect", { x: x, y: cy - 4.5, width: 9, height: 9, rx: 2, fill: s.color });
                add(g, "text", { x: x + 14, y: cy + 3.7, "class": "c-sec", "font-size": FS_SERIES, text: s.label });
            }
        }
        return g;
    }

    /** Thin out category labels when the slot is narrower than a label. */
    function labelStride(n, width, approxLabelPx) {
        if (n < 2 || width <= 0) return 1;
        return Math.max(1, Math.ceil((n * approxLabelPx) / width));
    }

    /* ------------------------------------------------------- spectral profile */

    function spectralProfile(host, cfg, ref) {
        var W = 640, H = 280;          // 16:7
        var bands = Array.isArray(cfg.bands) ? cfg.bands.map(String) : [];
        var n = bands.length;
        var series = normSeries(cfg.series, n);
        if (!n || !series.length) return notice(ref, "No spectral profile to display");

        var vals = finiteValues(series);
        if (!vals.length) return notice(ref, "Spectral profile contains no finite values");

        var lo = Math.min.apply(null, vals);
        var hi = Math.max.apply(null, vals);
        var span = hi - lo;
        // A padded data range, not a zero baseline: the native/super-resolved deltas
        // this chart exists to show are far smaller than the reflectance itself.
        var headroom = span > 0 ? span * 0.08 : (Math.abs(hi) * 0.1 || 0.1);
        var yLo = lo - headroom;
        if (lo >= 0 && yLo < 0) yLo = 0;
        var scale = niceScale(yLo, hi + headroom, 5);

        var wl = Array.isArray(cfg.wavelengths) ? cfg.wavelengths : null;
        var hasWl = false;
        var i, k;
        if (wl) {
            for (i = 0; i < n; i++) { if (isNum(toNum(wl[i]))) { hasWl = true; break; } }
        }

        var legend = layoutLegend(series, W - 2 * PAD, 16);
        var top = PAD + legend.height + (legend.height ? 6 : 0);
        var catH = 3 + 12 + (hasWl ? 10 : 0);
        var xLabel = hasWl ? "Central wavelength (nm)" : null;
        var box = { x: leftGutter(scale, !!cfg.yLabel), y: top, w: 0, h: 0 };
        box.w = W - box.x - (PAD + 14);
        box.h = H - top - catH - (xLabel ? 12 : 0) - PAD;
        if (box.w < 40 || box.h < 30) return notice(ref, "Not enough room to draw the profile");

        var y = makeY(box, scale);
        var cx = function (idx) { return n > 1 ? box.x + (box.w * idx) / (n - 1) : box.x + box.w / 2; };

        var title = cfg.title ? String(cfg.title) : "Spectral reflectance profile";
        var names = [];
        for (k = 0; k < series.length; k++) names.push(series[k].label);
        var svg = begin(host, {
            cls: "spectral", w: W, h: H, title: title,
            aria: title + ": " + names.join(" versus ") + " across " + n + " Sentinel-2 bands, " +
                bands.join(", ") + ". Use the arrow keys to read individual values."
        });

        var stride = labelStride(n, box.w, 26);
        var xTicks = [];
        for (i = 0; i < n; i++) {
            if (i % stride !== 0 && i !== n - 1) continue;
            var nm = wl ? toNum(wl[i]) : null;
            xTicks.push({ at: cx(i), label: bands[i], strong: true, sub: isNum(nm) ? fmt(nm, 0) : null });
        }
        drawFrame(svg, {
            box: box, scale: scale, y: y, xTicks: xTicks,
            yLabel: cfg.yLabel ? String(cfg.yLabel) : null,
            xLabel: xLabel, xLabelY: box.y + box.h + catH + 10
        });

        var plot = add(svg, "g", { "class": "c-plot", "aria-hidden": "true" });
        for (k = 0; k < series.length; k++) {
            var s = series[k];
            var d = "", pen = false;
            for (i = 0; i < n; i++) {
                if (!isNum(s.values[i])) { pen = false; continue; }   // break the line across gaps
                d += (pen ? "L" : "M") + r2(cx(i)) + " " + r2(y(s.values[i])) + " ";
                pen = true;
            }
            if (d) {
                add(plot, "path", {
                    d: d.trim(), fill: "none", stroke: s.color, "stroke-width": 2,
                    "stroke-linejoin": "round", "stroke-linecap": "round",
                    "stroke-dasharray": s.dashed ? "5 4" : null
                });
            }
            for (i = 0; i < n; i++) {
                if (!isNum(s.values[i])) continue;
                add(plot, "circle", {
                    cx: r2(cx(i)), cy: r2(y(s.values[i])), r: 3, fill: s.color,
                    stroke: "var(--bg-card, #ffffff)", "stroke-width": 1
                });
            }
        }

        drawLegend(svg, legend, PAD, true);

        /* ---- hover / keyboard read-out ---- */
        var tip = add(svg, "g", { "class": "c-tip", opacity: 0, "aria-hidden": "true" });
        var active = -1;

        function showTip(idx) {
            if (idx === active) return;
            active = idx;
            clear(tip);
            if (idx < 0 || idx >= n) { tip.setAttribute("opacity", 0); return; }

            var gx = cx(idx);
            add(tip, "line", { x1: r2(gx), y1: box.y, x2: r2(gx), y2: box.y + box.h, "class": "c-guide" });

            var rows = [], j;
            for (j = 0; j < series.length; j++) {
                if (!isNum(series[j].values[idx])) continue;
                rows.push({
                    label: series[j].label,
                    color: series[j].color,
                    text: fmt(series[j].values[idx], 4),
                    cy: y(series[j].values[idx])
                });
            }
            for (j = 0; j < rows.length; j++) {
                add(tip, "circle", {
                    cx: r2(gx), cy: r2(rows[j].cy), r: 5.5, fill: "none",
                    stroke: rows[j].color, "stroke-width": 1.5, opacity: 0.85
                });
            }

            var nmv = wl ? toNum(wl[idx]) : null;
            var head = bands[idx] + (isNum(nmv) ? "  ·  " + fmt(nmv, 0) + " nm" : "");
            var lw = 0, vw = 0;
            for (j = 0; j < rows.length; j++) {
                lw = Math.max(lw, textW(rows[j].label, FS_TICK));
                vw = Math.max(vw, textW(rows[j].text, FS_TICK, "mono"));
            }
            var body = rows.length ? 12 + lw + 16 + vw : textW("No value at this band", FS_TICK);
            var pw = Math.min(Math.max(textW(head, 10.5, "bold"), body) + 18, W - 2 * PAD);
            var ph = 22 + (rows.length ? rows.length * 14 : 14) + 6;

            // Flip to the left of the guide once past the midpoint so the panel
            // always stays inside the viewBox.
            var px = gx > box.x + box.w / 2 ? gx - 12 - pw : gx + 12;
            px = clamp(px, PAD, W - PAD - pw);
            var py = clamp(box.y + 4, PAD, H - PAD - ph);

            add(tip, "rect", { x: r2(px), y: r2(py), width: r2(pw), height: ph, rx: 6, "class": "c-panel" });
            add(tip, "text", {
                x: r2(px + 9), y: r2(py + 14), "class": "c-pri", "font-size": 10.5,
                "font-weight": 700, "letter-spacing": ".02em", text: head
            });
            add(tip, "line", {
                x1: r2(px + 1), y1: r2(py + 21), x2: r2(px + pw - 1), y2: r2(py + 21),
                "class": "c-grid", "shape-rendering": "crispEdges"
            });
            if (!rows.length) {
                add(tip, "text", {
                    x: r2(px + 9), y: r2(py + 35), "class": "c-muted", "font-size": FS_TICK,
                    text: "No value at this band"
                });
            }
            for (j = 0; j < rows.length; j++) {
                var ry = py + 22 + j * 14 + 10;
                add(tip, "rect", {
                    x: r2(px + 9), y: r2(ry - 7), width: 8, height: 8, rx: 2, fill: rows[j].color
                });
                add(tip, "text", {
                    x: r2(px + 21), y: r2(ry), "class": "c-sec", "font-size": FS_TICK,
                    text: truncate(rows[j].label, pw - 38 - vw, FS_TICK)
                });
                add(tip, "text", {
                    x: r2(px + pw - 9), y: r2(ry), "class": "c-pri c-mono", "font-size": FS_TICK,
                    "text-anchor": "end", text: rows[j].text
                });
            }
            tip.setAttribute("opacity", 1);
        }

        function nearest(clientX) {
            var rect = svg.getBoundingClientRect();
            if (!rect.width) return -1;
            var vx = ((clientX - rect.left) / rect.width) * W;
            var best = 0, bd = Infinity;
            for (var j = 0; j < n; j++) {
                var dist = Math.abs(cx(j) - vx);
                if (dist < bd) { bd = dist; best = j; }
            }
            return best;
        }

        var hit = add(svg, "rect", {
            "class": "c-hit", x: box.x - 8, y: box.y - 4,
            width: box.w + 16, height: box.h + 20
        });
        hit.addEventListener("mousemove", function (ev) { showTip(nearest(ev.clientX)); });
        hit.addEventListener("mouseleave", function () { showTip(-1); });
        hit.addEventListener("touchstart", function (ev) {
            if (ev.touches && ev.touches.length) showTip(nearest(ev.touches[0].clientX));
        }, { passive: true });
        hit.addEventListener("touchmove", function (ev) {
            if (ev.touches && ev.touches.length) showTip(nearest(ev.touches[0].clientX));
        }, { passive: true });

        svg.setAttribute("tabindex", "0");
        svg.addEventListener("keydown", function (ev) {
            if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") {
                var step = ev.key === "ArrowRight" ? 1 : -1;
                showTip(clamp(active < 0 ? (step > 0 ? 0 : n - 1) : active + step, 0, n - 1));
                ev.preventDefault();
            } else if (ev.key === "Escape" || ev.key === "Esc") {
                showTip(-1);
            }
        });
        svg.addEventListener("blur", function () { showTip(-1); });

        return svg;
    }

    /* ------------------------------------------------------------ metric gauge */

    function metricGauge(host, cfg, ref) {
        var W = hostWidth(host, 280), H = 64;
        var value = toNum(cfg.value);
        var good = toNum(cfg.good);
        var excellent = toNum(cfg.excellent);
        var better = cfg.better === "lower" ? "lower" : "higher";
        var label = cfg.label != null ? String(cfg.label) : "Metric";
        var unit = cfg.unit != null ? String(cfg.unit) : "";

        var lo = isNum(toNum(cfg.min)) ? toNum(cfg.min) : 0;
        var hi = toNum(cfg.max);
        if (!isNum(hi) || hi <= lo) {
            var pool = [];
            if (isNum(value)) pool.push(value);
            if (isNum(good)) pool.push(good);
            if (isNum(excellent)) pool.push(excellent);
            var pmax = pool.length ? Math.max.apply(null, pool) : lo + 1;
            hi = pmax > lo ? pmax + Math.abs(pmax - lo) * 0.12 : lo + 1;
        }

        var rating = null;
        if (isNum(value)) {
            if (!isNum(good) && !isNum(excellent)) {
                rating = "good";                                   // no thresholds: stay neutral
            } else if (better === "higher") {
                rating = isNum(excellent) && value >= excellent ? "excellent"
                    : isNum(good) && value >= good ? "good" : "fair";
            } else {
                rating = isNum(excellent) && value <= excellent ? "excellent"
                    : isNum(good) && value <= good ? "good" : "fair";
            }
        }
        var color = rating ? RATING_COLOR[rating] : "var(--border-medium, #cbd5e1)";
        var pos = function (v) { return clamp((v - lo) / ((hi - lo) || 1), 0, 1) * W; };

        var shown = isNum(value) ? fmt(value) : EM_DASH;
        var aria = label + ": " + shown + (unit ? " " + unit : "") +
            (rating ? " — rated " + rating : " — no value") +
            (isNum(good) ? ", good " + (better === "higher" ? "≥ " : "≤ ") + fmt(good) : "") +
            (isNum(excellent) ? ", excellent " + (better === "higher" ? "≥ " : "≤ ") + fmt(excellent) : "");
        var svg = begin(host, { cls: "gauge", w: W, h: H, fixed: true, title: aria });

        add(svg, "text", {
            x: 0, y: 11, "class": "c-sec", "font-size": FS_SERIES, "font-weight": 700,
            "letter-spacing": ".02em", text: truncate(label, W - 62, FS_SERIES, "bold")
        });
        if (rating) {
            add(svg, "text", {
                x: W, y: 11, "font-size": 9.5, "font-weight": 700, "letter-spacing": ".07em",
                "text-anchor": "end", fill: color, text: rating.toUpperCase()
            });
        }

        var valueText = add(svg, "text", { x: 0, y: 35 });
        valueText.appendChild(svgEl("tspan", {
            "class": "c-pri c-mono", "font-size": 18, "font-weight": 600, text: shown
        }));
        if (unit) {
            valueText.appendChild(svgEl("tspan", { "class": "c-muted", "font-size": FS_SERIES, dx: 4, text: unit }));
        }

        add(svg, "rect", { x: 0, y: 43, width: W, height: 6, rx: 3, fill: "var(--bg-subtle, #f1f5f9)" });
        if (isNum(value)) {
            var fw = Math.max(pos(value), 2);
            add(svg, "rect", { x: 0, y: 43, width: r2(fw), height: 6, rx: 3, fill: color, "class": "c-fill" });
        }
        [
            { v: good, name: "good" },
            { v: excellent, name: "excellent" }
        ].forEach(function (t) {
            if (!isNum(t.v)) return;
            var tx = r2(clamp(pos(t.v), 0.5, W - 0.5));
            var tick = add(svg, "line", {
                x1: tx, y1: 40, x2: tx, y2: 52, stroke: "var(--text-muted, #64748b)",
                "stroke-width": 1, opacity: 0.65, "shape-rendering": "crispEdges"
            });
            add(tick, "title", {
                text: t.name + " " + (better === "higher" ? "≥ " : "≤ ") + fmt(t.v) + (unit ? " " + unit : "")
            });
        });

        var rangeText = fmt(lo) + " – " + fmt(hi) + (unit ? " " + unit : "");
        var rangeW = textW(rangeText, FS_SUB, "mono");
        add(svg, "text", {
            x: W, y: 62, "class": "c-muted c-mono", "font-size": FS_SUB,
            "text-anchor": "end", text: rangeText
        });
        if (cfg.description) {
            add(svg, "text", {
                x: 0, y: 62, "class": "c-muted", "font-size": FS_TICK,
                text: truncate(String(cfg.description), W - rangeW - 10, FS_TICK)
            });
        }
        return svg;
    }

    /* -------------------------------------------------------------- histogram */

    function histogram(host, cfg, ref) {
        var W = 640, H = 240;          // 16:6
        var rawEdges = Array.isArray(cfg.edges) ? cfg.edges : [];
        var rawCounts = Array.isArray(cfg.counts) ? cfg.counts : [];
        var nb = Math.min(rawCounts.length, rawEdges.length - 1);
        if (nb < 1) return notice(ref, "No histogram data to display");

        var edges = [], counts = [], i, c;
        for (i = 0; i <= nb; i++) edges.push(toNum(rawEdges[i]));
        for (i = 0; i < nb; i++) { c = toNum(rawCounts[i]); counts.push(isNum(c) && c > 0 ? c : 0); }

        var e0 = null, e1 = null;
        for (i = 0; i <= nb; i++) {
            if (!isNum(edges[i])) continue;
            if (e0 === null) e0 = edges[i];
            e1 = edges[i];
        }
        if (e0 === null || e1 === null || e1 <= e0) return notice(ref, "Histogram bin edges are not usable");

        var maxC = 0, allInt = true;
        for (i = 0; i < nb; i++) {
            if (counts[i] > maxC) maxC = counts[i];
            if (Math.abs(counts[i] - Math.round(counts[i])) > 1e-9) allInt = false;
        }
        var scale = allInt ? countScale(maxC) : niceScale(0, maxC || 1, 4);
        scale.min = 0;

        var yLabel = cfg.yLabel != null ? String(cfg.yLabel) : "Count";
        var xLabel = cfg.xLabel != null && cfg.xLabel !== "" ? String(cfg.xLabel) : null;
        var box = { x: leftGutter(scale, true), y: PAD, w: 0, h: 0 };
        box.w = W - box.x - PAD - 6;
        box.h = H - PAD - 15 - (xLabel ? 12 : 0) - PAD;
        if (box.w < 40 || box.h < 30) return notice(ref, "Not enough room to draw the histogram");

        var y = makeY(box, scale);
        var x = function (v) { return box.x + ((v - e0) / (e1 - e0)) * box.w; };

        var title = cfg.title ? String(cfg.title) : "Value distribution";
        var svg = begin(host, {
            cls: "histogram", w: W, h: H, title: title,
            aria: title + ": " + nb + " bins from " + fmt(e0) + " to " + fmt(e1) +
                ", peak count " + fmt(maxC, 0)
        });

        var ts = niceScale(e0, e1, 5);
        var xTicks = [];
        var xdec = decimalsFor(ts.step);
        for (i = 0; i < ts.ticks.length; i++) {
            var tv = ts.ticks[i];
            if (tv < e0 - 1e-9 || tv > e1 + 1e-9) continue;
            xTicks.push({ at: x(tv), label: fmt(tv, xdec) });
        }
        drawFrame(svg, {
            box: box, scale: scale, y: y, xTicks: xTicks, yLabel: yLabel,
            xLabel: xLabel, xLabelY: box.y + box.h + 27
        });

        var color = typeof cfg.color === "string" && cfg.color ? cfg.color : PALETTE[0];
        var bars = add(svg, "g", { "class": "c-bars", "aria-hidden": "true" });
        var baseY = box.y + box.h;
        for (i = 0; i < nb; i++) {
            if (!isNum(edges[i]) || !isNum(edges[i + 1])) continue;
            var x0 = x(edges[i]), x1 = x(edges[i + 1]);
            if (!(x1 > x0)) continue;
            var raw = x1 - x0;
            var gap = raw > 3 ? 1 : 0;
            var h = baseY - y(counts[i]);
            if (counts[i] > 0 && h < 1) h = 1;              // keep tiny bins visible
            var rect = add(bars, "rect", {
                x: r2(x0 + gap / 2), y: r2(baseY - h), width: r2(Math.max(raw - gap, 0.6)),
                height: r2(Math.max(h, 0)), fill: color, "fill-opacity": 0.9,
                rx: raw > 6 ? 1.5 : 0
            });
            add(rect, "title", {
                text: "[" + fmt(edges[i]) + ", " + fmt(edges[i + 1]) + ")  ·  " + fmt(counts[i], 0)
            });
        }
        return svg;
    }

    /* --------------------------------------------------------------- class bar */

    function classBar(host, cfg, ref) {
        var W = hostWidth(host, 300);
        var barH = 26;
        var raw = Array.isArray(cfg.classes) ? cfg.classes : [];
        var items = [], total = 0, i;
        for (i = 0; i < raw.length; i++) {
            var c = raw[i];
            if (!c || typeof c !== "object") continue;
            var f = toNum(c.fraction);
            f = isNum(f) && f > 0 ? f : 0;
            items.push({
                label: c.label != null && String(c.label) !== "" ? String(c.label) : "Class " + (items.length + 1),
                color: typeof c.color === "string" && c.color ? c.color : PALETTE[items.length % PALETTE.length],
                f: f
            });
            total += f;
        }
        if (!items.length) return notice(ref, "No class composition to display");
        if (total <= 0) return notice(ref, "Class composition is empty — every fraction is zero");

        var legendItems = [];
        for (i = 0; i < items.length; i++) {
            items[i].p = items[i].f / total;                 // tolerate fractions that sum to 100 or 1
            var pc = items[i].p * 100;
            items[i].pct = fmt(pc, pc >= 10 ? 0 : 1) + "%";
            legendItems.push({ label: items[i].label + "  " + items[i].pct, color: items[i].color, ref: items[i] });
        }

        var legend = layoutLegend(legendItems, W - 2 * PAD, 9);
        var H = barH + 10 + legend.height + 2;

        var title = cfg.title ? String(cfg.title) : "Class composition";
        var parts = [];
        for (i = 0; i < items.length; i++) parts.push(items[i].label + " " + items[i].pct);
        var svg = begin(host, { cls: "classbar", w: W, h: H, fixed: true, title: title, aria: title + ": " + parts.join(", ") });

        var clipId = uid("clip");
        var defs = add(svg, "defs", {});
        var clip = add(defs, "clipPath", { id: clipId });
        add(clip, "rect", { x: 0, y: 0, width: W, height: barH, rx: 4 });

        var g = add(svg, "g", { "clip-path": "url(#" + clipId + ")", "aria-hidden": "true" });
        add(g, "rect", { x: 0, y: 0, width: W, height: barH, fill: "var(--bg-subtle, #f1f5f9)" });

        var cursor = 0;
        for (i = 0; i < items.length; i++) {
            var w = items[i].p * W;
            if (w <= 0) continue;
            var seg = add(g, "rect", { x: r2(cursor), y: 0, width: r2(w), height: barH, fill: items[i].color });
            add(seg, "title", { text: items[i].label + "  ·  " + items[i].pct });
            if (i > 0 && w >= 4) {
                add(g, "line", {
                    x1: r2(cursor), y1: 0, x2: r2(cursor), y2: barH,
                    stroke: "var(--bg-panel, #ffffff)", "stroke-width": 1, opacity: 0.55,
                    "shape-rendering": "crispEdges"
                });
            }
            // Below 3% there is no room for honest inline text; the legend carries it.
            if (items[i].p >= 0.03 && w >= textW(items[i].pct, FS_TICK, "bold") + 10) {
                add(g, "text", {
                    x: r2(cursor + w / 2), y: 17, "font-size": FS_TICK, "font-weight": 700,
                    "text-anchor": "middle", fill: readableOn(items[i].color), text: items[i].pct
                });
            }
            cursor += w;
        }
        add(svg, "rect", {
            x: 0.5, y: 0.5, width: Math.max(W - 1, 0), height: barH - 1, rx: 4,
            fill: "none", stroke: "var(--border-subtle, #e2e8f0)", "stroke-width": 1
        });

        var lg = add(svg, "g", { "class": "c-legend", "aria-hidden": "true" });
        for (i = 0; i < legend.placed.length; i++) {
            var p = legend.placed[i];
            var lx = PAD + p.x;
            var cy = barH + 10 + p.row * 15 + 7;
            add(lg, "rect", { x: lx, y: cy - 4.5, width: 9, height: 9, rx: 2, fill: p.item.color });
            var t = add(lg, "text", { x: lx + 14, y: cy + 3.7, "class": "c-sec", "font-size": FS_SERIES });
            t.appendChild(svgEl("tspan", { text: p.item.ref.label }));
            t.appendChild(svgEl("tspan", { "class": "c-muted c-mono", dx: 5, text: p.item.ref.pct }));
        }
        return svg;
    }

    /* --------------------------------------------------------- gradient legend */

    function gradientLegend(host, cfg, ref) {
        var W = hostWidth(host, 260), H = 40;
        var stops = [];
        var src = Array.isArray(cfg.stops) ? cfg.stops : [];
        for (var i = 0; i < src.length; i++) {
            if (typeof src[i] === "string" && src[i]) stops.push(src[i]);
        }
        if (!stops.length) return notice(ref, "No colour ramp to display");
        if (stops.length === 1) stops.push(stops[0]);

        var min = toNum(cfg.min), max = toNum(cfg.max);
        var mid = isNum(min) && isNum(max) ? (min + max) / 2 : null;
        var unit = cfg.unit != null ? String(cfg.unit) : "";
        var label = cfg.label != null ? String(cfg.label) : "";

        var title = (label || "Colour ramp") +
            (isNum(min) && isNum(max) ? ": " + fmt(min) + " to " + fmt(max) + (unit ? " " + unit : "") : "");
        var svg = begin(host, { cls: "ramp", w: W, h: H, fixed: true, title: title });

        var gid = uid("grad");
        var defs = add(svg, "defs", {});
        var grad = add(defs, "linearGradient", { id: gid, x1: "0%", y1: "0%", x2: "100%", y2: "0%" });
        for (i = 0; i < stops.length; i++) {
            add(grad, "stop", { offset: r2((i / (stops.length - 1)) * 100) + "%", "stop-color": stops[i] });
        }

        var unitW = 0;
        if (unit) {
            unitW = textW(unit, FS_SUB, "mono") + 8;
            add(svg, "text", {
                x: W, y: 9, "class": "c-muted c-mono", "font-size": FS_SUB,
                "text-anchor": "end", text: unit
            });
        }
        if (label) {
            add(svg, "text", {
                x: 0, y: 9, "class": "c-sec", "font-size": 10.5, "font-weight": 700,
                "letter-spacing": ".02em", text: truncate(label, W - unitW, 10.5, "bold")
            });
        }

        add(svg, "rect", { x: 0, y: 14, width: W, height: 10, rx: 2, fill: "url(#" + gid + ")" });
        add(svg, "rect", {
            x: 0.5, y: 14.5, width: Math.max(W - 1, 0), height: 9, rx: 2,
            fill: "none", stroke: "var(--border-subtle, #e2e8f0)", "stroke-width": 1
        });

        [
            { at: 0.5, v: min, anchor: "start" },
            { at: W / 2, v: mid, anchor: "middle" },
            { at: W - 0.5, v: max, anchor: "end" }
        ].forEach(function (t) {
            add(svg, "line", {
                x1: r2(t.at), y1: 25, x2: r2(t.at), y2: 28, "class": "c-rule", "shape-rendering": "crispEdges"
            });
            add(svg, "text", {
                x: r2(t.at), y: 37, "class": "c-muted c-mono", "font-size": FS_SUB,
                "text-anchor": t.anchor, text: fmt(t.v)
            });
        });
        return svg;
    }

    /* ------------------------------------------------------------ grouped bars */

    function groupedBars(host, cfg, ref) {
        var W = 640, H = 320;          // 16:8
        var cats = Array.isArray(cfg.categories) ? cfg.categories.map(String) : [];
        var n = cats.length;
        var series = normSeries(cfg.series, n);
        if (!n || !series.length) return notice(ref, "No comparison data to display");

        var vals = finiteValues(series);
        if (!vals.length) return notice(ref, "Comparison contains no finite values");

        var dmin = Math.min.apply(null, vals);
        var dmax = Math.max.apply(null, vals);
        var lo = Math.min(0, dmin);
        // A tightly clustered positive range (per-band PSNR, say) collapses into a
        // flat row of full-height bars on a zero baseline, so truncate the axis.
        if (dmin > 0 && (dmax - dmin) < dmin * 0.5) lo = dmin - ((dmax - dmin) || dmin * 0.1) * 0.6;
        var top = dmax > lo ? dmax + (dmax - lo) * 0.05 : lo + 1;
        var scale = niceScale(lo, top, 5);

        var legend = layoutLegend(series, W - 2 * PAD, 9);
        var plotTop = PAD + legend.height + (legend.height ? 6 : 0);
        var box = { x: leftGutter(scale, !!cfg.yLabel), y: plotTop, w: 0, h: 0 };
        box.w = W - box.x - PAD - 6;
        box.h = H - plotTop - 15 - PAD;
        if (box.w < 40 || box.h < 30) return notice(ref, "Not enough room to draw the comparison");

        var y = makeY(box, scale);
        var title = cfg.title ? String(cfg.title) : "Per-category comparison";
        var names = [], i, k;
        for (k = 0; k < series.length; k++) names.push(series[k].label);
        var svg = begin(host, {
            cls: "grouped", w: W, h: H, title: title,
            aria: title + ": " + names.join(" versus ") + " across " + n + " categories, " + cats.join(", ")
        });

        var slot = box.w / n;
        var stride = labelStride(n, box.w, 26);
        var xTicks = [];
        for (i = 0; i < n; i++) {
            if (i % stride !== 0 && i !== n - 1) continue;
            xTicks.push({ at: box.x + slot * (i + 0.5), label: cats[i], strong: true });
        }
        drawFrame(svg, {
            box: box, scale: scale, y: y, xTicks: xTicks,
            yLabel: cfg.yLabel ? String(cfg.yLabel) : null
        });

        var base = clamp(0, scale.min, scale.max);
        var baseY = y(base);
        if (scale.min < 0 && scale.max > 0) {
            add(svg, "line", {
                x1: box.x, y1: r2(baseY), x2: box.x + box.w, y2: r2(baseY),
                "class": "c-rule", "shape-rendering": "crispEdges"
            });
        }

        var m = series.length;
        var inner = slot * (n === 1 ? 0.5 : 0.72);
        var bw = inner / m;
        var gap = bw > 4 ? 1 : 0;
        var bars = add(svg, "g", { "class": "c-bars", "aria-hidden": "true" });
        for (i = 0; i < n; i++) {
            var gx = box.x + slot * i + (slot - inner) / 2;
            for (k = 0; k < m; k++) {
                var v = series[k].values[i];
                if (!isNum(v)) continue;                        // skip null / NaN samples
                var vy = y(v);
                var y0 = Math.min(vy, baseY);
                var h = Math.abs(baseY - vy);
                if (v !== base && h < 1) { h = 1; y0 = Math.min(y0, baseY - 1); }
                var w = Math.max(bw - gap, 0.6);
                var rect = add(bars, "rect", {
                    x: r2(gx + k * bw + gap / 2), y: r2(y0), width: r2(w), height: r2(h),
                    fill: series[k].color, rx: w > 5 ? 1.5 : 0
                });
                add(rect, "title", { text: cats[i] + "  ·  " + series[k].label + ": " + fmt(v) });
            }
        }

        drawLegend(svg, legend, PAD, false);
        return svg;
    }

    /* ---------------------------------------------------------------- exports */

    NTRO.charts = {
        spectralProfile: guard("spectralProfile", spectralProfile),
        metricGauge: guard("metricGauge", metricGauge),
        histogram: guard("histogram", histogram),
        classBar: guard("classBar", classBar),
        gradientLegend: guard("gradientLegend", gradientLegend),
        groupedBars: guard("groupedBars", groupedBars)
    };
})(window, document);
