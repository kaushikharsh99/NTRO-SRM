/**
 * NTRO-SRM — Pixel Inspector
 * Renders the spectral / thematic probe panel for one clicked pixel of a super-resolved patch.
 */
(function (window, document) {
    "use strict";

    var NTRO = window.NTRO = window.NTRO || {};

    var EM_DASH = "—";
    var DEFAULT_EMPTY_MSG = "Click anywhere on the super-resolved patch to inspect its spectral signature.";

    /* Sentinel-2 L2A central wavelengths (nm) in the canonical NTRO-SRM band order.
       Used whenever the payload omits `wavelengths_nm`. */
    var CANONICAL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"];
    var CANONICAL_NM = [492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 1613.7, 2202.4];

    var MAX_BANDS = 64; // sanity ceiling so a malformed payload cannot build a runaway table

    var RISK = {
        low:      { label: "Low risk",      fg: "var(--accent-emerald, #059669)", bg: "var(--accent-emerald-bg, #ecfdf5)" },
        moderate: { label: "Moderate risk", fg: "var(--accent-blue, #0284c7)",    bg: "var(--accent-blue-bg, #f0f9ff)" },
        elevated: { label: "Elevated risk", fg: "var(--accent-amber, #d97706)",   bg: "var(--accent-amber-bg, #fffbeb)" },
        high:     { label: "High risk",     fg: "var(--accent-red, #dc2626)",     bg: "var(--accent-red-bg, #fef2f2)" }
    };

    var SVG_OPEN = '<svg width="W" height="W" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">';
    function svgIcon(body, size) { return SVG_OPEN.replace(/W/g, String(size || 13)) + body + "</svg>"; }

    var TRIANGLE = '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>' +
        '<line x1="12" y1="9" x2="12" y2="13.5"/><line x1="12" y1="17" x2="12.01" y2="17"/>';

    var IC_COPY = svgIcon('<rect x="9" y="9" width="12" height="12" rx="2"/>' +
        '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>');
    var IC_CHECK = svgIcon('<polyline points="20 6 9 17 4 12"/>');
    var IC_ALERT = svgIcon(TRIANGLE);
    var IC_BROKEN = svgIcon(TRIANGLE, 28);
    var IC_TARGET = svgIcon('<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="2.4"/>' +
        '<line x1="12" y1="1" x2="12" y2="4.6"/><line x1="12" y1="19.4" x2="12" y2="23"/>' +
        '<line x1="1" y1="12" x2="4.6" y2="12"/><line x1="19.4" y1="12" x2="23" y2="12"/>', 30);

    /* Copy-feedback timers, keyed by the panel container so clear() can cancel them. */
    var timers = new WeakMap();

    /* ---------------------------------------------------------------- styling */

    function injectStyles() {
        if (document.querySelector('style[data-ntro-style="inspector"]')) return;
        var s = document.createElement("style");
        s.setAttribute("data-ntro-style", "inspector");
        s.textContent = [
            '.ntro-insp{font-family:var(--font-family,system-ui,sans-serif);color:var(--text-primary,#0f172a);}',
            '.ntro-insp-sr-only{position:absolute;width:1px;height:1px;margin:-1px;padding:0;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap;border:0;}',

            '.ntro-insp-head{display:flex;align-items:flex-start;justify-content:space-between;gap:.5rem;padding:.6rem .7rem;background:var(--bg-subtle,#f1f5f9);border:1px solid var(--border-subtle,#e2e8f0);border-radius:var(--radius-sm,6px);}',
            '.ntro-insp-coord{min-width:0;}',
            '.ntro-insp-latlon{font-family:var(--font-mono,monospace);font-size:12.5px;font-weight:600;letter-spacing:-.2px;color:var(--text-primary,#0f172a);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}',
            '.ntro-insp-datum{margin-left:.35rem;font-family:var(--font-family,sans-serif);font-size:9px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;color:var(--text-muted,#64748b);}',
            '.ntro-insp-sub{margin-top:.18rem;font-family:var(--font-mono,monospace);font-size:10.5px;line-height:1.5;color:var(--text-muted,#64748b);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}',
            '.ntro-insp-copy{flex:0 0 auto;display:inline-flex;align-items:center;gap:.3rem;padding:.25rem .5rem;font-family:var(--font-family,sans-serif);font-size:10.5px;font-weight:600;cursor:pointer;color:var(--text-secondary,#475569);background:var(--bg-panel,#fff);border:1px solid var(--border-medium,#cbd5e1);border-radius:var(--radius-sm,6px);transition:background-color .15s ease,color .15s ease,border-color .15s ease;}',
            '.ntro-insp-copy:hover:not(:disabled){background:var(--bg-hover,#e2e8f0);color:var(--text-primary,#0f172a);}',
            '.ntro-insp-copy:disabled{opacity:.45;cursor:default;}',
            '.ntro-insp-copy:focus-visible{outline:2px solid var(--border-focus,#3b82f6);outline-offset:2px;}',
            '.ntro-insp-copy.is-ok{color:var(--accent-emerald,#059669);border-color:var(--accent-emerald,#059669);}',
            '.ntro-insp-copy.is-err{color:var(--accent-red,#dc2626);border-color:var(--accent-red,#dc2626);}',
            '.ntro-insp-ico{display:inline-flex;}',

            '.ntro-insp-sect{margin-top:.85rem;}',
            '.ntro-insp-sect-head{display:flex;align-items:baseline;justify-content:space-between;gap:.5rem;margin-bottom:.4rem;}',
            '.ntro-insp-sect-title{margin:0;font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.45px;color:var(--text-muted,#64748b);}',
            '.ntro-insp-hint{font-family:var(--font-mono,monospace);font-size:10px;color:var(--text-muted,#64748b);opacity:.85;white-space:nowrap;}',
            '.ntro-insp-chart{min-height:120px;}',
            '.ntro-insp-note{padding:.45rem .55rem;font-size:11px;line-height:1.5;color:var(--text-muted,#64748b);background:var(--bg-subtle,#f1f5f9);border:1px dashed var(--border-subtle,#e2e8f0);border-radius:var(--radius-sm,6px);}',
            '.ntro-insp-note.is-warn{margin-top:.5rem;color:var(--text-secondary,#475569);background:var(--accent-amber-bg,#fffbeb);border:1px solid var(--accent-amber-border,#fde68a);}',

            '.ntro-insp-table{width:100%;border-collapse:collapse;font-size:11.5px;}',
            '.ntro-insp-table th{padding:.28rem .3rem;font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;color:var(--text-muted,#64748b);text-align:left;white-space:nowrap;border-bottom:1px solid var(--border-medium,#cbd5e1);}',
            '.ntro-insp-table td{padding:.32rem .3rem;vertical-align:top;border-bottom:1px dashed var(--border-subtle,#e2e8f0);}',
            '.ntro-insp-table tbody tr:last-child td{border-bottom:0;}',
            '.ntro-insp-num,.ntro-insp-table th.ntro-insp-num{text-align:right;font-family:var(--font-mono,monospace);font-variant-numeric:tabular-nums;white-space:nowrap;}',
            '.ntro-insp-table td.ntro-insp-num{color:var(--text-secondary,#475569);}',
            '.ntro-insp-table td.ntro-insp-num.is-sr{color:var(--primary-600,#2563eb);font-weight:600;}',
            '.ntro-insp-band{font-family:var(--font-mono,monospace);font-size:11px;font-weight:600;color:var(--text-primary,#0f172a);}',
            '.ntro-insp-idx-name{font-size:11.5px;font-weight:600;line-height:1.3;color:var(--text-primary,#0f172a);}',
            '.ntro-insp-idx-key{display:block;font-family:var(--font-mono,monospace);font-size:9.5px;letter-spacing:.3px;text-transform:uppercase;color:var(--text-muted,#64748b);}',
            '.ntro-insp-delta{display:block;font-family:var(--font-mono,monospace);font-size:9.5px;font-weight:400;color:var(--text-muted,#64748b);}',
            '.ntro-insp-pill{display:inline-block;max-width:100%;padding:.1rem .45rem;border-radius:var(--radius-full,9999px);font-size:10px;font-weight:700;letter-spacing:.2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;background:var(--bg-subtle,#f1f5f9);color:var(--text-muted,#64748b);border:1px solid var(--border-subtle,#e2e8f0);}',

            '.ntro-insp-details{margin-top:.5rem;}',
            '.ntro-insp-details>summary{display:inline-flex;align-items:center;gap:.3rem;padding:.2rem .1rem;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;color:var(--text-muted,#64748b);cursor:pointer;list-style:none;}',
            '.ntro-insp-details>summary::-webkit-details-marker{display:none;}',
            '.ntro-insp-details>summary::before{content:"\\25B8";display:inline-block;transition:transform .15s ease;}',
            '.ntro-insp-details[open]>summary::before{transform:rotate(90deg);}',
            '.ntro-insp-details>summary:hover{color:var(--text-secondary,#475569);}',
            '.ntro-insp-details>summary:focus-visible{outline:2px solid var(--border-focus,#3b82f6);outline-offset:2px;border-radius:3px;}',

            '.ntro-insp-conf{margin-top:.85rem;padding:.65rem .7rem;background:var(--bg-card,#fff);border:1px solid var(--border-subtle,#e2e8f0);border-radius:var(--radius-sm,6px);}',
            '.ntro-insp-conf-top{display:flex;align-items:center;justify-content:space-between;gap:.5rem;}',
            '.ntro-insp-conf-label{font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.45px;color:var(--text-muted,#64748b);}',
            '.ntro-insp-risk{padding:.1rem .4rem;border-radius:var(--radius-full,9999px);font-size:9.5px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;white-space:nowrap;}',
            '.ntro-insp-conf-val{display:flex;align-items:baseline;gap:.1rem;margin:.28rem 0 .4rem;}',
            '.ntro-insp-conf-num{font-family:var(--font-mono,monospace);font-size:18px;font-weight:600;line-height:1.1;font-variant-numeric:tabular-nums;color:var(--text-primary,#0f172a);}',
            '.ntro-insp-conf-pct{font-family:var(--font-mono,monospace);font-size:11px;color:var(--text-muted,#64748b);}',
            '.ntro-insp-bar{height:6px;border-radius:3px;background:var(--bg-hover,#e2e8f0);overflow:hidden;}',
            '.ntro-insp-bar>i{display:block;height:100%;width:0;border-radius:3px;transition:width .45s cubic-bezier(.4,0,.2,1);}',
            '.ntro-insp-conf-meta{margin-top:.4rem;font-size:11px;line-height:1.45;color:var(--text-muted,#64748b);}',
            '.ntro-insp-conf-meta b{font-family:var(--font-mono,monospace);font-weight:600;font-variant-numeric:tabular-nums;color:var(--text-secondary,#475569);}',
            '.ntro-insp-caution{margin-top:.5rem;padding:.4rem .55rem;font-size:11px;line-height:1.45;color:var(--text-secondary,#475569);border-left:2px solid;border-radius:0 var(--radius-sm,6px) var(--radius-sm,6px) 0;}',

            '.ntro-insp-empty{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:.55rem;padding:1.6rem 1rem;text-align:center;color:var(--text-muted,#64748b);}',
            '.ntro-insp-empty .ntro-insp-ico{opacity:.5;}',
            '.ntro-insp-empty p{max-width:250px;margin:0;font-size:11.5px;line-height:1.55;}',
            '.ntro-insp-empty.is-error .ntro-insp-ico{opacity:1;color:var(--accent-red,#dc2626);}',

            '@media (prefers-reduced-motion: reduce){',
            '.ntro-insp *,.ntro-insp *::before{transition:none!important;animation:none!important;}',
            '}'
        ].join("\n");
        (document.head || document.documentElement).appendChild(s);
    }

    /* ---------------------------------------------------------------- helpers */

    function num(v) { return typeof v === "number" && isFinite(v); }
    function fmt(v, dp) { return num(v) ? v.toFixed(dp) : EM_DASH; }
    function str(v) { return typeof v === "string" && v.trim() ? v.trim() : ""; }

    function el(tag, cls, text) {
        var n = document.createElement(tag);
        if (cls) n.className = cls;
        if (text != null) n.textContent = String(text);
        return n;
    }

    // `markup` is always a module-owned literal, never payload-derived.
    function icon(markup) {
        var n = el("span", "ntro-insp-ico");
        n.innerHTML = markup;
        return n;
    }

    function reducedMotion() {
        try { return window.matchMedia("(prefers-reduced-motion: reduce)").matches; } catch (e) { return false; }
    }

    // Resolve a token to a literal colour: series colours cross a module boundary as plain
    // data, and charts.js may apply them as SVG attributes, where var() would be inert.
    function cssVar(name, fallback) {
        try {
            var v = window.getComputedStyle(document.documentElement).getPropertyValue(name);
            return (v || "").trim() || fallback;
        } catch (e) { return fallback; }
    }

    function hexToRgba(hex, alpha) {
        var h = str(hex).replace(/^#/, "");
        if (h.length === 3) h = h.charAt(0) + h.charAt(0) + h.charAt(1) + h.charAt(1) + h.charAt(2) + h.charAt(2);
        if (!/^[0-9a-fA-F]{6}$/.test(h)) return null;
        var n = parseInt(h, 16);
        return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + alpha + ")";
    }

    function fmtMeters(v) {
        if (!num(v)) return EM_DASH;
        var parts = Math.abs(v).toFixed(1).split(".");
        parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, " ");
        return (v < 0 ? "-" : "") + parts.join(".");
    }

    function fmtIndex(v) {
        if (!num(v)) return EM_DASH;
        var a = Math.abs(v);
        return (a !== 0 && (a >= 1e6 || a < 1e-4)) ? v.toExponential(2) : v.toFixed(4);
    }

    function fmtDelta(sr, lr) {
        if (!num(sr) || !num(lr)) return null;
        var d = sr - lr;
        if (Math.abs(d) < 5e-5) return null; // below the displayed precision
        return (d > 0 ? "+" : "−") + Math.abs(d).toFixed(4);
    }

    /* ------------------------------------------------------- payload coercion */

    // Widest of the labels and every reflectance array, so a server that disagrees with
    // itself about band count still shows every measured value (and every label).
    function bandCount(p) {
        var n = Array.isArray(p.band_names) ? p.band_names.length : 0;
        var sources = [p.sr, p.lr, p.bicubic], i, r;
        for (i = 0; i < sources.length; i++) {
            r = sources[i] && sources[i].reflectance;
            if (Array.isArray(r)) n = Math.max(n, r.length);
        }
        return Math.min(n || CANONICAL_BANDS.length, MAX_BANDS);
    }

    function bandNames(p, n) {
        var names = Array.isArray(p.band_names) ? p.band_names : [], out = [], i;
        for (i = 0; i < n; i++) out.push(str(names[i]) || CANONICAL_BANDS[i] || "B" + (i + 1));
        return out;
    }

    function wavelengths(p, n) {
        var src = Array.isArray(p.wavelengths_nm) ? p.wavelengths_nm : [], out = [], any = false, i, v;
        for (i = 0; i < n; i++) {
            v = num(src[i]) ? src[i] : (num(CANONICAL_NM[i]) ? CANONICAL_NM[i] : null);
            if (v !== null) any = true;
            out.push(v);
        }
        return any ? out : null;
    }

    // Pads / truncates to n bands and replaces every non-finite entry with null,
    // so a partially corrupt array still plots the samples that are real.
    function coerceValues(arr, n) {
        if (!Array.isArray(arr)) return null;
        var out = [], finite = 0, i, v;
        for (i = 0; i < n; i++) {
            v = arr[i];
            if (num(v)) { out.push(v); finite++; } else { out.push(null); }
        }
        return finite ? out : null;
    }

    function buildSeries(p, n) {
        var defs = [
            { role: "lr", src: p.lr, label: "Native 10 m", short: "10 m", color: cssVar("--text-muted", "#64748b") },
            { role: "sr", src: p.sr, label: "Super-resolved 2.5 m", short: "2.5 m", color: cssVar("--primary-600", "#2563eb") },
            { role: "bicubic", src: p.bicubic, label: "Bicubic 2.5 m", short: "Bicubic", color: cssVar("--accent-amber", "#d97706"), dashed: true }
        ];
        var out = [];
        defs.forEach(function (d) {
            var values = coerceValues(d.src && d.src.reflectance, n);
            if (!values) return;
            out.push({
                role: d.role, label: d.label, short: d.short,
                color: d.color, values: values, dashed: !!d.dashed
            });
        });
        return out;
    }

    /* ---------------------------------------------------------------- header */

    function copyText(text) {
        var legacy = function () {
            return new Promise(function (resolve, reject) {
                try {
                    var ta = el("textarea");
                    ta.value = text;
                    ta.setAttribute("readonly", "");
                    ta.style.cssText = "position:fixed;top:-9999px;opacity:0;";
                    document.body.appendChild(ta);
                    ta.select();
                    var ok = document.execCommand("copy");
                    document.body.removeChild(ta);
                    if (ok) { resolve(); } else { reject(new Error("copy rejected")); }
                } catch (e) { reject(e); }
            });
        };
        var nav = window.navigator;
        if (nav && nav.clipboard && nav.clipboard.writeText) {
            // Rejects outside a secure context; fall through to the selection hack.
            return nav.clipboard.writeText(text).catch(legacy);
        }
        return legacy();
    }

    function flashCopy(btn, status, container, ok) {
        var label = btn.querySelector(".ntro-insp-copy-label");
        var ico = btn.querySelector(".ntro-insp-ico");
        btn.classList.remove("is-ok", "is-err");
        btn.classList.add(ok ? "is-ok" : "is-err");
        if (ico) ico.innerHTML = ok ? IC_CHECK : IC_ALERT;
        if (label) label.textContent = ok ? "Copied" : "Failed";
        status.textContent = ok ? "Coordinates copied to clipboard" : "Could not copy coordinates";

        var prev = timers.get(container);
        if (prev) clearTimeout(prev);
        timers.set(container, setTimeout(function () {
            timers.delete(container);
            if (!btn.isConnected) return;
            btn.classList.remove("is-ok", "is-err");
            if (ico) ico.innerHTML = IC_COPY;
            if (label) label.textContent = "Copy";
            status.textContent = "";
        }, 1800));
    }

    function buildHeader(p, container) {
        var head = el("div", "ntro-insp-head");
        var col = el("div", "ntro-insp-coord");

        var hasLatLon = num(p.lat) && num(p.lon);
        var latlon = hasLatLon ? p.lat.toFixed(5) + ", " + p.lon.toFixed(5) : EM_DASH + ", " + EM_DASH;
        var line = el("div", "ntro-insp-latlon", latlon);
        line.appendChild(el("span", "ntro-insp-datum", "WGS84"));
        line.title = latlon;
        col.appendChild(line);

        var crs = str(p.crs);
        if (num(p.easting) || num(p.northing) || crs) {
            var proj = el("div", "ntro-insp-sub",
                "E " + fmtMeters(p.easting) + "   N " + fmtMeters(p.northing) + (crs ? "   " + crs : ""));
            proj.title = proj.textContent;
            col.appendChild(proj);
        }

        var grid = [];
        if (num(p.row) && num(p.col)) grid.push("LR r" + Math.round(p.row) + " c" + Math.round(p.col));
        if (p.sr && num(p.sr.row) && num(p.sr.col)) grid.push("SR r" + Math.round(p.sr.row) + " c" + Math.round(p.sr.col));
        if (grid.length) col.appendChild(el("div", "ntro-insp-sub", grid.join("   ·   ")));
        head.appendChild(col);

        var btn = el("button", "ntro-insp-copy");
        btn.type = "button";
        btn.title = "Copy latitude, longitude";
        btn.setAttribute("aria-label", "Copy coordinates to clipboard");
        btn.appendChild(icon(IC_COPY));
        btn.appendChild(el("span", "ntro-insp-copy-label", "Copy"));

        var status = el("span", "ntro-insp-sr-only");
        status.setAttribute("role", "status");
        status.setAttribute("aria-live", "polite");

        // Copy exactly what the header shows: 5 dp is ~1.1 m, well inside the 2.5 m pixel.
        var plain = hasLatLon ? p.lat.toFixed(5) + ", " + p.lon.toFixed(5) : "";
        btn.disabled = !plain;
        btn.addEventListener("click", function () {
            copyText(plain).then(
                function () { flashCopy(btn, status, container, true); },
                function () { flashCopy(btn, status, container, false); }
            );
        });

        head.appendChild(btn);
        head.appendChild(status);
        return head;
    }

    /* -------------------------------------------------------------- sections */

    function sectionHead(title, hint) {
        var h = el("div", "ntro-insp-sect-head");
        h.appendChild(el("h4", "ntro-insp-sect-title", title));
        if (hint) h.appendChild(el("span", "ntro-insp-hint", hint));
        return h;
    }

    function bandTable(bands, wl, series) {
        var table = el("table", "ntro-insp-table");
        table.appendChild(el("caption", "ntro-insp-sr-only", "Surface reflectance per Sentinel-2 band"));

        var hr = el("tr");
        hr.appendChild(el("th", null, "Band"));
        hr.appendChild(el("th", "ntro-insp-num", "λ nm"));
        series.forEach(function (s) { hr.appendChild(el("th", "ntro-insp-num", s.short)); });
        for (var i = 0; i < hr.children.length; i++) hr.children[i].setAttribute("scope", "col");
        var thead = el("thead");
        thead.appendChild(hr);
        table.appendChild(thead);

        var tbody = el("tbody");
        bands.forEach(function (name, i) {
            var tr = el("tr");
            var th = el("th", "ntro-insp-band", name);
            th.setAttribute("scope", "row");
            tr.appendChild(th);
            tr.appendChild(el("td", "ntro-insp-num", wl && num(wl[i]) ? wl[i].toFixed(1) : EM_DASH));
            series.forEach(function (s) {
                tr.appendChild(el("td", "ntro-insp-num" + (s.role === "sr" ? " is-sr" : ""), fmt(s.values[i], 4)));
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        return table;
    }

    function buildSpectral(p, bands, wl, series) {
        var finiteWl = (wl || []).filter(num);
        var hint = bands.length + " bands";
        if (finiteWl.length) {
            hint += "  ·  " + Math.round(Math.min.apply(null, finiteWl)) + "–" +
                Math.round(Math.max.apply(null, finiteWl)) + " nm";
        }

        var sect = el("section", "ntro-insp-sect");
        sect.appendChild(sectionHead("Surface reflectance", hint));

        if (!series.length) {
            sect.appendChild(el("div", "ntro-insp-note", "No reflectance values were returned for this pixel."));
            return sect;
        }

        var host = el("div", "ntro-insp-chart");
        sect.appendChild(host);

        var drawn = false;
        if (NTRO.charts && typeof NTRO.charts.spectralProfile === "function") {
            try {
                drawn = !!NTRO.charts.spectralProfile(host, {
                    bands: bands,
                    series: series,
                    yLabel: "Reflectance",
                    title: "Surface reflectance per Sentinel-2 band",
                    wavelengths: wl
                });
            } catch (e) { drawn = false; }
        }

        if (!drawn) {
            host.textContent = "";
            host.appendChild(bandTable(bands, wl, series));
        } else {
            // The chart carries the shape; the exact numbers stay one disclosure away.
            var det = el("details", "ntro-insp-details");
            det.appendChild(el("summary", null, "Band values"));
            det.appendChild(bandTable(bands, wl, series));
            sect.appendChild(det);
        }

        if (!p.sr || !Array.isArray(p.sr.reflectance)) {
            sect.appendChild(el("div", "ntro-insp-note is-warn",
                "Super-resolved reflectance is missing from this response — only the reference values are shown."));
        }
        return sect;
    }

    function classPill(label, color) {
        var pill = el("span", "ntro-insp-pill", label);
        pill.title = label;
        var bg = hexToRgba(color, 0.12);
        if (bg) {
            pill.style.backgroundColor = bg;
            pill.style.color = str(color);
            pill.style.borderColor = hexToRgba(color, 0.28);
        }
        return pill;
    }

    function buildIndices(p) {
        var rows = (Array.isArray(p.indices) ? p.indices : []).filter(function (it) {
            return it && typeof it === "object";
        });

        var sect = el("section", "ntro-insp-sect");
        sect.appendChild(sectionHead("Spectral indices", rows.length ? rows.length + " computed" : ""));

        if (!rows.length) {
            sect.appendChild(el("div", "ntro-insp-note", "No spectral indices were computed for this pixel."));
            return sect;
        }

        var table = el("table", "ntro-insp-table");
        table.appendChild(el("caption", "ntro-insp-sr-only", "Spectral index values, native versus super-resolved"));

        var hr = el("tr");
        hr.appendChild(el("th", null, "Index"));
        hr.appendChild(el("th", "ntro-insp-num", "Native"));
        hr.appendChild(el("th", "ntro-insp-num", "2.5 m"));
        hr.appendChild(el("th", null, "Class"));
        for (var i = 0; i < hr.children.length; i++) hr.children[i].setAttribute("scope", "col");
        var thead = el("thead");
        thead.appendChild(hr);
        table.appendChild(thead);

        var tbody = el("tbody");
        rows.forEach(function (it) {
            var key = str(it.key);
            var name = str(it.name) || key || "Index";
            var unit = str(it.unit);
            var tr = el("tr");

            var nameCell = el("td");
            nameCell.appendChild(el("div", "ntro-insp-idx-name", name));
            if (key && key.toUpperCase() !== name.toUpperCase()) {
                nameCell.appendChild(el("span", "ntro-insp-idx-key", unit ? key + " · " + unit : key));
            } else if (unit) {
                nameCell.appendChild(el("span", "ntro-insp-idx-key", unit));
            }
            tr.appendChild(nameCell);

            tr.appendChild(el("td", "ntro-insp-num", fmtIndex(it.lr)));

            var srCell = el("td", "ntro-insp-num is-sr", fmtIndex(it.sr));
            var delta = fmtDelta(it.sr, it.lr);
            if (delta) srCell.appendChild(el("span", "ntro-insp-delta", delta));
            tr.appendChild(srCell);

            var classCell = el("td");
            var label = str(it.class_label);
            classCell.appendChild(label ? classPill(label, it.class_color) : el("span", "ntro-insp-delta", EM_DASH));
            tr.appendChild(classCell);

            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        sect.appendChild(table);
        return sect;
    }

    function resolveRisk(u, pct) {
        var key = str(u.risk).toLowerCase();
        if (RISK[key]) return { key: key, def: RISK[key] };
        // No usable risk word: derive a band from the confidence itself.
        key = pct === null ? "moderate" : (pct >= 80 ? "low" : pct >= 60 ? "moderate" : pct >= 40 ? "elevated" : "high");
        return { key: key, def: RISK[key] };
    }

    function buildConfidence(u) {
        var pct = null;
        if (num(u.confidence)) {
            // Accept either a 0-1 fraction or an already-scaled percentage.
            pct = Math.max(0, Math.min(100, u.confidence <= 1 ? u.confidence * 100 : u.confidence));
        }
        var risk = resolveRisk(u, pct);

        var box = el("section", "ntro-insp-conf");
        var top = el("div", "ntro-insp-conf-top");
        top.appendChild(el("div", "ntro-insp-conf-label", "Reconstruction confidence"));
        var pill = el("span", "ntro-insp-risk", risk.def.label);
        pill.style.color = risk.def.fg;
        pill.style.backgroundColor = risk.def.bg;
        top.appendChild(pill);
        box.appendChild(top);

        var valRow = el("div", "ntro-insp-conf-val");
        valRow.appendChild(el("span", "ntro-insp-conf-num", pct === null ? EM_DASH : pct.toFixed(1)));
        valRow.appendChild(el("span", "ntro-insp-conf-pct", "%"));
        box.appendChild(valRow);

        var bar = el("div", "ntro-insp-bar");
        bar.setAttribute("role", "progressbar");
        bar.setAttribute("aria-label", "Reconstruction confidence");
        bar.setAttribute("aria-valuemin", "0");
        bar.setAttribute("aria-valuemax", "100");
        if (pct !== null) bar.setAttribute("aria-valuenow", String(Math.round(pct)));
        var fill = el("i");
        fill.style.backgroundColor = risk.def.fg;
        bar.appendChild(fill);
        box.appendChild(bar);

        var width = (pct === null ? 0 : pct) + "%";
        if (reducedMotion()) { fill.style.width = width; }
        else { window.requestAnimationFrame(function () { fill.style.width = width; }); }

        var meta = el("div", "ntro-insp-conf-meta");
        meta.appendChild(document.createTextNode("Ensemble spread σ "));
        meta.appendChild(el("b", null, fmt(u.std, 4)));
        meta.appendChild(document.createTextNode("   ·   novelty "));
        meta.appendChild(el("b", null, fmt(u.novelty, 3)));
        box.appendChild(meta);

        if (risk.key === "elevated" || risk.key === "high") {
            var caution = el("div", "ntro-insp-caution",
                "Fine detail at this pixel is largely model-inferred and is not directly observed by Sentinel-2.");
            caution.style.borderLeftColor = risk.def.fg;
            caution.style.backgroundColor = risk.def.bg;
            box.appendChild(caution);
        }
        return box;
    }

    function stateBlock(container, iconMarkup, message, isError) {
        injectStyles();
        clear(container);
        var wrap = el("div", "ntro-insp");
        var box = el("div", "ntro-insp-empty" + (isError ? " is-error" : ""));
        box.appendChild(icon(iconMarkup));
        box.appendChild(el("p", null, message));
        wrap.appendChild(box);
        container.appendChild(wrap);
        return wrap;
    }

    /* ------------------------------------------------------------ public API */

    function clear(container) {
        if (!container || !container.nodeType) return;
        var t = timers.get(container);
        if (t) { clearTimeout(t); timers.delete(container); }
        while (container.firstChild) container.removeChild(container.firstChild);
    }

    function renderEmpty(container, message) {
        if (!container || !container.nodeType) return null;
        return stateBlock(container, IC_TARGET, str(message) || DEFAULT_EMPTY_MSG, false);
    }

    function render(container, payload) {
        if (!container || !container.nodeType) return null;
        if (!payload || typeof payload !== "object") {
            return stateBlock(container, IC_BROKEN,
                "The pixel response could not be read. Click the patch again to retry.", true);
        }

        try {
            injectStyles();
            clear(container);

            var n = bandCount(payload);
            var bands = bandNames(payload, n);
            var wl = wavelengths(payload, n);
            var series = buildSeries(payload, n);

            var root = el("div", "ntro-insp");
            root.setAttribute("role", "region");
            root.setAttribute("aria-label", "Pixel spectral inspector");

            root.appendChild(buildHeader(payload, container));
            root.appendChild(buildSpectral(payload, bands, wl, series));
            root.appendChild(buildIndices(payload));

            if (payload.uncertainty && typeof payload.uncertainty === "object") {
                root.appendChild(buildConfidence(payload.uncertainty));
            }

            container.appendChild(root);
            return root;
        } catch (e) {
            if (window.console && window.console.warn) window.console.warn("[NTRO.inspector] render failed", e);
            return stateBlock(container, IC_BROKEN,
                "This pixel could not be rendered. Click the patch again to retry.", true);
        }
    }

    NTRO.inspector = {
        render: render,
        clear: clear,
        renderEmpty: renderEmpty
    };
})(window, document);
