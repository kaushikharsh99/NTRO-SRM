/**
 * NTRO-SRM — Loupe
 * A circular cursor-following magnifier that renders a second, deeper-zoomed Leaflet
 * map over the main map, so a reviewer can judge whether 2.5 m reconstruction is
 * resolving real structure.
 */
(function (window, document) {
    "use strict";

    var NTRO = (window.NTRO = window.NTRO || {});

    var STYLE_FLAG = "loupe";
    var DEFAULT_SIZE = 220;
    var DEFAULT_ZOOM_OFFSET = 3;
    var MIN_SIZE = 96;
    var MAX_SIZE = 520;
    var MIN_OFFSET = -4;
    var MAX_OFFSET = 10;

    // The SR image overlays carry no maxZoom, but the basemap tile layer does. Without an
    // explicit map-level maxZoom Leaflet clamps to the basemap's value and silently refuses
    // the setView, freezing the loupe short of the requested magnification.
    var INSET_MAX_ZOOM = 24;

    var CURSOR_GAP = 26; // px between the pointer and the nearest edge of the loupe
    var EDGE_PAD = 10;   // keep the loupe this far inside the container

    var state = {
        mainMap: null,
        mapContainer: null, // mainMap.getContainer() — the mousemove surface
        container: null,    // the viewport element the loupe is positioned inside
        buildLayers: null,
        zoomOffset: DEFAULT_ZOOM_OFFSET,
        size: DEFAULT_SIZE,
        enabled: false,
        bound: false,
        visible: false,
        suspended: false,   // hidden for the duration of a drag / zoom gesture
        inside: false,      // pointer is currently over the map container
        pointer: null,      // last { clientX, clientY }
        frame: 0,
        wrapper: null,
        canvas: null,
        cross: null,
        note: null,
        badge: null,
        insetMap: null,
        layers: []
    };

    /* ---------------------------------------------------------------- styling */

    // z-index 0 on the canvas makes the inset map its own stacking context, so Leaflet's
    // panes (z-index up to 800) stay beneath the crosshair, badge and note.
    var CSS = [
        ".ntro-loupe{position:absolute;top:0;left:0;display:none;box-sizing:border-box;width:220px;height:220px;border-radius:50%;overflow:hidden;pointer-events:none;z-index:var(--z-loupe,8000);opacity:0;visibility:hidden;background:var(--bg-subtle,#f1f5f9);border:3px solid var(--bg-panel,#ffffff);box-shadow:0 0 0 1px var(--border-medium,#cbd5e1),var(--shadow-floating,0 8px 24px -4px rgba(15,23,42,.12));transition:opacity 110ms linear;will-change:transform}",
        ".ntro-loupe.is-active{display:block}",
        ".ntro-loupe.is-visible{opacity:1;visibility:visible}",
        ".ntro-loupe-canvas{position:absolute;top:0;right:0;bottom:0;left:0;z-index:0;background:var(--bg-subtle,#f1f5f9)}",
        ".ntro-loupe-canvas .leaflet-container{width:100%;height:100%;background:transparent;outline:0;font-family:var(--font-family,sans-serif)}",
        ".ntro-loupe-canvas .leaflet-control-container{display:none}",
        ".ntro-loupe-cross{position:absolute;left:50%;top:50%;width:0;height:0;z-index:2}",
        '.ntro-loupe-cross::before,.ntro-loupe-cross::after{content:"";position:absolute;background:var(--loupe-reticle,#ffffff);box-shadow:0 0 0 1px var(--loupe-reticle-edge,rgba(15,23,42,.55))}',
        ".ntro-loupe-cross::before{left:-6px;top:0;width:12px;height:1px;transform:translateY(-50%)}",
        ".ntro-loupe-cross::after{top:-6px;left:0;width:1px;height:12px;transform:translateX(-50%)}",
        ".ntro-loupe.has-note .ntro-loupe-cross{display:none}",
        ".ntro-loupe-badge{position:absolute;left:50%;bottom:8px;z-index:2;transform:translateX(-50%);padding:3px 7px;border-radius:var(--radius-full,9999px);font-family:var(--font-mono,monospace);font-size:10px;line-height:1;font-weight:600;letter-spacing:.02em;white-space:nowrap;color:#ffffff;background:var(--loupe-badge-bg,var(--overlay-scrim,rgba(15,23,42,.62)));border:1px solid rgba(255,255,255,.16)}",
        ".ntro-loupe-note{position:absolute;left:50%;top:50%;z-index:2;transform:translate(-50%,-50%);max-width:76%;text-align:center;font-family:var(--font-family,sans-serif);font-size:11px;font-weight:600;line-height:1.35;letter-spacing:.01em;color:var(--text-muted,#64748b)}",
        ".ntro-loupe-note.is-error{color:var(--accent-red,#dc2626)}",
        ".ntro-loupe-note[hidden]{display:none}",
        "@media (prefers-reduced-motion: reduce){.ntro-loupe{transition:none}}"
    ].join("");

    function injectStyles() {
        if (document.querySelector('style[data-ntro-style="' + STYLE_FLAG + '"]')) return;
        var style = document.createElement("style");
        style.setAttribute("data-ntro-style", STYLE_FLAG);
        style.textContent = CSS;
        (document.head || document.documentElement).appendChild(style);
    }

    /* -------------------------------------------------------------- utilities */

    function ready() {
        return !!(window.L && state.mainMap && state.mapContainer && state.container);
    }

    function normaliseOffset(value, fallback) {
        var n = Number(value);
        if (!isFinite(n)) return fallback;
        return Math.max(MIN_OFFSET, Math.min(MAX_OFFSET, Math.round(n)));
    }

    function normaliseSize(value, fallback) {
        var n = Number(value);
        if (!isFinite(n)) return fallback;
        return Math.max(MIN_SIZE, Math.min(MAX_SIZE, Math.round(n)));
    }

    function warn(message, err) {
        if (window.console && window.console.warn) window.console.warn("[NTRO.loupe] " + message, err);
    }

    /* -------------------------------------------------------------------- DOM */

    function ensureWrapper() {
        if (state.wrapper) return true;
        if (!state.container) return false;

        var wrapper = document.createElement("div");
        wrapper.className = "ntro-loupe";
        // Purely a pointer-driven visual aid duplicating map content already on screen.
        wrapper.setAttribute("aria-hidden", "true");
        wrapper.style.width = state.size + "px";
        wrapper.style.height = state.size + "px";

        var canvas = document.createElement("div");
        canvas.className = "ntro-loupe-canvas";
        wrapper.appendChild(canvas);

        var cross = document.createElement("div");
        cross.className = "ntro-loupe-cross";
        wrapper.appendChild(cross);

        var note = document.createElement("div");
        note.className = "ntro-loupe-note";
        note.hidden = true;
        wrapper.appendChild(note);

        var badge = document.createElement("div");
        badge.className = "ntro-loupe-badge";
        badge.textContent = formatBadge(state.zoomOffset, null);
        wrapper.appendChild(badge);

        state.container.appendChild(wrapper);
        state.wrapper = wrapper;
        state.canvas = canvas;
        state.cross = cross;
        state.note = note;
        state.badge = badge;
        return true;
    }

    function setNote(text, isError) {
        if (!state.note || !state.wrapper) return;
        if (text) {
            state.note.textContent = text;
            state.note.hidden = false;
            if (isError) state.note.classList.add("is-error");
            else state.note.classList.remove("is-error");
            state.wrapper.classList.add("has-note");
        } else {
            state.note.hidden = true;
            state.note.classList.remove("is-error");
            state.wrapper.classList.remove("has-note");
        }
    }

    function formatBadge(offset, zoom) {
        // Escapes, not literals: the badge must survive being served as anything but UTF-8.
        var head = (offset < 0 ? "\u2212" : "+") + Math.abs(offset) + "\u00d7";
        if (zoom === null || zoom === undefined || !isFinite(zoom)) return head;
        var z = Math.round(zoom * 10) % 10 === 0 ? String(Math.round(zoom)) : zoom.toFixed(1);
        return head + " \u00b7 z" + z;
    }

    function updateBadge(zoom) {
        if (state.badge) state.badge.textContent = formatBadge(state.zoomOffset, zoom);
    }

    function show() {
        if (state.visible || !state.wrapper) return;
        state.wrapper.classList.add("is-visible");
        state.visible = true;
    }

    function hide() {
        if (!state.visible || !state.wrapper) return;
        state.wrapper.classList.remove("is-visible");
        state.visible = false;
    }

    /* -------------------------------------------------------------- inset map */

    function ensureInsetMap() {
        if (state.insetMap) return true;
        var L = window.L;
        if (!L || !state.canvas) return false;

        var center = null;
        var zoom = 2;
        try {
            center = state.mainMap.getCenter();
            zoom = state.mainMap.getZoom();
        } catch (err) {
            center = null; // the main map has no view yet
        }
        if (!center) { center = L.latLng(0, 0); zoom = 2; }

        try {
            state.insetMap = L.map(state.canvas, {
                center: center,
                zoom: Math.min(zoom + state.zoomOffset, INSET_MAX_ZOOM),
                maxZoom: INSET_MAX_ZOOM,
                zoomSnap: 0,
                zoomControl: false, attributionControl: false,
                dragging: false, scrollWheelZoom: false, doubleClickZoom: false,
                boxZoom: false, keyboard: false, touchZoom: false, tap: false,
                inertia: false, fadeAnimation: false, zoomAnimation: false,
                markerZoomAnimation: false, trackResize: false
            });
        } catch (err) {
            state.insetMap = null;
            warn("could not create the inset map", err);
            setNote("Magnifier unavailable", true);
            return false;
        }

        applyLayers();
        return true;
    }

    function applyLayers() {
        if (!state.insetMap) return;

        var i;
        for (i = 0; i < state.layers.length; i++) {
            try { state.insetMap.removeLayer(state.layers[i]); } catch (err) { /* already gone */ }
        }
        state.layers = [];

        var built = null;
        var failed = false;
        if (typeof state.buildLayers === "function") {
            try { built = state.buildLayers(); }
            catch (err) { failed = true; warn("buildLayers() threw", err); }
        }
        if (!Array.isArray(built)) built = built ? [built] : [];

        for (i = 0; i < built.length; i++) {
            var layer = built[i];
            if (!layer || typeof layer.addTo !== "function") continue;
            // A Leaflet layer instance can only live on one map: adopting one that is still
            // on the main map would silently strip it from the main view.
            if (layer._map && layer._map !== state.insetMap) {
                warn("skipped a layer still attached to another map", null);
                continue;
            }
            try { layer.addTo(state.insetMap); state.layers.push(layer); }
            catch (err) { failed = true; warn("a layer failed to attach", err); }
        }

        if (failed) setNote("Layer error", true);
        else if (!state.layers.length) setNote("No imagery loaded", false);
        else setNote("", false);
    }

    function invalidateSoon() {
        if (!state.insetMap) return;
        // The map may have been created — or last measured — while the wrapper was
        // display:none, in which case Leaflet cached a 0x0 size.
        try { state.insetMap.invalidateSize(false); } catch (err) { /* not loaded yet */ }
        window.requestAnimationFrame(function () {
            if (!state.insetMap || !state.enabled) return;
            try { state.insetMap.invalidateSize(false); } catch (err) { /* ignore */ }
        });
    }

    /* -------------------------------------------------------------- placement */

    function place(clientX, clientY) {
        var box = state.container;
        var rect = box.getBoundingClientRect();
        var px = clientX - rect.left - (box.clientLeft || 0);
        var py = clientY - rect.top - (box.clientTop || 0);
        var size = state.size;

        var x = px + CURSOR_GAP;
        var y = py + CURSOR_GAP;
        if (x + size > rect.width - EDGE_PAD) x = px - CURSOR_GAP - size;
        if (y + size > rect.height - EDGE_PAD) y = py - CURSOR_GAP - size;

        var maxX = Math.max(EDGE_PAD, rect.width - size - EDGE_PAD);
        var maxY = Math.max(EDGE_PAD, rect.height - size - EDGE_PAD);
        x = Math.max(EDGE_PAD, Math.min(x, maxX));
        y = Math.max(EDGE_PAD, Math.min(y, maxY));

        state.wrapper.style.transform =
            "translate(" + Math.round(x) + "px, " + Math.round(y) + "px)";
    }

    function targetZoom() {
        var base;
        try { base = state.mainMap.getZoom(); } catch (err) { base = null; }
        if (typeof base !== "number" || !isFinite(base)) return null;

        var z = base + state.zoomOffset;
        var maxZ = state.insetMap.getMaxZoom();
        var minZ = state.insetMap.getMinZoom();
        if (isFinite(maxZ)) z = Math.min(z, maxZ);
        if (isFinite(minZ)) z = Math.max(z, minZ);
        return z;
    }

    function renderFrame() {
        state.frame = 0;
        if (!state.enabled || state.suspended || !state.insetMap || !state.pointer) return;

        var latlng = null;
        try { latlng = state.mainMap.mouseEventToLatLng(state.pointer); } catch (err) { latlng = null; }
        var zoom = latlng ? targetZoom() : null;
        if (!latlng || zoom === null) { hide(); return; }

        place(state.pointer.clientX, state.pointer.clientY);
        try { state.insetMap.setView(latlng, zoom, { animate: false }); }
        catch (err) { warn("setView failed", err); }
        updateBadge(zoom);
        show(); // after positioning, so the loupe never flashes at its previous spot
    }

    function schedule() {
        if (state.frame || !state.enabled) return;
        state.frame = window.requestAnimationFrame(renderFrame);
    }

    /* ----------------------------------------------------------------- events */

    function onMouseMove(ev) {
        state.inside = true;
        if (!state.enabled) return;
        // Leaflet's mouseEventToLatLng only reads clientX/clientY, so a plain point is enough
        // and it survives being held across a drag gesture.
        state.pointer = { clientX: ev.clientX, clientY: ev.clientY };
        if (state.suspended) return;
        schedule();
    }

    function onMouseOut(ev) {
        var to = ev.relatedTarget;
        // mouseout also fires when crossing between children of the map container.
        if (to && state.mapContainer && state.mapContainer.contains(to)) return;
        state.inside = false;
        hide();
    }

    function onGestureStart() {
        state.suspended = true;
        hide();
    }

    function onGestureEnd() {
        state.suspended = false;
        if (state.enabled && state.inside && state.pointer) schedule();
    }

    function bindListeners() {
        if (state.bound || !ready()) return;
        state.mapContainer.addEventListener("mousemove", onMouseMove, { passive: true });
        state.mapContainer.addEventListener("mouseout", onMouseOut);
        state.mainMap.on("dragstart", onGestureStart);
        state.mainMap.on("zoomstart", onGestureStart);
        state.mainMap.on("dragend", onGestureEnd);
        state.mainMap.on("zoomend", onGestureEnd);
        state.bound = true;
    }

    function unbindListeners() {
        if (!state.bound) return;
        if (state.mapContainer) {
            state.mapContainer.removeEventListener("mousemove", onMouseMove);
            state.mapContainer.removeEventListener("mouseout", onMouseOut);
        }
        if (state.mainMap && typeof state.mainMap.off === "function") {
            state.mainMap.off("dragstart", onGestureStart);
            state.mainMap.off("zoomstart", onGestureStart);
            state.mainMap.off("dragend", onGestureEnd);
            state.mainMap.off("zoomend", onGestureEnd);
        }
        state.bound = false;
    }

    function cancelFrame() {
        if (!state.frame) return;
        window.cancelAnimationFrame(state.frame);
        state.frame = 0;
    }

    /* ------------------------------------------------------------- public API */

    function attach(mainMap, config) {
        if (!window.L) return false;
        if (!mainMap || typeof mainMap.getContainer !== "function") return false;

        var mapContainer;
        try { mapContainer = mainMap.getContainer(); } catch (err) { return false; }
        if (!mapContainer) return false;

        var cfg = config || {};
        var container = cfg.container || mapContainer;
        if (!container || container.nodeType !== 1) return false;

        if (state.mainMap) detach(); // re-attaching must not leave a stale inset map behind

        injectStyles();
        state.mainMap = mainMap;
        state.mapContainer = mapContainer;
        state.container = container;
        state.buildLayers = typeof cfg.buildLayers === "function" ? cfg.buildLayers : null;
        state.zoomOffset = normaliseOffset(cfg.zoomOffset, DEFAULT_ZOOM_OFFSET);
        state.size = normaliseSize(cfg.size, DEFAULT_SIZE);

        // translate() offsets resolve against the nearest positioned ancestor.
        try {
            if (window.getComputedStyle(container).position === "static") {
                container.style.position = "relative";
            }
        } catch (err) { /* detached node */ }
        return true;
    }

    function enable() {
        if (!ready() || state.enabled) return;
        if (!ensureWrapper()) return;

        state.enabled = true;
        // Reveal before building the map: Leaflet measures its container at construction,
        // and a map created inside a display:none element renders at 0x0 forever.
        state.wrapper.classList.add("is-active");

        var isFirst = !state.insetMap;
        if (!ensureInsetMap()) {
            state.enabled = false;
            state.wrapper.classList.remove("is-active");
            return;
        }
        // ensureInsetMap() builds the layers itself the first time. On every later enable()
        // rebuild them, so a job that finished while the loupe was off is not shown stale.
        if (!isFirst) applyLayers();

        bindListeners();
        invalidateSoon();
        if (state.inside && state.pointer) schedule();
    }

    function disable() {
        if (!state.enabled) return;
        state.enabled = false;
        cancelFrame();
        unbindListeners();
        hide();
        if (state.wrapper) state.wrapper.classList.remove("is-active");
        state.suspended = false;
    }

    function toggle() {
        if (state.enabled) disable();
        else enable();
        return state.enabled;
    }

    function isEnabled() {
        return state.enabled;
    }

    function refresh() {
        if (!state.insetMap) return; // layers are built fresh on the next enable()
        applyLayers();
        if (state.enabled && state.visible) schedule();
    }

    function setZoomOffset(n) {
        var next = normaliseOffset(n, state.zoomOffset);
        if (next === state.zoomOffset) return;
        state.zoomOffset = next;
        if (!state.insetMap) {
            updateBadge(null);
            return;
        }
        if (state.enabled && state.visible && state.pointer) schedule();
        else updateBadge(targetZoom());
    }

    // Not in the contract — provided so a host page can tear the loupe down (hot reload,
    // map rebuild) without leaking listeners or stranding a second Leaflet map.
    function detach() {
        cancelFrame();
        unbindListeners();
        state.enabled = false;
        state.visible = false;
        state.suspended = false;
        state.inside = false;
        state.pointer = null;

        if (state.insetMap) {
            try { state.insetMap.remove(); } catch (err) { warn("inset map teardown failed", err); }
        }
        state.insetMap = null;
        state.layers = [];

        if (state.wrapper && state.wrapper.parentNode) {
            state.wrapper.parentNode.removeChild(state.wrapper);
        }
        state.wrapper = null;
        state.canvas = null;
        state.cross = null;
        state.note = null;
        state.badge = null;

        state.mainMap = null;
        state.mapContainer = null;
        state.container = null;
        state.buildLayers = null;
    }

    NTRO.loupe = {
        attach: attach,
        detach: detach,
        enable: enable,
        disable: disable,
        toggle: toggle,
        isEnabled: isEnabled,
        refresh: refresh,
        setZoomOffset: setZoomOffset
    };
})(window, document);
