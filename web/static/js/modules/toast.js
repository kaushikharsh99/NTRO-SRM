/**
 * NTRO-SRM — Toast notifications
 * Non-blocking status messages with an optional long-running progress handle.
 */
(function (window, document) {
    "use strict";

    var NTRO = window.NTRO = window.NTRO || {};

    const REGION_ID = "ntro-toast-region";
    const STYLE_KEY = "toast";
    const MAX_VISIBLE = 5;
    const DEFAULT_DURATION = 5000;
    const SETTLE_DURATION = 4000; // how long a succeeded/failed progress toast lingers
    const LEAVE_MS = 170;

    const TYPES = { info: true, success: true, warning: true, error: true };

    const ICON_PATHS = {
        info: '<circle cx="12" cy="12" r="9"/><line x1="12" y1="11" x2="12" y2="16.5"/><line x1="12" y1="7.6" x2="12.01" y2="7.6"/>',
        success: '<path d="M21 11.2V12a9 9 0 1 1-5.33-8.22"/><polyline points="21.5 4.7 12 14.2 9.3 11.5"/>',
        warning: '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/><line x1="12" y1="9.2" x2="12" y2="13.4"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
        error: '<circle cx="12" cy="12" r="9"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>',
        close: '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>'
    };

    /** id -> record { id, el, refs, timer, expiresAt, remaining, sticky, paused, finished } */
    const toasts = new Map();
    /** id -> element still playing its exit animation (node present, record already gone) */
    const leaving = new Map();

    let seq = 0;
    let region = null;

    const CSS = [
        '#' + REGION_ID + '{position:fixed;right:16px;bottom:16px;z-index:var(--z-toast,9000);display:flex;flex-direction:column;gap:8px;width:min(380px,calc(100vw - 32px));max-height:calc(100vh - 32px);pointer-events:none;font-family:var(--font-family,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif)}',
        '.ntro-toast{position:relative;display:flex;align-items:flex-start;gap:10px;padding:10px 11px 11px 12px;background-color:var(--bg-card,#ffffff);border:1px solid var(--border-subtle,#e2e8f0);border-left:4px solid var(--nt-accent);border-radius:var(--radius-md,8px);box-shadow:var(--shadow-floating,0 8px 24px -4px rgba(15,23,42,.12),0 2px 6px -1px rgba(15,23,42,.06));overflow:hidden;pointer-events:auto;animation:ntro-toast-in 180ms cubic-bezier(.16,.84,.44,1) both}',
        '.ntro-toast[data-type="info"]{--nt-accent:var(--accent-blue,#0284c7)}',
        '.ntro-toast[data-type="success"]{--nt-accent:var(--accent-emerald,#059669)}',
        '.ntro-toast[data-type="warning"]{--nt-accent:var(--accent-amber,#d97706)}',
        '.ntro-toast[data-type="error"]{--nt-accent:var(--accent-red,#dc2626)}',
        '.ntro-toast.has-bar{padding-bottom:14px}',
        '.ntro-toast.is-leaving{opacity:0;transform:translateX(10px);transition:opacity 160ms ease,transform 160ms ease}',
        '.ntro-toast:focus-visible,.ntro-toast-action:focus-visible,.ntro-toast-close:focus-visible{outline:2px solid var(--border-focus,#3b82f6);outline-offset:1px}',
        '.ntro-toast-icon{flex:0 0 auto;display:flex;margin-top:1px;color:var(--nt-accent)}',
        '.ntro-toast-body{flex:1 1 auto;min-width:0}',
        '.ntro-toast-title{font-size:13px;font-weight:600;letter-spacing:-.1px;line-height:1.35;color:var(--text-primary,#0f172a);margin-bottom:1px}',
        '.ntro-toast-msg{font-size:12.5px;line-height:1.45;color:var(--text-secondary,#475569);overflow-wrap:anywhere}',
        '.ntro-toast-msg:empty{display:none}',
        '.ntro-toast-actions{display:flex;flex-wrap:wrap;gap:2px;margin:7px 0 0 -6px}',
        '.ntro-toast-action{font-family:inherit;font-size:11.5px;font-weight:600;letter-spacing:.1px;color:var(--nt-accent);background:transparent;border:1px solid transparent;border-radius:var(--radius-sm,6px);padding:2px 6px;cursor:pointer}',
        '.ntro-toast-action:hover{background-color:var(--bg-subtle,#f1f5f9)}',
        '.ntro-toast-close{flex:0 0 auto;display:flex;align-items:center;justify-content:center;width:22px;height:22px;margin:-1px -2px 0 0;padding:0;background:transparent;border:0;border-radius:var(--radius-sm,6px);color:var(--text-muted,#64748b);cursor:pointer}',
        '.ntro-toast-close:hover{background-color:var(--bg-subtle,#f1f5f9);color:var(--text-primary,#0f172a)}',
        '.ntro-toast-track{position:absolute;left:0;right:0;bottom:0;height:3px;background-color:var(--bg-subtle,#f1f5f9);overflow:hidden}',
        '.ntro-toast-bar{height:100%;width:0;background-color:var(--nt-accent);transition:width 200ms ease}',
        '.ntro-toast-title[hidden],.ntro-toast-actions[hidden],.ntro-toast-track[hidden]{display:none}',
        '.ntro-toast-bar.is-indeterminate{width:34%;transition:none;animation:ntro-toast-shuttle 1.4s ease-in-out infinite}',
        '@keyframes ntro-toast-in{from{opacity:0;transform:translateX(14px) scale(.99)}to{opacity:1;transform:none}}',
        '@keyframes ntro-toast-shuttle{0%{transform:translateX(-105%)}100%{transform:translateX(310%)}}',
        '@media (max-width:640px){#' + REGION_ID + '{left:12px;right:12px;bottom:12px;width:auto}}',
        '@media (prefers-reduced-motion:reduce){',
        '.ntro-toast{animation:none}',
        '.ntro-toast.is-leaving{transition:none}',
        '.ntro-toast-bar{transition:none}',
        '.ntro-toast-bar.is-indeterminate{animation:none;width:100%;opacity:.45}',
        '}'
    ].join("\n");

    /* --------------------------------------------------------------- helpers */

    function reducedMotion() {
        try {
            return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
        } catch (err) {
            return false;
        }
    }

    function svgIcon(paths, size) {
        return '<svg viewBox="0 0 24 24" width="' + size + '" height="' + size +
            '" fill="none" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" ' +
            'stroke-linejoin="round" aria-hidden="true" focusable="false">' + paths + "</svg>";
    }

    // Accepts anything a caller might realistically pass to error(): string, Error, number, null.
    function asText(value) {
        if (value === null || value === undefined) return "";
        if (typeof value === "string") return value;
        if (value instanceof Error) return value.message || String(value);
        return String(value);
    }

    function clamp(n, lo, hi) {
        return n < lo ? lo : (n > hi ? hi : n);
    }

    // 0 and Infinity both mean "sticky"; anything unusable falls back.
    function resolveDuration(opts, fallback) {
        const d = opts.duration;
        if (d === 0 || d === Infinity) return 0;
        if (typeof d === "number" && isFinite(d) && d > 0) return d;
        return fallback;
    }

    /* ---------------------------------------------------------- dom plumbing */

    function injectStyles() {
        if (document.querySelector('style[data-ntro-style="' + STYLE_KEY + '"]')) return;
        const style = document.createElement("style");
        style.setAttribute("data-ntro-style", STYLE_KEY);
        style.textContent = CSS;
        (document.head || document.documentElement).appendChild(style);
    }

    function ensureRegion() {
        if (region && region.parentNode) return region;
        injectStyles();
        region = document.getElementById(REGION_ID);
        if (!region) {
            region = document.createElement("div");
            region.id = REGION_ID;
            region.setAttribute("role", "region");
            region.setAttribute("aria-live", "polite");
            region.setAttribute("aria-label", "Notifications");
        }
        if (!region.parentNode) (document.body || document.documentElement).appendChild(region);
        return region;
    }

    function buildToast(id) {
        const el = document.createElement("div");
        el.className = "ntro-toast";
        el.tabIndex = 0;

        const icon = document.createElement("span");
        icon.className = "ntro-toast-icon";
        icon.setAttribute("aria-hidden", "true");

        const body = document.createElement("div");
        body.className = "ntro-toast-body";

        const title = document.createElement("div");
        title.className = "ntro-toast-title";
        title.hidden = true;

        const msg = document.createElement("p");
        msg.className = "ntro-toast-msg";

        const actions = document.createElement("div");
        actions.className = "ntro-toast-actions";
        actions.hidden = true;

        const close = document.createElement("button");
        close.type = "button";
        close.className = "ntro-toast-close";
        close.setAttribute("aria-label", "Dismiss notification");
        close.innerHTML = svgIcon(ICON_PATHS.close, 14);

        const track = document.createElement("div");
        track.className = "ntro-toast-track";
        track.hidden = true;

        const bar = document.createElement("div");
        bar.className = "ntro-toast-bar";

        track.appendChild(bar);
        body.appendChild(title);
        body.appendChild(msg);
        body.appendChild(actions);
        el.appendChild(icon);
        el.appendChild(body);
        el.appendChild(close);
        el.appendChild(track);

        close.addEventListener("click", function () { dismissToast(id); });
        el.addEventListener("mouseenter", function () { hold(id, "hover", true); });
        el.addEventListener("mouseleave", function () { hold(id, "hover", false); });
        el.addEventListener("focusin", function () { hold(id, "focused", true); });
        el.addEventListener("focusout", function (e) {
            if (!el.contains(e.relatedTarget)) hold(id, "focused", false);
        });
        el.addEventListener("keydown", function (e) {
            if (e.target !== el) return; // buttons inside handle their own activation
            if (e.key !== "Enter" && e.key !== " " && e.key !== "Spacebar") return;
            e.preventDefault();
            dismissToast(id);
        });

        return {
            el: el,
            refs: { icon: icon, title: title, msg: msg, actions: actions, track: track, bar: bar }
        };
    }

    /* ------------------------------------------------------------- rendering */

    function applyType(rec, type) {
        rec.el.setAttribute("data-type", type);
        rec.refs.icon.innerHTML = svgIcon(ICON_PATHS[type] || ICON_PATHS.info, 16);
    }

    function setTitle(rec, text) {
        const t = text ? String(text) : "";
        rec.refs.title.textContent = t;
        rec.refs.title.hidden = !t;
    }

    function setActions(rec, list) {
        const host = rec.refs.actions;
        while (host.firstChild) host.removeChild(host.firstChild);
        const items = Array.isArray(list) ? list : [];
        let count = 0;
        items.forEach(function (action) {
            if (!action || !action.label) return;
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "ntro-toast-action";
            btn.textContent = String(action.label);
            btn.addEventListener("click", function () {
                try {
                    if (typeof action.onClick === "function") action.onClick();
                } catch (err) {
                    if (window.console && console.error) console.error("[NTRO.toast] action handler failed", err);
                }
                if (action.keepOpen !== true) dismissToast(rec.id);
            });
            host.appendChild(btn);
            count += 1;
        });
        host.hidden = count === 0;
    }

    /* ---------------------------------------------------------------- timers */

    function clearTimer(rec) {
        if (rec.timer) {
            window.clearTimeout(rec.timer);
            rec.timer = null;
        }
    }

    // arm(rec)     -> resume with whatever time is left
    // arm(rec, ms) -> (re)start the countdown; ms <= 0 or non-finite means sticky
    function arm(rec, ms) {
        clearTimer(rec);
        if (typeof ms === "number") {
            rec.sticky = !(isFinite(ms) && ms > 0);
            rec.remaining = rec.sticky ? 0 : ms;
        }
        if (rec.sticky || rec.paused) return;
        rec.expiresAt = Date.now() + rec.remaining;
        rec.timer = window.setTimeout(function () {
            rec.timer = null;
            dismissToast(rec.id);
        }, rec.remaining);
    }

    // The countdown is held while the pointer is over the toast OR focus is inside it,
    // so leaving with the mouse while a button still has focus must not restart it.
    function hold(id, flag, on) {
        const rec = toasts.get(id);
        if (!rec) return;
        rec[flag] = !!on;
        const held = !!(rec.hover || rec.focused);
        if (held === rec.paused) return;
        if (held) {
            if (rec.timer) rec.remaining = Math.max(0, rec.expiresAt - Date.now());
            clearTimer(rec);
            rec.paused = true;
        } else {
            rec.paused = false;
            arm(rec);
        }
    }

    /* ------------------------------------------------------------- lifecycle */

    function trim() {
        if (toasts.size <= MAX_VISIBLE) return;
        // Map keeps insertion order, so the oldest live toasts come first.
        const ids = Array.from(toasts.keys());
        const excess = ids.length - MAX_VISIBLE;
        for (let i = 0; i < excess; i++) dismissToast(ids[i]);
    }

    function removeNode(id, el) {
        if (leaving.get(id) === el) leaving.delete(id);
        if (el && el.parentNode) el.parentNode.removeChild(el);
    }

    function dismissToast(id) {
        const key = (id === null || id === undefined) ? "" : String(id);
        const rec = toasts.get(key);
        if (!rec) return; // unknown id, or a toast whose timer already tore it down
        clearTimer(rec);
        toasts.delete(key);

        const el = rec.el;
        // Keep keyboard users inside the stack instead of dumping focus onto <body>.
        if (el.contains(document.activeElement)) {
            const next = el.nextElementSibling || el.previousElementSibling;
            if (next && typeof next.focus === "function") next.focus();
        }
        el.setAttribute("aria-hidden", "true");
        el.tabIndex = -1;
        leaving.set(key, el);

        if (reducedMotion()) {
            removeNode(key, el);
            return;
        }
        el.classList.add("is-leaving");
        window.setTimeout(function () { removeNode(key, el); }, LEAVE_MS);
    }

    function clearAll() {
        Array.from(toasts.keys()).forEach(dismissToast);
    }

    /* ---------------------------------------------------------- progress bar */

    function setBar(rec, pct) {
        const track = rec.refs.track;
        const bar = rec.refs.bar;
        const value = (typeof pct === "number" && isFinite(pct)) ? clamp(pct, 0, 100) : null;
        if (value === null) {
            bar.classList.add("is-indeterminate");
            bar.style.width = "";
            track.removeAttribute("aria-valuenow");
            track.setAttribute("aria-valuetext", "In progress");
        } else {
            bar.classList.remove("is-indeterminate");
            bar.style.width = value.toFixed(1) + "%";
            track.removeAttribute("aria-valuetext");
            track.setAttribute("aria-valuenow", String(Math.round(value)));
        }
    }

    function showBar(rec, pct) {
        const track = rec.refs.track;
        track.hidden = false;
        track.setAttribute("role", "progressbar");
        track.setAttribute("aria-valuemin", "0");
        track.setAttribute("aria-valuemax", "100");
        rec.el.classList.add("has-bar");
        setBar(rec, pct);
    }

    // Freeze the bar. A finished run reads 100%; a failure that never reported a
    // percentage also fills, but one that did keeps its last figure — how far the
    // job got before it died is the useful diagnostic.
    function stopBar(rec, fillFull) {
        const bar = rec.refs.bar;
        const indeterminate = bar.classList.contains("is-indeterminate");
        const current = parseFloat(bar.style.width);
        setBar(rec, (fillFull || indeterminate || !isFinite(current)) ? 100 : current);
    }

    /* ------------------------------------------------------------ public API */

    function show(message, options) {
        const opts = (options && typeof options === "object") ? options : {};
        const id = (opts.id !== null && opts.id !== undefined && String(opts.id) !== "")
            ? String(opts.id)
            : "ntro-toast-" + (++seq);
        try {
            const type = Object.prototype.hasOwnProperty.call(TYPES, opts.type) ? opts.type : "info";
            const title = (opts.title === null || opts.title === undefined) ? "" : String(opts.title).trim();
            let text = asText(message).trim();
            if (!text && !title) text = "Notification"; // never render a blank card

            let rec = toasts.get(id);
            const isNew = !rec;

            if (isNew) {
                const host = ensureRegion();
                // A node with this id still fading out would collide — drop it now.
                const ghost = leaving.get(id);
                if (ghost) removeNode(id, ghost);

                const built = buildToast(id);
                rec = {
                    id: id, el: built.el, refs: built.refs,
                    timer: null, expiresAt: 0, remaining: 0,
                    sticky: false, paused: false, finished: false,
                    hover: false, focused: false
                };
                host.insertBefore(rec.el, host.firstChild); // newest on top
                toasts.set(id, rec);
            } else {
                rec.finished = false;
            }

            applyType(rec, type);
            setTitle(rec, title);
            rec.refs.msg.textContent = text;
            setActions(rec, opts.actions);
            arm(rec, resolveDuration(opts, DEFAULT_DURATION));
            if (isNew) trim();
        } catch (err) {
            if (window.console && console.error) console.error("[NTRO.toast] show failed", err);
        }
        return id;
    }

    function typed(type) {
        return function (message, options) {
            const opts = (options && typeof options === "object") ? options : {};
            return show(message, Object.assign({}, opts, { type: type }));
        };
    }

    function progress(message, options) {
        const opts = (options && typeof options === "object") ? options : {};
        // Sticky until succeed() / fail() / dismiss() says otherwise.
        const id = show(message, Object.assign({}, opts, { duration: 0 }));
        const rec = toasts.get(id);
        if (rec) showBar(rec, opts.percent);

        function update(msg, pct) {
            const r = toasts.get(id);
            if (!r) return;
            if (msg !== null && msg !== undefined) r.refs.msg.textContent = asText(msg);
            if (!r.finished) setBar(r, pct);
        }

        // Morphs the existing node in place — same element, new accent / icon / title.
        function settle(type, msg, defaultTitle, fillFull) {
            const r = toasts.get(id);
            if (!r) return; // already dismissed by the user
            r.finished = true;
            applyType(r, type);
            setTitle(r, (type === "success" ? opts.successTitle : opts.failTitle) || defaultTitle);
            if (msg !== null && msg !== undefined) r.refs.msg.textContent = asText(msg);
            stopBar(r, fillFull);
            arm(r, SETTLE_DURATION);
        }

        return {
            id: id,
            update: update,
            succeed: function (msg) { settle("success", msg, "Completed", true); },
            fail: function (msg) { settle("error", msg, "Failed", false); },
            dismiss: function () { dismissToast(id); }
        };
    }

    NTRO.toast = {
        show: show,
        info: typed("info"),
        success: typed("success"),
        warning: typed("warning"),
        error: typed("error"),
        dismiss: dismissToast,
        clear: clearAll,
        progress: progress
    };
})(window, document);
