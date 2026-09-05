/**
 * NTRO-SRM — Theme
 * Light / dark / system colour-scheme preference: early (flash-free) application,
 * persistence, live tracking of the OS setting, and the header toggle control.
 *
 * Anti-flash integration — put this in <head> BEFORE any stylesheet, with this
 * file loaded immediately above it (or with the body of applyStoredEarly inlined):
 *
 *     <script src="/static/js/modules/theme.js"></script>
 *     <script>window.NTRO.theme.applyStoredEarly();</script>
 *
 * Then once on DOMContentLoaded:
 *
 *     NTRO.theme.init({ toggleSelector: "#btn-theme" });
 */
(function (window, document) {
    "use strict";

    var NTRO = window.NTRO = window.NTRO || {};

    var STORAGE_KEY = "ntro-srm.theme";
    var DARK_QUERY = "(prefers-color-scheme: dark)";
    var SWITCH_CLASS = "ntro-theme-switching";
    var DEFAULT_SELECTOR = "[data-ntro-theme-toggle]";

    // 16px / 24-viewBox stroke icons, matching the icon vocabulary in index.html.
    var ICON_ATTRS =
        'viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" ' +
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" focusable="false" aria-hidden="true"';

    var ICON_SUN = '<svg ' + ICON_ATTRS + '><circle cx="12" cy="12" r="4.2"></circle>' +
        '<line x1="12" y1="1.6" x2="12" y2="3.6"></line><line x1="12" y1="20.4" x2="12" y2="22.4"></line>' +
        '<line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>' +
        '<line x1="1.6" y1="12" x2="3.6" y2="12"></line><line x1="20.4" y1="12" x2="22.4" y2="12"></line>' +
        '<line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>';

    var ICON_MOON = '<svg ' + ICON_ATTRS + '><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>';

    // "Auto" reads as the device setting, so a monitor — and it stays pure stroke,
    // unlike a half-filled contrast disc, and cannot be mistaken for the moon at 16px.
    var ICON_AUTO = '<svg ' + ICON_ATTRS + '><rect x="2" y="3.5" width="20" height="14" rx="2"></rect>' +
        '<line x1="8" y1="21" x2="16" y2="21"></line><line x1="12" y1="17.5" x2="12" y2="21"></line></svg>';

    var MODE_META = {
        light:  { label: "Light", next: "dark",   icon: ICON_SUN },
        dark:   { label: "Dark",  next: "system", icon: ICON_MOON },
        system: { label: "Auto",  next: "light",  icon: ICON_AUTO }
    };

    var state = {
        preference: null,   // "light" | "dark" | "system" — lazily read from storage
        resolved: null,     // "light" | "dark"
        storageOk: true,    // false once localStorage has thrown; preference then lives in memory
        initialised: false
    };

    var subscribers = [];
    var toggles = [];
    var stylesInjected = false;
    var mediaBound = false;
    var storageBound = false;
    var switchTimer = null;

    function warn(message, err) {
        if (window.console && window.console.warn) window.console.warn("[NTRO.theme] " + message, err || "");
    }

    // --- 1. Preference storage and resolution ------------------------------------

    function normalisePref(value) {
        return (value === "light" || value === "dark" || value === "system") ? value : null;
    }

    function readStored() {
        if (!state.storageOk) return state.preference || "system";
        try {
            return normalisePref(window.localStorage.getItem(STORAGE_KEY)) || "system";
        } catch (err) {
            state.storageOk = false;
            return state.preference || "system";
        }
    }

    function writeStored(pref) {
        try {
            window.localStorage.setItem(STORAGE_KEY, pref);
        } catch (err) {
            // Private-browsing / disabled storage: the preference still holds for this session.
            state.storageOk = false;
        }
    }

    function systemPrefersDark() {
        try {
            return !!(window.matchMedia && window.matchMedia(DARK_QUERY).matches);
        } catch (err) {
            return false;
        }
    }

    function resolveMode(pref) {
        if (pref === "light" || pref === "dark") return pref;
        return systemPrefersDark() ? "dark" : "light";
    }

    // --- 2. Applying the resolution to the document ------------------------------

    function updateMetaColorScheme(resolved) {
        var head = document.head || document.getElementsByTagName("head")[0];
        if (!head) return;
        try {
            var meta = head.querySelector('meta[name="color-scheme"]');
            if (!meta) {
                meta = document.createElement("meta");
                meta.setAttribute("name", "color-scheme");
                head.appendChild(meta);
            }
            meta.setAttribute("content", resolved === "dark" ? "dark light" : "light dark");
        } catch (err) {
            warn('could not maintain <meta name="color-scheme">', err);
        }
    }

    /**
     * Kill every CSS transition on the page for one beat while the tokens swap, so the
     * theme change lands as a single clean repaint instead of hundreds of staggered fades.
     */
    function suppressTransitions() {
        var root = document.documentElement;
        if (!root || !root.classList) return;
        root.classList.add(SWITCH_CLASS);
        if (switchTimer) window.clearTimeout(switchTimer);
        switchTimer = window.setTimeout(function () {
            switchTimer = null;
            root.classList.remove(SWITCH_CLASS);
        }, 140);
    }

    /**
     * @param {String} pref one of light|dark|system
     * @param {Object} [options] { guard: suppress transitions, force: notify even if unchanged }
     */
    function apply(pref, options) {
        var opts = options || {};
        var resolved = resolveMode(pref);
        var changed = (resolved !== state.resolved) || (pref !== state.preference);
        var root = document.documentElement;

        if (changed && opts.guard) suppressTransitions();

        if (root) {
            root.setAttribute("data-theme", resolved);
            root.setAttribute("data-theme-pref", pref);
            try { root.style.colorScheme = resolved; } catch (err) { /* older engines */ }
        }
        updateMetaColorScheme(resolved);

        state.preference = pref;
        state.resolved = resolved;

        decorateAll();
        if (changed || opts.force) notify();
        return resolved;
    }

    /**
     * Synchronous, dependency-free, DOM-ready-free. Safe to call from an inline <script>
     * in <head> before the stylesheet loads. The body is deliberately self-contained
     * (duplicating a few lines above) so it can also be hand-inlined verbatim; the two
     * `state` assignments near the end are the only lines a hand-inlined copy should drop.
     */
    function applyStoredEarly() {
        var pref = "system";
        var resolved = "light";
        try {
            try {
                var raw = window.localStorage.getItem(STORAGE_KEY);
                if (raw === "light" || raw === "dark" || raw === "system") pref = raw;
            } catch (storageErr) { pref = "system"; }

            resolved = pref !== "system" ? pref
                : ((window.matchMedia && window.matchMedia(DARK_QUERY).matches) ? "dark" : "light");

            var root = document.documentElement;
            if (root) {
                root.setAttribute("data-theme", resolved);
                root.setAttribute("data-theme-pref", pref);
                try { root.style.colorScheme = resolved; } catch (csErr) { /* older engines */ }
            }

            state.preference = pref;
            state.resolved = resolved;
        } catch (err) {
            // Never let the anti-flash hook break page parsing.
        }
        return resolved;
    }

    // --- 3. Subscribers ----------------------------------------------------------

    function notify() {
        var snapshot = subscribers.slice();
        for (var i = 0; i < snapshot.length; i++) {
            try {
                snapshot[i](state.resolved, state.preference);
            } catch (err) {
                warn("an onChange subscriber threw", err);
            }
        }
    }

    function onChange(fn) {
        if (typeof fn !== "function") {
            warn("onChange expects a function");
            return function () {};
        }
        subscribers.push(fn);
        var live = true;
        return function unsubscribe() {
            if (!live) return;
            live = false;
            var idx = subscribers.indexOf(fn);
            if (idx !== -1) subscribers.splice(idx, 1);
        };
    }

    // --- 4. Toggle control -------------------------------------------------------

    function titleFor(pref) {
        var meta = MODE_META[pref] || MODE_META.system;
        var nextLabel = MODE_META[meta.next].label;
        if (pref === "system") {
            return "Colour theme: Auto — following the system setting (currently " +
                (state.resolved || resolveMode("system")) + "). Click for " + nextLabel + ".";
        }
        return "Colour theme: " + meta.label + ". Click for " + nextLabel + ".";
    }

    function decorate(entry) {
        if (!entry || !entry.el) return;
        var pref = state.preference || "system";
        var meta = MODE_META[pref] || MODE_META.system;
        var nextLabel = MODE_META[meta.next].label;

        entry.el.setAttribute("data-theme-mode", pref);
        entry.el.setAttribute("aria-label", "Colour theme: " + meta.label + " (click for " + nextLabel + ")");
        entry.el.setAttribute("title", titleFor(pref));

        if (entry.icon.getAttribute("data-icon") !== pref) {
            entry.icon.setAttribute("data-icon", pref);
            entry.icon.innerHTML = meta.icon;   // replacing the node re-triggers the keyframe
        }
        if (entry.label.textContent !== meta.label) entry.label.textContent = meta.label;

        // Arm the live region only after the first paint, so populating the button on
        // page load is not announced as if the user had changed something.
        if (!entry.primed) {
            entry.primed = true;
            window.setTimeout(function () {
                if (entry.label) entry.label.setAttribute("aria-live", "polite");
            }, 0);
        }
    }

    function decorateAll() {
        for (var i = 0; i < toggles.length; i++) decorate(toggles[i]);
    }

    function onToggleClick(event) {
        if (event && event.preventDefault) event.preventDefault();
        toggle();
    }

    function onToggleKeydown(event) {
        if (!event) return;
        if (event.key === "Enter" || event.key === " " || event.key === "Spacebar") {
            event.preventDefault();
            toggle();
        }
    }

    function attachToggle(el) {
        if (!el || el.nodeType !== 1) return;
        if (el.getAttribute("data-ntro-theme-bound") === "1") return;
        el.setAttribute("data-ntro-theme-bound", "1");

        if (el.tagName === "BUTTON") {
            if (!el.getAttribute("type")) el.setAttribute("type", "button");   // never submit a form
        } else {
            // A non-button host still has to behave like one.
            if (!el.getAttribute("role")) el.setAttribute("role", "button");
            if (el.getAttribute("tabindex") === null) el.setAttribute("tabindex", "0");
            el.addEventListener("keydown", onToggleKeydown);
        }
        if (el.classList) el.classList.add("ntro-theme-toggle");

        var icon = document.createElement("span");
        icon.className = "ntro-theme-toggle__icon";
        icon.setAttribute("aria-hidden", "true");

        var label = document.createElement("span");
        label.className = "ntro-theme-toggle__label";

        el.innerHTML = "";
        el.appendChild(icon);
        el.appendChild(label);
        el.addEventListener("click", onToggleClick);

        var entry = { el: el, icon: icon, label: label, primed: false };
        toggles.push(entry);
        decorate(entry);
    }

    function bindToggles(selector) {
        var nodes;
        try {
            nodes = document.querySelectorAll(selector);
        } catch (err) {
            warn('invalid toggleSelector "' + selector + '"', err);
            return 0;   // a bad selector must not stop init() from applying the theme
        }
        for (var i = 0; i < nodes.length; i++) attachToggle(nodes[i]);
        return nodes.length;
    }

    // --- 5. Live sources: the OS setting, and this app open in another tab -------

    function ensureMediaWatcher() {
        if (mediaBound || !window.matchMedia) return;
        var mql;
        try {
            mql = window.matchMedia(DARK_QUERY);
        } catch (err) {
            return;
        }
        if (!mql) return;

        var handler = function () {
            if (state.preference !== "system") return;
            apply("system", { guard: true });
        };
        if (mql.addEventListener) mql.addEventListener("change", handler);
        else if (mql.addListener) mql.addListener(handler);
        else return;
        mediaBound = true;
    }

    function ensureStorageWatcher() {
        if (storageBound) return;
        storageBound = true;
        window.addEventListener("storage", function (event) {
            if (!event || event.key !== STORAGE_KEY) return;
            apply(normalisePref(event.newValue) || "system", { guard: true });
        });
    }

    // --- 6. Styles ---------------------------------------------------------------

    var CSS = [
        // Layout and type only, so a host that already dressed the element as .btn keeps its chrome.
        ".ntro-theme-toggle{display:inline-flex;align-items:center;gap:.4rem;font-family:var(--font-family,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif);font-size:.8rem;font-weight:600;line-height:1;cursor:pointer;-webkit-user-select:none;user-select:none}",
        // Standalone chrome, applied only when the host has NOT already made it a .btn.
        ".ntro-theme-toggle:not(.btn){padding:.45rem .7rem;border-radius:var(--radius-sm,6px);border:1px solid var(--border-subtle,#e2e8f0);background-color:var(--bg-subtle,#f1f5f9);color:var(--text-secondary,#475569);transition:background-color .15s ease,border-color .15s ease,color .15s ease}",
        ".ntro-theme-toggle:not(.btn):hover{background-color:var(--bg-hover,#e2e8f0);border-color:var(--border-medium,#cbd5e1);color:var(--text-primary,#0f172a)}",
        ".ntro-theme-toggle:not(.btn):active{transform:translateY(1px)}",
        ".ntro-theme-toggle:focus-visible{outline:2px solid var(--border-focus,#3b82f6);outline-offset:2px}",
        ".ntro-theme-toggle__icon{display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;flex:0 0 16px}",
        ".ntro-theme-toggle__icon>svg{display:block;width:16px;height:16px;animation:ntro-theme-icon-in .18s ease-out}",
        // Auto is a deferral rather than a choice, so its glyph sits one step back.
        '.ntro-theme-toggle[data-theme-mode="system"] .ntro-theme-toggle__icon{color:var(--text-muted,#64748b)}',
        // Fixed label box so the header does not reflow as Light/Dark/Auto cycle.
        ".ntro-theme-toggle__label{display:inline-block;min-width:2.5em;text-align:left;letter-spacing:.1px}",
        "@keyframes ntro-theme-icon-in{from{opacity:0;transform:rotate(-30deg) scale(.82)}to{opacity:1;transform:none}}",
        // One clean repaint on swap instead of hundreds of staggered token transitions.
        "html." + SWITCH_CLASS + " *,html." + SWITCH_CLASS + " *::before,html." + SWITCH_CLASS + " *::after{transition:none!important;animation-duration:.001ms!important}",
        "@media (prefers-reduced-motion:reduce){.ntro-theme-toggle:not(.btn){transition:none}.ntro-theme-toggle:not(.btn):active{transform:none}.ntro-theme-toggle__icon>svg{animation:none}}"
    ].join("");

    function injectStyles() {
        if (stylesInjected) return;
        var head = document.head || document.getElementsByTagName("head")[0];
        if (!head) return; // called before <head> exists; retried on the next public call
        if (document.querySelector('style[data-ntro-style="theme"]')) {
            stylesInjected = true;
            return;
        }
        var style = document.createElement("style");
        style.setAttribute("data-ntro-style", "theme");
        style.textContent = CSS;
        head.appendChild(style);
        stylesInjected = true;
    }

    // --- 7. Public API -----------------------------------------------------------

    function getPreference() {
        if (state.preference === null) state.preference = readStored();
        return state.preference;
    }

    function getResolved() {
        if (state.resolved === null) return resolveMode(getPreference());
        return state.resolved;
    }

    function setPreference(mode) {
        var pref = normalisePref(mode);
        if (!pref) {
            warn("set() ignored an unknown mode: " + String(mode));
            return;
        }
        injectStyles();
        ensureMediaWatcher();
        writeStored(pref);
        apply(pref, { guard: true });
    }

    function toggle() {
        var current = getPreference();
        setPreference((MODE_META[current] || MODE_META.system).next);
        return state.preference;
    }

    /**
     * @param {Object} [options] { toggleSelector: String|null }
     *   omitted -> the default "[data-ntro-theme-toggle]"; null -> bind nothing.
     */
    function init(options) {
        var opts = options || {};
        injectStyles();
        ensureMediaWatcher();
        ensureStorageWatcher();
        getPreference();   // resolve before binding, so a toggle's first paint is already correct

        var selector = opts.toggleSelector === undefined ? DEFAULT_SELECTOR : opts.toggleSelector;
        if (typeof selector === "string" && selector) {
            var matched = bindToggles(selector);
            // init() may legitimately run before the toggle exists; try once more when it can.
            if (matched === 0 && document.readyState === "loading") {
                document.addEventListener("DOMContentLoaded", function once() {
                    document.removeEventListener("DOMContentLoaded", once);
                    bindToggles(selector);
                    decorateAll();
                });
            }
        }

        state.initialised = true;
        // force: subscribers registered before init() still hear the initial resolution.
        return apply(readStored(), { force: true });
    }

    NTRO.theme = {
        init: init,
        get: getPreference,
        resolved: getResolved,
        set: setPreference,
        toggle: toggle,
        onChange: onChange,
        applyStoredEarly: applyStoredEarly
    };
})(window, document);
