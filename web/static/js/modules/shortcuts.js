/**
 * NTRO-SRM - Keyboard Shortcuts
 * Central keybinding registry, dispatcher and the "Keyboard shortcuts" help dialog.
 */
(function (window, document) {
    "use strict";

    var NTRO = window.NTRO = window.NTRO || {};

    var STYLE_KEY = "shortcuts";
    var DEFAULT_GROUP = "General";
    var CLOSE_MS = 170;
    var IS_APPLE = /Mac|iPhone|iPad|iPod/i.test(
        (window.navigator && (window.navigator.platform || window.navigator.userAgent)) || ""
    );

    var registry = Object.create(null);   // normalised combo -> spec
    var order = [];                       // combos in registration order
    var enabled = true;
    var listening = false;
    var helpOpen = false;
    var lastFocused = null;
    var closeTimer = null;
    var dom = null;                       // { root, dialog, body, count, close }

    function hasOwn(obj, key) {
        return Object.prototype.hasOwnProperty.call(obj, key);
    }

    // =====================================================================
    // 1. Combo normalisation
    // =====================================================================

    var MOD_ALIASES = {
        ctrl: "ctrl", control: "ctrl", cmd: "ctrl", command: "ctrl", meta: "ctrl", mod: "ctrl",
        alt: "alt", option: "alt", opt: "alt",
        shift: "shift"
    };

    var KEY_ALIASES = {
        spacebar: "space", esc: "escape", "return": "enter",
        del: "delete", ins: "insert", pgup: "pageup", pgdn: "pagedown",
        left: "arrowleft", right: "arrowright", up: "arrowup", down: "arrowdown",
        plus: "+", minus: "-", question: "?", slash: "/"
    };

    /** Split "ctrl++" and "shift+/" safely: a trailing "+" is the literal plus key. */
    function tokenise(raw) {
        if (raw === "+") return ["+"];
        if (raw.length > 1 && raw.charAt(raw.length - 1) === "+") {
            return raw.slice(0, -1).split("+").concat("+");
        }
        return raw.split("+");
    }

    /** "Shift+D" -> "shift+d". Modifiers are always emitted as ctrl+alt+shift+key. */
    function normalizeCombo(input) {
        if (typeof input !== "string") return "";
        // The space key may be given as " " itself, which trimming would otherwise erase.
        var raw = (input === " " ? "space" : input.trim()).toLowerCase();
        if (!raw) return "";
        var tokens = tokenise(raw);
        var ctrl = false, alt = false, shift = false, key = "";
        for (var i = 0; i < tokens.length; i++) {
            var t = tokens[i].trim();
            if (!t) continue;
            // A modifier name in the final slot is the key itself, e.g. "shift" alone.
            if (hasOwn(MOD_ALIASES, t) && i < tokens.length - 1) {
                var m = MOD_ALIASES[t];
                if (m === "ctrl") ctrl = true;
                else if (m === "alt") alt = true;
                else shift = true;
            } else {
                key = hasOwn(KEY_ALIASES, t) ? KEY_ALIASES[t] : t;
            }
        }
        if (!key) return "";
        return (ctrl ? "ctrl+" : "") + (alt ? "alt+" : "") + (shift ? "shift+" : "") + key;
    }

    function eventKey(e) {
        var k = e.key;
        if (typeof k !== "string" || !k) return "";
        return k === " " ? "space" : k.toLowerCase();
    }

    /** metaKey folds into ctrl so a single binding covers Windows and macOS. */
    function eventCombo(e, withShift) {
        var k = eventKey(e);
        if (!k) return "";
        return ((e.ctrlKey || e.metaKey) ? "ctrl+" : "") +
               (e.altKey ? "alt+" : "") +
               (withShift && e.shiftKey ? "shift+" : "") + k;
    }

    /** "?" and friends already carry shift on most layouts, so shift is optional for them. */
    function isPrintablePunctuation(k) {
        return k.length === 1 && !/[a-z0-9]/.test(k);
    }

    function resolve(e) {
        var exact = eventCombo(e, true);
        if (exact && hasOwn(registry, exact)) return registry[exact];
        if (e.shiftKey && isPrintablePunctuation(eventKey(e))) {
            var loose = eventCombo(e, false);
            if (loose && hasOwn(registry, loose)) return registry[loose];
        }
        return null;
    }

    // =====================================================================
    // 2. Dispatch
    // =====================================================================

    function isEditableTarget(el) {
        if (!el || el.nodeType !== 1) return false;
        if (el.isContentEditable) return true;
        var tag = el.tagName;
        return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
    }

    /** Any open aria-modal dialog blocks non-Escape keys, ours or another module's. */
    function isModalOpen() {
        if (helpOpen) return true;
        var nodes = document.querySelectorAll('[role="dialog"][aria-modal="true"]');
        for (var i = 0; i < nodes.length; i++) {
            var n = nodes[i];
            if (dom && n === dom.dialog) continue;
            if (!n.hidden && n.getClientRects().length > 0) return true;
        }
        return false;
    }

    /** Space/Enter on a control belongs to that control, not to a shortcut. */
    function isNativeActivation(spec, e) {
        if (spec.combo !== "space" && spec.combo !== "enter") return false;
        var t = e.target;
        if (!t || t.nodeType !== 1) return false;
        var tag = t.tagName;
        return tag === "BUTTON" || tag === "A" || tag === "SUMMARY" ||
               t.getAttribute("role") === "button";
    }

    function onKeydown(e) {
        if (e.defaultPrevented) return;
        if (e.isComposing || e.keyCode === 229) return;   // mid IME composition

        // Escape closes the help dialog even when focus has slipped outside it
        // (clicking the scrim parks focus on <body>, which bypasses the dialog listener).
        if (helpOpen && eventKey(e) === "escape") {
            e.preventDefault();
            hideHelp();
            return;
        }
        if (!enabled) return;

        var spec = resolve(e);
        if (!spec) return;

        if (spec.combo !== "escape" && !spec.allowInInput) {
            if (isEditableTarget(e.target)) return;
            if (isModalOpen()) return;
            if (isNativeActivation(spec, e)) return;
        }

        if (spec.preventDefault) e.preventDefault();
        try {
            spec.handler(e);
        } catch (err) {
            if (window.console && window.console.error) {
                window.console.error('[NTRO.shortcuts] handler for "' + spec.combo + '" threw:', err);
            }
            if (NTRO.toast && typeof NTRO.toast.error === "function") {
                try {
                    NTRO.toast.error("The action bound to " + spec.combo + " could not be completed.",
                        { title: "Shortcut failed" });
                } catch (ignored) { /* a broken toast must never break the dispatcher */ }
            }
        }
    }

    // =====================================================================
    // 3. Registry API
    // =====================================================================

    function register(combo, spec) {
        var key = normalizeCombo(combo);
        if (!key || !spec || typeof spec.handler !== "function") return false;
        if (!hasOwn(registry, key)) order.push(key);
        registry[key] = {
            combo: key,
            description: String(spec.description == null ? key : spec.description),
            group: String(spec.group == null || spec.group === "" ? DEFAULT_GROUP : spec.group),
            handler: spec.handler,
            allowInInput: spec.allowInInput === true,
            preventDefault: spec.preventDefault !== false
        };
        if (helpOpen) renderBody();
        return true;
    }

    function unregister(combo) {
        var key = normalizeCombo(combo);
        if (!key || !hasOwn(registry, key)) return false;
        delete registry[key];
        var i = order.indexOf(key);
        if (i !== -1) order.splice(i, 1);
        if (helpOpen) renderBody();
        return true;
    }

    function list() {
        var out = [];
        for (var i = 0; i < order.length; i++) {
            var s = registry[order[i]];
            if (s) out.push({ combo: s.combo, description: s.description, group: s.group });
        }
        return out;
    }

    // =====================================================================
    // 4. Styles
    // =====================================================================

    var CSS = `
.ntro-sc-scrim{position:fixed;inset:0;z-index:var(--z-modal,9500);display:flex;align-items:center;justify-content:center;padding:1.5rem;background:var(--overlay-scrim,rgba(15,23,42,.55));-webkit-backdrop-filter:blur(2px);backdrop-filter:blur(2px);opacity:0;transition:opacity .16s ease;font-family:var(--font-family,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif)}
.ntro-sc-scrim[hidden]{display:none}
.ntro-sc-scrim:not(.is-open){pointer-events:none}
.ntro-sc-scrim.is-open{opacity:1}
.ntro-sc-scrim :focus-visible{outline:2px solid var(--border-focus,#3b82f6);outline-offset:2px}
.ntro-sc-dialog{display:flex;flex-direction:column;width:100%;max-width:640px;max-height:80vh;overflow:hidden;background:var(--bg-panel,#ffffff);color:var(--text-primary,#0f172a);border:1px solid var(--border-subtle,#e2e8f0);border-radius:var(--radius-lg,12px);box-shadow:var(--shadow-floating,0 8px 24px -4px rgba(15,23,42,.18));transform:translateY(6px) scale(.99);transition:transform .16s cubic-bezier(.2,.7,.3,1)}
.ntro-sc-scrim.is-open .ntro-sc-dialog{transform:none}
.ntro-sc-dialog:focus,.ntro-sc-body:focus{outline:none}
.ntro-sc-head{flex:none;display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;padding:.8rem 1rem;background:var(--bg-subtle,#f1f5f9);border-bottom:1px solid var(--border-subtle,#e2e8f0)}
.ntro-sc-title{margin:0;font-size:.95rem;font-weight:800;letter-spacing:-.2px;color:var(--text-primary,#0f172a)}
.ntro-sc-sub{margin:.15rem 0 0;font-size:.7rem;letter-spacing:.2px;color:var(--text-muted,#64748b)}
.ntro-sc-close{flex:none;display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;padding:0;cursor:pointer;color:var(--text-muted,#64748b);background:var(--bg-panel,#ffffff);border:1px solid var(--border-subtle,#e2e8f0);border-radius:var(--radius-sm,6px);transition:background .15s ease,color .15s ease,border-color .15s ease}
.ntro-sc-close:hover{background:var(--bg-hover,#e2e8f0);color:var(--text-primary,#0f172a);border-color:var(--border-medium,#cbd5e1)}
.ntro-sc-body{flex:1 1 auto;overflow-y:auto;overscroll-behavior:contain;padding:0 1rem 1rem}
.ntro-sc-body::-webkit-scrollbar{width:10px}
.ntro-sc-body::-webkit-scrollbar-thumb{background:var(--border-medium,#cbd5e1);border-radius:9999px;border:3px solid transparent;background-clip:content-box}
.ntro-sc-gtitle{position:sticky;top:0;z-index:1;margin:0;padding:.85rem 0 .3rem;font-size:.66rem;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:var(--text-muted,#64748b);background:var(--bg-panel,#ffffff);border-bottom:1px solid var(--border-subtle,#e2e8f0)}
.ntro-sc-row{display:grid;grid-template-columns:1fr auto;align-items:center;gap:1rem;padding:.34rem .45rem;border-radius:var(--radius-sm,6px)}
.ntro-sc-row:hover{background:var(--bg-subtle,#f1f5f9)}
.ntro-sc-desc{font-size:.8rem;line-height:1.35;color:var(--text-secondary,#475569)}
.ntro-sc-keys{display:inline-flex;align-items:center;gap:3px;white-space:nowrap;justify-self:end}
.ntro-sc-kbd{display:inline-block;min-width:20px;padding:2px 6px;text-align:center;font-family:var(--font-mono,SFMono-Regular,Menlo,Consolas,monospace);font-size:11px;font-weight:600;line-height:1.45;color:var(--text-primary,#0f172a);background:var(--bg-subtle,#f1f5f9);border:1px solid var(--border-medium,#cbd5e1);border-radius:4px;box-shadow:inset 0 -1px 0 rgba(15,23,42,.10)}
[data-theme="dark"] .ntro-sc-kbd{box-shadow:inset 0 -1px 0 rgba(0,0,0,.4)}
.ntro-sc-plus{padding:0 1px;font-size:10px;line-height:1;color:var(--text-muted,#64748b)}
.ntro-sc-note{margin:1.25rem 0 .35rem;padding:1.4rem 1rem;text-align:center;font-size:.78rem;line-height:1.5;color:var(--text-muted,#64748b);background:var(--bg-subtle,#f1f5f9);border:1px dashed var(--border-medium,#cbd5e1);border-radius:var(--radius-md,8px)}
.ntro-sc-note b{display:block;margin-bottom:.2rem;font-size:.82rem;font-weight:700;color:var(--text-secondary,#475569)}
.ntro-sc-note.is-error{color:var(--accent-red,#dc2626);background:var(--accent-red-bg,#fef2f2);border-color:var(--accent-red,#dc2626)}
.ntro-sc-note.is-error b{color:var(--accent-red,#dc2626)}
.ntro-sc-foot{flex:none;display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:.5rem 1rem;font-size:.7rem;color:var(--text-muted,#64748b);background:var(--bg-subtle,#f1f5f9);border-top:1px solid var(--border-subtle,#e2e8f0)}
.ntro-sc-foot .ntro-sc-kbd{background:var(--bg-panel,#ffffff)}
@media (max-width:640px){.ntro-sc-scrim{padding:.75rem}.ntro-sc-dialog{max-height:88vh}.ntro-sc-row{grid-template-columns:1fr;gap:.15rem}.ntro-sc-keys{justify-self:start}.ntro-sc-foot{display:none}}
@media (prefers-reduced-motion:reduce){.ntro-sc-scrim,.ntro-sc-dialog,.ntro-sc-close{transition:none!important}.ntro-sc-dialog{transform:none!important}}
`;

    function injectStyles() {
        if (!document.head) return;
        if (document.querySelector('style[data-ntro-style="' + STYLE_KEY + '"]')) return;
        var style = document.createElement("style");
        style.setAttribute("data-ntro-style", STYLE_KEY);
        style.textContent = CSS;
        document.head.appendChild(style);
    }

    // =====================================================================
    // 5. Help dialog
    // =====================================================================

    var LABELS = {
        ctrl: IS_APPLE ? "⌘" : "Ctrl",
        alt: IS_APPLE ? "⌥" : "Alt",
        shift: IS_APPLE ? "⇧" : "Shift",
        escape: "Esc", space: "Space", enter: "Enter", tab: "Tab",
        backspace: "⌫", "delete": "Del", insert: "Ins",
        arrowleft: "←", arrowright: "→", arrowup: "↑", arrowdown: "↓",
        pageup: "PgUp", pagedown: "PgDn", home: "Home", end: "End"
    };

    var CLOSE_SVG = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"' +
        ' stroke-width="2.5" stroke-linecap="round" aria-hidden="true" focusable="false">' +
        '<line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>';

    function el(tag, cls, text) {
        var n = document.createElement(tag);
        if (cls) n.className = cls;
        if (text != null) n.textContent = text;
        return n;
    }

    function labelFor(token) {
        if (hasOwn(LABELS, token)) return LABELS[token];
        return token.length === 1
            ? token.toUpperCase()
            : token.charAt(0).toUpperCase() + token.slice(1);
    }

    function keysFor(combo) {
        var wrap = el("span", "ntro-sc-keys");
        var parts = tokenise(combo);
        for (var i = 0; i < parts.length; i++) {
            if (i) wrap.appendChild(el("span", "ntro-sc-plus", "+"));
            wrap.appendChild(el("kbd", "ntro-sc-kbd", labelFor(parts[i])));
        }
        return wrap;
    }

    function note(title, detail, isError) {
        var box = el("div", "ntro-sc-note" + (isError ? " is-error" : ""));
        box.appendChild(el("b", null, title));
        box.appendChild(document.createTextNode(detail));
        return box;
    }

    function renderBody() {
        if (!dom) return;
        var body = dom.body;
        body.textContent = "";
        var items = list();
        dom.count.textContent = items.length === 1
            ? "1 binding registered"
            : items.length + " bindings registered";

        if (!items.length) {
            body.appendChild(note("No shortcuts registered",
                "Bindings appear here as the map, comparison and inspector modules initialise."));
            return;
        }

        try {
            var names = [];
            var buckets = Object.create(null);
            items.forEach(function (item) {
                if (!hasOwn(buckets, item.group)) {
                    buckets[item.group] = [];
                    names.push(item.group);
                }
                buckets[item.group].push(item);
            });

            names.forEach(function (name, gi) {
                var titleId = "ntro-sc-g" + gi;
                var section = el("section", "ntro-sc-group");
                section.setAttribute("role", "group");
                section.setAttribute("aria-labelledby", titleId);
                var heading = el("h3", "ntro-sc-gtitle", name);
                heading.id = titleId;
                section.appendChild(heading);
                buckets[name].forEach(function (item) {
                    var row = el("div", "ntro-sc-row");
                    row.appendChild(el("span", "ntro-sc-desc", item.description));
                    row.appendChild(keysFor(item.combo));
                    section.appendChild(row);
                });
                body.appendChild(section);
            });
        } catch (err) {
            if (window.console && window.console.error) {
                window.console.error("[NTRO.shortcuts] could not render the shortcut list:", err);
            }
            body.textContent = "";
            body.appendChild(note("Shortcut list unavailable",
                "The bindings are still active — only this listing failed to render.", true));
        }
    }

    function focusable() {
        if (!dom) return [];
        var nodes = dom.dialog.querySelectorAll(
            'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),' +
            'textarea:not([disabled]),[tabindex]:not([tabindex="-1"])');
        var out = [];
        for (var i = 0; i < nodes.length; i++) {
            if (nodes[i].getClientRects().length > 0) out.push(nodes[i]);
        }
        return out;
    }

    function onModalKeydown(e) {
        // The dialog owns Escape, and a second "?", while it is open.
        if (e.key === "Escape" || (e.key === "?" && !isEditableTarget(e.target))) {
            e.preventDefault();
            e.stopPropagation();
            hideHelp();
            return;
        }
        if (e.key !== "Tab") return;

        var items = focusable();
        if (!items.length) {
            e.preventDefault();
            dom.dialog.focus();
            return;
        }
        var first = items[0];
        var last = items[items.length - 1];
        var active = document.activeElement;
        var inside = dom.dialog.contains(active);
        if (e.shiftKey && (!inside || active === first)) {
            e.preventDefault();
            last.focus();
        } else if (!e.shiftKey && (!inside || active === last)) {
            e.preventDefault();
            first.focus();
        }
    }

    /** Backstop for focus that escapes the dialog by any route other than Tab. */
    function onDocFocusIn(e) {
        if (!helpOpen || !dom || dom.dialog.contains(e.target)) return;
        var items = focusable();
        (items[0] || dom.dialog).focus();
    }

    function ensureModal() {
        if (dom) return dom;
        if (!document.body) return null;
        injectStyles();

        var root = el("div", "ntro-sc-scrim");
        root.hidden = true;

        var dialog = el("div", "ntro-sc-dialog");
        dialog.setAttribute("role", "dialog");
        dialog.setAttribute("aria-modal", "true");
        dialog.setAttribute("aria-labelledby", "ntro-sc-title");
        dialog.tabIndex = -1;

        var title = el("h2", "ntro-sc-title", "Keyboard shortcuts");
        title.id = "ntro-sc-title";
        var count = el("p", "ntro-sc-sub", "");
        var headText = el("div");
        headText.appendChild(title);
        headText.appendChild(count);

        var close = el("button", "ntro-sc-close");
        close.type = "button";
        close.setAttribute("aria-label", "Close keyboard shortcuts");
        close.title = "Close (Esc)";
        close.innerHTML = CLOSE_SVG;
        close.addEventListener("click", hideHelp);

        var head = el("header", "ntro-sc-head");
        head.appendChild(headText);
        head.appendChild(close);

        var body = el("div", "ntro-sc-body");
        body.tabIndex = 0;                 // a scrollable region must be keyboard reachable

        var footLeft = el("span");
        footLeft.appendChild(el("kbd", "ntro-sc-kbd", "Esc"));
        footLeft.appendChild(document.createTextNode(" closes this dialog"));
        var foot = el("footer", "ntro-sc-foot");
        foot.appendChild(footLeft);
        foot.appendChild(el("span", null,
            "Shortcuts pause while you type in a field · Ctrl and Cmd are interchangeable"));

        dialog.appendChild(head);
        dialog.appendChild(body);
        dialog.appendChild(foot);
        root.appendChild(dialog);

        root.addEventListener("keydown", onModalKeydown);
        root.addEventListener("click", function (e) {
            if (e.target === root) hideHelp();
        });

        document.body.appendChild(root);
        dom = { root: root, dialog: dialog, body: body, count: count, close: close };
        return dom;
    }

    function showHelp() {
        var m = ensureModal();
        if (!m || helpOpen) return;
        if (closeTimer) {
            window.clearTimeout(closeTimer);
            closeTimer = null;
        }

        lastFocused = document.activeElement;
        renderBody();
        m.root.hidden = false;
        helpOpen = true;
        void m.dialog.offsetWidth;         // flush layout so the open transition runs
        m.root.classList.add("is-open");
        document.addEventListener("focusin", onDocFocusIn, true);
        try { m.close.focus(); } catch (err) { /* focus is best effort */ }
    }

    function hideHelp() {
        if (!dom || !helpOpen) return;
        helpOpen = false;
        dom.root.classList.remove("is-open");
        document.removeEventListener("focusin", onDocFocusIn, true);
        if (closeTimer) window.clearTimeout(closeTimer);
        closeTimer = window.setTimeout(function () {
            closeTimer = null;
            if (!helpOpen && dom) dom.root.hidden = true;
        }, CLOSE_MS);

        var target = lastFocused;
        lastFocused = null;
        if (target && typeof target.focus === "function" && document.contains(target)) {
            try { target.focus(); } catch (err) { /* the trigger may have been removed */ }
        }
    }

    function toggleHelp() {
        if (helpOpen) hideHelp();
        else showHelp();
    }

    // =====================================================================
    // 6. Lifecycle
    // =====================================================================

    function init() {
        if (listening) return;
        injectStyles();
        document.addEventListener("keydown", onKeydown, false);
        listening = true;
    }

    function setEnabled(value) {
        enabled = !!value;
    }

    NTRO.shortcuts = {
        init: init,
        register: register,
        unregister: unregister,
        list: list,
        showHelp: showHelp,
        hideHelp: hideHelp,
        toggleHelp: toggleHelp,
        setEnabled: setEnabled,
        isModalOpen: isModalOpen
    };
})(window, document);
