/**
 * NTRO-SRM Sentinel-2 Super-Resolution Mapping Web UI
 * Interactive Leaflet Map, AOI Manager, STAC Client, Split-Slider Visualizer,
 * Layer/index picker, quality-assessment workspace, spectral probe,
 * 1-Click Auto-Patch Upscaling, and Direct GeoTIFF Upload.
 */

(function () {
    "use strict";

    var NTRO = window.NTRO = window.NTRO || {};

    // Application State
    const state = {
        map: null,
        baseLayer: null,
        aoiLayer: null,
        patchOutline: null,
        isDrawingAoi: false,
        aoiStartLatLng: null,
        currentAoi: null, // { min_lon, min_lat, max_lon, max_lat }
        selectedScene: null,
        selectedModel: "lite", // "lite" | "swin2sr"
        activeJobId: null,
        jobPollTimer: null,
        jobStartedAt: 0,
        jobTimerHandle: null,
        activeLayer: "rgb",       // composite | index | analysis key
        layerKind: "composite",   // "composite" | "index" | "analysis"
        leftCompareMode: "lr",    // "lr" | "bicubic"
        viewMode: "split",        // "split" | "lr_only" | "sr_only" | "blend"
        srOpacity: 100,           // 0 to 100
        sliderPosRatio: 0.5,      // 0.0 to 1.0
        isDraggingSlider: false,
        layers: {
            left: null,
            right: null,
        },
        jobResult: null,
        analysis: null,
        uploadedFile: null,
        catalog: null,            // /api/layers payload
        inspectArmed: false,
        probeMarker: null,
        historyOpen: false,
        sidebarCollapsed: false,
        lastFocused: null,
    };

    // DOM Element References (initialized on start)
    let elements = {};

    function $(id) { return document.getElementById(id); }

    function getElements() {
        return {
            map: $("map"),
            mapViewport: $("map-viewport"),
            appMain: $("app-main"),
            drawBanner: $("draw-banner"),
            btnCancelDraw: $("btn-cancel-draw"),
            btnLoadDemo: $("btn-load-demo"),
            btnDrawAoi: $("btn-draw-aoi"),
            btnCenterPatch: $("btn-center-patch"),
            btnClearAoi: $("btn-clear-aoi"),
            btnSearchScenes: $("btn-search-scenes"),
            btnRunSr: $("btn-run-sr"),
            selectLayer: $("select-layer"),
            layerHint: $("layer-hint"),
            selectLeftLayer: $("select-left-layer"),
            comparisonToggleGroup: $("comparison-toggle-group"),
            viewModeGroup: $("view-mode-group"),
            btnViewSplit: $("btn-view-split"),
            btnViewLr: $("btn-view-lr"),
            btnViewSr: $("btn-view-sr"),
            btnViewBlend: $("btn-view-blend"),
            blendSliderGroup: $("blend-slider-group"),
            srOpacitySlider: $("sr-opacity-slider"),
            srOpacityVal: $("sr-opacity-val"),
            btnZoomPatch: $("btn-zoom-patch"),
            btnLoupe: $("btn-loupe"),
            btnInspect: $("btn-inspect"),
            dateFrom: $("date-from"),
            dateTo: $("date-to"),
            cloudCover: $("cloud-cover"),
            cloudVal: $("cloud-val"),
            aoiBadge: $("aoi-badge"),
            aoiCenter: $("aoi-center"),
            aoiDims: $("aoi-dims"),
            aoiPixels: $("aoi-pixels"),
            aoiSrPixels: $("aoi-sr-pixels"),
            aoiWarning: $("aoi-warning"),
            coordDisplay: $("coord-display"),
            aoiDisplayStatus: $("aoi-display-status"),
            searchResultsContainer: $("search-results-container"),
            searchPlaceholder: $("search-placeholder"),
            searchLoading: $("search-loading"),
            scenesList: $("scenes-list"),
            selectedSceneCard: $("selected-scene-card"),
            selectedSceneId: $("selected-scene-id"),
            selectedSceneMeta: $("selected-scene-meta"),
            progressCard: $("progress-card"),
            progressFill: $("progress-fill"),
            progressStepText: $("progress-step-text"),
            progressPercentText: $("progress-percent-text"),
            progressTimer: $("progress-timer"),
            resultsCard: $("results-card"),
            resCrs: $("res-crs"),
            resTime: $("res-time"),
            resDevice: $("res-device"),
            resModel: $("res-model"),
            resVram: $("res-vram"),
            resGrid: $("res-grid"),
            btnDownloadGeotiff: $("btn-download-geotiff"),
            btnDownloadNative: $("btn-download-native"),
            btnDownloadConfidence: $("btn-download-confidence"),
            btnDownloadRgb: $("btn-download-rgb"),
            btnDownloadCir: $("btn-download-cir"),
            btnDownloadReport: $("btn-download-report"),
            btnDownloadReportMd: $("btn-download-report-md"),
            sliderContainer: $("slider-container"),
            sliderDivider: $("slider-divider"),
            compareLabels: $("compare-labels"),
            labelLeftText: $("label-left-text"),
            labelRightText: $("label-right-text"),

            // Legend
            mapLegend: $("map-legend"),
            legendTitle: $("legend-title"),
            legendKey: $("legend-key"),
            legendFormula: $("legend-formula"),
            legendGradient: $("legend-gradient"),
            legendClasses: $("legend-classes"),

            // Result workspace
            resultTabs: document.querySelectorAll(".result-tab"),
            qualityContent: $("quality-content"),
            appsContent: $("apps-content"),
            probeContent: $("probe-content"),

            // Analysis options
            optAnalysis: $("opt-analysis"),
            optWald: $("opt-wald"),
            optEnsemble: $("opt-ensemble"),

            // History drawer
            btnHistory: $("btn-history"),
            btnHistoryClose: $("btn-history-close"),
            historyDrawer: $("history-drawer"),
            historyScrim: $("history-scrim"),
            historyBody: $("history-body"),

            // Header controls
            btnTheme: $("btn-theme"),
            btnHelp: $("btn-help"),
            btnSidebarToggle: $("btn-sidebar-toggle"),
            sidebarPanel: $("sidebar-panel"),

            // Tabs
            tabBtns: document.querySelectorAll(".tab-btn"),
            tabContents: document.querySelectorAll(".tab-content"),

            // Upload
            uploadDropzone: $("upload-dropzone"),
            fileUploadInput: $("file-upload-input"),
            uploadFileCard: $("upload-file-card"),
            uploadFilename: $("upload-filename"),
            uploadFileSize: $("upload-file-size"),
            uploadFileDims: $("upload-file-dims"),
            uploadFileBands: $("upload-file-bands"),
            uploadFileCrs: $("upload-file-crs"),
            btnRunUploadSr: $("btn-run-upload-sr"),

            // Manual Coordinates
            coordMinLon: $("coord-min-lon"),
            coordMinLat: $("coord-min-lat"),
            coordMaxLon: $("coord-max-lon"),
            coordMaxLat: $("coord-max-lat"),
            btnApplyCoords: $("btn-apply-coords"),
        };
    }

    // =========================================================================
    // 0. Small utilities
    // =========================================================================

    /** Non-blocking notification, degrading to console when the module is absent. */
    function notify(type, message, options) {
        if (NTRO.toast && typeof NTRO.toast[type] === "function") {
            return NTRO.toast[type](message, options || {});
        }
        console[type === "error" ? "error" : "log"]("[NTRO-SRM] " + message);
        return null;
    }

    function fmt(value, digits) {
        if (value === null || value === undefined || !isFinite(value)) return "—";
        var d = digits === undefined ? 3 : digits;
        return Number(value).toFixed(d);
    }

    function escapeHtml(text) {
        return String(text === null || text === undefined ? "" : text)
            .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    }

    function cssVar(name, fallback) {
        try {
            var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
            return v || fallback;
        } catch (e) {
            return fallback;
        }
    }

    /** Rating of a metric value against its METRIC_META thresholds. */
    function rateMetric(value, meta) {
        if (!meta || value === null || value === undefined || !isFinite(value)) return "fair";
        if (meta.better === "lower") {
            if (value <= meta.excellent) return "excellent";
            if (value <= meta.good) return "good";
            return "fair";
        }
        if (value >= meta.excellent) return "excellent";
        if (value >= meta.good) return "good";
        return "fair";
    }

    function metricMeta(key) {
        var meta = state.catalog && state.catalog.metric_meta;
        return (meta && meta[key]) || null;
    }

    // =========================================================================
    // 1. Leaflet Map Initialization (Google Maps Default + Fast Tile Endpoints)
    // =========================================================================
    function initMap() {
        if (state.map) return;

        // 1. Google Maps Streets (DEFAULT: The real Google Maps!)
        const googleRoads = L.tileLayer(
            "https://mt{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
            {
                subdomains: ["0", "1", "2", "3"],
                attribution: "&copy; Google Maps",
                maxZoom: 20,
            }
        );

        // 2. Google Maps Satellite / Hybrid (Satellite imagery + Roads & Labels)
        const googleHybrid = L.tileLayer(
            "https://mt{s}.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
            {
                subdomains: ["0", "1", "2", "3"],
                attribution: "&copy; Google Satellite",
                maxZoom: 20,
            }
        );

        // 3. Google Maps Terrain
        const googleTerrain = L.tileLayer(
            "https://mt{s}.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
            {
                subdomains: ["0", "1", "2", "3"],
                attribution: "&copy; Google Terrain",
                maxZoom: 20,
            }
        );

        // 4. CartoDB Voyager (Modern clean pastel street map)
        const cartoVoyager = L.tileLayer(
            "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
            {
                subdomains: "abcd",
                attribution: '&copy; CARTO &copy; OpenStreetMap',
                maxZoom: 20,
            }
        );

        // 5. CartoDB Dark Matter (pairs with the dark UI theme)
        const cartoDark = L.tileLayer(
            "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
            {
                subdomains: "abcd",
                attribution: '&copy; CARTO &copy; OpenStreetMap',
                maxZoom: 20,
            }
        );

        // 6. OpenStreetMap Standard
        const osmStreets = L.tileLayer(
            "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            {
                subdomains: ["a", "b", "c"],
                attribution: '&copy; OpenStreetMap contributors',
                maxZoom: 19,
            }
        );

        // Initialize Map centered at Mountain Lake Biological Station, VA
        state.map = L.map("map", {
            center: [37.4255, -80.5723],
            zoom: 14,
            zoomControl: true,
            layers: [googleRoads], // DEFAULT: Real Google Maps Streets
        });

        state.baseLayer = googleRoads;

        // Base Layer Switcher Control
        const baseMaps = {
            "🗺️ Google Maps (Streets)": googleRoads,
            "🛰️ Google Satellite (Hybrid)": googleHybrid,
            "🏔️ Google Terrain": googleTerrain,
            "🎨 CartoDB Voyager": cartoVoyager,
            "🌑 CartoDB Dark Matter": cartoDark,
            "🌐 OpenStreetMap": osmStreets,
        };

        L.control.layers(baseMaps, null, { position: "topright" }).addTo(state.map);
        state.map.on("baselayerchange", function (e) { state.baseLayer = e.layer; refreshLoupe(); });

        // Mouse Coordinates Tracker
        state.map.on("mousemove", (e) => {
            const lat = e.latlng.lat.toFixed(4);
            const lon = e.latlng.lng.toFixed(4);
            const zoom = state.map.getZoom();
            if (elements.coordDisplay) {
                elements.coordDisplay.textContent = `Lat: ${lat}, Lon: ${lon} • Zoom: ${zoom}`;
            }
        });

        // Spectral probe: a click while the probe is armed samples that pixel.
        state.map.on("click", (e) => {
            if (!state.inspectArmed || state.isDrawingAoi) return;
            probePixel(e.latlng.lat, e.latlng.lng);
        });

        // Set default patch at initial view (Mountain Lake)
        const initLat = 37.4255;
        const initLon = -80.5723;
        const deltaLat = (1.5 / 110.574) / 2;
        const deltaLon = (1.5 / (111.32 * Math.cos((initLat * Math.PI) / 180))) / 2;
        const initBounds = L.latLngBounds(
            [initLat - deltaLat, initLon - deltaLon],
            [initLat + deltaLat, initLon + deltaLon]
        );
        setAoiFromBounds(initBounds);

        // Setup Box Drawing Events
        setupDrawingHandlers();

        // Invalidate size once map is attached
        setTimeout(() => {
            if (state.map) state.map.invalidateSize();
        }, 200);
    }

    // =========================================================================
    // 2. Quick Location Selection (Handles both floating bar & sidebar chips)
    // =========================================================================
    function selectQuickLocation(lat, lon, name) {
        if (!state.map) return;
        clearOverlays();

        const deltaLat = (1.5 / 110.574) / 2;
        const deltaLon = (1.5 / (111.32 * Math.cos((lat * Math.PI) / 180))) / 2;
        const bounds = L.latLngBounds(
            [lat - deltaLat, lon - deltaLon],
            [lat + deltaLat, lon + deltaLon]
        );

        // Update active class on all matching preset chips
        document.querySelectorAll(".preset-chip").forEach((el) => {
            const elLat = parseFloat(el.getAttribute("data-lat"));
            const elLon = parseFloat(el.getAttribute("data-lon"));
            const isMatch = Math.abs(elLat - lat) < 0.001 && Math.abs(elLon - lon) < 0.001;
            el.classList.toggle("active", isMatch);
        });

        // Pan and fit map to new location
        state.map.setView([lat, lon], 14, { animate: true });
        setAoiFromBounds(bounds);

        if (elements.aoiBadge) {
            elements.aoiBadge.textContent = name || "Preset Loaded";
            elements.aoiBadge.className = "badge badge-success";
        }
        if (elements.aoiDisplayStatus) {
            elements.aoiDisplayStatus.textContent = `Location: ${name} • Ready to Upscale`;
        }

        // Switch to Tab 1 if currently on another tab
        const tabBtnSelect = $("tab-btn-select");
        if (tabBtnSelect && !tabBtnSelect.classList.contains("active")) {
            tabBtnSelect.click();
        }
    }

    function initQuickLocations() {
        // Event delegation on document so any .preset-chip anywhere is caught
        document.addEventListener("click", (e) => {
            const chip = e.target.closest(".preset-chip");
            if (!chip) return;
            e.preventDefault();
            e.stopPropagation();

            const lat = parseFloat(chip.getAttribute("data-lat"));
            const lon = parseFloat(chip.getAttribute("data-lon"));
            const name = chip.getAttribute("data-name") || chip.textContent.trim();

            if (!isNaN(lat) && !isNaN(lon)) {
                selectQuickLocation(lat, lon, name);
            }
        });
    }

    // =========================================================================
    // 3. Tab Navigation
    // =========================================================================
    function initTabs() {
        const upscaleCard = $("upscale-action-card");
        const sceneCard = $("selected-scene-card");

        elements.tabBtns.forEach((btn) => {
            btn.addEventListener("click", () => {
                elements.tabBtns.forEach((b) => {
                    b.classList.remove("active");
                    b.setAttribute("aria-selected", "false");
                });
                elements.tabContents.forEach((c) => c.classList.remove("active"));

                btn.classList.add("active");
                btn.setAttribute("aria-selected", "true");
                const tabId = btn.getAttribute("data-tab");
                const content = $(tabId);
                if (content) content.classList.add("active");

                if (tabId === "tab-upload") {
                    if (upscaleCard) upscaleCard.classList.add("hidden");
                    if (sceneCard) sceneCard.classList.add("hidden");
                } else {
                    if (upscaleCard) upscaleCard.classList.remove("hidden");
                    if (sceneCard && state.selectedScene) sceneCard.classList.remove("hidden");
                }
            });
        });
    }

    // =========================================================================
    // 4. Interactive AOI Drawing & Box Geometry
    // =========================================================================
    function setupDrawingHandlers() {
        const mapContainer = state.map.getContainer();

        if (elements.btnDrawAoi) {
            elements.btnDrawAoi.addEventListener("click", () => {
                toggleDrawMode(!state.isDrawingAoi);
            });
        }

        if (elements.btnCancelDraw) {
            elements.btnCancelDraw.addEventListener("click", () => {
                toggleDrawMode(false);
            });
        }

        mapContainer.addEventListener("mousedown", onDrawStart);
        mapContainer.addEventListener("mousemove", onDrawMove);
        window.addEventListener("mouseup", onDrawEnd);
    }

    function toggleDrawMode(active) {
        state.isDrawingAoi = active;
        const container = state.map.getContainer();

        if (active) {
            if (state.inspectArmed) setInspectArmed(false);
            container.style.cursor = "crosshair";
            state.map.dragging.disable();
            elements.btnDrawAoi.classList.add("drawing-active");
            elements.btnDrawAoi.innerHTML = `
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect></svg>
                Exit Draw Mode
            `;
            elements.drawBanner.classList.remove("hidden");
            elements.aoiDisplayStatus.textContent = "Draw mode active • Drag on map to define patch";
        } else {
            container.style.cursor = "";
            state.map.dragging.enable();
            elements.btnDrawAoi.classList.remove("drawing-active");
            elements.btnDrawAoi.innerHTML = `
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>
                Draw Box
            `;
            elements.drawBanner.classList.add("hidden");
        }
    }

    function onDrawStart(e) {
        if (!state.isDrawingAoi || e.button !== 0) return;
        const latlng = state.map.mouseEventToLatLng(e);
        state.aoiStartLatLng = latlng;

        if (state.aoiLayer) {
            state.map.removeLayer(state.aoiLayer);
            state.aoiLayer = null;
        }

        state.aoiLayer = L.rectangle([latlng, latlng], {
            color: "#2563eb",
            weight: 2.5,
            fillColor: "#3b82f6",
            fillOpacity: 0.2,
            dashArray: "4, 4",
        }).addTo(state.map);
    }

    function onDrawMove(e) {
        if (!state.isDrawingAoi || !state.aoiStartLatLng || !state.aoiLayer) return;
        const currentLatLng = state.map.mouseEventToLatLng(e);
        const bounds = L.latLngBounds(state.aoiStartLatLng, currentLatLng);
        state.aoiLayer.setBounds(bounds);
        updateAoiMetrics(bounds);
    }

    function onDrawEnd(e) {
        if (!state.isDrawingAoi || !state.aoiStartLatLng) return;
        const endLatLng = state.map.mouseEventToLatLng(e);
        const bounds = L.latLngBounds(state.aoiStartLatLng, endLatLng);
        state.aoiStartLatLng = null;

        const p1 = state.map.latLngToContainerPoint(bounds.getSouthWest());
        const p2 = state.map.latLngToContainerPoint(bounds.getNorthEast());
        const pixelDist = Math.hypot(p1.x - p2.x, p1.y - p2.y);

        if (pixelDist < 15) {
            // Click without drag: create ~1.5km x 1.5km patch centered on click
            const lat = endLatLng.lat;
            const lon = endLatLng.lng;
            const deltaLat = (1.5 / 110.574) / 2;
            const deltaLon = (1.5 / (111.32 * Math.cos((lat * Math.PI) / 180))) / 2;
            const clickBounds = L.latLngBounds(
                [lat - deltaLat, lon - deltaLon],
                [lat + deltaLat, lon + deltaLon]
            );
            setAoiFromBounds(clickBounds);
        } else {
            setAoiFromBounds(bounds);
        }

        toggleDrawMode(false);
    }

    function setAoiFromBounds(bounds) {
        if (state.aoiLayer) {
            state.map.removeLayer(state.aoiLayer);
        }

        state.aoiLayer = L.rectangle(bounds, {
            color: "#2563eb",
            weight: 2.5,
            fillColor: "#3b82f6",
            fillOpacity: 0.15,
            dashArray: null,
        }).addTo(state.map);

        const sw = bounds.getSouthWest();
        const ne = bounds.getNorthEast();

        state.currentAoi = {
            min_lon: Math.min(sw.lng, ne.lng),
            min_lat: Math.min(sw.lat, ne.lat),
            max_lon: Math.max(sw.lng, ne.lng),
            max_lat: Math.max(sw.lat, ne.lat),
        };

        // Update coordinate input fields
        if (elements.coordMinLon) elements.coordMinLon.value = state.currentAoi.min_lon.toFixed(4);
        if (elements.coordMinLat) elements.coordMinLat.value = state.currentAoi.min_lat.toFixed(4);
        if (elements.coordMaxLon) elements.coordMaxLon.value = state.currentAoi.max_lon.toFixed(4);
        if (elements.coordMaxLat) elements.coordMaxLat.value = state.currentAoi.max_lat.toFixed(4);

        updateAoiMetrics(bounds);

        if (!state.selectedScene) {
            elements.selectedSceneCard.classList.remove("hidden");
            elements.selectedSceneId.textContent = "Auto-Fetch Lowest Cloud Scene";
            elements.selectedSceneMeta.textContent = "Automatic Copernicus CDSE / AWS L2A streaming";
        }

        // Enable 1-Click Upscale immediately
        elements.btnRunSr.disabled = false;
    }

    function updateAoiMetrics(bounds) {
        const sw = bounds.getSouthWest();
        const ne = bounds.getNorthEast();

        const centerLat = (sw.lat + ne.lat) / 2;
        const centerLon = (sw.lng + ne.lng) / 2;

        const latRad = (centerLat * Math.PI) / 180;
        const widthKm = Math.abs(ne.lng - sw.lng) * 111.32 * Math.cos(latRad);
        const heightKm = Math.abs(ne.lat - sw.lat) * 110.574;
        const areaKm2 = widthKm * heightKm;

        const wPx = Math.max(1, Math.round((widthKm * 1000) / 10));
        const hPx = Math.max(1, Math.round((heightKm * 1000) / 10));
        const totalPx = wPx * hPx;

        const wSr = wPx * 4;
        const hSr = hPx * 4;

        elements.aoiBadge.textContent = "Patch Ready";
        elements.aoiBadge.className = "badge badge-success";
        elements.aoiCenter.textContent = `${centerLat.toFixed(4)}, ${centerLon.toFixed(4)}`;
        elements.aoiDims.textContent = `${widthKm.toFixed(2)} × ${heightKm.toFixed(2)} km (${areaKm2.toFixed(1)} km²)`;
        elements.aoiPixels.textContent = `${wPx} × ${hPx} px (~${totalPx.toLocaleString()} px)`;
        if (elements.aoiSrPixels) {
            elements.aoiSrPixels.textContent = `${wSr} × ${hSr} px (4× Super-Resolved)`;
        }

        const maxPx = 512 * 512;
        if (totalPx > maxPx) {
            elements.aoiWarning.textContent = `AOI has ~${totalPx.toLocaleString()} px, exceeding 512×512 GPU tile limit. Please draw a smaller patch.`;
            elements.aoiWarning.classList.remove("hidden");
            elements.btnRunSr.disabled = true;
        } else {
            elements.aoiWarning.classList.add("hidden");
            elements.btnRunSr.disabled = false;
        }

        elements.aoiDisplayStatus.textContent = `Patch: ${wPx}×${hPx} px (10m) → ${wSr}×${hSr} px (2.5m) • Ready to Upscale`;
    }

    function clearAoi() {
        if (state.aoiLayer) {
            state.map.removeLayer(state.aoiLayer);
            state.aoiLayer = null;
        }
        state.currentAoi = null;
        state.selectedScene = null;
        document.querySelectorAll(".preset-chip").forEach((c) => c.classList.remove("active"));
        elements.aoiBadge.textContent = "No Patch Selected";
        elements.aoiBadge.className = "badge badge-gray";
        elements.aoiCenter.textContent = "--";
        elements.aoiDims.textContent = "--";
        elements.aoiPixels.textContent = "--";
        if (elements.aoiSrPixels) elements.aoiSrPixels.textContent = "--";
        elements.aoiWarning.classList.add("hidden");
        elements.btnRunSr.disabled = true;
        elements.selectedSceneCard.classList.add("hidden");
        elements.aoiDisplayStatus.textContent = "Ready • Click 'Draw Box' or select a quick location to upscale";
    }

    function initAoiControls() {
        if (elements.btnClearAoi) {
            elements.btnClearAoi.addEventListener("click", clearAoi);
        }

        // "Use Map Center as Patch" Button
        if (elements.btnCenterPatch) {
            elements.btnCenterPatch.addEventListener("click", () => {
                if (!state.map) return;
                const center = state.map.getCenter();
                const lat = center.lat;
                const lon = center.lng;
                const deltaLat = (1.5 / 110.574) / 2;
                const deltaLon = (1.5 / (111.32 * Math.cos((lat * Math.PI) / 180))) / 2;
                const bounds = L.latLngBounds(
                    [lat - deltaLat, lon - deltaLon],
                    [lat + deltaLat, lon + deltaLon]
                );
                state.map.fitBounds(bounds, { padding: [60, 60] });
                setAoiFromBounds(bounds);
            });
        }

        // Apply Manual Coordinates
        if (elements.btnApplyCoords) {
            elements.btnApplyCoords.addEventListener("click", () => {
                const minLon = parseFloat(elements.coordMinLon.value);
                const minLat = parseFloat(elements.coordMinLat.value);
                const maxLon = parseFloat(elements.coordMaxLon.value);
                const maxLat = parseFloat(elements.coordMaxLat.value);

                if (isNaN(minLon) || isNaN(minLat) || isNaN(maxLon) || isNaN(maxLat)) {
                    notify("warning", "Please enter valid decimal coordinates in all four fields.");
                    return;
                }
                if (minLon >= maxLon || minLat >= maxLat) {
                    notify("warning", "Minimum coordinates must be strictly smaller than the maximum coordinates.");
                    return;
                }

                const bounds = L.latLngBounds([minLat, minLon], [maxLat, maxLon]);
                state.map.fitBounds(bounds, { padding: [40, 40] });
                setAoiFromBounds(bounds);
                notify("success", "Area of interest updated from manual coordinates.");
            });
        }
    }

    // =========================================================================
    // 5. Direct GeoTIFF File Upload
    // =========================================================================
    function initUpload() {
        if (!elements.uploadDropzone || !elements.fileUploadInput) return;

        elements.uploadDropzone.addEventListener("click", () => {
            elements.fileUploadInput.click();
        });

        elements.uploadDropzone.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                elements.fileUploadInput.click();
            }
        });

        elements.uploadDropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            elements.uploadDropzone.classList.add("drag-over");
        });

        elements.uploadDropzone.addEventListener("dragleave", () => {
            elements.uploadDropzone.classList.remove("drag-over");
        });

        elements.uploadDropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            elements.uploadDropzone.classList.remove("drag-over");
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleUploadedFile(e.dataTransfer.files[0]);
            }
        });

        elements.fileUploadInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleUploadedFile(e.target.files[0]);
            }
        });

        if (elements.btnRunUploadSr) {
            elements.btnRunUploadSr.addEventListener("click", async () => {
                if (!state.uploadedFile) return;

                elements.btnRunUploadSr.disabled = true;
                beginJobUi("Uploading GeoTIFF to the processing server...");

                const formData = new FormData();
                formData.append("file", state.uploadedFile);

                const params = new URLSearchParams(analysisQueryParams());
                params.set("model", state.selectedModel);

                try {
                    const resp = await fetch(`/api/sr/upload?${params.toString()}`, {
                        method: "POST",
                        body: formData,
                    });

                    if (!resp.ok) {
                        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
                        throw new Error(err.detail || "Failed to upload GeoTIFF.");
                    }

                    const data = await resp.json();
                    state.activeJobId = data.job_id;
                    elements.uploadFileDims.textContent = `${data.dimensions[0]} × ${data.dimensions[1]} px`;
                    elements.uploadFileBands.textContent = `${data.bands} Bands`;
                    elements.uploadFileCrs.textContent = data.crs || "Projected";

                    notify("info", `${escapeHtml(data.filename)} accepted — ${data.dimensions[0]}×${data.dimensions[1]}, ${data.bands} bands.`);
                    pollJobStatus(state.activeJobId);
                } catch (err) {
                    failJobUi(err.message);
                    elements.btnRunUploadSr.disabled = false;
                }
            });
        }
    }

    function handleUploadedFile(file) {
        if (!file.name.toLowerCase().endsWith(".tif") && !file.name.toLowerCase().endsWith(".tiff")) {
            notify("warning", "Please select a GeoTIFF image (.tif or .tiff).");
            return;
        }

        state.uploadedFile = file;
        elements.uploadFilename.textContent = file.name;
        elements.uploadFileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
        elements.uploadFileDims.textContent = "Inspecting...";
        elements.uploadFileBands.textContent = "Inspecting...";
        elements.uploadFileCrs.textContent = "--";
        elements.uploadFileCard.classList.remove("hidden");
    }

    // =========================================================================
    // 6. Sentinel-2 STAC Catalog Search (Optional / Advanced)
    // =========================================================================
    function initCatalog() {
        if (elements.cloudCover && elements.cloudVal) {
            elements.cloudCover.addEventListener("input", (e) => {
                elements.cloudVal.textContent = `${e.target.value}%`;
            });
        }

        if (elements.btnSearchScenes) {
            elements.btnSearchScenes.addEventListener("click", async () => {
                if (!state.currentAoi) {
                    notify("warning", "Define an area of interest on the map before searching the catalogue.");
                    return;
                }

                elements.searchPlaceholder.classList.add("hidden");
                elements.scenesList.classList.add("hidden");
                elements.searchLoading.classList.remove("hidden");

                const payload = {
                    aoi: state.currentAoi,
                    date_from: elements.dateFrom.value,
                    date_to: elements.dateTo.value,
                    max_cloud_cover: parseFloat(elements.cloudCover.value),
                    limit: 10,
                };

                try {
                    const resp = await fetch("/api/sentinel/search", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload),
                    });

                    if (!resp.ok) {
                        const errData = await resp.json().catch(() => ({ detail: resp.statusText }));
                        throw new Error(errData.detail || "STAC search request failed.");
                    }

                    const data = await resp.json();
                    renderSearchResults(data.scenes);
                    if (data.scenes && data.scenes.length) {
                        notify("success", `${data.scenes.length} Sentinel-2 L2A acquisitions found.`);
                    }
                } catch (err) {
                    elements.searchLoading.classList.add("hidden");
                    elements.searchPlaceholder.classList.remove("hidden");
                    elements.searchPlaceholder.textContent = `Search failed: ${err.message}`;
                    notify("error", `Catalogue search failed: ${err.message}`);
                }
            });
        }
    }

    function renderSearchResults(scenes) {
        elements.searchLoading.classList.add("hidden");
        elements.scenesList.innerHTML = "";

        if (!scenes || scenes.length === 0) {
            elements.searchPlaceholder.classList.remove("hidden");
            elements.searchPlaceholder.textContent = "No cloud-free Sentinel-2 scenes found for the selected date range.";
            return;
        }

        elements.searchPlaceholder.classList.add("hidden");
        elements.scenesList.classList.remove("hidden");

        scenes.forEach((s, idx) => {
            const li = document.createElement("li");
            li.className = "scene-item";
            if (idx === 0) {
                li.classList.add("selected");
                selectScene(s, li);
            }

            const dateStr = s.datetime ? s.datetime.split("T")[0] : "Unknown date";
            li.innerHTML = `
                <div class="scene-info">
                    <span class="scene-date">${escapeHtml(dateStr)}</span>
                    <span class="scene-cloud">Cloud: ${escapeHtml(s.cloud_cover)}% • ${escapeHtml(String(s.id).substring(0, 22))}...</span>
                </div>
                <button type="button" class="btn btn-secondary btn-xs select-btn">Select</button>
            `;

            const btn = li.querySelector(".select-btn");
            btn.addEventListener("click", () => selectScene(s, li));
            elements.scenesList.appendChild(li);
        });
    }

    function selectScene(scene, liElement) {
        document.querySelectorAll(".scene-item").forEach((el) => el.classList.remove("selected"));
        if (liElement) liElement.classList.add("selected");

        state.selectedScene = scene;
        elements.selectedSceneId.textContent = scene.id;
        const dateStr = scene.datetime ? scene.datetime.split("T")[0] : "";
        elements.selectedSceneMeta.textContent = `Date: ${dateStr} • Cloud: ${scene.cloud_cover}% • ${scene.provider}`;
        elements.selectedSceneCard.classList.remove("hidden");

        if (state.currentAoi && elements.aoiWarning.classList.contains("hidden")) {
            elements.btnRunSr.disabled = false;
        }
    }

    // =========================================================================
    // 7. Demo Scene Workflow
    // =========================================================================
    function initDemo() {
        if (!elements.btnLoadDemo) return;

        elements.btnLoadDemo.addEventListener("click", async () => {
            elements.btnLoadDemo.disabled = true;
            elements.btnLoadDemo.textContent = "Loading Demo...";

            try {
                const resp = await fetch("/api/demo/info");
                if (!resp.ok) throw new Error("Demo scene not available.");
                const demoInfo = await resp.json();

                // Set AOI
                const aoi = demoInfo.aoi;
                const bounds = L.latLngBounds([aoi.min_lat, aoi.min_lon], [aoi.max_lat, aoi.max_lon]);
                state.map.fitBounds(bounds, { padding: [40, 40] });
                setAoiFromBounds(bounds);

                // Set Scene
                state.selectedScene = {
                    id: demoInfo.scene_id,
                    datetime: demoInfo.datetime,
                    cloud_cover: demoInfo.cloud_cover,
                    provider: "Local Demo Archive",
                };
                elements.selectedSceneId.textContent = demoInfo.scene_id;
                elements.selectedSceneMeta.textContent = `${demoInfo.tile_id} • 2018-08-25 • 0.0% Cloud`;
                elements.selectedSceneCard.classList.remove("hidden");

                // Automatically launch super-resolution inference
                await launchSrProcessing(true);
            } catch (err) {
                notify("error", `Failed to load the sample scene: ${err.message}`);
            } finally {
                elements.btnLoadDemo.disabled = false;
                elements.btnLoadDemo.innerHTML = `
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                    Load Sample Scene
                `;
            }
        });
    }

    // =========================================================================
    // 8. Super-Resolution Execution & Progress Tracking
    // =========================================================================
    function initSrExecution() {
        if (elements.btnRunSr) {
            elements.btnRunSr.addEventListener("click", () => {
                launchSrProcessing(false);
            });
        }
    }

    /** Analysis toggles as a plain object, shared by the JSON and query-string paths. */
    function analysisQueryParams() {
        const ensembleRaw = elements.optEnsemble ? elements.optEnsemble.value : "auto";
        const params = {
            run_analysis: elements.optAnalysis ? String(!!elements.optAnalysis.checked) : "true",
            run_wald_validation: elements.optWald ? String(!!elements.optWald.checked) : "true",
        };
        if (ensembleRaw !== "auto") params.uncertainty_members = ensembleRaw;
        return params;
    }

    function analysisPayload() {
        const ensembleRaw = elements.optEnsemble ? elements.optEnsemble.value : "auto";
        return {
            run_analysis: elements.optAnalysis ? !!elements.optAnalysis.checked : true,
            run_wald_validation: elements.optWald ? !!elements.optWald.checked : true,
            uncertainty_members: ensembleRaw === "auto" ? null : parseInt(ensembleRaw, 10),
        };
    }

    function beginJobUi(message) {
        elements.progressCard.classList.remove("hidden");
        elements.resultsCard.classList.add("hidden");
        clearOverlays();
        resetProgressSteps();
        state.jobStartedAt = Date.now();
        startJobTimer();
        updateProgressUI(5, message);
    }

    function failJobUi(message) {
        stopJobTimer();
        updateProgressUI(100, `Failed: ${message}`);
        elements.progressFill.classList.remove("indeterminate");
        elements.btnRunSr.disabled = false;
        if (elements.btnRunUploadSr) elements.btnRunUploadSr.disabled = false;
        notify("error", message, { title: "Super-resolution failed", duration: 12000 });
    }

    function startJobTimer() {
        stopJobTimer();
        state.jobTimerHandle = setInterval(() => {
            if (!elements.progressTimer) return;
            const secs = (Date.now() - state.jobStartedAt) / 1000;
            elements.progressTimer.textContent = `${secs.toFixed(1)} s`;
        }, 100);
    }

    function stopJobTimer() {
        if (state.jobTimerHandle) {
            clearInterval(state.jobTimerHandle);
            state.jobTimerHandle = null;
        }
    }

    async function launchSrProcessing(isDemo = false) {
        elements.btnRunSr.disabled = true;
        const modelLabel = state.selectedModel === "swin2sr" ? "SEN2SR-Swin2SR" : "SEN2SR-Lite";
        beginJobUi(`Submitting the patch to ${modelLabel}...`);

        const sceneId = isDemo ? null : (state.selectedScene ? state.selectedScene.id : "auto");

        const payload = Object.assign({
            aoi: isDemo ? null : state.currentAoi,
            scene_id: sceneId,
            is_demo: isDemo,
            model: state.selectedModel,
            overlap: 32,
            clamp_output: true,
        }, analysisPayload());

        try {
            const resp = await fetch("/api/sr/process", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ detail: resp.statusText }));
                throw new Error(err.detail || "Failed to start super-resolution job.");
            }

            const data = await resp.json();
            state.activeJobId = data.job_id;
            pollJobStatus(state.activeJobId);
        } catch (err) {
            failJobUi(err.message);
        }
    }

    function pollJobStatus(jobId) {
        if (state.jobPollTimer) clearInterval(state.jobPollTimer);

        state.jobPollTimer = setInterval(async () => {
            try {
                const resp = await fetch(`/api/sr/jobs/${jobId}`);
                if (!resp.ok) return;
                const job = await resp.json();

                updateProgressUI(job.progress_percent, job.progress_step);

                if (job.status === "completed") {
                    clearInterval(state.jobPollTimer);
                    onJobCompleted(job);
                } else if (job.status === "failed") {
                    clearInterval(state.jobPollTimer);
                    failJobUi(job.error_message || "Unknown processing error.");
                }
            } catch (e) {
                console.error("Polling error:", e);
            }
        }, 500);
    }

    function resetProgressSteps() {
        ["step-aoi", "step-stream", "step-model", "step-geotiff", "step-qa", "step-visual"].forEach((id) => {
            const el = $(id);
            if (el) el.className = "step-item";
        });
    }

    function updateProgressUI(pct, stepMsg) {
        elements.progressFill.style.width = `${pct}%`;
        elements.progressPercentText.textContent = `${pct}%`;
        elements.progressStepText.textContent = stepMsg;

        markStep("step-aoi", pct >= 10);
        markStep("step-stream", pct >= 30);
        markStep("step-model", pct >= 55);
        markStep("step-geotiff", pct >= 78);
        markStep("step-qa", pct >= 96);
        markStep("step-visual", pct >= 100);
    }

    function markStep(stepId, isDone) {
        const el = $(stepId);
        if (el) el.className = isDone ? "step-item done" : "step-item";
    }

    // =========================================================================
    // 9. Visual Overlay Rendering & Results Display
    // =========================================================================
    function onJobCompleted(job) {
        stopJobTimer();
        state.jobResult = job.result;
        state.analysis = job.result.analysis || null;
        elements.btnRunSr.disabled = false;
        if (elements.btnRunUploadSr) elements.btnRunUploadSr.disabled = false;

        // Populate Result Details
        elements.resCrs.textContent = job.result.crs || "—";
        elements.resTime.textContent = `${job.result.processing_time_sec} s`;
        elements.resDevice.textContent = formatDevice(job.result.device_used);
        if (elements.resModel) {
            elements.resModel.textContent = job.result.model || (state.selectedModel === "swin2sr" ? "SEN2SR-Swin2SR" : "SEN2SR-Lite");
        }
        if (elements.resVram) {
            elements.resVram.textContent = job.result.peak_vram_mb ? `${job.result.peak_vram_mb} MB` : "N/A";
        }
        if (elements.resGrid && job.result.output_shape) {
            const s = job.result.output_shape;
            elements.resGrid.textContent = `${s[2]} × ${s[1]} px × ${s[0]} bands`;
        }

        const base = `/api/sr/jobs/${job.job_id}/download`;
        setDownload(elements.btnDownloadGeotiff, `${base}/geotiff`, true);
        setDownload(elements.btnDownloadNative, `${base}/native`, true);
        setDownload(elements.btnDownloadRgb, `${base}/rgb`, true);
        setDownload(elements.btnDownloadCir, `${base}/cir`, true);
        setDownload(elements.btnDownloadConfidence, `${base}/confidence`, !!job.result.confidence_geotiff_path);
        const hasReport = !!(state.analysis && state.analysis.report);
        setDownload(elements.btnDownloadReport, `${base}/report`, hasReport);
        setDownload(elements.btnDownloadReportMd, `${base}/report-md`, hasReport);

        elements.resultsCard.classList.remove("hidden");
        elements.viewModeGroup.style.display = "flex";
        elements.comparisonToggleGroup.style.display = "flex";

        renderAnalysis();
        displayJobLayers(job.result);
        refreshHistory();

        const verdict = state.analysis && state.analysis.summary && state.analysis.summary.verdict;
        const gsd = job.result.output_gsd || "2.50m";
        notify("success",
            `Patch reconstructed at ${gsd} in ${job.result.processing_time_sec} s${verdict ? " · QA verdict: " + verdict.replace(/-/g, " ") : ""}.`,
            { title: "Super-resolution complete", duration: 8000 });

        const warnings = (state.analysis && state.analysis.warnings) || [];
        if (warnings.length) {
            notify("warning", warnings.join(" "), { title: "Quality assessment notes", duration: 12000 });
        }
    }

    function formatDevice(device) {
        if (!device) return "—";
        return device.toUpperCase();
    }

    function setDownload(anchor, href, enabled) {
        if (!anchor) return;
        if (enabled) {
            anchor.href = href;
            anchor.removeAttribute("aria-disabled");
            anchor.style.opacity = "";
            anchor.style.pointerEvents = "";
        } else {
            anchor.removeAttribute("href");
            anchor.setAttribute("aria-disabled", "true");
            anchor.style.opacity = "0.45";
            anchor.style.pointerEvents = "none";
        }
    }

    function displayJobLayers(result) {
        clearOverlays();

        const bounds = result.leaflet_bounds;
        state.map.fitBounds(bounds, { padding: [50, 50] });

        state.layers.left = L.imageOverlay(getLayerUrl("left"), bounds, {
            opacity: 1.0,
            interactive: false,
        }).addTo(state.map);

        state.layers.right = L.imageOverlay(getLayerUrl("right"), bounds, {
            opacity: 1.0,
            interactive: false,
        }).addTo(state.map);

        // Patch Border Outline
        state.patchOutline = L.rectangle(bounds, {
            color: "#059669",
            weight: 2,
            fill: false,
            dashArray: "3, 3",
        }).addTo(state.map);

        // Activate UI Controls
        elements.sliderContainer.classList.remove("hidden");
        elements.compareLabels.classList.remove("hidden");

        updateLabelTexts();
        setViewMode(state.viewMode);
        updateSplitClipping();
        refreshLoupe();
    }

    function getLayerUrl(side) {
        const r = state.jobResult;
        if (!r || !r.previews) return "";
        const key = state.activeLayer;

        if (side === "right") {
            return r.previews[`sr_${key}`] || "";
        }
        // Analysis rasters have no native-resolution counterpart; swipe them against
        // the super-resolved true-colour image so the confidence field is interpretable.
        if (state.layerKind === "analysis") {
            return r.previews["sr_rgb"] || "";
        }
        return r.previews[`${state.leftCompareMode}_${key}`] || r.previews[`lr_${key}`] || "";
    }

    function clearOverlays() {
        if (state.layers.left) {
            state.map.removeLayer(state.layers.left);
            state.layers.left = null;
        }
        if (state.layers.right) {
            state.map.removeLayer(state.layers.right);
            state.layers.right = null;
        }
        if (state.patchOutline) {
            state.map.removeLayer(state.patchOutline);
            state.patchOutline = null;
        }
        if (state.probeMarker) {
            state.map.removeLayer(state.probeMarker);
            state.probeMarker = null;
        }
        elements.sliderContainer.classList.add("hidden");
        elements.compareLabels.classList.add("hidden");
        refreshLoupe();
    }

    // =========================================================================
    // 10. Split Slider & View Modes
    // =========================================================================
    function updateSplitClipping() {
        if (!state.layers.left || !state.layers.right || state.viewMode !== "split") return;

        const leftEl = state.layers.left.getElement();
        const rightEl = state.layers.right.getElement();
        if (!leftEl || !rightEl) return;

        const mapRect = elements.mapViewport.getBoundingClientRect();
        const splitX = mapRect.width * state.sliderPosRatio;

        elements.sliderDivider.style.left = `${splitX}px`;
        elements.sliderDivider.setAttribute("aria-valuenow", String(Math.round(state.sliderPosRatio * 100)));

        const leftBbox = leftEl.getBoundingClientRect();
        const rightBbox = rightEl.getBoundingClientRect();

        const clipLeftRight = Math.max(0, splitX - (leftBbox.left - mapRect.left));
        leftEl.style.clipPath = `polygon(0 0, ${clipLeftRight}px 0, ${clipLeftRight}px 100%, 0 100%)`;

        const clipRightLeft = Math.max(0, splitX - (rightBbox.left - mapRect.left));
        rightEl.style.clipPath = `polygon(${clipRightLeft}px 0, 100% 0, 100% 100%, ${clipRightLeft}px 100%)`;
    }

    function setViewMode(mode) {
        state.viewMode = mode;

        [elements.btnViewSplit, elements.btnViewLr, elements.btnViewSr, elements.btnViewBlend].forEach((b) => {
            if (b) b.classList.remove("active");
        });

        if (mode === "split") {
            elements.btnViewSplit.classList.add("active");
            elements.sliderContainer.classList.remove("hidden");
            elements.compareLabels.classList.remove("hidden");
            elements.blendSliderGroup.style.display = "none";
            if (state.layers.left) state.layers.left.setOpacity(1.0);
            if (state.layers.right) state.layers.right.setOpacity(1.0);
            updateSplitClipping();
        } else if (mode === "lr_only") {
            elements.btnViewLr.classList.add("active");
            elements.sliderContainer.classList.add("hidden");
            elements.compareLabels.classList.add("hidden");
            elements.blendSliderGroup.style.display = "none";
            if (state.layers.left) {
                state.layers.left.setOpacity(1.0);
                const el = state.layers.left.getElement();
                if (el) el.style.clipPath = "none";
            }
            if (state.layers.right) state.layers.right.setOpacity(0.0);
        } else if (mode === "sr_only") {
            elements.btnViewSr.classList.add("active");
            elements.sliderContainer.classList.add("hidden");
            elements.compareLabels.classList.add("hidden");
            elements.blendSliderGroup.style.display = "none";
            if (state.layers.left) state.layers.left.setOpacity(0.0);
            if (state.layers.right) {
                state.layers.right.setOpacity(1.0);
                const el = state.layers.right.getElement();
                if (el) el.style.clipPath = "none";
            }
        } else if (mode === "blend") {
            elements.btnViewBlend.classList.add("active");
            elements.sliderContainer.classList.add("hidden");
            elements.compareLabels.classList.add("hidden");
            elements.blendSliderGroup.style.display = "flex";
            if (state.layers.left) {
                state.layers.left.setOpacity(1.0);
                const el = state.layers.left.getElement();
                if (el) el.style.clipPath = "none";
            }
            if (state.layers.right) {
                state.layers.right.setOpacity(state.srOpacity / 100);
                const el = state.layers.right.getElement();
                if (el) el.style.clipPath = "none";
            }
        }
        refreshLoupe();
    }

    function nudgeSlider(delta) {
        if (state.viewMode !== "split") return;
        state.sliderPosRatio = Math.max(0.02, Math.min(0.98, state.sliderPosRatio + delta));
        updateSplitClipping();
    }

    function initSliderAndModes() {
        if (elements.btnViewSplit) elements.btnViewSplit.addEventListener("click", () => setViewMode("split"));
        if (elements.btnViewLr) elements.btnViewLr.addEventListener("click", () => setViewMode("lr_only"));
        if (elements.btnViewSr) elements.btnViewSr.addEventListener("click", () => setViewMode("sr_only"));
        if (elements.btnViewBlend) elements.btnViewBlend.addEventListener("click", () => setViewMode("blend"));

        if (elements.srOpacitySlider) {
            elements.srOpacitySlider.addEventListener("input", (e) => {
                state.srOpacity = parseInt(e.target.value, 10);
                elements.srOpacityVal.textContent = `${state.srOpacity}%`;
                if (state.layers.right && state.viewMode === "blend") {
                    state.layers.right.setOpacity(state.srOpacity / 100);
                }
            });
        }

        // Slider Dragging Events
        if (elements.sliderDivider) {
            elements.sliderDivider.addEventListener("mousedown", (e) => {
                state.isDraggingSlider = true;
                e.preventDefault();
            });

            elements.sliderDivider.addEventListener("touchstart", () => {
                state.isDraggingSlider = true;
            }, { passive: true });

            // Keyboard operability for the swipe handle
            elements.sliderDivider.addEventListener("keydown", (e) => {
                const step = e.shiftKey ? 0.1 : 0.02;
                if (e.key === "ArrowLeft") { nudgeSlider(-step); e.preventDefault(); }
                else if (e.key === "ArrowRight") { nudgeSlider(step); e.preventDefault(); }
                else if (e.key === "Home") { state.sliderPosRatio = 0.02; updateSplitClipping(); e.preventDefault(); }
                else if (e.key === "End") { state.sliderPosRatio = 0.98; updateSplitClipping(); e.preventDefault(); }
            });
        }

        window.addEventListener("mousemove", (e) => {
            if (!state.isDraggingSlider) return;
            const rect = elements.mapViewport.getBoundingClientRect();
            let posX = e.clientX - rect.left;
            posX = Math.max(30, Math.min(rect.width - 30, posX));
            state.sliderPosRatio = posX / rect.width;
            updateSplitClipping();
        });

        window.addEventListener("mouseup", () => {
            if (state.isDraggingSlider) {
                state.isDraggingSlider = false;
            }
        });

        window.addEventListener("touchmove", (e) => {
            if (!state.isDraggingSlider || !e.touches[0]) return;
            const rect = elements.mapViewport.getBoundingClientRect();
            let posX = e.touches[0].clientX - rect.left;
            posX = Math.max(30, Math.min(rect.width - 30, posX));
            state.sliderPosRatio = posX / rect.width;
            updateSplitClipping();
        }, { passive: true });

        window.addEventListener("touchend", () => {
            state.isDraggingSlider = false;
        });

        if (elements.selectLeftLayer) {
            elements.selectLeftLayer.addEventListener("change", (e) => {
                state.leftCompareMode = e.target.value;
                if (state.jobResult && state.layers.left) {
                    state.layers.left.setUrl(getLayerUrl("left"));
                }
                updateLabelTexts();
                refreshLoupe();
            });
        }

        if (elements.btnZoomPatch) {
            elements.btnZoomPatch.addEventListener("click", zoomToPatch);
        }
    }

    function zoomToPatch() {
        if (state.jobResult && state.jobResult.leaflet_bounds) {
            state.map.fitBounds(state.jobResult.leaflet_bounds, { padding: [50, 50] });
        } else if (state.aoiLayer) {
            state.map.fitBounds(state.aoiLayer.getBounds(), { padding: [50, 50] });
        }
    }

    // =========================================================================
    // 11. Layer catalogue, picker and legend
    // =========================================================================
    async function loadLayerCatalog() {
        try {
            const resp = await fetch("/api/layers");
            if (!resp.ok) throw new Error(resp.statusText);
            state.catalog = await resp.json();
            buildLayerPicker();
        } catch (e) {
            console.warn("Layer catalogue unavailable:", e);
        }
    }

    function buildLayerPicker() {
        const sel = elements.selectLayer;
        if (!sel || !state.catalog) return;

        sel.innerHTML = "";
        const groups = [
            ["Composites", state.catalog.composites || [], "composite"],
            ["Spectral indices (2.5 m products)", state.catalog.indices || [], "index"],
            ["Reconstruction analysis", state.catalog.analysis || [], "analysis"],
        ];

        groups.forEach(([label, items, kind]) => {
            if (!items.length) return;
            const og = document.createElement("optgroup");
            og.label = label;
            items.forEach((item) => {
                const opt = document.createElement("option");
                opt.value = item.key;
                opt.dataset.kind = kind;
                if (kind === "composite") {
                    opt.textContent = `${item.label} (${(item.detail || "").split("—")[0].trim()})`;
                } else if (kind === "index") {
                    opt.textContent = `${item.name} — ${item.key.toUpperCase()}`;
                } else {
                    opt.textContent = item.label;
                }
                og.appendChild(opt);
            });
            sel.appendChild(og);
        });

        sel.value = state.activeLayer;
        sel.addEventListener("change", (e) => {
            const opt = e.target.selectedOptions[0];
            setActiveLayer(e.target.value, opt ? opt.dataset.kind : "composite");
        });
        updateLayerHint();
    }

    function findLayerSpec(key, kind) {
        if (!state.catalog) return null;
        const list = kind === "index" ? state.catalog.indices
            : kind === "analysis" ? state.catalog.analysis
                : state.catalog.composites;
        return (list || []).filter((x) => x.key === key)[0] || null;
    }

    function setActiveLayer(key, kind) {
        state.activeLayer = key;
        state.layerKind = kind || "composite";

        if (elements.selectLayer && elements.selectLayer.value !== key) {
            elements.selectLayer.value = key;
        }

        // Bicubic has no meaningful counterpart for the analysis rasters.
        if (elements.selectLeftLayer) {
            elements.selectLeftLayer.disabled = state.layerKind === "analysis";
        }

        if (state.jobResult) {
            if (state.layers.left) state.layers.left.setUrl(getLayerUrl("left"));
            if (state.layers.right) state.layers.right.setUrl(getLayerUrl("right"));
        }
        updateLabelTexts();
        updateLayerHint();
        renderLegend();
        refreshLoupe();
    }

    function updateLayerHint() {
        if (!elements.layerHint) return;
        const spec = findLayerSpec(state.activeLayer, state.layerKind);
        if (!spec) { elements.layerHint.textContent = ""; return; }
        elements.layerHint.textContent = spec.detail || spec.application || spec.description || "";
    }

    function renderLegend() {
        const box = elements.mapLegend;
        if (!box) return;

        if (state.layerKind === "composite" || !state.jobResult) {
            box.hidden = true;
            return;
        }

        const spec = findLayerSpec(state.activeLayer, state.layerKind);
        if (!spec) { box.hidden = true; return; }

        box.hidden = false;
        elements.legendTitle.textContent = spec.name || spec.label || state.activeLayer;
        elements.legendKey.textContent = state.activeLayer.toUpperCase();
        elements.legendFormula.textContent = spec.formula || spec.detail || "";

        const stops = spec.legend_hex || null;
        elements.legendGradient.innerHTML = "";
        if (stops && NTRO.charts && NTRO.charts.gradientLegend) {
            NTRO.charts.gradientLegend(elements.legendGradient, {
                stops: stops,
                min: spec.vmin !== undefined ? spec.vmin : 0,
                max: spec.vmax !== undefined ? spec.vmax : 1,
                label: spec.name || spec.label,
                unit: "",
            });
        } else if (stops) {
            const bar = document.createElement("div");
            bar.style.cssText = `height:12px;border-radius:3px;background:linear-gradient(90deg,${stops.join(",")});`;
            elements.legendGradient.appendChild(bar);
        }

        elements.legendClasses.innerHTML = "";
        const classes = spec.classes || [];
        classes.forEach((c) => {
            // Registry classes arrive as [lo, hi, label, colour].
            const lo = Array.isArray(c) ? c[0] : c.lo;
            const hi = Array.isArray(c) ? c[1] : c.hi;
            const label = Array.isArray(c) ? c[2] : c.label;
            const color = Array.isArray(c) ? c[3] : c.color;
            const row = document.createElement("div");
            row.className = "legend-class";
            row.innerHTML = `<span class="legend-swatch" style="background:${escapeHtml(color)}"></span>
                <span>${escapeHtml(label)} <span class="mono" style="opacity:.65">${fmt(lo, 2)}…${fmt(hi, 2)}</span></span>`;
            elements.legendClasses.appendChild(row);
        });
    }

    function updateLabelTexts() {
        const spec = findLayerSpec(state.activeLayer, state.layerKind);
        const layerLabel = spec ? (spec.name || spec.label || state.activeLayer) : state.activeLayer.toUpperCase();

        if (state.layerKind === "analysis") {
            elements.labelLeftText.textContent = "Super-resolved 2.5m (natural colour)";
            elements.labelRightText.textContent = `${layerLabel} (2.50m)`;
            return;
        }

        if (state.leftCompareMode === "bicubic") {
            elements.labelLeftText.textContent = `Bicubic baseline (2.50m • ${layerLabel})`;
        } else {
            elements.labelLeftText.textContent = `Original Sentinel-2 (10.0m native • ${layerLabel})`;
        }

        const modelName = (state.jobResult && state.jobResult.model)
            ? state.jobResult.model
            : (state.selectedModel === "swin2sr" ? "SEN2SR-Swin2SR" : "SEN2SR-Lite");
        elements.labelRightText.textContent = `${modelName} (2.50m neural SR • ${layerLabel})`;
    }

    // =========================================================================
    // 12. Model selector
    // =========================================================================
    function initModelSelector() {
        const radios = document.querySelectorAll('input[name="sr-model"]');
        radios.forEach((radio) => {
            radio.addEventListener("change", (e) => {
                state.selectedModel = e.target.value;
                const isSwin = state.selectedModel === "swin2sr";

                const labelLite = $("label-model-lite");
                const labelSwin = $("label-model-swin");
                if (labelLite) labelLite.classList.toggle("active", !isSwin);
                if (labelSwin) labelSwin.classList.toggle("active", isSwin);

                const chip = $("chip-model-name");
                if (chip) chip.textContent = isSwin ? "SEN2SR-Swin2SR" : "SEN2SR-Lite";

                const explainer = $("upscale-explainer-text");
                if (explainer) {
                    explainer.innerHTML = isSwin
                        ? `Selected <strong>SEN2SR-Swin2SR</strong> &bull; Higher-capacity 4&times; spatial upscaling (10m &rarr; 2.5m)`
                        : `Selected <strong>SEN2SR-Lite</strong> &bull; 4&times; spatial upscaling (10m &rarr; 2.5m)`;
                }

                if (isSwin) {
                    notify("info",
                        "SEN2SR-Swin2SR runs a pure-PyTorch state-space scan and can take several minutes per patch. The uncertainty ensemble is disabled by default for this model.",
                        { title: "Higher-capacity model selected", duration: 9000 });
                }
                updateLabelTexts();
            });
        });
    }

    // =========================================================================
    // 13. Quality-assessment workspace
    // =========================================================================
    function initResultTabs() {
        elements.resultTabs.forEach((tab) => {
            tab.addEventListener("click", () => selectResultTab(tab.id));
        });
        selectResultTab("rtab-product");
    }

    function selectResultTab(tabId) {
        elements.resultTabs.forEach((tab) => {
            const on = tab.id === tabId;
            tab.setAttribute("aria-selected", on ? "true" : "false");
            const pane = $(tab.dataset.pane);
            if (pane) pane.hidden = !on;
        });
    }

    function renderAnalysis() {
        renderQualityPane();
        renderAppsPane();
        renderProbePane(null);
        renderLegend();
    }

    function emptyState(message) {
        return `<div class="empty-state">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><circle cx="12" cy="12" r="9"></circle><line x1="12" y1="8" x2="12" y2="13"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
            <span>${escapeHtml(message)}</span>
        </div>`;
    }

    function renderQualityPane() {
        const host = elements.qualityContent;
        if (!host) return;
        const a = state.analysis;

        if (!a || (!a.wald && !a.consistency && !a.uncertainty)) {
            host.innerHTML = emptyState(
                "No quality assessment was run for this job. Enable “Run quality assessment” under the model selector before upscaling."
            );
            return;
        }

        host.innerHTML = "";

        // --- Verdict banner ---
        const summary = a.summary || {};
        const verdict = summary.verdict || "inconclusive";
        const banner = document.createElement("div");
        banner.className = "verdict-banner";
        banner.setAttribute("data-verdict", verdict);
        banner.innerHTML = `
            <span class="verdict-icon">${verdictIcon(verdict)}</span>
            <span>
                <span class="verdict-title">${escapeHtml(verdict.replace(/-/g, " "))}</span>
                <span class="verdict-detail">${escapeHtml(verdictDetail(verdict, a))}</span>
            </span>`;
        host.appendChild(banner);

        // --- Wald synthesis protocol ---
        if (a.wald && a.wald.metrics) {
            const sec = section("Accuracy — Wald synthesis protocol");
            const note = document.createElement("p");
            note.className = "analysis-note";
            note.textContent =
                "The observed 10 m image is degraded to 40 m, reconstructed back to 10 m by the same network, " +
                "and scored against the real observation. This is a genuine quantitative accuracy figure obtained " +
                "without any commercial high-resolution reference, but it validates the model at the 40→10 m scale, " +
                "not directly at 2.5 m.";
            sec.appendChild(note);

            const grid = document.createElement("div");
            grid.className = "metric-grid";
            ["psnr_db", "ssim", "sam_deg", "ergas"].forEach((key) => {
                const meta = metricMeta(key);
                const value = a.wald.metrics[key];
                const cell = document.createElement("div");
                cell.className = "metric-cell";
                if (NTRO.charts && NTRO.charts.metricGauge && meta) {
                    NTRO.charts.metricGauge(cell, {
                        value: value,
                        min: 0,
                        max: gaugeMax(key, value, meta),
                        good: meta.good,
                        excellent: meta.excellent,
                        better: meta.better,
                        label: meta.label,
                        unit: meta.unit,
                        description: meta.description,
                    });
                } else {
                    cell.innerHTML = `<div style="font-size:.68rem;color:var(--text-muted)">${escapeHtml((meta && meta.label) || key)}</div>
                        <div class="mono" style="font-size:1.05rem;font-weight:600">${fmt(value, 3)}</div>`;
                }
                grid.appendChild(cell);
            });
            sec.appendChild(grid);
            sec.appendChild(waldComparisonTable(a.wald));

            const perBandChart = document.createElement("div");
            sec.appendChild(perBandChart);
            renderPerBandChart(perBandChart, a.wald);

            host.appendChild(sec);
        }

        // --- Radiometric consistency ---
        if (a.consistency) {
            const c = a.consistency;
            const sec = section("Radiometric consistency", c.passed
                ? '<span class="pill pill-pass">Within tolerance</span>'
                : '<span class="pill pill-fail">Out of tolerance</span>');
            const note = document.createElement("p");
            note.className = "analysis-note";
            note.textContent =
                "Downsampling the 2.5 m product back onto the native 10 m grid must reproduce the observed " +
                "Sentinel-2 reflectance. This is Wald's consistency property and it costs no extra inference.";
            sec.appendChild(note);

            const tol = c.tolerance || {};
            const rows = [
                ["Max absolute band bias", fmt(c.max_abs_bias, 5), `≤ ${fmt(tol.max_abs_bias, 3)}`],
                ["Mean spectral angle", `${fmt(c.spectral_angle_deg, 3)}°`, `≤ ${fmt(tol.spectral_angle_deg, 2)}°`],
                ["Round-trip PSNR", `${fmt(c.metrics && c.metrics.psnr_db, 2)} dB`, "—"],
                ["Round-trip SSIM", fmt(c.metrics && c.metrics.ssim, 4), "—"],
            ];
            sec.appendChild(simpleTable(["Check", "Measured", "Tolerance"], rows, [false, true, true]));

            if (c.band_names && c.per_band_bias) {
                const det = document.createElement("details");
                det.innerHTML = `<summary style="cursor:pointer;font-size:.7rem;font-weight:700;color:var(--text-secondary);margin-top:.5rem">Per-band bias</summary>`;
                const rows2 = c.band_names.map((b, i) => [
                    b,
                    fmt(c.per_band_bias[i], 5),
                    fmt(c.per_band_rmse ? c.per_band_rmse[i] : null, 5),
                ]);
                const wrap = document.createElement("div");
                wrap.className = "scroll-x";
                wrap.appendChild(simpleTable(["Band", "Bias", "RMSE"], rows2, [false, true, true]));
                det.appendChild(wrap);
                sec.appendChild(det);
            }
            host.appendChild(sec);
        }

        // --- Uncertainty ---
        if (a.uncertainty) {
            const u = a.uncertainty;
            const riskPill = `<span class="pill ${riskPillClass(u.hallucination_risk)}">${escapeHtml(String(u.hallucination_risk || "").toUpperCase())} RISK</span>`;
            const sec = section("Reconstruction uncertainty", riskPill);

            const gaugeHost = document.createElement("div");
            gaugeHost.className = "metric-cell";
            gaugeHost.style.marginBottom = "8px";
            if (NTRO.charts && NTRO.charts.metricGauge) {
                NTRO.charts.metricGauge(gaugeHost, {
                    value: u.reliability_score,
                    min: 0, max: 100, good: 70, excellent: 85, better: "higher",
                    label: "Reliability score",
                    unit: "%",
                    description: "Share of the reconstruction supported by the observation rather than synthesised.",
                });
            } else {
                gaugeHost.innerHTML = `<div class="mono" style="font-size:1.1rem;font-weight:600">${fmt(u.reliability_score, 1)}%</div>`;
            }
            sec.appendChild(gaugeHost);

            sec.appendChild(simpleTable(
                ["Quantity", "Mean", "95th pct"],
                [
                    ["Ensemble spread (reflectance)", fmt(u.mean_std, 5), fmt(u.p95_std, 5)],
                    ["Synthesised detail", fmt(u.mean_novelty, 5), fmt(u.p95_novelty, 5)],
                ],
                [false, true, true]
            ));

            const meta = document.createElement("p");
            meta.className = "analysis-note";
            meta.textContent = `Method: ${u.method || "—"} · ${u.n_ensemble || 1} ensemble member(s).`;
            sec.appendChild(meta);

            if (u.interpretation) {
                const interp = document.createElement("p");
                interp.className = "analysis-note";
                interp.style.color = "var(--text-secondary)";
                interp.textContent = u.interpretation;
                sec.appendChild(interp);
            }

            if (u.histogram && NTRO.charts && NTRO.charts.histogram) {
                const hist = document.createElement("div");
                hist.style.marginTop = "6px";
                NTRO.charts.histogram(hist, {
                    edges: u.histogram.edges,
                    counts: u.histogram.counts,
                    color: cssVar("--primary-500", "#3b82f6"),
                    title: "Distribution of per-pixel confidence",
                    xLabel: "Confidence (0–1)",
                });
                sec.appendChild(hist);
            }

            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "btn btn-secondary btn-sm btn-block mt-2";
            btn.textContent = "Show the confidence map on the map";
            btn.addEventListener("click", () => {
                setActiveLayer("confidence", "analysis");
                setViewMode("split");
            });
            sec.appendChild(btn);

            host.appendChild(sec);
        }

        // --- Spectral fidelity ---
        if (a.spectral_fidelity && a.spectral_fidelity.bands && a.spectral_fidelity.bands.length) {
            const sf = a.spectral_fidelity;
            const sec = section("Spectral fidelity (band means)");
            const rows = sf.bands.map((b) => [
                b.band, fmt(b.lr_mean, 4), fmt(b.sr_mean, 4), fmt(b.rel_pct, 2) + " %",
            ]);
            const wrap = document.createElement("div");
            wrap.className = "scroll-x";
            wrap.appendChild(simpleTable(["Band", "10 m mean", "2.5 m mean", "Δ"], rows, [false, true, true, true]));
            sec.appendChild(wrap);
            host.appendChild(sec);
        }

        // --- Caveats ---
        const caveats = (summary.caveats || []);
        if (caveats.length) {
            const sec = section("Caveats");
            const ul = document.createElement("ul");
            ul.style.cssText = "margin:0;padding-left:1.05rem;";
            caveats.forEach((c) => {
                const li = document.createElement("li");
                li.className = "analysis-note";
                li.style.marginTop = "3px";
                li.textContent = c;
                ul.appendChild(li);
            });
            sec.appendChild(ul);
            host.appendChild(sec);
        }
    }

    function gaugeMax(key, value, meta) {
        if (key === "ssim") return 1;
        if (meta.better === "lower") return Math.max(meta.good * 2, (value || 0) * 1.3, 1);
        return Math.max(meta.excellent * 1.35, (value || 0) * 1.1);
    }

    function verdictIcon(verdict) {
        if (verdict === "validated") {
            return '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" aria-hidden="true"><polyline points="20 6 9 17 4 12"></polyline></svg>';
        }
        if (verdict === "validated-with-caveats") {
            return '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true"><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>';
        }
        return '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" aria-hidden="true"><circle cx="12" cy="12" r="9"></circle><line x1="12" y1="8" x2="12" y2="13"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>';
    }

    function verdictDetail(verdict, a) {
        const psnr = a.wald && a.wald.metrics ? a.wald.metrics.psnr_db : null;
        const ssim = a.wald && a.wald.metrics ? a.wald.metrics.ssim : null;
        const rel = a.uncertainty ? a.uncertainty.reliability_score : null;
        const bits = [];
        if (psnr !== null && psnr !== undefined) bits.push(`Wald PSNR ${fmt(psnr, 2)} dB`);
        if (ssim !== null && ssim !== undefined) bits.push(`SSIM ${fmt(ssim, 3)}`);
        if (rel !== null && rel !== undefined) bits.push(`reliability ${fmt(rel, 0)}%`);
        if (a.consistency) bits.push(a.consistency.passed ? "consistency passed" : "consistency out of tolerance");
        return bits.length ? bits.join(" · ") : "No quantitative validation was available for this product.";
    }

    function riskPillClass(risk) {
        if (risk === "low") return "pill-pass";
        if (risk === "moderate") return "pill-pass";
        if (risk === "elevated") return "pill-warn";
        return "pill-fail";
    }

    function section(title, badgeHtml) {
        const sec = document.createElement("div");
        sec.className = "analysis-section";
        const h = document.createElement("div");
        h.className = "analysis-section-title";
        h.innerHTML = `<span>${escapeHtml(title)}</span>${badgeHtml || ""}`;
        sec.appendChild(h);
        return sec;
    }

    function simpleTable(headers, rows, numeric) {
        const t = document.createElement("table");
        t.className = "qa-table";
        const thead = document.createElement("thead");
        const tr = document.createElement("tr");
        headers.forEach((h, i) => {
            const th = document.createElement("th");
            th.scope = "col";
            th.textContent = h;
            if (numeric && numeric[i]) th.className = "num";
            tr.appendChild(th);
        });
        thead.appendChild(tr);
        t.appendChild(thead);

        const tbody = document.createElement("tbody");
        rows.forEach((r) => {
            const row = document.createElement("tr");
            r.forEach((cell, i) => {
                const td = document.createElement("td");
                if (numeric && numeric[i]) td.className = "num";
                if (cell && cell.html) td.innerHTML = cell.html;
                else td.textContent = cell === null || cell === undefined ? "—" : String(cell);
                row.appendChild(td);
            });
            tbody.appendChild(row);
        });
        t.appendChild(tbody);
        return t;
    }

    function waldComparisonTable(wald) {
        const keys = ["psnr_db", "ssim", "sam_deg", "ergas", "rmse", "scc"];
        const rows = keys.map((k) => {
            const meta = metricMeta(k);
            const m = wald.metrics ? wald.metrics[k] : null;
            const b = wald.baseline_metrics ? wald.baseline_metrics[k] : null;
            const imp = wald.improvement ? wald.improvement[k] : null;
            const cls = (imp === null || imp === undefined || !isFinite(imp)) ? "" : (imp >= 0 ? "delta-pos" : "delta-neg");
            const sign = (imp !== null && imp !== undefined && isFinite(imp) && imp >= 0) ? "+" : "";
            const digits = k === "psnr_db" ? 2 : 4;
            return [
                (meta && meta.label) || k,
                fmt(m, digits),
                fmt(b, digits),
                { html: `<span class="${cls}">${imp === null || imp === undefined || !isFinite(imp) ? "—" : sign + fmt(imp, digits)}</span>` },
            ];
        });
        const wrap = document.createElement("div");
        wrap.className = "scroll-x";
        const t = simpleTable(["Metric", "Model", "Bicubic", "Gain"], rows, [false, true, true, true]);
        const cap = document.createElement("caption");
        cap.textContent = "Model versus bicubic on the same degraded input — positive gain means the network beat interpolation.";
        t.insertBefore(cap, t.firstChild);
        wrap.appendChild(t);
        return wrap;
    }

    function renderPerBandChart(host, wald) {
        if (!NTRO.charts || !NTRO.charts.groupedBars) return;
        const per = (wald.metrics && wald.metrics.per_band) || [];
        if (!per.length) return;
        const baselinePer = (wald.baseline_metrics && wald.baseline_metrics.per_band) || [];
        const series = [{
            label: "Model",
            color: cssVar("--primary-600", "#2563eb"),
            values: per.map((p) => p.psnr_db),
        }];
        if (baselinePer.length === per.length) {
            series.push({
                label: "Bicubic",
                color: cssVar("--text-muted", "#64748b"),
                values: baselinePer.map((p) => p.psnr_db),
            });
        }
        NTRO.charts.groupedBars(host, {
            categories: per.map((p) => p.band),
            series: series,
            title: "Per-band reconstruction PSNR",
            yLabel: "dB",
        });
    }

    function renderAppsPane() {
        const host = elements.appsContent;
        if (!host) return;
        const a = state.analysis;
        const indices = (a && a.indices) || [];

        if (!indices.length) {
            host.innerHTML = emptyState(
                "No thematic products were derived. Enable “Run quality assessment” before upscaling to compute spectral indices at 2.5 m."
            );
            return;
        }

        host.innerHTML = "";
        const intro = document.createElement("p");
        intro.className = "analysis-note";
        intro.style.marginBottom = "0.6rem";
        intro.textContent =
            "Thematic indices computed on the super-resolved 10-band stack. “Edge gain” compares the thematic " +
            "gradient energy of the 2.5 m product against the interpolated 10 m index — above 1.0 means the " +
            "product resolves boundaries the native index could not.";
        host.appendChild(intro);

        indices.forEach((stat) => {
            const spec = findLayerSpec(stat.key, "index") || {};
            const card = document.createElement("div");
            card.className = "index-card";

            const head = document.createElement("div");
            head.className = "index-card-head";
            head.innerHTML = `<span class="index-card-name">${escapeHtml(stat.name || spec.name || stat.key)}</span>
                <span class="index-card-key">${escapeHtml(stat.key)}</span>`;
            card.appendChild(head);

            if (spec.application || spec.description) {
                const app = document.createElement("div");
                app.className = "index-card-app";
                app.textContent = spec.application || spec.description;
                card.appendChild(app);
            }

            if (stat.classes && stat.classes.length && NTRO.charts && NTRO.charts.classBar) {
                const bar = document.createElement("div");
                NTRO.charts.classBar(bar, { classes: stat.classes, title: "" });
                card.appendChild(bar);
            }

            const delta = stat.delta || {};
            const stats = document.createElement("div");
            stats.className = "index-card-stats";
            stats.innerHTML = `
                <span>Mean <b>${fmt(stat.mean, 3)}</b></span>
                <span>P05–P95 <b>${fmt(stat.p05, 2)}–${fmt(stat.p95, 2)}</b></span>
                <span>Edge gain <b>${fmt(delta.edge_gain, 2)}×</b></span>
                <span>Valid <b>${fmt((stat.valid_fraction || 0) * 100, 1)}%</b></span>`;
            card.appendChild(stats);

            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "index-show-btn";
            btn.style.marginTop = "6px";
            btn.textContent = "Show on map →";
            btn.addEventListener("click", () => {
                setActiveLayer(stat.key, "index");
                setViewMode("split");
            });
            card.appendChild(btn);

            host.appendChild(card);
        });
    }

    // =========================================================================
    // 14. Spectral probe
    // =========================================================================
    function setInspectArmed(on) {
        state.inspectArmed = !!on;
        if (elements.btnInspect) {
            elements.btnInspect.setAttribute("aria-pressed", state.inspectArmed ? "true" : "false");
            elements.btnInspect.classList.toggle("active", state.inspectArmed);
        }
        if (elements.mapViewport) {
            elements.mapViewport.classList.toggle("inspect-armed", state.inspectArmed);
        }
        if (state.inspectArmed) {
            selectResultTab("rtab-probe");
            notify("info", "Probe armed — click anywhere on the patch to read its 10-band spectral signature.", { duration: 5000 });
        }
    }

    function renderProbePane(payload) {
        const host = elements.probeContent;
        if (!host) return;
        if (!NTRO.inspector) {
            host.innerHTML = emptyState("The spectral inspector module failed to load.");
            return;
        }
        if (!payload) {
            NTRO.inspector.renderEmpty(host,
                state.jobResult
                    ? "Enable the Probe tool above the map, then click a point on the patch to read its spectral signature."
                    : "Run a super-resolution job first, then probe any pixel of the result.");
            return;
        }
        NTRO.inspector.render(host, payload);
    }

    async function probePixel(lat, lon) {
        if (!state.jobResult || !state.activeJobId) {
            notify("warning", "Run a super-resolution job before probing pixels.");
            return;
        }
        selectResultTab("rtab-probe");
        elements.probeContent.innerHTML = `<div class="empty-state"><div class="skeleton" style="height:120px;width:100%"></div></div>`;

        try {
            const resp = await fetch(`/api/sr/jobs/${state.activeJobId}/pixel?lat=${lat}&lon=${lon}`);
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ detail: resp.statusText }));
                throw new Error(err.detail || "Pixel probe failed.");
            }
            const payload = await resp.json();
            renderProbePane(payload);
            placeProbeMarker(lat, lon);
        } catch (e) {
            elements.probeContent.innerHTML = emptyState(e.message);
        }
    }

    function placeProbeMarker(lat, lon) {
        if (state.probeMarker) {
            state.map.removeLayer(state.probeMarker);
            state.probeMarker = null;
        }
        state.probeMarker = L.circleMarker([lat, lon], {
            radius: 6,
            color: "#ffffff",
            weight: 2,
            fillColor: cssVar("--primary-600", "#2563eb"),
            fillOpacity: 0.9,
            interactive: false,
        }).addTo(state.map);
    }

    // =========================================================================
    // 15. Magnifier loupe
    // =========================================================================
    function loupeLayers() {
        const out = [];
        try {
            if (state.baseLayer && state.baseLayer._url) {
                out.push(L.tileLayer(state.baseLayer._url, state.baseLayer.options));
            }
            if (state.jobResult && state.jobResult.leaflet_bounds) {
                const bounds = state.jobResult.leaflet_bounds;
                const url = state.viewMode === "lr_only" ? getLayerUrl("left") : getLayerUrl("right");
                if (url) out.push(L.imageOverlay(url, bounds, { opacity: 1.0, interactive: false }));
            }
        } catch (e) {
            console.warn("Loupe layer construction failed:", e);
        }
        return out;
    }

    function refreshLoupe() {
        if (NTRO.loupe && NTRO.loupe.isEnabled && NTRO.loupe.isEnabled()) {
            NTRO.loupe.refresh();
        }
    }

    function initLoupe() {
        if (!NTRO.loupe || !elements.btnLoupe) return;
        const ok = NTRO.loupe.attach(state.map, {
            container: elements.mapViewport,
            buildLayers: loupeLayers,
            zoomOffset: 3,
            size: 220,
        });
        if (!ok) {
            elements.btnLoupe.disabled = true;
            return;
        }
        elements.btnLoupe.addEventListener("click", toggleLoupe);
    }

    function toggleLoupe() {
        if (!NTRO.loupe) return;
        const on = NTRO.loupe.toggle();
        elements.btnLoupe.setAttribute("aria-pressed", on ? "true" : "false");
        elements.btnLoupe.classList.toggle("active", on);
    }

    // =========================================================================
    // 16. Session job history
    // =========================================================================
    function initHistory() {
        if (elements.btnHistory) elements.btnHistory.addEventListener("click", () => toggleHistory(true));
        if (elements.btnHistoryClose) elements.btnHistoryClose.addEventListener("click", () => toggleHistory(false));
        if (elements.historyScrim) elements.historyScrim.addEventListener("click", () => toggleHistory(false));
    }

    function toggleHistory(open) {
        const willOpen = open === undefined ? !state.historyOpen : open;
        state.historyOpen = willOpen;

        if (willOpen) {
            state.lastFocused = document.activeElement;
            elements.historyScrim.hidden = false;
            elements.historyDrawer.hidden = false;
            // Force a reflow so the transition runs from the off-screen position.
            void elements.historyDrawer.offsetWidth;
            elements.historyScrim.classList.add("open");
            elements.historyDrawer.classList.add("open");
            elements.btnHistory.setAttribute("aria-expanded", "true");
            refreshHistory();
            if (elements.btnHistoryClose) elements.btnHistoryClose.focus();
        } else {
            elements.historyScrim.classList.remove("open");
            elements.historyDrawer.classList.remove("open");
            elements.btnHistory.setAttribute("aria-expanded", "false");
            setTimeout(() => {
                if (!state.historyOpen) {
                    elements.historyScrim.hidden = true;
                    elements.historyDrawer.hidden = true;
                }
            }, 220);
            if (state.lastFocused && state.lastFocused.focus) state.lastFocused.focus();
        }
    }

    async function refreshHistory() {
        if (!elements.historyBody) return;
        try {
            const resp = await fetch("/api/sr/jobs?limit=25");
            if (!resp.ok) throw new Error(resp.statusText);
            const data = await resp.json();
            renderHistory(data.jobs || []);
        } catch (e) {
            elements.historyBody.innerHTML = `<div class="history-empty">Job history unavailable: ${escapeHtml(e.message)}</div>`;
        }
    }

    function renderHistory(jobs) {
        const host = elements.historyBody;
        host.innerHTML = "";
        if (!jobs.length) {
            host.innerHTML = `<div class="history-empty">No jobs yet in this session.<br>Select an area and press Upscale to get started.</div>`;
            return;
        }

        jobs.forEach((j) => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "history-item";
            if (j.job_id === state.activeJobId) btn.setAttribute("aria-current", "true");

            const when = new Date(j.created_at * 1000);
            const shape = j.output_shape ? `${j.output_shape[2]}×${j.output_shape[1]}` : "—";
            const statusPill = j.status === "completed"
                ? '<span class="pill pill-pass">done</span>'
                : j.status === "failed"
                    ? '<span class="pill pill-fail">failed</span>'
                    : `<span class="pill pill-warn">${escapeHtml(j.progress_percent)}%</span>`;

            btn.innerHTML = `
                <div class="history-row">
                    <span class="history-scene">${escapeHtml(j.scene_id || j.job_id)}</span>
                    ${statusPill}
                </div>
                <div class="history-meta">${escapeHtml(when.toLocaleTimeString())} · ${escapeHtml(j.model || "—")} · ${shape} px${j.processing_time_sec ? " · " + j.processing_time_sec + " s" : ""}</div>
                ${j.verdict ? `<div class="history-meta">QA: ${escapeHtml(j.verdict.replace(/-/g, " "))}</div>` : ""}
                ${j.error_message ? `<div class="history-meta" style="color:var(--accent-red)">${escapeHtml(j.error_message.substring(0, 90))}</div>` : ""}`;

            if (j.status === "completed") {
                btn.addEventListener("click", () => loadHistoricJob(j.job_id));
            } else {
                btn.disabled = true;
            }
            host.appendChild(btn);
        });
    }

    async function loadHistoricJob(jobId) {
        try {
            const resp = await fetch(`/api/sr/jobs/${jobId}`);
            if (!resp.ok) throw new Error(resp.statusText);
            const job = await resp.json();
            if (job.status !== "completed") throw new Error("That job has not finished.");
            state.activeJobId = jobId;
            onJobCompleted(job);
            toggleHistory(false);
            zoomToPatch();
        } catch (e) {
            notify("error", `Could not reload that job: ${e.message}`);
        }
    }

    // =========================================================================
    // 17. Theme, shortcuts and sidebar chrome
    // =========================================================================
    function initTheme() {
        if (!NTRO.theme) return;
        NTRO.theme.init({ toggleSelector: "#btn-theme" });
        NTRO.theme.onChange(() => {
            // Re-render the charts so they pick up the new token values.
            if (state.analysis) {
                renderQualityPane();
                renderAppsPane();
                renderLegend();
            }
        });
    }

    function initSidebarToggle() {
        if (!elements.btnSidebarToggle) return;
        elements.btnSidebarToggle.addEventListener("click", toggleSidebar);
    }

    function toggleSidebar() {
        state.sidebarCollapsed = !state.sidebarCollapsed;
        elements.appMain.classList.toggle("sidebar-collapsed", state.sidebarCollapsed);
        elements.btnSidebarToggle.setAttribute("aria-expanded", state.sidebarCollapsed ? "false" : "true");
        elements.btnSidebarToggle.style.transform = state.sidebarCollapsed
            ? "translate(50%, -50%) rotate(180deg)"
            : "translate(50%, -50%)";
        setTimeout(() => {
            if (state.map) state.map.invalidateSize();
            updateSplitClipping();
        }, 220);
    }

    function initShortcuts() {
        if (!NTRO.shortcuts) return;
        NTRO.shortcuts.init();

        const reg = (combo, group, description, handler) =>
            NTRO.shortcuts.register(combo, { group: group, description: description, handler: handler });

        reg("d", "Area of interest", "Toggle draw-box mode", () => toggleDrawMode(!state.isDrawingAoi));
        reg("c", "Area of interest", "Create a patch at the map centre", () => elements.btnCenterPatch && elements.btnCenterPatch.click());
        reg("escape", "Area of interest", "Cancel drawing / close panels", () => {
            if (state.isDrawingAoi) { toggleDrawMode(false); return; }
            if (state.historyOpen) { toggleHistory(false); return; }
            if (state.inspectArmed) { setInspectArmed(false); return; }
        });

        reg("u", "Processing", "Upscale the selected patch", () => {
            if (!elements.btnRunSr.disabled) elements.btnRunSr.click();
        });
        reg("shift+s", "Processing", "Load the bundled sample scene", () => elements.btnLoadDemo && elements.btnLoadDemo.click());

        reg("1", "Map view", "Swipe split comparison", () => setViewMode("split"));
        reg("2", "Map view", "Show the native 10 m image", () => setViewMode("lr_only"));
        reg("3", "Map view", "Show the 2.5 m super-resolved image", () => setViewMode("sr_only"));
        reg("4", "Map view", "Opacity blend", () => setViewMode("blend"));
        reg("arrowleft", "Map view", "Move the swipe divider left", () => nudgeSlider(-0.02));
        reg("arrowright", "Map view", "Move the swipe divider right", () => nudgeSlider(0.02));
        reg("z", "Map view", "Zoom to the processed patch", zoomToPatch);
        reg("l", "Map view", "Toggle the magnifier loupe", toggleLoupe);
        reg("i", "Map view", "Toggle the spectral probe", () => setInspectArmed(!state.inspectArmed));
        reg("n", "Map view", "Cycle to the next layer", () => cycleLayer(1));
        reg("shift+n", "Map view", "Cycle to the previous layer", () => cycleLayer(-1));

        reg("h", "Workspace", "Open the session job history", () => toggleHistory());
        reg("t", "Workspace", "Cycle the colour theme", () => NTRO.theme && NTRO.theme.toggle());
        reg("\\", "Workspace", "Collapse or expand the side panel", toggleSidebar);
        reg("?", "Workspace", "Show this shortcut list", () => NTRO.shortcuts.toggleHelp());

        if (elements.btnHelp) {
            elements.btnHelp.addEventListener("click", () => NTRO.shortcuts.showHelp());
        }
    }

    function cycleLayer(direction) {
        const sel = elements.selectLayer;
        if (!sel || !sel.options.length) return;
        let idx = sel.selectedIndex + direction;
        if (idx < 0) idx = sel.options.length - 1;
        if (idx >= sel.options.length) idx = 0;
        sel.selectedIndex = idx;
        const opt = sel.options[idx];
        setActiveLayer(opt.value, opt.dataset.kind || "composite");
    }

    function initInspectToggle() {
        if (elements.btnInspect) {
            elements.btnInspect.addEventListener("click", () => setInspectArmed(!state.inspectArmed));
        }
    }

    // =========================================================================
    // 18. Application Lifecycle Bootstrap
    // =========================================================================
    function startApp() {
        elements = getElements();
        initTheme();
        initMap();
        initQuickLocations();
        initTabs();
        initAoiControls();
        initUpload();
        initCatalog();
        initDemo();
        initModelSelector();
        initSrExecution();
        initSliderAndModes();
        initResultTabs();
        initHistory();
        initSidebarToggle();
        initInspectToggle();
        initLoupe();
        initShortcuts();
        loadLayerCatalog();
        renderProbePane(null);

        if (state.map) {
            state.map.on("move", updateSplitClipping);
            state.map.on("zoom", updateSplitClipping);
            state.map.on("resize", updateSplitClipping);
        }
    }

    if (document.readyState === "complete" || document.readyState === "interactive") {
        setTimeout(startApp, 1);
    } else {
        document.addEventListener("DOMContentLoaded", startApp);
    }

    window.addEventListener("load", () => {
        if (!state.map) {
            startApp();
        } else {
            state.map.invalidateSize();
        }
    });

})();
