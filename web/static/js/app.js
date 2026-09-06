/**
 * NTRO-SRM Sentinel-2 Super-Resolution Mapping Web UI
 * Interactive Leaflet Map, AOI Manager, STAC Client, Split-Slider Visualizer,
 * 1-Click Auto-Patch Upscaling, and Direct GeoTIFF Upload
 */

(function () {
    "use strict";

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
        selectedModel: "lite", // "lite" | "lite-ft" | "swin2sr"
        activeJobId: null,
        jobPollTimer: null,
        colorMode: "rgb", // "rgb" | "cir"
        leftCompareMode: "lr", // "lr" | "bicubic" | "prev"
        viewMode: "split", // "split" | "lr_only" | "sr_only" | "blend"
        srOpacity: 100, // 0 to 100
        sliderPosRatio: 0.5, // 0.0 to 1.0
        isDraggingSlider: false,
        layers: {
            left: null,
            right: null,
        },
        jobResult: null,
        prevJobResult: null, // last completed run kept for cross-model compare
        compareRunning: false,
        diffShown: false,
        diffAmp: 0,
        analysisLayer: "rgb", // "rgb" | "confidence" | "novelty" | "ndvi"
        pixelMarker: null,
        pixelProbeRequest: 0,
        uploadedFile: null,
        userLocationMarker: null,
        userLocationCircle: null,
        isLocating: false,
    };

    // DOM Element References (initialized on start)
    let elements = {};

    function getElements() {
        return {
            map: document.getElementById("map"),
            mapViewport: document.getElementById("map-viewport"),
            drawBanner: document.getElementById("draw-banner"),
            btnCancelDraw: document.getElementById("btn-cancel-draw"),
            btnLoadDemo: document.getElementById("btn-load-demo"),
            btnDrawAoi: document.getElementById("btn-draw-aoi"),
            btnCenterPatch: document.getElementById("btn-center-patch"),
            btnClearAoi: document.getElementById("btn-clear-aoi"),
            btnSearchScenes: document.getElementById("btn-search-scenes"),
            btnRunSr: document.getElementById("btn-run-sr"),
            btnCompareModels: document.getElementById("btn-compare-models"),
            btnModeRgb: document.getElementById("btn-mode-rgb"),
            btnModeCir: document.getElementById("btn-mode-cir"),
            selectLeftLayer: document.getElementById("select-left-layer"),
            comparisonToggleGroup: document.getElementById("comparison-toggle-group"),
            viewModeGroup: document.getElementById("view-mode-group"),
            btnViewSplit: document.getElementById("btn-view-split"),
            btnViewLr: document.getElementById("btn-view-lr"),
            btnViewSr: document.getElementById("btn-view-sr"),
            btnViewBlend: document.getElementById("btn-view-blend"),
            blendSliderGroup: document.getElementById("blend-slider-group"),
            srOpacitySlider: document.getElementById("sr-opacity-slider"),
            srOpacityVal: document.getElementById("sr-opacity-val"),
            btnZoomPatch: document.getElementById("btn-zoom-patch"),
            btnToggleDiff: document.getElementById("btn-toggle-diff"),
            btnLocateMe: document.getElementById("btn-locate-me"),
            dateFrom: document.getElementById("date-from"),
            dateTo: document.getElementById("date-to"),
            cloudCover: document.getElementById("cloud-cover"),
            cloudVal: document.getElementById("cloud-val"),
            aoiBadge: document.getElementById("aoi-badge"),
            aoiCenter: document.getElementById("aoi-center"),
            aoiDims: document.getElementById("aoi-dims"),
            aoiPixels: document.getElementById("aoi-pixels"),
            aoiSrPixels: document.getElementById("aoi-sr-pixels"),
            aoiWarning: document.getElementById("aoi-warning"),
            coordDisplay: document.getElementById("coord-display"),
            aoiDisplayStatus: document.getElementById("aoi-display-status"),
            searchResultsContainer: document.getElementById("search-results-container"),
            searchPlaceholder: document.getElementById("search-placeholder"),
            searchLoading: document.getElementById("search-loading"),
            scenesList: document.getElementById("scenes-list"),
            selectedSceneCard: document.getElementById("selected-scene-card"),
            selectedSceneId: document.getElementById("selected-scene-id"),
            selectedSceneMeta: document.getElementById("selected-scene-meta"),
            progressCard: document.getElementById("progress-card"),
            progressFill: document.getElementById("progress-fill"),
            progressStepText: document.getElementById("progress-step-text"),
            progressPercentText: document.getElementById("progress-percent-text"),
            resultsCard: document.getElementById("results-card"),
            resCrs: document.getElementById("res-crs"),
            resTime: document.getElementById("res-time"),
            resDevice: document.getElementById("res-device"),
            resModel: document.getElementById("res-model"),
            resVram: document.getElementById("res-vram"),
            btnDownloadGeotiff: document.getElementById("btn-download-geotiff"),
            btnDownloadRgb: document.getElementById("btn-download-rgb"),
            btnDownloadCir: document.getElementById("btn-download-cir"),
            btnDownloadReport: document.getElementById("btn-download-report"),
            btnDownloadReportMd: document.getElementById("btn-download-report-md"),
            sliderContainer: document.getElementById("slider-container"),
            sliderDivider: document.getElementById("slider-divider"),
            compareLabels: document.getElementById("compare-labels"),
            labelLeftText: document.getElementById("label-left-text"),
            labelRightText: document.getElementById("label-right-text"),
            validationDock: document.getElementById("validation-dock"),
            validationVerdict: document.getElementById("validation-verdict"),
            validationScope: document.getElementById("validation-scope"),
            validationMetrics: document.getElementById("validation-metrics"),
            comparisonSummary: document.getElementById("comparison-summary"),
            analysisLayerSwitcher: document.getElementById("analysis-layer-switcher"),
            btnOpenReport: document.getElementById("btn-open-report"),
            validationModal: document.getElementById("validation-modal"),
            btnCloseReport: document.getElementById("btn-close-report"),
            validationModalSubtitle: document.getElementById("validation-modal-subtitle"),
            validationReportContent: document.getElementById("validation-report-content"),
            pixelInspector: document.getElementById("pixel-inspector"),
            btnClosePixel: document.getElementById("btn-close-pixel"),
            btnInspectCenter: document.getElementById("btn-inspect-center"),
            pixelCoordinate: document.getElementById("pixel-coordinate"),
            pixelSummary: document.getElementById("pixel-summary"),
            pixelSpectrum: document.getElementById("pixel-spectrum"),

            // Tabs
            tabBtns: document.querySelectorAll(".tab-btn"),
            tabContents: document.querySelectorAll(".tab-content"),

            // Upload
            uploadDropzone: document.getElementById("upload-dropzone"),
            fileUploadInput: document.getElementById("file-upload-input"),
            uploadFileCard: document.getElementById("upload-file-card"),
            uploadFilename: document.getElementById("upload-filename"),
            uploadFileSize: document.getElementById("upload-file-size"),
            uploadFileDims: document.getElementById("upload-file-dims"),
            uploadFileBands: document.getElementById("upload-file-bands"),
            uploadFileCrs: document.getElementById("upload-file-crs"),
            btnRunUploadSr: document.getElementById("btn-run-upload-sr"),

            // Manual Coordinates
            coordMinLon: document.getElementById("coord-min-lon"),
            coordMinLat: document.getElementById("coord-min-lat"),
            coordMaxLon: document.getElementById("coord-max-lon"),
            coordMaxLat: document.getElementById("coord-max-lat"),
            btnApplyCoords: document.getElementById("btn-apply-coords"),
        };
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

        // 5. OpenStreetMap Standard
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
            "🌐 OpenStreetMap": osmStreets,
        };

        L.control.layers(baseMaps, null, { position: "topright" }).addTo(state.map);

        // Mouse Coordinates Tracker
        state.map.on("mousemove", (e) => {
            const lat = e.latlng.lat.toFixed(4);
            const lon = e.latlng.lng.toFixed(4);
            const zoom = state.map.getZoom();
            if (elements.coordDisplay) {
                elements.coordDisplay.textContent = `Lat: ${lat}, Lon: ${lon} • Zoom: ${zoom}`;
            }
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
            elements.aoiBadge.textContent = name || "Selected";
            elements.aoiBadge.className = "badge badge-success";
        }
        if (elements.aoiDisplayStatus) {
            elements.aoiDisplayStatus.textContent = `${name} selected`;
        }

        // Switch to Tab 1 if currently on another tab
        const tabBtnSelect = document.getElementById("tab-btn-select");
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
    // 2b. User Geolocation — center map + AOI on device location
    // =========================================================================
    function initUserLocation() {
        if (!elements.btnLocateMe) return;
        elements.btnLocateMe.addEventListener("click", () => {
            if (!state.map || state.isLocating) return;

            if (!("geolocation" in navigator)) {
                if (elements.aoiDisplayStatus) {
                    elements.aoiDisplayStatus.textContent = "Geolocation not supported in this browser";
                }
                return;
            }

            state.isLocating = true;
            elements.btnLocateMe.disabled = true;
            elements.btnLocateMe.classList.add("locating");
            const originalLabel = elements.btnLocateMe.innerHTML;
            elements.btnLocateMe.innerHTML = "Locating…";
            if (elements.aoiDisplayStatus) {
                elements.aoiDisplayStatus.textContent = "Locating… allow browser location access";
            }

            const resetButton = () => {
                state.isLocating = false;
                if (elements.btnLocateMe) {
                    elements.btnLocateMe.disabled = false;
                    elements.btnLocateMe.classList.remove("locating");
                    elements.btnLocateMe.innerHTML = originalLabel;
                }
            };

            navigator.geolocation.getCurrentPosition(
                (pos) => {
                    resetButton();
                    centerOnUserLocation(pos.coords.latitude, pos.coords.longitude, pos.coords.accuracy);
                },
                (err) => {
                    resetButton();
                    let msg = "Unable to get your location";
                    if (err && err.code === 1) {
                        msg = "Location permission denied — allow access in browser site settings";
                    } else if (err && err.code === 2) {
                        msg = "Location unavailable — check GPS/network connection";
                    } else if (err && err.code === 3) {
                        msg = "Location request timed out — try again";
                    }
                    if (elements.aoiDisplayStatus) {
                        elements.aoiDisplayStatus.textContent = msg;
                    }
                },
                { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 }
            );
        });
    }

    function centerOnUserLocation(lat, lon, accuracy) {
        if (!state.map || isNaN(lat) || isNaN(lon)) return;
        clearOverlays();

        document.querySelectorAll(".preset-chip").forEach((el) => el.classList.remove("active"));

        const deltaLat = (1.5 / 110.574) / 2;
        const deltaLon = (1.5 / (111.32 * Math.cos((lat * Math.PI) / 180))) / 2;
        const bounds = L.latLngBounds(
            [lat - deltaLat, lon - deltaLon],
            [lat + deltaLat, lon + deltaLon]
        );

        state.map.setView([lat, lon], 14, { animate: true });
        setAoiFromBounds(bounds);

        if (state.userLocationMarker) {
            state.map.removeLayer(state.userLocationMarker);
            state.userLocationMarker = null;
        }
        if (state.userLocationCircle) {
            state.map.removeLayer(state.userLocationCircle);
            state.userLocationCircle = null;
        }

        if (accuracy && isFinite(accuracy)) {
            state.userLocationCircle = L.circle([lat, lon], {
                radius: Math.max(accuracy, 10),
                color: "#007aff",
                weight: 1.5,
                fillColor: "#007aff",
                fillOpacity: 0.12,
            }).addTo(state.map);
        }
        state.userLocationMarker = L.circleMarker([lat, lon], {
            radius: 8,
            color: "#ffffff",
            weight: 2.5,
            fillColor: "#007aff",
            fillOpacity: 1,
        }).addTo(state.map);

        if (elements.aoiBadge) {
            elements.aoiBadge.textContent = "My location";
            elements.aoiBadge.className = "badge badge-success";
        }
        if (elements.aoiDisplayStatus) {
            elements.aoiDisplayStatus.textContent = "Centered on your location";
        }

        const tabBtnSelect = document.getElementById("tab-btn-select");
        if (tabBtnSelect && !tabBtnSelect.classList.contains("active")) {
            tabBtnSelect.click();
        }
    }

    // =========================================================================
    // 3. Tab Navigation
    // =========================================================================
    function initTabs() {
        const upscaleCard = document.getElementById("upscale-action-card");
        const sceneCard = document.getElementById("selected-scene-card");

        elements.tabBtns.forEach((btn) => {
            btn.addEventListener("click", () => {
                elements.tabBtns.forEach((b) => b.classList.remove("active"));
                elements.tabContents.forEach((c) => c.classList.remove("active"));

                btn.classList.add("active");
                const tabId = btn.getAttribute("data-tab");
                const content = document.getElementById(tabId);
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
            container.style.cursor = "crosshair";
            state.map.dragging.disable();
            elements.btnDrawAoi.classList.add("drawing-active");
            elements.btnDrawAoi.innerHTML = `
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect></svg>
                Exit Draw Mode
            `;
            elements.drawBanner.classList.remove("hidden");
            elements.aoiDisplayStatus.textContent = "Draw mode active • Drag on map to define patch";
        } else {
            container.style.cursor = "";
            state.map.dragging.enable();
            elements.btnDrawAoi.classList.remove("drawing-active");
            elements.btnDrawAoi.innerHTML = `
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>
                Draw Box on Map
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
        setRunButtonsDisabled(false);
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

        elements.aoiBadge.textContent = "Selected";
        elements.aoiBadge.className = "badge badge-success";
        elements.aoiCenter.textContent = `${centerLat.toFixed(4)}, ${centerLon.toFixed(4)}`;
        elements.aoiDims.textContent = `${widthKm.toFixed(2)} × ${heightKm.toFixed(2)} km (${areaKm2.toFixed(1)} km²)`;
        elements.aoiPixels.textContent = `${wPx} × ${hPx} px (~${totalPx.toLocaleString()} px)`;
        if (elements.aoiSrPixels) {
            elements.aoiSrPixels.textContent = `${wSr} × ${hSr} px`;
        }

        const maxPx = 512 * 512;
        if (totalPx > maxPx) {
            elements.aoiWarning.textContent = `This area contains about ${totalPx.toLocaleString()} pixels. Draw a smaller region.`;
            elements.aoiWarning.classList.remove("hidden");
            setRunButtonsDisabled(true);
        } else {
            elements.aoiWarning.classList.add("hidden");
            setRunButtonsDisabled(false);
        }

        elements.aoiDisplayStatus.textContent = `${wPx}×${hPx} px at 10 m → ${wSr}×${hSr} px at 2.5 m`;
    }

    // Clear AOI
    function initAoiControls() {
        if (elements.btnClearAoi) {
            elements.btnClearAoi.addEventListener("click", () => {
                if (state.aoiLayer) {
                    state.map.removeLayer(state.aoiLayer);
                    state.aoiLayer = null;
                }
                state.currentAoi = null;
                state.selectedScene = null;
                document.querySelectorAll(".preset-chip").forEach((c) => c.classList.remove("active"));
                elements.aoiBadge.textContent = "Not selected";
                elements.aoiBadge.className = "badge badge-gray";
                elements.aoiCenter.textContent = "--";
                elements.aoiDims.textContent = "--";
                elements.aoiPixels.textContent = "--";
                if (elements.aoiSrPixels) elements.aoiSrPixels.textContent = "--";
                elements.aoiWarning.classList.add("hidden");
                setRunButtonsDisabled(true);
                elements.selectedSceneCard.classList.add("hidden");
                elements.aoiDisplayStatus.textContent = "Draw an area or choose a location";
            });
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
                    alert("Please enter valid decimal coordinates.");
                    return;
                }
                if (minLon >= maxLon || minLat >= maxLat) {
                    alert("Min coordinates must be strictly smaller than Max coordinates.");
                    return;
                }

                const bounds = L.latLngBounds([minLat, minLon], [maxLat, maxLon]);
                state.map.fitBounds(bounds, { padding: [40, 40] });
                setAoiFromBounds(bounds);
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
                elements.progressCard.classList.remove("hidden");
                elements.resultsCard.classList.add("hidden");
                if (elements.validationDock) elements.validationDock.classList.add("hidden");
                closePixelInspector();
                clearOverlays();
                resetProgressSteps();
                updateProgressUI(10, "Uploading GeoTIFF to backend server...");

                const formData = new FormData();
                formData.append("file", state.uploadedFile);

                try {
                    const uploadQuery = new URLSearchParams({
                        model: state.selectedModel,
                        run_analysis: "true",
                        run_wald_validation: "true",
                        uncertainty_members: "0",
                    });
                    const resp = await fetch(`/api/sr/upload?${uploadQuery.toString()}`, {
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

                    pollJobStatus(state.activeJobId);
                } catch (err) {
                    updateProgressUI(100, `Upload error: ${err.message}`);
                    elements.btnRunUploadSr.disabled = false;
                }
            });
        }
    }

    function handleUploadedFile(file) {
        if (!file.name.toLowerCase().endsWith(".tif") && !file.name.toLowerCase().endsWith(".tiff")) {
            alert("Please select a GeoTIFF image (.tif or .tiff).");
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
                    alert("Please define an Area of Interest (AOI) on the map first.");
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
                } catch (err) {
                    elements.searchLoading.classList.add("hidden");
                    elements.searchPlaceholder.classList.remove("hidden");
                    elements.searchPlaceholder.textContent = `Error: ${err.message}`;
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
                    <span class="scene-date">${dateStr}</span>
                    <span class="scene-cloud">Cloud: ${s.cloud_cover}% • ${s.id.substring(0, 22)}...</span>
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
            setRunButtonsDisabled(false);
        }
    }

    // =========================================================================
    // 7. Demo Scene Workflow
    // =========================================================================
    function initDemo() {
        if (!elements.btnLoadDemo) return;

        elements.btnLoadDemo.addEventListener("click", async () => {
            elements.btnLoadDemo.disabled = true;
            elements.btnLoadDemo.textContent = "Opening sample…";

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
                alert(`Failed to load demo scene: ${err.message}`);
            } finally {
                elements.btnLoadDemo.disabled = false;
                elements.btnLoadDemo.innerHTML = `
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                    Open sample
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

    async function launchSrProcessing(isDemo = false) {
        setRunButtonsDisabled(true);
        elements.progressCard.classList.remove("hidden");
        elements.resultsCard.classList.add("hidden");
        if (elements.validationDock) elements.validationDock.classList.add("hidden");
        closePixelInspector();
        clearOverlays();

        resetProgressSteps();
        updateProgressUI(5, "Preparing image…");

        const sceneId = isDemo ? null : (state.selectedScene ? state.selectedScene.id : "auto");

        const payload = {
            aoi: isDemo ? null : state.currentAoi,
            scene_id: sceneId,
            is_demo: isDemo,
            model: state.selectedModel,
            overlap: 32,
            clamp_output: true,
            run_analysis: true,
            run_wald_validation: true,
            uncertainty_members: 0,
        };

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
            updateProgressUI(100, `Error: ${err.message}`);
            setRunButtonsDisabled(false);
        }
    }

    // =========================================================================
    // 8b. One-Click Compare: Lite vs Lite-FT on the same AOI, then auto-swipe
    // =========================================================================
    function runSrJobOnce(payload, phaseLabel) {
        return new Promise((resolve, reject) => {
            fetch("/api/sr/process", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            }).then(async (resp) => {
                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
                    throw new Error(err.detail || "Failed to start super-resolution job.");
                }
                const data = await resp.json();
                state.activeJobId = data.job_id;
                updateProgressUI(10, `${phaseLabel}: processing…`);
                pollJobStatus(data.job_id, (job, err) => {
                    if (err || !job) {
                        reject(err || new Error("Job failed."));
                    } else {
                        resolve(job);
                    }
                });
            }).catch(reject);
        });
    }

    async function compareLiteVsFt() {
        if (!state.map || state.compareRunning) return;
        if (!state.currentAoi || (elements.btnRunSr && elements.btnRunSr.disabled)) {
            alert("Please define an Area of Interest (AOI) on the map first.");
            return;
        }
        state.compareRunning = true;
        setRunButtonsDisabled(true);
        elements.progressCard.classList.remove("hidden");
        elements.resultsCard.classList.add("hidden");
        if (elements.validationDock) elements.validationDock.classList.add("hidden");
        closePixelInspector();
        clearOverlays();
        resetProgressSteps();

        const sceneId = state.selectedScene ? state.selectedScene.id : "auto";
        const useDemoScene = sceneId === "DEMO_MLBS_20180825_S2L2A";
        const variants = ["lite", "lite-ft"];
        const done = {};
        try {
            for (let i = 0; i < variants.length; i++) {
                const variant = variants[i];
                updateProgressUI(5, `Compare ${i + 1}/2: launching ${displayNameForModel(variant)}…`);
                const job = await runSrJobOnce({
                    aoi: state.currentAoi,
                    scene_id: useDemoScene ? null : sceneId,
                    is_demo: useDemoScene,
                    model: variant,
                    overlap: 32,
                    clamp_output: true,
                    run_analysis: true,
                    run_wald_validation: true,
                    uncertainty_members: 0,
                }, `Compare ${i + 1}/2 ${displayNameForModel(variant)}`);
                done[variant] = job;
            }
        } catch (err) {
            updateProgressUI(100, `Compare failed: ${err.message}`);
            state.compareRunning = false;
            setRunButtonsDisabled(false);
            return;
        }

        // Present FT on the right, Lite on the left, swipe ready.
        onJobCompleted({ job_id: done["lite-ft"].job_id, result: done["lite-ft"].result });
        state.prevJobResult = done["lite"].result;
        state.leftCompareMode = "prev";
        if (elements.selectLeftLayer) elements.selectLeftLayer.value = "prev";
        refreshPrevRunOption();
        if (state.layers.left) {
            state.layers.left.setUrl(getLayerUrl("prev", state.colorMode));
        }
        updateLabelTexts();
        renderValidationDock();
        state.compareRunning = false;
        setRunButtonsDisabled(false);
    }

    function initCompare() {
        if (elements.btnCompareModels) {
            elements.btnCompareModels.addEventListener("click", () => {
                compareLiteVsFt();
            });
        }
    }

    function setRunButtonsDisabled(disabled) {
        if (elements.btnRunSr) elements.btnRunSr.disabled = disabled;
        if (elements.btnCompareModels) elements.btnCompareModels.disabled = disabled;
    }

    function pollJobStatus(jobId, onDone) {
        if (state.jobPollTimer) clearInterval(state.jobPollTimer);

        state.jobPollTimer = setInterval(async () => {
            try {
                const resp = await fetch(`/api/sr/jobs/${jobId}`);
                if (!resp.ok) return;
                const job = await resp.json();

                updateProgressUI(job.progress_percent, job.progress_step);

                if (job.status === "completed") {
                    clearInterval(state.jobPollTimer);
                    if (onDone) {
                        onDone(job);
                    } else {
                        onJobCompleted(job);
                    }
                } else if (job.status === "failed") {
                    clearInterval(state.jobPollTimer);
                    const errMsg = job.error_message || "Unknown error";
                    updateProgressUI(100, `Failed: ${errMsg}`);
                    setRunButtonsDisabled(false);
                    if (elements.btnRunUploadSr) elements.btnRunUploadSr.disabled = false;
                    if (onDone) {
                        onDone(null, new Error(errMsg));
                    }
                }
            } catch (e) {
                console.error("Polling error:", e);
            }
        }, 500);
    }

    function resetProgressSteps() {
        ["step-aoi", "step-stream", "step-model", "step-geotiff", "step-visual"].forEach((id) => {
            const el = document.getElementById(id);
            if (el) el.className = "step-item";
        });
    }

    function updateProgressUI(pct, stepMsg) {
        elements.progressFill.style.width = `${pct}%`;
        elements.progressPercentText.textContent = `${pct}%`;
        elements.progressStepText.textContent = stepMsg;

        if (pct >= 10) markStep("step-aoi", true);
        if (pct >= 30) markStep("step-stream", true);
        if (pct >= 55) markStep("step-model", true);
        if (pct >= 85) markStep("step-geotiff", true);
        if (pct >= 100) markStep("step-visual", true);
    }

    function markStep(stepId, isDone) {
        const el = document.getElementById(stepId);
        if (el) {
            el.className = isDone ? "step-item done" : "step-item";
        }
    }

    const QUALITY_METRICS = [
        { key: "psnr_db", label: "PSNR", unit: "dB", digits: 2, better: "higher", help: "Signal fidelity against the observed 10 m reference. Higher is better." },
        { key: "ssim", label: "SSIM", unit: "", digits: 3, better: "higher", help: "Structural similarity to the observed reference. 1.0 is a perfect match." },
        { key: "sam_deg", label: "SAM", unit: "°", digits: 2, better: "lower", help: "Spectral angle between reconstructed and observed pixels. Lower is better." },
        { key: "ergas", label: "ERGAS", unit: "%", digits: 2, better: "lower", help: "Relative global reconstruction error. Lower is better." },
    ];

    function escapeHtml(value) {
        return String(value == null ? "" : value)
            .replace(/&/g, "&amp;").replace(/</g, "&lt;")
            .replace(/>/g, "&gt;").replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function finiteNumber(value) {
        if (value === null || value === undefined || value === "") return null;
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : null;
    }

    function formatMetric(value, spec) {
        const n = finiteNumber(value);
        if (n === null) return "—";
        return `${n.toFixed(spec.digits)}${spec.unit ? ` ${spec.unit}` : ""}`;
    }

    function qualityBlock(result) {
        const analysis = (result && result.analysis) || {};
        const wald = analysis.wald || null;
        const consistency = analysis.consistency || null;
        return {
            analysis,
            metrics: (wald && wald.metrics) || (consistency && consistency.metrics) || {},
            scope: wald ? "Wald · 40 m → 10 m" : "Source consistency · 10 m",
            isWald: !!wald,
        };
    }

    function comparisonFor(current, previous) {
        if (!current || !previous || !sameBounds(current.leaflet_bounds, previous.leaflet_bounds)) return null;
        const currentQuality = qualityBlock(current);
        const previousQuality = qualityBlock(previous);
        if (currentQuality.isWald !== previousQuality.isWald) return null;
        if (!Object.keys(currentQuality.metrics).length || !Object.keys(previousQuality.metrics).length) return null;

        let currentWins = 0;
        let previousWins = 0;
        const metrics = QUALITY_METRICS.map((spec) => {
            const currentValue = finiteNumber(currentQuality.metrics[spec.key]);
            const previousValue = finiteNumber(previousQuality.metrics[spec.key]);
            if (currentValue === null || previousValue === null) return { spec, currentValue, previousValue, winner: "none" };
            const delta = currentValue - previousValue;
            const threshold = Math.max(Math.abs(currentValue), Math.abs(previousValue), 1) * 1e-5;
            let winner = "tie";
            if (Math.abs(delta) > threshold) {
                const currentBetter = spec.better === "higher" ? delta > 0 : delta < 0;
                winner = currentBetter ? "current" : "previous";
                if (currentBetter) currentWins += 1;
                else previousWins += 1;
            }
            return { spec, currentValue, previousValue, delta, winner };
        });
        return { metrics, currentWins, previousWins, currentQuality, previousQuality };
    }

    function verdictText(summary, hasWald) {
        const raw = String((summary && summary.verdict) || "").toLowerCase();
        if (raw.includes("validated")) return raw.includes("caveat") ? "Validated · caveats" : "Validated";
        if (hasWald) return "Validation complete";
        return "Consistency checked";
    }

    function renderValidationDock() {
        if (!state.jobResult || !elements.validationDock) return;
        const current = qualityBlock(state.jobResult);
        const summary = current.analysis.summary || {};
        const uncertainty = current.analysis.uncertainty || {};
        const comparison = comparisonFor(state.jobResult, state.prevJobResult);

        elements.validationVerdict.textContent = verdictText(summary, current.isWald);
        elements.validationVerdict.className = `validation-verdict ${current.isWald ? "is-validated" : "is-limited"}`;
        elements.validationScope.textContent = current.scope;

        const reliability = finiteNumber(uncertainty.reliability_score);
        const cards = QUALITY_METRICS.map((spec) => {
            const value = current.metrics[spec.key];
            const compared = comparison && comparison.metrics.find((item) => item.spec.key === spec.key);
            let deltaHtml = "";
            if (compared && finiteNumber(compared.delta) !== null) {
                const sign = compared.delta > 0 ? "+" : "";
                const deltaText = `${sign}${compared.delta.toFixed(spec.digits)}`;
                const deltaClass = compared.winner === "current" ? "is-better" : (compared.winner === "previous" ? "is-worse" : "is-even");
                deltaHtml = `<span class="metric-delta ${deltaClass}">Δ ${deltaText}</span>`;
            }
            return `<div class="validation-metric" title="${escapeHtml(spec.help)}">
                <div class="metric-label">${spec.label}</div>
                <div class="metric-value">${formatMetric(value, spec)}</div>${deltaHtml}
            </div>`;
        });
        cards.push(`<div class="validation-metric" title="Scene-level confidence derived from reconstruction novelty.">
            <div class="metric-label">Reliability</div>
            <div class="metric-value">${reliability === null ? "—" : `${reliability.toFixed(1)}%`}</div>
            <span class="metric-delta risk-${escapeHtml(uncertainty.hallucination_risk || "unknown")}">${escapeHtml(uncertainty.hallucination_risk || "unrated")} risk</span>
        </div>`);
        elements.validationMetrics.innerHTML = cards.join("");

        if (comparison) {
            const currentName = state.jobResult.model || "Current run";
            const previousName = state.prevJobResult.model || "Previous run";
            let leadText = "The runs are tied across the headline metrics.";
            if (comparison.currentWins > comparison.previousWins) {
                leadText = `${currentName} leads ${comparison.currentWins} of ${QUALITY_METRICS.length} metrics.`;
            } else if (comparison.previousWins > comparison.currentWins) {
                leadText = `${previousName} leads ${comparison.previousWins} of ${QUALITY_METRICS.length} metrics.`;
            }
            elements.comparisonSummary.innerHTML = `<span class="comparison-icon">↗</span><span><b>${escapeHtml(leadText)}</b> Deltas compare the right-side run with the left.</span>`;
            elements.comparisonSummary.classList.remove("hidden");
        } else {
            elements.comparisonSummary.classList.add("hidden");
            elements.comparisonSummary.innerHTML = "";
        }

        elements.validationDock.classList.remove("hidden");
        renderValidationReport();
    }

    function metricTableRows(metrics, previousMetrics) {
        return QUALITY_METRICS.map((spec) => {
            const currentValue = finiteNumber(metrics && metrics[spec.key]);
            const previousValue = finiteNumber(previousMetrics && previousMetrics[spec.key]);
            const delta = currentValue !== null && previousValue !== null ? currentValue - previousValue : null;
            const better = delta === null ? "" : ((spec.better === "higher" ? delta > 0 : delta < 0) ? "is-better" : "is-worse");
            return `<tr><th>${spec.label}<small>${escapeHtml(spec.help)}</small></th>
                ${previousMetrics ? `<td>${formatMetric(previousValue, spec)}</td>` : ""}
                <td><b>${formatMetric(currentValue, spec)}</b></td>
                ${previousMetrics ? `<td class="${better}">${delta === null ? "—" : `${delta > 0 ? "+" : ""}${delta.toFixed(spec.digits)}`}</td>` : ""}
            </tr>`;
        }).join("");
    }

    function renderValidationReport() {
        if (!state.jobResult || !elements.validationReportContent) return;
        const current = qualityBlock(state.jobResult);
        const previous = comparisonFor(state.jobResult, state.prevJobResult);
        const consistency = current.analysis.consistency || {};
        const uncertainty = current.analysis.uncertainty || {};
        const caveats = (current.analysis.summary && current.analysis.summary.caveats) || [];
        const perBand = (current.metrics && current.metrics.per_band) || [];
        const currentName = state.jobResult.model || "Current run";
        const previousName = state.prevJobResult && (state.prevJobResult.model || "Previous run");
        const previousMetrics = previous ? previous.previousQuality.metrics : null;

        elements.validationModalSubtitle.textContent = current.isWald
            ? "Quantitative Wald validation at 40 m → 10 m, plus 2.5 m source-consistency checks"
            : "2.5 m product checked by downsampling it to the observed 10 m source";

        const bandRows = perBand.map((band) => `<tr><th>${escapeHtml(band.band)}</th><td>${finiteNumber(band.psnr_db) === null ? "—" : band.psnr_db.toFixed(2)}</td><td>${finiteNumber(band.ssim) === null ? "—" : band.ssim.toFixed(3)}</td><td>${finiteNumber(band.rmse) === null ? "—" : band.rmse.toFixed(4)}</td><td>${finiteNumber(band.cc) === null ? "—" : band.cc.toFixed(3)}</td></tr>`).join("");
        const consistencyMetrics = consistency.metrics || {};
        const reliability = finiteNumber(uncertainty.reliability_score);
        const lowConfidence = finiteNumber(uncertainty.low_confidence_fraction);

        elements.validationReportContent.innerHTML = `
            <section class="report-callout">
                <div><span>Result</span><b>${escapeHtml(verdictText(current.analysis.summary, current.isWald))}</b></div>
                <div><span>Validation scope</span><b>${escapeHtml(current.scope)}</b></div>
                <div><span>Sampling density</span><b>10,000 → 160,000 px/km²</b></div>
            </section>
            <p class="report-method-note">${current.isWald
                ? "Accuracy metrics are measured by degrading the observed 10 m image to 40 m, reconstructing it to 10 m, and comparing against the real 10 m image. They indicate model accuracy without claiming unavailable 2.5 m ground truth."
                : "No direct high-resolution reference was available. The report therefore measures how faithfully the 2.5 m reconstruction returns to the observed 10 m source."}</p>
            <section class="report-section">
                <div class="report-section-title"><h3>${previous ? "Run benchmark" : "Headline metrics"}</h3><span>${escapeHtml(current.scope)}</span></div>
                <div class="report-table-wrap"><table class="report-table"><thead><tr><th>Metric</th>${previous ? `<th>${escapeHtml(previousName)}</th>` : ""}<th>${escapeHtml(currentName)}</th>${previous ? "<th>Δ</th>" : ""}</tr></thead><tbody>${metricTableRows(current.metrics, previousMetrics)}</tbody></table></div>
            </section>
            <section class="report-grid">
                <div class="report-stat"><span>Reliability</span><b>${reliability === null ? "—" : `${reliability.toFixed(1)}%`}</b><small>${escapeHtml(uncertainty.hallucination_risk || "unrated")} hallucination risk</small></div>
                <div class="report-stat"><span>Low-confidence pixels</span><b>${lowConfidence === null ? "—" : `${(lowConfidence * 100).toFixed(1)}%`}</b><small>confidence below 0.5</small></div>
                <div class="report-stat"><span>10 m consistency</span><b>${consistency.passed ? "Pass" : "Review"}</b><small>SAM ${finiteNumber(consistencyMetrics.sam_deg) === null ? "—" : `${consistencyMetrics.sam_deg.toFixed(2)}°`}</small></div>
            </section>
            ${bandRows ? `<section class="report-section"><div class="report-section-title"><h3>Per-band validation</h3><span>10 Sentinel-2 bands</span></div><div class="report-table-wrap"><table class="report-table compact"><thead><tr><th>Band</th><th>PSNR</th><th>SSIM</th><th>RMSE</th><th>Corr.</th></tr></thead><tbody>${bandRows}</tbody></table></div></section>` : ""}
            ${caveats.length ? `<section class="report-section caveat-section"><h3>Interpretation limits</h3><ul>${caveats.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul></section>` : ""}`;
    }

    function setAnalysisLayer(layer) {
        if (!state.jobResult || !state.layers.right) return;
        state.diffShown = false;
        if (elements.btnToggleDiff) elements.btnToggleDiff.classList.remove("active");
        state.analysisLayer = layer;
        const jobId = state.jobResult.job_id || state.activeJobId;
        let url;
        if (layer === "rgb") url = getLayerUrl("sr", state.colorMode);
        else url = `/api/sr/jobs/${encodeURIComponent(jobId)}/preview/sr_${encodeURIComponent(layer)}`;
        state.layers.right.setUrl(url);
        elements.analysisLayerSwitcher.querySelectorAll(".analysis-layer-btn").forEach((button) => {
            button.classList.toggle("active", button.dataset.layer === layer);
        });
        if (layer !== "rgb" && state.viewMode === "lr_only") setViewMode("split");
        updateLabelTexts();
    }

    function closePixelInspector() {
        if (elements.pixelInspector) elements.pixelInspector.classList.add("hidden");
        if (state.pixelMarker && state.map) state.map.removeLayer(state.pixelMarker);
        state.pixelMarker = null;
    }

    async function inspectPixel(latlng) {
        if (!state.jobResult || state.isDrawingAoi) return;
        const bounds = L.latLngBounds(state.jobResult.leaflet_bounds);
        if (!bounds.contains(latlng)) return;
        const requestId = ++state.pixelProbeRequest;
        if (state.pixelMarker) state.map.removeLayer(state.pixelMarker);
        state.pixelMarker = L.circleMarker(latlng, { radius: 6, color: "#fff", weight: 2, fillColor: "#007aff", fillOpacity: 1 }).addTo(state.map);
        elements.pixelInspector.classList.remove("hidden");
        elements.pixelCoordinate.textContent = `${latlng.lat.toFixed(5)}, ${latlng.lng.toFixed(5)}`;
        elements.pixelSummary.innerHTML = '<span class="pixel-loading">Sampling 10 bands…</span>';
        elements.pixelSpectrum.innerHTML = "";
        try {
            const jobId = state.jobResult.job_id || state.activeJobId;
            const params = new URLSearchParams({ lat: latlng.lat, lon: latlng.lng });
            const response = await fetch(`/api/sr/jobs/${encodeURIComponent(jobId)}/pixel?${params}`);
            if (!response.ok) throw new Error("Pixel could not be sampled");
            const data = await response.json();
            if (requestId !== state.pixelProbeRequest) return;
            const uncertainty = data.uncertainty || {};
            const confidence = finiteNumber(uncertainty.confidence);
            const indices = Object.fromEntries((data.indices || []).map((item) => [String(item.key).toLowerCase(), item]));
            const indexCards = ["ndvi", "ndwi"].filter((key) => indices[key]).map((key) => {
                const item = indices[key];
                return `<div><span>${key.toUpperCase()}</span><b>${finiteNumber(item.sr) === null ? "—" : item.sr.toFixed(3)}</b><small>${escapeHtml(item.class_label || "")}</small></div>`;
            }).join("");
            elements.pixelSummary.innerHTML = `<div class="pixel-confidence"><span>Confidence</span><b>${confidence === null ? "—" : `${(confidence * 100).toFixed(0)}%`}</b><small>${escapeHtml(uncertainty.risk || "unrated")} risk</small></div>${indexCards}`;

            const lr = (data.lr && data.lr.reflectance) || [];
            const sr = (data.sr && data.sr.reflectance) || [];
            const maxValue = Math.max(0.01, ...lr.map(Number), ...sr.map(Number));
            elements.pixelSpectrum.innerHTML = (data.band_names || []).map((band, index) => {
                const lrValue = finiteNumber(lr[index]) || 0;
                const srValue = finiteNumber(sr[index]) || 0;
                return `<div class="spectrum-row" title="${escapeHtml(band)} · observed ${lrValue.toFixed(4)} · reconstructed ${srValue.toFixed(4)}">
                    <span>${escapeHtml(band)}</span><div class="spectrum-bars"><i class="bar-observed" style="width:${Math.max(1, lrValue / maxValue * 100)}%"></i><i class="bar-reconstructed" style="width:${Math.max(1, srValue / maxValue * 100)}%"></i></div><b>${srValue.toFixed(3)}</b>
                </div>`;
            }).join("") + '<div class="spectrum-legend"><span><i class="observed"></i>Observed 10 m</span><span><i class="reconstructed"></i>Result 2.5 m</span></div>';
        } catch (error) {
            if (requestId === state.pixelProbeRequest) elements.pixelSummary.textContent = error.message;
        }
    }

    function initValidationUi() {
        if (elements.btnOpenReport) elements.btnOpenReport.addEventListener("click", () => elements.validationModal.classList.remove("hidden"));
        if (elements.btnCloseReport) elements.btnCloseReport.addEventListener("click", () => elements.validationModal.classList.add("hidden"));
        if (elements.validationModal) elements.validationModal.addEventListener("click", (event) => {
            if (event.target.dataset.closeReport) elements.validationModal.classList.add("hidden");
        });
        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape") elements.validationModal.classList.add("hidden");
        });
        if (elements.analysisLayerSwitcher) elements.analysisLayerSwitcher.addEventListener("click", (event) => {
            const button = event.target.closest(".analysis-layer-btn");
            if (button) setAnalysisLayer(button.dataset.layer);
        });
        if (elements.btnClosePixel) elements.btnClosePixel.addEventListener("click", closePixelInspector);
        if (elements.btnInspectCenter) elements.btnInspectCenter.addEventListener("click", () => {
            if (state.jobResult && state.jobResult.leaflet_bounds) {
                inspectPixel(L.latLngBounds(state.jobResult.leaflet_bounds).getCenter());
            }
        });
    }

    // =========================================================================
    // 9. Visual Overlay Rendering & Results Display
    // =========================================================================
    function onJobCompleted(job) {
        if (state.jobResult && state.jobResult !== job.result) {
            state.prevJobResult = state.jobResult;
        }
        state.jobResult = job.result;
        state.jobResult.job_id = state.jobResult.job_id || job.job_id;
        state.analysisLayer = "rgb";
        if (elements.analysisLayerSwitcher) {
            elements.analysisLayerSwitcher.querySelectorAll(".analysis-layer-btn").forEach((button) => {
                button.classList.toggle("active", button.dataset.layer === "rgb");
            });
        }
        state.diffShown = false;
        if (elements.btnToggleDiff) elements.btnToggleDiff.classList.remove("active");
        refreshPrevRunOption();
        setRunButtonsDisabled(false);
        if (elements.btnRunUploadSr) elements.btnRunUploadSr.disabled = false;

        // Populate Result Details
        elements.resCrs.textContent = job.result.crs || "EPSG:32617";
        elements.resTime.textContent = `${job.result.processing_time_sec} s`;
        elements.resDevice.textContent = job.result.device_used.toUpperCase();
        if (elements.resModel) {
            elements.resModel.textContent = job.result.model || displayNameForModel(state.selectedModel);
        }
        if (elements.resVram) {
            elements.resVram.textContent = job.result.peak_vram_mb ? `${job.result.peak_vram_mb} MB` : "N/A";
        }

        elements.btnDownloadGeotiff.href = `/api/sr/jobs/${job.job_id}/download/geotiff`;
        elements.btnDownloadRgb.href = `/api/sr/jobs/${job.job_id}/download/rgb`;
        elements.btnDownloadCir.href = `/api/sr/jobs/${job.job_id}/download/cir`;
        if (elements.btnDownloadReport) elements.btnDownloadReport.href = `/api/sr/jobs/${job.job_id}/download/report`;
        if (elements.btnDownloadReportMd) elements.btnDownloadReportMd.href = `/api/sr/jobs/${job.job_id}/download/report-md`;

        elements.resultsCard.classList.remove("hidden");
        elements.progressCard.classList.add("hidden");
        elements.viewModeGroup.style.display = "flex";
        elements.comparisonToggleGroup.style.display = "flex";

        // Display on Map
        displayJobLayers(job.result);
        renderValidationDock();
    }

    function displayJobLayers(result) {
        clearOverlays();

        const bounds = result.leaflet_bounds;
        state.map.fitBounds(bounds, { padding: [50, 50] });

        // Left Layer (Default: Native 10m LR RGB)
        const leftUrl = getLayerUrl(state.leftCompareMode, state.colorMode);
        state.layers.left = L.imageOverlay(leftUrl, bounds, {
            opacity: 1.0,
            interactive: true,
        }).addTo(state.map).on("click", (event) => inspectPixel(event.latlng));

        // Right Layer (Super-Resolved 2.5m SR RGB)
        const rightUrl = getLayerUrl("sr", state.colorMode);
        state.layers.right = L.imageOverlay(rightUrl, bounds, {
            opacity: 1.0,
            interactive: true,
        }).addTo(state.map).on("click", (event) => inspectPixel(event.latlng));

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
    }

    function sameBounds(a, b) {
        return !!a && !!b && JSON.stringify(a) === JSON.stringify(b);
    }

    function refreshPrevRunOption() {
        const opt = document.getElementById("opt-prev-layer");
        const usable = !!(
            state.prevJobResult &&
            state.jobResult &&
            state.prevJobResult.previews &&
            sameBounds(state.prevJobResult.leaflet_bounds, state.jobResult.leaflet_bounds)
        );
        if (opt) {
            opt.disabled = !usable;
            opt.textContent = usable
                ? `Previous: ${state.prevJobResult.model || "run"}`
                : "Previous run";
        }
        if (elements.btnToggleDiff) {
            elements.btnToggleDiff.disabled = !usable;
            if (!usable && state.diffShown) {
                hideDiffView();
            }
        }
        if (!usable && state.leftCompareMode === "prev") {
            state.leftCompareMode = "lr";
            if (elements.selectLeftLayer) elements.selectLeftLayer.value = "lr";
            if (state.jobResult && state.layers.left) {
                state.layers.left.setUrl(getLayerUrl("lr", state.colorMode));
            }
        }
        updateLabelTexts();
    }

    function getLayerUrl(type, mode) {
        if (type === "prev") {
            if (!state.prevJobResult) return "";
            const key = `sr_${mode}`;
            return state.prevJobResult.previews[key] || "";
        }
        if (!state.jobResult) return "";
        const key = `${type}_${mode}`;
        return state.jobResult.previews[key] || "";
    }

    // =========================================================================
    // 8c. Amplified difference heatmap (previous run vs current, client-side)
    // =========================================================================
    // Approx inferno ramp stops: [pos, r, g, b]
    const DIFF_RAMP = [
        [0.0, 0, 0, 4], [0.13, 31, 12, 72], [0.25, 63, 23, 108],
        [0.38, 95, 32, 120], [0.5, 129, 41, 107], [0.63, 165, 54, 82],
        [0.75, 201, 73, 52], [0.88, 230, 107, 31], [1.0, 252, 255, 164],
    ];

    function rampColor(t) {
        const x = Math.max(0, Math.min(1, t));
        for (let i = 1; i < DIFF_RAMP.length; i++) {
            if (x <= DIFF_RAMP[i][0]) {
                const [p0, r0, g0, b0] = DIFF_RAMP[i - 1];
                const [p1, r1, g1, b1] = DIFF_RAMP[i];
                const f = (x - p0) / Math.max(1e-6, p1 - p0);
                return [r0 + (r1 - r0) * f, g0 + (g1 - g0) * f, b0 + (b1 - b0) * f];
            }
        }
        return DIFF_RAMP[DIFF_RAMP.length - 1].slice(1);
    }

    function loadPreviewImage(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error("Could not load preview image."));
            img.src = url;
        });
    }

    async function buildDiffDataURL(urlPrev, urlCurr) {
        const [imgA, imgB] = await Promise.all([loadPreviewImage(urlPrev), loadPreviewImage(urlCurr)]);
        const w = Math.min(imgA.naturalWidth || imgA.width, imgB.naturalWidth || imgB.width);
        const h = Math.min(imgA.naturalHeight || imgA.height, imgB.naturalHeight || imgB.height);
        if (!w || !h) throw new Error("Empty preview image.");
        const read = (img) => {
            const c = document.createElement("canvas");
            c.width = w; c.height = h;
            const ctx = c.getContext("2d");
            ctx.drawImage(img, 0, 0, w, h);
            return ctx.getImageData(0, 0, w, h).data;
        };
        const dA = read(imgA);
        const dB = read(imgB);
        const n = w * h;
        const mag = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            const o = i * 4;
            mag[i] = (Math.abs(dA[o] - dB[o]) + Math.abs(dA[o + 1] - dB[o + 1]) + Math.abs(dA[o + 2] - dB[o + 2])) / (3 * 255);
        }
        const sorted = Array.from(mag).sort((a, b) => a - b);
        const p99 = sorted[Math.floor(0.99 * (n - 1))] || 0;
        const amp = 1 / Math.max(p99, 1e-6);
        state.diffAmp = Math.round(amp);
        const out = document.createElement("canvas");
        out.width = w; out.height = h;
        const octx = out.getContext("2d");
        const outImg = octx.createImageData(w, h);
        for (let i = 0; i < n; i++) {
            const [r, g, b] = rampColor(Math.min(1, mag[i] * amp));
            const o = i * 4;
            outImg.data[o] = r; outImg.data[o + 1] = g; outImg.data[o + 2] = b; outImg.data[o + 3] = 255;
        }
        octx.putImageData(outImg, 0, 0);
        return out.toDataURL("image/png");
    }

    async function toggleDiffView() {
        const btn = elements.btnToggleDiff;
        if (!btn || btn.disabled || state.compareRunning) return;
        if (state.diffShown) {
            hideDiffView();
            return;
        }
        btn.disabled = true;
        try {
            state.analysisLayer = "rgb";
            if (elements.analysisLayerSwitcher) {
                elements.analysisLayerSwitcher.querySelectorAll(".analysis-layer-btn").forEach((button) => {
                    button.classList.toggle("active", button.dataset.layer === "rgb");
                });
            }
            const dataUrl = await buildDiffDataURL(
                getLayerUrl("prev", state.colorMode),
                getLayerUrl("sr", state.colorMode)
            );
            state.diffShown = true;
            btn.classList.add("active");
            if (state.layers.right) state.layers.right.setUrl(dataUrl);
            updateLabelTexts();
        } catch (err) {
            if (elements.aoiDisplayStatus) {
                elements.aoiDisplayStatus.textContent = `Diff view unavailable: ${err.message}`;
            }
        } finally {
            btn.disabled = false;
        }
    }

    function hideDiffView() {
        state.diffShown = false;
        if (elements.btnToggleDiff) elements.btnToggleDiff.classList.remove("active");
        if (state.layers.right) {
            state.layers.right.setUrl(getLayerUrl("sr", state.colorMode));
        }
        updateLabelTexts();
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
        elements.sliderContainer.classList.add("hidden");
        elements.compareLabels.classList.add("hidden");
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

        const leftBbox = leftEl.getBoundingClientRect();
        const rightBbox = rightEl.getBoundingClientRect();

        const clipLeftRight = Math.max(0, splitX - leftBbox.left);
        leftEl.style.clipPath = `polygon(0 0, ${clipLeftRight}px 0, ${clipLeftRight}px 100%, 0 100%)`;

        const clipRightLeft = Math.max(0, splitX - rightBbox.left);
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

        // Color Mode & Layer Selector
        if (elements.btnModeRgb) elements.btnModeRgb.addEventListener("click", () => setColorMode("rgb"));
        if (elements.btnModeCir) elements.btnModeCir.addEventListener("click", () => setColorMode("cir"));

        if (elements.selectLeftLayer) {
            elements.selectLeftLayer.addEventListener("change", (e) => {
                state.leftCompareMode = e.target.value;
                if (state.jobResult && state.layers.left) {
                    state.layers.left.setUrl(getLayerUrl(state.leftCompareMode, state.colorMode));
                }
                updateLabelTexts();
            });
        }

        if (elements.btnZoomPatch) {
            elements.btnZoomPatch.addEventListener("click", () => {
                if (state.jobResult && state.jobResult.leaflet_bounds) {
                    state.map.fitBounds(state.jobResult.leaflet_bounds, { padding: [50, 50] });
                } else if (state.aoiLayer) {
                    state.map.fitBounds(state.aoiLayer.getBounds(), { padding: [50, 50] });
                }
            });
        }

        if (elements.btnToggleDiff) {
            elements.btnToggleDiff.addEventListener("click", () => {
                toggleDiffView();
            });
        }
    }

    function setColorMode(mode) {
        state.colorMode = mode;
        elements.btnModeRgb.classList.toggle("active", mode === "rgb");
        elements.btnModeCir.classList.toggle("active", mode === "cir");

        if (state.diffShown) {
            state.diffShown = false;
            if (elements.btnToggleDiff) elements.btnToggleDiff.classList.remove("active");
        }

        if (state.jobResult) {
            if (state.layers.left) {
                state.layers.left.setUrl(getLayerUrl(state.leftCompareMode, mode));
            }
            if (state.layers.right) {
                if (state.analysisLayer === "rgb") state.layers.right.setUrl(getLayerUrl("sr", mode));
            }
        }
        updateLabelTexts();
    }

    function displayNameForModel(variant) {
        if (variant === "swin2sr") return "SEN2SR-Swin2SR";
        if (variant === "lite-ft") return "SEN2SR-Lite FT";
        return "SEN2SR-Lite";
    }

    function initModelSelector() {
        const radios = document.querySelectorAll('input[name="sr-model"]');
        radios.forEach((radio) => {
            radio.addEventListener("change", (e) => {
                state.selectedModel = e.target.value;

                const labelLite = document.getElementById("label-model-lite");
                const labelSwin = document.getElementById("label-model-swin");
                const labelLiteFt = document.getElementById("label-model-liteft");
                if (labelLite) labelLite.classList.toggle("active", state.selectedModel === "lite");
                if (labelSwin) labelSwin.classList.toggle("active", state.selectedModel === "swin2sr");
                if (labelLiteFt) labelLiteFt.classList.toggle("active", state.selectedModel === "lite-ft");

                const explainer = document.getElementById("upscale-explainer-text");
                if (explainer) {
                    explainer.innerHTML = `<strong>${displayNameForModel(state.selectedModel)}</strong> &bull; 10 m &rarr; 2.5 m`;
                }
                updateLabelTexts();
            });
        });
    }

    function updateLabelTexts() {
        const modeLabel = state.colorMode === "rgb" ? "Natural" : "Infrared";
        if (state.leftCompareMode === "bicubic") {
            elements.labelLeftText.textContent = `Bicubic · 2.5 m · ${modeLabel}`;
        } else if (state.leftCompareMode === "prev" && state.prevJobResult) {
            const prevName = state.prevJobResult.model || "Previous";
            elements.labelLeftText.textContent = `${prevName} · 2.5 m · ${modeLabel}`;
        } else {
            elements.labelLeftText.textContent = `Original · 10 m · ${modeLabel}`;
        }
        const modelName = (state.jobResult && state.jobResult.model)
            ? state.jobResult.model
            : displayNameForModel(state.selectedModel);
        if (state.diffShown) {
            elements.labelRightText.textContent = `Amplified Δ ×${state.diffAmp} · FT vs prev`;
        } else if (state.analysisLayer !== "rgb") {
            const layerLabels = { confidence: "Reconstruction confidence", novelty: "Added neural detail", ndvi: "NDVI" };
            elements.labelRightText.textContent = `${layerLabels[state.analysisLayer] || state.analysisLayer} · ${modelName}`;
        } else {
            elements.labelRightText.textContent = `${modelName} · 2.5 m · ${modeLabel}`;
        }
    }

    // =========================================================================
    // 11. Application Lifecycle Bootstrap
    // =========================================================================
    function startApp() {
        elements = getElements();
        initMap();
        initQuickLocations();
        initUserLocation();
        initTabs();
        initAoiControls();
        initUpload();
        initCatalog();
        initDemo();
        initModelSelector();
        initSrExecution();
        initCompare();
        initSliderAndModes();
        initValidationUi();

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
