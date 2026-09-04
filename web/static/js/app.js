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
        selectedModel: "lite", // "lite" | "swin2sr"
        activeJobId: null,
        jobPollTimer: null,
        colorMode: "rgb", // "rgb" | "cir"
        leftCompareMode: "lr", // "lr" | "bicubic"
        viewMode: "split", // "split" | "lr_only" | "sr_only" | "blend"
        srOpacity: 100, // 0 to 100
        sliderPosRatio: 0.5, // 0.0 to 1.0
        isDraggingSlider: false,
        layers: {
            left: null,
            right: null,
        },
        jobResult: null,
        uploadedFile: null,
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
            sliderContainer: document.getElementById("slider-container"),
            sliderDivider: document.getElementById("slider-divider"),
            compareLabels: document.getElementById("compare-labels"),
            labelLeftText: document.getElementById("label-left-text"),
            labelRightText: document.getElementById("label-right-text"),

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
            elements.aoiBadge.textContent = name || "Preset Loaded";
            elements.aoiBadge.className = "badge badge-success";
        }
        if (elements.aoiDisplayStatus) {
            elements.aoiDisplayStatus.textContent = `Location: ${name} • Ready to Upscale`;
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
                clearOverlays();
                resetProgressSteps();
                updateProgressUI(10, "Uploading GeoTIFF to backend server...");

                const formData = new FormData();
                formData.append("file", state.uploadedFile);

                try {
                    const resp = await fetch(`/api/sr/upload?model=${encodeURIComponent(state.selectedModel)}`, {
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
                alert(`Failed to load demo scene: ${err.message}`);
            } finally {
                elements.btnLoadDemo.disabled = false;
                elements.btnLoadDemo.innerHTML = `
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
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

    async function launchSrProcessing(isDemo = false) {
        elements.btnRunSr.disabled = true;
        elements.progressCard.classList.remove("hidden");
        elements.resultsCard.classList.add("hidden");
        clearOverlays();

        resetProgressSteps();
        updateProgressUI(5, "Submitting super-resolution task to RTX 3050 GPU...");

        const sceneId = isDemo ? null : (state.selectedScene ? state.selectedScene.id : "auto");

        const payload = {
            aoi: isDemo ? null : state.currentAoi,
            scene_id: sceneId,
            is_demo: isDemo,
            model: state.selectedModel,
            overlap: 32,
            clamp_output: true,
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
            elements.btnRunSr.disabled = false;
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
                    updateProgressUI(100, `Failed: ${job.error_message || "Unknown error"}`);
                    elements.btnRunSr.disabled = false;
                    if (elements.btnRunUploadSr) elements.btnRunUploadSr.disabled = false;
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

    // =========================================================================
    // 9. Visual Overlay Rendering & Results Display
    // =========================================================================
    function onJobCompleted(job) {
        state.jobResult = job.result;
        elements.btnRunSr.disabled = false;
        if (elements.btnRunUploadSr) elements.btnRunUploadSr.disabled = false;

        // Populate Result Details
        elements.resCrs.textContent = job.result.crs || "EPSG:32617";
        elements.resTime.textContent = `${job.result.processing_time_sec} s`;
        elements.resDevice.textContent = `${job.result.device_used.toUpperCase()} (NVIDIA RTX 3050)`;
        if (elements.resModel) {
            elements.resModel.textContent = job.result.model || (state.selectedModel === "swin2sr" ? "SEN2SR-Swin2SR" : "SEN2SR-Lite");
        }
        if (elements.resVram) {
            elements.resVram.textContent = job.result.peak_vram_mb ? `${job.result.peak_vram_mb} MB` : "N/A";
        }

        elements.btnDownloadGeotiff.href = `/api/sr/jobs/${job.job_id}/download/geotiff`;
        elements.btnDownloadRgb.href = `/api/sr/jobs/${job.job_id}/download/rgb`;
        elements.btnDownloadCir.href = `/api/sr/jobs/${job.job_id}/download/cir`;

        elements.resultsCard.classList.remove("hidden");
        elements.viewModeGroup.style.display = "flex";
        elements.comparisonToggleGroup.style.display = "flex";

        // Display on Map
        displayJobLayers(job.result);
    }

    function displayJobLayers(result) {
        clearOverlays();

        const bounds = result.leaflet_bounds;
        state.map.fitBounds(bounds, { padding: [50, 50] });

        // Left Layer (Default: Native 10m LR RGB)
        const leftUrl = getLayerUrl(state.leftCompareMode, state.colorMode);
        state.layers.left = L.imageOverlay(leftUrl, bounds, {
            opacity: 1.0,
            interactive: false,
        }).addTo(state.map);

        // Right Layer (Super-Resolved 2.5m SR RGB)
        const rightUrl = getLayerUrl("sr", state.colorMode);
        state.layers.right = L.imageOverlay(rightUrl, bounds, {
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
    }

    function getLayerUrl(type, mode) {
        if (!state.jobResult) return "";
        const key = `${type}_${mode}`;
        return state.jobResult.previews[key] || "";
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
    }

    function setColorMode(mode) {
        state.colorMode = mode;
        elements.btnModeRgb.classList.toggle("active", mode === "rgb");
        elements.btnModeCir.classList.toggle("active", mode === "cir");

        if (state.jobResult) {
            if (state.layers.left) {
                state.layers.left.setUrl(getLayerUrl(state.leftCompareMode, mode));
            }
            if (state.layers.right) {
                state.layers.right.setUrl(getLayerUrl("sr", mode));
            }
        }
        updateLabelTexts();
    }

    function initModelSelector() {
        const radios = document.querySelectorAll('input[name="sr-model"]');
        radios.forEach((radio) => {
            radio.addEventListener("change", (e) => {
                state.selectedModel = e.target.value;
                const isSwin = state.selectedModel === "swin2sr";

                const labelLite = document.getElementById("label-model-lite");
                const labelSwin = document.getElementById("label-model-swin");
                if (labelLite) labelLite.classList.toggle("active", !isSwin);
                if (labelSwin) labelSwin.classList.toggle("active", isSwin);

                const explainer = document.getElementById("upscale-explainer-text");
                if (explainer) {
                    if (isSwin) {
                        explainer.innerHTML = `Selected <strong>SEN2SR-Swin2SR</strong> &bull; Higher-capacity 4&times; spatial upscaling (10m &rarr; 2.5m)`;
                    } else {
                        explainer.innerHTML = `Selected <strong>SEN2SR-Lite</strong> &bull; 4&times; spatial upscaling (10m &rarr; 2.5m)`;
                    }
                }
                updateLabelTexts();
            });
        });
    }

    function updateLabelTexts() {
        const modeLabel = state.colorMode === "rgb" ? "Natural RGB" : "Infrared CIR";
        if (state.leftCompareMode === "bicubic") {
            elements.labelLeftText.textContent = `Bicubic Baseline (2.5m • ${modeLabel})`;
        } else {
            elements.labelLeftText.textContent = `Original Sentinel-2 (10.0m Native • ${modeLabel})`;
        }
        const modelName = (state.jobResult && state.jobResult.model)
            ? state.jobResult.model
            : (state.selectedModel === "swin2sr" ? "SEN2SR-Swin2SR" : "SEN2SR-Lite");
        elements.labelRightText.textContent = `${modelName} (2.50m Neural SR • ${modeLabel})`;
    }

    // =========================================================================
    // 11. Application Lifecycle Bootstrap
    // =========================================================================
    function startApp() {
        elements = getElements();
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
