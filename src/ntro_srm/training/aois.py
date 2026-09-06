"""Indian-priority + global AOI catalog for SIH26142 dataset builds.

Each entry is a small (~2.5 km) box centred on the given lat/lon, sized to
yield a 256x256 Sentinel-2 10m window. Indian sites are flagged
indian_priority=True and are drawn from the three PS application axes:
crop monitoring, urban analysis, disaster assessment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogAOI:
    site_id: str
    name: str
    lat: float
    lon: float
    half_deg: float = 0.0115  # ~1.28 km half-side -> ~2.56 km box
    landcover: str = "mixed"
    task: str = "general"
    indian_priority: bool = False
    split: str = "train"  # train | val | test (geographic split)

    @property
    def bbox(self) -> list[float]:
        return [
            self.lon - self.half_deg,
            self.lat - self.half_deg,
            self.lon + self.half_deg,
            self.lat + self.half_deg,
        ]


# --- Indian priority sites (held-out test bias) ---
INDIAN_AOIS: list[CatalogAOI] = [
    CatalogAOI("IND_PUNJAB_AGRI", "Ludhiana agri belt (field boundaries, crop)", 30.9010, 75.8573,
               landcover="cropland", task="crop", indian_priority=True, split="train"),
    CatalogAOI("IND_DELHI_URBAN", "South Delhi urban (buildings, narrow roads)", 28.5355, 77.2090,
               landcover="urban", task="urban", indian_priority=True, split="test"),
    CatalogAOI("IND_MUMBAI_COAST", "Mumbai coast/port (water edges, infra)", 19.0760, 72.8777,
               landcover="coastal_urban", task="urban", indian_priority=True, split="test"),
    CatalogAOI("IND_ASSAM_FLOOD", "Brahmaputra floodplain Assam (disaster)", 26.1778, 91.7500,
               landcover="floodplain", task="disaster", indian_priority=True, split="test"),
    CatalogAOI("IND_BENGALURU_LAKE", "Bengaluru lakes/urban fringe", 12.9716, 77.5946,
               landcover="urban_lake", task="urban", indian_priority=True, split="val"),
    CatalogAOI("IND_JAIPUR_SEMIARID", "Jaipur semi-arid fringe", 26.9124, 75.7873,
               landcover="semiarid", task="crop", indian_priority=True, split="train"),
]

# --- Global context sites (train diversity + real-HR pairing where NAIP exists) ---
GLOBAL_AOIS: list[CatalogAOI] = [
    CatalogAOI("USA_MOUNTAIN_LAKE", "Mountain Lake forest/lake", 37.4255, -80.5723,
               landcover="forest_lake", task="general", split="train"),
    CatalogAOI("USA_SALINAS_AGRI", "Salinas Valley farmland", 36.6777, -121.6555,
               landcover="cropland", task="crop", split="train"),
    CatalogAOI("USA_TAHOE_ALPINE", "Lake Tahoe alpine", 39.0968, -120.0324,
               landcover="alpine_lake", task="general", split="val"),
    CatalogAOI("DE_FRANKFURT_URBAN", "Frankfurt airport/city", 50.0379, 8.5622,
               landcover="urban", task="urban", split="train"),
    CatalogAOI("NL_ROTTERDAM_PORT", "Rotterdam port/harbor", 51.9244, 4.4777,
               landcover="port", task="disaster", split="val"),
    CatalogAOI("UAE_DUBAI_COAST", "Dubai Palm/coast", 25.1124, 55.1390,
               landcover="coastal_urban", task="urban", split="train"),
]

ALL_AOIS: list[CatalogAOI] = [*INDIAN_AOIS, *GLOBAL_AOIS]
