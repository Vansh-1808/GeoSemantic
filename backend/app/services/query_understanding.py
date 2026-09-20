"""
Query Understanding Foundation:
Transforms natural language analyst queries into a structured, canonical internal representation.
Performs:
  1. Lowercasing, punctuation, and whitespace normalization
  2. Singular/plural normalization for satellite & geographical entities
  3. Domain synonym handling
  4. Satellite-domain terminology normalization
  5. Intent classification (change_search vs semantic_search vs temporal_search)
  6. Extraction of target, concept, relationship, location, temporal bounds, and quality constraints
  7. Generation of canonical embedding text for RemoteCLIP
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Set
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.services.gazetteer import GeographicEntity, gazetteer_service
from app.services.spelling_correction import SpellingCorrection, spelling_correction_service

logger = get_logger(__name__)


# ── Geographic Knowledge Base (Known Locations with EPSG:4326 BBoxes) ───────────
KNOWN_LOCATIONS: Dict[str, Tuple[float, float, float, float]] = {
    "rajasthan": (69.4, 23.0, 78.3, 30.2),
    "tamil nadu": (76.2, 8.0, 80.4, 13.6),
    "bhadla": (71.8, 27.4, 72.2, 27.6),
    "thar": (69.5, 24.5, 76.0, 29.5),
    "noida": (77.2, 28.4, 77.5, 28.7),
    "greater noida": (77.4, 28.3, 77.6, 28.6),
    "jewar": (77.4, 28.0, 77.7, 28.3),
    "delhi": (76.8, 28.4, 77.3, 28.9),
    "chennai": (80.1, 12.9, 80.3, 13.2),
    "varanasi": (82.9, 25.2, 83.1, 25.4),
    "ganga": (78.0, 24.0, 88.0, 31.0),
    "ladakh": (75.5, 32.2, 79.5, 36.0),
    "pangong": (78.3, 33.5, 79.2, 34.1),
    "srinagar": (74.7, 34.0, 75.0, 34.2),
    "kashmir": (73.5, 33.0, 76.5, 35.0),
    "himalayas": (74.0, 28.0, 88.0, 35.0),
    "sardar sarovar": (73.7, 21.8, 73.9, 22.0),
    "mumbai": (72.7, 18.8, 73.0, 19.3),
    "bangalore": (77.4, 12.8, 77.8, 13.1),
    "bengaluru": (77.4, 12.8, 77.8, 13.1),
    "kolkata": (88.2, 22.4, 88.5, 22.7),
    "hyderabad": (78.3, 17.2, 78.6, 17.6),
    "kochi": (76.2, 9.9, 76.4, 10.1),
    "kerala": (74.8, 8.2, 77.6, 12.8),
}

# ── Domain Synonyms (Mapping variations to canonical terms) ────────────────────
SYNONYMS_MAP: Dict[str, str] = {
    # Energy / Solar
    "photovoltaic": "solar",
    "pv": "solar",
    "solar park": "solar",
    "solar array": "solar",
    "solar plant": "solar",
    "solar power": "solar",
    "solar farm": "solar",
    # Aviation
    "aerodrome": "airport",
    "airstrip": "airport",
    "airfield": "airport",
    "tarmac": "runway",
    "air apron": "apron",
    # Transportation / Roads
    "expressway": "road",
    "expressways": "road",
    "highway": "road",
    "highways": "road",
    "freeway": "road",
    "motorway": "road",
    "street": "road",
    "avenue": "road",
    "corridor": "road",
    # Water
    "water body": "water",
    "water bodies": "water",
    "waterbody": "water",
    "water surface": "water",
    "waterway": "water",
    "waterways": "water",
    "canal": "water",
    "stream": "river",
    "dam lake": "reservoir",
    "ocean": "sea",
    "harbour": "port",
    "dock": "port",
    "docks": "port",
    "pier": "port",
    "marina": "port",
    # Vegetation / Agriculture
    "deforestation": "vegetation_clearance",
    "logging": "vegetation_clearance",
    "tree cutting": "vegetation_clearance",
    "tree removal": "vegetation_clearance",
    "tree felling": "vegetation_clearance",
    "agricultural land": "farmland",
    "agricultural": "farmland",
    "agriculture": "farmland",
    "cultivation": "farmland",
    "crop field": "farmland",
    "crop fields": "farmland",
    "crop": "crops",
    "paddy": "farmland",
    "greenery": "vegetation",
    "trees": "forest",
    "dense forest": "forest",
    # Arid / Desert
    "sand dune": "desert",
    "sand dunes": "desert",
    "arid land": "desert",
    "arid region": "desert",
    "barren land": "desert",
    "wasteland": "desert",
    # Urban / Built-up
    "built-up": "urban",
    "builtup": "urban",
    "housing colony": "residential",
    "housing": "residential",
    "apartments": "residential",
    "commercial area": "commercial",
    "factory": "industrial",
    "factories": "industrial",
    "warehouse": "industrial",
    "warehouses": "industrial",
    "open pit": "mining",
    "quarry": "mining",
}

# ── Singular/Plural Normalization Exceptions & Rules ───────────────────────────
PLURAL_EXCEPTIONS: Dict[str, str] = {
    "series": "series",
    "species": "species",
    "oasis": "oasis",
    "debris": "debris",
    "crossroads": "crossroad",
    "headquarters": "headquarters",
    "barracks": "barracks",
}

DOMAIN_LEMMAS: Dict[str, str] = {
    "airports": "airport",
    "runways": "runway",
    "terminals": "terminal",
    "hangars": "hangar",
    "airfields": "airfield",
    "airstrips": "airstrip",
    "aircrafts": "aircraft",
    "airplanes": "airplane",
    "roads": "road",
    "highways": "highway",
    "expressways": "expressway",
    "bridges": "bridge",
    "buildings": "building",
    "structures": "structure",
    "houses": "house",
    "colonies": "colony",
    "suburbs": "suburb",
    "facilities": "facility",
    "factories": "factory",
    "warehouses": "warehouse",
    "panels": "panel",
    "reservoirs": "reservoir",
    "rivers": "river",
    "lakes": "lake",
    "waterways": "waterway",
    "canals": "canal",
    "ports": "port",
    "harbors": "harbor",
    "docks": "dock",
    "vessels": "vessel",
    "ships": "ship",
    "boats": "boat",
    "ferries": "ferry",
    "forests": "forest",
    "trees": "tree",
    "farmlands": "farmland",
    "fields": "field",
    "crops": "crop",
    "orchards": "orchard",
    "mountains": "mountain",
    "glaciers": "glacier",
    "clouds": "cloud",
    "shadows": "shadow",
    "dams": "dam",
    "cities": "city",
    "towns": "town",
}

# ── Intent Trigger Keywords ───────────────────────────────────────────────────
CHANGE_INTENT_WORDS: Set[str] = {
    "new", "expanded", "expansion", "expand", "constructing", "construction",
    "demolished", "demolition", "cleared", "clearance", "clearing", "removal",
    "disappeared", "disappearance", "appeared", "appearance", "built", "change",
    "changes", "changed", "changing", "delta", "shift", "shifts", "loss", "gain",
    "increase", "increasing", "decrease", "decreasing", "growth", "shrinkage",
    "contraction", "development", "developing", "altered", "alteration", "deforested",
    "deforestation", "reforested", "afforestation", "transformed", "transformation",
}

RELATIONSHIP_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(near|along|adjacent to|beside|by|close to)\s+(roads?|highways?|expressways?|streets?)\b"), "near_road"),
    (re.compile(r"\b(near|along|adjacent to|beside|by|close to|around)\s+(water|water\s*bodies|rivers?|lakes?|reservoirs?|oceans?|coasts?|coastlines?)\b"), "near_water"),
    (re.compile(r"\b(in|inside|within|through|across)\s+(forests?|woodlands?|jungles?)\b"), "in_forest"),
    (re.compile(r"\b(near|around|adjacent to)\s+(airports?|runways?)\b"), "near_airport"),
    (re.compile(r"\b(near|around|adjacent to)\s+(cities|urban|buildings?|residential)\b"), "near_urban"),
]


class ParsedQuery(BaseModel):
    """Structured internal query representation produced by the Query Understanding Engine."""
    raw_query: str
    normalized_query: str
    intent: str  # "change_search" | "semantic_search" | "temporal_search"
    
    # Semantic Extractions
    target: Optional[str] = None  # e.g. "construction", "vegetation_clearance", "water", "road"
    concept: Optional[str] = None  # e.g. "desert", "airport", "solar", "farmland", "reservoir"
    relationship: Optional[str] = None  # e.g. "near_road", "near_water", "in_forest"
    location: Optional[str] = None  # e.g. "Rajasthan", "Noida", "Jewar", "Bhadla"
    location_bbox: Optional[List[float]] = None  # [west, south, east, north] in EPSG:4326
    location_type: Optional[str] = None  # "country" | "state" | "city" | "region"
    location_wkt: Optional[str] = None  # WKT representation for PostGIS
    
    # Constraints
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    time_anchor: Optional[str] = None  # e.g. "2023", "recent", "historical"
    max_cloud_cover: Optional[float] = None  # e.g. 5.0 for "cloud-free"
    min_quality: Optional[float] = None
    sensor_constraints: Optional[List[str]] = None
    
    # Physical landcover indicators
    requires_water: bool = False
    requires_vegetation: bool = False
    requires_urban: bool = False
    
    # Canonical string to pass to foundation embedding model
    canonical_embedding_text: str

    # Phase 13: Offline Spelling Correction & Query Robustness
    corrected_query: Optional[str] = None
    did_you_mean: Optional[str] = None
    corrections: List[SpellingCorrection] = Field(default_factory=list)


class QueryNormalizer:
    """Normalizes raw query strings through lowercasing, punctuation, lemmatization, and synonym substitution."""

    @staticmethod
    def cleanup(text: str) -> str:
        """Lowercases, normalizes whitespace, and strips non-printable characters."""
        if not text:
            return ""
        # Lowercase
        text = text.lower()
        # Replace non-breaking spaces and tabs with standard space
        text = re.sub(r"[\s\xa0\t\r\n]+", " ", text)
        return text.strip()

    @staticmethod
    def normalize_punctuation(text: str) -> str:
        """
        Normalizes punctuation to spaces while preserving:
          - hyphens in dates (e.g. 2023-01-01)
          - decimal points in coordinates/numbers (e.g. 28.5)
          - hyphens in compound words (e.g. cloud-free, built-up)
        """
        # Replace brackets, quotes, slashes, ampersands, commas, semicolons with space
        cleaned = re.sub(r"[\[\]\(\)\{\}\"\'/\\&,;:!?*~#^+=<>|]", " ", text)
        # Remove dots that are not between digits (e.g. ellipses '...' or periods at end of words/sentences)
        cleaned = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", cleaned)
        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def lemmatize_word(cls, word: str) -> str:
        """Converts plural nouns to singular using domain dictionary and standard rules."""
        if word in PLURAL_EXCEPTIONS:
            return PLURAL_EXCEPTIONS[word]
        if word in DOMAIN_LEMMAS:
            return DOMAIN_LEMMAS[word]

        # Standard English plural reductions
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        if word.endswith("es") and len(word) > 4:
            # e.g. boxes, beaches
            if any(word.endswith(sfx) for sfx in ["shes", "ches", "xes", "zes"]):
                return word[:-2]
        if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            return word[:-1]
        return word

    @classmethod
    def apply_synonyms(cls, text: str) -> str:
        """Substitutes domain synonyms with canonical terms."""
        # Replace multi-word phrases first
        for phrase, canonical in sorted(SYNONYMS_MAP.items(), key=lambda x: -len(x[0])):
            pattern = r"\b" + re.escape(phrase) + r"\b"
            text = re.sub(pattern, canonical, text)
        return text

    @classmethod
    def normalize(cls, text: str) -> str:
        """Full normalization pipeline."""
        cleaned = cls.cleanup(text)
        punct_norm = cls.normalize_punctuation(cleaned)
        syn_applied = cls.apply_synonyms(punct_norm)
        
        # Lemmatize individual tokens
        tokens = syn_applied.split()
        lemmatized_tokens = [cls.lemmatize_word(t) for t in tokens]
        return " ".join(lemmatized_tokens)


class QueryUnderstandingEngine:
    """
    Analyzes analyst queries to produce a rich structured internal representation.
    """

    def __init__(self) -> None:
        self.normalizer = QueryNormalizer()
        self.spelling_corrector = spelling_correction_service
        self.gazetteer = gazetteer_service

    def parse(self, raw_query: str) -> ParsedQuery:
        """Parses raw natural language string into ParsedQuery with offline spelling correction."""
        raw_clean = raw_query.strip()

        # Step 0: Offline Spelling Correction & Typo Detection
        spell_res = self.spelling_corrector.correct_query(raw_clean)
        effective_query = spell_res.corrected_query if spell_res.is_corrected else raw_clean

        norm_text = self.normalizer.normalize(effective_query)
        norm_words = set(norm_text.split())

        # 1. Temporal Constraints Extraction
        start_date, end_date, time_anchor = self._extract_temporal(effective_query)

        # 2. Quality & Sensor Constraints
        max_cloud_cover, min_quality, sensors = self._extract_quality_and_sensor(effective_query)

        # 3. Geographic Entity & Location Extraction (Phase 14)
        location, location_bbox, location_type, location_wkt = self._extract_location(effective_query)

        # 4. Relationship Extraction (e.g. near_road, near_water)
        relationship = self._extract_relationship(effective_query)

        # 5. Intent Classification
        is_change = (
            bool(norm_words & CHANGE_INTENT_WORDS) or
            ("between" in norm_words and ("and" in norm_words or "to" in norm_words)) or
            ("since" in norm_words) or
            ("after" in norm_words and time_anchor is not None) or
            (relationship is not None and any(w in norm_words for w in ["construction", "expansion", "clearance", "growth", "loss", "development"]))
        )
        intent = "change_search" if is_change else "semantic_search"

        # 6. Target and Concept Extraction
        target, concept = self._extract_target_and_concept(norm_text, norm_words, is_change)

        # 7. Physical Landcover Indicators
        requires_water = bool(norm_words & {"water", "river", "lake", "reservoir", "sea", "ocean", "coast", "coastal", "port", "canal", "estuary"}) or (relationship == "near_water")
        requires_vegetation = bool(norm_words & {"farmland", "forest", "vegetation", "crop", "crops", "agriculture", "trees", "woodland"})
        requires_urban = bool(norm_words & {"urban", "building", "residential", "commercial", "industrial", "road", "construction", "concrete"}) or (relationship == "near_road") or (relationship == "near_urban")

        # 8. Generate Canonical Embedding Text for RemoteCLIP
        canonical_text = self._build_canonical_embedding_text(
            norm_text=norm_text,
            concept=concept,
            target=target,
            relationship=relationship,
            location=location,
        )

        result = ParsedQuery(
            raw_query=raw_clean,
            normalized_query=norm_text,
            intent=intent,
            target=target,
            concept=concept,
            relationship=relationship,
            location=location,
            location_bbox=location_bbox,
            location_type=location_type,
            location_wkt=location_wkt,
            start_date=start_date,
            end_date=end_date,
            time_anchor=time_anchor,
            max_cloud_cover=max_cloud_cover,
            min_quality=min_quality,
            sensor_constraints=sensors,
            requires_water=requires_water,
            requires_vegetation=requires_vegetation,
            requires_urban=requires_urban,
            canonical_embedding_text=canonical_text,
            corrected_query=spell_res.corrected_query if spell_res.is_corrected else None,
            did_you_mean=spell_res.did_you_mean,
            corrections=spell_res.corrections,
        )
        logger.debug(f"Parsed query '{raw_clean}' -> intent={result.intent}, concept={result.concept}, target={result.target}, location={result.location} ({location_type}), corrected={spell_res.is_corrected}")
        return result

    def _extract_temporal(self, text: str) -> Tuple[Optional[datetime], Optional[datetime], Optional[str]]:
        """Extracts date ranges (after YYYY, before YYYY, between YYYY and YYYY, since YYYY)."""
        lower = text.lower()
        start_date: Optional[datetime] = None
        end_date: Optional[datetime] = None
        time_anchor: Optional[str] = None

        # 1. "between YYYY and YYYY" / "between YYYY to YYYY"
        m_between = re.search(r"\bbetween\s+(\d{4})\s+(?:and|to)\s+(\d{4})\b", lower)
        if m_between:
            y1, y2 = int(m_between.group(1)), int(m_between.group(2))
            start_date = datetime(y1, 1, 1, tzinfo=timezone.utc)
            end_date = datetime(y2, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            time_anchor = f"{y1}-{y2}"
            return start_date, end_date, time_anchor

        # 2. "after YYYY" / "since YYYY" / "from YYYY" / "post YYYY"
        m_after = re.search(r"\b(?:after|since|from|post)\s+(\d{4})\b", lower)
        if m_after:
            y = int(m_after.group(1))
            start_date = datetime(y, 1, 1, tzinfo=timezone.utc)
            time_anchor = str(y)

        # 3. "before YYYY" / "prior to YYYY" / "until YYYY" / "up to YYYY"
        m_before = re.search(r"\b(?:before|prior to|until|up to)\s+(\d{4})\b", lower)
        if m_before:
            y = int(m_before.group(1))
            end_date = datetime(y, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            time_anchor = time_anchor or str(y)

        # 4. Standalone 4-digit year if preceded by "in" (e.g. "in 2024")
        m_in = re.search(r"\bin\s+(\d{4})\b", lower)
        if m_in and not (start_date or end_date):
            y = int(m_in.group(1))
            start_date = datetime(y, 1, 1, tzinfo=timezone.utc)
            end_date = datetime(y, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            time_anchor = str(y)

        return start_date, end_date, time_anchor

    def _extract_quality_and_sensor(self, text: str) -> Tuple[Optional[float], Optional[float], Optional[List[str]]]:
        """Extracts cloud cover limits, quality thresholds, and satellite sensors."""
        lower = text.lower()
        max_cloud: Optional[float] = None
        min_quality: Optional[float] = None
        sensors: Optional[List[str]] = None

        # Cloud constraints
        if any(w in lower for w in ["cloud-free", "cloudless", "no clouds", "clear sky", "zero clouds", "without clouds"]):
            max_cloud = 5.0
        elif "low cloud" in lower:
            max_cloud = 15.0

        # Quality
        if "high quality" in lower or "crisp" in lower or "clear" in lower:
            min_quality = 0.75

        # Sensors
        found_sensors = []
        if "sentinel-2" in lower or "sentinel 2" in lower:
            found_sensors.append("Sentinel-2")
        elif "sentinel" in lower:
            found_sensors.append("Sentinel-2")
        if "landsat-8" in lower or "landsat 8" in lower:
            found_sensors.append("Landsat-8")
        elif "landsat" in lower:
            found_sensors.append("Landsat")
        if "planet" in lower:
            found_sensors.append("PlanetScope")

        sensors = found_sensors if found_sensors else None
        return max_cloud, min_quality, sensors

    def _extract_location(self, text: str) -> Tuple[Optional[str], Optional[List[float]], Optional[str], Optional[str]]:
        """
        Detects geographic locations using GeographicGazetteer with KNOWN_LOCATIONS fallback.
        Returns: (canonical_name, bbox, entity_type, wkt_polygon)
        """
        # 1. Primary: Gazetteer resolution with hierarchical entity classification
        entity = self.gazetteer.resolve_entity(text)
        if entity:
            return entity.canonical_name, list(entity.bbox), entity.entity_type, entity.wkt_polygon

        # 2. Fallback: KNOWN_LOCATIONS dictionary
        lower = text.lower()
        for loc_name, bbox in sorted(KNOWN_LOCATIONS.items(), key=lambda x: -len(x[0])):
            pattern = r"\b" + re.escape(loc_name) + r"\b"
            if re.search(pattern, lower):
                e_type = "state" if loc_name in ["rajasthan", "kerala", "kashmir", "ladakh", "tamil nadu"] else "city"
                w, s, e, n = bbox
                wkt = f"POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))"
                return loc_name.title(), list(bbox), e_type, wkt

        return None, None, None, None

    def _extract_relationship(self, text: str) -> Optional[str]:
        """Identifies spatial or semantic relationships (near_road, near_water, in_forest)."""
        lower = text.lower()
        for pattern, rel_name in RELATIONSHIP_PATTERNS:
            if pattern.search(lower):
                return rel_name
        return None

    def _extract_target_and_concept(
        self,
        norm_text: str,
        norm_words: Set[str],
        is_change: bool,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extracts primary concept (for semantic search) and target (for change search).
        Examples:
          "new construction near roads" -> target: "construction", concept: "road"
          "Rajasthan desert" -> concept: "desert"
          "vegetation clearance" -> target: "vegetation_clearance"
        """
        target: Optional[str] = None
        concept: Optional[str] = None

        # Change Targets
        if "demolition" in norm_words or "demolished" in norm_words:
            target = "demolition"
        elif "vegetation_clearance" in norm_words or "clearance" in norm_words or "deforestation" in norm_words:
            target = "vegetation_clearance"
        elif "expansion" in norm_words or "development" in norm_words:
            if "urban" in norm_words:
                target = "urban_expansion"
            elif "airport" in norm_words or "runway" in norm_words:
                target = "airport_expansion"
            elif "solar" in norm_words:
                target = "solar_development"
            elif "road" in norm_words:
                target = "road_expansion"
            else:
                target = "urban_expansion"
        elif "construction" in norm_words or "building" in norm_words:
            target = "construction"
        elif any(w in norm_words for w in ["water", "reservoir", "lake", "river"]):
            target = "water"
        elif any(w in norm_words for w in ["road", "highway"]):
            target = "road"

        # Semantic Concepts
        if "desert" in norm_words or "dune" in norm_words:
            concept = "desert"
        elif "airport" in norm_words or "runway" in norm_words or "terminal" in norm_words:
            concept = "airport"
        elif "solar" in norm_words:
            concept = "solar"
        elif "farmland" in norm_words or "crop" in norm_words or "agriculture" in norm_words:
            concept = "farmland"
        elif "forest" in norm_words:
            concept = "forest"
        elif "reservoir" in norm_words:
            concept = "reservoir"
        elif "lake" in norm_words:
            concept = "lake"
        elif "river" in norm_words:
            concept = "river"
        elif "port" in norm_words or "harbor" in norm_words:
            concept = "port"
        elif "mountain" in norm_words or "glacier" in norm_words:
            concept = "mountain"
        elif "residential" in norm_words:
            concept = "residential"
        elif "industrial" in norm_words:
            concept = "industrial"
        elif "urban" in norm_words:
            concept = "urban"
        elif "road" in norm_words or "highway" in norm_words:
            concept = "road"
        elif "water" in norm_words:
            concept = "water"
        elif "construction" in norm_words:
            concept = "construction"

        # If it's a pure semantic search and no target was set, target can be the concept
        if not is_change and not concept and target:
            concept = target
        elif is_change and not target and concept:
            target = concept

        return target, concept

    def _build_canonical_embedding_text(
        self,
        norm_text: str,
        concept: Optional[str],
        target: Optional[str],
        relationship: Optional[str],
        location: Optional[str],
    ) -> str:
        """
        Synthesizes a clean, high-signal semantic string to pass to RemoteCLIP.
        Eliminates temporal noise ('after 2023', 'between 2018 and 2022') and
        extraneous prepositions while retaining the visual features.
        """
        # Remove temporal phrases from embedding text
        cleaned = re.sub(r"\b(after|since|before|between|from|until|in|to|and|post)\s+\d{4}\b", "", norm_text)
        cleaned = re.sub(r"\b\d{4}\b", "", cleaned)
        cleaned = re.sub(r"\b(cloud-free|cloudless|no clouds|clear sky|sentinel-2|sentinel|landsat-8|landsat)\b", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # If location is known, cleanly strip it and any prepositions attaching to it
        if location:
            loc_lower = location.lower()
            # Remove "in <loc>", "near <loc>", "at <loc>", "<loc>"
            cleaned = re.sub(r"\b(in|at|within|inside|near|around|across|along|of)\s+" + re.escape(loc_lower) + r"\b", " ", cleaned)
            cleaned = re.sub(r"\b" + re.escape(loc_lower) + r"\b", " ", cleaned)
            # Strip trailing/leading prepositions
            cleaned = re.sub(r"\b(in|at|within|of|near|around)\s*$", "", cleaned)
            cleaned = re.sub(r"^\s*(in|at|within|of|near|around)\b", "", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Augment with canonical relationship terms if present
        if relationship == "near_road" and "road" not in cleaned:
            cleaned += " road infrastructure"
        elif relationship == "near_water" and "water" not in cleaned:
            cleaned += " water surface"

        if not cleaned:
            # Fallback to concept or target
            tokens = [concept, target]
            cleaned = " ".join([t for t in tokens if t]) or norm_text

        return cleaned.strip()

    async def parse_async(self, raw_query: str, use_llm: bool = True) -> ParsedQuery:
        """
        Async version of parse() that integrates the Phase 14 Local LLM pipeline.

        Full pipeline:
          raw_query
            → QueryNormalizer (Phase 12)
            → SpellingCorrector (Phase 13)
            → LocalLLM / parse_analyst_query (Phase 14)
            → GeographicGazetteer (Phase 14 — PostGIS coords)
            → ParsedQuery (canonical internal representation)

        If the LLM is unavailable, falls back to the synchronous deterministic
        parse() without interruption.  The caller cannot distinguish a LLM-enhanced
        result from a deterministic one at the ParsedQuery schema level — which is
        intentional; both paths produce the same Pydantic model.
        """
        # Always run deterministic pipeline first (Phase 12 + 13) — this is
        # the safety layer that never fails.
        base = self.parse(raw_query)

        if not use_llm:
            return base

        # Lazy import to avoid circular dependency and to ensure the LLM
        # singleton is only initialised when it is actually needed.
        try:
            from app.services.llm_service import llm_service
            from app.core.config import settings
            if not settings.llm_enabled:
                return base
        except ImportError:
            return base

        try:
            llm_result = await llm_service.parse_analyst_query(
                raw_query=raw_query,
                corrected_query=base.corrected_query,
            )
        except Exception as exc:
            logger.warning("llm_parse_async_error", error=str(exc))
            return base

        if not llm_result.llm_used:
            # LLM was offline or produced invalid output — deterministic result
            # is already correct; just return it.
            return base

        aq = llm_result.analyst_query

        # ── Merge LLM semantics into the deterministic base ──────────────────
        # Strategy: LLM wins for intent/concept/target/change_type/relationships.
        # Deterministic wins for location/bbox (Gazetteer is authoritative for
        # PostGIS coordinates) and temporal (regex is more reliable than LLM
        # date parsing).

        # 1. Intent
        if aq.intent == "change_analysis":
            merged_intent = "change_search"
        else:
            merged_intent = "semantic_search"

        # 2. Concept — prefer first LLM concept, fall back to deterministic
        merged_concept = base.concept
        if aq.concepts:
            primary_concept = aq.concepts[0].lower().replace(" ", "_")
            # Only use LLM concept if it's a recognisable landcover term
            _known_concepts = {
                "desert", "airport", "solar", "farmland", "forest", "reservoir",
                "lake", "river", "port", "mountain", "residential", "industrial",
                "urban", "road", "water", "construction", "vegetation", "glacier",
            }
            if primary_concept in _known_concepts:
                merged_concept = primary_concept

        # 3. Target
        merged_target = base.target
        if aq.target and base.target is None:
            merged_target = aq.target.lower().replace(" ", "_")

        # 4. Relationships — LLM may add extra relationships not caught by regex
        merged_relationship = base.relationship
        if aq.relationships and base.relationship is None:
            merged_relationship = aq.relationships[0] if aq.relationships else None

        # 5. Rebuild canonical embedding text with merged concept/target
        merged_canonical = self._build_canonical_embedding_text(
            norm_text=base.normalized_query,
            concept=merged_concept,
            target=merged_target,
            relationship=merged_relationship,
            location=base.location,
        )

        logger.info(
            "llm_parse_async_merged",
            raw_query=raw_query,
            deterministic_intent=base.intent,
            llm_intent=merged_intent,
            merged_concept=merged_concept,
            merged_target=merged_target,
            llm_model=llm_result.llm_model,
            duration_ms=llm_result.inference_duration_ms,
        )

        return base.model_copy(update={
            "intent": merged_intent,
            "concept": merged_concept,
            "target": merged_target,
            "relationship": merged_relationship,
            "canonical_embedding_text": merged_canonical,
        })


# Singleton Export
query_understanding_service = QueryUnderstandingEngine()
