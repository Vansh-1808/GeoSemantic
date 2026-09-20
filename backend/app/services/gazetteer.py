"""
Geographic Entity Understanding & Gazetteer:
Maintains hierarchical geographic knowledge (countries, states, cities, regions)
and resolves entity names to spatial bounding envelopes (EPSG:4326) and WKT polygons.

Ensures geographic entities are cleanly separated from visual concepts so:
  - "desert in Rajasthan" means "Find desert visual concept INSIDE Rajasthan state boundary"
  - "industrial areas in Tamil Nadu" means "Find industrial visual concept INSIDE Tamil Nadu boundary"
  - "urban development in Chennai" means "Find urban development INSIDE Chennai city boundary"
  - "water bodies in Kerala" means "Find water bodies INSIDE Kerala state boundary"
"""
from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

from app.core.logging import get_logger

logger = get_logger(__name__)


EntityType = Literal["country", "state", "city", "region"]


class GeographicEntity(BaseModel):
    """Structured representation of an administrative or physical geographic entity."""
    canonical_name: str
    aliases: List[str] = Field(default_factory=list)
    entity_type: EntityType
    country: str = "India"
    bbox: Tuple[float, float, float, float]  # (west, south, east, north) in EPSG:4326
    description: Optional[str] = None

    @property
    def wkt_polygon(self) -> str:
        """Returns WKT POLYGON for the bounding envelope."""
        w, s, e, n = self.bbox
        return f"POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))"


# ── Hierarchical Gazetteer Knowledge Base ─────────────────────────────────────
# Covers all regions in the indexed dataset + major administrative entities of India
GAZETTEER_ENTITIES: List[GeographicEntity] = [
    # ── 1. Countries ──────────────────────────────────────────────────────────
    GeographicEntity(
        canonical_name="India",
        aliases=["republic of india", "bharat", "hindustan"],
        entity_type="country",
        country="India",
        bbox=(68.1, 6.7, 97.4, 35.7),
        description="Republic of India",
    ),

    # ── 2. States & Union Territories ─────────────────────────────────────────
    GeographicEntity(
        canonical_name="Rajasthan",
        aliases=["rajasthani", "marwar", "mewar"],
        entity_type="state",
        country="India",
        bbox=(69.4, 23.0, 78.3, 30.2),
        description="State of Rajasthan, Northwestern India",
    ),
    GeographicEntity(
        canonical_name="Tamil Nadu",
        aliases=["tamilnadu", "tamil nadu state", "tn"],
        entity_type="state",
        country="India",
        bbox=(76.2, 8.0, 80.4, 13.6),
        description="State of Tamil Nadu, Southern India",
    ),
    GeographicEntity(
        canonical_name="Kerala",
        aliases=["keralam", "malabar"],
        entity_type="state",
        country="India",
        bbox=(74.8, 8.2, 77.6, 12.8),
        description="State of Kerala, Southwestern Malabar Coast",
    ),
    GeographicEntity(
        canonical_name="Uttar Pradesh",
        aliases=["uttarpradesh", "up"],
        entity_type="state",
        country="India",
        bbox=(77.0, 23.8, 84.7, 30.4),
        description="State of Uttar Pradesh, Northern India",
    ),
    GeographicEntity(
        canonical_name="Gujarat",
        aliases=["gujarat state"],
        entity_type="state",
        country="India",
        bbox=(68.1, 20.1, 74.5, 24.7),
        description="State of Gujarat, Western India",
    ),
    GeographicEntity(
        canonical_name="Jammu and Kashmir",
        aliases=["jammu & kashmir", "kashmir", "j&k", "jammu kashmir"],
        entity_type="state",
        country="India",
        bbox=(73.5, 32.2, 77.5, 35.5),
        description="Union Territory of Jammu and Kashmir",
    ),
    GeographicEntity(
        canonical_name="Ladakh",
        aliases=["ladakh ut", "leh ladakh"],
        entity_type="state",
        country="India",
        bbox=(75.5, 32.2, 80.0, 36.0),
        description="Union Territory of Ladakh",
    ),
    GeographicEntity(
        canonical_name="Maharashtra",
        aliases=["maharashtra state"],
        entity_type="state",
        country="India",
        bbox=(72.6, 15.6, 80.9, 22.0),
        description="State of Maharashtra, Western India",
    ),
    GeographicEntity(
        canonical_name="Karnataka",
        aliases=["karnataka state"],
        entity_type="state",
        country="India",
        bbox=(74.0, 11.5, 78.6, 18.5),
        description="State of Karnataka, Southwestern India",
    ),
    GeographicEntity(
        canonical_name="West Bengal",
        aliases=["bengal", "paschim banga"],
        entity_type="state",
        country="India",
        bbox=(85.8, 21.5, 89.9, 27.2),
        description="State of West Bengal, Eastern India",
    ),
    GeographicEntity(
        canonical_name="Delhi",
        aliases=["new delhi", "ncr", "national capital region", "delhi ncr"],
        entity_type="state",
        country="India",
        bbox=(76.8, 28.4, 77.3, 28.9),
        description="National Capital Territory of Delhi",
    ),

    # ── 3. Cities & Metropolitan Areas ────────────────────────────────────────
    GeographicEntity(
        canonical_name="Chennai",
        aliases=["madras", "chennai city"],
        entity_type="city",
        country="India",
        bbox=(80.1, 12.9, 80.38, 13.25),
        description="Capital city of Tamil Nadu (Sentinel-2 indexed scene area)",
    ),
    GeographicEntity(
        canonical_name="Perambur",
        aliases=["perambur chennai", "perambur loc"],
        entity_type="city",
        country="India",
        bbox=(80.2, 13.1, 80.26, 13.16),
        description="Industrial and rail transit hub in Northern Chennai (Sentinel-2 scene area)",
    ),
    GeographicEntity(
        canonical_name="Noida",
        aliases=["new okhla industrial development authority"],
        entity_type="city",
        country="India",
        bbox=(77.25, 28.45, 77.45, 28.65),
        description="Major planned urban hub in Gautam Buddha Nagar, Uttar Pradesh",
    ),
    GeographicEntity(
        canonical_name="Greater Noida",
        aliases=["greater noida industrial area"],
        entity_type="city",
        country="India",
        bbox=(77.4, 28.3, 77.6, 28.55),
        description="Planned city adjacent to Noida in Gautam Buddha Nagar",
    ),
    GeographicEntity(
        canonical_name="Jewar",
        aliases=["jewar airport", "noida international airport"],
        entity_type="city",
        country="India",
        bbox=(77.45, 28.05, 77.68, 28.28),
        description="Location of Noida International Airport / Jewar (Indexed scene area)",
    ),
    GeographicEntity(
        canonical_name="Varanasi",
        aliases=["banaras", "kashi", "benares"],
        entity_type="city",
        country="India",
        bbox=(82.9, 25.2, 83.1, 25.42),
        description="Historic city on the Ganga River, Uttar Pradesh (Indexed scene area)",
    ),
    GeographicEntity(
        canonical_name="Srinagar",
        aliases=["srinagar city", "dal lake area"],
        entity_type="city",
        country="India",
        bbox=(74.7, 34.0, 75.0, 34.2),
        description="Summer capital of Jammu and Kashmir surrounding Dal Lake (Indexed scene area)",
    ),
    GeographicEntity(
        canonical_name="Kochi",
        aliases=["cochin", "ernakulam"],
        entity_type="city",
        country="India",
        bbox=(76.15, 9.9, 76.35, 10.1),
        description="Major coastal port city in Kerala",
    ),
    GeographicEntity(
        canonical_name="Bhadla",
        aliases=["bhadla solar", "bhadla park"],
        entity_type="city",
        country="India",
        bbox=(71.8, 27.42, 72.05, 27.62),
        description="Site of Bhadla Solar Park in Jodhpur district, Rajasthan (Indexed scene area)",
    ),
    GeographicEntity(
        canonical_name="Mumbai",
        aliases=["bombay"],
        entity_type="city",
        country="India",
        bbox=(72.75, 18.88, 73.0, 19.28),
        description="Financial capital of India, Maharashtra",
    ),
    GeographicEntity(
        canonical_name="Bangalore",
        aliases=["bengaluru", "bangalore city"],
        entity_type="city",
        country="India",
        bbox=(77.45, 12.85, 77.75, 13.15),
        description="Capital city of Karnataka",
    ),
    GeographicEntity(
        canonical_name="Kolkata",
        aliases=["calcutta"],
        entity_type="city",
        country="India",
        bbox=(88.25, 22.45, 88.45, 22.65),
        description="Capital city of West Bengal on the Hooghly River",
    ),
    GeographicEntity(
        canonical_name="Hyderabad",
        aliases=["hyderabad city"],
        entity_type="city",
        country="India",
        bbox=(78.35, 17.3, 78.6, 17.55),
        description="Capital city of Telangana",
    ),

    # ── 4. Geographic Regions, Waterbodies & Physiographic Features ───────────
    GeographicEntity(
        canonical_name="Thar",
        aliases=["thar desert", "great indian desert"],
        entity_type="region",
        country="India",
        bbox=(69.5, 24.5, 76.0, 29.5),
        description="Thar Desert region covering western Rajasthan",
    ),
    GeographicEntity(
        canonical_name="Pangong",
        aliases=["pangong tso", "pangong lake", "pangong tso lake"],
        entity_type="region",
        country="India",
        bbox=(78.35, 33.65, 78.6, 33.85),
        description="High-altitude endorheic lake in Ladakh Himalayas (Indexed scene area)",
    ),
    GeographicEntity(
        canonical_name="Sardar Sarovar",
        aliases=["sardar sarovar dam", "narmada dam", "kevadiya"],
        entity_type="region",
        country="India",
        bbox=(73.65, 21.78, 73.85, 21.92),
        description="Major concrete gravity dam on the Narmada River in Gujarat (Indexed scene area)",
    ),
    GeographicEntity(
        canonical_name="Ganga",
        aliases=["ganga river", "ganges", "ganges river", "gangetic plain"],
        entity_type="region",
        country="India",
        bbox=(78.0, 24.0, 88.0, 31.0),
        description="Ganga river channel and fertile plains across Northern India",
    ),
    GeographicEntity(
        canonical_name="Himalayas",
        aliases=["himalaya", "himalayan range", "himalayan mountains"],
        entity_type="region",
        country="India",
        bbox=(74.0, 28.0, 88.0, 35.0),
        description="Himalayan mountain range across northern India",
    ),
]


class GeographicGazetteer:
    """
    Gazetteer service that maps natural language mentions of geographic entities
    (countries, states, cities, regions) to standardized entities and PostGIS-compatible boundaries.
    """

    def __init__(self, entities: Optional[List[GeographicEntity]] = None) -> None:
        self.entities = entities or GAZETTEER_ENTITIES
        # Build lookup table sorted by descending length to match specific entities first
        # (e.g. 'Greater Noida' before 'Noida', 'Tamil Nadu' before single words)
        self._lookup: List[Tuple[re.Pattern, GeographicEntity]] = []
        for ent in sorted(self.entities, key=lambda e: -len(e.canonical_name)):
            # All names to match: canonical and aliases
            all_names = [ent.canonical_name] + ent.aliases
            for name in sorted(all_names, key=lambda n: -len(n)):
                pattern = re.compile(r"\b" + re.escape(name.lower()) + r"\b", re.IGNORECASE)
                self._lookup.append((pattern, ent))

    def resolve_entity(self, text: str) -> Optional[GeographicEntity]:
        """
        Resolves a natural language string to the first matching GeographicEntity.
        Uses greedy scanning with priority given to longest matching entity names.
        """
        if not text:
            return None
        text_clean = text.lower().strip()
        for pattern, entity in self._lookup:
            if pattern.search(text_clean):
                return entity
        return None

    def strip_entity_mentions(self, text: str, entity: GeographicEntity) -> str:
        """
        Removes the entity name and any immediately preceding or following spatial
        prepositions ('in', 'at', 'within', 'near', 'around', 'across') from the text.
        This produces a pure semantic description suitable for visual embedding.
        Example:
          "desert in Rajasthan" -> "desert"
          "industrial areas in Tamil Nadu" -> "industrial areas"
          "urban development in Chennai" -> "urban development"
          "water bodies in Kerala" -> "water bodies"
        """
        cleaned = text
        names = [entity.canonical_name] + entity.aliases
        for name in sorted(names, key=lambda n: -len(n)):
            # Match preposition + entity name, e.g. "in Rajasthan", "within Tamil Nadu", "near Chennai"
            pattern_with_prep = re.compile(
                r"\b(in|at|within|inside|around|near|across|along|through|of)\s+" + re.escape(name) + r"\b",
                re.IGNORECASE,
            )
            cleaned = pattern_with_prep.sub(" ", cleaned)

            # Match standalone entity name
            pattern_standalone = re.compile(r"\b" + re.escape(name) + r"\b", re.IGNORECASE)
            cleaned = pattern_standalone.sub(" ", cleaned)

        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned


# Singleton instance
gazetteer_service = GeographicGazetteer()
