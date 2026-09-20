"""
Offline Spelling Correction & Query Robustness Engine:
Detects and corrects typos, letter transpositions, phonetic/homophone confusions,
and satellite domain terminology errors without external cloud APIs.

Supports:
  - dessert -> desert
  - constrution -> construction
  - buildng -> building
  - developement -> development
  - vegitaiton -> vegetation
  - and general satellite domain vocabulary typos via Damerau-Levenshtein distance.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from app.core.logging import get_logger

logger = get_logger(__name__)


class SpellingCorrection(BaseModel):
    """Details of a single word-level spelling correction."""
    original_word: str
    corrected_word: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    edit_distance: int = Field(..., ge=0)


class SpellingResult(BaseModel):
    """Complete result of spell correction over an entire query string."""
    original_query: str
    corrected_query: str
    corrections: List[SpellingCorrection] = Field(default_factory=list)
    did_you_mean: Optional[str] = None
    is_corrected: bool = False


# ── 1. Domain Confusables ─────────────────────────────────────────────────────
# Words that are valid English words, but in satellite remote sensing queries
# almost universally represent a user misspelling/homophone confusion.
DOMAIN_CONFUSABLES: Dict[str, str] = {
    "dessert": "desert",
    "desserts": "desert",
    "cite": "site",
    "cites": "sites",
    "plain": "plane",
    "plains": "planes",
    "damn": "dam",
    "damns": "dams",
    "bare": "barren",
    "flod": "flood",
    "floding": "flooding",
    "floded": "flooded",
}

# ── 2. Satellite & Geographic Domain Vocabulary ─────────────────────────────────
# Comprehensive lexicon of domain words for offline fuzzy matching.
DOMAIN_VOCABULARY_LIST: List[str] = [
    # Geographic proper nouns (India & Global regions)
    "rajasthan", "bhadla", "thar", "noida", "delhi", "chennai", "varanasi", "ganga",
    "ladakh", "pangong", "srinagar", "kashmir", "himalayas", "sardar", "sarovar",
    "mumbai", "bangalore", "bengaluru", "kolkata", "hyderabad", "kochi", "kerala",
    "jewar", "india", "asia", "gangetic", "yamuna", "brahmaputra", "deccan",

    # Core concepts & Physical features
    "desert", "deserts", "dune", "dunes", "airport", "airports", "runway", "runways",
    "terminal", "terminals", "hangar", "hangars", "airstrip", "airstrips", "airfield",
    "solar", "photovoltaic", "panel", "panels", "farmland", "farmlands", "agriculture",
    "agricultural", "crop", "crops", "field", "fields", "cultivation", "paddy", "orchard",
    "farm", "farms", "farming", "farmer", "farmers", "park", "parks", "plant", "plants",
    "body", "bodies", "waterbody", "waterbodies", "surface", "surfaces", "channel", "channels",
    "forest", "forests", "tree", "trees", "woodland", "jungle", "greenery", "vegetation",
    "water", "waters", "waterway", "waterways", "reservoir", "reservoirs",
    "river", "rivers", "lake", "lakes", "canal", "canals", "stream", "streams", "ocean",
    "sea", "port", "ports", "harbor", "harbors", "harbour", "dock", "docks", "pier",
    "marina", "coast", "coastal", "coastline", "shore", "shoreline", "beach", "beaches",
    "mountain", "mountains", "peak", "glacier", "glaciers", "snow", "ice", "island",
    "valley", "delta", "estuary", "wetland", "wetlands", "marsh", "swamp", "basin", "bay",

    # Urban, Built-up & Infrastructure
    "construction", "constructing", "constructed", "building", "buildings", "built",
    "structure", "structures", "urban", "city", "cities", "town", "towns", "suburb",
    "residential", "commercial", "industrial", "house", "houses", "housing", "colony",
    "apartment", "apartments", "warehouse", "warehouses", "factory", "factories",
    "facility", "facilities", "infrastructure", "concrete", "asphalt", "quarry", "mining",
    "road", "roads", "highway", "highways", "expressway", "expressways", "freeway",
    "motorway", "street", "streets", "avenue", "corridor", "bridge", "bridges", "flyover",
    "railway", "railways", "track", "tracks", "dam", "dams",

    # Change actions & Temporal verbs
    "new", "old", "expansion", "expanded", "expand", "expanding", "growth", "growing",
    "development", "developing", "developed", "clearance", "clearing", "cleared",
    "deforestation", "reforestation", "afforestation", "demolition", "demolished",
    "demolishing", "removal", "removed", "change", "changes", "changed", "changing",
    "delta", "loss", "gain", "increase", "increasing", "decrease", "decreasing",
    "shrinkage", "contraction", "appearance", "appeared", "disappearance", "disappeared",
    "alteration", "altered", "transformation", "transformed",

    # Satellite, Sensor & Spectral terminology
    "satellite", "satellites", "imagery", "images", "image", "tile", "tiles", "scene",
    "scenes", "sentinel", "landsat", "planet", "planetscope", "modis", "optical",
    "spectral", "infrared", "radiometric", "resolution", "pixel", "pixels", "raster",
    "multispectral", "band", "bands", "sensor", "sensors", "platform", "platforms",
    "ndvi", "ndwi", "ndbi", "cloud", "clouds", "cloudless", "cloud-free", "shadow",
    "shadows", "haze", "atmospheric", "quality", "usable", "clear", "crisp",

    # Spatial prepositions & Search operators
    "near", "along", "adjacent", "beside", "by", "close", "around", "inside", "within",
    "through", "across", "between", "after", "before", "since", "from", "until", "post",
    "prior", "with", "without", "and", "or", "in", "to", "at", "on", "of", "for", "the",
    "a", "an", "all", "some", "both", "high", "low", "zero", "deep", "dense", "wide",
]

# Set representation for O(1) membership checks
DOMAIN_VOCABULARY: Set[str] = set(DOMAIN_VOCABULARY_LIST)

# High-priority domain concepts for weighted scoring when ranking multiple candidates
DOMAIN_WEIGHTS: Dict[str, float] = {
    "desert": 1.5,
    "construction": 1.5,
    "building": 1.4,
    "development": 1.4,
    "vegetation": 1.4,
    "airport": 1.4,
    "reservoir": 1.4,
    "highway": 1.3,
    "road": 1.3,
    "solar": 1.3,
    "farmland": 1.3,
    "forest": 1.3,
    "water": 1.3,
    "satellite": 1.3,
    "infrastructure": 1.3,
    "sentinel": 1.3,
    "landsat": 1.3,
    "demolition": 1.3,
    "clearance": 1.3,
    "expansion": 1.3,
}


def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes the true Damerau-Levenshtein edit distance between two strings,
    supporting:
      1. Insertions
      2. Deletions
      3. Substitutions
      4. Adjacent character transpositions (e.g. 'vegitaiton' -> 'vegetation', 'teh' -> 'the')
    """
    len1, len2 = len(s1), len(s2)
    # 2D table initialized with boundary conditions
    d = [[0] * (len2 + 2) for _ in range(len1 + 2)]
    max_dist = len1 + len2
    d[0][0] = max_dist

    for i in range(0, len1 + 1):
        d[i + 1][0] = max_dist
        d[i + 1][1] = i
    for j in range(0, len2 + 1):
        d[0][j + 1] = max_dist
        d[1][j + 1] = j

    last_row: Dict[str, int] = {}

    for i in range(1, len1 + 1):
        db = 0
        for j in range(1, len2 + 1):
            k = last_row.get(s2[j - 1], 0)
            l = db
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            if cost == 0:
                db = j

            d[i + 1][j + 1] = min(
                d[i][j + 1] + 1,        # Deletion
                d[i + 1][j] + 1,        # Insertion
                d[i][j] + cost,          # Substitution
                d[k][l] + (i - k - 1) + 1 + (j - l - 1)  # Transposition
            )
        last_row[s1[i - 1]] = i

    return d[len1 + 1][len2 + 1]


class SpellingCorrectionEngine:
    """
    General, offline spelling correction engine designed for geospatial and satellite queries.
    Uses Damerau-Levenshtein fuzzy matching, domain confusables mapping, and weighted vocabulary ranking.
    """

    def __init__(
        self,
        vocabulary: Optional[Set[str]] = None,
        confusables: Optional[Dict[str, str]] = None,
    ) -> None:
        self.vocabulary = vocabulary or DOMAIN_VOCABULARY
        self.confusables = confusables or DOMAIN_CONFUSABLES
        self.weights = DOMAIN_WEIGHTS

    def correct_word(self, raw_word: str) -> Optional[SpellingCorrection]:
        """
        Attempts to correct a single word token.
        Returns a SpellingCorrection if a typo/confusable is detected and fixed, or None if valid.
        """
        word = raw_word.lower().strip()
        if not word:
            return None

        # Ignore pure numeric tokens (e.g. '2023', '28.5', '100')
        if re.match(r"^[\d\.\-]+$", word):
            return None

        # 1. Direct Confusable Check (Highest precedence for homophones like 'dessert' -> 'desert')
        if word in self.confusables:
            corrected = self.confusables[word]
            dist = damerau_levenshtein_distance(word, corrected)
            return SpellingCorrection(
                original_word=raw_word,
                corrected_word=corrected,
                confidence=0.96,
                edit_distance=dist,
            )

        # 2. In-Vocabulary Check (Word is already correct)
        if word in self.vocabulary:
            return None

        # Short words (< 4 letters) that aren't in confusables or vocab are skipped to avoid over-correcting
        if len(word) < 4:
            return None

        # 3. Damerau-Levenshtein Fuzzy Search across domain vocabulary
        max_allowed_dist = 2 if len(word) >= 5 else 1
        best_candidate: Optional[str] = None
        best_score = -1.0
        best_dist = 999

        for candidate in self.vocabulary:
            # Quick length pruning: candidates must be within max_allowed_dist of target length
            if abs(len(candidate) - len(word)) > max_allowed_dist:
                continue

            dist = damerau_levenshtein_distance(word, candidate)
            if dist <= max_allowed_dist:
                # If distance is 2, first letter must match to prevent false replacements (e.g. farms -> dams)
                if dist == 2 and candidate[0] != word[0]:
                    continue

                # Calculate composite scoring:
                # - Inverse edit distance
                # - Length similarity ratio
                # - Prefix match bonus (users usually get the 1st letter right)
                # - Domain concept weight
                len_ratio = 1.0 - (dist / max(len(word), len(candidate)))
                prefix_bonus = 0.10 if (candidate[0] == word[0]) else 0.0
                weight = self.weights.get(candidate, 1.0)

                # Overall score
                score = (len_ratio + prefix_bonus) * weight

                if score > best_score:
                    best_score = score
                    best_candidate = candidate
                    best_dist = dist

        if best_candidate and best_dist <= max_allowed_dist:
            # Map score to normalized confidence [0.70, 0.95]
            confidence = 0.95 if best_dist == 1 else 0.80
            if best_candidate[0] == word[0]:
                confidence = min(1.0, confidence + 0.03)

            return SpellingCorrection(
                original_word=raw_word,
                corrected_word=best_candidate,
                confidence=round(confidence, 2),
                edit_distance=best_dist,
            )

        return None

    def correct_query(self, query: str) -> SpellingResult:
        """
        Analyzes a full natural language query, detects misspelled tokens,
        reconstructs the clean corrected string, and formats 'Did you mean: ...' suggestions.
        """
        if not query or not query.strip():
            return SpellingResult(
                original_query=query,
                corrected_query=query,
                corrections=[],
                did_you_mean=None,
                is_corrected=False,
            )

        raw_clean = query.strip()
        tokens = raw_clean.split()
        corrected_tokens: List[str] = []
        corrections: List[SpellingCorrection] = []

        for token in tokens:
            # Strip outer punctuation for spelling check while preserving context
            stripped_token = re.sub(r"^[^\w]+|[^\w]+$", "", token)
            if not stripped_token:
                corrected_tokens.append(token)
                continue

            corr = self.correct_word(stripped_token)
            if corr:
                corrections.append(corr)
                # Preserve surrounding punctuation if any
                prefix = token[:token.find(stripped_token)]
                suffix = token[token.find(stripped_token) + len(stripped_token):]
                corrected_tokens.append(f"{prefix}{corr.corrected_word}{suffix}")
            else:
                corrected_tokens.append(token)

        is_corrected = len(corrections) > 0
        corrected_str = " ".join(corrected_tokens)

        # Generate "Did you mean: [Corrected Query]?"
        did_you_mean: Optional[str] = None
        if is_corrected:
            # Capitalize geographic place names nicely if present in corrected text
            suggested_text = self._format_suggestion(corrected_str)
            did_you_mean = f"Did you mean: {suggested_text}?"

        return SpellingResult(
            original_query=raw_clean,
            corrected_query=corrected_str,
            corrections=corrections,
            did_you_mean=did_you_mean,
            is_corrected=is_corrected,
        )

    def _format_suggestion(self, text: str) -> str:
        """Applies proper title casing for known geographic locations in suggestions."""
        words = text.split()
        formatted = []
        known_proper = {
            "rajasthan": "Rajasthan",
            "bhadla": "Bhadla",
            "thar": "Thar",
            "noida": "Noida",
            "delhi": "Delhi",
            "chennai": "Chennai",
            "varanasi": "Varanasi",
            "ganga": "Ganga",
            "ladakh": "Ladakh",
            "pangong": "Pangong",
            "srinagar": "Srinagar",
            "kashmir": "Kashmir",
            "jewar": "Jewar",
            "mumbai": "Mumbai",
            "bangalore": "Bangalore",
            "bengaluru": "Bengaluru",
            "kolkata": "Kolkata",
            "hyderabad": "Hyderabad",
            "kochi": "Kochi",
            "kerala": "Kerala",
            "sentinel": "Sentinel",
            "landsat": "Landsat",
        }
        for w in words:
            clean = re.sub(r"[^\w]", "", w.lower())
            if clean in known_proper:
                # Replace with capitalized form preserving punctuation
                rep = known_proper[clean]
                formatted.append(w.replace(clean, rep))
            else:
                formatted.append(w)
        return " ".join(formatted)


# Singleton instance
spelling_correction_service = SpellingCorrectionEngine()
