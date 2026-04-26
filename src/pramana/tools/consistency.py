"""Heart of the project: 8 contradiction rules + trust score.

Used in two places:
  1. Batch — applied to every row in `notebooks/04_silver_contradictions.py`
     to materialise `silver.contradictions` (Spark UDF over the table).
  2. Online — registered as UC function `score_claim_consistency(facility_id STRING)`
     and called by the Verifier node mid-conversation.

Keep this file pure-python (stdlib only). UC python functions cannot import
heavy libs at registration time.
"""
from __future__ import annotations
import json
import re
from typing import Any


HOSPITAL_TYPES = {"hospital"}
NON_HOSPITAL_TYPES = {"clinic", "dentist", "pharmacy", "farmacy", "doctor"}

ADVANCED_KEYWORDS = {
    "advanced surgery", "cardiac", "cardiology", "cath lab", "oncology",
    "neonatal icu", "nicu", "icu", "trauma", "neurosurgery", "transplant",
    "open heart", "bypass", "chemotherapy", "radiotherapy",
}

EMERGENCY_PATTERNS = [
    r"24\s*[/x×]\s*7", r"round[- ]the[- ]clock", r"24\s*hour", r"emergency care",
    r"trauma center", r"casualty",
]

FAKE_AWARD_PATTERNS = [
    r"w\.?\s*h\.?\s*o\.?\s+(award|certif|recogn)",
    r"who\s+(award|certif|recogn)",
    r"iso\s*9001\s*[:\-]?\s*20(2[5-9]|[3-9]\d)",
    r"nobel\s+(prize|award)",
    r"unesco\s+(award|certif)",
]

SPECIALTY_EQUIPMENT_HINTS: dict[str, list[str]] = {
    "cardiology":       ["ecg", "echo", "cath", "angio", "defibrillator", "stent", "tmt"],
    "cardiacsurg":      ["cath", "perfusion", "ventilator", "icu", "echo"],
    "oncology":         ["chemo", "linac", "radiotherapy", "ct", "pet", "mri", "brachy"],
    "radiology":        ["ct", "mri", "x-ray", "ultrasound", "xray", "usg"],
    "nephrology":       ["dialysis", "hemodialysis", "ro plant"],
    "orthopedic":       ["x-ray", "xray", "c-arm", "arthroscope", "image intensifier"],
    "neurology":        ["mri", "ct", "eeg", "emg"],
    "neurosurg":        ["mri", "ct", "microscope", "navigation"],
    "obstetric":        ["ultrasound", "ctg", "delivery", "incubator"],
    "gynecolog":        ["ultrasound", "colposcope", "ot"],
    "pediatric":        ["incubator", "phototherapy", "warmer", "nicu"],
    "neonato":          ["incubator", "warmer", "phototherapy", "nicu", "cpap", "ventilator"],
    "ophthalmology":    ["slit lamp", "phaco", "yag", "tonometer", "fundus"],
    "pulmonology":      ["spirometer", "ventilator", "bipap", "cpap", "bronchoscope"],
    "gastroenterology": ["endoscope", "colonoscope", "gastroscope", "ercp"],
    "urology":          ["cystoscope", "lithotripter", "uro"],
    "otolaryngology":   ["audiometer", "endoscope", "tympanometer"],
}

PEDIATRIC_DENTAL_FALSE_POSITIVE = re.compile(r"pediatric.*dent|dent.*pediatric", re.I)


def _to_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip().lower() for x in v if x]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                return _to_list(json.loads(s))
            except Exception:
                pass
        return [p.strip().lower() for p in s.replace("|", ",").split(",") if p.strip()]
    return [str(v).strip().lower()]


def _has_any(text: str, needles) -> bool:
    t = (text or "").lower()
    return any(n in t for n in needles)


def _flag(rule: str, severity: str, message: str, evidence: str, citation_column: str) -> dict:
    return {
        "rule_id": rule,
        "severity": severity,
        "message": message,
        "evidence": (evidence or "")[:500],
        "citation_column": citation_column,
    }


def _coords_in_india(lat, lon) -> bool:
    try:
        return 6.5 <= float(lat) <= 35.5 and 68.0 <= float(lon) <= 97.5
    except Exception:
        return False


def _coords_in_state(lat, lon, state, state_bbox: dict) -> bool:
    if not state or state not in state_bbox:
        return True
    bb = state_bbox[state]
    try:
        return bb["min_lat"] <= float(lat) <= bb["max_lat"] and bb["min_lon"] <= float(lon) <= bb["max_lon"]
    except Exception:
        return True


def evaluate_facility(row: dict, state_bbox: dict | None = None) -> list[dict]:
    """Apply rules R1–R8. `row` is a dict with the silver columns."""
    state_bbox = state_bbox or {}
    flags: list[dict] = []

    def _s(x):
        """NaN-safe string coercion (pandas null is float('nan'), not None)."""
        if x is None:
            return ""
        try:
            if isinstance(x, float) and x != x:
                return ""
        except Exception:
            pass
        return str(x)

    specialties = _to_list(row.get("specialties"))
    procedure   = _to_list(row.get("procedure"))
    equipment   = _to_list(row.get("equipment"))
    capability  = _to_list(row.get("capability"))
    description = _s(row.get("description"))
    ftype = _s(row.get("facility_type") or row.get("facilityTypeId")).lower()
    capacity = row.get("capacity")
    n_doctors = row.get("number_doctors") or row.get("numberDoctors")
    lat, lon = row.get("latitude"), row.get("longitude")
    state = _s(row.get("state") or row.get("address_stateOrRegion")) or None

    claims_blob = " ".join(specialties + capability + procedure)

    if _has_any(claims_blob, ADVANCED_KEYWORDS):
        if (not equipment) or (n_doctors in (None, 0) and capacity in (None, 0)):
            flags.append(_flag(
                "R1", "HIGH",
                "Claims advanced/critical care but missing equipment or staffing evidence.",
                f"specialties={specialties[:3]} equipment={equipment[:3]} doctors={n_doctors} capacity={capacity}",
                "specialties|capability|equipment|numberDoctors|capacity",
            ))

    if any(re.search(p, description, re.I) for p in EMERGENCY_PATTERNS):
        if ftype in NON_HOSPITAL_TYPES:
            flags.append(_flag(
                "R2", "HIGH",
                f"Claims 24/7 emergency care but is registered as a {ftype}.",
                description[:200],
                "description|facilityTypeId",
            ))

    if lat is not None and lon is not None:
        if not _coords_in_india(lat, lon):
            flags.append(_flag(
                "R3", "HIGH",
                "Coordinates fall outside India bounding box.",
                f"lat={lat}, lon={lon}",
                "latitude|longitude",
            ))
        elif not _coords_in_state(lat, lon, state, state_bbox):
            flags.append(_flag(
                "R3", "HIGH",
                f"Coordinates do not fall inside claimed state '{state}'.",
                f"lat={lat}, lon={lon}, state={state}",
                "latitude|longitude|state",
            ))

    if ftype in HOSPITAL_TYPES:
        if (capacity in (None, 0)) and (n_doctors in (None, 0)) and (not equipment):
            flags.append(_flag(
                "R4", "HIGH",
                "Hospital with zero capacity, zero doctors, and no equipment listed (ghost capability).",
                f"capacity={capacity} doctors={n_doctors} equipment={equipment}",
                "capacity|numberDoctors|equipment",
            ))

    ftype_raw = _s(row.get("facility_type_raw") or row.get("facilityTypeId")).lower()
    if ftype == "farmacy" or ftype_raw == "farmacy":
        flags.append(_flag(
            "R5", "MED",
            "Facility type 'farmacy' is a data-entry typo of 'pharmacy'.",
            f"facilityTypeId='{ftype_raw or 'farmacy'}'",
            "facilityTypeId",
        ))

    if any(re.search(p, description, re.I) for p in FAKE_AWARD_PATTERNS):
        m = next((re.search(p, description, re.I) for p in FAKE_AWARD_PATTERNS
                  if re.search(p, description, re.I)), None)
        flags.append(_flag(
            "R6", "MED",
            "Fabricated or implausible award/certification mentioned in description.",
            (m.group(0) if m else description[:120]),
            "description",
        ))

    eq_blob = " ".join(equipment)
    for sp in specialties:
        sp_norm = sp.lower().strip()
        if PEDIATRIC_DENTAL_FALSE_POSITIVE.search(sp_norm):
            continue
        for key, hints in SPECIALTY_EQUIPMENT_HINTS.items():
            if key in sp_norm and not _has_any(eq_blob, hints):
                flags.append(_flag(
                    "R7", "MED",
                    f"Lists '{sp}' specialty but no expected equipment ({', '.join(hints[:3])}…) found.",
                    f"specialty={sp} equipment={equipment[:5]}",
                    "specialties|equipment",
                ))
                break

    months_old = row.get("recency_months") or row.get("recency_of_page_update_months")
    socials = [row.get(k) for k in ("facebookLink", "twitterLink", "instagramLink",
                                       "linkedinLink", "websites", "officialWebsite")]
    socials_present = [bool(s) for s in socials]
    if (months_old is not None and months_old > 24) or not any(socials_present):
        flags.append(_flag(
            "R8", "LOW",
            "Stale page (>24 months) or zero web/social presence.",
            f"recency_months={months_old} socials_present={socials_present}",
            "recency_of_page_update|facebookLink|twitterLink|instagramLink|websites",
        ))

    return flags


def trust_score(flags: list[dict]) -> int:
    h = sum(1 for f in flags if f["severity"] == "HIGH")
    m = sum(1 for f in flags if f["severity"] == "MED")
    l = sum(1 for f in flags if f["severity"] == "LOW")
    return max(0, 100 - (35 * h + 15 * m + 5 * l))


def score_facility(facility_id: str) -> str:
    """Run all 8 consistency rules (R1-R8) for a single facility and return a
    JSON envelope with its trust score and any flags fired.

    Reads the row from ``silver_facilities_clean`` and the state bounding-box
    table ``ref_state_bbox`` from the catalog/schema configured in
    ``pramana.config`` (default ``workspace.pramana``). Use this to verify a
    specific facility's claims mid-conversation rather than scanning the whole
    table.

    Args:
        facility_id: Pramana-synthesised identifier of the form ``F######``
            (zero-padded to 6 digits, e.g. ``"F000123"``). Required. If the id
            is not found, returns ``{"facility_id": <id>, "error": "not found"}``.

    Returns:
        JSON string with keys ``facility_id``, ``trust_score`` (int 0-100; lower
        is worse) and ``flags`` (list of dicts with ``rule_id``, ``severity``,
        ``message``, ``evidence`` and ``citation_column``).
    """
    from pyspark.sql import SparkSession
    from pramana.config import CATALOG, SCHEMA
    spark = SparkSession.builder.getOrCreate()
    row = spark.sql(
        f"SELECT * FROM {CATALOG}.{SCHEMA}.silver_facilities_clean WHERE facility_id = :fid LIMIT 1",
        args={"fid": facility_id},
    ).collect()
    if not row:
        return json.dumps({"facility_id": facility_id, "error": "not found"})
    d = row[0].asDict(recursive=True)
    try:
        bbox_rows = spark.sql(
            f"SELECT state, min_lat, max_lat, min_lon, max_lon FROM {CATALOG}.{SCHEMA}.ref_state_bbox"
        ).collect()
        bbox = {r["state"]: r.asDict() for r in bbox_rows}
    except Exception:
        bbox = {}
    flags = evaluate_facility(d, bbox)
    return json.dumps({
        "facility_id": facility_id,
        "trust_score": trust_score(flags),
        "flags": flags,
    })
