"""Cross-source disagreement check used by the Verifier node.

Given a `facility_id` and a single textual `claim`, returns whether the claim
is corroborated by ≥2 independent sources (specialties array, equipment array,
description text, capability array, capacity/doctors fill-rate).
"""
from __future__ import annotations

from pramana.config import CATALOG, SCHEMA


def cross_source_disagree(facility_id: str, claim: str) -> str:
    import json
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    rows = spark.sql(
        f"SELECT specialties, capability, equipment, procedure, description, "
        f"capacity, number_doctors FROM {CATALOG}.{SCHEMA}.silver_facilities_clean "
        f"WHERE facility_id = :fid LIMIT 1",
        args={"fid": facility_id},
    ).collect()
    if not rows:
        return json.dumps({"facility_id": facility_id, "claim": claim, "agree": False,
                            "reason": "facility not found", "sources_supporting": [], "sources_total": 0})

    r = rows[0].asDict(recursive=True)
    needles = [w.strip().lower() for w in str(claim).lower().split() if len(w) > 3]

    def hit(field_value) -> bool:
        if field_value is None:
            return False
        if isinstance(field_value, list):
            text = " ".join(str(x).lower() for x in field_value)
        else:
            text = str(field_value).lower()
        return any(n in text for n in needles)

    source_hits = {
        "specialties": hit(r.get("specialties")),
        "capability":  hit(r.get("capability")),
        "equipment":   hit(r.get("equipment")),
        "procedure":   hit(r.get("procedure")),
        "description": hit(r.get("description")),
        "capacity_or_doctors": (r.get("capacity") not in (None, 0)) or (r.get("number_doctors") not in (None, 0)),
    }
    supporting = [k for k, v in source_hits.items() if v]
    return json.dumps({
        "facility_id": facility_id,
        "claim": claim,
        "agree": len(supporting) >= 2,
        "sources_supporting": supporting,
        "sources_total": len(source_hits),
    })
