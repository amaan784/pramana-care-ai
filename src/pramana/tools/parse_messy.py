"""Parse a single messy free-form facility text field with `ai_extract`.

UC function name: `main.pramana.parse_messy_field(text STRING) -> STRING (JSON)`.
"""
from __future__ import annotations


def parse_messy_field(text: str) -> str:
    """Extract structured fields from a messy facility note. Returns JSON string."""
    import json
    from pyspark.sql import SparkSession
    if not text or not str(text).strip():
        return json.dumps({})
    spark = SparkSession.builder.getOrCreate()
    labels = (
        "specialties, equipment, awards, certifications, services, "
        "operating_hours, languages, insurance_accepted"
    )
    df = spark.sql(
        "SELECT ai_extract(:t, array("
        "'specialties','equipment','awards','certifications','services',"
        "'operating_hours','languages','insurance_accepted'"
        ")) AS x",
        args={"t": str(text)[:4000]},
    )
    val = df.collect()[0]["x"]
    try:
        return json.dumps(val if isinstance(val, dict) else json.loads(val))
    except Exception:
        return json.dumps({"raw": str(val), "labels": labels})
