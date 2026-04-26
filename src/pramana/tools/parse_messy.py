"""Parse a single messy free-form facility text field with `ai_extract`.

UC function name: `{CATALOG}.{SCHEMA}.parse_messy_field(text STRING) -> STRING (JSON)`,
where CATALOG/SCHEMA come from `pramana.config` (default: `workspace.pramana`).
"""
from __future__ import annotations


def parse_messy_field(text: str) -> str:
    """Extract structured fields from a messy free-form facility text snippet.

    Calls Databricks' ``ai_extract`` foundation-model function over the labels
    ``specialties``, ``equipment``, ``awards``, ``certifications``, ``services``,
    ``operating_hours``, ``languages``, ``insurance_accepted``. Useful when the
    ``description`` column contains a paragraph of marketing copy and you need
    structured fields back.

    Args:
        text: Raw free-form text to parse. Will be truncated to the first 4000
            characters before the LLM call. Pass an empty string to short-circuit
            and get back ``{}``.

    Returns:
        JSON string with one key per extracted label, mapping to a list of
        strings (or null if the label was not found in the text).
    """
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
