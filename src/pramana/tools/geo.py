"""Geo radius search powered by H3 + ST_DistanceSpheroid (no Mosaic library)."""
from __future__ import annotations


def geo_radius(lat: float, lon: float, radius_km: float,
               specialty: str, limit: int) -> str:
    """Return facilities within ``radius_km`` of (lat, lon), optionally filtered by specialty.

    Uses an H3 res-8 ring as a coarse pre-filter, then ST_DistanceSpheroid for the
    exact distance. Returns a JSON list ordered by distance ascending.

    Args:
        lat: Center-point latitude in decimal degrees (WGS-84). Indian range
            roughly 6.5 to 35.5.
        lon: Center-point longitude in decimal degrees (WGS-84). Indian range
            roughly 68.0 to 97.5.
        radius_km: Search radius in kilometres. Reasonable values are 5–500.
            Pass 50.0 if the user did not specify a radius.
        specialty: Optional specialty filter as a lowercase substring matched
            against the ``specialties`` array (e.g. ``"oncolog"``,
            ``"cardio"``, ``"orthopedi"``). Pass an empty string ``""`` for
            no specialty filter.
        limit: Maximum number of facilities to return, ordered by ascending
            distance. Pass 50 if the user did not specify a limit.

    Returns:
        JSON string with keys ``center``, ``radius_km``, ``specialty`` and
        ``results`` (a list of facility dicts including ``facility_id``,
        ``name``, ``state``, ``city``, ``trust_score``, ``distance_km`` and
        ``specialties``).
    """
    import json
    import math
    from pramana.config import CATALOG, SCHEMA
    from pyspark.sql import SparkSession

    cat = os.environ.get("PRAMANA_CATALOG", "workspace")
    sch = os.environ.get("PRAMANA_SCHEMA", "pramana")

    spark = SparkSession.builder.getOrCreate()

    edge_len_km_res8 = 0.461
    k = max(1, int(math.ceil(float(radius_km) / edge_len_km_res8)))

    spec = (specialty or "").strip().lower().replace("'", "")
    spec_clause = (
        f"AND exists(specialties, x -> contains(lower(x), '{spec}'))" if spec else ""
    )

    sql = f"""
    WITH center AS (
      SELECT h3_longlatash3({float(lon)}, {float(lat)}, 8) AS h
    ),
    ring AS (
      SELECT explode(h3_kring((SELECT h FROM center), {k})) AS hex
    )
    SELECT
      facility_id, name, facility_type, state, city, latitude, longitude,
      trust_score,
      ST_DistanceSpheroid(
        ST_Point({float(lon)}, {float(lat)}),
        ST_Point(longitude, latitude)
      ) / 1000.0 AS distance_km,
      specialties
    FROM {CATALOG}.{SCHEMA}.gold_facilities
    WHERE h3_8 IN (SELECT hex FROM ring)
      {spec_clause}
    HAVING distance_km <= {float(radius_km)}
    ORDER BY distance_km ASC
    LIMIT {int(limit)}
    """
    rows = spark.sql(sql).toJSON().collect()
    return json.dumps({
        "center": {"lat": lat, "lon": lon},
        "radius_km": radius_km,
        "specialty": spec,
        "results": [json.loads(r) for r in rows],
    })
