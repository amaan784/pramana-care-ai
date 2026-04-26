"""Geo radius search powered by H3 + ST_DistanceSpheroid (no Mosaic library)."""
from __future__ import annotations


def geo_radius(lat: float, lon: float, radius_km: float = 50.0,
               specialty: str = "", limit: int = 50) -> str:
    """Return facilities within `radius_km` of (lat,lon), optionally filtered by specialty.

    Uses H3 res-8 ring as a coarse pre-filter, then ST_DistanceSpheroid for the
    exact distance. Returns a JSON list ordered by distance ascending.
    """
    import json
    import math
    from pyspark.sql import SparkSession
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
      facility_id, name, facility_type, state, district, latitude, longitude,
      trust_score,
      ST_DistanceSpheroid(
        ST_Point({float(lon)}, {float(lat)}),
        ST_Point(longitude, latitude)
      ) / 1000.0 AS distance_km,
      specialties
    FROM main.pramana.gold_facilities
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
