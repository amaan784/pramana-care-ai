"""Register Unity Catalog **SQL** scalar functions for tools that need lakehouse I/O.

Unity Catalog **Python** scalar UDFs run in an isolated CPython worker with **no**
``SparkSession`` — ``SparkSession.builder.getOrCreate()`` inside the UDF tears down
the JVM gateway (``JAVA_GATEWAY_EXITED``).  SQL ``LANGUAGE SQL`` functions run on
the warehouse/cluster SQL engine and may read Delta tables, ``ai_extract``, H3,
etc.

Facility search is handled by ``VectorSearchRetrieverTool`` in the deployed agent,
not by a Unity Catalog function.
"""
from __future__ import annotations


def register_uc_sql_functions(spark, catalog: str, schema: str) -> list[str]:
    """Run ``CREATE OR REPLACE FUNCTION`` for the four table-backed tools.

    Must be called from a Databricks notebook/job where ``spark`` is the active
    session (driver), not from inside a UC Python UDF.
    """
    ns = f"{catalog}.{schema}"
    out: list[str] = []

    # Remove prior Python-UDF versions (or failed experiments) so LANGUAGE SQL
    # can bind cleanly.
    for fn in (
        "score_claim_consistency",
        "parse_messy_field",
        "geo_radius",
        "cross_source_disagree",
    ):
        spark.sql(f"DROP FUNCTION IF EXISTS {ns}.{fn}")

    # --- score_claim_consistency: read materialised gold (trust + flags) -------
    spark.sql(
        f"""
CREATE OR REPLACE FUNCTION {ns}.score_claim_consistency(fid STRING)
RETURNS STRING
LANGUAGE SQL
COMMENT 'Trust score + contradiction flags for one facility (from gold_facilities).'
RETURN (
  WITH params AS (
    SELECT fid AS q_fid
  )
  SELECT CASE
    WHEN g.facility_id IS NULL THEN
      to_json(named_struct('facility_id', p.q_fid, 'error', 'not found'))
    ELSE
      to_json(
        named_struct(
          'facility_id', g.facility_id,
          'trust_score', g.trust_score,
          'flags', g.flags
        )
      )
    END
  FROM params p
  LEFT JOIN {ns}.gold_facilities g
    ON g.facility_id = p.q_fid
  LIMIT 1
);
"""
    )
    out.append(f"{ns}.score_claim_consistency")

    # --- parse_messy_field: ai_extract (SQL engine) ---------------------------
    spark.sql(
        f"""
CREATE OR REPLACE FUNCTION {ns}.parse_messy_field(p_text STRING)
RETURNS STRING
LANGUAGE SQL
COMMENT 'Structured extraction from messy free text via ai_extract.'
RETURN (
  SELECT CASE
    WHEN p_text IS NULL OR length(trim(p_text)) = 0 THEN '{{}}'
    ELSE to_json(
      ai_extract(
        p_text,
        array(
          'specialties', 'equipment', 'awards', 'certifications', 'services',
          'operating_hours', 'languages', 'insurance_accepted'
        )
      )
    )
  END
);
"""
    )
    out.append(f"{ns}.parse_messy_field")

    # --- geo_radius: same logic as tools/geo.py, no PySpark in a UDF -----------
    spark.sql(
        f"""
CREATE OR REPLACE FUNCTION {ns}.geo_radius(
  p_lat DOUBLE,
  p_lon DOUBLE,
  p_radius_km DOUBLE,
  p_specialty STRING,
  p_limit INT
)
RETURNS STRING
LANGUAGE SQL
COMMENT 'Facilities within p_radius_km of (p_lat,p_lon); optional specialty substring filter.'
RETURN (
  WITH params AS (
    SELECT
      p_lat AS q_lat,
      p_lon AS q_lon,
      p_radius_km AS q_radius_km,
      lower(trim(coalesce(p_specialty, ''))) AS q_specialty,
      p_limit AS q_limit
  ),
  scored AS (
    SELECT
      g.facility_id,
      g.name,
      g.facility_type,
      g.state,
      g.city,
      g.latitude,
      g.longitude,
      g.trust_score,
      g.specialties,
      p.q_lat,
      p.q_lon,
      p.q_radius_km,
      p.q_specialty,
      p.q_limit,
      ST_DistanceSpheroid(
        ST_Point(p.q_lon, p.q_lat),
        ST_Point(g.longitude, g.latitude)
      ) / 1000.0 AS distance_km
    FROM {ns}.gold_facilities g
    CROSS JOIN params p
    WHERE g.latitude IS NOT NULL
      AND g.longitude IS NOT NULL
      AND (
        p.q_specialty = ''
        OR exists(
          g.specialties,
          x -> contains(lower(x), p.q_specialty)
        )
      )
  ),
  ranked AS (
    SELECT
      *,
      row_number() OVER (ORDER BY distance_km ASC) AS rn
    FROM scored
    WHERE distance_km <= q_radius_km
  )
  SELECT to_json(
    named_struct(
      'center', named_struct('lat', max(q_lat), 'lon', max(q_lon)),
      'radius_km', max(q_radius_km),
      'specialty', max(q_specialty),
      'results', coalesce(
        collect_list(
          named_struct(
            'facility_id', facility_id,
            'name', name,
            'facility_type', facility_type,
            'state', state,
            'city', city,
            'latitude', latitude,
            'longitude', longitude,
            'trust_score', trust_score,
            'distance_km', distance_km,
            'specialties', specialties
          )
        ),
        array()
      )
    )
  )
  FROM ranked
  WHERE rn <= q_limit
);
"""
    )
    out.append(f"{ns}.geo_radius")

    # --- cross_source_disagree --------------------------------------------------
    spark.sql(
        f"""
CREATE OR REPLACE FUNCTION {ns}.cross_source_disagree(p_fid STRING, p_claim STRING)
RETURNS STRING
LANGUAGE SQL
COMMENT 'True if >=2 independent source columns support tokens (>3 chars) from p_claim.'
RETURN (
  WITH params AS (
    SELECT
      p_fid AS q_fid,
      p_claim AS q_claim,
      filter(
        split(trim(lower(p_claim)), ' '),
        x -> length(x) > 3
      ) AS q_needles
  ),
  base AS (
    SELECT
      p.q_fid,
      p.q_claim,
      p.q_needles,
      s.facility_id,
      s.specialties,
      s.capability,
      s.equipment,
      s.procedure,
      s.description,
      s.capacity,
      s.number_doctors
    FROM params p
    LEFT JOIN {ns}.silver_facilities_clean s
      ON s.facility_id = p.q_fid
    LIMIT 1
  ),
  hits AS (
    SELECT
      q_fid,
      q_claim,
      facility_id,
      CASE WHEN facility_id IS NULL THEN false ELSE aggregate(
        q_needles,
        false,
        (acc, n) -> acc OR exists(
          specialties,
          s -> contains(lower(cast(s AS STRING)), n)
        )
      ) END AS hit_spec,
      CASE WHEN facility_id IS NULL THEN false ELSE aggregate(
        q_needles,
        false,
        (acc, n) -> acc OR exists(
          capability,
          s -> contains(lower(cast(s AS STRING)), n)
        )
      ) END AS hit_cap,
      CASE WHEN facility_id IS NULL THEN false ELSE aggregate(
        q_needles,
        false,
        (acc, n) -> acc OR exists(
          equipment,
          s -> contains(lower(cast(s AS STRING)), n)
        )
      ) END AS hit_eq,
      CASE WHEN facility_id IS NULL THEN false ELSE aggregate(
        q_needles,
        false,
        (acc, n) -> acc OR exists(
          procedure,
          s -> contains(lower(cast(s AS STRING)), n)
        )
      ) END AS hit_proc,
      CASE WHEN facility_id IS NULL THEN false ELSE aggregate(
        q_needles,
        false,
        (acc, n) -> acc OR contains(lower(coalesce(description, '')), n)
      ) END AS hit_desc,
      CASE
        WHEN facility_id IS NULL THEN false
        ELSE (coalesce(capacity, 0) != 0 OR coalesce(number_doctors, 0) != 0)
      END AS hit_cd
    FROM base
  )
  SELECT CASE
    WHEN facility_id IS NULL THEN
      to_json(
        named_struct(
          'facility_id', q_fid,
          'claim', q_claim,
          'agree', false,
          'reason', 'facility not found',
          'sources_supporting', cast(array() AS ARRAY<STRING>),
          'sources_total', 0
        )
      )
    ELSE
      to_json(
        named_struct(
          'facility_id', facility_id,
          'claim', q_claim,
          'agree',
            cast(
              cast(hit_spec AS INT) + cast(hit_cap AS INT) + cast(hit_eq AS INT)
              + cast(hit_proc AS INT) + cast(hit_desc AS INT) + cast(hit_cd AS INT)
              >= 2 AS BOOLEAN
            ),
          'sources_supporting',
            filter(
              array(
                IF(hit_spec, 'specialties', CAST(NULL AS STRING)),
                IF(hit_cap, 'capability', CAST(NULL AS STRING)),
                IF(hit_eq, 'equipment', CAST(NULL AS STRING)),
                IF(hit_proc, 'procedure', CAST(NULL AS STRING)),
                IF(hit_desc, 'description', CAST(NULL AS STRING)),
                IF(hit_cd, 'capacity_or_doctors', CAST(NULL AS STRING))
              ),
              x -> x IS NOT NULL
            ),
          'sources_total', 6
        )
      )
    END
  FROM hits
);
"""
    )
    out.append(f"{ns}.cross_source_disagree")

    return out
