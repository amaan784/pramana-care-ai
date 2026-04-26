"""Register Unity Catalog **SQL** scalar functions for tools that need lakehouse I/O.

Unity Catalog **Python** scalar UDFs run in an isolated CPython worker with **no**
``SparkSession`` — ``SparkSession.builder.getOrCreate()`` inside the UDF tears down
the JVM gateway (``JAVA_GATEWAY_EXITED``).  SQL ``LANGUAGE SQL`` functions run on
the warehouse/cluster SQL engine and may read Delta tables, ``ai_extract``, H3,
etc.

``search_facilities`` stays a Python UC function (Vector Search REST client only —
see ``registration.py``).
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
  SELECT IFNULL(
    (
      SELECT to_json(
        named_struct(
          'facility_id', g.facility_id,
          'trust_score', g.trust_score,
          'flags', g.flags
        )
      )
      FROM {ns}.gold_facilities g
      WHERE g.facility_id = fid
      LIMIT 1
    ),
    to_json(named_struct('facility_id', fid, 'error', 'not found'))
  )
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
  SELECT to_json(
    named_struct(
      'center', named_struct('lat', p_lat, 'lon', p_lon),
      'radius_km', p_radius_km,
      'specialty', lower(trim(coalesce(p_specialty, ''))),
      'results', coalesce(w.j, array())
    )
  )
  FROM (
    SELECT collect_list(
      named_struct(
        'facility_id', t.facility_id,
        'name', t.name,
        'facility_type', t.facility_type,
        'state', t.state,
        'city', t.city,
        'latitude', t.latitude,
        'longitude', t.longitude,
        'trust_score', t.trust_score,
        'distance_km', t.distance_km,
        'specialties', t.specialties
      )
    ) AS j
    FROM (
      SELECT *
      FROM (
        SELECT
          facility_id,
          name,
          facility_type,
          state,
          city,
          latitude,
          longitude,
          trust_score,
          specialties,
          ST_DistanceSpheroid(
            ST_Point(p_lon, p_lat),
            ST_Point(longitude, latitude)
          ) / 1000.0 AS distance_km
        FROM {ns}.gold_facilities
        WHERE h3_8 IN (
          SELECT h3_h3tostring(h3idx)
          FROM (
            SELECT explode(
              h3_kring(
                h3_longlatash3(p_lon, p_lat, 8),
                greatest(1, cast(ceil(p_radius_km / 0.461) AS INT))
              )
            ) AS h3idx
          ) ring_cells
        )
        AND (
          trim(coalesce(p_specialty, '')) = ''
          OR exists(
            specialties,
            x -> contains(lower(x), lower(trim(p_specialty)))
          )
        )
      ) raw
      WHERE raw.distance_km <= p_radius_km
      ORDER BY raw.distance_km ASC
      LIMIT p_limit
    ) t
  ) w
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
  SELECT IFNULL(
    (
      SELECT to_json(
        named_struct(
          'facility_id', b.facility_id,
          'claim', p_claim,
          'agree',
            cast(
              cast(b.hit_spec AS INT) + cast(b.hit_cap AS INT) + cast(b.hit_eq AS INT)
              + cast(b.hit_proc AS INT) + cast(b.hit_desc AS INT) + cast(b.hit_cd AS INT)
              >= 2 AS BOOLEAN
            ),
          'sources_supporting',
            filter(
              array(
                IF(b.hit_spec, 'specialties', CAST(NULL AS STRING)),
                IF(b.hit_cap, 'capability', CAST(NULL AS STRING)),
                IF(b.hit_eq, 'equipment', CAST(NULL AS STRING)),
                IF(b.hit_proc, 'procedure', CAST(NULL AS STRING)),
                IF(b.hit_desc, 'description', CAST(NULL AS STRING)),
                IF(b.hit_cd, 'capacity_or_doctors', CAST(NULL AS STRING))
              ),
              x -> x IS NOT NULL
            ),
          'sources_total', 6
        )
      )
      FROM (
        SELECT
          facility_id,
          aggregate(
            needle_arr,
            false,
            (acc, n) -> acc OR exists(
              specialties,
              s -> contains(lower(cast(s AS STRING)), n)
            )
          ) AS hit_spec,
          aggregate(
            needle_arr,
            false,
            (acc, n) -> acc OR exists(
              capability,
              s -> contains(lower(cast(s AS STRING)), n)
            )
          ) AS hit_cap,
          aggregate(
            needle_arr,
            false,
            (acc, n) -> acc OR exists(
              equipment,
              s -> contains(lower(cast(s AS STRING)), n)
            )
          ) AS hit_eq,
          aggregate(
            needle_arr,
            false,
            (acc, n) -> acc OR exists(
              procedure,
              s -> contains(lower(cast(s AS STRING)), n)
            )
          ) AS hit_proc,
          aggregate(
            needle_arr,
            false,
            (acc, n) -> acc OR contains(lower(coalesce(description, '')), n)
          ) AS hit_desc,
          (coalesce(capacity, 0) != 0 OR coalesce(number_doctors, 0) != 0) AS hit_cd
        FROM (
          SELECT
            s.*,
            filter(
              split(trim(lower(p_claim)), ' '),
              x -> length(x) > 3
            ) AS needle_arr
          FROM {ns}.silver_facilities_clean s
          WHERE s.facility_id = p_fid
          LIMIT 1
        ) inner0
      ) b
    ),
    to_json(
      named_struct(
        'facility_id', p_fid,
        'claim', p_claim,
        'agree', false,
        'reason', 'facility not found',
        'sources_supporting', cast(array() AS ARRAY<STRING>),
        'sources_total', 0
      )
    )
  )
);
"""
    )
    out.append(f"{ns}.cross_source_disagree")

    return out
