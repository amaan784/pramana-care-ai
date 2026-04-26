# 07 — Genie space setup (manual UI clicks, ~5 minutes)

Genie spaces cannot be created cleanly via SDK on Free Edition without
service-principal gymnastics, so we drive this once by hand.

## Steps

1. **Workspace → Genie → New space**.
2. **Name:** `pramana_facilities`.
3. **Add tables:** `main.pramana.gold_facilities`, `main.pramana.silver_contradictions`, `main.pramana.silver_claims_long`.
4. **Warehouse:** the 2X-Small serverless SQL warehouse used by the bundle.
5. **Instructions** (paste verbatim):

> You answer questions about Indian healthcare facilities. PIN codes are STRINGS with leading zeros — never cast to INT. Always prefer `gold_facilities`. `trust_score` is 0–100 with lower = more contradictions. `flags` is an array of struct{rule_id, severity, message, evidence, citation_column}. `h3_6` and `h3_8` are STRING H3 cell ids. The `facility_type` column normalises the bug `farmacy → pharmacy`; original is in `facility_type_raw`. The `state` column is the canonical state name resolved via `ref_state_aliases`; the verbatim source is in `state_raw`. The dataset has no district column, so use `city` (sourced from `address_city`) — for most rows city ≈ district HQ. Specialty values are camelCase concatenated tokens (e.g. `'medicaloncology'`, `'orthopedicsurgery'`); always match with `contains(lower(x), 'oncolog')` style substrings, not equality. Coordinates have ~23% inaccuracy concentrated in NITI Aspirational Districts.

## 5 Example SQL queries (paste as Genie examples)

```sql
-- Q1: How many facilities per state, ranked.
SELECT state, COUNT(*) AS n FROM main.pramana.gold_facilities
GROUP BY state ORDER BY n DESC;

-- Q2: How many entries had the 'farmacy' typo?
SELECT facility_type_raw, COUNT(*) AS n FROM main.pramana.gold_facilities
WHERE facility_type_raw = 'farmacy' GROUP BY 1;

-- Q3: Cities with zero oncology coverage in a state (city ≈ district HQ).
WITH onc AS (
  SELECT DISTINCT city FROM main.pramana.gold_facilities
  WHERE exists(specialties, x -> contains(lower(x), 'oncolog'))
)
SELECT DISTINCT city FROM main.pramana.gold_facilities
WHERE city NOT IN (SELECT city FROM onc) AND state = 'Bihar';

-- Q4: Lowest-trust hospitals.
SELECT facility_id, name, state, city, trust_score
FROM main.pramana.gold_facilities
WHERE facility_type = 'hospital'
ORDER BY trust_score ASC LIMIT 25;

-- Q5: Counts per contradiction rule and severity.
SELECT rule_id, severity, COUNT(*) AS n
FROM main.pramana.silver_contradictions
GROUP BY rule_id, severity ORDER BY rule_id, severity;
```

## 5 Benchmark questions to seed Genie sample-questions panel

1. *How many facilities per state, ranked descending?*
2. *Which facilities in Bihar claim cardiac surgery but have an empty equipment array?*
3. *How many entries had the 'farmacy' typo and what is their state distribution?*
4. *Which cities in Bihar have zero oncology coverage?*
5. *Show me the 25 lowest-trust hospitals.*

## After saving

Copy the **Genie space ID** from the URL and export it:

```bash
export GENIE_SPACE_ID=<paste-id-here>
```

Add it to `databricks.yml` variables and to the App's environment (see `resources/app.yml`).
