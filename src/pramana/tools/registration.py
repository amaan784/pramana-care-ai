"""Register Unity Catalog tools used by the agent.

Four table-backed tools are SQL functions. Facility search is provided by the
agent's VectorSearchRetrieverTool, not by a UC Python UDF.
"""
from __future__ import annotations

from pramana.config import CATALOG, SCHEMA

def register_all(spark) -> list[str]:
    """Register the four UC functions in `pramana.config.UC_TOOLS`."""
    from pramana.tools.uc_sql_register import register_uc_sql_functions

    spark.sql(f"DROP FUNCTION IF EXISTS {CATALOG}.{SCHEMA}.search_facilities")
    return list(register_uc_sql_functions(spark, CATALOG, SCHEMA))
