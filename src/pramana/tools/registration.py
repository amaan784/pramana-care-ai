"""Register Unity Catalog tools used by the agent.

- **Four** table-backed tools are **SQL** scalar functions (``LANGUAGE SQL``) — UC
  Python UDFs cannot call ``SparkSession`` / ``spark.sql`` (see
  ``uc_sql_register.py``).
- Facility search is **not** registered as a UC function. The deployed agent uses
  ``databricks_langchain.VectorSearchRetrieverTool`` directly (see
  ``agent/agent.py``), which runs in the model-serving environment where
  ``databricks-vectorsearch`` is installed. UC Python UDF sandboxes do not inherit
  notebook ``%pip`` packages, so a UC ``search_facilities`` Python function fails
  at runtime with ``ModuleNotFoundError: databricks.vector_search``.

Run from notebook ``08_register_uc_tools`` with the active ``spark`` session.
"""
from __future__ import annotations

from pramana.config import CATALOG, SCHEMA


def register_all(spark) -> list[str]:
    """Register the four UC tools in ``pramana.config.UC_TOOLS``."""
    from pramana.tools.uc_sql_register import register_uc_sql_functions

    # Clean up the previous UC-Python implementation. The agent uses
    # VectorSearchRetrieverTool for this name, not a UC function.
    spark.sql(f"DROP FUNCTION IF EXISTS {CATALOG}.{SCHEMA}.search_facilities")

    return list(register_uc_sql_functions(spark, CATALOG, SCHEMA))
