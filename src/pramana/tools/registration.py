"""Register Unity Catalog tools used by the agent.

- **Four** table-backed tools are **SQL** scalar functions (``LANGUAGE SQL``) — UC
  Python UDFs cannot call ``SparkSession`` / ``spark.sql`` (see
  ``uc_sql_register.py``).
- **search_facilities** stays a **Python** UC function: it only calls the Vector
  Search REST client (no Spark in the UDF body).

Run from notebook ``08_register_uc_tools`` with the active ``spark`` session.
"""
from __future__ import annotations

from pramana.config import CATALOG, SCHEMA


def register_all(spark) -> list[str]:
    """Register all five UC tools. Pass the notebook/job ``spark`` session."""
    from pramana.tools.uc_sql_register import register_uc_sql_functions
    from unitycatalog.ai.core.databricks import DatabricksFunctionClient
    from pramana.tools.search import search_facilities

    created: list[str] = list(register_uc_sql_functions(spark, CATALOG, SCHEMA))

    client = DatabricksFunctionClient()
    full = f"{CATALOG}.{SCHEMA}.{search_facilities.__name__}"
    try:
        client.delete_function(full)
    except Exception:
        pass
    info = client.create_python_function(
        func=search_facilities,
        catalog=CATALOG,
        schema=SCHEMA,
        replace=True,
    )
    created.append(info.full_name if hasattr(info, "full_name") else full)
    return created
