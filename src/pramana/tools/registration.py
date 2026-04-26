"""Register the 5 Python UC functions used by the agent.

Run from notebook 08. Idempotent (drops + recreates).
"""
from __future__ import annotations
from pramana.config import CATALOG, SCHEMA


def register_all() -> list[str]:
    from unitycatalog.ai.core.databricks import DatabricksFunctionClient
    from pramana.tools.consistency import score_facility
    from pramana.tools.parse_messy import parse_messy_field
    from pramana.tools.geo import geo_radius
    from pramana.tools.cross_source import cross_source_disagree
    from pramana.tools.search import search_facilities

    client = DatabricksFunctionClient()

    fns = [
        ("score_claim_consistency", score_facility),
        ("parse_messy_field",       parse_messy_field),
        ("geo_radius",              geo_radius),
        ("cross_source_disagree",   cross_source_disagree),
        ("search_facilities",       search_facilities),
    ]

    created: list[str] = []
    for uc_name, py_fn in fns:
        full = f"{CATALOG}.{SCHEMA}.{uc_name}"
        try:
            client.delete_function(full)
        except Exception:
            pass
        info = client.create_python_function(
            func=py_fn, catalog=CATALOG, schema=SCHEMA, replace=True,
        )
        created.append(info.full_name if hasattr(info, "full_name") else full)
    return created
