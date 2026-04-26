"""Register the 5 Python UC functions used by the agent.

Run from notebook 08. Idempotent (drops + recreates).
"""
from __future__ import annotations
from pramana.config import CATALOG, SCHEMA


def register_all() -> list[str]:
    from unitycatalog.ai.core.databricks import DatabricksFunctionClient
    from pramana.tools.consistency import score_claim_consistency
    from pramana.tools.parse_messy import parse_messy_field
    from pramana.tools.geo import geo_radius
    from pramana.tools.cross_source import cross_source_disagree
    from pramana.tools.search import search_facilities

    client = DatabricksFunctionClient()

    # `create_python_function` registers under the callable's ``__name__``, not an
    # arbitrary alias. Always delete/create using ``py_fn.__name__`` so we don't
    # orphan e.g. ``score_facility`` while the agent expects ``score_claim_consistency``.
    for stale in ("score_facility",):
        try:
            client.delete_function(f"{CATALOG}.{SCHEMA}.{stale}")
        except Exception:
            pass

    fns = (
        score_claim_consistency,
        parse_messy_field,
        geo_radius,
        cross_source_disagree,
        search_facilities,
    )

    created: list[str] = []
    for py_fn in fns:
        full = f"{CATALOG}.{SCHEMA}.{py_fn.__name__}"
        try:
            client.delete_function(full)
        except Exception:
            pass
        info = client.create_python_function(
            func=py_fn, catalog=CATALOG, schema=SCHEMA, replace=True,
        )
        created.append(info.full_name if hasattr(info, "full_name") else full)
    return created
