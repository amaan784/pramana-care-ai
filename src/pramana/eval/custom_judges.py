"""Custom MLflow GenAI judges for Pramana."""
from __future__ import annotations

from mlflow.genai.scorers import Guidelines


cites_facility_id = Guidelines(
    name="cites_facility_id",
    guidelines=(
        "The assistant's answer must cite at least one specific facility_id from the "
        "tool result set, formatted in square brackets like [F12345]. Citations to a "
        "facility name without an id are not sufficient. If the question is purely "
        "definitional ('what is X?'), this guideline is not applicable."
    ),
)

flags_known_bugs = Guidelines(
    name="flags_known_bugs",
    guidelines=(
        "When the user asks about data quality, pharmacy/farmacy typos, fake 'W.HO' "
        "awards, or coordinate accuracy in Aspirational Districts, the answer must "
        "explicitly call out the bug (typo / fabricated award / inaccurate coords) "
        "and propose a remediation. If unrelated, this guideline is not applicable."
    ),
)

confidence_line = Guidelines(
    name="ends_with_confidence_line",
    guidelines=(
        "The final line of the answer must be exactly 'Confidence: high', "
        "'Confidence: medium', or 'Confidence: low'."
    ),
)


PRAMANA_JUDGES = [cites_facility_id, flags_known_bugs, confidence_line]
