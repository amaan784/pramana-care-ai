"""Unit tests for the 8 contradiction rules. Pure-python; no Databricks needed."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pramana.tools.consistency import evaluate_facility, trust_score


BBOX = {
    "Bihar":   {"min_lat": 24.3, "max_lat": 27.5, "min_lon": 83.3, "max_lon": 88.2},
    "Kerala":  {"min_lat": 8.2,  "max_lat": 12.8, "min_lon": 74.8, "max_lon": 77.4},
}


def _ids(flags) -> set[str]:
    return {f["rule_id"] for f in flags}


def test_r1_advanced_claim_no_evidence():
    row = {
        "facility_id": "T1", "facility_type": "hospital", "state": "Bihar",
        "specialties": ["cardiology"], "capability": ["advanced surgery"],
        "equipment": [], "number_doctors": None, "capacity": None,
        "description": "", "latitude": 25.5, "longitude": 85.0,
    }
    flags = evaluate_facility(row, BBOX)
    assert "R1" in _ids(flags)
    assert any(f["severity"] == "HIGH" for f in flags if f["rule_id"] == "R1")


def test_r1_satisfied_with_equipment():
    row = {
        "facility_id": "T1b", "facility_type": "hospital", "state": "Bihar",
        "specialties": ["cardiology"], "capability": ["cardiac surgery"],
        "equipment": ["ECG", "Echo", "Cath Lab"], "number_doctors": 12, "capacity": 80,
        "description": "", "latitude": 25.5, "longitude": 85.0,
    }
    flags = evaluate_facility(row, BBOX)
    assert "R1" not in _ids(flags)


def test_r2_clinic_claims_24x7():
    row = {
        "facility_id": "T2", "facility_type": "clinic", "state": "Kerala",
        "specialties": [], "equipment": [], "capability": [],
        "description": "We provide 24x7 emergency care", "latitude": 10.0, "longitude": 76.0,
    }
    assert "R2" in _ids(evaluate_facility(row, BBOX))


def test_r2_hospital_claims_24x7_ok():
    row = {
        "facility_id": "T2b", "facility_type": "hospital", "state": "Kerala",
        "specialties": [], "equipment": ["ECG"], "capability": [],
        "description": "We provide 24x7 emergency care", "latitude": 10.0, "longitude": 76.0,
        "capacity": 50, "number_doctors": 5,
    }
    assert "R2" not in _ids(evaluate_facility(row, BBOX))


def test_r3_outside_india():
    row = {"facility_id": "T3", "facility_type": "hospital", "state": "Bihar",
           "specialties": [], "equipment": [], "capability": [], "description": "",
           "latitude": 51.5, "longitude": -0.1, "capacity": 10, "number_doctors": 1}
    assert "R3" in _ids(evaluate_facility(row, BBOX))


def test_r3_state_mismatch():
    row = {"facility_id": "T3b", "facility_type": "hospital", "state": "Kerala",
           "specialties": [], "equipment": [], "capability": [], "description": "",
           "latitude": 25.5, "longitude": 85.0, "capacity": 10, "number_doctors": 1}
    assert "R3" in _ids(evaluate_facility(row, BBOX))


def test_r4_ghost_hospital():
    row = {"facility_id": "T4", "facility_type": "hospital", "state": "Bihar",
           "specialties": [], "equipment": [], "capability": [], "description": "",
           "latitude": 25.5, "longitude": 85.0, "capacity": None, "number_doctors": None}
    assert "R4" in _ids(evaluate_facility(row, BBOX))


def test_r5_farmacy_typo():
    row = {"facility_id": "T5", "facility_type": "farmacy", "state": "Bihar",
           "specialties": [], "equipment": [], "capability": [], "description": "",
           "latitude": 25.5, "longitude": 85.0}
    flags = evaluate_facility(row, BBOX)
    assert "R5" in _ids(flags)
    assert any(f["severity"] == "MED" for f in flags if f["rule_id"] == "R5")


def test_r6_fake_who_award():
    row = {"facility_id": "T6", "facility_type": "hospital", "state": "Bihar",
           "specialties": [], "equipment": ["x-ray"], "capability": [],
           "description": "Recipient of W.HO award 2019 and ISO 9001:2027 certification",
           "latitude": 25.5, "longitude": 85.0, "capacity": 10, "number_doctors": 1}
    assert "R6" in _ids(evaluate_facility(row, BBOX))


def test_r7_specialty_without_equipment():
    row = {"facility_id": "T7", "facility_type": "hospital", "state": "Bihar",
           "specialties": ["cardiology"], "equipment": ["bandage", "thermometer"],
           "capability": [], "description": "", "latitude": 25.5, "longitude": 85.0,
           "capacity": 30, "number_doctors": 5}
    assert "R7" in _ids(evaluate_facility(row, BBOX))


def test_r8_stale_low_signal():
    row = {"facility_id": "T8", "facility_type": "clinic", "state": "Bihar",
           "specialties": [], "equipment": [], "capability": [], "description": "",
           "latitude": 25.5, "longitude": 85.0,
           "recency_of_page_update_months": 36,
           "social_facebook": None, "social_twitter": None, "social_instagram": None,
           "website": None}
    assert "R8" in _ids(evaluate_facility(row, BBOX))


def test_trust_score_math():
    flags = [
        {"rule_id": "R1", "severity": "HIGH", "message": "", "evidence": "", "citation_column": ""},
        {"rule_id": "R5", "severity": "MED",  "message": "", "evidence": "", "citation_column": ""},
        {"rule_id": "R8", "severity": "LOW",  "message": "", "evidence": "", "citation_column": ""},
    ]
    assert trust_score(flags) == 100 - 35 - 15 - 5  # 45


def test_trust_score_floor_zero():
    flags = [{"rule_id": f"R{i}", "severity": "HIGH", "message": "", "evidence": "", "citation_column": ""}
             for i in range(10)]
    assert trust_score(flags) == 0
