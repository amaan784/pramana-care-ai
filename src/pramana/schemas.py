"""Pydantic v2 contracts shared by tools, agent nodes, and eval."""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict

Severity = Literal["HIGH", "MED", "LOW"]


class Facility(BaseModel):
    model_config = ConfigDict(extra="ignore")
    facility_id: str
    name: Optional[str] = None
    facility_type: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = Field(default=None, description="City / district-HQ proxy from address_city")
    pin: Optional[str] = Field(default=None, description="Always string; preserves leading zeros")
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    specialties: list[str] = Field(default_factory=list)
    procedure: list[str] = Field(default_factory=list)
    equipment: list[str] = Field(default_factory=list)
    capability: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    capacity: Optional[int] = None
    number_doctors: Optional[int] = None
    year_established: Optional[int] = None


class Contradiction(BaseModel):
    rule_id: str
    severity: Severity
    message: str
    evidence: str = Field(description="Verbatim span or column value cited as evidence")
    citation_column: str


class TrustScore(BaseModel):
    facility_id: str
    score: int = Field(ge=0, le=100)
    flags: list[Contradiction] = Field(default_factory=list)


class ClaimEvidence(BaseModel):
    claim: str
    facility_id: str
    sources: list[str] = Field(default_factory=list, description="e.g. ['equipment', 'description']")
    agree: bool
    notes: Optional[str] = None


class AgentResponse(BaseModel):
    answer: str
    citations: list[ClaimEvidence] = Field(default_factory=list)
    trust_summary: Optional[str] = None
    confidence: Literal["high", "medium", "low"] = "medium"
