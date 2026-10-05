"""systems/compass/sandbox.py: Tidewater state for agent runs.

A deterministic, in-memory stand-in for Tidewater's systems.
Each run starts from a named seed and can be inspected.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Shipment(BaseModel):
    service: Literal["standard", "priority"]
    status: str
    address_id: str


class Claim(BaseModel):
    shipment_id: str
    claim_type: Literal["damage", "loss"]
    status: Literal["open", "approved", "rejected"]


class Sandbox(BaseModel):
    shipments: dict[str, Shipment]
    claims: dict[str, Claim] = {}
    credits: dict[str, str] = {}     # claim id -> shipment id

    # Tools the agent may call ---------------------------------
    def reroute_shipment(self, shipment_id: str,
                         address_id: str) -> str:
        s = self.shipments[shipment_id]
        if s.status == "delivered":
            return "error: already delivered"
        s.address_id = address_id
        return "ok"

    def open_claim(self, shipment_id: str,
                   claim_type: Literal["damage", "loss"]) -> str:
        cid = f"CLM-{len(self.claims) + 1:04d}"
        self.claims[cid] = Claim(shipment_id=shipment_id,
                                 claim_type=claim_type,
                                 status="open")
        return cid

    def issue_credit(self, claim_id: str) -> str:
        c = self.claims[claim_id]
        s = self.shipments[c.shipment_id]
        if s.service != "priority" or c.status != "approved":
            return "error: not eligible"
        self.credits[claim_id] = c.shipment_id
        return "ok"

    # Inspection -----------------------------------------------
    def snapshot(self) -> dict[str, object]:
        flat: dict[str, object] = {}
        for sid, s in self.shipments.items():
            for k, v in s.model_dump().items():
                flat[f"shipments.{sid}.{k}"] = v
        for cid, c in self.claims.items():
            for k, v in c.model_dump().items():
                flat[f"claims.{cid}.{k}"] = v
        for cid, sid in self.credits.items():
            flat[f"credits.{cid}"] = sid
        return flat


SEEDS: dict[str, Sandbox] = {
    "two-shipments": Sandbox(shipments={
        "TW-1042": Shipment(service="priority",
                            status="in_transit",
                            address_id="ADDR-1"),
        "TW-2210": Shipment(service="standard",
                            status="in_transit",
                            address_id="ADDR-7")}),
}


def fresh(seed: str) -> Sandbox:
    """A new, independent copy of a named starting state."""
    return SEEDS[seed].model_copy(deep=True)
