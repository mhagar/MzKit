"""
AnalyteAnnotation: the user's identification of one AlignedAnalyte - a name
and an (exact) structure. AnalyteGroup is the other annotation kind: a claim
about a basket of analytes ("these are psammaplins"), whose SMILES is a
family scaffold rather than this analyte's structure.

Standalone sidecar keyed by AnalyteUUID, like AnalyteGroup, so it is mutable
while alignments are not, and shows in every (filtered) alignment that
contains the analyte.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.data_structs.uuid_types import AnalyteUUID


@dataclass(frozen=True)
class AnalyteAnnotation:
    analyte_uuid: 'AnalyteUUID'
    name: str = ""      # e.g. "Psammaplin A"
    smiles: str = ""    # this analyte's structure

    @property
    def is_empty(self) -> bool:
        return not (self.name or self.smiles)
