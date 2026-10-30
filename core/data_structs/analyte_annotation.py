"""
AnalyteAnnotation: what's known about one AlignedAnalyte - a list of
hypotheses (proposed structures, from the user or a tool), the one chosen as
its structure, and the analyte's name, MSI confidence level and notes.
AnalyteGroup is the other annotation kind: a claim about a basket of analytes
("these are psammaplins"), whose SMILES is a family scaffold rather than this
analyte's structure.

Standalone sidecar keyed by AnalyteUUID, like AnalyteGroup,
so it is mutable while alignments are not, and shows in every (filtered)
 alignment that contains the analyte. Frozen: edits replace the whole
annotation in the registry (dataclasses.replace), which signals the change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.data_structs.uuid_types import AnalyteUUID, SampleUUID

USER = "user"   # Hypothesis.origin of a hand-typed structure
# MSI (Metabolomics Standards Initiative) identification levels
MSI_LEVELS = {
    1: "Identified compound (vs. a reference standard)",
    2: "Putatively annotated compound",
    3: "Putatively characterised compound class",
    4: "Unknown compound",
}


@dataclass(frozen=True)
class Hypothesis:
    smiles: str
    name: str = ""
    origin: str = USER   # USER, or the database / tool it came from, e.g. "COCONUT"
    # What's claimed: 'structure' (the analyte's exact structure). Room for
    # hazier claims from classifiers / SNAP-MF ('substructure', 'class')
    claim: str = 'structure'
    db_id: str = ""
    sample_uuid: Optional['SampleUUID'] = None   # whose MS2 it was proposed / scored for
    scores: dict = field(default_factory=dict)   # e.g. {"explained": "7/10", "cosine": 0.61}


@dataclass(frozen=True)
class AnalyteAnnotation:
    analyte_uuid: 'AnalyteUUID'
    name: str = ""      # e.g. "Psammaplin A"; "" goes by the chosen hypothesis's
    msi_level: Optional[int] = None
    notes: str = ""
    hypotheses: tuple[Hypothesis, ...] = ()
    chosen_idx: Optional[int] = None

    @property
    def chosen(self) -> Optional[Hypothesis]:
        return None if self.chosen_idx is None else self.hypotheses[self.chosen_idx]

    @property
    def label(self) -> str:
        return self.name or (self.chosen.name if self.chosen else "")

    @property
    def smiles(self) -> str:
        """The chosen structure's SMILES ("" if none)."""
        return self.chosen.smiles if self.chosen else ""

    @property
    def is_empty(self) -> bool:
        return not (self.name or self.notes or self.msi_level or self.hypotheses)

    @classmethod
    def from_dict(cls, d: dict) -> 'AnalyteAnnotation':
        """Inverse of dataclasses.asdict (persistence)."""
        return cls(**{**d, 'hypotheses': tuple(Hypothesis(**h) for h in d.get('hypotheses', ()))})
