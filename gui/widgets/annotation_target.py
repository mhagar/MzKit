"""
What a host (Alignment Viewer, Ensemble Viewer) hands the annotation tabs for
its focus: where edits go, and what to search with. Qt-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.data_structs import Ensemble
    from core.utils.array_types import SpectrumArray


@dataclass
class AnnotationTarget:
    key: Hashable                        # per-focus cache key (rankings, search results, scores)
    ensemble: Optional['Ensemble']       # whose EnsembleAnnotation edits go to (None: nothing to edit)
    ms2: Optional['SpectrumArray'] = None
    precursor_mz: Optional[float] = None
    formula: Optional[str] = None        # structure-search query prefill
    adduct: Optional[str] = None         # e.g. '[M+H]+'; None: unknown
    noun: str = "ensemble"               # "This <noun> has no MS2"
    header: str = ""                     # the editor's "whose annotation" line
    header_tip: str = ""
    members: list['Ensemble'] = field(default_factory=list)   # "Apply to all" (hidden when <= 1)
    embedding_id: Optional[str] = None   # key into imported query embeddings (analyte uuid str)


def ensemble_adduct(ensemble: 'Ensemble') -> Optional[str]:
    """The ion name (e.g. '[M+Na]+') of an ensemble's chosen (else top) formula candidate."""
    from core.cli.rank_candidates import ion_name

    assignment = ensemble.formula_assignment
    candidate = assignment and (assignment.chosen or assignment.top)
    if candidate and candidate.adduct:
        return ion_name(candidate.adduct, assignment.charge)
    return None
