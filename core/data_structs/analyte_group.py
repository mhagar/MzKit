"""
AnalyteGroup: a user annotation (title, notes, SMILES) of a basket of
AlignedAnalytes, e.g. a named cluster ("Psammaplins") or a single-analyte note.

Standalone sidecar keyed by AnalyteUUIDs, meaning groups stay mutable even if
alignments are immutable.

Filtered child alignments share their parent's analyte UUIDs, so a
group shows in any alignment containing at least one of its members.
"""
from __future__ import annotations

import time
import uuid as uuid_module
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.data_structs.uuid_types import AnalyteUUID, GroupUUID


@dataclass
class AnalyteGroup:
    analyte_uuids: list['AnalyteUUID']          # ordered, no duplicates
    title: str = ""                             # short name, e.g. "Psammaplins"
    notes: str = ""                             # free text
    smiles: str = ""                            # optional structure
    uuid: 'GroupUUID' = field(default_factory=lambda: uuid_module.uuid4().int)
    created_at: float = field(default_factory=time.time)
    # Breadcrumb only (may dangle): the alignment UUID + clustering params
    # the group was created under, or None.
    provenance: Optional[dict] = None
