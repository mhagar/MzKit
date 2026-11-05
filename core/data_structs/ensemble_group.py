"""
EnsembleGroup: a user annotation (title, notes, SMILES) of a basket of
Ensembles, e.g. a named cluster ("Psammaplins").

Keyed by EnsembleUUIDs, so a group is independent of any alignment: it shows
in every alignment containing one of its members (an analyte is in it when any
of its member ensembles is), and survives re-aligning. A group that looks odd
against another sample's members is useful signal.
"""
from __future__ import annotations

import time
import uuid as uuid_module
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.data_structs.uuid_types import EnsembleUUID, GroupUUID


@dataclass
class EnsembleGroup:
    ensemble_uuids: list['EnsembleUUID']        # ordered, no duplicates
    title: str = ""                             # short name, e.g. "Psammaplins"
    notes: str = ""                             # free text
    smiles: str = ""                            # optional structure
    uuid: 'GroupUUID' = field(default_factory=lambda: uuid_module.uuid4().int)
    created_at: float = field(default_factory=time.time)
    # Breadcrumb only (may dangle): the alignment UUID + clustering params
    # the group was created under, or None.
    provenance: Optional[dict] = None
