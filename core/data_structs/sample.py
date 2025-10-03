"""
Dataclass representing a *sample*, i.e. a unique chemical mixture
that was either chemically or biologically analyzed
"""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from core.data_structs import (
        SampleUUID,
        Injection, Fingerprint,
    )


@dataclass
class Sample:
    name: str
    uuid: 'SampleUUID' = field(
        default_factory=lambda: uuid.uuid4().int
    )
    injection: Optional['Injection'] = None
    fingerprint: Optional['Fingerprint'] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        # if not self.injection and not self.fingerprint:
        #     raise ValueError(
        #         f"Neither injection nor fingeprint argument given. "
        #         f"(sample {self.name}"
        #     )
        pass

    def set_injection(
        self,
        injection: 'Injection',
    ):
        if self.injection:
            raise ValueError(
                f"Sample already has an Injection registered: {self.injection}"
            )

        self.injection = injection

    def set_fingerprint(
        self,
        fingerprint: 'Fingerprint',
    ):
        if self.fingerprint:
            raise ValueError(
                f"Sample already has a Fingeprint registered: {self.fingerprint}"
            )

        self.fingerprint = fingerprint

    def __repr__(self):
        return (f"Sample("
                f"name={self.name},"
                f"uuid=...{str(self.uuid)[-5:]}, "
                f"injection: {(self.injection is not None)}, "
                f"fingerprint: {(self.fingerprint is not None)}, "
                f"metadata: {self.metadata}")
