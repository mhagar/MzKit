"""
Different types of UUIDs defined here for easy type checking
"""
from typing import NewType

SampleUUID = NewType(
    'SampleUUID',
    int,
)
FingerprintUUID = NewType(
    'FingerprintUUID',
    int,
)
InjectionUUID = NewType(
    'InjectionUUID',
    int,
)
EnsembleUUID = NewType(
    'EnsembleUUID',
    int,
)
ScanArrayUUID = NewType(
    'ScanArrayUUID',
    int,
)
AnalyteTableUUID = NewType(
    'AnalyteTableUUID',
    int,
)
AnalyteID = NewType(
    'AnalyteID',
    int,
)
