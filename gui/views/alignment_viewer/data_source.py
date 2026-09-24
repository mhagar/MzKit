"""
Data-source contract specific to the AlignmentViewer.

The viewer needs three surfaces over the same source:
 - Sample lookups (to resolve an analyte's member ensembles + sample names)
 - Alignment lookups (the alignments it displays)
 - FormulaAssignment lookups (to show ensemble formulae)

This just composes them into a single type for the viewer to depend on
"""
from typing import Protocol

from core.interfaces.data_sources import (
    SampleDataSource,
    AlignmentDataSource,
    AssignmentDataSource,
)


class AlignmentViewerDataSource(
    SampleDataSource,
    AlignmentDataSource,
    AssignmentDataSource,
    Protocol,
):
    """
    Composite of SampleDataSource + AlignmentDataSource + AssignmentDataSource.
    """
    ...
