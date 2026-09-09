"""
Data-source contract specific to the EnsembleViewer.

The viewer needs two surfaces over the same source:
 - Sample lookups (to resolve an ensemble's owning sample name)
  - FormulaAssignment lookups (to render the compound formula label)

This just composes them into a single type for the viewer to depend on
"""

from typing import Protocol

from core.interfaces.data_sources import (
    SampleDataSource,
    AssignmentDataSource,
)


class EnsembleViewerDataSource(
    SampleDataSource,
    AssignmentDataSource,
    Protocol,
):
    """
    Composite of SampleDataSource + AssignmentDataSource.
    """
    ...
