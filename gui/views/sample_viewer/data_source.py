"""
Data-source contract specific to the SampleViewer

The viewer needs two surfaces over the same source:
 - Sample lookups (to resolve an ensemble's owning sample name)
 - Ensemble lookups (its formula assignment, for the compound formula label)

This just composes them into a single type for the viewer to depend on
"""

from typing import Protocol

from core.interfaces.data_sources import (
    SampleDataSource,
    EnsembleDataSource,
)

class SampleViewerDataSource(
    SampleDataSource,
    EnsembleDataSource,
    Protocol,
):
    """
    Composite of SampleDataSource + EnsembleDataSource.
    """
    ...
