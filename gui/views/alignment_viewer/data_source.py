"""
Data-source contract specific to the AlignmentViewer.

The viewer needs five surfaces over the same source:
 - Sample lookups (to resolve an analyte's member ensembles + sample names)
 - Alignment lookups (the alignments it displays)
 - Ensemble lookups (formula assignments + annotations of the members)
 - EnsembleGroup lookups (user annotations of baskets of ensembles)
 - SimilarityMatrix lookups (cluster mode)

This just composes them into a single type for the viewer to depend on
"""
from typing import Protocol

from core.interfaces.data_sources import (
    SampleDataSource,
    AlignmentDataSource,
    EnsembleDataSource,
    GroupDataSource,
    MatrixDataSource,
)


class AlignmentViewerDataSource(
    SampleDataSource,
    AlignmentDataSource,
    EnsembleDataSource,
    GroupDataSource,
    MatrixDataSource,
    Protocol,
):
    """
    Composite of SampleDataSource + AlignmentDataSource + EnsembleDataSource
     + GroupDataSource + MatrixDataSource.
    """
    ...
