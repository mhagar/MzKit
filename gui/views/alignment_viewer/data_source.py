"""
Data-source contract specific to the AlignmentViewer.

The viewer needs six surfaces over the same source:
 - Sample lookups (to resolve an analyte's member ensembles + sample names)
 - Alignment lookups (the alignments it displays)
 - Ensemble lookups (formula assignments, to show ensemble formulae)
 - AnalyteGroup lookups (user annotations of analyte baskets)
 - AnalyteAnnotation lookups (a single analyte's name + structure)
 - SimilarityMatrix lookups (cluster mode)

This just composes them into a single type for the viewer to depend on
"""
from typing import Protocol

from core.interfaces.data_sources import (
    SampleDataSource,
    AlignmentDataSource,
    EnsembleDataSource,
    GroupDataSource,
    AnnotationDataSource,
    MatrixDataSource,
)


class AlignmentViewerDataSource(
    SampleDataSource,
    AlignmentDataSource,
    EnsembleDataSource,
    GroupDataSource,
    AnnotationDataSource,
    MatrixDataSource,
    Protocol,
):
    """
    Composite of SampleDataSource + AlignmentDataSource + EnsembleDataSource
     + GroupDataSource + AnnotationDataSource + MatrixDataSource.
    """
    ...
