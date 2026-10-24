"""
Resolution helpers for the AlignmentViewer: turns plot targets (ensemble /
analyte HoverTargets) into the domain data the panels display - member
ensembles, representative spectra, formulae and pairwise scores.

No widgets here; one AlignmentContext exists per displayed alignment.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from core.cli.align_ensembles import PairScore, score_spectrum_pair
from core.data_structs.alignment import RepresentativeSpectrum
from core.utils.formula_formatting import format_assignment_formula_text

if TYPE_CHECKING:
    from core.data_structs import Ensemble, SampleUUID
    from core.data_structs.alignment import AlignedAnalyte, EnsembleAlignment
    from core.utils.array_types import SpectrumArray
    from gui.views.alignment_viewer.data_source import AlignmentViewerDataSource
    from gui.widgets.alignment_plot.layout import HoverTarget


@dataclass(frozen=True)
class ItemSpectra:
    """
    What a selected target contributes to a comparison: its representative
    spectra plus the numbers the compare table reports.
    """
    label: str
    ms1: 'SpectrumArray'
    ms2: Optional['SpectrumArray']
    rt: float
    base_mz: float
    formula_label: str = ""   # for display, e.g. "C10H12O" or "C10H12O?"
    formula_count: str = ""   # analytes: members agreeing, e.g. "3/4"
    formula: str = ""         # bare, for arithmetic
    analyte_uuid: Optional[int] = None


class AlignmentContext:
    def __init__(
        self,
        alignment: 'EnsembleAlignment',
        data_source: 'AlignmentViewerDataSource',
    ):
        self.alignment = alignment
        self.data_source = data_source

    # -- lookups ------------------------------------------------------------

    def sample_name(self, sample_uuid: 'SampleUUID') -> str:
        sample = self.data_source.get_sample(sample_uuid)
        return sample.name if sample else f"...{str(sample_uuid)[-5:]}"

    def members(
        self,
        analyte: 'AlignedAnalyte',
    ) -> dict['SampleUUID', 'Ensemble']:
        """The analyte's resolvable member ensembles, keyed by sample uuid."""
        return analyte.resolve_members(self.data_source.get_sample)

    def representative(
        self,
        analyte: 'AlignedAnalyte',
    ) -> Optional[RepresentativeSpectrum]:
        return analyte.representative_spectrum(self.members(analyte))

    def formula(
            self,
            ensemble: 'Ensemble',
    ) -> str:
        """
        Plain-text formula for an ensemble (see format_assignment_formula_text)
        """
        return format_assignment_formula_text(
            self.data_source.get_assignment_for_source(ensemble.uuid)
        )

    def formula_agreement(
        self,
        analyte: 'AlignedAnalyte',
    ) -> Optional[tuple[str, int, int]]:
        """
        Most common member formula as (formula, count, n_members), or None
        if no member has a formula. Unaccepted ('?') candidates count as
        their bare formula.
        """
        members = self.members(analyte)
        formulas = [
            f.rstrip('?') for f in (self.formula(e) for e in members.values())
            if f
        ]
        if not formulas:
            return None
        formula, count = Counter(formulas).most_common(1)[0]
        return formula, count, len(members)

    def analyte_label(self, analyte: 'AlignedAnalyte') -> str:
        """"m/z @ RT", prefixed by the analyte's annotated name, if any."""
        label = f"{analyte.consensus_mz:.4f} @ {analyte.consensus_rt:.1f}s"
        annotation = self.data_source.get_annotation(analyte.uuid)
        return f"{annotation.name} ({label})" if annotation and annotation.name else label

    def formula_label(self, analyte: 'AlignedAnalyte') -> str:
        """`formula_agreement` as "C10H12O (3/4)"; empty if no formula."""
        agreement = self.formula_agreement(analyte)
        return f"{agreement[0]} ({agreement[1]}/{agreement[2]})" if agreement else ""

    # -- comparison ---------------------------------------------------------

    def item_spectra(self, target: 'HoverTarget') -> Optional[ItemSpectra]:
        if target.kind == 'ensemble' and target.ensemble is not None:
            ensemble = target.ensemble
            composite = ensemble.composite_spectrum
            formula = self.formula(ensemble)
            return ItemSpectra(
                label=self.sample_name(target.sample_uuid),
                ms1=composite.ms1,
                ms2=composite.ms2,
                rt=float(ensemble.peak_rt),
                base_mz=float(ensemble.base_mz),
                formula_label=formula,
                formula=formula.rstrip('?'),
            )

        if target.kind == 'analyte' and target.analyte is not None:
            analyte = target.analyte
            rep = self.representative(analyte)
            if rep is None:
                return None
            agreement = self.formula_agreement(analyte)
            return ItemSpectra(
                label=self.analyte_label(analyte),
                ms1=rep.composite.ms1,
                ms2=rep.composite.ms2,
                rt=float(analyte.consensus_rt),
                base_mz=rep.base_mz,
                formula_label=agreement[0] if agreement else "",
                formula_count=(
                    f"{agreement[1]}/{agreement[2]}" if agreement else ""
                ),
                formula=agreement[0] if agreement else "",
                analyte_uuid=analyte.uuid,
            )

        return None

    def score(
            self,
            a: ItemSpectra,
            b: ItemSpectra,
    ) -> PairScore:
        """
        Cosine scores between two items, as alignment computes them
        """
        return score_spectrum_pair(
            a.ms1, b.ms1, a.ms2, b.ms2, self.alignment.parameters,
        )
