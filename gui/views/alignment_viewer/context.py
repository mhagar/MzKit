"""
Resolution helpers for the AlignmentViewer: turns plot targets (ensemble /
analyte HoverTargets) into the domain data the panels display - member
ensembles, consensus spectra, formulae and pairwise scores.

No widgets here; one AlignmentContext exists per displayed alignment.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from core.cli.align_ensembles import PairScore, score_spectrum_pair
from core.data_structs.alignment import ConsensusSpectrum
from gui.widgets.alignment_plot.layout import resolve_ensemble

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


class AlignmentContext:
    def __init__(
        self,
        alignment: 'EnsembleAlignment',
        data_source: 'AlignmentViewerDataSource',
        saturation_threshold: float,
    ):
        self.alignment = alignment
        self.data_source = data_source
        self.saturation_threshold = saturation_threshold

    # -- lookups ------------------------------------------------------------

    def sample_name(self, sample_uuid: 'SampleUUID') -> str:
        sample = self.data_source.get_sample(sample_uuid)
        return sample.name if sample else f"...{str(sample_uuid)[-5:]}"

    def members(
        self,
        analyte: 'AlignedAnalyte',
    ) -> dict['SampleUUID', 'Ensemble']:
        """The analyte's resolvable member ensembles, keyed by sample uuid."""
        out = {}
        for sample_uuid in analyte.ensemble_map:
            ensemble = resolve_ensemble(analyte, sample_uuid, self.data_source)
            if ensemble is not None:
                out[sample_uuid] = ensemble
        return out

    def consensus(
        self,
        analyte: 'AlignedAnalyte',
    ) -> Optional[ConsensusSpectrum]:
        return analyte.consensus_spectrum(
            self.members(analyte), self.saturation_threshold,
        )

    def formula(self, ensemble: 'Ensemble') -> str:
        """
        Plain-text formula for an ensemble: the accepted assignment
        candidate, else the free-text proposed formula, else the top
        (unaccepted) candidate marked with '?'. Empty if none.
        """
        assignment = self.data_source.get_assignment_for_source(ensemble.uuid)
        chosen = assignment.chosen if assignment is not None else None
        if chosen is not None:
            return chosen.formula_str
        if ensemble.proposed_formula:
            return str(ensemble.proposed_formula)
        top = assignment.top if assignment is not None else None
        if top is not None:
            return f"{top.formula_str}?"
        return ""

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

    # -- comparison ---------------------------------------------------------

    def item_spectra(self, target: 'HoverTarget') -> Optional[ItemSpectra]:
        if target.kind == 'ensemble' and target.ensemble is not None:
            ensemble = target.ensemble
            composite = ensemble.composite_spectrum
            return ItemSpectra(
                label=self.sample_name(target.sample_uuid),
                ms1=composite.ms1,
                ms2=composite.ms2,
                rt=float(ensemble.peak_rt),
                base_mz=float(ensemble.base_mz),
            )

        if target.kind == 'analyte' and target.analyte is not None:
            analyte = target.analyte
            consensus = self.consensus(analyte)
            if consensus is None:
                return None
            return ItemSpectra(
                label=(
                    f"Analyte {analyte.consensus_mz:.4f} @ "
                    f"{analyte.consensus_rt:.1f}s "
                    f"({len(analyte.ensemble_map)}/{self.alignment.sample_count})"
                ),
                ms1=consensus.composite.ms1,
                ms2=consensus.composite.ms2,
                rt=float(analyte.consensus_rt),
                base_mz=consensus.base_mz,
            )

        return None

    def score(self, a: ItemSpectra, b: ItemSpectra) -> PairScore:
        """Cosine scores between two items, as alignment computes them."""
        return score_spectrum_pair(
            a.ms1, b.ms1, a.ms2, b.ms2, self.alignment.parameters,
        )
