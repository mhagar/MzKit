"""
Resolution helpers for the AlignmentViewer: turns plot targets (ensemble /
analyte HoverTargets) into the domain data the panels display - member
ensembles, representative spectra, formulae and pairwise scores.

Annotations live on the member Ensembles; an analyte's are display-only
aggregates over them (e.g. "Psammaplin A 5/6"), never stored.

No widgets here; one AlignmentContext exists per displayed alignment.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from core.cli.align_ensembles import PairScore, score_spectrum_pair
from core.data_structs.alignment import RepresentativeSpectrum
from core.utils.formula_formatting import format_assignment_formula_text
from gui.widgets.annotation_target import AnnotationTarget, ensemble_adduct

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
    precursor_mz: Optional[float] = None


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
        return format_assignment_formula_text(ensemble.formula_assignment)

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

    def focus_formula(
        self,
        analyte: 'AlignedAnalyte',
        ensemble: Optional['Ensemble'] = None,
    ) -> Optional[str]:
        """The focused member `ensemble`'s formula, else the analyte's most common one."""
        if ensemble is not None:
            return self.formula(ensemble).rstrip('?') or None
        agreement = self.formula_agreement(analyte)
        return agreement[0] if agreement else None

    def target_ensemble(
        self,
        analyte: 'AlignedAnalyte',
        ensemble: Optional['Ensemble'] = None,
    ) -> Optional['Ensemble']:
        """
        The ensemble a focus's annotation edits go to: `ensemble` (a focused
        member), else the analyte's representative member.
        """
        if ensemble is not None:
            return ensemble
        rep = self.representative(analyte)
        return None if rep is None else self.members(analyte).get(rep.sample_uuid)

    def annotation_target(
        self,
        analyte: 'AlignedAnalyte',
        ensemble: Optional['Ensemble'] = None,
    ) -> AnnotationTarget:
        """
        The annotation tabs' target for a focus: `ensemble` (a focused member)
        with its own MS2, precursor and formula, else the analyte, searched
        with its representative's MS2 at the consensus m/z and its most common
        formula, edits going to the representative member. The adduct goes
        with the formula: the focused (else representative) member's, else
        any member's.
        """
        members = self.members(analyte)
        target = self.target_ensemble(analyte, ensemble)
        if ensemble is not None:
            composite = ensemble.composite_spectrum
            ms2, precursor_mz = composite.ms2, composite.precursor_mz
        else:
            rep = self.representative(analyte)
            ms2, precursor_mz = (None if rep is None else rep.composite.ms2), analyte.consensus_mz
        firsts = [target] if target is not None else []
        adduct = next(filter(None, map(ensemble_adduct, firsts + list(members.values()))), None)

        header = tip = ""
        if target is not None:
            sample = next((s for s, e in members.items() if e.uuid == target.uuid), None)
            header = self.sample_name(sample) if sample is not None else "This ensemble"
            if ensemble is None:
                header += " (representative member)"
            if len(members) > 1 and (agreement := self.structure_agreement(analyte)):
                header += f" · members: {agreement[0]} {agreement[1]}/{agreement[2]}"
            tip = "\n".join(self.notes(analyte))
        else:
            header = "No member ensemble to annotate"

        return AnnotationTarget(
            key=(analyte.uuid, ensemble and ensemble.uuid),
            ensemble=target,
            ms2=ms2,
            precursor_mz=precursor_mz,
            formula=self.focus_formula(analyte, ensemble),
            adduct=adduct,
            noun="ensemble" if ensemble is not None else "analyte",
            header=header,
            header_tip=tip,
            members=list(members.values()),
            embedding_id=str(analyte.uuid) if ensemble is None else None,
        )

    def structure_agreement(
        self,
        analyte: 'AlignedAnalyte',
    ) -> Optional[tuple[str, int, int]]:
        """
        Most common member label (name, else the chosen structure's name, else
        its SMILES) as (label, count, n_members); None if no member has one.
        """
        members = self.members(analyte)
        labels = [
            a.label or a.smiles
            for a in (e.annotation for e in members.values()) if a is not None
        ]
        labels = [label for label in labels if label]
        if not labels:
            return None
        label, count = Counter(labels).most_common(1)[0]
        return label, count, len(members)

    def notes(self, analyte: 'AlignedAnalyte') -> list[str]:
        """Every member's notes, deduplicated, in member order."""
        notes = (e.annotation.notes for e in self.members(analyte).values() if e.annotation)
        return list(dict.fromkeys(n for n in notes if n))

    def analyte_label(self, analyte: 'AlignedAnalyte') -> str:
        """"m/z @ RT", followed by its members' most common name, if any."""
        label = f"{analyte.consensus_mz:.4f} @ {analyte.consensus_rt:.1f}s"
        agreement = self.structure_agreement(analyte)
        return f"{label} ({agreement[0]})" if agreement else label

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
                precursor_mz=composite.precursor_mz,
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
                precursor_mz=rep.composite.precursor_mz,
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
            a.precursor_mz, b.precursor_mz,
        )
