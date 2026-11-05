"""
Ensemble annotation types and the Ensemble mixin that adds them.

Annotation types are primitive-only, so persistence round-trips them with
plain `asdict` / `Cls(**d)`.
"""
from dataclasses import dataclass, field, replace
import uuid
from typing import Literal, Optional, TYPE_CHECKING

import numpy as np
from find_mfs import get_isotope_envelope
from molmass import Formula

from core.utils.formula_formatting import format_formula_obj_to_html

if TYPE_CHECKING:
    from find_mfs import FormulaCandidate


class EnsembleAnnotationMixin:
    """
    Ensemble's annotation adders. Mixed into `Ensemble` only.
    """

    def add_mz_diff_annot(
        self,
        cofeature_a_idx: int,
        cofeature_b_idx: int,
        ms_level: Literal[1, 2],
        delta_mz: float,
        scan_num: Optional[int] = None,
        label: Optional[str] = None,
        formula: Optional[str] = None,
    ) -> 'MzDiffAnnotation':
        """
        Snapshot of a delta m/z measurement. `delta_mz` is whatever the
        user saw on screen when committing the click — we don't recompute
        from the scan array (the previous behaviour returned 0 / wrong
        values whenever the cofeature lane had no signal at peak_rt,
        which is the common case for sparse DDA MS2 lanes).

        `formula` is an optional neutral-loss formula chosen by the user
        in the formula finder. No validation against the rest of the
        spectrum — it's purely a labelling aid.
        """
        cofeatures = self._get_cofeatures(ms_level)
        for idx in (cofeature_a_idx, cofeature_b_idx):
            if not (0 <= idx <len(cofeatures)):
                raise ValueError(
                    f"Invalid cofeature idx: {idx}, "
                    f"Ensemble only contains {len(cofeatures)} cofeatures "
                )

        annot = MzDiffAnnotation(
            cofeature_a_idx=cofeature_a_idx,
            cofeature_b_idx=cofeature_b_idx,
            ms_level=ms_level,
            delta_mz=delta_mz,
            user_label=label,
            scan_num=scan_num,
            formula=formula,
        )

        self.mz_diffs.append(annot)
        return annot

    def add_ion_annot(
        self,
        cofeature_idxs: list[int],
        ms_level: Literal[1, 2],
        formula: 'FormulaCandidate',
        label: Optional[str],
        scan_num: Optional[int] = None,
    ) -> 'IonAnnotation':
        """
        Create, validate, and add an ion annotation
        """
        # Validate that indices are real
        cofeatures = self._get_cofeatures(ms_level)
        for idx in cofeature_idxs:
            if not (0 <= idx < len(cofeatures)):
                raise ValueError(
                    f"Invalid cofeature_idx: {idx}. Ensemble only contains "
                    f"{len(cofeatures)} cofeatures"
                )

        annot = IonAnnotation(
            cofeature_idxs=cofeature_idxs,
            ms_level=ms_level,
            formula=str(formula.formula),
            error_ppm=formula.error_ppm,
            adduct=formula.adduct,
            user_label=label,
            scan_num=scan_num,
        )

        self.ion_annots[annot.uuid] = annot

        return annot

    def add_generic_annot(
        self,
        cofeature_idxs: list[int],
        ms_level: Literal[1, 2],
        text: str,
        scan_num: Optional[int] = None,
    ) -> 'GenericAnnotation':
        """
        Create and add a free-form annotation on one or more peaks.
        """
        n = len(self._get_cofeatures(ms_level))
        if not cofeature_idxs or not all(0 <= i < n for i in cofeature_idxs):
            raise ValueError(
                f"Invalid cofeature_idxs: {cofeature_idxs}. Ensemble only "
                f"contains {n} cofeatures"
            )

        annot = GenericAnnotation(
            cofeature_idxs=list(cofeature_idxs),
            ms_level=ms_level,
            text=text,
            scan_num=scan_num,
        )

        self.generic_annots[annot.uuid] = annot
        return annot

    def add_ion_pair_annot(
        self,
        ion_a_uuid: int,
        ion_b_uuid: int,
        relationship: Literal[
            "adduct", "neutral_loss", "charge_state"
        ],
        label: Optional[str] = None,
    ) -> 'IonPairAnnotation':
        """
        Create and add an IonPairAnnotation, while validating
        """

        # Validate that UUIDs are real:
        for ion_uuid in (ion_a_uuid, ion_b_uuid):
            if ion_uuid not in self.ion_annots.keys():
                raise ValueError(
                    f"Ion UUID {ion_uuid} not found in this ensemble"
                )

        # Determine formula difference:
        ## TODO: THIS DOESN'T WORK FOR ADDUCTS. MUST FIX BEFORE USE
        formula_diff = str(
            Formula(self.ion_annots[ion_a_uuid].formula)
            - Formula(self.ion_annots[ion_b_uuid].formula)
        )

        annot = IonPairAnnotation(
            ion_a_uuid=ion_a_uuid,
            ion_b_uuid=ion_b_uuid,
            relationship=relationship,
            formula_diff=formula_diff,
            user_label=label,
        )

        self.ion_pair_annots.append(annot)

        return annot


####    Ensemble Annotations    ####

@dataclass
class MzDiffAnnotation:
    """
    A record of m/z difference between two co-features.

    `scan_num` identifies the scan (column index into the ms_level's
    ScanArray) at which the user made the measurement. The viewer shows
    the annotation only when displaying that scan. `None` = scan-agnostic
    (back-compat for .mzk files saved before scan-tied annotations).
    """
    cofeature_a_idx: int
    cofeature_b_idx: int
    ms_level: Literal[1, 2]
    delta_mz: float
    uuid: int = field(default_factory=lambda: uuid.uuid4().int)
    user_label: Optional[str] = None
    scan_num: Optional[int] = None
    formula: Optional[str] = None  # neutral-loss formula, e.g. 'H2O'


@dataclass
class GenericAnnotation:
    """
    Free-form user label on one or more peaks, drawn at the first
    (the tallest, when made). `scan_num`: see `MzDiffAnnotation.scan_num`.
    """
    cofeature_idxs: list[int]
    ms_level: Literal[1, 2]
    text: str
    uuid: int = field(default_factory=lambda: uuid.uuid4().int)
    scan_num: Optional[int] = None


@dataclass
class AdductNetwork:
    """
    Which ion each MS1 isotope envelope of an ensemble is: find-mfs's
    mass-difference network result (`Solution`), or the user's pick in the
    formula finder (a one-envelope network, `by='user'`).

    Upstream of the formula: a FormulaAssignment records the `M` it searched
    with, and is stale once the network's M moves (core/formula/network.py).

    `ions` maps an envelope's monoisotopic MS1 cofeature idx to its ion label
    ("[M+Na]+"); `base_envelope` holds the base envelope's cofeature idxs.
    """
    M: float                       # analyte neutral mass
    base_envelope: list[int]
    ions: dict[int, str]
    # Base-envelope (adduct, charge) readings the network couldn't rule out, winner first
    near_tie_adducts: list[tuple[Optional[str], int]] = field(default_factory=list)
    by: str = 'auto'               # 'auto' (find-mfs) or 'user'

    @property
    def base_ion(self) -> Optional[str]:
        """The base envelope's ion label, e.g. '[M+H]+'."""
        return next(
            (self.ions[i] for i in self.base_envelope if i in self.ions), None
        )


@dataclass
class IonAnnotation:
    """
    Claim a group of features are isotopoogues.

    `scan_num`: see `MzDiffAnnotation.scan_num`.
    """
    cofeature_idxs: list[int]
    ms_level: Literal[1, 2]
    formula: str  # core formula (without adduct), str(molmass.Formula)
    error_ppm: Optional[float]
    adduct: Optional[str] = None  # find-mfs adduct string, e.g. 'Na', '-H'
    uuid: int = field(default_factory=lambda: uuid.uuid4().int)
    user_label: Optional[str] = None
    scan_num: Optional[int] = None

    @property
    def format_string(self) -> str:
        """
        Returns an HTML-formatted string suitable for
        display in MSPlotWidget
        """
        formula_html = format_formula_obj_to_html(Formula(self.formula))
        if self.error_ppm is None:   # e.g. a typed-in formula
            return formula_html
        return f"{formula_html}<br>{self.error_ppm:.1f} ppm"

    @property
    def isotope_envelope(self) -> np.ndarray:
        """
        Assembles the neutral formula + adduct + charge and then
        returns the isotope envelope as a numpy array
        (used for plotting)
        """
        # TODO: Crude/inelegant. Can probably just fix on find-mfs side
        # Assemble formula:
        ion_formula = self.formula
        if self.adduct:
            ion_formula += self.adduct

            charge = Formula(self.formula).charge
            if charge > 0:
                ion_formula += "+"*charge
            if charge < 0:
                ion_formula += "-"*charge

        envelope = get_isotope_envelope(
            formula=Formula(ion_formula),
            mz_tolerance=0.1,
            threshold=0.005,
        )

        return envelope


@dataclass
class IonPairAnnotation:
    """
    Claim about the relationship between two ions
    """
    ion_a_uuid: int
    ion_b_uuid: int
    relationship: Literal[
        "adduct", "neutral_loss", "charge_state"
    ]
    formula_diff: str
    user_label: Optional[str] = None


####    Structure annotation    ####

USER = "user"   # Hypothesis.origin of a hand-typed structure


@dataclass(frozen=True)
class Hypothesis:
    """A proposed structure for an ensemble, from the user or a tool."""
    smiles: str
    name: str = ""
    origin: str = USER   # USER, or the database / tool it came from, e.g. "COCONUT"
    # What's claimed: 'structure' (the exact structure). Room for hazier
    # claims from classifiers / SNAP-MF ('substructure', 'class', 'analogue')
    claim: str = 'structure'
    db_id: str = ""
    scores: dict = field(default_factory=dict)   # against this ensemble's MS2, e.g. {"cosine": 0.61}


@dataclass(frozen=True)
class EnsembleAnnotation:
    """
    What's known about one Ensemble's compound: hypotheses (proposed
    structures), the one chosen as its structure, and its name and notes.

    Lives on the ensemble (`Ensemble.annotation`), so it outlives any
    alignment. Frozen: an edit replaces the whole annotation through
    DataRegistry.update_ensemble (dataclasses.replace), which signals it.
    """
    name: str = ""      # e.g. "Psammaplin A"; "" goes by the chosen hypothesis's
    notes: str = ""
    hypotheses: tuple[Hypothesis, ...] = ()
    chosen_idx: Optional[int] = None

    @property
    def chosen(self) -> Optional[Hypothesis]:
        return None if self.chosen_idx is None else self.hypotheses[self.chosen_idx]

    @property
    def label(self) -> str:
        return self.name or (self.chosen.name if self.chosen else "")

    @property
    def smiles(self) -> str:
        """The chosen structure's SMILES ("" if none)."""
        return self.chosen.smiles if self.chosen else ""

    @property
    def is_empty(self) -> bool:
        return not (self.name or self.notes or self.hypotheses)

    def with_hypothesis(self, h: Hypothesis) -> 'EnsembleAnnotation':
        """`h` added (or found already there, by smiles / origin / db_id) and chosen."""
        hyps = list(self.hypotheses)
        idx = next((i for i, o in enumerate(hyps)
                    if (o.smiles, o.origin, o.db_id) == (h.smiles, h.origin, h.db_id)), None)
        if idx is None:
            hyps.append(h)
            idx = len(hyps) - 1
        return replace(self, hypotheses=tuple(hyps), chosen_idx=idx)

    def with_scores(self, smiles: str, scores: dict) -> 'EnsembleAnnotation':
        """Unscored hypotheses of `smiles` get `scores` (against this ensemble's own MS2)."""
        return replace(self, hypotheses=tuple(
            replace(h, scores=scores) if h.smiles == smiles and not h.scores else h
            for h in self.hypotheses
        ))

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional['EnsembleAnnotation']:
        """Inverse of dataclasses.asdict (persistence); None passes through."""
        if d is None:
            return None
        return cls(**{**d, 'hypotheses': tuple(Hypothesis(**h) for h in d.get('hypotheses', ()))})
