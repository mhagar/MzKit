"""
Ensemble annotation types and the Ensemble mixin that adds them.

Annotation types are primitive-only, so persistence round-trips them with
plain `asdict` / `Cls(**d)`.
"""
from dataclasses import dataclass, field
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
        cofeature_idx: int,
        ms_level: Literal[1, 2],
        text: str,
        scan_num: Optional[int] = None,
        source: str = 'user',
    ) -> 'GenericAnnotation':
        """
        Create and add a free-form annotation anchored to a peak.

        `source` tags the origin ('user', or e.g. 'auto_adduct'); see
        `remove_generic_annots_by_source`.
        """
        cofeatures = self._get_cofeatures(ms_level)
        if not (0 <= cofeature_idx < len(cofeatures)):
            raise ValueError(
                f"Invalid cofeature_idx: {cofeature_idx}. Ensemble only "
                f"contains {len(cofeatures)} cofeatures"
            )

        annot = GenericAnnotation(
            cofeature_idx=cofeature_idx,
            ms_level=ms_level,
            text=text,
            scan_num=scan_num,
            source=source,
        )

        self.generic_annots[annot.uuid] = annot
        return annot

    def remove_generic_annots_by_source(
        self,
        source: str,
    ) -> None:
        """
        Drop every generic annotation with the given `source` tag, leaving all
        others intact. Used to clear a prior automated pass's labels (e.g.
        'auto_adduct') before re-attaching, so re-runs don't stack duplicates.
        """
        self.generic_annots = {
            uuid_: annot
            for uuid_, annot in self.generic_annots.items()
            if annot.source != source
        }

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
    Free-form user annotation anchored to a single cofeature peak.
    `scan_num`: see `MzDiffAnnotation.scan_num`.
    `source`: who created it -- 'user' for manual annotations, or a tag like
    'auto_adduct' for programmatically-attached labels, so an automated pass can
    clear+replace only its own labels without touching the user's.
    """
    cofeature_idx: int
    ms_level: Literal[1, 2]
    text: str
    uuid: int = field(default_factory=lambda: uuid.uuid4().int)
    scan_num: Optional[int] = None
    source: str = 'user'


@dataclass
class IonAnnotation:
    """
    Claim a group of features are isotopoogues.

    `scan_num`: see `MzDiffAnnotation.scan_num`.
    """
    cofeature_idxs: list[int]
    ms_level: Literal[1, 2]
    formula: str  # core formula (without adduct), str(molmass.Formula)
    error_ppm: float
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
        return (f"{formula_html}<br>"
                f"{self.error_ppm:.1f} ppm")

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
