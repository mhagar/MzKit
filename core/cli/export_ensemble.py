"""
Render a single Ensemble's spectra + metadata to export formats
(MGF, JSON, .ms) for ingestion by external tools (SIRIUS, GNPS, …).

Spectrum production is done by the Ensemble itself
(`Ensemble.get_ms2_spectra`). This module only
maps to export metadata tags / serializes format
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, TYPE_CHECKING

import numpy as np

from core.data_structs.ensemble import MS2Spectrum
from core.utils.spectra import normalize_spectrum, threshold_consensus
from core.utils.spectrum_export import to_mgf, to_sirius_ms

if TYPE_CHECKING:
    from core.data_structs import Ensemble
    from core.data_structs.ensemble import MS2Mode

logger = logging.getLogger(__name__)

# Output formats understood by write_ensemble_export().
VALID_FORMATS = ('mgf', 'ms', 'json')

# user_metadata keys with special export meaning (matched case-insensitively).
# These drive typed export fields rather than being emitted as generic tags;
# every other user_metadata entry passes through as an MGF tag.
_RESERVED_META_KEYS = {'charge', 'adduct', 'ionization', 'feature_id'}


def _export_metadata(ensemble: 'Ensemble') -> dict[str, str]:
    """
    Build the MGF metadata tag block from the Ensemble's typed fields and
    user_metadata (reserved keys handled specially; the rest pass through
    as uppercased generic tags).
    """
    md: dict[str, str] = {}

    if ensemble.identity:
        md['NAME'] = ensemble.identity
    if ensemble.proposed_formula:
        md['FORMULA'] = ensemble.proposed_formula

    adduct = ensemble.get_meta('adduct') or ensemble.get_meta('ionization')
    if adduct:
        md['ADDUCT'] = adduct

    feature_id = ensemble.get_meta('feature_id')
    if feature_id:
        md['FEATURE_ID'] = feature_id

    for key, value in ensemble.user_metadata.items():
        if key.lower() in _RESERVED_META_KEYS:
            continue
        md[key.upper()] = str(value)

    return md


@dataclass
class EnsembleExport:
    """
    In-memory export artifacts for a single ensemble, rendered to specific
    formats via the to_*() methods.

    `ms2_spectra` holds the produced MS2 spectra
    (see Ensemble.get_ms2_spectra):
    - none for MS1-only ensembles
    - one for a DIA consensus / tallest scan
    - or several for DDA (per matched scan, or per precursor when merged)
    """
    base_name: str
    parent_mz: float
    charge: int
    rt: float
    ms1_spectrum: np.ndarray
    ms2_spectra: list[MS2Spectrum] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def _nonempty_ms2(self) -> list[MS2Spectrum]:
        return [s for s in self.ms2_spectra if s.spectrum.size > 0]

    def _precursor_blocks(
        self,
    ) -> Iterator[tuple[float, int, Optional[MS2Spectrum]]]:
        """
        Yield one `(precursor_mz, charge, ms2_or_None)` per export block.

        Both MGF and SIRIUS group an MS1 spectrum with each MS2
        by a shared precursor, so this emits one block per nonempty
         MS2 (each carrying its own precursor),
         falling back to a single MS1-only block when the ensemble
          has no MS2.
        """
        nonempty = self._nonempty_ms2()
        if not nonempty:
            yield self.parent_mz, self.charge, None
            return
        for ms2 in nonempty:
            yield ms2.precursor_mz, ms2.charge, ms2

    def to_mgf_text(self) -> str:
        """
        MGF string.

        For each precursor block this emits an MS1 block followed
        by the MS2 block, both stamped with that block's precursor m/z.

        SIRIUS groups MS1/MS2 by a shared PEPMASS,
        so the MS1 is repeated (once per precursor) to keep each
         MS1/MS2 pair associated.

        MS1-only ensembles fall back to a single MS1 block.
        """
        blocks: list[str] = []
        for pepmass, charge, ms2 in self._precursor_blocks():
            blocks.append(
                to_mgf(
                    pepmass=pepmass,
                    charge=charge,
                    mslevel=1,
                    spec_arr=self.ms1_spectrum,
                    metadata=self.metadata,
                )
            )
            if ms2 is not None:
                blocks.append(
                    to_mgf(
                        pepmass=pepmass,
                        charge=charge,
                        mslevel=2,
                        spec_arr=ms2.spectrum,
                        metadata=self.metadata,
                    )
                )
        return '\n\n'.join(blocks)

    def to_sirius_text(self) -> str:
        """
        SIRIUS .ms string.
        Mirrors the MGF layout: one compound block per
        precursor, each carrying a copy of the MS1 spectrum and the
        matching precursor as >parentmass.

        MS1-only ensembles fall back to a single compound block.
        """
        compound = self.metadata.get('NAME', self.base_name)
        empty_ms2 = np.empty(0, dtype=self.ms1_spectrum.dtype)

        blocks = [
            to_sirius_ms(
                compound=compound,
                parent_mz=pepmass,
                ms1_spec_arr=self.ms1_spectrum,
                ms2_spec_arr=ms2.spectrum if ms2 is not None else empty_ms2,
            )
            for pepmass, _charge, ms2 in self._precursor_blocks()
        ]
        return '\n\n'.join(blocks)

    def to_json_obj(self) -> dict:
        """
        Structured dict ready for JSON serialization.

        `ms2_spectra` is the full list; `ms2_spectrum` is kept for
        backwards compatibility and holds the first MS2 spectrum (or None).
        """
        ms2_list = [
            {
                'mz': s.spectrum['mz'].tolist(),
                'intsy': s.spectrum['intsy'].tolist(),
                'precursor_mz': s.precursor_mz,
                'charge': s.charge,
                'rt': s.rt,
            }
            for s in self._nonempty_ms2()
        ]
        return {
            'name': self.base_name,
            'parent_mz': self.parent_mz,
            'charge': self.charge,
            'rt': self.rt,
            'metadata': self.metadata,
            'ms1_spectrum': {
                'mz': self.ms1_spectrum['mz'].tolist(),
                'intsy': self.ms1_spectrum['intsy'].tolist(),
            },
            'ms2_spectrum': ms2_list[0] if ms2_list else None,
            'ms2_spectra': ms2_list,
        }


def build_ensemble_export(
    ensemble: 'Ensemble',
    *,
    rt: Optional[float] = None,
    ms2_mode: 'MS2Mode' = 'consensus',
    freq_threshold: float = 0.25,
    normalize: bool = True,
) -> EnsembleExport:
    """
    Gather MS1 + MS2 spectra + metadata for a single ensemble.

    MS2 selection is governed by `ms2_mode` (delegated to
    Ensemble.get_ms2_spectra); `rt` only governs the MS1 spectrum.

    Export is the "print" boundary of the BIN strategy: consensus MS2
    spectra (which retain per-bin frequency) are thresholded here, keeping
    only bins present in at least `freq_threshold` of the merged scans.

    :param rt: scan retention time to pull the MS1 spectrum from. If None,
        the ensemble apex (peak_rt) is used.
    :param ms2_mode: 'tallest' | 'all' | 'consensus'.
    :param freq_threshold: min relative frequency to keep a consensus bin
        (Bittremieux et al. use 0.25). Ignored for non-consensus spectra.
    :param normalize: normalize each spectrum to 0-100.
    """
    scan_rt = ensemble.peak_rt if rt is None else float(rt)

    ms1 = ensemble.get_spectrum(ms_level=1, scan_rt=scan_rt)
    if normalize:
        ms1 = normalize_spectrum(ms1, max_range=100.0)

    # Flatten each MS2 spectrum to plain (mz, intsy), thresholding consensus
    # spectra by frequency first (a no-op for single-scan tallest/all).
    ms2_spectra: list[MS2Spectrum] = []
    for s in ensemble.get_ms2_spectra(mode=ms2_mode):
        spectrum = threshold_consensus(s.spectrum, min_freq=freq_threshold)
        if normalize:
            spectrum = normalize_spectrum(spectrum, max_range=100.0)
        ms2_spectra.append(
            MS2Spectrum(
                spectrum=spectrum,
                precursor_mz=s.precursor_mz,
                charge=s.charge,
                rt=s.rt,
            )
        )

    return EnsembleExport(
        base_name=ensemble.identity or ensemble.format_string,
        parent_mz=ensemble.resolved_precursor_mz,
        charge=ensemble.resolved_charge,
        rt=float(scan_rt),
        ms1_spectrum=ms1,
        ms2_spectra=ms2_spectra,
        metadata=_export_metadata(ensemble),
    )


def safe_filename(name: str) -> str:
    """
    Turn an arbitrary label into a filesystem-safe file stem.
    """
    keep = [
        c if (c.isalnum() or c in ('_', '-', '.')) else '_'
        for c in name
    ]
    return ''.join(keep).strip('_') or 'compound'


def write_ensemble_export(
    export: EnsembleExport,
    output_dir: Path,
    formats: Iterable[str] = ('mgf',),
) -> list[Path]:
    """
    Write the requested formats for one ensemble export into output_dir.

    :return: the list of written file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = {f.lower() for f in formats}
    unknown = requested - set(VALID_FORMATS)
    if unknown:
        raise ValueError(
            f"Unknown export format(s): {sorted(unknown)}. "
            f"Valid formats: {sorted(VALID_FORMATS)}"
        )

    stem = safe_filename(export.base_name)
    renderers = {
        'mgf': ('mgf', export.to_mgf_text),
        'ms': ('ms', export.to_sirius_text),
        'json': ('json', lambda: json.dumps(export.to_json_obj(), indent=2)),
    }

    written: list[Path] = []
    for fmt in VALID_FORMATS:  # deterministic order
        if fmt not in requested:
            continue
        suffix, render = renderers[fmt]
        path = output_dir / f"{stem}.{suffix}"
        path.write_text(render())
        written.append(path)
        logger.info("Wrote %s", path)

    return written
