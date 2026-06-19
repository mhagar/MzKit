"""
Export a single Ensemble's spectra + metadata for ingestion by external
tools (SIRIUS, GNPS, etc.).

Qt-free single source of truth: used by both the GUI (EnsembleViewer)
and the CLI (export_compound.py). All exportable metadata is read off
the Ensemble itself — its typed fields (`identity`, `proposed_formula`)
and its `user_metadata` dict — so there is no separate data-entry step.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, TYPE_CHECKING

import numpy as np

from core.utils.spectrum_export import to_mgf, to_sirius_ms

if TYPE_CHECKING:
    from core.data_structs import Ensemble

logger = logging.getLogger(__name__)

# Output formats understood by write_ensemble_export().
VALID_FORMATS = ('mgf', 'ms', 'json')

# user_metadata keys with special export meaning (matched case-insensitively).
# These drive typed export fields rather than being emitted as generic tags;
# every other user_metadata entry passes through as an MGF tag.
_RESERVED_META_KEYS = {'charge', 'adduct', 'ionization', 'feature_id'}


def _normalize_spectrum(spec: np.ndarray) -> np.ndarray:
    """
    Normalize spectrum intensities to a 0-100 range.
    """
    if spec.size == 0:
        return spec

    max_intsy = spec['intsy'].max()
    if max_intsy <= 0:
        return spec

    out = spec.copy()
    out['intsy'] = out['intsy'] / max_intsy * 100.0
    return out


def _get_meta(ensemble: 'Ensemble', key: str) -> Optional[str]:
    """
    Case-insensitive lookup into the ensemble's user_metadata.
    """
    for k, v in ensemble.user_metadata.items():
        if k.lower() == key:
            return v
    return None


def _resolve_charge(ensemble: 'Ensemble') -> int:
    """
    Charge precedence: explicit `charge` in user_metadata, then the DDA
    precursor charge, else 1.
    """
    raw = _get_meta(ensemble, 'charge')
    if raw:
        try:
            return int(str(raw).strip().rstrip('+-') or '1')
        except ValueError:
            logger.warning("Could not parse charge %r; defaulting to 1", raw)

    if ensemble.precursor_charge:
        return int(ensemble.precursor_charge)

    return 1


def _resolve_parent_mz(ensemble: 'Ensemble') -> float:
    """
    Precursor m/z: the DDA precursor if known, otherwise the ensemble's
    base (most-intense MS1) m/z.
    """
    if ensemble.precursor_mz is not None:
        return float(ensemble.precursor_mz)
    return float(ensemble.base_mz)


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

    adduct = _get_meta(ensemble, 'adduct') or _get_meta(ensemble, 'ionization')
    if adduct:
        md['ADDUCT'] = adduct

    feature_id = _get_meta(ensemble, 'feature_id')
    if feature_id:
        md['FEATURE_ID'] = feature_id

    for key, value in ensemble.user_metadata.items():
        if key.lower() in _RESERVED_META_KEYS:
            continue
        md[key.upper()] = str(value)

    return md


@dataclass
class MS2SpectrumExport:
    """
    A single MS2 spectrum with its own precursor metadata.

    For DIA (or MS1-only) ensembles there is at most one of these, pulled
    at the selected/apex RT. For DDA ensembles there is one per matched
    MS2 scan, and `precursor_mz`/`charge` carry whatever the instrument
    actually designated as the precursor for that scan.
    """
    spectrum: np.ndarray
    precursor_mz: float
    charge: int
    rt: float


@dataclass
class EnsembleExport:
    """
    In-memory export artifacts for a single ensemble, format-agnostic.
    Render to a specific format with the to_*() methods.
    """
    base_name: str
    parent_mz: float
    charge: int
    rt: float
    ms1_spectrum: np.ndarray
    ms2_spectra: list[MS2SpectrumExport] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def _nonempty_ms2(self) -> list[MS2SpectrumExport]:
        return [s for s in self.ms2_spectra if s.spectrum.size > 0]

    def _has_ms2(self) -> bool:
        return len(self._nonempty_ms2()) > 0

    def to_mgf_text(self) -> str:
        """
        MGF string. For each MS2 spectrum we emit an MS1 block followed by
        the MS2 block, both stamped with that spectrum's precursor m/z —
        SIRIUS groups MS1/MS2 by a shared PEPMASS, so the MS1 is repeated
        (once per precursor) to keep each MS1/MS2 pair associated.

        MS1-only ensembles fall back to a single MS1 block.
        """
        nonempty = self._nonempty_ms2()
        if not nonempty:
            return to_mgf(
                pepmass=self.parent_mz,
                charge=self.charge,
                mslevel=1,
                spec_arr=self.ms1_spectrum,
                metadata=self.metadata,
            )

        blocks: list[str] = []
        for ms2 in nonempty:
            blocks.append(
                to_mgf(
                    pepmass=ms2.precursor_mz,
                    charge=ms2.charge,
                    mslevel=1,
                    spec_arr=self.ms1_spectrum,
                    metadata=self.metadata,
                )
            )
            blocks.append(
                to_mgf(
                    pepmass=ms2.precursor_mz,
                    charge=ms2.charge,
                    mslevel=2,
                    spec_arr=ms2.spectrum,
                    metadata=self.metadata,
                )
            )
        return '\n\n'.join(blocks)

    def to_sirius_text(self) -> str:
        """
        SIRIUS .ms string. Mirrors the MGF layout: one compound block per
        MS2 spectrum, each carrying a copy of the MS1 spectrum and the
        matching precursor as >parentmass.

        MS1-only ensembles fall back to a single compound block.
        """
        compound = self.metadata.get('NAME', self.base_name)
        nonempty = self._nonempty_ms2()
        if not nonempty:
            return to_sirius_ms(
                compound=compound,
                parent_mz=self.parent_mz,
                ms1_spec_arr=self.ms1_spectrum,
                ms2_spec_arr=np.empty(0, dtype=self.ms1_spectrum.dtype),
            )

        blocks = [
            to_sirius_ms(
                compound=compound,
                parent_mz=ms2.precursor_mz,
                ms1_spec_arr=self.ms1_spectrum,
                ms2_spec_arr=ms2.spectrum,
            )
            for ms2 in nonempty
        ]
        return '\n\n'.join(blocks)

    def to_json_obj(self) -> dict:
        """
        Structured dict ready for JSON serialization.

        `ms2_spectra` is the full list (one entry per MS2 scan for DDA);
        `ms2_spectrum` is kept for backwards compatibility and holds the
        first MS2 spectrum (or None).
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
        out: dict = {
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
        return out


def _is_dda(ensemble: 'Ensemble') -> bool:
    """
    True if this ensemble's MS2 comes from DDA acquisition (each MS2 scan
    is a discrete fragmentation of a designated precursor).
    """
    return bool(
        ensemble.injection is not None
        and getattr(ensemble.injection, 'acquisition_mode', None) == 'dda'
    )


def _build_dia_ms2_spectra(
    ensemble: 'Ensemble',
    scan_rt: float,
    normalize: bool,
) -> list[MS2SpectrumExport]:
    """
    DIA / MS1-only: a single reconstructed MS2 spectrum at `scan_rt`, with
    the ensemble's resolved precursor m/z and charge.
    """
    try:
        ms2 = ensemble.get_spectrum(ms_level=2, scan_rt=scan_rt)
    except (ValueError, IndexError):
        # No MS2 data available for this ensemble/scan.
        return []

    if ms2.size == 0:
        return []

    if normalize:
        ms2 = _normalize_spectrum(ms2)

    return [
        MS2SpectrumExport(
            spectrum=ms2,
            precursor_mz=_resolve_parent_mz(ensemble),
            charge=_resolve_charge(ensemble),
            rt=float(scan_rt),
        )
    ]


def _build_dda_ms2_spectra(
    ensemble: 'Ensemble',
    normalize: bool,
) -> list[MS2SpectrumExport]:
    """
    DDA: dump every MS2 scan matched to this ensemble. Each matched scan
    is one precursor's fragmentation, so we emit the full scan spectrum
    (zero-intensity lanes dropped) tagged with the precursor m/z and
    charge the instrument designated for that scan.
    """
    ms2_arr = ensemble.injection.scan_array_ms2
    if ms2_arr is None or not ensemble.ms2_cofeatures:
        return []

    # All MS2 cofeatures share the same matched scan_idxs by construction.
    scan_idxs = np.unique(np.asarray(ensemble.ms2_cofeatures[0].scan_idxs))
    if scan_idxs.size == 0:
        return []

    default_charge = _resolve_charge(ensemble)

    out: list[MS2SpectrumExport] = []
    for raw_idx in scan_idxs:
        scan_idx = int(raw_idx)
        spec = ms2_arr.get_spectrum(scan_idx)
        spec = spec[spec['intsy'] > 0]
        if spec.size == 0:
            continue
        if normalize:
            spec = _normalize_spectrum(spec)

        if ms2_arr.precursor_mz_arr is not None:
            precursor_mz = float(ms2_arr.precursor_mz_arr[scan_idx])
        else:
            precursor_mz = _resolve_parent_mz(ensemble)

        charge = default_charge
        if ms2_arr.precursor_charge_arr is not None:
            scan_charge = int(ms2_arr.precursor_charge_arr[scan_idx])
            if scan_charge:
                charge = abs(scan_charge)

        rt = (
            float(ms2_arr.rt_arr[scan_idx])
            if ms2_arr.rt_arr is not None
            else 0.0
        )

        out.append(
            MS2SpectrumExport(
                spectrum=spec,
                precursor_mz=precursor_mz,
                charge=charge,
                rt=rt,
            )
        )

    return out


def build_ensemble_export(
    ensemble: 'Ensemble',
    *,
    rt: Optional[float] = None,
    normalize: bool = True,
) -> EnsembleExport:
    """
    Gather MS1/MS2 spectra + metadata for a single ensemble.

    For DIA / MS1-only ensembles a single MS2 spectrum is pulled at `rt`.
    For DDA ensembles, `rt` only governs the MS1 spectrum — every matched
    MS2 scan is dumped, each tagged with its own designated precursor.

    :param rt: scan retention time to pull the MS1 (and, for DIA, MS2)
        spectrum from. If None, the ensemble apex (peak_rt) is used.
    :param normalize: normalize spectra to 0-100.
    """
    scan_rt = ensemble.peak_rt if rt is None else float(rt)

    ms1 = ensemble.get_spectrum(ms_level=1, scan_rt=scan_rt)
    if normalize:
        ms1 = _normalize_spectrum(ms1)

    if _is_dda(ensemble):
        ms2_spectra = _build_dda_ms2_spectra(ensemble, normalize)
    else:
        ms2_spectra = _build_dia_ms2_spectra(ensemble, scan_rt, normalize)

    return EnsembleExport(
        base_name=ensemble.identity or ensemble.format_string,
        parent_mz=_resolve_parent_mz(ensemble),
        charge=_resolve_charge(ensemble),
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
