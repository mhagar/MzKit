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
    ms2_spectrum: Optional[np.ndarray]
    metadata: dict[str, str] = field(default_factory=dict)

    def _has_ms2(self) -> bool:
        return self.ms2_spectrum is not None and self.ms2_spectrum.size > 0

    def to_mgf_text(self) -> str:
        """
        MGF string: one BEGIN IONS block for MS1, plus one for MS2 if
        present.
        """
        blocks = [
            to_mgf(
                pepmass=self.parent_mz,
                charge=self.charge,
                mslevel=1,
                spec_arr=self.ms1_spectrum,
                metadata=self.metadata,
            )
        ]
        if self._has_ms2():
            blocks.append(
                to_mgf(
                    pepmass=self.parent_mz,
                    charge=self.charge,
                    mslevel=2,
                    spec_arr=self.ms2_spectrum,
                    metadata=self.metadata,
                )
            )
        return '\n\n'.join(blocks)

    def to_sirius_text(self) -> str:
        """
        SIRIUS .ms string (MS1 + MS2).
        """
        if self._has_ms2():
            ms2 = self.ms2_spectrum
        else:
            ms2 = np.empty(0, dtype=self.ms1_spectrum.dtype)
        return to_sirius_ms(
            compound=self.metadata.get('NAME', self.base_name),
            parent_mz=self.parent_mz,
            ms1_spec_arr=self.ms1_spectrum,
            ms2_spec_arr=ms2,
        )

    def to_json_obj(self) -> dict:
        """
        Structured dict ready for JSON serialization.
        """
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
            'ms2_spectrum': None,
        }
        if self._has_ms2():
            out['ms2_spectrum'] = {
                'mz': self.ms2_spectrum['mz'].tolist(),
                'intsy': self.ms2_spectrum['intsy'].tolist(),
            }
        return out


def build_ensemble_export(
    ensemble: 'Ensemble',
    *,
    rt: Optional[float] = None,
    normalize: bool = True,
) -> EnsembleExport:
    """
    Gather MS1/MS2 spectra + metadata for a single ensemble.

    :param rt: scan retention time to pull spectra from. If None, the
        ensemble apex (peak_rt) is used.
    :param normalize: normalize spectra to 0-100.
    """
    scan_rt = ensemble.peak_rt if rt is None else float(rt)

    ms1 = ensemble.get_spectrum(ms_level=1, scan_rt=scan_rt)
    if normalize:
        ms1 = _normalize_spectrum(ms1)

    ms2: Optional[np.ndarray] = None
    try:
        _ms2 = ensemble.get_spectrum(ms_level=2, scan_rt=scan_rt)
        if _ms2.size > 0:
            ms2 = _normalize_spectrum(_ms2) if normalize else _ms2
    except (ValueError, IndexError):
        # No MS2 data available for this ensemble/scan.
        pass

    return EnsembleExport(
        base_name=ensemble.identity or ensemble.format_string,
        parent_mz=_resolve_parent_mz(ensemble),
        charge=_resolve_charge(ensemble),
        rt=float(scan_rt),
        ms1_spectrum=ms1,
        ms2_spectrum=ms2,
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
