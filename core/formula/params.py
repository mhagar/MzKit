"""
FindMfsParams: the one set of find-mfs parameters every MzKit path uses.

The ion search, the compound (MS2) search, the Ensemble Viewer's auto button,
batch auto-annotation and the CLI all build their find-mfs call from this, so
a setting means the same thing everywhere. It round-trips through the
`[findmfs]` config section; the GUI's FindMfsParamWidget edits it.

Qt-free (used by the CLI and background workers).
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from configparser import ConfigParser

SECTION = 'findmfs'


@dataclass
class FindMfsParams:
    # --- Search space ---
    charge: int = 1
    max_counts: str = 'C*H*N*O*P*S*'    # also defines the element set
    min_counts: str = ''
    detect_halogens: bool = True
    halogen_cap: str = 'Cl4Br4'         # Cl/Br bounds used when detection fires
    min_rdbe: float = -1.0
    max_rdbe: float = 99.0
    check_octet: bool = True

    # --- Mass error ---
    error_ppm: float = 5.0
    error_da: float = 0.01              # search-window floor (matters at low m/z)
    mass_weight: float = 1.0

    # --- Isotope envelope ---
    iso_weight: float = 1.0
    iso_ppm: float = 5.0
    iso_mz_match_da: float = 0.02
    iso_min_rel: float = 0.02

    # --- Chemical prior ---
    chem_weight: float = 1.0
    chem_strength: float = 1.0
    chem_softness: float = 1.0

    # --- MS2 (MistNet) ---
    ms2_weight: float = 1.0
    instrument: str = 'unknown'
    top_n: int = 50                     # ranked candidates kept per assignment

    # -- config -----------------------------------------------------------

    @classmethod
    def from_config(cls, config: Optional['ConfigParser']) -> 'FindMfsParams':
        """Read `[findmfs]`; any missing or unparsable key keeps its default."""
        params = cls()
        if config is None or not config.has_section(SECTION):
            return params

        for f in fields(cls):
            if not config.has_option(SECTION, f.name):
                continue
            getter = {
                bool: config.getboolean,
                int: config.getint,
                float: config.getfloat,
            }.get(type(getattr(params, f.name)), config.get)
            try:
                setattr(params, f.name, getter(SECTION, f.name))
            except ValueError:
                pass
        # Strings may carry stray whitespace from hand-edited configs
        params.max_counts = params.max_counts.strip()
        params.min_counts = params.min_counts.strip()
        params.halogen_cap = params.halogen_cap.strip()
        return params

    def to_config(self, config: 'ConfigParser') -> None:
        """Write every field into `[findmfs]` (does not save to disk)."""
        if not config.has_section(SECTION):
            config.add_section(SECTION)
        for f in fields(self):
            config.set(SECTION, f.name, str(getattr(self, f.name)))

    # -- derived ----------------------------------------------------------

    @property
    def effective_halogen_cap(self) -> Optional[str]:
        """The cap handed to find-mfs; None (detection off) if unchecked or blank."""
        if self.detect_halogens and self.halogen_cap:
            return self.halogen_cap
        return None

    def validate(self) -> None:
        """
        Raise ValueError if find-mfs would reject the count constraints, so a
        typo fails once up front instead of once per ensemble in a batch.
        """
        from find_mfs import resolve_search_bounds

        # Both outcomes of halogen detection must give a searchable space
        # (e.g. 'C0H0' is only rescued by the cap when Cl/Br is detected)
        for halogenated in (False, True):
            resolve_search_bounds(
                self.max_counts or _default_max_counts(),
                self.min_counts or None,
                self.effective_halogen_cap,
                halogenated=halogenated,
            )

    def search_kwargs(self) -> dict:
        """
        Keyword arguments shared by find-mfs `annotate_precursor` and
        `annotate_analyte_dia`: constraints, tolerances and every scoring term.
        Callers add what is path-specific (adducts, spectra, scorer).
        """
        return dict(
            max_counts=self.max_counts or _default_max_counts(),
            min_counts=self.min_counts or None,
            halogen_cap=self.effective_halogen_cap,
            error_ppm=self.error_ppm,
            instrument=self.instrument,
            ms2_weight=self.ms2_weight,
            finder_kwargs=dict(
                error_da=self.error_da,
                filter_rdbe=(self.min_rdbe, self.max_rdbe),
                check_octet=self.check_octet,
            ),
            # FormulaScorer.score terms. The search window (error_ppm) is
            # treated as 3 sigma of the mass error.
            mass_weight=self.mass_weight,
            mass_sigma_ppm=self.error_ppm / 3,
            iso_weight=self.iso_weight,
            iso_ppm=self.iso_ppm,
            iso_mz_match_da=self.iso_mz_match_da,
            iso_min_rel=self.iso_min_rel,
            chem_weight=self.chem_weight,
            chem_strength=self.chem_strength,
            chem_softness=self.chem_softness,
        )


def _default_max_counts() -> str:
    from find_mfs import DEFAULT_MAX_COUNTS
    return DEFAULT_MAX_COUNTS


def counts_to_str(counts: Optional[dict]) -> Optional[str]:
    """
    Bounds dict from find-mfs `query_params` ({'C': inf, 'S': 2, ...}) -> a
    constraint string ("C*S2..."), for provenance that survives JSON.
    """
    if not counts:
        return None
    return ''.join(
        f"{sym}{'*' if n == float('inf') else int(n)}"
        for sym, n in counts.items()
    )
