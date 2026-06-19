"""
Adduct / neutral-loss / charge-state mass arithmetic
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from molmass import Formula
from molmass.elements import ELECTRON

ELECTRON_MASS = float(ELECTRON.mass)

ISOTOPE_SPACING = (
    Formula("[13C]").monoisotopic_mass - Formula("C").monoisotopic_mass
)  # ~= 1.0033548


@lru_cache(maxsize=None)
def _formula_mass(formula: str) -> float:
    """
    Monoisotopic mass of a neutral formula string ("" -> 0.0)
    """
    if not formula:
        return 0.0
    return float(Formula(formula).monoisotopic_mass)


@dataclass(frozen=True)
class Adduct:
    """
    One ion species relative to a neutral molecule M

    add/remove are neutral formula strings of atoms gained/lost (e.g. add="Na"
    for [M+Na]+, remove="H2O" for [M+H-H2O]+). nummol is the count of M units
    (2 -> dimer). z is the signed charge.
    """
    name: str
    add: str
    remove: str
    nummol: int
    z: int

    @property
    def mass_shift(self) -> float:
        """
        add_mass - remove_mass - z*m_e: per-ion shift before /|z|
        """
        return (
            _formula_mass(self.add)
            - _formula_mass(self.remove)
            - self.z * ELECTRON_MASS
        )

    def mz(self, neutral_mass: float) -> float:
        """
        Predicted observed m/z for a neutral mass M under this adduct
        """
        return (self.nummol * neutral_mass + self.mass_shift) / abs(self.z)

    def neutral_mass(self, mz: float) -> float:
        """
        Back out the neutral mass M implied by an observed m/z
        """
        return (mz * abs(self.z) - self.mass_shift) / self.nummol


# Adduct tables
_POSITIVE = [
    Adduct("[M+H]+",      add="H",      remove="",    nummol=1, z=1),
    Adduct("[M+Na]+",     add="Na",     remove="",    nummol=1, z=1),
    # Adduct("[M+K]+",      add="K",      remove="",    nummol=1, z=1),
    Adduct("[M+NH4]+",    add="NH4",    remove="",    nummol=1, z=1),
    Adduct("[M+H-H2O]+",  add="H",      remove="H2O", nummol=1, z=1),
    Adduct("[M+H-NH3]+",  add="H",      remove="NH3", nummol=1, z=1),
    Adduct("[M+2H]2+",    add="H2",     remove="",    nummol=1, z=2),
    Adduct("[2M+H]+",     add="H",      remove="",    nummol=2, z=1),
    Adduct("[2M+Na]+",    add="Na",     remove="",    nummol=2, z=1),
]
_NEGATIVE = [
    Adduct("[M-H]-",      add="",       remove="H",   nummol=1, z=-1),
    Adduct("[M+Cl]-",     add="Cl",     remove="",    nummol=1, z=-1),
    # Adduct("[M+HCOO]-",   add="HCOO",   remove="",    nummol=1, z=-1),  # formate
    # Adduct("[M+CH3COO]-", add="C2H3O2", remove="",    nummol=1, z=-1),  # acetate
    Adduct("[M-H-H2O]-",  add="",       remove="H3O", nummol=1, z=-1),
    Adduct("[M-2H]2-",    add="",       remove="H2",  nummol=1, z=-2),
    Adduct("[2M-H]-",     add="",       remove="H",   nummol=2, z=-1),
]

# Common in-source neutral losses (direct m/z delta), for labeling ISF edges.
_NEUTRAL_LOSSES: dict[str, float] = {
    # name: _formula_mass(f)
    # for name, f in {
    #     "-H2O": "H2O",
    #     "-2H2O": "H4O2",
    #     "-NH3": "NH3",
        # "-CO": "CO",
        # "-CO2": "CO2",
        # "-HCOOH": "CH2O2",     # formic acid
        # "-CH2O": "CH2O",       # formaldehyde
        # "-C2H4O2": "C2H4O2",   # acetic acid
        # "-hexose": "C6H10O5",  # glycoside loss (~162)
        # "-pentose": "C5H8O4",  # ~132
    # }.items()
}


def adducts_for_polarity(polarity: Optional[int]) -> list[Adduct]:
    """
    polarity: 1 positive, 0 negative, None/other unknown -> both signs
    """
    if polarity == 1:
        return list(_POSITIVE)
    if polarity == 0:
        return list(_NEGATIVE)
    return list(_POSITIVE) + list(_NEGATIVE)


def neutral_losses() -> dict[str, float]:
    return dict(_NEUTRAL_LOSSES)


def _intra_pair_deltas(
        adducts: list[Adduct]
) -> list[tuple[float, str]]:
    """
    Pairwise m/z deltas between same-charge (|z| =1), single-molecule
     adducts of one polarity.

    Cross-polarity pairs are never formed.
    """
    singles = [a for a in adducts if a.nummol == 1 and abs(a.z) == 1]
    out: list[tuple[float, str]] = []
    for i, a in enumerate(singles):
        for b in singles[i + 1:]:
            d = abs(a.mass_shift - b.mass_shift)
            if d > 1e-6:
                out.append((d, f"{a.name}/{b.name}"))
    return out


@lru_cache(maxsize=None)
def _pair_deltas(polarity: Optional[int]) -> tuple[tuple[float, str], ...]:
    """
    Cached adduct-pair deltas for a polarity (intra-polarity only)
    """
    if polarity == 1:
        pairs = _intra_pair_deltas(_POSITIVE)
    elif polarity == 0:
        pairs = _intra_pair_deltas(_NEGATIVE)
    else:
        pairs = _intra_pair_deltas(_POSITIVE) + _intra_pair_deltas(_NEGATIVE)
    return tuple(pairs)


def relationship_label(
    seed_mz: float,
    cand_mz: float,
    ppm_tol: float,
    polarity: Optional[int] = None,
) -> Optional[str]:
    """
    If `|cand_mz - seed_mz|` matches a known same-entity relationship
    (within ppm_tol of seed m/z), return a human-readable label; else
    None.
    """
    delta = abs(cand_mz - seed_mz)
    tol = ppm_tol * 1e-6 * seed_mz

    # Isotopologues.
    for k in (1, 2):
        if abs(delta - k * ISOTOPE_SPACING) <= tol:
            return f"isotope +{k}"

    # In-source neutral losses (charge-1 m/z delta).
    for name, mass in _NEUTRAL_LOSSES.items():
        if abs(delta - mass) <= tol:
            return name

    # Same-charge adduct pairs (constant m/z deltas; no neutral mass posited).
    for d, label in _pair_deltas(polarity):
        if abs(delta - d) <= tol:
            return label

    return None


if __name__ == "__main__":
    # Self-check against textbook values (glucose, C6H12O6).
    M = _formula_mass("C6H12O6")
    print(f"electron mass    = {ELECTRON_MASS:.9f}")
    print(f"isotope spacing  = {ISOTOPE_SPACING:.7f}  (expect ~1.0033548)")
    print(f"glucose neutral  = {M:.5f}  (expect ~180.06339)")

    mh = next(a for a in _POSITIVE if a.name == "[M+H]+").mz(M)
    mna = next(a for a in _POSITIVE if a.name == "[M+Na]+").mz(M)
    print(f"[M+H]+={mh:.5f}  [M+Na]+={mna:.5f}")
    print(f"Na/H pair label  = {relationship_label(mh, mna, 20.0, polarity=1)}")
    print(f"H2O-loss label   = {relationship_label(mh, mh - _formula_mass('H2O'), 20.0, 1)}")
    print(f"unrelated label  = {relationship_label(mh, mh + 5.0, 20.0, 1)}")
