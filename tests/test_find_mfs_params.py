"""
Tests for core/formula/params.py: FindMfsParams, the one parameter set shared
by every find-mfs path (ion search, compound search, auto/batch, CLI).
"""
from configparser import ConfigParser

import pytest

from core.formula.params import FindMfsParams, counts_to_str


def _config(**findmfs) -> ConfigParser:
    cfg = ConfigParser()
    cfg['findmfs'] = {k: str(v) for k, v in findmfs.items()}
    return cfg


def test_config_roundtrip():
    p = FindMfsParams(
        charge=-1, max_counts='C*H*N*O*S2', detect_halogens=False,
        halogen_cap='Cl6Br2', iso_weight=0.25, instrument='orbitrap', top_n=12,
    )
    cfg = ConfigParser()
    p.to_config(cfg)
    assert FindMfsParams.from_config(cfg) == p


def test_from_config_keeps_defaults_for_missing_or_bad_keys():
    p = FindMfsParams.from_config(_config(error_ppm='eight', top_n='30'))
    assert p.error_ppm == FindMfsParams().error_ppm
    assert p.top_n == 30
    # Keys from older configs are ignored, not errors
    assert FindMfsParams.from_config(_config(autodetect_cl_br='True')) == FindMfsParams()


def test_from_config_without_section():
    assert FindMfsParams.from_config(ConfigParser()) == FindMfsParams()
    assert FindMfsParams.from_config(None) == FindMfsParams()


@pytest.mark.parametrize('detect, cap, expected', [
    (True, 'Cl4Br4', 'Cl4Br4'),
    (False, 'Cl4Br4', None),     # unticked
    (True, '', None),            # ticked but blank: off, not a silent no-op cap
])
def test_effective_halogen_cap(detect, cap, expected):
    p = FindMfsParams(detect_halogens=detect, halogen_cap=cap)
    assert p.effective_halogen_cap == expected
    assert p.search_kwargs()['halogen_cap'] == expected


def test_search_kwargs_carry_constraints_and_every_score_term():
    p = FindMfsParams(
        max_counts='C*H*O*', min_counts='C1', min_rdbe=0.0, max_rdbe=20.0,
        check_octet=False, error_ppm=3.0, error_da=0.002,
        mass_weight=2.0, iso_weight=0.5, iso_ppm=4.0, iso_mz_match_da=0.01,
        iso_min_rel=0.05, chem_weight=0.0, chem_strength=2.0, chem_softness=0.5,
    )
    kw = p.search_kwargs()
    assert kw['max_counts'] == 'C*H*O*'
    assert kw['min_counts'] == 'C1'
    assert kw['finder_kwargs'] == dict(
        error_da=0.002, filter_rdbe=(0.0, 20.0), check_octet=False,
    )
    assert kw['mass_sigma_ppm'] == pytest.approx(1.0)      # window = 3 sigma
    for key in ('mass_weight', 'iso_weight', 'iso_ppm', 'iso_mz_match_da',
                'iso_min_rel', 'chem_weight', 'chem_strength', 'chem_softness'):
        assert kw[key] == getattr(p, key)


def test_blank_counts_fall_back_to_find_mfs_defaults():
    from find_mfs import DEFAULT_MAX_COUNTS
    kw = FindMfsParams(max_counts='', min_counts='').search_kwargs()
    assert kw['max_counts'] == DEFAULT_MAX_COUNTS
    assert kw['min_counts'] is None


@pytest.mark.parametrize('overrides', [
    dict(max_counts='C*H*Xx2'),                  # not an element
    dict(halogen_cap='F2'),                      # only Cl/Br are detectable
    dict(max_counts='C*H*O*', min_counts='N1'),  # min needs what max forbids
])
def test_validate_rejects_bad_constraints(overrides):
    with pytest.raises(ValueError):
        FindMfsParams(**overrides).validate()


def test_validate_accepts_defaults_and_disabled_bad_cap():
    FindMfsParams().validate()
    # A bad cap only matters while detection is on
    FindMfsParams(detect_halogens=False, halogen_cap='F2').validate()


def test_counts_to_str():
    inf = float('inf')
    assert counts_to_str({'C': inf, 'H': inf, 'S': 2, 'Cl': 4}) == 'C*H*S2Cl4'
    assert counts_to_str({}) is None
    assert counts_to_str(None) is None
