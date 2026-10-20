import numpy as np

from core.utils.spectrum_export import mgf_block, to_mgf


def _spec():
    arr = np.zeros(2, dtype=[('mz', 'f8'), ('intsy', 'f8')])
    arr['mz'] = [100.0, 200.5]
    arr['intsy'] = [10.0, 20.0]
    return arr


def test_mgf_block_bare():
    assert mgf_block(_spec()).splitlines() == [
        "BEGIN IONS", "100.00000 10.00", "200.50000 20.00", "END IONS",
    ]


def test_mgf_block_headers_and_to_mgf():
    lines = mgf_block(_spec(), {"RTINSECONDS": "12.3"}).splitlines()
    assert lines[1] == "RTINSECONDS=12.3"
    assert to_mgf(301.1, 1, 2, _spec()).splitlines()[1:4] == [
        "PEPMASS=301.1", "CHARGE=1+", "MSLEVEL=2",
    ]
