import pytest

from core.formula import delta_formula


@pytest.mark.parametrize("anchor, item, expected", [
    ("CH2", "C6H6", "+C5H4"),
    ("C6H6", "CH2", "−C5H4"),
    ("C6H6", "C6H6", "0"),
    ("CH4", "C2H2", "n/a"),
    ("", "CH2", "n/a"),
    ("C6H6", "", "n/a"),
    ("not a formula", "CH2", "n/a"),
])
def test_delta_formula(anchor, item, expected):
    assert delta_formula(anchor, item) == expected
