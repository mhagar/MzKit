import pytest

from core.utils.render_molecule import smiles_to_svg


@pytest.mark.parametrize("smiles", [
    "CCO",
    r"C(=C/c1ccccc1)\c1ccccc1",  # E-stilbene
    r"C(=C\c1ccccc1)\c1ccccc1",  # Z-stilbene
    r"C/C=C/C1CCCCC1",  # E double bond adjacent to a ring
])
def test_renders_without_warnings(smiles):
    svg, warnings = smiles_to_svg(smiles)
    assert "<svg" in svg
    assert warnings == []


@pytest.mark.parametrize("smiles", ["", "not a smiles", "C1CC"])
def test_invalid_smiles_raises(smiles):
    with pytest.raises(ValueError):
        smiles_to_svg(smiles)
