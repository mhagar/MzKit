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


def test_highlight_atoms_draws_highlight():
    plain, _ = smiles_to_svg("CCO")
    lit, _ = smiles_to_svg("CCO", highlight_atoms=[1, 2])
    assert lit != plain


def test_highlight_does_not_move_the_drawing():
    """Highlighted and plain renders share one layout, so fragments don't jitter."""
    import re
    bond_paths = lambda svg: re.findall(r"class='bond-\d+ atom-\d+ atom-\d+' d='([^']+)'", svg)  # noqa: E731
    plain, _ = smiles_to_svg("CCOC(=O)c1ccc(N)cc1")
    lit, _ = smiles_to_svg("CCOC(=O)c1ccc(N)cc1", highlight_atoms=[0, 1, 2])
    assert re.search(r"width='[\d.]+px'", lit).group() == re.search(r"width='[\d.]+px'", plain).group()
    assert set(bond_paths(plain)) <= set(bond_paths(lit))


def test_sized_render():
    svg, _ = smiles_to_svg("CCO", size=(120, 80), dark=True)
    assert "width='120px'" in svg and "height='80px'" in svg
    assert "#1F3F7F00" not in svg  # unlit highlights stripped


def test_keeps_element_colours_and_can_skip_labels():
    svg, _ = smiles_to_svg("CCO", size=(120, 80), labels=False)
    assert "#FF0000" in svg            # oxygen's bond half stays red despite the all-atom highlight
    labelled, _ = smiles_to_svg("CCO", size=(120, 80))
    assert "class='atom-2'" in labelled and "class='atom-2'" not in svg   # the O label's glyphs
