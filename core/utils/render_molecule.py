"""
Render a SMILES string as an ACS1996-style SVG using RDKit CoordGen layout.
"""
# rdkit is imported lazily: its pip wheel bundles its own font/cairo libs, and
# loading it before the QApplication exists segfaults the first widget paint.


def _ez_labels(
        mol
) -> dict[int, str]:
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler
    rdCIPLabeler.AssignCIPLabels(mol)
    return {
        b.GetIdx(): b.GetProp("_CIPCode")
        for b in mol.GetBonds()
        if b.GetBondType() == Chem.BondType.DOUBLE and b.HasProp("_CIPCode")
    }


def smiles_to_svg(
        smiles: str,
        highlight_atoms: list[int] | None = None,
) -> tuple[str, list[str]]:
    """
    Return (svg_text, warnings). Raises ValueError on empty/unparseable SMILES.
    highlight_atoms: atom indices (as parsed from `smiles`) to highlight, with the bonds between them.
    """
    from rdkit import Chem
    from rdkit.Chem import rdCoordGen
    from rdkit.Chem.Draw import rdMolDraw2D

    if not smiles or not smiles.strip():
        raise ValueError("SMILES string is empty")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles!r}")

    rdCoordGen.AddCoords(mol)
    d = rdMolDraw2D.MolDraw2DSVG(-1, -1)
    rdMolDraw2D.SetACS1996Mode(
        d.drawOptions(), rdMolDraw2D.MeanBondLength(mol)
    )
    if highlight_atoms:
        keep = set(highlight_atoms)
        bonds = [b.GetIdx() for b in mol.GetBonds()
                 if b.GetBeginAtomIdx() in keep and b.GetEndAtomIdx() in keep]
        d.DrawMolecule(mol, highlightAtoms=list(keep), highlightBonds=bonds)
    else:
        d.DrawMolecule(mol)
    d.FinishDrawing()
    svg = d.GetDrawingText()

    # Check the 2D layout preserved E/Z: stereo re-perceived from coords must match the input.
    original = _ez_labels(Chem.Mol(mol))
    reread = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol))
    redrawn = _ez_labels(reread) if reread is not None else {}
    warnings = []
    for idx, label in original.items():
        if redrawn.get(idx) != label:
            b = mol.GetBondWithIdx(idx)
            warnings.append(
                f"Double bond between atoms {b.GetBeginAtomIdx()} and {b.GetEndAtomIdx()} "
                f"is {label} in the SMILES but drawn as {redrawn.get(idx, 'unspecified')}"
            )
    return svg, warnings
