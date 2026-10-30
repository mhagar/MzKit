"""
Render a SMILES string as an SVG using RDKit CoordGen layout: ACS1996 style on a
canvas fitted to the molecule, or drawn at a given pixel size.
"""
import re
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


# Highlight colour for atoms/bonds not asked for: every atom is highlighted so the
# drawing's bounding box doesn't depend on which are (no jitter between fragments).
# Qt's SVG renderer paints 8-digit (alpha) colours opaque, so these are stripped.
_HIGHLIGHT = (0.93, 0.42, 0.42, 1.0)  # dark enough to read on white or black
_UNLIT = (0.125, 0.25, 0.5, 0.0)
_UNLIT_RE = re.compile(r"<[^<>]*#1F3F7F00[^<>]*>\n?")


def smiles_to_svg(
        smiles: str,
        highlight_atoms: list[int] | None = None,
        size: tuple[int, int] | None = None,
        dark: bool = False,
        labels: bool = True,
) -> tuple[str, list[str]]:
    """
    Return (svg_text, warnings). Raises ValueError on empty/unparseable SMILES.
    highlight_atoms: atom indices (as parsed from `smiles`) to highlight, with the bonds between them.
    size: (width, height) px to draw at, with fixed-px bonds (so small renders keep
        visible lines) and labels that shrink with the drawing. None: ACS1996 style,
        canvas fitted to the molecule.
    dark: RDKit's dark mode (light lines on black).
    labels: False draws no atom labels (heteroatoms show only as bond colour), for thumbnails.
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
    if size is None:
        d = rdMolDraw2D.MolDraw2DSVG(-1, -1)
        rdMolDraw2D.SetACS1996Mode(
            d.drawOptions(), rdMolDraw2D.MeanBondLength(mol)
        )
    else:
        d = rdMolDraw2D.MolDraw2DSVG(*size)
        opts = d.drawOptions()
        opts.bondLineWidth = 1.5 if min(size) < 200 else 2
        opts.minFontSize = 4   # labels shrink rather than crash into each other
        opts.padding = 0.05
    if dark:
        rdMolDraw2D.SetDarkMode(d.drawOptions())
    d.drawOptions().noAtomLabels = not labels
    d.drawOptions().standardColoursForHighlightedAtoms = True   # every atom is highlighted (below)
    keep = set(highlight_atoms or ())
    lit = lambda on: _HIGHLIGHT if on else _UNLIT  # noqa: E731
    d.DrawMolecule(
        mol,
        highlightAtoms=[a.GetIdx() for a in mol.GetAtoms()],
        highlightAtomColors={a.GetIdx(): lit(a.GetIdx() in keep) for a in mol.GetAtoms()},
        highlightBonds=[b.GetIdx() for b in mol.GetBonds()],
        highlightBondColors={
            b.GetIdx(): lit(b.GetBeginAtomIdx() in keep and b.GetEndAtomIdx() in keep)
            for b in mol.GetBonds()
        },
    )
    d.FinishDrawing()
    svg = _UNLIT_RE.sub('', d.GetDrawingText())

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
