"""
Contains functions for formatting Formula texts
"""
from molmass import Formula

import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from molmass import CompositionItem
    from core.data_structs.formula_assignment import FormulaAssignment


def format_formula_str_to_html(
    formula_string: str
) -> str:
    """
    Convert 'C32H66O10' to HTML with subscripts
    """
    return re.sub(
        pattern=r"(\d+)",
        repl="<sub>\1</sub>",
        string=formula_string,
    )


def format_formula_obj_to_html(
    formula: Formula
) -> str:
    """
    Convert a molmass.Formula object into HTML with subscripts and charge notation.
    Elements are ordered as: CHNOPS, halogens (F, Cl, Br, I), then Na and K.
    """
    # Define element ordering priority
    priority_order = ['C', 'H', 'N', 'O', 'P', 'S',
                      'F', 'Cl', 'Br', 'I', 'Na', 'K']

    composition = formula.composition()
    output: list[str] = []

    # First, add elements in priority order
    for symbol in priority_order:
        if symbol in composition:
            comp_item: 'CompositionItem' = composition[symbol]

            if symbol == 'e-':
                continue

            output.append(symbol)

            if comp_item.count > 1:
                output.append(
                    f"<sub>{comp_item.count}</sub>"
                )

    # Then add any remaining elements not in priority list
    for symbol, comp_item in composition.items():
        comp_item: 'CompositionItem'

        if symbol == 'e-' or symbol in priority_order:
            continue

        output.append(symbol)

        if comp_item.count > 1:
            output.append(
                f"<sub>{comp_item.count}</sub>"
            )

    # Add charge notation
    charge = formula.charge
    if charge > 0:
        output.append('+' * charge)
    elif charge < 0:
        output.append('-' * abs(charge))

    return "".join(output)


def format_assignment_label_html(
    assignment: Optional['FormulaAssignment'],
) -> Optional[str]:
    """
    The compound-formula portion of an ensemble's label: the accepted
    candidate of its FormulaAssignment (formula, adduct, ppm), or None if
    nothing is accepted.

    This is the single source of truth for "how do we render an ensemble's
    formula", shared by the EnsembleViewer's MS1 title strip and the
    sample-viewer peak overlays, so the two never disagree.
    """
    chosen = assignment.chosen if assignment is not None else None

    if chosen is not None:
        formula_html = format_formula_obj_to_html(
            Formula(chosen.formula_str)
        )
        parts = [f"<b>{formula_html}</b>"]
        if chosen.adduct:
            parts.append(f"({chosen.adduct})")
        if chosen.error_ppm is not None:
            parts.append(f"<br>{chosen.error_ppm:.1f}δ")
        return "".join(parts)

    return None

