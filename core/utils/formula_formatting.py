"""
Contains functions for formatting Formula texts
"""
from molmass import Formula

import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from molmass import CompositionItem
    from core.data_structs import Ensemble
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
    ensemble: 'Ensemble',
    assignment: Optional['FormulaAssignment'],
) -> Optional[str]:
    """
    The compound-formula portion of an ensemble's label.

    Prefers the accepted candidate from a structured FormulaAssignment
    (formula &middot; adduct &middot; ppm); falls back to the ensemble's
    free-text `proposed_formula`; returns None if neither is available.

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
            parts.append(chosen.adduct)
        if chosen.error_ppm is not None:
            parts.append(f"{chosen.error_ppm:.1f} ppm")
        return "  &middot;  ".join(parts)

    # Fallback: free-text proposed formula (may not parse as a Formula).
    proposed = ensemble.proposed_formula
    if proposed:
        try:
            fhtml = format_formula_obj_to_html(Formula(proposed))
        except Exception:
            fhtml = proposed
        return f"<b>{fhtml}</b>"

    return None

