from molmass import Formula, FormulaError


def delta_formula(anchor: str, item: str) -> str:
    """
    Signed formula difference `item - anchor`, i.e. +CH2 or −C2H4
    Returns "n/a" if either formula is empty or unparseable,
     or the difference isn't expressible (each has atoms the other lacks);
     "0" if identical.
    """
    if not anchor or not item:
        return "n/a"
    try:
        a, b = Formula(anchor), Formula(item)
        try:
            diff, sign = b - a, "+"
        except ValueError:
            diff, sign = a - b, "−"
        if not str(diff):  # molmass raises on `.formula` of an empty Formula
            return "0"
        return f"{sign}{diff.formula}"
    except (ValueError, FormulaError):
        return "n/a"
