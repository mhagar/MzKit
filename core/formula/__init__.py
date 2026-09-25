"""
core/formula: MS2-based molecular-formula assignment.

Thin orchestration over the find-mfs `annotate_precursor` primitive

Spectrum interpretation (monoisotopic pick, halogen detection, element widening)
 is done by find-mfs. This package:
  - extracts spectra from a source
  - assembles a query
  - stores ranked results

    FindMfsParams        (params.py)  : constraints + scoring, shared by every find-mfs path
    query_from_ensemble  (extract.py) : Takes Ensemble, returns FormulaQuery
    assign_formula       (assign_formula.py)  Takes FormulaQuery, returns FormulaAssignment
"""
from core.formula.params import FindMfsParams
from core.formula.query import FormulaQuery
from core.formula.extract import query_from_ensemble, query_from_signals
from core.formula.assign_formula import assign_formula

__all__ = [
    "FindMfsParams",
    "FormulaQuery",
    "query_from_ensemble",
    "query_from_signals",
    "assign_formula",
]
