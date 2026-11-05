"""
AlignmentContext's annotation aggregates: an analyte's are display-only
summaries of its member ensembles' annotations.
"""
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import gui.controllers.main_controller  # noqa: E402,F401  (app import order)
from core.data_structs.ensemble import EnsembleAnnotation, Hypothesis  # noqa: E402
from gui.views.alignment_viewer.context import AlignmentContext  # noqa: E402


def _context(annotations: dict):
    members = {s: SimpleNamespace(uuid=10 + s, annotation=a) for s, a in annotations.items()}
    analyte = SimpleNamespace(
        uuid=99, consensus_mz=181.0712, consensus_rt=60.0,
        resolve_members=lambda get_sample: members,
        # the representative is sample 2's member
        representative_spectrum=lambda m: SimpleNamespace(sample_uuid=2, ensemble_uuid=12),
    )
    ctx = AlignmentContext(alignment=None, data_source=SimpleNamespace(get_sample=lambda u: None))
    return ctx, analyte, members


def test_target_is_the_focused_member_else_the_representative():
    ctx, analyte, members = _context({1: None, 2: None})
    assert ctx.target_ensemble(analyte) is members[2]
    assert ctx.target_ensemble(analyte, members[1]) is members[1]


def test_structure_agreement_and_notes():
    psa = EnsembleAnnotation(hypotheses=(Hypothesis("CCO", "Psammaplin A"),), chosen_idx=0)
    ctx, analyte, _ = _context({
        1: psa,
        2: EnsembleAnnotation(name="Psammaplin A", notes="sponge"),
        3: EnsembleAnnotation(name="other", notes="sponge"),
        4: None,
    })
    assert ctx.structure_agreement(analyte) == ("Psammaplin A", 2, 4)
    assert ctx.notes(analyte) == ["sponge"]
    assert ctx.analyte_label(analyte) == "181.0712 @ 60.0s (Psammaplin A)"

    ctx, analyte, _ = _context({1: None, 2: EnsembleAnnotation(notes="x")})
    assert ctx.structure_agreement(analyte) is None
    assert ctx.analyte_label(analyte) == "181.0712 @ 60.0s"
