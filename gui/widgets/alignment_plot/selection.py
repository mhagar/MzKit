"""
The alignment map's selection state, shared by the plot and the Inspector.
Three independent pieces, each with its own signal:

 - selection: ordered analyte targets. Ensembles are never listed: selecting
              one selects its analyte (and focuses the ensemble), and a
              selected analyte covers its member ensembles
 - focus:     the one target being inspected, a listed analyte or one of its
              members (None only when the selection is empty)
 - anchor:    the reference for mirror comparisons; any target, kept across
              selection changes

Targets are compared by `target_key`: re-rendering the map creates new
HoverTarget objects for the same items.
"""
from typing import Callable, Optional

from PyQt5 import QtCore

from gui.widgets.alignment_plot.layout import HoverTarget


def target_key(target: Optional[HoverTarget]) -> Optional[tuple]:
    """Identity of a selectable target, independent of the objects it holds."""
    if target is None:
        return None
    return target.kind, target.analyte_id, target.sample_uuid


def display_order(targets: list[HoverTarget]) -> list[HoverTarget]:
    """
    Targets grouped by analyte, in order of first appearance, each group's
    analyte target first (the Inspector tree's row order).
    """
    groups: dict = {}
    for target in targets:
        groups.setdefault(target.analyte_id, []).append(target)
    return [
        t for group in groups.values()
        for t in sorted(group, key=lambda t: t.kind != 'analyte')
    ]


def _covered(target: HoverTarget, keys) -> bool:
    """`target` is in `keys` (target keys), or is an ensemble of an analyte in it."""
    return target_key(target) in keys or (
        target.kind == 'ensemble' and ('analyte', target.analyte_id, None) in keys
    )


def _analyte_of(target: HoverTarget) -> HoverTarget:
    """The analyte target an ensemble target belongs to (analytes: themselves)."""
    if target.kind == 'analyte':
        return target
    return target._replace(kind='analyte', sample_uuid=None, ensemble=None)


class SelectionModel(QtCore.QObject):
    sigSelectionChanged = QtCore.pyqtSignal(object)  # list[HoverTarget]
    sigFocusChanged = QtCore.pyqtSignal(object)      # HoverTarget | None
    sigAnchorChanged = QtCore.pyqtSignal(object)     # HoverTarget | None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._selection: list[HoverTarget] = []
        self._focus: Optional[HoverTarget] = None
        self._anchor: Optional[HoverTarget] = None

    @property
    def selection(self) -> list[HoverTarget]:
        return list(self._selection)

    @property
    def focus(self) -> Optional[HoverTarget]:
        return self._focus

    @property
    def anchor(self) -> Optional[HoverTarget]:
        return self._anchor

    def contains(self, target: HoverTarget) -> bool:
        """Listed, or an ensemble of a listed analyte."""
        return _covered(target, {target_key(t) for t in self._selection})

    def lists(self, target: Optional[HoverTarget]) -> bool:
        """Listed itself (not only through its analyte)."""
        key = target_key(target)
        return any(target_key(t) == key for t in self._selection)

    def set_selection(
        self,
        targets: list[HoverTarget],
        focus: Optional[HoverTarget] = None,
    ):
        """
        Replace the selection with the analytes of `targets`; `focus` (if
        covered) else the last of `targets` (an ensemble stays the focus) is
        focused.
        """
        if focus is None and targets:
            focus = targets[-1]
        unique: dict = {}
        for target in map(_analyte_of, targets):
            unique.setdefault(target_key(target), target)
        targets = list(unique.values())
        if focus is None or not _covered(focus, unique):
            focus = targets[-1] if targets else None
        else:
            focus = unique.get(target_key(focus), focus)

        selection_changed = list(unique) != [target_key(t) for t in self._selection]
        focus_changed = target_key(focus) != target_key(self._focus)
        # Both are updated before either signal goes out
        self._selection, self._focus = targets, focus
        if selection_changed:
            self.sigSelectionChanged.emit(self.selection)
        if focus_changed:
            self.sigFocusChanged.emit(focus)

    def toggle(self, target: HoverTarget):
        """
        Ctrl+click: add `target`'s analyte and focus `target`, or remove
        the analyte if it's already selected.
        """
        key = target_key(_analyte_of(target))
        kept = [t for t in self._selection if target_key(t) != key]
        if len(kept) < len(self._selection):
            self.set_selection(kept, focus=self._focus)
        else:
            self.set_selection(self._selection + [target], focus=target)

    def set_focus(self, target: Optional[HoverTarget]):
        """Focus a selected (listed or covered) target; anything else is ignored."""
        key = target_key(target)
        if target is None or not self.contains(target) or key == target_key(self._focus):
            return
        self._focus = next((t for t in self._selection if target_key(t) == key), target)
        self.sigFocusChanged.emit(self._focus)

    def set_anchor(self, target: Optional[HoverTarget]):
        changed = target_key(target) != target_key(self._anchor)
        self._anchor = target
        if changed:
            self.sigAnchorChanged.emit(target)

    def remap(self, resolve: Callable[[tuple], Optional[HoverTarget]]):
        """
        After a re-render: swap every target for `resolve(its key)`, dropping
        those that resolve to None (no longer drawn).
        """
        def again(target):
            return None if target is None else resolve(target_key(target))

        anchor = again(self._anchor)
        self.set_selection(
            [t for t in map(again, self._selection) if t is not None],
            focus=again(self._focus),
        )
        self.set_anchor(anchor)

    def clear(self, anchor: bool = True):
        self.set_selection([])
        if anchor:
            self.set_anchor(None)
