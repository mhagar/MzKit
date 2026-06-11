from typing import Optional, TYPE_CHECKING
from PyQt5 import QtCore
from PyQt5.QtCore import QObject, QEvent
from PyQt5.QtWidgets import QMdiSubWindow

from gui.views.process_monitor import ProcessMonitorWindow
from gui.views.sample_viewer import SampleViewer
from gui.views.fingerprint_viewer import FingerprintViewerWindow
from gui.views.ensemble_viewer import EnsembleViewer
from gui.views.alignment_viewer import AlignmentViewer

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QMdiArea
    from core.controllers.ProcessController import ProcessController
    from core.data_structs import DataRegistry


class _HideOnCloseFilter(QObject):
    """
    Event filter that intercepts close events on a QMdiSubWindow and
    converts them into hide() calls, so the window persists and can be
    re-shown later without losing its state.

    Emits visibilityChanged(bool) on show/hide transitions so external
    UI (e.g. the Window menu) can sync its checked state.
    """
    visibilityChanged = QtCore.pyqtSignal(bool)  # True = visible

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Close:
            obj.hide()
            event.ignore()
            self.visibilityChanged.emit(False)
            return True
        if event.type() == QEvent.ShowToParent:
            self.visibilityChanged.emit(True)
        elif event.type() == QEvent.HideToParent:
            self.visibilityChanged.emit(False)
        return False


class SubWindowManager(QObject):
    """
    Manages MDI sub-windows with factory pattern and centralized
     lifecycle management

    Emits visibility_changed(window_type: str, is_visible: bool) whenever
    any managed subwindow is shown or hidden, including via the X button.
    """

    visibility_changed = QtCore.pyqtSignal(str, bool)
    
    # Window configuration registry
    WINDOW_CONFIGS = {
        'process_monitor': {
            'class': ProcessMonitorWindow,
            'dependencies': ['process_controller'],
            'title': 'Process Monitor'
        },
        'sample_viewer': {
            'class': SampleViewer,
            'dependencies': ['data_registry'],
            'title': 'Sample Viewer'
        },
        'fingerprint_viewer': {
            'class': FingerprintViewerWindow,
            'dependencies': ['data_registry'],
            'title': 'Fingerprint Viewer'
        },
        'ensemble_viewer': {
            'class': EnsembleViewer,
            'dependencies': ['data_registry'],
            'title': 'Ensemble Viewer',
        },
        'alignment_viewer': {
            'class': AlignmentViewer,
            'dependencies': ['data_registry'],
            'title': 'Alignment Viewer',
        }
    }
    
    def __init__(
        self,
        mdi_area: 'QMdiArea',
        process_controller: 'ProcessController',
        data_registry: 'DataRegistry'
    ):
        super().__init__()
        self.mdi_area = mdi_area
        self.dependencies = {
            'process_controller': process_controller,
            'data_registry': data_registry
        }
        
        # Track created windows
        self.windows: dict[str, any] = {}
        self.sub_windows: dict[str, QMdiSubWindow] = {}

        # One event filter per subwindow (intercepts close → hide)
        self._close_filters: dict[str, _HideOnCloseFilter] = {}


    def create_window(
        self,
        window_type: str,
    ) -> any:
        """
        Factory method to create windows w dependency injection
        """
        if window_type in self.windows:
            return self.windows[window_type]
            
        if window_type not in self.WINDOW_CONFIGS:
            raise ValueError(
                f"Unknown window type: {window_type}"
            )
        
        config = self.WINDOW_CONFIGS[window_type]
        
        # Inject required dependencies
        kwargs = {}
        for dep_name in config['dependencies']:
            if dep_name == 'process_controller':
                kwargs['process_controller'] = self.dependencies['process_controller']
            elif dep_name == 'data_registry':
                kwargs['data_source'] = self.dependencies['data_registry']
        
        # Create the window
        window = config['class'](**kwargs)
        self.windows[window_type] = window
        
        return window
    
    def add_to_mdi(
        self,
        window_type: str,
    ) -> QMdiSubWindow:
        """
        Create window and add it to MDI area.
        """
        window = self.create_window(window_type)
        sub_window = self.mdi_area.addSubWindow(window)

        config = self.WINDOW_CONFIGS[window_type]
        sub_window.setWindowTitle(config['title'])

        # Make sure the X button hides instead of destroying. By default
        # QMdiSubWindow without WA_DeleteOnClose is hidden, but we install
        # an explicit filter so behaviour is consistent across Qt versions
        # and so we have a hook for future visibility-change reporting.
        close_filter = _HideOnCloseFilter(parent=sub_window)
        sub_window.installEventFilter(close_filter)
        close_filter.visibilityChanged.connect(
            lambda visible, wt=window_type: self.visibility_changed.emit(
                wt, visible
            )
        )
        self._close_filters[window_type] = close_filter

        self.sub_windows[window_type] = sub_window
        return sub_window
    
    def get_window(
        self,
        window_type: str,
    ) -> Optional[any]:
        """
        Get a window instance if it exists.
        """
        return self.windows.get(window_type)
    
    def show_window(self, window_type: str) -> None:
        """
        Show a specific window, creating it if necessary. Brings the
        subwindow back if the user previously closed it via the X button
        (which is intercepted to hide-not-destroy).
        """
        if window_type not in self.sub_windows:
            self.add_to_mdi(window_type)

        sub_window = self.sub_windows[window_type]
        # Always show the inner widget too — some callers used to call
        # widget.show() directly, which would only un-hide the inner widget
        # but leave the QMdiSubWindow itself hidden.
        sub_window.widget().show()
        sub_window.show()
        sub_window.raise_()
        sub_window.setFocus()

    def hide_window(self, window_type: str) -> None:
        """
        Hide a subwindow without destroying it. Preserves state for
        later re-display via show_window().
        """
        if window_type in self.sub_windows:
            self.sub_windows[window_type].hide()

    def is_window_visible(self, window_type: str) -> bool:
        """
        Returns True if the named subwindow is currently visible.
        """
        sub_window = self.sub_windows.get(window_type)
        return bool(sub_window and sub_window.isVisible())

    # Subwindows whose content is tied to the loaded project. These get
    # reset/hidden when the workspace is cleared (New/Open/Close project).
    DATA_WINDOW_TYPES = (
        'sample_viewer',
        'ensemble_viewer',
        'alignment_viewer',
        'fingerprint_viewer',
    )

    def reset_data_windows(self) -> None:
        """
        Reset and hide every data-bound subwindow so no stale content
        from a previous project remains visible after the workspace is
        cleared.

        A window is reset via its own reset_for_new_project() hook if it
        provides one; either way it is hidden. The window instances (and
        their signal connections) are preserved.
        """
        for window_type in self.DATA_WINDOW_TYPES:
            window = self.windows.get(window_type)
            if window is not None and hasattr(window, 'reset_for_new_project'):
                window.reset_for_new_project()
            self.hide_window(window_type)

    def initialize_all_windows(self) -> None:
        """
        Create all windows and add them to MDI area.
        """
        for window_type in self.WINDOW_CONFIGS:
            self.add_to_mdi(window_type)

    def close_window(self, window_type: str) -> None:
        """
        Hide a subwindow (kept for backwards compatibility). This used
        to destroy the window; it now just hides it. Use
        destroy_window() if you really want to release the resources.
        """
        self.hide_window(window_type)

    def destroy_window(self, window_type: str) -> None:
        """
        Permanently remove a subwindow and release its widget. Rarely
        needed — usually hide_window() is what you want.
        """
        if window_type in self.sub_windows:
            sub_window = self.sub_windows.pop(window_type)
            self.mdi_area.removeSubWindow(sub_window)
            sub_window.deleteLater()
            self.windows.pop(window_type, None)
            self._close_filters.pop(window_type, None)