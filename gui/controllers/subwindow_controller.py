from typing import Optional, TYPE_CHECKING
from PyQt5.QtWidgets import QMdiSubWindow

from gui.views.process_monitor import ProcessMonitorWindow
from gui.views.sample_viewer import SampleViewer
from gui.views.fingerprint_viewer import FingerprintViewerWindow
from gui.views.analyte_table_viewer import AnalyteTableViewerWindow
from gui.views.ensemble_viewer import EnsembleViewer

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QMdiArea
    from core.controllers.ProcessController import ProcessController
    from core.data_structs import DataRegistry


class SubWindowManager:
    """
    Manages MDI sub-windows with factory pattern and centralized
     lifecycle management
    """
    
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
        'analyte_viewer': {
            'class': AnalyteTableViewerWindow,
            'dependencies': ['data_registry'],
            'title': 'Analyte Table Viewer'
        },
        'ensemble_viewer': {
            'class': EnsembleViewer,
            'dependencies': ['data_registry'],
            'title': 'Ensemble Viewer',
        }
    }
    
    def __init__(
        self,
        mdi_area: 'QMdiArea',
        process_controller: 'ProcessController',
        data_registry: 'DataRegistry'
    ):
        self.mdi_area = mdi_area
        self.dependencies = {
            'process_controller': process_controller,
            'data_registry': data_registry
        }
        
        # Track created windows
        self.windows: dict[str, any] = {}
        self.sub_windows: dict[str, QMdiSubWindow] = {}


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
        Show a specific window, creating it if necessary.
        """
        if window_type not in self.sub_windows:
            self.add_to_mdi(window_type)
        
        sub_window = self.sub_windows[window_type]
        sub_window.show()
        sub_window.raise_()
    
    def initialize_all_windows(self) -> None:
        """
        Create all windows and add them to MDI area.
        """
        for window_type in self.WINDOW_CONFIGS:
            self.add_to_mdi(window_type)
    
    def close_window(self, window_type: str) -> None:
        """
        Close a specific window.
        """
        if window_type in self.sub_windows:
            self.sub_windows[window_type].close()
            del self.sub_windows[window_type]
            del self.windows[window_type]