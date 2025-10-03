from core.cli.process_runner import ProcessRunner

from PyQt5.QtCore import (
    QObject, pyqtSignal, QTimer, QAbstractTableModel, Qt, QModelIndex, QVariant,
    QSize
)
import logging
import threading
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from gui.controllers.main_controller import MainController


class ProcessSignals(QObject):
    """
    Defines the signals available for commnuicating process events
    """
    output_ready = pyqtSignal(int, str, str)  # process_id, level, msg
    status_changed = pyqtSignal(int, str)  # process_id, new_status
    process_finished = pyqtSignal(int, object)  # process_id, result


class ProcessController:
    """
    Controls processes (lol)
    """
    def __init__(
            self,
            main_controller: 'MainController',
            log_level: int = logging.INFO,
    ):
        self.main_controller = main_controller

        # Counter used to generate process IDs
        self.process_counter = 0

        # Registry of functions to return process results to
        self.return_func_registry: dict[int, callable] = {}

        # QAbstractTableModel for storing info about processes
        self.model = ProcessTableModel()

        # Signals object for process communication
        self.process_signals = ProcessSignals()
        self.process_signals.output_ready.connect(self._handle_process_output)
        self.process_signals.status_changed.connect(self._handle_status_change)
        self.process_signals.process_finished.connect(self._handle_process_finished)

        # Create a timer for continuously polling processes
        self.process_poll_timer = QTimer()
        self.process_poll_timer.setInterval(100)  # 100ms
        self.process_poll_timer.timeout.connect(self._poll_active_processes)
        self.process_poll_timer.start()

        # Use a thread-safe dictionary for processes
        self._processes_lock = threading.RLock()
        self.running_processes: dict[int, ProcessRunner] = {}
        self.last_known_status: dict[int, str] = {}

        # Set up logger
        self.logger = logging.getLogger("process_controller")
        self.logger.setLevel(log_level)
        self.log_level = log_level

    def start_process(
            self,
            module_path: str,
            function_name: str = 'main',
            parameters: Optional[dict[str, any]] = None,
            on_completion_func: Optional[callable] = None,
            log_level: Optional[int] = None,
    ) -> int:
        """
        Start a function/process in the background
        :param module_path: Import path or file path to the Python module
        :param function_name: Name of the function to call (default: "main")
        :param parameters: Dictionary of parameters to pass to function
        :param on_completion_func: Function that's called upon process completion,
            with the results
        :param log_level: Optional log level override
        :return: process_id (int)
        """
        if not log_level:
            log_level = self.log_level

        process = self._create_process(
            module_path=module_path,
            function_name=function_name,
            parameters=parameters,
            log_level=log_level,
        )

        process_id = self.process_counter
        self.process_counter += 1

        with self._processes_lock:
            self.running_processes[process_id] = process
            self.last_known_status[process_id] = process.status

        # Register the 'on_completion' function, so it receives the output
        #  of the process when it's ready
        if on_completion_func:
            self.return_func_registry[process_id] = on_completion_func

        process.start()
        self.model.addProcess(
            process_id=process_id,
            process_name=f"{module_path}",
            process_status=self.last_known_status[process_id],
            process_progress="",
        )

        self.logger.debug(
            f"Started process {process_id}: {module_path}.{function_name}"
        )

        return process_id


    @staticmethod
    def _create_process(
            module_path: str,
            function_name: str,
            parameters: Optional[dict[str, any]] = None,
            log_level: Optional[int] = None,
    ) -> ProcessRunner:
        """
        Create a process runner for the specified Python function

        :param module_path: Import path or file path to the Python module
        :param function_name: Name of the function to call
        :param parameters: Dictionary of parameters to pass to the function
        :param log_level: Optional log level override
        :return: ProcessRunner instance
        """
        return ProcessRunner(
            module_path=module_path,
            function_name=function_name,
            parameters=parameters,
            log_level=log_level,
        )


    def get_process(
            self,
            process_id: int,
    ) -> Optional[ProcessRunner]:
        """
        Get a process by ID (thread-safe)

        :param process_id: ID of the process
        :return: ProcessRunner instance, or None
        """
        with self._processes_lock:
            return self.running_processes.get(
                process_id
            )


    def get_process_status(
            self,
            process_id: int
    ) -> Optional[str]:
        """
        Get the status of a process

        :param process_id: ID of the process
        :return: Status string, or None if not found
        """
        process = self.get_process(
            process_id
        )
        if process:
            return process.status
        return None


    def get_process_output(
            self,
            process_id: int,
            block: bool = False,
            timeout: Optional[int] = None,
    ) -> Optional[tuple[str, str]]:
        """
        Get the next output item from a process

        :param process_id: ID of the process
        :param block: Whether to block until output available
        :param timeout: Seconds to timeout (if blocking)
        :return: (stream, message) tuple, or None
        """
        process = self.get_process(
            process_id
        )

        if process:
            return process.get_output(
                block=block,
                timeout=timeout,
            )

        return None


    def get_all_process_output(
            self,
            process_id: int
    ) -> list[tuple]:
        """
        Get all available output from a process

        :param process_id: ID of the process
        :return: List of (stream, message) tuples
        """
        process = self.get_process(
            process_id
        )

        if process:
            return process.get_all_output()

        return []


    def get_process_result(
            self,
            process_id: int,
    ) -> any:
        """
        Get the return value from a completed process

        :param process_id: process ID
        :return: The return value of the function, or None
        """
        process = self.get_process(
            process_id
        )

        if process:
            return process.result

        return None


    def cleanup_completed_processes(
            self
    ) -> list[int]:
        """
        Remove completed or failed processes from the running list

        :return: List of process IDs that were cleaned up
        """
        to_remove = []
        for pid, process in self.running_processes.items():
            if process.status in ["completed", "failed", "error"]:
                to_remove.append(pid)

        for pid in to_remove:
            del self.running_processes[pid]
            self.logger.debug(
                f"Cleaned up process {pid}"
            )

        return to_remove


    def terminate_process(
            self,
            process_id: int,
    ) -> bool:
        """
        Mark a process for termination

        :param process_id: ID of process to terminate
        :return: True if the process was found and marked for termination
        """
        process = self.get_process(
            process_id
        )

        if process:
            # Set status to indicate that it should be cleaned up
            process.status = "error"
            self.logger.info(
                f"Marked process {process_id} for termination"
            )
            return True
        return False


    def _poll_active_processes(
            self
    ) -> None:
        """
        Poll all active processes for output and status changes.

        Called by a QTimer at regular intervals
        :return:
        """
        processes_to_remove = []

        for process_id, status in self.last_known_status.items():
            # Check for new output:
            while True:
                output = self.get_process_output(
                    process_id,
                    block=False,
                )

                if not output:
                    break

                level, message = output

                # Emit signal with the output
                self.process_signals.output_ready.emit(
                    process_id, level, message
                )

            # Check for status changes
            current_status = self.get_process_status(
                process_id
            )

            last_known_status = status

            if current_status != last_known_status:
                self.last_known_status[process_id] = current_status
                self.process_signals.status_changed.emit(
                    process_id, current_status,
                )

                # If process is completed, failed, or has an error:
                if current_status in ["completed", "failed", "error"]:
                    result = self.get_process_result(
                        process_id
                    )

                    self.process_signals.process_finished.emit(
                        process_id, result,
                    )

                    processes_to_remove.append(process_id)

                # Need to break because dictionary was changed mid-loop!
                break

        # Remove finished processes from having status tracked
        for process_id in processes_to_remove:
            del self.last_known_status[process_id]


    def _handle_process_output(
            self,
            process_id: int,
            level: str,
            message: str,
    ) -> None:
        """
        Handle output from a process
        :param process_id:
        :param level:
        :param message:
        :return:
        """
        logging.info(
            f"[Process {process_id}] [{level}] {message}"
        )
        pass


    def _handle_status_change(
            self,
            process_id: int,
            status: str,
    ) -> None:
        """
        Handle status change from a process
        :param process_id:
        :param status:
        :return:
        """
        logging.info(
            f"Process {process_id} status changed to: {status}"
        )

        # Update process monitor UI
        self.model.updateProcess(
            process_id=process_id,
            status=status,
        )


    def _handle_process_finished(
            self,
            process_id: int,
            result: any,
    ) -> None:
        """
        Handle a process finishing.
        :param process_id:
        :param result:
        :return:
        """
        logging.info(
            f"Process {process_id} finished with result: {result}"
        )

        # Call the 'on completion' function corresponding to this process,
        # (if it has one)
        if process_id in self.return_func_registry.keys():
            self.return_func_registry[process_id](result)

        # Clean up the process
        self.cleanup_completed_processes()

    def cancel_process(
            self,
            process_id: int,
    ) -> None:
        """
        Cancel a running process, # TODO: Implement
        :param process_id:
        :return:
        """
        if self.terminate_process(process_id):
            logging.info(
                f"Process {process_id} cancelled"
            )


# noinspection PyMethodOverriding
class ProcessTableModel(QAbstractTableModel):
    """
    Model for storing information about running processes
    """
    def __init__(
            self,
            parent=None,
    ):
        super().__init__(parent)
        self.headers = ["Process ID", "Name", "Status", "Progress"]

        # Initialize empty data list
        self.processes = []


    def rowCount(
            self,
            index,
    ):
        # The length of the outer list
        return len(self.processes)


    def columnCount(
            self,
            index: any,
    ):
        return len(self.headers)


    def data(
            self,
            index: QModelIndex,
            role: int,
    ):
        """
        Return data for display in the table cells
        :param index:
        :param role:
        :return:
        """
        if not index.isValid() or not (0 <= index.row() < len(self.processes)):
            return QVariant()

        if role == Qt.DisplayRole:
            if index.isValid():
                return str(self.processes[index.row()][index.column()])

        if role == Qt.TextAlignmentRole:
            return Qt.AlignHCenter + Qt.AlignVCenter

        return QVariant()


    def headerData(
            self,
            section: int,
            orientation: int,
            role: int = Qt.DisplayRole,
    ):
        """
        Return header data
        :param section:
        :param orientation:
        :param role:
        :return:
        """
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal and self.headers:
                if 0 <= section < len(self.headers):
                    return self.headers[section]

        if role == Qt.SizeHintRole:
            if orientation == Qt.Horizontal:
                if section == 1:
                    return QVariant(QSize(250, 15))

        return QVariant()


    def addProcess(
            self,
            process_id: int,
            process_name: str,
            process_status: str,
            process_progress: str,
    ):
        """
        Add a new record of a process to the table
        :param process_id:
        :param process_name:
        :param process_status:
        :param process_progress:
        :return:
        """
        self.beginInsertRows(
            QModelIndex(),
            len(self.processes),
            len(self.processes),
        )

        self.processes.append(
            [
                str(process_id),
                process_name,
                process_status,
                process_progress,
            ]
        )

        self.endInsertRows()

        return True


    def updateProcess(
            self,
            process_id: int,
            status: Optional[str] = None,
            progress: Optional[str] = None,
    ):
        """
        Given a process_id, update the status and/or progress
        :param process_id:
        :param status:
        :param progress:
        :return:
        """

        for row, process in enumerate(self.processes):
            if process[0] == str(process_id):
                if status:
                    process[2] = status

                if progress:
                    process[3] = progress

                # Emit data changed signal
                top_left = self.index(row, 0)
                bottom_right = self.index(
                    row,
                    len(self.headers) - 1,
                )

                self.dataChanged.emit(
                    top_left,
                    bottom_right,
                )

                return True

        # Process not found
        return False

    def deleteAllProcesses(self):
        """
        Deletes all processes from the table
        :return:
        """

        self.beginResetModel()
        self.processes.clear()
        self.endResetModel()

        return True


