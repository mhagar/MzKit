import threading
import queue
import importlib.util
import importlib
import traceback
import logging
from typing import Literal, Any, Union, Optional, TextIO


from typing import Literal

class ProcessRunner(
    threading.Thread,
):
    """
    Runs a Python function/script in a background thread
    """
    def __init__(
            self,
            module_path: str,
            function_name: str = "main",
            parameters: Optional[dict[str, Any]] = None,
            log_level: int = logging.INFO,
    ):
        super().__init__()
        self.module_path = module_path
        self.function_name = function_name
        self.parameters = parameters or {}
        self.output_queue: queue.Queue = queue.Queue()
        self.status: Literal['ready','running', 'completed', 'failed', 'error'] = 'ready'
        self.progress: str = ""
        self.daemon = True  # Allow main program to exit
        self.result = None

        # Set up a logger for this process
        self.logger = logging.getLogger(
            f"process.{id(self)}"
        )

        print(
            f"Setting logger level to log_level = {log_level}"
        )

        self.logger.setLevel(log_level)

        # Handler that puts log messages into queue
        self.queue_handler = QueueLogHandler(
            self.output_queue
        )
        self.logger.addHandler(self.queue_handler)


    def run(
            self
    ) -> None:
        """
        Import the module and run the specified function
        """
        try:
            self.status = "running"
            self.logger.info(
                f"Starting process: {self.module_path}.{self.function_name}"
            )

            # Import the module
            try:
                # First, try importing as a regular package
                module = importlib.import_module(
                    self.module_path
                )
            except ImportError:
                # If that fails, try loading from a filepath
                try:
                    spec = importlib.util.spec_from_file_location(
                        name="dynamic_module",
                        location=self.module_path,
                    )
                    if spec is None or spec.loader is None:
                        raise ImportError(
                            f"Could not load module from {self.module_path}"
                        )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                except Exception as e:
                    self.logger.error(
                        f"Failed to import module: {e}"
                    )
                    raise

            # Get the function
            if not hasattr(module, self.function_name):
                raise AttributeError(
                    f"Module {self.module_path} has no function {self.function_name}"
                )

            func = getattr(module, self.function_name)

            # Create a log handler for the module and inject it
            module_logger = logging.getLogger(
                self.module_path
            )
            original_handlers = list(module_logger.handlers)
            module_logger.addHandler(self.queue_handler)

            # Run the function with provided parameters
            try:
                self.result = func(**self.parameters)
                self.status = "completed"
                self.logger.info(
                    f"Process completed: {self.module_path}.{self.function_name}",
                )
            except Exception as e:
                self.status = "failed"
                self.logger.error(
                    f"Error running {self.function_name}: {str(e)}"
                )

                self.logger.error(
                    str(traceback.format_exc())
                )
            finally:
                # Restore original logger configuration
                module_logger.handlers = original_handlers

        except Exception as e:
            self.status = "error"
            self.logger.error(
                f"Process runner error: {str(e)}",
            )
            self.logger.error(
                str(traceback.format_exc()),
            )


    def get_output(
            self,
            block: bool = False,
            timeout: Optional[int] = None,
    ) -> Optional[tuple[str, str]]:
        """
        Get the next output message from the queue

        :param block: Whether to block until output available
        :param timeout: Seconds to timeout (if blocking)
        :return: (stream, message) tuple, or None
        """
        try:
            return self.output_queue.get(
                block=block,
                timeout=timeout,
            )
        except queue.Empty:
            return None


    def get_all_output(self):
        """
        Get all available output, without blocking
        """
        all_output = []
        while True:
            try:
                output = self.output_queue.get_nowait()
                all_output.append(output)
            except queue.Empty:
                break

        return all_output


class QueueLogHandler(logging.Handler):
    """
    A logging handler that puts logs into a queue
    """
    def __init__(
            self,
            output_queue,
    ):
        super().__init__()
        self.output_queue = output_queue


    def emit(
            self,
            record,
    ):
        try:
            # Format the record
            msg = self.format(record)

            # Put in queue with level as first element
            self.output_queue.put(
                (record.levelname.lower(), msg)
            )
        except Exception as e:
            self.handleError(record)