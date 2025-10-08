import abc
import logging

class CLITool(abc.ABC):
    """
    Generic class for all CLI tools
    """

    def __init__(
            self,
            logger=None,
    ):
        self.logger = logger or logging.getLogger(self.__class__.name__)

    @abc.abstractmethod
    def run(
            self,
            *args,
            **kwargs,
    ):
        """
        Run the tool with the given arguments
        """

    def report_progress(
            self,
            percentage: float,
            message: str,
    ):
        self.logger.info(
            f"Progress: {percentage}% - {message}"
        )
        # Will be captured by the ProcessRunner