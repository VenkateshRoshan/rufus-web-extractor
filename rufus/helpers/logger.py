import logging
import os
import sys
import inspect
from datetime import datetime
from pathlib import Path
import colorama
from colorama import Fore, Style

# Initialize colorama to work properly on all platforms
colorama.init(autoreset=True)


class RufusLogger:
    """
    Custom logger for Rufus web extractor.
    Provides colored console output and file logging with detailed context information.
    """

    # ANSI color codes for terminal output
    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.BLUE,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one logger instance exists."""
        if cls._instance is None:
            cls._instance = super(RufusLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, log_level=logging.DEBUG, log_dir="logs"):
        """
        Initialize the Rufus logger.

        Args:
            log_level: Minimum logging level (default: DEBUG)
            log_dir: Directory to store log files (default: 'logs')
        """
        if self._initialized:
            return

        self.log_level = log_level
        self.log_dir = log_dir

        # Create logs directory if it doesn't exist
        Path(log_dir).mkdir(exist_ok=True, parents=True)

        # Create a logger
        self.logger = logging.getLogger("rufus")
        self.logger.setLevel(log_level)
        self.logger.propagate = False

        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(ColoredFormatter())

        # Create file handler for all logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_logs_file = os.path.join(log_dir, f"rufus_{timestamp}.log")
        file_handler = logging.FileHandler(all_logs_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(filename)s:%(funcName)s:%(lineno)d - %(message)s"
            )
        )

        # Create file handler for error logs only
        error_logs_file = os.path.join(log_dir, f"rufus_errors_{timestamp}.log")
        error_file_handler = logging.FileHandler(error_logs_file)
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(filename)s:%(funcName)s:%(lineno)d - %(message)s"
            )
        )

        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_file_handler)

        self._initialized = True

    def _get_caller_info(self):
        """Extract caller's filename, function name and line number."""
        # Get the stack frame of the caller (depth=2 to skip this method and the logging method)
        frame = inspect.currentframe().f_back.f_back
        filename = os.path.basename(frame.f_code.co_filename)
        func_name = frame.f_code.co_name
        line_no = frame.f_lineno
        return filename, func_name, line_no

    def debug(self, message):
        """Log a debug message with caller information."""
        filename, func_name, line_no = self._get_caller_info()
        self.logger.debug(f"{filename}:{func_name}:{line_no} - {message}")

    def info(self, message):
        """Log an info message with caller information."""
        filename, func_name, line_no = self._get_caller_info()
        self.logger.info(f"{filename}:{func_name}:{line_no} - {message}")

    def warning(self, message):
        """Log a warning message with caller information."""
        filename, func_name, line_no = self._get_caller_info()
        self.logger.warning(f"{filename}:{func_name}:{line_no} - {message}")

    def error(self, message, exc_info=None):
        """
        Log an error message with caller information and optional exception info.

        Args:
            message: Error message
            exc_info: Exception information (default: None)
        """
        filename, func_name, line_no = self._get_caller_info()
        if exc_info:
            self.logger.error(
                f"{filename}:{func_name}:{line_no} - {message}", exc_info=exc_info
            )
        else:
            self.logger.error(f"{filename}:{func_name}:{line_no} - {message}")

    def critical(self, message, exc_info=None):
        """
        Log a critical message with caller information and optional exception info.

        Args:
            message: Critical error message
            exc_info: Exception information (default: None)
        """
        filename, func_name, line_no = self._get_caller_info()
        if exc_info:
            self.logger.critical(
                f"{filename}:{func_name}:{line_no} - {message}", exc_info=exc_info
            )
        else:
            self.logger.critical(f"{filename}:{func_name}:{line_no} - {message}")

    def exception(self, message):
        """
        Log an exception message with full stack trace.
        Should only be called from an exception handler.

        Args:
            message: Exception message
        """
        filename, func_name, line_no = self._get_caller_info()
        self.logger.exception(f"{filename}:{func_name}:{line_no} - {message}")


class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to console output."""

    def __init__(self):
        super().__init__("%(message)s")

    def format(self, record):
        """Format the log record with appropriate colors."""
        # Get the original formatted message
        message = super().format(record)

        # Add color based on log level
        if record.levelname in RufusLogger.COLORS:
            color = RufusLogger.COLORS[record.levelname]
            return f"{color}{message}{Style.RESET_ALL}"

        return message


# Create a singleton instance to be imported by other modules
logger = RufusLogger()


# Example usage
if __name__ == "__main__":
    # Test the logger
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    try:
        # Generate an exception
        1 / 0
    except Exception as e:
        logger.exception(f"An exception occurred: {str(e)}")
        logger.critical("Critical error in application", exc_info=True)
