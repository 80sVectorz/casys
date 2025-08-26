import logging
from typing import cast
from rich.logging import RichHandler
from rich.console import Console
from rich.syntax import Syntax
from rich.traceback import install

# Install rich traceback handler for pretty tracebacks
install(show_locals=True)

console = Console()

# ——————————————
# Logging Setup
# ——————————————

def setup_logging(verbose: bool = False) -> None:
    """
    Configure the root logger to use RichHandler and set level based on verbose flag.

    Args:
        verbose: If True, set level to DEBUG, else INFO.
    """
    logger = logging.getLogger('casys')
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[RichHandler(rich_tracebacks=True, console=console)]
    )


def log_warning(message: str) -> None:
    """
    Log a warning-level message.

    Args:
        message: The message to log.
    """
    if logging.getLogger('casys').level <= logging.WARN:
        console.print(f'[bold yellow]WARNING: [/bold yellow]{message}')

def log_debug(message: str) -> None:
    """
    Log a debug-level message.

    Args:
        message: The message to log.
    """
    if logging.getLogger('casys').level <= logging.DEBUG:
        console.print(message)

# ——————————————
# Exception types & error display utilities
# ——————————————

class DefinitionException(Exception):
    """
    Base exception for errors that occur before preprocessing or transpiling.  
    For example while applying the `@cac_type` decorator.

    Attributes:
        message: Human-readable error message.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

class PreprocessError(Exception):
    """
    Base exception for errors encountered during preprocessing.

    Attributes:
        message: Human-readable error message.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

class TranspileError(Exception):
    """
    Base exception for errors encountered during transpilation.

    Attributes:
        message: Human-readable error message.
        code: The source code where the error occurred.
        line: Line number of the error (1-based).
        column: Column number of the error (1-based).
    """

    def __init__(self, message: str, code: str, line: int, column: int) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.line = line
        self.column = column


def display_error(err: TranspileError | PreprocessError | DefinitionException, context_radius: int = 2) -> None:
    """
    Render a transpilation error with syntax-highlighted snippet and pointer to the error location.

    Args:
        err: The TranspileError or PreprocessError instance.
        context_radius: Number of lines of context around the error line.
    """

    base_msg = f"[bold red]Error:[/] {err.message}"
   
    match err:
        case TranspileError():
            lines = err.code.splitlines()
            start = max(err.line - context_radius - 1, 0)
            end = min(err.line + context_radius, len(lines))
            snippet = "\n".join(lines[start:end])

            syntax = Syntax(
                snippet,
                'python',
                start_line=start + 1,
                highlight_lines={err.line},
                line_numbers=True,
                word_wrap=True,
                theme='solarized-dark', background_color='#1a1b26'
            )
            final_msg = base_msg + f'(line {err.line}, col {err.column})'

            console.print(final_msg)
            console.print(syntax)
            pointer = ' ' * (err.column + len(str(err.line)) + 1) + '^'
            console.print(pointer)

        case PreprocessError():
            console.print(base_msg)

        case DefinitionException:
            base_msg = f"[bold red]Exception:[/] {err.message}"
            console.print(base_msg)

# Example usage:
# setup_logging(verbose=True)
# try:
#     # transpilation logic...
#     raise TranspileError('Unexpected token', code_str, 10, 5)
# except TranspileError as e:
#     display_error(e)
