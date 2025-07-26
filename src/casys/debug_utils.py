import ast
from rich.console import Console
from rich.syntax import Syntax


RC = Console()
def header(*txt):
    RC.print(f"[bold blue]{'\n'.join(txt)}")

def preview(node: ast.AST | None = None, txt: None | str = None):
    RC.print(Syntax(txt if txt else ast.unparse(node), 'python', theme='solarized-dark', background_color='#1a1b26', word_wrap=True, line_numbers=True))