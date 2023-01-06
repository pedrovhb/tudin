"""
Use libcst to generate an __all__ variable for a given Python file.

The code analyzes the AST of the file and provides a list of all classes, functions,
and top-level variables which are defined in the file (that is, were not imported from another
file or module).
"""
import argparse
import difflib
import json
import sys
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from pathlib import Path

from rich_argparse import RichHelpFormatter
import black
import libcst as cst
from rich.console import Console, ConsoleOptions, RenderResult
from rich.padding import Padding
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Column, Table

_console = Console()
__version__ = "0.1.0"


def get_all_exports(file: Path, include_private: bool = False) -> tuple[str, ...]:
    """Get the list of all exports from the file.

    The returned list is sorted alphabetically. Names that start with an underscore are not included
    unless `include_private` is True. Names that are imported from other modules are not included.

    Args:
        file: The file to analyze.
        include_private: Whether to include names that start with an underscore.

    Returns:
        A tuple of all exports from the file.

    Examples:
        >>> get_all_exports(Path("tudin/__main__.py"))
        ('AllExportsDisplay', 'interactive_change_all_vars', 'get_all_exports')
    """
    code = file.read_text()
    module = cst.parse_module(code)
    names_set: set[str] = set()
    for node in module.body:
        if isinstance(node, cst.ClassDef):
            names_set.add(node.name.value)
        elif isinstance(node, cst.FunctionDef):
            names_set.add(node.name.value)
        elif isinstance(node, cst.SimpleStatementLine):
            for ss_node in node.body:
                for target in ss_node.targets if isinstance(ss_node, cst.Assign) else ():
                    if isinstance(target, cst.AssignTarget) and isinstance(target.target, cst.Name):
                        names_set.add(target.target.value)

    if not include_private:
        names = (name for name in names_set if not name.startswith("_"))
    else:
        names = names_set

    sorted_names = sorted(names, key=lambda name: name.lower())
    return tuple(sorted_names)


@dataclass(frozen=True)
class ExportsDisplay:
    """
    Class which pretty prints the code for a file.

    Uses the @group decorator to yield items which fit into a Panel. One of these is a Syntax
    object with the text "__all__ = (...)
    """

    file: Path
    include_private: bool = False

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield f" [bold green]{'=' * (len(str(self.file)) + 8)}[/]"
        non_bold, bold = self.file.parent, self.file.name
        yield f"  [bold green]==[/] [blue]{non_bold}/[bold]{bold}[/] [bold green]=="
        yield ""
        yield self.get_syntax()

    def get_syntax(self) -> Syntax:
        return Syntax(self.code, "python", background_color="default")

    @property
    def code(self) -> str:
        if len(self.all_names) == 0:
            names_str = "__all__ = ()"
        elif len(self.all_names) == 1:
            names_str = f"__all__ = ({self.all_names[0]!r},)"
        else:
            names_str = f"__all__ = ({', '.join(map(repr, self.all_names))})"

        return black.format_str(
            names_str,
            mode=black.Mode(
                target_versions={black.TargetVersion.PY310},
                magic_trailing_comma=True,
            ),
        )

    @cached_property
    def all_names(self) -> tuple[str, ...]:
        return get_all_exports(self.file, self.include_private)

    @classmethod
    def from_file(cls, file: str | Path) -> "ExportsDisplay":
        return cls(Path(file))

    def get_file_code(self) -> str:
        return self.file.read_text()

    def get_current_file_all(self) -> cst.SimpleStatementLine | None:
        """Get the current __all__ variable from the file."""
        module = cst.parse_module(self.get_file_code())
        return next(
            (
                node
                for node in module.body
                if isinstance(node, cst.SimpleStatementLine)
                and isinstance(ass := node.body[0], cst.Assign)
                and isinstance(name_node := ass.targets[0].target, cst.Name)
                and name_node.value == "__all__"
            ),
            None,
        )

    def data(self) -> dict:
        return {
            "file": str(self.file),
            "include_private": self.include_private,
            "code": self.code,
            "all_names": self.all_names,
        }

    def json(self) -> str:
        return json.dumps(self.data())


def _interactive_change_all_vars(file: str | Path, dry_run: bool = True) -> None:
    """Go through given files one by one, displaying the calculated __all__ exports,
    and the __all__ exports currently present in the file, if any. Prompt the user to apply the
    __all__ export if the file does not have one, or replace the current one if it does.

    Todo - rework this function to be more readable and modular.
    """

    display = ExportsDisplay.from_file(file)

    if not display.all_names:
        _console.print(f"[red]No exportable names found in [bold blue]{file}[/]")
        return

    module = cst.parse_module(display.get_file_code())
    current_all_statement = display.get_current_file_all()
    proposed_all = cst.parse_module(ExportsDisplay.from_file(file).code).body[0]
    # print(proposed_all)
    proposed_all_str = cst.Module(body=[proposed_all]).code.strip()

    if current_all_statement is None:
        _console.print(f"The file [bold]{file}[/] does not currently have an __all__ variable.\n")
        _console.print("Calculated __all__ exports:")
        _console.print(Syntax(proposed_all_str, "python", background_color="default"))

        user_input = _console.input("Would you like to set this as the __all__ variable? [Y/n] ")
        if user_input.lower().strip() in ("y", "", "yes"):
            if dry_run:
                _console.print(
                    Syntax(
                        display.code + "\n\n" + proposed_all_str,
                        "python",
                        background_color="default",
                    )
                )
            else:
                with file.open("a") as f:
                    f.write("\n\n" + proposed_all_str)
            _console.print(f"Added __all__ variable to [bold]{file}[/]")
        else:
            _console.print(f"Skipped [bold]{file}[/]")

    else:  # File already has an __all__ variable
        current_all_str = cst.Module(body=[current_all_statement]).code.strip()

        if proposed_all_str == current_all_str:
            _console.print(f"[green]No changes proposed to [bold blue]{file}[/]")
            return

        _console.print(f"The file [bold]{file}[/] currently has an __all__ variable.")

        current_all_str = black.format_str(current_all_str, mode=black.Mode(line_length=20))
        proposed_all_str = black.format_str(proposed_all_str, mode=black.Mode(line_length=20))

        syntax_current = Syntax(current_all_str, "python", background_color="default")
        syntax_proposed = Syntax(proposed_all_str, "python", background_color="default")

        diff_str = "\n".join(
            difflib.ndiff(
                current_all_str.splitlines(),
                proposed_all_str.splitlines(),
            )
        )

        syntax_diff = Syntax(
            diff_str,
            "diff",
            background_color="default",
        )
        table = Table.grid(
            Column("Current", no_wrap=True),
            Column("Proposed", no_wrap=True),
            Column("Diff", no_wrap=True),
            padding=(1, 4),
        )
        table.show_header = True
        _console.print(syntax_diff)

        table.add_row(syntax_current, syntax_proposed, syntax_diff)

        # Make code wrap
        table._column_widths = [None, None, None]

        _console.print(
            Padding(
                Panel(table, title=str(file), title_align="left", padding=(2, 2)),
                (1, 2),
            )
        )

        _console.print()

        user_input = _console.input(
            "Would you like to replace the current __all__ variable? [Y/n] "
        )
        if user_input.lower().strip() in ("y", "yes", ""):
            module = module.deep_replace(current_all_statement, proposed_all)

            if dry_run:
                _console.print(
                    Syntax(
                        module.code,
                        "python",
                        background_color="default",
                    )
                )
            else:
                with file.open("w") as f:
                    # f.write(module.code + "\n")
                    f.write(
                        module.code.replace(
                            cst.Module(body=[current_all_statement]).code,
                            f"\n\n{proposed_all_str}\n",
                        )
                    )

    _console.print("\n")


def _app() -> None:
    """Main application function."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate __all__ variables for a given Python file, "
            "or all Python files in a directory."
        ),
        formatter_class=RichHelpFormatter,
        prog="tudin",
    )
    parser.add_argument(
        "input_filenames",
        metavar="filename",
        type=Path,
        nargs="+",
        help="Python file or directory to generate __all__ variables for.",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        help="Recursively search for Python files in the given directories.",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    mutually_exclusive = parser.add_mutually_exclusive_group()
    mutually_exclusive.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactively replace __all__ variables in given files in-place.",
    )
    mutually_exclusive.add_argument(
        "--inplace",
        "--in-place",
        "-I",
        action="store_true",
        help="Replace __all__ variables in given files in-place with no interaction.",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Do not write to files when using --interactive.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--include-private",
        "-p",
        action="store_true",
        help="Include private variables in __all__.",
    )
    mutually_exclusive.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output JSON instead of Python code.",
    )
    parser.add_argument(
        "--raw",
        "-R",
        action="store_true",
        help="Do not pretty-print output.",
    )
    parser.add_argument(
        "--exclude",
        "-e",
        type=Path,
        nargs="*",
        help="Exclude files or directories from being processed.",
        default=["test", "tests", "venv"],
        action="append",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    input_paths: set[Path] = set(args.input_filenames)
    try:
        input_paths.remove(Path("-"))
    except KeyError:
        pass
    else:
        for line in sys.stdin:
            input_paths.add(Path(line.strip()))

    paths = []
    for path in input_paths:
        if path.is_dir():
            for file in path.glob("**/*.py" if args.recursive else "*.py"):
                paths.append(file)
        elif path.is_file():
            paths.append(path)

    prev_paths_len = len(paths)
    exclude = [Path(p) for p in chain.from_iterable(args.exclude)]
    paths = [p for p in paths if not any(p.is_relative_to(e) for e in exclude)]
    if len(paths) != prev_paths_len:
        _console.print(
            f"Excluded {prev_paths_len - len(paths)} paths from processing.",
            style="yellow",
        )

    if not paths:
        _console.print(
            "No Python files matched the given input.\n"
            "Check the arguments or consider using --recursive/-r.",
            style="red",
        )
        sys.exit(1)

    def output_data(display: ExportsDisplay) -> None:

        if args.json:
            if args.raw:
                sys.stdout.write(display.json())
            else:
                _console.print_json(display.json())
        elif args.inplace:
            display.file.write_text(display.code)
        else:
            if args.raw:
                sys.stdout.write(f"# {display.file}\n{display.code}\n\n")
            else:
                _console.print(display)

    # Process the list of Path objects
    for path in paths:
        if args.interactive:
            _interactive_change_all_vars(path, args.dry_run)
        else:
            exports = ExportsDisplay.from_file(path)
            output_data(exports)


if __name__ == "__main__":
    _app()


__all__ = (
    "ExportsDisplay",
    "get_all_exports",
)
