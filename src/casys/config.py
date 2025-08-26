from __future__ import annotations
from dataclasses import dataclass, field
import inspect
import pathlib
import sys
import time
from typing import Literal

class CasysConfig:
    """ Configuration for the CASys Transpiler.
    The CONFIG singleton can be modified before any Casys components are used.

    :param debug_dynsrc_mode: Level of generated code file storage.

    :param debug_jit_disable_parallel: Disables numba JIT parallelization
    :param debug_jit_nopython: Default = True, Wether or not JIT should use nopython mode.
    :param debug_disable_jit: Allows easy debugging by disabling numba JIT
     
    :param debug_ast_origin_tracking: Enable AST origin tracking for debugging. 
        This will allow error messages to include original source code locations.

    :param debug_files_directory: The directory where debug files will be saved.

    :param debug_ast_timeline: Enable AST timeline tracking.  
        This will create debug files that can be viewed using the CasysASTDebugger tool.
    :param debug_timeline_file_directory: The directory where timeline files will be saved.
    """

    debug_dynsrc_mode: Literal['off', 'virtual', 'mirror'] = 'virtual'

    debug_disable_cpu_parallelization: bool = False
    debug_jit_nopython: bool = True
    debug_disable_jit: Literal['full', 'step_func', False] = False

    debug_ast_origin_tracking: bool = True

    _debug_files_dir: pathlib.Path | None = None

    @property
    def debug_files_dir(self) -> pathlib.Path:
        if self._debug_files_dir: return self._debug_files_dir
        self._debug_files_dir = get_entry_point_path().parent / '__casys_debug__'
        self._debug_files_dir.mkdir(exist_ok=True)
        return self._debug_files_dir

    @debug_files_dir.setter
    def debug_files_dir(self, v: pathlib.Path):
        self._debug_files_dir = v

    debug_ast_timeline: bool = False
    _debug_timeline_file: pathlib.Path | None = None
    @property
    def debug_timeline_file(self) -> pathlib.Path:
        if self._debug_timeline_file: return self._debug_timeline_file
        path = self.debug_files_dir / 'ast_timelines'
        path.mkdir(exist_ok=True)
        file_name = f'{int(time.time())}.json'
        if sys.argv and sys.argv[0] and sys.argv[0] != '-c':
            file_path = pathlib.Path(sys.argv[0])
            if path.exists():
                file_name = f'{file_path.name.split('.')[0]}.json'
        self._debug_timeline_file = path / file_name
        return self._debug_timeline_file

    @debug_timeline_file.setter
    def debug_timeline_file(self, v: pathlib.Path):
        self._debug_timeline_file = v

def get_entry_point_path() -> pathlib.Path:
    """Try to determine the original project entry point file path."""
    # 1. Check if running as a script (sys.argv[0] is a file)
    if sys.argv and sys.argv[0] and sys.argv[0] != '-c':
        path = pathlib.Path(sys.argv[0])
        if path.exists():
            return path.resolve()

    # 2. Fallback: walk the call stack for the outermost user file
    for frame_info in reversed(inspect.stack()):
        frame_path = pathlib.Path(frame_info.filename)
        # skip stdlib, site-packages, and <string>/<stdin>
        if (
            frame_path.exists()
            and 'site-packages' not in frame_path.parts
            and 'lib' not in frame_path.parts
            and not frame_path.name.startswith('<')
        ):
            return frame_path.resolve()

    # 3. As a last resort, use CWD
    return pathlib.Path.cwd()


CASYS_CONFIG = CasysConfig()