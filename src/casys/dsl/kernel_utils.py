from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal, Sequence, cast

if TYPE_CHECKING:
    from casys.spec.cac_type import cact_field
    from casys.wrappers import CaCellTypeSpec

import numpy as np

from casys.dsl._core.core_macros import MacroSpec

@MacroSpec(required=(), optional=())
def step_func_split():
    """
    Dummy function for denoting kernel grouping in CA step function definitions.
    """
    ...

@MacroSpec(required=('layers',), optional=())
def step_func_swap(layers: list[CaCellTypeSpec | object]):
    """
    Dummy function for denoting double buffer swap of `layers`.
    Tells the transpiler that the SoA field buffers of the given layers need to be swapped
    before any new kernel calls are made.

    **Notes**:
        The swap placement will be optimized by the transpiler.
        This is done by for example analyzing where SoA field buffers are actually used.
    """
    ...

@MacroSpec(required=('expr',), optional=())
def k_eval[T](
    expr: T,
) -> T:
    """
    Evaluates the expression at transpilation time as a string and inserts the result.

    :param expr: Any python expression that will evaluate to a single value.
    """
    return cast(T, 0)

@MacroSpec(required=(), optional=())
def k_get_timestamp() -> int:
    """Returns the current timestamp

    :return: current simulation timestamp
    :rtype: int
    """
    return 0

@MacroSpec(required=(), optional=())
def k_get_pos() -> Sequence[int]:
    """Returns the cell position per axis for this kernel instance
    This function automatically marks the target variables as positions.

    :return: x-pos, y-pos, z-pos, etc. Depends on n_dims
    :rtype: Sequence[int]
    """
    return 0,0,0,0

@MacroSpec(required=(), optional=())
def k_get_dims() -> Sequence[int]:
    """Returns the simulation-grid dimensions

    **Note**:
        This function freezes the target variables
    """
    return 0,0,0,0


@MacroSpec(required=('scalar_type', 'name'), optional=())
def k_get_const[DT: np.generic](
    scalar_type: type[DT],
    name: str,
) -> DT:
    """
    Returns the requested simulation constant

    :param scalar_type: The numpy scalar type of the constant (included for static type checkers)
    """
    return scalar_type(0)

@MacroSpec(
    required=(),
    optional=('args',),
)
def k_mark_pos(*args: int):
    """
    Used to notify the transpiler that the target variables are position values.
    The transpiler will automatically insert logic to ensure positions are within the simulation bounds.
    By default the axis that each variable represents is inferred automatically. 
    But axis mapping can be specified using `*args`.

    Note:
        Bounds logic may add extra overhead. `k_assure_bounds()` can be used to skip bounds logic insertion.
        Only use this when it's certain that the position stays within the bounds.

    Examples: 
        >>> # Same line:
        next_x,next_y = x+dx, y+dy ; k_mark_coords()
        # Or on new line:
        next_x,next_y = x+dx, y+dy
        k_mark_coords()
    """

@MacroSpec(
    required=(),
    optional=(),
)
def k_assure_bounds():
    """
    Used to assure the transpiler that position marked variables modifications are within sim bounds.
    Should be placed directly after the variable assign line.

    Examples: 
        >>> # Same line:
        next_x,next_y = x+dx, y+dy ; k_assure_bounds()
        # Or on new line:
        next_x,next_y = x+dx, y+dy
        k_assure_bounds()
    """

@MacroSpec(
    required=('op', 'buffer', 'x', 'y', 'width', 'height'),
    optional=('weights',)
)
def k_patch_op(
    op: Literal['sum', 'mean', 'product', 'bit_or', 'logical_or', 'logical_and'],
    buffer: np.ndarray | cact_field[Any, Any],
    x: int,
    y: int,
    width: int,
    height: int,
    weights: list[list[int | float]] | None = None
) -> int | float | bool:
    """Performs an operation on a patch of the buffer.
    Acts as snippet generating dummy function, requires static parameters

    **Note**:
        Center value is not included in auto generated weight matrix

    Args:
        op: The operation to perform, one of 'sum', 'mean', 'product', etc
        buffer: The buffer to perform the operation on
        x: The x position of the patch center
        y: The y position of the patch center
        width: The width of the patch
        height: The height of the patch
        weights: Optional weights to apply to the patch, defaults to 1 for all Moore-neighbors.

    Returns:
        The result of the patch operation
    """
    return 0 # dummy result


@MacroSpec(required=('expr','x','y'), optional=('d',))
def k_neighbor_mask(
    expr: np.ndarray | Any,
    x: int,
    y: int,
    d: int = 0,
) -> np.uint8:
    """
    Returns an 8-bit mask of the 8 Moore-neighbors of (x,y) in `buffer`.

    Bits (LSB=bit 0) map as:
       bit 0 (1<<0): N
       bit 1 (1<<1): NE
       bit 2 (1<<2): E
       bit 3 (1<<3): SE
       bit 4 (1<<4): S
       bit 5 (1<<5): SW
       bit 6 (1<<6): W
       bit 7 (1<<7): NW

    Each bit is set if the corresponding neighbor value != 0.

    :param expr: Any boolean expression. The transpiler will correctly handle placeholders like x and y
    :param x: The x position of the patch
    :param y: The y position of the patch
    :param d: Use walrus operator to create placeholder that will be treated as the current Moore neighbor direction
    """
    return expr[np.uint8(0+x+y)]

@MacroSpec(required=('snippet_call',))
def k_snippet[T](snippet_call: T) -> T:
    """
    Allows for the insertion of user defined snippets.
    The `snippet_call` argument should be a call expression.
    It will automatically insert the body of the function you called.
    Any arguments you pass will be treated as find & replace targets.
    They'll be copied directly instead of being evaluated as python code.

    Example input:
        >>> def snippet(x,y):
            x * y
        # Somewhere in a @ca_kernel function:
        y = k_snippet(snippet((1+2),2))

    Resulting code:
        >>> y = (1+2) * 2

    """
    return snippet_call