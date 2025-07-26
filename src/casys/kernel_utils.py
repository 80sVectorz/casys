
import ast
from dataclasses import dataclass, field
from typing import Callable, Any, Literal
import numpy as np

def step_func_split():
    """
    Dummy function for denoting kernel grouping in CA step function definitions.
    """
    ...

@dataclass
class Spec:
    """Decorator to attach a simple signature spec (required and optional args) to dummy functions.

    :param required: tuple of required argument names
    :param optional: tuple of optional argument names
    """
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()

    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        fn._spec_required = self.required # type: ignore
        fn._spec_optional = self.optional # type: ignore
        return fn

    @classmethod
    def parse(cls, fn: Callable[..., Any], call: ast.Call) -> dict[str, ast.expr]:
        """Parse an AST Call node into a dict mapping argument names to AST expr nodes.

        :param fn: function previously decorated with @spec
        :param call: AST Call node
        :return: mapping from argument names to AST expression nodes
        :raises ValueError: if there are missing required args, too many positionals, unexpected or duplicated keywords
        """
        required = getattr(fn, '_spec_required', ())
        optional = getattr(fn, '_spec_optional', ())
        names = list(required) + list(optional)
        args_dict: dict[str, ast.expr] = {}

        # positional arguments
        for i, arg in enumerate(call.args):
            if i >= len(names):
                raise ValueError('Too many positional arguments')
            args_dict[names[i]] = arg

        # keyword arguments
        for kw in call.keywords:
            if kw.arg not in names:
                raise ValueError(f'Unexpected keyword argument {kw.arg!r}')
            if kw.arg in args_dict:
                raise ValueError(f'Duplicate argument {kw.arg!r}')
            args_dict[kw.arg] = kw.value

        # ensure all required present
        missing = [n for n in required if n not in args_dict]
        if missing:
            raise ValueError(f'Missing required arguments {missing}')
        return args_dict
    

@Spec(required=(), optional=())
def k_get_timestamp() -> int:
    """Returns the current timestamp

    :return: current simulation timestamp
    :rtype: int
    """
    return 0


@Spec(required=(), optional=())
def k_get_pos() -> tuple[int, int]:
    """Returns the x,y cell position for this kernel instance

    :return: x-pos, y-pos
    :rtype: tuple[int, int]
    """
    return 0, 0


@Spec(required=(), optional=())
def k_get_dims() -> tuple[int, int]:
    """Returns the simulation grid's width and height

    :return: width, height
    :rtype: tuple[int, int]
    """
    return 0, 0

def k_get_wr_idx() -> int:
    """Returns the current double-buffer write target index

    Returns:
        int: index of write buffer (0 or 1)
    """
    return 0


@Spec(required=('scalar_type', 'name'), optional=())
def k_get_const[DT: np.generic](
    scalar_type: type[DT],
    name: str
) -> DT:
    """Returns the requested simulation constant

    :param scalar_type: The numpy scalar type of the constant (included for static type checkers)
    """
    return scalar_type(0)


@Spec(
    required=('op', 'buffer', 'x', 'y', 'width', 'height'),
    optional=('mask',)
)
def k_patch_op(
    op: Literal['sum', 'mean', 'product', 'bit_or'],
    buffer: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    mask: list[list[Literal[0, 1]]] | None = None
) -> int | float:
    """Performs an operation on a patch of the buffer.
    Acts as snippet generating dummy function, requires static parameters

    :param op: The operation to perform, one of 'sum', 'mean', or 'product'
    :param buffer: The buffer to perform the operation on
    :param x: The x position of the patch
    :param y: The y position of the patch
    :param width: The width of the patch
    :param height: The height of the patch
    :param mask: Optional mask to apply to the patch, defaults to None
    :return: The result of the patch operation
    """
    return np.sum(buffer[x:x+width, y:y+height])  # dummy result


@Spec(required=('expr','x','y'), optional=('d',))
def k_neighbor_mask(
    expr: np.ndarray,
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