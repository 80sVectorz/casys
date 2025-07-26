from copy import deepcopy
from dataclasses import dataclass
import pprint
from numba import njit
from typing import Callable, Any, cast
import inspect
import ast
from typing import get_type_hints
import astpretty
from numba.experimental.structref import new
from numba.np.random.new_random_methods import buffered_bounded_bool
import numpy as np
from numpy.ma import getmask

from .debug_utils import preview, header
from .wrappers import CAKernel
from .soa import SOASchema
from rich.console import Console
from rich.syntax import Syntax
from .utils import type_to_ast_node, namespace_canonicalize_modules
from .kernel_func_handlers import  NT_KFuncParser
from .kernel_ast import K_META, KernelASTNodeMeta, copy_meta, get_meta
from .kernel_values import BASE_RESERVED_NAMES, KV_PX, KV_PY, KV_TIMESTAMP, KV_WIDTH, KV_HEIGHT, KV_LD_IDX, KV_WR_IDX
    
class KernelDescriptor:
    """Holds raw kernel function data, which will be handed off to the pipeline."""
    name: str
    func: Callable[..., Any]
    src: str
    namespace: dict[str, Any]
    hints: dict[str, Any]

    def __init__(self, wrapped_func: CAKernel) -> None:
        func = wrapped_func.func
        self.name = func.__name__
        self.func = func
        self.src = inspect.getsource(func)
        self.namespace = func.__globals__
        namespace_canonicalize_modules(self.namespace)

        self.hints = get_type_hints(func, include_extras=True)

class NV_ConstantsCollector(ast.NodeVisitor):
    consts: list[tuple[str,ast.expr]]

    def __init__(self):
        self.consts = []
        self.calls = set()
    
    def visit_Call(self, node: ast.Call):
        if isinstance(n := node.func, ast.Name) and n.id == 'k_get_const':
            args = node.args
            print(args)
            assert isinstance(c := args[1], ast.Constant) and isinstance(c.value, str)
            self.consts.append((c.value,args[0]))

@dataclass(frozen=True)
class KernelMetadata:
    desc: KernelDescriptor
    fndef: ast.FunctionDef
    soa: SOASchema
    consts: list[tuple[str, ast.expr]]

class KernelPreprocessor:
    def preprocess(self, desc: KernelDescriptor) -> KernelMetadata:
        func = desc.func
        tree = ast.parse(desc.src)    
        fndef = cast(ast.FunctionDef, tree.body[0])

        hints = get_type_hints(func, include_extras=True)

        soa_input_schema = tuple((
            (k, t.__cac_type__)
            for k,t in hints.items()
            if hasattr(t, "__cac_type__"))
        )

        soa = SOASchema(soa_input_schema)

        consts_collector = NV_ConstantsCollector()
        consts_collector.visit(fndef)

        return KernelMetadata(
            desc=desc,
            fndef=fndef,
            soa=soa,
            consts=consts_collector.consts,
        )
    
def ast_assign_get_targets(node: ast.Assign) -> list[ast.expr]:
    if isinstance(node.targets[0], ast.Tuple):
        return node.targets[0].elts
    elif isinstance(node.targets[0], ast.Name):
        return [node.targets[0]]
    return []
    
    
def ast_assign_get_value(node: ast.Assign, i: int) -> ast.expr:
    """Returns the value of the assignment at index i in the node."""
    if isinstance(node.targets[0], ast.Tuple) and (i < 0 or i >= len(cast(ast.Tuple,node.targets[0]).elts)):
        raise IndexError(f'Index {i} out of range for assignment targets, got {len(node.targets)} target(s)')
    
    value = node.value
    if isinstance(value, ast.Tuple):
        if i >= len(value.elts):
            raise IndexError(f'Index {i} out of range for assignment value tuple, got {len(value.elts)} elements')
        return value.elts[i]
    return value
    
class NV_IllegalWriteDetector(ast.NodeVisitor):
    reserved_names: set[str]

    def __init__(self, reserved_names: set[str]) -> None:
        self.reserved_names = reserved_names
    
    def visit_Name(self, node: ast.Name):
        if node.id in self.reserved_names and isinstance(node.ctx, ast.Store):
            raise ValueError(f'Illegal write to reserved name {node.id} at line {node.lineno}, column {node.col_offset}')
    
    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id in self.reserved_names and isinstance(node.ctx, ast.Store):
            raise ValueError(f'Illegal write to reserved name {node.value.id} at line {node.lineno}, column {node.col_offset}')
        
        # Visit the slice to ensure we catch any illegal writes there as well
        self.generic_visit(node)

class NV_BufferReferenceCollector(ast.NodeVisitor):
    refs: list[ast.Name]
    meta: KernelMetadata

    def __init__(self, meta: KernelMetadata) -> None:
        self.meta = meta
        self.refs = []

    def visit_Name(self, node: ast.Name):
        buffer_names = self.meta.soa.names
        assert buffer_names is not None

        if node.id in buffer_names:
            self.refs.append(node)

class NT_BufferNamesConverter(ast.NodeTransformer):
    buffers: set[str]
    meta: KernelMetadata

    def __init__(self, meta: KernelMetadata) -> None:
        self.meta = meta
        self.buffers = set()

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute | ast.Subscript | ast.Name:
        soa = self.meta.soa
        assert (names_set := soa.names) is not None

        name: str = ''
        nv = node.value
        match nv:
            case ast.Subscript():
                if (isinstance(n := nv.value, ast.Name) and (name := n.id) in names_set):
                    new_name = soa.cvt(name,node.attr)
                    self.buffers.add(new_name)

                    new_node = nv
                    new_node.value = ast.Name(new_name,n.ctx)
                    ast.copy_location(new_node,node)
                    return new_node
            
            case ast.Name():
                if ((name := nv.id) in names_set):
                    new_name = soa.cvt(name,node.attr)
                    self.buffers.add(new_name)

                    new_node = ast.Name(new_name, nv.ctx)
                    ast.copy_location(new_node,node)
                    return new_node

        return node
    
class NT_SubscriptEnforcer(ast.NodeTransformer):
    """Subscript Enforcer  
    Ensures that all buffer accesses are done through subscript notation at the current x and y.
    Only applies to cases where there's no subscript notation already present.
    """
    buffer_names: set[str]
    bypass_list: set[ast.Name]

    def __init__(self, buffer_names: set[str]) -> None:
        self.buffer_names = buffer_names
        self.bypass_list = set()

    def visit_Subscript(self, node: ast.Subscript) -> ast.Subscript:
        if isinstance(n := node.value, ast.Name) and n.id in self.buffer_names:
            # Check if the subscript has two elements, if not raise an error
            if not isinstance(node.slice, ast.Tuple) or len(node.slice.elts) not in (2,3):
                raise ValueError(f'Buffer {n.id} must be accessed with subscript notation at the current x and y coordinates, got {ast.dump(node.slice)} at line {node.lineno}, column {node.col_offset}')

            # Ensure the name is skipped since it already has subscript notation
            self.bypass_list.add(n)
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name | ast.Subscript:
        if node in self.bypass_list: return node

        if node.id in self.buffer_names:
            # If the node is a buffer name, we need to ensure it is accessed with subscript notation
            return ast.Subscript(
                value=node,
                slice=ast.Tuple(elts=[
                    ast.Name(KV_PX, ctx=ast.Load()),
                    ast.Name(KV_PY, ctx=ast.Load()),
                ]),
                ctx=ast.Load()
            )
        return node

@dataclass()
class VarDescriptor:
    name: str
    constant: bool
    ld_refs: int
    # wr_refs: number of write references after first assignment
    wr_refs: int
    # alias_of: if this variable's value is another variable that is assigned to it,
    # this will be the name of that variable
    alias_of: str | None = None 
    is_reserved: bool = False

class NV_VariablesCollector(ast.NodeVisitor):
    """Collects all user defined variables"""

    reserved_names: set[str]
    variables: dict[str, VarDescriptor]
    visited_nodes: set[ast.Name]

    def __init__(self, reserved_names: set[str]) -> None:
        self.variables = {}
        self.reserved_names = reserved_names
        self.visited_nodes = set()

    def visit_Call(self, node: ast.Call):
        [self.visit(arg) for arg in node.args]

    def visit_arg(self, node: ast.arg):
        return 

    def visit_Assign(self, node: ast.Assign) -> None:
        # We hook Assign visits so we can detect alias assignments
        # and update the variable descriptor accordingly.

        targets = ast_assign_get_targets(node)

        self.visit(node.value)

        for i,target in enumerate(targets):
            if isinstance(n := target, ast.Name) and n.id not in self.reserved_names:
                name = n.id
                if name not in self.variables:
                    # Existing variables will be handled in visit_Name
                    var_desc = VarDescriptor(
                        name=name,
                        constant=True,
                        ld_refs=0,
                        wr_refs=1,
                    )

                    value = ast_assign_get_value(node, i)

                    if isinstance(value, ast.Name):
                        var_desc.alias_of = value.id

                    self.variables[n.id] = var_desc

                    self.visited_nodes.add(n)
                else:
                    self.visit(target)
            else:
                self.visit(target)

    def visit_Name(self, node: ast.Name) -> None:
        name = node.id
        if node in self.visited_nodes: return

        if name in self.variables:
            var_desc = self.variables[name]
            if isinstance(node.ctx, ast.Load):
                var_desc.ld_refs += 1
            elif isinstance(node.ctx, ast.Store):
                var_desc.wr_refs += 1
            if var_desc.wr_refs > 1:
                var_desc.constant = False
                var_desc.alias_of = None
            self.variables[name] = var_desc

        elif name not in self.variables:
            if isinstance(node.ctx,ast.Load) and name not in self.reserved_names:
                raise ValueError(f'Variable {name} is used before assignment at line {node.lineno}, column {node.col_offset}') 

            var_desc = VarDescriptor(
                name=name,
                constant=True,
                ld_refs=1,
                wr_refs=0,
            )
            if name in self.reserved_names: var_desc.is_reserved = True

            self.variables[name] = var_desc

class NT_kvalsAliasRemover(ast.NodeTransformer):
    """Replaces variables that are aliases of any kvals or other reserved names with the original variable."""
    reserved_names: set[str]
    variables: dict[str, VarDescriptor]

    def __init__(self, reserved_names: set[str], variables: dict[str, VarDescriptor]) -> None:
        self.reserved_names = reserved_names
        self.variables = variables

    def visit_Assign(self, node: ast.Assign) -> ast.Assign | None:
        node.targets = [self.visit(t) for t in node.targets]
        node.value = self.visit(node.value)

        if not (isinstance(node.value, ast.Name) or isinstance(node.value, ast.Tuple)): return node

        targets = ast_assign_get_targets(node)
        if not targets: return node

        new_targets = []
        new_values = []

        for i, target in enumerate(targets):
            value = ast_assign_get_value(node, i)

            if not isinstance(target, ast.Name):
                new_targets.append(target)
                new_values.append(value)
                continue

            if isinstance(value, ast.Name) and target.id == value.id:
                # If the value is the same as the target, we can remove the assignment
                continue
            else:
                new_targets.append(target)
                new_values.append(value)

        if not new_targets or not new_values:
            return None

        if isinstance(node.value, ast.Tuple):
            new_node = ast.Assign(
                [ast.Tuple(elts=new_targets)],
                ast.Tuple(new_values)
            )
            return new_node

        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self.variables and node.id not in self.reserved_names:
            var_desc = self.variables[node.id]
            if var_desc.alias_of in self.reserved_names:
                node.id = var_desc.alias_of
                return ast.Name(var_desc.alias_of, ctx=node.ctx)
        return node

class NT_BoundaryWrapEnforcer(ast.NodeTransformer):
    """Ensures all elements in the subscripts for read and write operations are wrapped by a modulo operation with the width and height of the buffer.
    Skipped if the coordinates are not kval_x and kval_y.
    """
    buffer_names: set[str]
    variables: dict[str, VarDescriptor]

    def __init__(self, buffer_names: set[str], variables: dict[str, VarDescriptor]) -> None:
        self.buffer_names = buffer_names
        self.variables = variables

    def visit_Subscript(self, node: ast.Subscript) -> ast.Subscript:
        meta = get_meta(node)
        if meta is not None and meta.verified_bounds: return node

        if isinstance(n := node.value, ast.Name) and n.id in self.buffer_names:
            if isinstance(node.slice, ast.Tuple):
                new_slice = node.slice.elts[:-2]
                for i, elt in enumerate(node.slice.elts[-2:]):
                    if (
                        (isinstance(elt, ast.Name) and elt.id == (KV_PX,KV_PY)[i])
                        or (
                            isinstance(elt, ast.BinOp) and
                            isinstance(elt.op, ast.Mod) and
                            isinstance(elt.right, ast.Name) and
                            elt.right.id == (KV_WIDTH,KV_HEIGHT)[i]
                        )
                    ):
                        new_slice.append(elt)
                    else:
                        # Ensure the element is wrapped by a modulo operation
                        new_slice.append(
                            ast.BinOp(
                                left=elt,
                                op=ast.Mod(),
                                right=ast.Name((KV_WIDTH,KV_HEIGHT)[i], ctx=ast.Load())
                            )
                        )

                new_node = ast.Subscript(
                    value=ast.Name(n.id, ctx=ast.Load()),
                    slice=ast.Tuple(elts=new_slice),
                    ctx=node.ctx
                )

                copy_meta(new_node,node)
                if meta:
                    meta.verified_bounds = True

                ast.copy_location(new_node, node)
                return new_node

        return node
    
class NT_DoubleBufferIndexInserter(ast.NodeTransformer):
    """Adds the doubler buffer index to all buffer accesses. Using kval_ld_idx and kval_wr_idx based on their context"""
    buffer_names: set[str]
    assign_ctx_nodes: set[ast.Subscript]

    def __init__(self, buffer_names: set[str]) -> None:
        self.buffer_names = buffer_names
        self.assign_ctx_nodes = set()

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        for target in node.targets:
            if isinstance(n := target, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in self.buffer_names:
                self.assign_ctx_nodes.add(n)
        self.generic_visit(node)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.Subscript:
        if isinstance(n := node.value, ast.Name) and n.id in self.buffer_names:
            if (
                isinstance(node.slice, ast.Tuple)
                and len(node.slice.elts) == 3
                and isinstance(el1 := node.slice.elts[0],ast.Name)
                and el1.id in (KV_WR_IDX, KV_LD_IDX)
            ): return node

            idx_key = (KV_WR_IDX if node in self.assign_ctx_nodes else KV_LD_IDX)
            # If the node is in the assign context, we use kval_wr_idx
            new_slice = ast.Tuple(elts=[
                ast.Name(idx_key, ctx=ast.Load()),
                *cast(ast.Tuple,node.slice).elts,
            ])
            new_node = ast.Subscript(
                value=ast.Name(n.id, ctx=ast.Load()),
                slice=new_slice,
                ctx=node.ctx
            )
            copy_meta(new_node,node)
            ast.copy_location(new_node, node)
            return new_node

        return node
    
class KernelProcessor:
    kernels: dict[str, KernelMetadata]
    processed_kernels: dict[str, ast.FunctionDef]
    compiled_kernels: dict[str, Callable[..., Any]]

    def __init__(self) -> None:
        self.kernels = {}
        self.processed_kernels = {}
        self.compiled_kernels = {}

    def process(self, meta: KernelMetadata) -> None:
        assert meta.soa.names is not None and meta.soa.output_schema is not None


        fndef_processed = deepcopy(meta.fndef)
        fndef_processed.decorator_list = []


        preview(txt=astpretty.pformat(fndef_processed, show_offsets=False)+'\n')

        header('[Ensuring no illegal writes to reserved names]\n')

        buffer_names = set(meta.soa.names)
        reserved_names = BASE_RESERVED_NAMES | buffer_names | set(c[0] for c in meta.consts)
        illegal_write_detector = NV_IllegalWriteDetector(reserved_names)
        illegal_write_detector.visit(fndef_processed)

        header('[Parse kernel_util k functions]')
        kfunc_parser = NT_KFuncParser()
        kfunc_parser.visit(fndef_processed)
        preview(fndef_processed)

        header('[Buffer Name Rewrites]')
        names_converter = NT_BufferNamesConverter(meta)
        names_converter.visit(fndef_processed)
        used_buffers_list = [entry[0] for entry in meta.soa.output_schema if entry[0] in names_converter.buffers] 
        preview(fndef_processed)

        header('[Collecting Variables, and removing kval aliases]')
        var_collector = NV_VariablesCollector(reserved_names)
        var_collector.visit(fndef_processed)

        preview(txt=pprint.pformat(var_collector.variables, width = 500))
        print()

        alias_remover = NT_kvalsAliasRemover(
            reserved_names - buffer_names, # Buffer name aliases are lookups. So remove buffer_names to avoid removing lookup caching
            var_collector.variables
        )
        alias_remover.visit(fndef_processed)

        ast.fix_missing_locations(fndef_processed)
        preview(fndef_processed)

        header('[Enforcing Subscript Notation, and wrapped indexing]')
        subscript_enforcer = NT_SubscriptEnforcer(buffer_names)
        subscript_enforcer.visit(fndef_processed)

        boundary_wrap_enforcer = NT_BoundaryWrapEnforcer(buffer_names, var_collector.variables)
        boundary_wrap_enforcer.visit(fndef_processed)
        preview(fndef_processed)

        header('[Inserting Double Buffer Indices]')
        double_buffer_index_inserter = NT_DoubleBufferIndexInserter(meta.soa.names)
        double_buffer_index_inserter.visit(fndef_processed)
        preview(fndef_processed)

        header('[Finalizing function def signature]')

        # Rewrite function signature for converted buffer names
        arg_hints = {
            **{c[0]:c[1] for c in meta.consts}
        }
        arg_names = [b[0] for b in meta.soa.output_schema]
        arg_names += [
            KV_TIMESTAMP,
            KV_WIDTH, KV_HEIGHT,
            KV_PX, KV_PY,
            KV_LD_IDX, KV_WR_IDX
        ]
        arg_names += [var.name for var in var_collector.variables.values() if var.is_reserved and var.name not in arg_names]

        final_fndef_processed = ast.FunctionDef(
            fndef_processed.name,
            ast.arguments( args=[ast.arg(n, annotation=arg_hints.get(n)) for n in arg_names]),
            fndef_processed.body
        )

        ast.copy_location(final_fndef_processed, fndef_processed)
        preview(fndef_processed)

        # Store the processed function definition
        self.kernels[meta.desc.name] = meta
        self.processed_kernels[meta.desc.name] = final_fndef_processed

    def jit_kernel(self, name: str) -> Callable[..., Any]:
        """Returns the JIT compiled kernel function by name."""
        if name not in self.processed_kernels:
            raise ValueError(f'Kernel {name} has not been processed yet')

        fndef = self.processed_kernels[name]
        namespace = {**self.kernels[name].desc.namespace}

        exec(
            ast.unparse(fndef),
            namespace,
        )
        jit_kernel = njit(namespace[name], inline='always', nogil=True, fastmath=True, parallel=True)
        self.compiled_kernels[name] = jit_kernel

        return self.compiled_kernels[name]
    
    def jit_kernels(self) -> dict[str, Callable[..., Any]]:
        """Returns a dictionary of all JIT compiled kernel functions."""
        return {name: self.jit_kernel(name) for name in self.processed_kernels.keys()}