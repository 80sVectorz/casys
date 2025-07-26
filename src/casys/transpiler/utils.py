import ast
import types

def type_to_ast_node(t: type) -> ast.expr:
    if hasattr(t, '__module__') and hasattr(t, '__name__'):
        return ast.Attribute(
            value=ast.Name(id=t.__module__.split('.')[-1], ctx=ast.Load()),
            attr=t.__name__,
            ctx=ast.Load()
        )
    raise ValueError(f"Cannot convert type {t} to AST")

def namespace_canonicalize_modules(namespace: dict[str, object]) -> None:
    """
    Ensures all imported modules in the given namespace are also available under their canonical name.
    
    :param namespace: A dict representing the variable scope, e.g. globals() or locals().
    """
    for alias, obj in list(namespace.items()):
        # Check if the object is a module
        if isinstance(obj, types.ModuleType):
            # Canonical name of the module
            canonical_name = obj.__name__.split('.')[0]  # Handle submodules like numpy.typing
            
            # If the canonical name isn't in the namespace, assign it
            if canonical_name not in namespace:
                namespace[canonical_name] = obj