from __future__ import annotations

from .gui.overlay import OverlayWidget
from .tools.registry import ToolRegistry
from .tools.uimodel_binder import bind_registry_to_ui

# Optional: import and register built-in tools here (kept minimal on purpose)

def wire_up_viz_extensions(window) -> None:
    """Attach overlay and bind registered tools to the existing UIModel."""
    overlay = OverlayWidget(canvas=window.canvas, parent=window.canvas.native)
    overlay.raise_()
    window.overlay = overlay  # type: ignore

    registry = ToolRegistry()
    # Register built-ins if available

    bind_registry_to_ui(window, window.ui_model, registry)

    window._viz_extensions = {'overlay': overlay, 'tool_registry': registry}
