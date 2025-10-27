from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    from casys.viz.gui.main_window import MainWindow
    from casys.sim_manager import SimManager
    from casys.viz.gui.overlay import OverlayWidget
    from casys.viz.gui.canvas_widget import CanvasWidget

type tool_kind = Literal['modal', 'oneshot']

# modifier bit flags
MOD_SHIFT = 1 << 0
MOD_CTRL  = 1 << 1
MOD_ALT   = 1 << 2
MOD_META  = 1 << 3

@dataclass
class ToolEvent:
    """Normalized pointer/key events for tools."""
    kind: Literal['down', 'move', 'up', 'key']
    gpos: tuple[int,...]
    modifiers: int = 0
    button: int = 0
    key: int = 0

    _default_prevented: bool = False  # Internal flag to track default behavior prevention

    def prevent_default(self) -> None:
        """Mark this event as having its default action prevented."""
        self._default_prevented = True

    @property
    def default_prevented(self) -> bool:
        """Check if the default action has been prevented."""
        return self._default_prevented

@dataclass
class ToolInfo:
    """Public info for UI to render buttons/menus."""
    name: str
    kind: tool_kind

    group: str = 'default'
    order: int = 0
    tooltip: str = ''
    icon_id: str = ''

class ToolContext:
    """Stable services,  available to tools."""

    window: MainWindow
    canvas: CanvasWidget
    overlay: OverlayWidget
    sim_mgr: SimManager
    tool_mgr: ToolManager

    def __init__(
        self, *, 
        window: MainWindow,
        canvas: CanvasWidget,
        overlay: OverlayWidget,
        sim_mgr: SimManager,
    ) -> None:
        self.window = window
        self.canvas = canvas
        self.overlay = overlay
        self.sim_mgr = sim_mgr
    
class ToolPlugin:
    """Implement a tool by subclassing."""
    name: str = 'Unnamed'
    kind: tool_kind = 'oneshot'

    # Opt-in: keep a oneshot active across repeated clicks while Shift is held
    shift_multi_use: bool = False

    def on_activate(self, ctx: ToolContext) -> None: ...
    def on_deactivate(self) -> None: ...
    def on_event(self, ev: ToolEvent, ctx: ToolContext) -> None: ...
    def on_cancel(self, ctx: ToolContext) -> None: ...

class ToolManager:
    """Single source of truth for tools and routing."""
    def __init__(self, ctx: ToolContext) -> None:
        ctx.tool_mgr = self
        self._ctx = ctx
        self._tools: dict[str, ToolPlugin] = {}
        self._infos: dict[str, ToolInfo] = {}
        self._active_modal: str | None = None
        self._active_oneshot: str | None = None
        self._return_to_modal: str | None = None
        self._listeners: list[Callable] = []

    # ---- registration
    def register(self, plugin: ToolPlugin, *, group: str = 'default', order: int = 0, tooltip: str = '', icon_id: str = '') -> None:
        name = plugin.name
        if name in self._tools:
            raise ValueError(f'Tool already registered: {name}')
        self._tools[name] = plugin
        self._infos[name] = ToolInfo(name=name, kind=plugin.kind, group=group, order=order, tooltip=tooltip, icon_id=icon_id)
        self._notify()

    def unregister(self, name: str) -> None:
        if name in self._tools and self._active_modal == name:
            self.deactivate()
        self._tools.pop(name, None)
        self._infos.pop(name, None)
        self._notify()

    # ---- queries for UI
    def tools(self) -> list[ToolInfo]:
        return sorted(self._infos.values(), key=lambda i: (i.group, i.order, i.name.lower()))

    def active_tool(self) -> str | None:
        return self._active_modal

    def on_changed(self, cb: Callable) -> None:
        self._listeners.append(cb)

    def is_oneshot_active(self) -> bool:
        return self._active_oneshot is not None

    def has_active(self) -> bool:
        return (self._active_modal is not None) or (self._active_oneshot is not None)

    # ---- activation
    def activate_modal(self, name: str) -> None:
        self.deactivate()
        self._active_modal = name
        self._tools[name].on_activate(self._ctx)
        self._notify()

    def use_oneshot(self, name: str) -> None:
        self._return_to_modal = self._active_modal
        self.deactivate()
        self._active_oneshot = name
        self._tools[name].on_activate(self._ctx)
        # auto-return after receiving an 'up' in route()

    def deactivate(self) -> None:
        cur = self._active_modal
        if cur and cur in self._tools:
            self._tools[cur].on_deactivate()
        self._active_modal = None
        self._notify()

    # ---- routing from canvas
    def route(self, ev: ToolEvent) -> None:
        # route to active modal or active oneshot
        target: ToolPlugin | None = None
        if self._active_modal and self._active_modal in self._tools:
            target = self._tools[self._active_modal]
        elif self._active_oneshot and self._active_oneshot in self._tools:
            target = self._tools[self._active_oneshot]

        if target is None:
            return
        target.on_event(ev, self._ctx)

        if target.kind == 'oneshot' and ev.kind == 'up':
            keep = bool(getattr(target, 'shift_multi_use', False) and (ev.modifiers & MOD_SHIFT))
            if not keep:
                target.on_deactivate()
                self._active_oneshot = None
                if self._return_to_modal:
                    self.activate_modal(self._return_to_modal)
                    self._return_to_modal = None

    def cancel(self) -> None:
        if self._active_oneshot and self._active_oneshot in self._tools:
            t = self._tools[self._active_oneshot]
            t.on_cancel(self._ctx)
            t.on_deactivate()
            self._active_oneshot = None
            if self._return_to_modal:
                self.activate_modal(self._return_to_modal)
                self._return_to_modal = None
            self._notify()
            return
        if self._active_modal and self._active_modal in self._tools:
            self._tools[self._active_modal].on_cancel(self._ctx)

    def _notify(self) -> None:
        for cb in list(self._listeners):
            cb()