from dataclasses import dataclass
from typing import Literal
from PySide6 import QtCore, QtGui, QtWidgets


@dataclass
class RectItem:
    """Represents a single grid-aligned rectangle with an optional label."""
    gid: int
    gx: int
    gy: int
    w: int
    h: int
    label: str | None = None
    rgba: tuple[int, int, int, int] = (0, 255, 0, 180)
    z: int = 0


@dataclass
class NoteItem:
    """Represents a small pinned bubble of text near a grid position."""
    gid: int
    gx: int
    gy: int
    text: str
    rgba: tuple[int, int, int, int] = (20, 20, 20, 200)
    fg: tuple[int, int, int] = (240, 240, 240)
    z: int = 10


@dataclass
class PinnedWidgetItem:
    """A real QWidget pinned to a grid anchor."""
    gid: int
    gx: float
    gy: float
    widget: QtWidgets.QWidget
    mode: str = 'fixed'  # 'fixed' or 'canvas'
    w_cells: float = 3.0  # used when mode == 'canvas'
    h_cells: float = 1.5
    px_size: tuple[int, int] = (180, 96)  # used when mode == 'fixed'
    offset_px: tuple[int, int] = (10, 10)
    anchor_corner: Literal['tl', 'tr', 'bl', 'br', 'auto'] = 'auto'
    z: int = 20


@dataclass
class _DragState:
    """Internal pointer-sized drag info keyed by widget id()."""
    start_global: QtCore.QPoint
    start_geom: QtCore.QRect
    resizing: bool


class OverlayWidget(QtWidgets.QWidget):
    """
    Transparent painter sitting above the GL canvas.

    Uses a stateful tile selection per gid so widgets/rects/notes stay in the
    same torus tile until the viewport shows exactly one different tile.
    """

    def __init__(
        self,
        canvas,
        parent: QtWidgets.QWidget | None = None,
        widgets_edge_policy: Literal['clip', 'clamp'] = 'clip',
    ) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)
        self.raise_()

        self._rects: dict[int, RectItem] = {}
        self._notes: dict[int, NoteItem] = {}
        self._widgets: dict[int, PinnedWidgetItem] = {}
        self._widget_index: dict[int, int] = {}
        self._drag_states: dict[int, _DragState] = {}
        self._resize_margin: int = 14
        self._edge_policy: Literal['clip', 'clamp'] = widgets_edge_policy

        # Stateful tile selection: (kx, ky) per gid.
        self._tile_sel: dict[int, tuple[int, int]] = {}

        # Cache of tiles visible this frame.
        self._visible_tiles: list[tuple[int, int]] = []

        self._canvas = canvas
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self.update)
        self._timer.start()

    # --- public API ---

    def set_widgets_edge_policy(self, policy: Literal['clip', 'clamp']) -> None:
        """Change how pinned widgets behave at the overlay edges."""
        self._edge_policy = policy
        for it in self._widgets.values():
            self._layout_widget(it)
        self.update()

    # --- sizing/position sync ---

    def sizeHint(self) -> QtCore.QSize:
        if hasattr(self._canvas, 'size'):
            w, h = getattr(self._canvas, 'size', (800, 800))
            return QtCore.QSize(int(w), int(h))
        return super().sizeHint()

    def sync_to_canvas_geometry(self) -> None:
        """Keep overlay exactly on top of the canvas."""
        if isinstance(self.parent(), QtWidgets.QWidget):
            self.setGeometry(self.parent().rect()) # type: ignore

    # --- items management ---

    def upsert_rect(self, item: RectItem) -> None:
        self._rects[item.gid] = item

    def remove_rect(self, gid: int) -> None:
        self._rects.pop(gid, None)

    def upsert_note(self, item: NoteItem) -> None:
        self._notes[item.gid] = item

    def remove_note(self, gid: int) -> None:
        self._notes.pop(gid, None)

    def upsert_widget(self, item: PinnedWidgetItem) -> None:
        """Adopt/update a pinned QWidget and manage its geometry."""
        self._widgets[item.gid] = item
        w = item.widget
        wid = id(w)
        self._widget_index[wid] = item.gid
        if w.parent() is not self:
            w.setParent(self)
            w.installEventFilter(self)
            w.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
            w.show()

        # Initialize selection: choose any visible tile that puts the anchor
        # closest to the widget's current center. If none cached yet, keep (0,0).
        self._visible_tiles = getattr(self._canvas, 'visible_tiles', lambda: [(0, 0)])()
        if item.gid not in self._tile_sel and self._visible_tiles:
            ref = w.geometry().center()
            ax, ay, perx, pery = self._canvas.grid_to_unwrapped_px(item.gx, item.gy)
            best = None
            for kx, ky in self._visible_tiles:
                x = ax + kx * perx
                y = ay + ky * pery
                d2 = (x - ref.x()) ** 2 + (y - ref.y()) ** 2
                if best is None or d2 < best[0]:
                    best = (d2, (kx, ky))
            if best is not None:
                self._tile_sel[item.gid] = best[1]

        self._layout_widget(item)
        self.update()

    def remove_widget(self, gid: int) -> None:
        it = self._widgets.pop(gid, None)
        if it and it.widget:
            wid = id(it.widget)
            self._widget_index.pop(wid, None)
            self._drag_states.pop(wid, None)
            self._tile_sel.pop(gid, None)
            it.widget.removeEventFilter(self)
            it.widget.setParent(None)
            it.widget.deleteLater()
        self.update()

    # --- painting ---

    def paintEvent(self, ev: QtGui.QPaintEvent) -> None:
        self.sync_to_canvas_geometry()
        if not hasattr(self._canvas, 'grid_to_unwrapped_px'):
            return

        # Refresh visible tiles once per frame.
        self._visible_tiles = getattr(self._canvas, 'visible_tiles', lambda: [(0, 0)])()

        widgets_sorted = sorted(self._widgets.values(), key=lambda w: w.z)
        for item in widgets_sorted:
            self._layout_widget(item)

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        for item in widgets_sorted:
            self._paint_connector(painter, item)

        for item in sorted(self._rects.values(), key=lambda r: r.z):
            self._paint_rect(painter, item)

        for item in sorted(self._notes.values(), key=lambda n: n.z):
            self._paint_note(painter, item)

        painter.end()

    # --- tile selection helpers ---

    def _stable_tile(self, gid: int, fallback: tuple[int, int] = (0, 0)) -> tuple[int, int]:
        return self._tile_sel.get(gid, fallback)

    def _maybe_switch_tile(self, gid: int) -> None:
        """
        Only switch when:
          - current tile not visible AND
          - exactly one tile is visible.
        Otherwise keep the current tile for stability.
        """
        cur = self._stable_tile(gid)
        vis = self._visible_tiles
        if cur in vis:
            return
        if len(vis) == 1:
            self._tile_sel[gid] = vis[0]

    def _apply_tile(self, px: float, py: float, perx: float, pery: float, kx: int, ky: int) -> tuple[int, int]:
        return int(round(px + kx * perx)), int(round(py + ky * pery))

    # --- primitives ---

    def _paint_rect(self, p: QtGui.QPainter, it: RectItem) -> None:
        # Keep rect in the same tile as any widget/note with the same gid.
        kx, ky = self._stable_tile(it.gid, (0, 0))

        px0, py0, perx, pery = self._canvas.grid_to_unwrapped_px(it.gx, it.gy, align='origin')
        px1, py1, _, _ = self._canvas.grid_to_unwrapped_px(it.gx + it.w, it.gy + it.h, align='origin')

        x0, y0 = self._apply_tile(px0, py0, perx, pery, kx, ky)
        x1, y1 = self._apply_tile(px1, py1, perx, pery, kx, ky)

        x = min(x0, x1)
        y = min(y0, y1)
        w = max(1, abs(x1 - x0))
        h = max(1, abs(y1 - y0))

        pen = QtGui.QPen(QtGui.QColor(*it.rgba))
        pen.setWidth(2)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRect(x, y, w, h)

        if it.label:
            bg = QtGui.QColor(0, 0, 0, 160)
            fm = QtGui.QFontMetrics(p.font())
            tw = fm.horizontalAdvance(it.label) + 8
            th = fm.height() + 4
            p.fillRect(x + 4, y + 4, tw, th, bg)
            p.setPen(QtGui.QColor(255, 255, 255, 230))
            p.drawText(x + 8, y + 4 + fm.ascent(), it.label)

    def _paint_note(self, p: QtGui.QPainter, it: NoteItem) -> None:
        kx, ky = self._stable_tile(it.gid, (0, 0))
        px, py, perx, pery = self._canvas.grid_to_unwrapped_px(it.gx, it.gy)
        x, y = self._apply_tile(px, py, perx, pery, kx, ky)
        x += 10
        y += 10

        fm = QtGui.QFontMetrics(p.font())
        lines = it.text.splitlines() or ['']
        tw = max((fm.horizontalAdvance(s) for s in lines), default=0) + 12
        th = fm.height() * len(lines) + 10

        bg = QtGui.QColor(*it.rgba)
        fg = QtGui.QColor(*it.fg)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(x, y, tw, th, 6, 6)
        p.setPen(fg)
        ty = y + 6 + fm.ascent()
        for s in lines:
            p.drawText(x + 6, ty, s)
            ty += fm.height()

    # --- helpers ---

    def _pinned_child_at(self, pos: QtCore.QPoint) -> QtWidgets.QWidget | None:
        w = self.childAt(pos)
        if not w:
            return None
        for it in self._widgets.values():
            if w is it.widget or it.widget.isAncestorOf(w):
                return it.widget
        return None

    def _forward_mouse(self, ev: QtGui.QMouseEvent) -> None:
        """Re-send a copy of the mouse event to the underlying canvas."""
        target = getattr(self._canvas, 'native', None)
        if target is None:
            return
        local_in_target = self.mapTo(target, ev.position().toPoint())
        clone = QtGui.QMouseEvent(
            ev.type(),
            QtCore.QPointF(local_in_target),
            ev.globalPosition(),
            ev.button(),
            ev.buttons(),
            ev.modifiers(),
        )
        QtWidgets.QApplication.sendEvent(self._canvas.native, clone)

    def _forward_wheel(self, ev: QtGui.QWheelEvent) -> None:
        target = getattr(self._canvas, 'native', None)
        if target is None:
            return
        local_in_target = self.mapTo(target, ev.position().toPoint())
        clone = QtGui.QWheelEvent(
            QtCore.QPointF(local_in_target),
            ev.globalPosition(),
            ev.pixelDelta(),
            ev.angleDelta(),
            ev.buttons(),
            ev.modifiers(),
            ev.phase(),
            ev.inverted(),
        )
        QtWidgets.QApplication.sendEvent(self._canvas.native, clone)

    # --- layout & connector use stateful tile selection ---

    def _layout_widget(self, it: PinnedWidgetItem) -> None:
        """Compute position/size in screen px from grid-space anchor."""
        # Update tile selection based on current visibility policy.
        self._maybe_switch_tile(it.gid)
        kx, ky = self._stable_tile(it.gid, (0, 0))

        ax, ay, perx, pery = self._canvas.grid_to_unwrapped_px(it.gx, it.gy)
        ax, ay = self._apply_tile(ax, ay, perx, pery, kx, ky)

        if it.mode == 'canvas':
            bx, by, _, _ = self._canvas.grid_to_unwrapped_px(it.gx + it.w_cells, it.gy + it.h_cells)
            bx, by = self._apply_tile(bx, by, perx, pery, kx, ky)
            w = max(1, int(bx - ax))
            # Y grows downward in pixels -> height is ay - by
            h = max(1, int(ay - by))
        else:
            w, h = it.px_size

        x = int(ax + it.offset_px[0])
        y = int(ay + it.offset_px[1])

        if self._edge_policy == 'clamp':
            x = max(0, min(x, self.width() - w))
            y = max(0, min(y, self.height() - h))

        it.widget.setGeometry(x, y, w, h)

    def _paint_connector(self, p: QtGui.QPainter, it: PinnedWidgetItem) -> None:
        wrect = it.widget.geometry()
        if not self.rect().intersects(wrect):
            return

        kx, ky = self._stable_tile(it.gid, (0, 0))
        ax, ay, perx, pery = self._canvas.grid_to_unwrapped_px(it.gx, it.gy)
        x1, y1 = self._apply_tile(ax, ay, perx, pery, kx, ky)

        # Pick widget anchor corner.
        if it.anchor_corner != 'auto':
            x0 = wrect.left() if 'l' in it.anchor_corner else wrect.right()
            y0 = wrect.bottom() if 'b' in it.anchor_corner else wrect.top()
        else:
            dist_l = abs(wrect.left() - x1)
            dist_r = abs(wrect.right() - x1)
            dist_b = abs(wrect.bottom() - y1)
            dist_t = abs(wrect.top() - y1)
            x0 = wrect.left() if dist_l < dist_r else wrect.right()
            y0 = wrect.bottom() if dist_b < dist_t else wrect.top()

        midx = x0 + 0.3 * (x1 - x0)
        midy = y0 + 0.6 * (y1 - y0)

        pen = QtGui.QPen(QtGui.QColor(0, 220, 0, 220))
        pen.setWidth(2)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        path = QtGui.QPainterPath(QtCore.QPointF(x0, y0))
        path.lineTo(midx, y0)
        path.lineTo(midx, midy)
        path.lineTo(x1, y1)
        p.drawPath(path)

    # --- eventFilter and input forwarding unchanged below ---

    def eventFilter(self, watched: QtCore.QObject, ev: QtCore.QEvent) -> bool:
        if not isinstance(watched, QtWidgets.QWidget):
            return super().eventFilter(watched, ev)
        wid = id(watched)
        gid = self._widget_index.get(wid)
        if gid is None:
            return super().eventFilter(watched, ev)

        item = self._widgets.get(gid)
        if item is None:
            return super().eventFilter(watched, ev)

        et = ev.type()

        if et == QtCore.QEvent.Type.MouseButtonPress:
            if not isinstance(ev, QtGui.QMouseEvent):
                return False
            start_global = _global_point(ev)
            start_geom = watched.geometry()
            local_pos = _local_pointf(ev, watched)
            resizing = (
                local_pos.x() >= watched.width() - self._resize_margin
                and local_pos.y() >= watched.height() - self._resize_margin
            )
            self._drag_states[wid] = _DragState(
                start_global=start_global,
                start_geom=start_geom,
                resizing=resizing,
            )
            watched.setCursor(
                QtCore.Qt.CursorShape.SizeFDiagCursor
                if resizing
                else QtCore.Qt.CursorShape.SizeAllCursor
            )
            return True

        if et == QtCore.QEvent.Type.MouseMove:
            if not isinstance(ev, QtGui.QMouseEvent):
                return False
            st = self._drag_states.get(wid)
            if st is None:
                return False
            cur_global = _global_point(ev)
            delta = cur_global - st.start_global

            if st.resizing:
                new_w = max(60, st.start_geom.width() + delta.x())
                new_h = max(40, st.start_geom.height() + delta.y())
                watched.resize(new_w, new_h)
                if item.mode == 'fixed':
                    item.px_size = (new_w, new_h)
                else:
                    a = self._canvas.grid_to_unwrapped_px(item.gx, item.gy)[0:2]
                    bx = self._canvas.grid_to_unwrapped_px(item.gx + 1, item.gy)[0:2]
                    by = self._canvas.grid_to_unwrapped_px(item.gx, item.gy + 1)[0:2]
                    cell_w = max(1, abs(int(bx[0] - a[0])))
                    cell_h = max(1, abs(int(by[1] - a[1])))
                    item.w_cells = new_w / cell_w
                    item.h_cells = new_h / cell_h
            else:
                g = QtCore.QRect(st.start_geom)
                g.translate(delta)
                watched.setGeometry(g)
                kx, ky = self._stable_tile(item.gid, (0, 0))
                ax, ay, perx, pery = self._canvas.grid_to_unwrapped_px(item.gx, item.gy)
                ax, ay = self._apply_tile(ax, ay, perx, pery, kx, ky)
                item.offset_px = (g.x() - int(ax), g.y() - int(ay))

            self.update()
            return True

        if et == QtCore.QEvent.Type.MouseButtonRelease:
            if wid in self._drag_states:
                self._drag_states.pop(wid, None)
            watched.unsetCursor()
            return True

        if et == QtCore.QEvent.Type.HoverMove:
            if not isinstance(ev, QtGui.QHoverEvent):
                return False
            pos = ev.position()
            if (
                pos.x() >= watched.width() - self._resize_margin
                and pos.y() >= watched.height() - self._resize_margin
            ):
                watched.setCursor(QtCore.Qt.CursorShape.SizeFDiagCursor)
            else:
                watched.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            return True

        return super().eventFilter(watched, ev)

    # -- Input handling

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._pinned_child_at(ev.pos()) is None:
            self._forward_mouse(ev)
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._pinned_child_at(ev.pos()) is None:
            self._forward_mouse(ev)
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        if self._pinned_child_at(ev.pos()) is None:
            self._forward_mouse(ev)
        else:
            super().mouseReleaseEvent(ev)

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        if self._pinned_child_at(ev.position().toPoint()) is None:
            self._forward_wheel(ev)
        else:
            super().wheelEvent(ev)


def _global_point(ev: QtCore.QEvent) -> QtCore.QPoint:
    """Qt6-safe global position from a mouse event."""
    if isinstance(ev, QtGui.QMouseEvent):
        gp = getattr(ev, 'globalPosition', None)
        if callable(gp):
            return gp().toPoint()
        gp_old = getattr(ev, 'globalPos', None)
        if callable(gp_old):
            return gp_old()
    return QtGui.QCursor.pos()


def _local_pointf(ev: QtCore.QEvent, widget: QtWidgets.QWidget) -> QtCore.QPointF:
    """Qt6-safe local position from a mouse event."""
    if isinstance(ev, QtGui.QMouseEvent):
        lp = getattr(ev, 'position', None)
        if callable(lp):
            return lp()
        lp_old = getattr(ev, 'pos', None)
        if callable(lp_old):
            p = lp_old()
            return QtCore.QPointF(float(p.x()), float(p.y()))
    return widget.mapFromGlobal(_global_point(ev)).toPointF()