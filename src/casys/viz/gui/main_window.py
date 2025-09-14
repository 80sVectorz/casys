from typing import Any, cast
import threading
import pathlib

from PySide6 import QtWidgets
from PySide6 import QtCore
from PySide6 import QtGui
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QInputDialog,
    QMainWindow,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QCheckBox,
    QToolButton,
    QScrollArea,
    QSizePolicy,
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction

from casys.viz.gui.overlay import OverlayWidget
from casys.viz.tools.api import consume_preinit_tools
from casys.viz.tools.core import ToolContext, ToolManager

from .canvas_widget import CanvasWidget, ClickRequest, Layer, LayerSpec
from .ui_model import UIModel, InfoField, Tool
from casys.sim_manager import SimManager

def apply[T](widget: T, **kwargs) -> T:
    """
    Apply setter methods on a Qt widget using given keyword args.
    For each key, calls widget.set[Key](value).
    """
    
    for name, value in kwargs.items():
        setter = getattr(widget, 'set' + name[0].upper() + name[1:], None)
        if callable(setter):
            if isinstance(value, tuple):
                setter(*value)
            else:
                setter(value)

    return widget


class CollapsibleSection(QWidget):
    """
    A collapsible section widget with a toggle button and content area.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        tb = apply(
            QToolButton(self),
            text=title,
            checkable=True,
            checked=True,
            styleSheet='QToolButton { border: none; }',
            toolButtonStyle=Qt.ToolButtonStyle.ToolButtonTextBesideIcon,
            arrowType=Qt.ArrowType.DownArrow
        )
        self.toggle_button = tb
        tb.clicked.connect(self._on_toggle)

        hl = apply(
            QWidget(self),
            fixedHeight=1,
            sizePolicy=(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed),
            styleSheet='background-color: #c0c0c0;'
        )
        self.header_line = hl

        ca = QWidget(self)
        self.content_area = ca
        self.content_layout = QVBoxLayout(ca)
        apply(self.content_layout, contentsMargins=(0, 0, 0, 0), spacing=5)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(tb)
        main_layout.addWidget(hl)
        main_layout.addWidget(ca)

    def _on_toggle(self) -> None:
        expanded = self.toggle_button.isChecked()
        self.content_area.setVisible(expanded)
        self.header_line.setVisible(expanded)
        arrow = Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        self.toggle_button.setArrowType(arrow)

        from PySide6.QtWidgets import QDialog, QTreeWidget, QTreeWidgetItem, QVBoxLayout


class MainWindow(QMainWindow):
    """
    Main application window for Casys Visualizer.
    """

    tool_mgr: ToolManager
    overlay: OverlayWidget

    def __init__(
        self,
        sim: SimManager,
        layers: list[Any],
        ui_model: UIModel,
        fps: float = 60.0,
        window_size: tuple[int, int] = (800, 800),
    ) -> None:
        super().__init__()
        apply(self, windowTitle='Casys Visualizer')
        self.sim_mgr = sim
        self.layers = layers
        self.ui_model = ui_model
        self.lock = threading.Lock()

        self._last_dir: pathlib.Path | None = None

        # build menu bar
        self._build_menus()

        container = QWidget(self)
        self.setCentralWidget(container)
        layout = QHBoxLayout(container)
        apply(layout, contentsMargins=(0, 0, 0, 0))

        controls = self._create_controls_panel()
        apply(controls, sizePolicy=(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding))
        layout.addWidget(controls, 1)

        self.canvas = CanvasWidget(sim, layers, self.lock, fps, window_size)
        layout.addWidget(self.canvas.native, 4)

        overlay = OverlayWidget(canvas=self.canvas, parent=self.canvas.native)
        overlay.raise_()
        self.overlay = overlay

        self._info_timer = QTimer(self)
        self._info_timer.timeout.connect(self._update_info_labels)
        self._info_timer.start(1000)

        ctx = ToolContext(
            window=self,
            canvas=self.canvas,
            overlay=getattr(self, 'overlay', None),
            sim_mgr=self.sim_mgr,
            edit_bus=None,  # you said no EditBus yet
        )
        self.tool_mgr = ToolManager(ctx)
        self.canvas.set_tool_manager(self.tool_mgr)

        # Register any tools the user queued before starting viz
        for plugin in consume_preinit_tools():
            self.tool_mgr.register(plugin)

        self._build_tool_bar()

    def _create_controls_panel(self) -> QWidget:
        """
        Build and return the left-side controls panel containing Info, Controls, Layers, and Tools sections.
        """

        panel = QFrame(self)
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        vbox = QVBoxLayout(panel)
        apply(vbox, contentsMargins=(5, 5, 5, 5), spacing=10)

        # Info section
        info_section = CollapsibleSection('Info', panel)
        for field in self.ui_model.info_fields:
            lbl = QLabel(f'{field.label}: {field.callback()}', info_section)
            info_section.content_layout.addWidget(lbl)
        vbox.addWidget(info_section)

        # Controls section
        ctrl_section = CollapsibleSection('Controls', panel)
        row = QWidget(ctrl_section)
        row_layout = QHBoxLayout(row)
        apply(row_layout, contentsMargins=(0, 0, 0, 0))
        for name, slot in (('Start', self.sim_mgr.start), ('Pause', self.sim_mgr.pause)):
            btn = QPushButton(name, row)
            btn.clicked.connect(slot)
            row_layout.addWidget(btn)

        btn_step = QPushButton('Step', row)
        btn_step.clicked.connect(lambda: (self.lock.acquire(), self.sim_mgr.step(), self.lock.release()))
        btn_rewind = QPushButton('Rewind', row)
        btn_rewind.clicked.connect(lambda: (self.lock.acquire(), self.sim_mgr.rewind(), self.lock.release()))
        row_layout.addWidget(btn_step)
        row_layout.addWidget(btn_rewind)

        for b in (btn_step, btn_rewind):
            row_layout.addWidget(b)

        ctrl_section.content_layout.addWidget(row)

        lbl_sim_spd_slider = QLabel('Sec/Step', ctrl_section)
        def handle_sim_spd_slider(v):
            self.sim_mgr.timestep = v/1000
            lbl_sim_spd_slider.setText(f'Sec/Step {v/1000}')

        sim_spd_slider = QSlider(Qt.Orientation.Horizontal, ctrl_section)
        sim_spd_slider.setRange(0, 1000)
        sim_spd_slider.setValue(int(self.sim_mgr.timestep * 1000))
        sim_spd_slider.valueChanged.connect(handle_sim_spd_slider)
        ctrl_section.content_layout.addWidget(lbl_sim_spd_slider)
        ctrl_section.content_layout.addWidget(sim_spd_slider)

        lbl_sync_rate_slider = QLabel('Step/Sync', ctrl_section)
        def handle_sync_rate_slider(v):
            self.canvas.sync_rate = v
            lbl_sync_rate_slider.setText(f'Steps/Sync {v}')
            
        sync_rate_slider = QSlider(Qt.Orientation.Horizontal, ctrl_section)
        sync_rate_slider.setRange(1, 100)
        sync_rate_slider.setValue(1)
        sync_rate_slider.valueChanged.connect(handle_sync_rate_slider)
        ctrl_section.content_layout.addWidget(lbl_sync_rate_slider)
        ctrl_section.content_layout.addWidget(sync_rate_slider)

        vbox.addWidget(ctrl_section)

        # Layers section
        layer_section = CollapsibleSection('Layers', panel)
        scroll = QScrollArea(layer_section)
        scroll.setWidgetResizable(True)
        layer_container = QWidget(scroll)
        layer_layout = QVBoxLayout(layer_container)
        apply(layer_layout, contentsMargins=(0, 0, 0, 0), spacing=5)

        # build layer checkboxes with canvas redraw on toggle
        for layer in self.layers:
            cb = QCheckBox(layer.name, layer_container)
            # update visibility and request repaint
            def _on_layer_toggled(state: int, l: Layer = layer) -> None:
                l.visible = (state > 0)
                # trigger canvas redraw
                self.canvas.update()
            cb.stateChanged.connect(_on_layer_toggled)
            # set initial checked state without emitting signal
            cb.blockSignals(True)
            cb.setChecked(layer.visible)
            cb.blockSignals(False)
            layer_layout.addWidget(cb)
        scroll.setWidget(layer_container)
        layer_section.content_layout.addWidget(scroll)
        vbox.addWidget(layer_section)

        vbox.addStretch()
        return panel
    
    def _build_tool_bar(self) -> None:
        tb = QtWidgets.QToolBar('Tools', self)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, tb)

        def refresh() -> None:
            tb.clear()
            for info in self.tool_mgr.tools():
                act = QtGui.QAction(info.name, self)
                act.setToolTip(info.tooltip or info.name)
                if info.kind == 'modal':
                    act.triggered.connect(lambda _=False, n=info.name: self.tool_mgr.activate_modal(n))
                else:
                    act.triggered.connect(lambda _=False, n=info.name: self.tool_mgr.use_oneshot(n))
                tb.addAction(act)

        self.tool_mgr.on_changed(refresh)
        refresh()
    
    def _build_menus(self) -> None:
        """Create 'File' menu with Save/Load actions."""
        mb = self.menuBar()
        file_menu = mb.addMenu('File')

        act_save = QAction('Save…', self)
        act_save.setShortcut('Ctrl+S')
        act_save.triggered.connect(self._save_state_dialog)
        file_menu.addAction(act_save)

        act_load = QAction('Load…', self)
        act_load.setShortcut('Ctrl+O')
        act_load.triggered.connect(self._load_state_dialog)
        file_menu.addAction(act_load)
    
    def _save_state_dialog(self) -> None:
        start_dir = str(self._last_dir) if self._last_dir else ''
        path, _ = QFileDialog.getSaveFileName(
            self,
            'Save Snapshot',
            start_dir,
            'CASim snapshot (*.npz)'
        )
        if not path:
            return

        self._last_dir = pathlib.Path(path).parent
        max_hist = len(self.sim_mgr.history_buffer)
        steps, ok = QInputDialog.getInt(
            self,
            'History Length',
            f'How many history steps to save? (0-{max_hist})',
            max_hist, 0, max_hist
        )
        if ok:
            self.lock.acquire()
            try:
                self.sim_mgr.save_state(path, steps)  # SimManager appends .npz if missing
            finally:
                self.lock.release()

    def _load_state_dialog(self) -> None:
        start_dir = str(self._last_dir) if self._last_dir else ''
        path, _ = QFileDialog.getOpenFileName(
            self,
            'Load Snapshot',
            start_dir,
            'CASim snapshot (*.npz)'
        )
        if not path:
            return

        self._last_dir = pathlib.Path(path).parent
        self.lock.acquire()
        try:
            self.sim_mgr.load_state(path)
        finally:
            self.lock.release()
        self.canvas.update()

    def _update_info_labels(self) -> None:
        container = self.centralWidget()
        layout = cast(QHBoxLayout, container.layout())
        panel = cast(QWidget, layout.itemAt(0).widget())
        vbox = cast(QVBoxLayout, panel.layout())
        info_section = cast(CollapsibleSection, vbox.itemAt(0).widget())
        for i, field in enumerate(self.ui_model.info_fields):
            lbl = cast(QLabel, info_section.content_layout.itemAt(i).widget())
            lbl.setText(f'{field.label}: {field.callback()}')
