import sys
from typing import Any

from PySide6.QtWidgets import QApplication
from casys.sim_manager import SimManager

from .gui.main_window import MainWindow
from .gui.canvas_widget import Layer, LayerSpec
from .gui.ui_model import UIModel

class Visualizer:
    def __init__(
        self,
        sim_manager: SimManager,
        layers_cfg: list[dict[str, Any]],
        ui_model: UIModel,
        fps: float = 60.0,
        window_size: tuple[int,int] = (800, 800),
    ) -> None:
        self.sim = sim_manager
        self.ui_model = ui_model
        self.layers: list[Layer] = []
        self.fps = fps
        self.window_size = window_size
        w, h = self.sim.dims
        for cfg in layers_cfg:
            spec = LayerSpec(
                name=cfg['name'],
                field=cfg['field'],
                dtype=cfg['dtype'],
                shader=cfg.get('shader'),
                shader_uniforms=tuple(cfg.get('shader_uniforms', tuple())),
                visible=cfg.get('visible',True)
            )
            self.layers.append(Layer(w, h, spec))

    def start(self) -> None:
        app = QApplication(sys.argv)
        window = MainWindow(self.sim, self.layers, self.ui_model, self.fps, self.window_size)
        window.show()
        sys.exit(app.exec())