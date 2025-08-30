from typing import Any
from PySide6.QtCore import QEvent, QObject
from vispy.app import KeyEvent, MouseEvent, use_app
from vispy.util.keys import Key
use_app('pyside6')
from vispy.scene import SceneCanvas
from vispy import gloo
import numpy as np
from collections.abc import Callable
from dataclasses import dataclass
import threading
import pathlib

from casys.sim_manager import SimManager

# GLSL sources
VERT_QUAD = """
#version 330 core
in  vec2 in_pos;
in  vec2 in_uv;
out vec2 v_uv;
void main() { v_uv = in_uv; gl_Position = vec4(in_pos,0,1); }
"""

COMMON_GLSL = """
#version 330 core

uniform sampler2D u_tex;
uniform mat3      u_cam;   // zoom and pan matrix
uniform ivec2     u_dims;  // grid dims (width, height)

in  vec2  v_uv;            // normalized viewport coords [0..1]
out vec4  FragColor;

vec2 apply_cam(vec2 uv) {
    return (u_cam * vec3(uv, 1.0)).xy;
}

// wrap UV infinitely in [0..1]
vec2 wrapped_uv() {
    vec2 uvw = apply_cam(v_uv);
    return fract(uvw);
}

// continuous world coordinates in units of cells
vec2 world_uv() {
    return wrapped_uv() * vec2(u_dims);
}

// integer cell index
ivec2 cell_index() {
    return ivec2(floor(world_uv()));
}

// fractional UV inside the current cell
vec2 cell_uv() {
    return fract(world_uv());
}

// fetch the metadata (as uint) for that cell

float sample_meta_float() {
    return texelFetch(u_tex, cell_index(), 0).r;
}

uint sample_meta() {
    return uint(texelFetch(u_tex, cell_index(), 0).r*255);
}

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
"""

FALLBACK_FRAG = """
void main(){
    FragColor = vec4(float(sample_meta));
}
"""

CHECKERBOARD_FRAG = COMMON_GLSL+"""

uniform float u_checker_size;  // size of each square in pixels
uniform float u_border_radius; // border radius in world_space;
uniform ivec2 u_window_dims; // Window dimensions in pixels

void main() {
    ivec2 pix = ivec2(gl_FragCoord.xy);
    int cx = pix.x / int(u_checker_size);
    int cy = pix.y / int(u_checker_size);
    float checker = mod(float(cx + cy), 2.0);
    vec3 light = vec3(1-0.95);
    vec3 dark  = vec3(1-0.85);

    float screen_ratio = u_window_dims.y/u_window_dims.x;
    float border_radius = u_border_radius;
    vec2 border = abs(world_uv() - u_dims*0.5) * 2;
    float border_d = (max(border.x,border.y) - (u_dims.x-border_radius)) / border_radius;
    if (border_d <= 1 && border_d >= 0) {
        light = mix(light, vec3(0,1,0), border_d);
        dark = mix(dark, vec3(0,1,0), border_d);
    }

    FragColor = vec4(mix(light, dark, checker), 1.0);
}
"""

@dataclass
class ClickRequest:
    callback: Callable[[tuple[int,int], tuple[int,int]], None]
    lock: threading.Lock
    single_use: bool
    live: bool = False
    dirty: bool = False

@dataclass
class LayerSpec:
    name: str
    field: str
    dtype: np.dtype
    shader: str | Callable[[], str] | None
    shader_uniforms: tuple[tuple[str, Any],]
    visible: bool = True

class Layer:
    spec: LayerSpec
    visible: bool

    def __init__(self, width: int, height: int, spec: LayerSpec) -> None:
        self.spec = spec
        self.name = spec.name
        self.field = spec.field
        self.visible = spec.visible
        self.cpu = np.zeros((height, width), dtype=spec.dtype)
        self.tex = gloo.Texture2D(self.cpu, format='luminance', internalformat='R8')
        self.tex.interpolation = 'nearest'

        frag_src = self._resolve_shader(spec.shader)
        self.prog = gloo.Program(VERT_QUAD, frag_src)
        for key, val in spec.shader_uniforms:
            self.prog[key] = val
        self.prog['u_tex'] = self.tex
        self.prog['u_dims'] = (width, height)

    @staticmethod
    def _resolve_shader(src: str | Callable[[], str] | None) -> str:
        if src is None:
            raise ValueError('Shader source required')
        if callable(src):
            return src()
        path = pathlib.Path(src)
        if path.exists():
            return COMMON_GLSL + '\n' + path.read_text()
        return COMMON_GLSL + '\n' + str(src)

    def upload(self, data: np.ndarray) -> None:
        self.cpu[:, :] = data.T
        self.tex.set_data(self.cpu)

class CanvasWidget(SceneCanvas):
    sim: SimManager
    layers: list[Layer]
    lock: threading.Lock
    click_request: ClickRequest | None = None

    def __init__(
        self,
        sim: SimManager,
        layers: list[Layer],
        lock: threading.Lock,
        fps: float = 60.0,
        size: tuple[int,int] = (800, 800),
    ) -> None:
        super().__init__(keys='interactive', size=size, show=False,)
        self.unfreeze()

        # store the original Qt keyPressEvent
        self._orig_qt_keypress = self.native.keyPressEvent

        def _qt_keypress(evt):
            # swallow Esc at the Qt layer so VisPy never even sees it
            if evt.key() == 16777216:
                return
            # for all other keys, fall back to the original handler
            self._orig_qt_keypress(evt)

        # monkey‑patch the native widget
        self.native.keyPressEvent = _qt_keypress

        # store the original Qt resizeEvent handler
        self._orig_qt_resize = self.native.resizeEvent

        # monkey-patch in our own
        def _qt_resize(evt):
            # Pull new size out of the QResizeEvent
            w = evt.size().width()
            h = evt.size().height()
            self.handle_resize((w,h))
            # Call Qt’s original
            self._orig_qt_resize(evt)

        # Replace the widget’s resizeEvent
        self.native.resizeEvent = _qt_resize

        self.sim = sim
        self.layers = layers
        self.lock = lock

        # camera state
        self.zoom = 1.0
        self.pan = np.zeros(2, dtype='f4')
        self.screen_ratio = 1.0
        self._dragging = False
        self._last_pos: tuple[float,float] | None = None

        # initialize GL resources
        self._init_gl()

        # update timer
        from PySide6.QtCore import QTimer
        self._timer = QTimer(self.native)
        self._timer.timeout.connect(self.native.update)
        self._timer.start(int(1000 / fps))

    def _init_gl(self) -> None:
        # full-screen quad VBO
        quad = np.zeros(
            6,
            dtype=[("in_pos", np.float32, 2), ("in_uv", np.float32, 2)],
        )
        quad["in_pos"] = [
            (-1, -1), (1, -1), (-1,  1),
            (-1,  1), (1, -1), (1,  1),
        ]
        quad["in_uv"] = [
            (0, 0), (1, 0), (0, 1),
            (0, 1), (1, 0), (1, 1),
        ]
        self.vbo = gloo.VertexBuffer(quad)

        # checkerboard background program
        self.bg_prog = gloo.Program(VERT_QUAD, CHECKERBOARD_FRAG)
        self.bg_prog['u_dims'] = self.sim.sim.dims
        self.bg_prog['u_checker_size'] = 100

        gloo.set_clear_color('black')
        gloo.set_state(blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'), depth_test=False)

    def on_draw(self, event) -> None:
        gloo.clear(color=True)

        cam = self._compute_cam().T

        self.bg_prog.bind(self.vbo)
        self.bg_prog['u_cam'] = cam
        self.bg_prog['u_border_radius'] = 0.005 * self.sim.sim.dims[0] * self.zoom
        self.bg_prog['u_window_dims'] = self.size
        self.bg_prog.draw(mode='triangles')

        with self.lock:
            state = self.sim.get_current_state()

        for layer in self.layers[::-1]:
            if not layer.visible:
                continue
            layer.upload(state[layer.field])
            layer.prog['u_cam'] = cam
            layer.prog.bind(self.vbo)
            layer.prog.draw('triangles')

    # --- spatial math helpers ---
    def _compute_cam(self) -> np.ndarray:
        """Return 3x3 affine matrix (scale and translate) used in shaders."""
        s = self.zoom
        tx, ty = self.pan
        return np.array([[s, 0, tx], [0, s*self.screen_ratio, ty*self.screen_ratio], [0, 0, 1]], dtype="f4")

    def _screen_to_uv(self, x: float, y: float) -> np.ndarray:
        """
        Map pixel coords (origin top-left) → normalised UV ([0-1]x[0-1])
        **after** current pan/zoom.
        """
        w, h = self.size
        uv_screen = np.array([x / w, 1.0 - y / h], dtype="f4")   # flip Y
        # invert affine: uv_world = (uv_screen - pan) / zoom
        return (uv_screen - self.pan) / self.zoom
    
    def _screen_to_grid(self, x: float, y:float) -> tuple[int,int]:
        w, h = self.size
        dims = self.sim.sim.dims
        uv_screen = np.array([x / w, self.screen_ratio - y / h * self.screen_ratio], dtype="f4")
        uv_screen[0] += self.pan[0] / self.zoom
        uv_screen[1] += self.pan[1]*self.screen_ratio / self.zoom
        uv_screen *= self.zoom
        screen_coord = np.floor(uv_screen * np.array(self.sim.sim.dims)) % dims

        return (int(screen_coord[0]),int(screen_coord[1]))

    # --- mouse handlers --- #

    def on_mouse_press(self, ev: MouseEvent):
        if ev.button == 1:
            self._last_pos = ev.pos

            cr = self.click_request
            if cr is not None:
                if Key('Shift') not in ev.modifiers and cr.dirty:
                    self.click_request = None
                else:
                    cr.live = True
                    return

            self._dragging = True

    def on_mouse_release(self, ev: MouseEvent):
        if ev.button == 1:
            cr = self.click_request
            if cr is not None and cr.live:

                cr.lock.acquire()
                try:
                    was_playing = not self.sim.paused
                    if was_playing:
                        self.sim.pause()

                    cr.callback(self._screen_to_grid(*tuple(self._last_pos.astype(np.int_))), self._screen_to_grid(*ev.pos)) # type: ignore
                finally:
                    cr.lock.release()

                if was_playing: self.sim.start()

                if cr.single_use or Key('Shift') not in ev.modifiers:
                    self.click_request = None

                else:
                    cr.dirty = True
                    cr.live = False

            self._dragging = False
            self._last_pos = None

    def on_mouse_move(self, ev: MouseEvent):
        if not self._dragging:
            return

        # compute how many pixels we moved since last event
        new_x, new_y = ev.pos
        old_x, old_y = self._last_pos # type: ignore
        dx, dy = new_x - old_x, new_y - old_y
        self._last_pos = (new_x, new_y)

        w, h = self.size
        # pan in UV‐space: drag right → pan right, drag up → pan up
        self.pan += np.array([-dx / w,  dy / h], dtype="f4") * self.zoom
        self.update()

    def on_mouse_wheel(self, ev):
        # mouse position as normalized [0...1]x[0...1]
        mx, my = ev.pos
        w, h   = self.size
        uv_screen = np.array([mx / w, 1 - (my / h)], dtype="f4")

        # compute new zoom
        old_z = self.zoom
        factor = 1.1 ** -ev.delta[1]    # ev.delta[1] is wheel scroll
        new_z = np.clip(old_z * factor, 0.001, 512.0)
        self.zoom = new_z

        # adjust pan so (uv_screen) stays fixed
        self.pan += (old_z - new_z) * uv_screen
        # print(f"[DEBUG zoom] zoom={self.zoom:.4f}, pan={self.pan}")

        self.update()

    def handle_resize(self, size: tuple[int,int]):
        if not hasattr(self,'_last_size'):
            self._last_size = size

        new_w, new_h = size
        old_w, old_h = self._last_size
        # keep the same pixel-per-world‐unit ratio
        if old_w:
            self.zoom *= new_w / old_w
        gloo.set_viewport(0, 0, new_w, new_h)
        # update GL viewport and aspect ratio
        self.screen_ratio = new_h / new_w
        # store for next resize
        self._last_size = (new_w, new_h)

    # ---------------------

    def on_key_press(self, ev: KeyEvent):
        match ev.key:
            case 'Escape':
                ev.handled = True
                return    

    def on_key_release(self, ev: KeyEvent):
        match ev.key:
            case 'Left' | 'A':
                self.sim.rewind()
            case 'Right' | 'D':
                self.sim.step()
            case ' ':
                self.sim.start() if self.sim.paused else self.sim.pause()
            case 'Escape':
                return