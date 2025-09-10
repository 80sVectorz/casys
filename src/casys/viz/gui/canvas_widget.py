from typing import Any
from PySide6.QtCore import QEvent, QObject
from vispy.app import KeyEvent, MouseEvent, use_app
from vispy.util.keys import Key

from casys.viz.tools.core import ToolEvent, ToolManager, MOD_SHIFT, MOD_CTRL, MOD_ALT, MOD_META
use_app('pyside6')
from vispy.scene import SceneCanvas
from vispy import gloo
import numpy as np
import math
from collections.abc import Callable
from dataclasses import dataclass
import threading
import pathlib

from casys.sim_manager import SimManager

# GLSL sources
VERT_QUAD = """//glsl
#version 330 core
in  vec2 in_pos;
in  vec2 in_uv;
out vec2 v_uv;
void main() { v_uv = in_uv; gl_Position = vec4(in_pos,0,1); }
"""

COMMON_GLSL = """//glsl
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

// ---- Analytic AA helpers -----------------------------------------------------
// You can override the pixel softness globally by #defining U_AA_SCALE in user code.
#ifndef U_AA_SCALE
#define U_AA_SCALE 1.0
#endif

// Anti-aliased hard step using derivative-based width.
float aa_step(float edge, float x) {
    float w = fwidth(x) * U_AA_SCALE;
    return smoothstep(edge - w, edge + w, x);
}

// Given a signed distance (sdf) where sdf < 0 is "inside",
// return a smooth coverage [0..1].
float aa_coverage(float sdf) {
    float w = fwidth(sdf) * U_AA_SCALE + 1e-7;
    return clamp(0.5 - sdf / w, 0.0, 1.0);
}

// --- SDF primitives in *cell space* (use p = cell_uv() - 0.5) ----------------
float sd_box(vec2 p, vec2 half_size, float radius) {
    vec2 q = abs(p) - half_size + radius;
    return length(max(q, 0.0)) - radius;
}

float sd_circle(vec2 p, float r) {
    return length(p) - r;
}

// Thick line segment from a to b with radius r, all in cell space.
float sd_segment(vec2 p, vec2 a, vec2 b, float r) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

// AA wrappers that directly return a coverage [0..1]
float aa_box(vec2 p, vec2 half_size, float radius) {
    return aa_coverage(sd_box(p, half_size, radius));
}

float aa_circle(vec2 p, float radius) {
    return aa_coverage(sd_circle(p, radius));
}

float aa_segment(vec2 p, vec2 a, vec2 b, float radius) {
    return aa_coverage(sd_segment(p, a, b, radius));
}
"""

FALLBACK_FRAG = """//glsl
void main(){
    FragColor = vec4(float(sample_meta));
}
"""

CHECKERBOARD_FRAG = COMMON_GLSL+"""//glsl

uniform float u_checker_size;  // size of each square in pixels
uniform float u_border_radius; // border radius in world cells
uniform ivec2 u_window_dims;   // Window dimensions in pixels

void main() {
    ivec2 pix = ivec2(gl_FragCoord.xy);
    int cx = pix.x / int(u_checker_size);
    int cy = pix.y / int(u_checker_size);
    float checker = mod(float(cx + cy), 2.0);
    vec3 light = vec3(1.0 - 0.95);
    vec3 dark  = vec3(1.0 - 0.85);

    // Rectangular border in world (cell) units
    float br = u_border_radius;
    vec2 half = vec2(u_dims) * 0.5;
    vec2 d = abs(world_uv() - half) * 2.0;
    vec2 inner = vec2(u_dims) - vec2(br) * 2.0;

    // Rectangular single-edge feather, identical to square behavior when u_dims.x==u_dims.y
    vec2 b = abs(world_uv() - u_dims * 0.5) * 2.0;

    // Equalize aspect: measure Y in X-units so the Chebyshev radius works on rectangles.
    float cheb = max(b.x, b.y * (u_dims.x / u_dims.y));

    float feather = (cheb - (u_dims.x - u_border_radius)) / u_border_radius;

    if (feather >= 0.0 && feather <= 1.0) {
        vec3 tint = vec3(0.0, 1.0, 0.0);
        light = mix(light, tint, feather);
        dark  = mix(dark,  tint, feather);
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
    rect_wrap: bool = False  # opt-in: split rectangles across torus seams

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

    tool_mgr: ToolManager | None = None

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
        W, H = self.sim.sim.dims
        self.bg_prog['u_border_radius'] = 0.005 * min(W, H) * self.zoom
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
        W, H = map(float, self.sim.sim.dims)
        y_corr = self.screen_ratio * (W / H)  # = (h/w) * (W/H)
        return np.array(
            [[s, 0, tx],
            [0, s * y_corr, ty * y_corr],
            [0, 0, 1]],
            dtype='f4',
        )

    def _screen_to_uv(self, x: float, y: float) -> np.ndarray:
        """Return wrapped UV in [0..1]^2 that matches the shader path."""
        w, h = self.size
        uv_screen = np.array([x / w, 1.0 - (y / h)], dtype='f4')

        s = self.zoom
        W, H = self.sim.sim.dims
        y_corr = self.screen_ratio * (W / H)  # must match _compute_cam

        tx, ty = self.pan
        uv_cam = np.array(
            [s * uv_screen[0] + tx, s * y_corr * uv_screen[1] + ty * y_corr],
            dtype='f4',
        )
        return uv_cam % 1.0

    def _screen_to_grid(self, x: float, y: float) -> tuple[int, int]:
        uv = self._screen_to_uv(x, y)
        W, H = self.sim.sim.dims
        ix = int(np.floor(uv[0] * W)) % W
        iy = int(np.floor(uv[1] * H)) % H
        return (ix, iy)
    
    def _wrap_span_shortest(self, i0: int, i1: int, n: int) -> list[tuple[int, int]]:
        """Return 1 or 2 inclusive index ranges for the shortest wrap path."""
        fwd = (i1 - i0) % n
        bwd = (i0 - i1) % n
        if fwd <= bwd:
            s, e = i0, (i0 + fwd) % n
        else:
            s, e = (i0 - bwd) % n, i0
        return [(s, e)] if s <= e else [(s, n - 1), (0, e)]

    def _decompose_wrapped_rect(
        self, a: tuple[int, int], b: tuple[int, int]
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Split an a->b rectangle into 1..4 inclusive subrects on a torus grid."""
        W, H = self.sim.sim.dims
        xs = self._wrap_span_shortest(a[0], b[0], W)
        ys = self._wrap_span_shortest(a[1], b[1], H)
        rects: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for (x0, x1) in xs:
            for (y0, y1) in ys:
                rects.append(((x0, y0), (x1, y1)))
        return rects
    
    def grid_to_screen_px(
        self,
        gx: float,
        gy: float,
        *,
        align: str = 'center',
        prefer_center_tile: bool = True,
        tile_ref_px: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """
        Map a grid coordinate (gx, gy) to a screen pixel position under the current
        camera and toroidal wrapping.

        If tile_ref_px is provided, the wrapped copy is chosen to be closest to that
        reference pixel in screen-UV space. This makes positions stable relative to
        the widget, preventing 'flips' when the cell is off-screen.

        Args:
            gx: Grid x index. Floats allowed for subcell positions.
            gy: Grid y index. Floats allowed for subcell positions.
            align: 'center' uses the cell center; 'origin' uses the cell's min corner.
            prefer_center_tile: If True and no tile_ref_px is given, choose the copy
                nearest the viewport center when zoomed in.
            tile_ref_px: Optional (x, y) screen pixel that acts as an anchor for
                choosing the wrap copy nearest to the widget.

        Returns:
            np.ndarray of shape (2,) as [px, py].
        """
        w, h = self.size
        W, H = self.sim.sim.dims

        s = self.zoom
        tx, ty = self.pan
        y_corr = self.screen_ratio * (W / H)

        if align == 'center':
            u = (gx + 0.5) / W
            v = (gy + 0.5) / H
        else:
            u = gx / W
            v = gy / H

        ux = (u - tx) / s
        uy = (v - ty * y_corr) / (s * y_corr)

        period_x = 1.0 / s
        period_y = 1.0 / (s * y_corr)

        if tile_ref_px is not None:
            rx = tile_ref_px[0] / w
            ry = 1.0 - (tile_ref_px[1] / h)
            if period_x > 0.0:
                nx = round((rx - ux) / period_x)
                ux += nx * period_x
            if period_y > 0.0:
                ny = round((ry - uy) / period_y)
                uy += ny * period_y
        else:
            if prefer_center_tile and period_x < 1.0:
                ux += round((0.5 - ux) / period_x) * period_x
            if prefer_center_tile and period_y < 1.0:
                uy += round((0.5 - uy) / period_y) * period_y

        ux %= 1.0
        uy %= 1.0

        px = int(ux * w)
        py = int((1.0 - uy) * h)
        return np.array([px, py])
    
    def grid_to_unwrapped_px(
        self,
        gx: float,
        gy: float,
        *,
        align: str = 'center',
    ) -> tuple[float, float, float, float]:
        """
        Return unwrapped pixel coordinates and tile periods in pixels.

        Args:
            gx: Grid x index.
            gy: Grid y index.
            align: 'center' for cell center, 'origin' for min corner.

        Returns:
            (px, py, period_x_px, period_y_px)
        """
        w, h = self.size
        W, H = self.sim.sim.dims

        s = self.zoom
        tx, ty = self.pan
        y_corr = self.screen_ratio * (W / H)

        if align == 'center':
            u = (gx + 0.5) / W
            v = (gy + 0.5) / H
        else:
            u = gx / W
            v = gy / H

        ux = (u - tx) / s
        uy = (v - ty * y_corr) / (s * y_corr)

        px = ux * w
        py = (1.0 - uy) * h

        period_x_px = w / s
        period_y_px = h / (s * y_corr)
        return px, py, period_x_px, period_y_px


    def visible_tiles(self) -> list[tuple[int, int]]:
        """
        Compute which torus tiles (kx, ky) intersect the viewport [0..w) x [0..h).

        Note: Y uses bottom-anchored intervals because grid_to_unwrapped_px(..., align='origin')
        returns the bottom edge for the origin tile due to the (1 - uy) flip.
        """
        w, h = self.size
        px0, py0, perx, pery = self.grid_to_unwrapped_px(0, 0, align='origin')

        def x_candidates(p0: float, per: float, L: float) -> list[int]:
            kmin = math.floor((-p0 - per) / per) - 1
            kmax = math.ceil((L - p0) / per) + 1
            out: list[int] = []
            for k in range(kmin, kmax + 1):
                left = p0 + k * per
                right = left + per
                if right > 0 and left < L:
                    out.append(k)
            return sorted(set(out))

        def y_candidates(p0: float, per: float, L: float) -> list[int]:
            # For each ky, the vertical span is [bottom - per, bottom]
            kmin = math.floor((-p0) / per) - 2
            kmax = math.ceil((L - p0) / per) + 2
            out: list[int] = []
            for k in range(kmin, kmax + 1):
                bottom = p0 + k * per
                top = bottom - per
                if bottom > 0 and top < L:
                    out.append(k)
            return sorted(set(out))

        xs = x_candidates(px0, perx, w)
        ys = y_candidates(py0, pery, h)
        return [(kx, ky) for kx in xs for ky in ys]

    # --- mouse handlers --- #

    def on_mouse_press(self, ev: MouseEvent):
        if ev.button == 1:
            if self.tool_mgr is not None and self._tool_active():
                gx, gy = self._screen_to_grid(*ev.pos)
                self.tool_mgr.route(ToolEvent(
                    kind='down',
                    gpos=(gx, gy),
                    button=ev.button,
                    modifiers=self._mods_mask(ev),
                ))
                ev.handled = True
                return

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
        if ev.button == 1 and self.tool_mgr is not None and self._tool_active():
            gx, gy = self._screen_to_grid(*ev.pos)
            self.tool_mgr.route(ToolEvent(
                kind='up',
                gpos=(gx, gy),
                button=ev.button,
                modifiers=self._mods_mask(ev),
            ))
            ev.handled = True
            return

        if ev.button == 1:
            cr = self.click_request
            if cr is not None and cr.live:

                cr.lock.acquire()
                was_playing = not self.sim.paused
                try:

                    p0 = self._screen_to_grid(*tuple(self._last_pos.astype(np.int_)))
                    p1 = self._screen_to_grid(*ev.pos)

                    if cr.rect_wrap:
                        for a, b in self._decompose_wrapped_rect(p0, p1):
                            cr.callback(a, b)
                    else:
                        cr.callback(p0, p1)
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
        if self.tool_mgr is not None and self._tool_active():
            gx, gy = self._screen_to_grid(*ev.pos)
            self.tool_mgr.route(ToolEvent(kind='move', gpos=(gx, gy)))
            ev.handled = True
            return

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

    # ---- ToolManager stuff ----

    def set_tool_manager(self, mgr: ToolManager) -> None:
        self.tool_mgr = mgr

    def _tool_active(self) -> bool:
        """True if a modal or oneshot tool is currently active."""
        tm = self.tool_mgr
        if tm is None:
            return False
        # Prefer public accessors if you have them:
        active_modal = getattr(tm, 'active_modal', lambda: None)()
        is_one = getattr(tm, 'is_oneshot_active', lambda: False)()
        return bool(active_modal) or bool(is_one)

    def _mods_mask(self, ev: MouseEvent) -> int:
        m = 0
        if Key('Shift') in ev.modifiers:
            m |= MOD_SHIFT
        if Key('Control') in ev.modifiers:
            m |= MOD_CTRL
        if Key('Alt') in ev.modifiers:
            m |= MOD_ALT
        if Key('Meta') in ev.modifiers:
            m |= MOD_META
        return m

    # ---------------------

    def on_key_press(self, ev: KeyEvent):
        match ev.key:
            case 'Escape':
                if self.tool_mgr is not None and self._tool_active():
                    self.tool_mgr.cancel()
                    ev.handled = True
                    return

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