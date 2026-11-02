"""OpenGL (moderngl) 3D view of the run: textured floor, a TurtleBot3 Waffle Pi-style robot built
from primitives at the real ROBOTIS dimensions, a camera mast with its field-of-view frustum,
planar shadows, the driven trail and the fitted line. GPU on Windows, EGL/llvmpipe on Linux."""
import sys
import numpy as np
import moderngl

# ---------------------------------------------------------------- geometry helpers
def _n(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


class Mesh:
    def __init__(self):
        self.v, self.n, self.c = [], [], []

    def tri(self, a, b, c, col, n=None):
        a, b, c = map(np.asarray, (a, b, c))
        nn = _n(np.cross(b - a, c - a)) if n is None else n
        for p, q in zip((a, b, c), nn if np.ndim(nn) == 2 else (nn, nn, nn)):
            self.v.append(p); self.n.append(q); self.c.append(col)

    def quad(self, a, b, c, d, col):
        self.tri(a, b, c, col); self.tri(a, c, d, col)

    def box(self, c0, size, col, R=None):
        cx, cy, cz = c0; sx, sy, sz = np.asarray(size) / 2
        P = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
        if R is not None:
            P = P @ R.T
        P = P + c0
        F = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        for f in F:
            self.quad(*[P[i] for i in f], col)

    def cyl(self, c0, axis, r, h, col, seg=28, cap_col=None, r_fn=None, rot=0.0):
        """Cylinder centred at c0 along unit axis. r_fn(angle)->radius gives tyre treads."""
        axis = _n(np.asarray(axis, float))
        u = _n(np.cross(axis, [0, 0, 1] if abs(axis[2]) < 0.9 else [1, 0, 0])); w = np.cross(axis, u)
        c0 = np.asarray(c0, float)
        a = np.linspace(0, 2 * np.pi, seg + 1) + rot
        rr = np.array([r_fn(x - rot) if r_fn else r for x in a])
        ring = lambda s: c0 + axis * s * h / 2 + (np.cos(a)[:, None] * u + np.sin(a)[:, None] * w) * rr[:, None]
        top, bot = ring(1), ring(-1)
        for i in range(seg):
            n0 = _n(np.cos(a[i]) * u + np.sin(a[i]) * w); n1 = _n(np.cos(a[i + 1]) * u + np.sin(a[i + 1]) * w)
            self.tri(bot[i], bot[i + 1], top[i + 1], col, np.array([n0, n1, n1]))
            self.tri(bot[i], top[i + 1], top[i], col, np.array([n0, n1, n0]))
            cc = cap_col or col
            self.tri(c0 + axis * h / 2, top[i], top[i + 1], cc, axis)
            self.tri(c0 - axis * h / 2, bot[i + 1], bot[i], cc, -axis)

    def sphere(self, c0, r, col, seg=12):
        c0 = np.asarray(c0, float)
        th = np.linspace(0, np.pi, seg + 1); ph = np.linspace(0, 2 * np.pi, 2 * seg + 1)
        P = lambda t, p: np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
        for i in range(seg):
            for j in range(2 * seg):
                q = [P(th[i], ph[j]), P(th[i + 1], ph[j]), P(th[i + 1], ph[j + 1]), P(th[i], ph[j + 1])]
                self.tri(c0 + r * q[0], c0 + r * q[1], c0 + r * q[2], col, np.array(q[:3]))
                self.tri(c0 + r * q[0], c0 + r * q[2], c0 + r * q[3], col, np.array([q[0], q[2], q[3]]))

    def array(self):
        return np.hstack([np.array(self.v, "f4"), np.array(self.n, "f4"), np.array(self.c, "f4")]).astype("f4")


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# ---------------------------------------------------------------- robot
# TurtleBot3 Waffle Pi (ROBOTIS e-manual): 281 x 306 x 141 mm, wheel radius 33 mm, wheel separation
# 287 mm, LDS lidar on top, Raspberry Pi camera at the front. Our line camera sits on an extra mast.
PLATE = (0.20, 0.20, 0.21); PLATE_EDGE = (0.15, 0.15, 0.16); TYRE = (0.06, 0.06, 0.07); HUB = (0.75, 0.76, 0.78)
ALU = (0.72, 0.74, 0.77); PCB_G = (0.05, 0.42, 0.22); PCB_B = (0.10, 0.25, 0.55); LDS = (0.10, 0.10, 0.11)
ORANGE = (0.95, 0.45, 0.10); CAM = (0.08, 0.08, 0.09); LENS = (0.15, 0.35, 0.65); BATT = (0.85, 0.75, 0.15)
X_FRONT, X_BACK, HALF_W = 0.090, -0.191, 0.133
WHEEL_R, WHEEL_W, WHEEL_Y = 0.033, 0.020, 0.1435


def robot_body(mount_x, mount_h, pitch):
    m = Mesh()
    cx = (X_FRONT + X_BACK) / 2; L = X_FRONT - X_BACK
    for z in (0.024, 0.066, 0.106):  # three waffle plates
        m.box((cx, 0, z), (L, 2 * HALF_W, 0.004), PLATE)
        # waffle pattern: raised ribs
        for gx in np.linspace(X_BACK + 0.02, X_FRONT - 0.02, 8):
            m.box((gx, 0, z + 0.0025), (0.004, 2 * HALF_W - 0.01, 0.001), PLATE_EDGE)
        for gy in np.linspace(-HALF_W + 0.02, HALF_W - 0.02, 8):
            m.box((cx, gy, z + 0.0025), (L - 0.01, 0.004, 0.001), PLATE_EDGE)
    for x in (X_BACK + 0.012, X_FRONT - 0.012):  # standoffs
        for y in (-HALF_W + 0.012, HALF_W - 0.012):
            m.cyl((x, y, 0.065), (0, 0, 1), 0.004, 0.084, ALU, seg=10)
    m.box((cx - 0.02, 0, 0.044), (0.12, 0.08, 0.035), BATT)              # LiPo battery
    m.box((cx - 0.01, 0, 0.0715), (0.10, 0.10, 0.008), (0.85, 0.85, 0.85))  # OpenCR board
    m.box((cx - 0.01, 0, 0.0765), (0.095, 0.095, 0.002), PCB_B)
    m.box((cx + 0.03, 0.04, 0.0845), (0.085, 0.056, 0.014), PCB_G)       # Raspberry Pi
    for s in (-1, 1):  # Dynamixel XM430 motors
        m.box((0.0, s * 0.085, 0.040), (0.046, 0.034, 0.028), (0.12, 0.12, 0.12))
    # LDS-01 lidar
    m.cyl((-0.064, 0, 0.118), (0, 0, 1), 0.035, 0.020, LDS, seg=32)
    m.cyl((-0.064, 0, 0.134), (0, 0, 1), 0.033, 0.012, (0.22, 0.22, 0.24), seg=32, cap_col=(0.16, 0.16, 0.17))
    m.box((-0.064 + 0.028, 0, 0.134), (0.012, 0.03, 0.008), ORANGE)  # lidar window
    # front Raspberry Pi camera bracket (stock)
    m.box((X_FRONT - 0.004, 0, 0.088), (0.006, 0.026, 0.024), PCB_G)
    # casters
    for y in (-0.07, 0.07):
        m.box((X_BACK + 0.03, y, 0.018), (0.02, 0.02, 0.012), ALU)
        m.sphere((X_BACK + 0.03, y, 0.0095), 0.0095, (0.8, 0.8, 0.82), seg=8)
    # line-camera mast
    m.cyl((mount_x, 0, (0.108 + mount_h) / 2), (0, 0, 1), 0.006, mount_h - 0.108, ALU, seg=14)
    R = rot_y(np.radians(pitch))
    m.box((mount_x, 0, mount_h), (0.024, 0.032, 0.024), CAM, R=R)
    ax = R @ np.array([1.0, 0, 0])
    m.cyl(np.array([mount_x, 0, mount_h]) + ax * 0.015, ax, 0.007, 0.010, CAM, seg=16, cap_col=LENS)
    return m


def wheel(side, angle):
    m = Mesh()
    tread = lambda a: WHEEL_R - 0.0016 * (np.cos(a * 24) > 0.3)
    m.cyl((0, side * WHEEL_Y, WHEEL_R), (0, 1, 0), WHEEL_R, WHEEL_W, TYRE, seg=96, r_fn=tread, rot=angle)
    m.cyl((0, side * (WHEEL_Y + WHEEL_W / 2 + 0.0005), WHEEL_R), (0, 1, 0), 0.022, 0.002, HUB, seg=6, rot=angle)
    return m


# ---------------------------------------------------------------- GL
VS = """#version 330
uniform mat4 mvp; uniform mat4 model;
in vec3 in_pos; in vec3 in_nrm; in vec3 in_col;
out vec3 n; out vec3 col; out vec3 wp;
void main(){ vec4 p = model*vec4(in_pos,1.0); wp=p.xyz; n=mat3(model)*in_nrm; col=in_col; gl_Position=mvp*p; }"""
FS = """#version 330
uniform vec3 light; uniform vec3 eye;
in vec3 n; in vec3 col; in vec3 wp; out vec4 f;
void main(){ vec3 N=normalize(n); vec3 L=normalize(-light); vec3 V=normalize(eye-wp);
 float d=max(dot(N,L),0.0); float s=pow(max(dot(reflect(-L,N),V),0.0),32.0);
 vec3 c=col*(0.32+0.75*d)+vec3(0.25)*s; f=vec4(pow(c,vec3(1.0/1.08)),1.0);}"""
FLOOR_VS = """#version 330
uniform mat4 mvp; in vec3 in_pos; out vec3 wp;
void main(){ wp=in_pos; gl_Position=mvp*vec4(in_pos,1.0);}"""
FLOOR_FS = """#version 330
uniform sampler2D tex; uniform sampler2D shadow; uniform vec2 half_size; uniform vec3 outside;
in vec3 wp; out vec4 f;
void main(){ vec2 q=wp.xy/half_size*0.5+0.5; vec3 c;
 if(q.x<0.0||q.x>1.0||q.y<0.0||q.y>1.0) c=outside; else c=texture(tex,vec2(q.x,1.0-q.y)).rgb;
 float sh=texture(shadow,q).r; f=vec4(c*(1.0-0.55*sh),1.0);}"""
SH_VS = """#version 330
uniform mat4 model; uniform vec3 light; uniform vec2 half_size;
in vec3 in_pos;
void main(){ vec3 p=(model*vec4(in_pos,1.0)).xyz; p=p - light*(p.z/light.z);
 gl_Position=vec4(p.xy/half_size,0.0,1.0);}"""
SH_FS = """#version 330
out vec4 f; void main(){ f=vec4(1.0); }"""
FLAT_VS = """#version 330
uniform mat4 mvp; in vec3 in_pos; void main(){ gl_Position=mvp*vec4(in_pos,1.0);}"""
FLAT_FS = """#version 330
uniform vec4 col; out vec4 f; void main(){ f=col; }"""


def look_at(eye, tgt, up=(0, 0, 1)):
    eye, tgt, up = map(lambda v: np.asarray(v, float), (eye, tgt, up))
    f = _n(tgt - eye); s = _n(np.cross(f, up)); u = np.cross(s, f)
    M = np.eye(4); M[0, :3], M[1, :3], M[2, :3] = s, u, -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def perspective(fovy, aspect, zn, zf):
    t = 1 / np.tan(np.radians(fovy) / 2)
    M = np.zeros((4, 4)); M[0, 0] = t / aspect; M[1, 1] = t
    M[2, 2] = (zf + zn) / (zn - zf); M[2, 3] = 2 * zf * zn / (zn - zf); M[3, 2] = -1
    return M


def ortho(l, r, b, t, zn, zf):
    M = np.eye(4); M[0, 0] = 2 / (r - l); M[1, 1] = 2 / (t - b); M[2, 2] = -2 / (zf - zn)
    M[0, 3] = -(r + l) / (r - l); M[1, 3] = -(t + b) / (t - b); M[2, 3] = -(zf + zn) / (zf - zn)
    return M


def pose_mat(x, y, th, scale=1.0):
    M = np.eye(4); c, s = np.cos(th), np.sin(th)
    M[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) * scale
    M[:3, 3] = (x, y, 0)
    return M


def _f4(M):
    return M.T.astype("f4").tobytes()


class Renderer:
    def __init__(self, floor_rgb, floor_w, floor_h, cam_mount=(0.05, 0.20, 45.0), samples=8):
        backend = None if sys.platform == "win32" else "egl"
        self.ctx = moderngl.create_standalone_context(backend=backend) if backend else moderngl.create_standalone_context()
        ctx = self.ctx
        self.samples = min(samples, ctx.max_samples)
        self.hs = (floor_w / 2, floor_h / 2)
        self.prog = ctx.program(vertex_shader=VS, fragment_shader=FS)
        self.fprog = ctx.program(vertex_shader=FLOOR_VS, fragment_shader=FLOOR_FS)
        self.sprog = ctx.program(vertex_shader=SH_VS, fragment_shader=SH_FS)
        self.flat = ctx.program(vertex_shader=FLAT_VS, fragment_shader=FLAT_FS)
        h, w = floor_rgb.shape[:2]
        self.tex = ctx.texture((w, h), 3, np.ascontiguousarray(floor_rgb).tobytes())
        self.tex.build_mipmaps(); self.tex.anisotropy = 16.0
        self.tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.outside = tuple(np.median(floor_rgb[:20, :20].reshape(-1, 3), 0) / 255)
        E = 12.0
        fl = np.array([[-E, -E, 0, 0, 0], [E, -E, 0, 1, 0], [E, E, 0, 1, 1], [-E, -E, 0, 0, 0], [E, E, 0, 1, 1], [-E, E, 0, 0, 1]], "f4")
        self.floor_vao = ctx.vertex_array(self.fprog, [(ctx.buffer(fl.tobytes()), "3f 2x4", "in_pos")])
        self.sh_tex = ctx.texture((1600, 1280), 1, dtype="f1")
        self.sh_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.sh_fbo_ms = ctx.framebuffer(color_attachments=[ctx.renderbuffer((1600, 1280), 1, samples=min(4, ctx.max_samples))])
        self.sh_fbo = ctx.framebuffer(color_attachments=[self.sh_tex])
        self.mount = cam_mount
        self.body = self._vao(robot_body(*cam_mount).array())
        self.light = _n(np.array([0.55, 0.40, -0.8]))
        self.fbos = {}

    def _vao(self, arr, prog=None):
        prog = prog or self.prog
        buf = self.ctx.buffer(arr.tobytes())
        return self.ctx.vertex_array(prog, [(buf, "3f 3f 3f", "in_pos", "in_nrm", "in_col")]), len(arr), buf

    def _fbo(self, size):
        if size not in self.fbos:
            ms = self.ctx.framebuffer(color_attachments=[self.ctx.renderbuffer(size, 4, samples=self.samples)],
                                      depth_attachment=self.ctx.depth_renderbuffer(size, samples=self.samples))
            out = self.ctx.framebuffer(color_attachments=[self.ctx.renderbuffer(size, 4)])
            self.fbos[size] = (ms, out)
        return self.fbos[size]

    def _robot_parts(self, pose, wheel_ang, scale):
        M = pose_mat(*pose, scale)
        parts = [(self.body, M)]
        for side, ang in zip((1, -1), wheel_ang):
            parts.append((self._vao(wheel(side, ang).array()), M))
        return parts

    def _shadow(self, parts):
        self.sh_fbo_ms.use(); self.sh_fbo_ms.clear(0, 0, 0, 0)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.sprog["light"].value = tuple(self.light); self.sprog["half_size"].value = self.hs
        for (vao, n, buf), M in parts:
            sv = self.ctx.vertex_array(self.sprog, [(buf, "3f 6x4", "in_pos")])
            self.sprog["model"].write(_f4(M)); sv.render(moderngl.TRIANGLES)
            sv.release()
        self.ctx.copy_framebuffer(self.sh_fbo, self.sh_fbo_ms)

    def _ribbon(self, pts, width, z):
        if len(pts) < 2:
            return None
        d = np.diff(pts, axis=0); d = np.vstack([d, d[-1:]]); nrm = _n(np.column_stack([-d[:, 1], d[:, 0]]))
        a = np.column_stack([pts + nrm * width / 2, np.full(len(pts), z)])
        b = np.column_stack([pts - nrm * width / 2, np.full(len(pts), z)])
        tri = np.stack([a[:-1], b[:-1], b[1:], a[:-1], b[1:], a[1:]], 1).reshape(-1, 3)
        return tri.astype("f4")

    def _flat(self, tri, col, mvp, mode=moderngl.TRIANGLES):
        if tri is None or len(tri) == 0:
            return
        buf = self.ctx.buffer(np.ascontiguousarray(tri, "f4").tobytes())
        vao = self.ctx.vertex_array(self.flat, [(buf, "3f", "in_pos")])
        self.flat["mvp"].write(_f4(mvp)); self.flat["col"].value = col
        vao.render(mode); vao.release(); buf.release()

    def render(self, *a, **k):
        # one GL context per scene: make this one current, or the last floor texture wins
        with self.ctx:
            return self._render(*a, **k)

    def _render(self, size, view, proj, eye, pose, wheel_ang, trail=None, trail_col=(0, 0, 1), fit=None,
               frustum=None, robot_scale=1.0, trail_w=0.012, fit_w=0.009):
        ctx = self.ctx
        parts = self._robot_parts(pose, wheel_ang, robot_scale)
        self._shadow(parts)
        ms, out = self._fbo(size)
        ms.use(); ms.clear(1, 1, 1, 1)
        ctx.enable(moderngl.DEPTH_TEST)
        vp = proj @ view
        self.fprog["mvp"].write(_f4(vp)); self.fprog["half_size"].value = self.hs; self.fprog["outside"].value = self.outside
        self.tex.use(0); self.sh_tex.use(1); self.fprog["tex"].value = 0; self.fprog["shadow"].value = 1
        self.floor_vao.render()
        ctx.enable(moderngl.BLEND); ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.depth_mask = False
        if trail is not None:
            self._flat(self._ribbon(trail, trail_w, 0.0015), (*trail_col, 0.95), vp)
        if fit is not None:
            self._flat(self._ribbon(fit, fit_w, 0.0025), (0.85, 0.27, 0.94, 0.95), vp)
        ctx.depth_mask = True
        self.prog["mvp"].write(_f4(vp)); self.prog["light"].value = tuple(self.light); self.prog["eye"].value = tuple(map(float, eye))
        for (vao, n, buf), M in parts:
            self.prog["model"].write(_f4(M)); vao.render(moderngl.TRIANGLES)
        for (vao, n, buf), M in parts[1:]:
            vao.release(); buf.release()
        if frustum is not None:  # (apex(3,), footprint(N,3)) in world coords
            apex, foot = frustum
            ctx.depth_mask = False
            n = len(foot)
            fan = np.array([[apex, foot[i], foot[(i + 1) % n]] for i in range(n)]).reshape(-1, 3)
            self._flat(fan, (0.98, 0.80, 0.08, 0.13), vp)
            gl = np.array([[foot[i], foot[(i + 1) % n]] for i in range(n)]).reshape(-1, 3) + [0, 0, 0.003]
            k = max(1, n // 4)
            ed = np.array([[apex, foot[i]] for i in range(0, n, k)]).reshape(-1, 3)
            self._flat(np.vstack([gl, ed]), (0.79, 0.54, 0.02, 0.9), vp, moderngl.LINES)
            ctx.depth_mask = True
        ctx.disable(moderngl.BLEND)
        ctx.copy_framebuffer(out, ms)
        img = np.frombuffer(out.read(components=3), np.uint8).reshape(size[1], size[0], 3)
        return img[::-1].copy()
