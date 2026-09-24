"""Global world/radar view — 3D configuration workspace.

Secondary to the camera-tracking workspace. Represents the actual
simulation environment for scenario configuration and observation.

Rendering:
  - Ground grid with axis markers
  - World boundary wireframe
  - Terminal A (diamond marker + orientation arrow)
  - Terminal B / beacons (color-coded, primary beacon distinguished)
  - Distractors (dim, small)
  - Camera frustum wireframe (approximate FOV cone)
  - Trajectory trails for all targets
  - Connection line between Terminal A and primary beacon

Interaction:
  - Left-drag: orbit
  - Right-drag: pan
  - Scroll: zoom
  - Left-click: select target
  - Double-click: focus on target
  - Right-click: context menu (inspect, set beacon, follow, place Terminal A)
  - Keyboard: C=center, T=toggle trajectories, F=toggle frustum
"""

from __future__ import annotations

import math

from fsoc_tracker.gui.state import ApplicationViewState
from fsoc_tracker.gui.theme import Colors
from fsoc_tracker.simulation.camera.geometry import camera_rotation_matrix
from fsoc_tracker.simulation.camera.projection import project_to_image
from fsoc_tracker.simulation.camera.state import CameraState, CameraViewMode

try:
    from PySide6.QtCore import QPointF, Qt, Signal
    from PySide6.QtGui import QAction, QColor, QPainter, QPainterPath, QPen
    from PySide6.QtWidgets import QFrame, QMenu
except ImportError:
    from PyQt5.QtCore import QPointF, Qt  # type: ignore
    from PyQt5.QtCore import pyqtSignal as Signal
    from PyQt5.QtGui import QAction, QColor, QPainter, QPainterPath, QPen  # type: ignore
    from PyQt5.QtWidgets import QFrame, QMenu  # type: ignore


# Fixed simulation camera position (PS-spec: mounted on Terminal A)
_SIM_CAM_X = 1000.0
_SIM_CAM_Y = 1000.0
_SIM_CAM_Z = 50.0

# Trajectory preview: sample positions along each target's path
_TRAJ_PREVIEW_SAMPLES = 80
_TRAJ_PREVIEW_DURATION = 30.0  # seconds to preview


class WorldViewWidget(QFrame):
    target_selected = Signal(int)
    terminal_selected = Signal()
    beacon_dragged = Signal(int, float, float, float)
    terminal_placed = Signal(float, float, float)
    scan_started = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("WorldViewport")
        self.setMinimumSize(300, 300)
        self._state: ApplicationViewState | None = None
        self._error_count = 0

        # Free exploration camera
        self.free_camera = CameraState(
            position_x=1000.0,
            position_y=1500.0,
            position_z=-800.0,
            pan_deg=0.0,
            tilt_deg=-25.0,
            horizontal_fov_deg=60.0,
            vertical_fov_deg=60.0,
            max_pan_speed_deg_s=5.0,
            max_tilt_speed_deg_s=5.0,
        )

        self.last_pos = None
        self.mouse_btn = None

        # Beacon drag state
        self._dragging_beacon = False
        self._drag_beacon_id: int | None = None

        # Display toggles
        self._show_trajectories = True
        self._show_frustum = True
        self._show_grid = True

        # Follow mode
        self._follow_target_id: int | None = None

    def _get_render_camera(self, s: ApplicationViewState) -> CameraState:
        render_cam = self.free_camera
        if s.camera_mode == CameraViewMode.TERMINAL_A_POV.value and hasattr(s, "terminal_a") or s.camera_mode == CameraViewMode.FOLLOW.value and hasattr(s, "terminal_a"):
            ta = s.terminal_a
            render_cam = CameraState(
                position_x=ta.world_x, position_y=ta.world_y, position_z=ta.world_z,
                pan_deg=ta.yaw_deg, tilt_deg=ta.pitch_deg,
                horizontal_fov_deg=ta.hfov_deg, vertical_fov_deg=ta.vfov_deg,
            )
        render_cam.width = self.width()
        render_cam.height = self.height()
        return render_cam

    # ------------------------------------------------------------------
    #  Mouse interaction
    # ------------------------------------------------------------------
    def mousePressEvent(self, event):
        self.last_pos = event.pos()
        self.mouse_btn = event.button()

        s = self._state
        if not s:
            return

        # Left click: select target
        if self.mouse_btn == Qt.LeftButton:
            render_cam = self._get_render_camera(s)
            intrinsics = render_cam.intrinsics
            rot_mat = camera_rotation_matrix(render_cam.pan_deg, render_cam.tilt_deg, render_cam.roll_deg)
            cam_pos = render_cam.position

            best_id = None
            best_dist = float("inf")

            for t in s.targets_all:
                if not t.visible:
                    continue
                res = project_to_image((t.world_x, t.world_y, t.world_z), cam_pos, intrinsics, rot_mat)
                if res.depth > 0:
                    r = max(4.0, min(16.0, 500.0 / res.depth))
                    dx = res.pixel_x - event.pos().x()
                    dy = res.pixel_y - event.pos().y()
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist <= r + 8.0 and dist < best_dist:
                        best_dist = dist
                        best_id = t.target_id

            if best_id is not None:
                self.target_selected.emit(best_id)
                self.last_pos = None
                return

            # Also check Terminal A click
            ta_res = project_to_image(
                (s.terminal_a.world_x, s.terminal_a.world_y, s.terminal_a.world_z),
                cam_pos, intrinsics, rot_mat,
            )
            if ta_res.depth > 0:
                dx = ta_res.pixel_x - event.pos().x()
                dy = ta_res.pixel_y - event.pos().y()
                if math.sqrt(dx*dx + dy*dy) < 16:
                    self.terminal_selected.emit()
                    self.last_pos = None
                    return

        # Middle click: start scan
        if self.mouse_btn == Qt.MiddleButton:
            self.scan_started.emit()
            self.last_pos = None
            return

    def mouseDoubleClickEvent(self, event) -> None:
        """Double-click: focus camera on clicked target."""
        if event.button() != Qt.LeftButton:
            return
        s = self._state
        if not s:
            return

        render_cam = self._get_render_camera(s)
        intrinsics = render_cam.intrinsics
        rot_mat = camera_rotation_matrix(render_cam.pan_deg, render_cam.tilt_deg, render_cam.roll_deg)
        cam_pos = render_cam.position

        for t in s.targets_all:
            if not t.visible:
                continue
            res = project_to_image((t.world_x, t.world_y, t.world_z), cam_pos, intrinsics, rot_mat)
            if res.depth > 0:
                r = max(4.0, min(16.0, 500.0 / res.depth))
                dx = res.pixel_x - event.pos().x()
                dy = res.pixel_y - event.pos().y()
                if math.sqrt(dx*dx + dy*dy) <= r + 8.0:
                    self._focus_on(t.world_x, t.world_y, t.world_z)
                    return

    def mouseMoveEvent(self, event):
        if not self.last_pos:
            return
        dx = event.pos().x() - self.last_pos.x()
        dy = event.pos().y() - self.last_pos.y()
        self.last_pos = event.pos()

        s = self._state

        # Beacon drag mode
        if self._dragging_beacon and self._drag_beacon_id is not None and s:
            render_cam = self._get_render_camera(s)
            camera_rotation_matrix(render_cam.pan_deg, render_cam.tilt_deg, render_cam.roll_deg)

            for t in s.targets_all:
                if t.target_id == self._drag_beacon_id:
                    world_dx = dx * 2.0
                    world_dy = -dy * 2.0
                    new_x = max(0.0, min(2000.0, t.world_x + world_dx))
                    new_y = max(0.0, min(2000.0, t.world_y + world_dy))
                    self.beacon_dragged.emit(self._drag_beacon_id, new_x, new_y, t.world_z)
                    break
            return

        # Free camera orbit
        if self.mouse_btn == Qt.LeftButton:
            self.free_camera.pan_deg += dx * 0.2
            self.free_camera.tilt_deg -= dy * 0.2
        elif self.mouse_btn == Qt.RightButton:
            yaw_rad = math.radians(self.free_camera.pan_deg)
            self.free_camera.position_x -= dx * 2.0 * math.cos(yaw_rad)
            self.free_camera.position_z += dx * 2.0 * math.sin(yaw_rad)
            self.free_camera.position_y += dy * 2.0

        self.update()

    def mouseReleaseEvent(self, event):
        self._dragging_beacon = False
        self._drag_beacon_id = None
        self.last_pos = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        yaw_rad = math.radians(self.free_camera.pan_deg)
        pitch_rad = math.radians(-self.free_camera.tilt_deg)
        self.free_camera.position_x += delta * 0.5 * math.sin(yaw_rad) * math.cos(pitch_rad)
        self.free_camera.position_y -= delta * 0.5 * math.sin(pitch_rad)
        self.free_camera.position_z += delta * 0.5 * math.cos(yaw_rad) * math.cos(pitch_rad)
        self.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_C:
            self._center_on_world()
        elif key == Qt.Key_T:
            self._show_trajectories = not self._show_trajectories
            self.update()
        elif key == Qt.Key_F:
            self._show_frustum = not self._show_frustum
            self.update()
        elif key == Qt.Key_G:
            self._show_grid = not self._show_grid
            self.update()
        elif key == Qt.Key_Escape:
            self._follow_target_id = None
            self.update()
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:
        """Right-click context menu for object interaction."""
        s = self._state
        if not s:
            return

        render_cam = self._get_render_camera(s)
        intrinsics = render_cam.intrinsics
        rot_mat = camera_rotation_matrix(render_cam.pan_deg, render_cam.tilt_deg, render_cam.roll_deg)
        cam_pos = render_cam.position

        # Find clicked target
        clicked_id = None
        for t in s.targets_all:
            if not t.visible:
                continue
            res = project_to_image((t.world_x, t.world_y, t.world_z), cam_pos, intrinsics, rot_mat)
            if res.depth > 0:
                r = max(4.0, min(16.0, 500.0 / res.depth))
                dx = res.pixel_x - event.pos().x()
                dy = res.pixel_y - event.pos().y()
                if math.sqrt(dx*dx + dy*dy) <= r + 8.0:
                    clicked_id = t.target_id
                    break

        menu = QMenu(self)

        if clicked_id is not None:
            target = next((t for t in s.targets_all if t.target_id == clicked_id), None)
            is_beacon = s.designated_beacon_id == clicked_id

            # Inspect
            inspect_act = QAction(f"Inspect OBJ-{clicked_id}", self)
            inspect_act.triggered.connect(lambda: self.target_selected.emit(clicked_id))
            menu.addAction(inspect_act)

            # Focus
            if target:
                focus_act = QAction("Focus Camera", self)
                focus_act.triggered.connect(
                    lambda: self._focus_on(target.world_x, target.world_y, target.world_z)
                )
                menu.addAction(focus_act)

            # Follow
            follow_act = QAction("Follow Target", self)
            follow_act.triggered.connect(lambda: self._start_follow(clicked_id))
            menu.addAction(follow_act)

            # Set as primary beacon
            if not is_beacon:
                beacon_act = QAction("Set as Primary Beacon", self)
                beacon_act.triggered.connect(lambda: self.target_selected.emit(clicked_id))
                menu.addAction(beacon_act)

            menu.addSeparator()

        # Place Terminal A at clicked world position
        place_ta = QAction("Place Terminal A Here", self)
        place_ta.triggered.connect(lambda: self._place_terminal_a(event.pos()))
        menu.addAction(place_ta)

        # Center camera
        center_act = QAction("Center View", self)
        center_act.triggered.connect(self._center_on_world)
        menu.addAction(center_act)

        menu.exec(event.globalPos())

    # ------------------------------------------------------------------
    #  Actions
    # ------------------------------------------------------------------
    def _focus_on(self, x: float, y: float, z: float) -> None:
        """Move free camera to look at a world point."""
        cam = self.free_camera
        dx = x - cam.position_x
        dy = y - cam.position_y
        dz = z - cam.position_z
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < 1.0:
            return
        # Point camera at target
        cam.pan_deg = math.degrees(math.atan2(dx, dz))
        cam.tilt_deg = math.degrees(math.atan2(-dy, math.sqrt(dx*dx + dz*dz)))
        self.update()

    def _start_follow(self, target_id: int) -> None:
        """Start following a target."""
        self._follow_target_id = target_id
        self.update()

    def _center_on_world(self) -> None:
        """Reset free camera to see entire world."""
        self.free_camera.position_x = 1000.0
        self.free_camera.position_y = 1500.0
        self.free_camera.position_z = -800.0
        self.free_camera.pan_deg = 0.0
        self.free_camera.tilt_deg = -25.0
        self.update()

    def _place_terminal_a(self, screen_pos: QPointF) -> None:
        """Place Terminal A at the clicked world position (approximate on ground plane)."""
        # Simplified: place at clicked position projected to z=50
        # In a full implementation, we'd do a ray-ground intersection
        s = self._state
        if not s:
            return
        render_cam = self._get_render_camera(s)
        intrinsics = render_cam.intrinsics
        rot_mat = camera_rotation_matrix(render_cam.pan_deg, render_cam.tilt_deg, render_cam.roll_deg)
        cam_pos = render_cam.position

        # Ray from screen point
        cam_x = (screen_pos.x() - intrinsics.cx) / intrinsics.fx
        cam_y = -(screen_pos.y() - intrinsics.cy) / intrinsics.fy
        # In camera space, ray direction is (cam_x, cam_y, 1) normalized
        ray_len = math.sqrt(cam_x*cam_x + cam_y*cam_y + 1.0)
        ray = (cam_x / ray_len, cam_y / ray_len, 1.0 / ray_len)

        # Transform to world space
        R = rot_mat
        world_dx = R[0][0]*ray[0] + R[0][1]*ray[1] + R[0][2]*ray[2]
        world_dy = R[1][0]*ray[0] + R[1][1]*ray[1] + R[1][2]*ray[2]
        world_dz = R[2][0]*ray[0] + R[2][1]*ray[1] + R[2][2]*ray[2]

        # Intersect with y=1000 (Terminal A height)
        if abs(world_dy) > 1e-6:
            t = (1000.0 - cam_pos[1]) / world_dy
            if t > 0:
                wx = cam_pos[0] + t * world_dx
                wz = cam_pos[2] + t * world_dz
                s.terminal_a.world_x = max(0.0, min(2000.0, wx))
                s.terminal_a.world_y = 1000.0
                s.terminal_a.world_z = max(0.0, min(2000.0, wz))
                s.add_event(f"Terminal A placed at ({wx:.0f}, 1000, {wz:.0f})", "INFO")
                self.terminal_placed.emit(
                    float(s.terminal_a.world_x),
                    float(s.terminal_a.world_y),
                    float(s.terminal_a.world_z),
                )

    def update_state(self, state: ApplicationViewState) -> None:
        self._state = state

        # Follow target
        if self._follow_target_id is not None and state:
            for t in state.targets_all:
                if t.target_id == self._follow_target_id and t.visible:
                    cam = self.free_camera
                    cam.position_x = t.world_x
                    cam.position_y = t.world_y + 200.0
                    cam.position_z = t.world_z - 400.0
                    cam.pan_deg = 0.0
                    cam.tilt_deg = -20.0
                    break

        self.update()

    # ------------------------------------------------------------------
    #  Rendering
    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        try:
            self._paint(painter)
        except Exception:
            self._error_count += 1
            try:
                painter.fillRect(self.rect(), QColor("#1A0000"))
                painter.setPen(QPen(QColor("#FF4444"), 1))
                painter.drawText(self.rect(), Qt.AlignCenter, f"Render error #{self._error_count}")
            except Exception:
                pass
        finally:
            if painter.isActive():
                painter.end()

    def _paint(self, painter: QPainter) -> None:
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0A0E17"))

        s = self._state
        render_cam = self._get_render_camera(s) if s else self.free_camera
        intrinsics = render_cam.intrinsics
        rot_mat = camera_rotation_matrix(render_cam.pan_deg, render_cam.tilt_deg, render_cam.roll_deg)
        cam_pos = render_cam.position

        # --- Ground grid ---
        if self._show_grid:
            self._draw_grid(painter, cam_pos, intrinsics, rot_mat)

        # --- World axes (at origin) ---
        self._draw_axes(painter, cam_pos, intrinsics, rot_mat)

        # --- Camera frustum ---
        if self._show_frustum and s:
            self._draw_camera_frustum(painter, s, cam_pos, intrinsics, rot_mat)

        # --- Scan sphere wave ---
        if s and s.scan_active and s.scan_radius > 0:
            self._draw_sphere_wave(painter, s, cam_pos, intrinsics, rot_mat)

        # --- Trajectory trails ---
        if self._show_trajectories and s:
            self._draw_trajectories(painter, s, cam_pos, intrinsics, rot_mat)

        # --- Targets / beacons ---
        if s:
            self._draw_targets(painter, s, cam_pos, intrinsics, rot_mat)

        # --- Terminal A ---
        if s and hasattr(s, "terminal_a") and s.terminal_a.active:
            self._draw_terminal_a(painter, s, cam_pos, intrinsics, rot_mat)

        # --- Terminal B ---
        if s and hasattr(s, "terminal_b") and s.terminal_b.active:
            self._draw_terminal_b(painter, s, cam_pos, intrinsics, rot_mat)

        # --- Connection line (Terminal A to primary beacon) ---
        if s and s.beacon_connected and s.terminal_a.active and s.terminal_b.active:
            self._draw_connection(painter, s, cam_pos, intrinsics, rot_mat)

        # --- HUD overlay ---
        self._draw_hud(painter, s)

    def _draw_grid(self, painter: QPainter, cam_pos, intrinsics, rot_mat) -> None:
        """Draw ground grid on Y=0 plane."""
        pen_grid = QPen(QColor(Colors.GRID), 1)
        painter.setPen(pen_grid)
        for i in range(-2000, 2001, 200):
            res1 = project_to_image((float(i), 0.0, -2000.0), cam_pos, intrinsics, rot_mat)
            res2 = project_to_image((float(i), 0.0, 2000.0), cam_pos, intrinsics, rot_mat)
            if res1.depth > 0 and res2.depth > 0:
                painter.drawLine(QPointF(res1.pixel_x, res1.pixel_y), QPointF(res2.pixel_x, res2.pixel_y))
            res3 = project_to_image((-2000.0, 0.0, float(i)), cam_pos, intrinsics, rot_mat)
            res4 = project_to_image((2000.0, 0.0, float(i)), cam_pos, intrinsics, rot_mat)
            if res3.depth > 0 and res4.depth > 0:
                painter.drawLine(QPointF(res3.pixel_x, res3.pixel_y), QPointF(res4.pixel_x, res4.pixel_y))

    def _draw_axes(self, painter: QPainter, cam_pos, intrinsics, rot_mat) -> None:
        """Draw world axes at origin: X=red, Y=green, Z=blue."""
        origin = project_to_image((0.0, 0.0, 0.0), cam_pos, intrinsics, rot_mat)
        if origin.depth <= 0:
            return

        axes = [
            ((200.0, 0.0, 0.0), "#FF4444", "X"),
            ((0.0, 200.0, 0.0), "#44FF44", "Y"),
            ((0.0, 0.0, 200.0), "#4488FF", "Z"),
        ]
        for (x, y, z), color, label in axes:
            end = project_to_image((x, y, z), cam_pos, intrinsics, rot_mat)
            if end.depth > 0:
                painter.setPen(QPen(QColor(color), 2))
                painter.drawLine(QPointF(origin.pixel_x, origin.pixel_y), QPointF(end.pixel_x, end.pixel_y))
                painter.setPen(QPen(QColor(color), 1))
                font = painter.font()
                font.setPointSize(8)
                painter.setFont(font)
                painter.drawText(QPointF(end.pixel_x + 4, end.pixel_y - 4), label)

    def _draw_camera_frustum(
        self, painter: QPainter, s: ApplicationViewState,
        cam_pos, intrinsics, rot_mat,
    ) -> None:
        """Draw the simulation camera's FOV frustum in the world."""
        # Simulation camera position and orientation
        sx, sy, sz = _SIM_CAM_X, _SIM_CAM_Y, _SIM_CAM_Z
        pan_rad = math.radians(s.camera.pan_deg)
        tilt_rad = math.radians(s.camera.tilt_deg)
        hfov_rad = math.radians(s.camera.hfov_deg)
        vfov_rad = math.radians(s.camera.vfov_deg)

        # Camera look direction
        look_x = math.sin(pan_rad) * math.cos(tilt_rad)
        look_y = math.sin(tilt_rad)
        look_z = math.cos(pan_rad) * math.cos(tilt_rad)

        # Camera right and up vectors (approximate)
        right_x = math.cos(pan_rad)
        right_z = -math.sin(pan_rad)
        up_x = -math.sin(pan_rad) * math.sin(tilt_rad)
        up_y = math.cos(tilt_rad)
        up_z = -math.cos(pan_rad) * math.sin(tilt_rad)

        near_dist = 10.0
        far_dist = 500.0

        half_hfov = hfov_rad / 2.0
        half_vfov = vfov_rad / 2.0

        # 4 corners at far distance
        corners = []
        for sign_h in (-1, 1):
            for sign_v in (-1, 1):
                dx = look_x + sign_h * math.sin(half_hfov) * right_x + sign_v * math.sin(half_vfov) * up_x
                dy = look_y + sign_h * math.sin(half_hfov) * 0 + sign_v * math.sin(half_vfov) * up_y
                dz = look_z + sign_h * math.sin(half_hfov) * right_z + sign_v * math.sin(half_vfov) * up_z
                # Normalize
                ln = math.sqrt(dx*dx + dy*dy + dz*dz)
                dx, dy, dz = dx/ln, dy/ln, dz/ln
                fx = sx + dx * far_dist
                fy = sy + dy * far_dist
                fz = sz + dz * far_dist
                corners.append((fx, fy, fz))

        # Project corners
        proj_corners = []
        for (fx, fy, fz) in corners:
            res = project_to_image((fx, fy, fz), cam_pos, intrinsics, rot_mat)
            if res.depth > 0:
                proj_corners.append(QPointF(res.pixel_x, res.pixel_y))

        # Draw frustum as wireframe (4 edges from near to far, 4 far edges)
        if len(proj_corners) == 4:
            pen = QPen(QColor(255, 255, 0, 100), 1, Qt.DashLine)
            painter.setPen(pen)
            # Far face
            for i in range(4):
                j = (i + 1) % 4
                painter.drawLine(proj_corners[i], proj_corners[j])

        # Draw near plane (small square near camera)
        near_corners = []
        for sign_h in (-1, 1):
            for sign_v in (-1, 1):
                dx = look_x + sign_h * math.sin(half_hfov) * right_x + sign_v * math.sin(half_vfov) * up_x
                dy = look_y + sign_h * math.sin(half_hfov) * 0 + sign_v * math.sin(half_vfov) * up_y
                dz = look_z + sign_h * math.sin(half_hfov) * right_z + sign_v * math.sin(half_vfov) * up_z
                ln = math.sqrt(dx*dx + dy*dy + dz*dz)
                dx, dy, dz = dx/ln, dy/ln, dz/ln
                nx = sx + dx * near_dist
                ny = sy + dy * near_dist
                nz = sz + dz * near_dist
                res = project_to_image((nx, ny, nz), cam_pos, intrinsics, rot_mat)
                if res.depth > 0:
                    near_corners.append(QPointF(res.pixel_x, res.pixel_y))

        if len(near_corners) == 4:
            pen = QPen(QColor(255, 255, 0, 150), 1)
            painter.setPen(pen)
            for i in range(4):
                j = (i + 1) % 4
                painter.drawLine(near_corners[i], near_corners[j])

        # Draw 4 edges connecting near to far
        if len(proj_corners) == 4 and len(near_corners) == 4:
            pen = QPen(QColor(255, 255, 0, 60), 1, Qt.DotLine)
            painter.setPen(pen)
            for i in range(4):
                painter.drawLine(near_corners[i], proj_corners[i])

        # Camera position marker
        cam_res = project_to_image((sx, sy, sz), cam_pos, intrinsics, rot_mat)
        if cam_res.depth > 0:
            painter.setPen(QPen(QColor(Colors.ACCENT), 2))
            painter.setBrush(QColor(Colors.ACCENT))
            painter.drawEllipse(QPointF(cam_res.pixel_x, cam_res.pixel_y), 5, 5)
            painter.setPen(QPen(QColor(Colors.ACCENT), 1))
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            painter.drawText(QPointF(cam_res.pixel_x + 8, cam_res.pixel_y - 4), "CAM")

        # Look direction line
        look_end = project_to_image(
            (sx + look_x * 200, sy + look_y * 200, sz + look_z * 200),
            cam_pos, intrinsics, rot_mat,
        )
        if look_end.depth > 0 and cam_res.depth > 0:
            painter.setPen(QPen(QColor(255, 255, 0, 180), 1))
            painter.drawLine(QPointF(cam_res.pixel_x, cam_res.pixel_y), QPointF(look_end.pixel_x, look_end.pixel_y))

    def _draw_trajectories(
        self, painter: QPainter, s: ApplicationViewState,
        cam_pos, intrinsics, rot_mat,
    ) -> None:
        """Draw target motion trails (fading polylines, newest brightest)."""
        if not s.show_trail:
            return
        for t in s.targets_all:
            if not t.visible or not t.trail:
                continue
            pts = t.trail
            if len(pts) < 2:
                continue
            is_primary = t.target_id == s.designated_beacon_id
            base = QColor("#FFB300") if is_primary else QColor("#4488FF")
            for i in range(1, len(pts)):
                a = pts[i - 1]
                b = pts[i]
                if len(a) < 3 or len(b) < 3:
                    continue
                ra = project_to_image((a[0], a[1], a[2]), cam_pos, intrinsics, rot_mat)
                rb = project_to_image((b[0], b[1], b[2]), cam_pos, intrinsics, rot_mat)
                if ra.depth <= 0 or rb.depth <= 0:
                    continue
                frac = i / len(pts)
                col = QColor(base)
                col.setAlpha(int(40 + 180 * frac))
                painter.setPen(QPen(col, 2 if is_primary else 1))
                painter.drawLine(QPointF(ra.pixel_x, ra.pixel_y),
                                 QPointF(rb.pixel_x, rb.pixel_y))
        self._draw_camera_trail(painter, s, cam_pos, intrinsics, rot_mat)

    def _draw_camera_trail(
        self, painter: QPainter, s: ApplicationViewState,
        cam_pos, intrinsics, rot_mat,
    ) -> None:
        """Draw the camera boresight sweep path (search trail).

        Shows where the camera has pointed: the spiral/search pattern
        is directly observable instead of inferred from motion.
        """
        if not s.show_trail:
            return
        pts = s.camera_trail
        if not pts or len(pts) < 2:
            return
        for i in range(1, len(pts)):
            a, b = pts[i - 1], pts[i]
            if len(a) < 3 or len(b) < 3:
                continue
            ra = project_to_image((a[0], a[1], a[2]), cam_pos, intrinsics, rot_mat)
            rb = project_to_image((b[0], b[1], b[2]), cam_pos, intrinsics, rot_mat)
            if ra.depth <= 0 or rb.depth <= 0:
                continue
            frac = i / len(pts)
            col = QColor(Colors.CROSSHAIR)
            col.setAlpha(int(30 + 150 * frac))
            painter.setPen(QPen(col, 1, Qt.DotLine))
            painter.drawLine(QPointF(ra.pixel_x, ra.pixel_y),
                             QPointF(rb.pixel_x, rb.pixel_y))

    def _draw_targets(
        self, painter: QPainter, s: ApplicationViewState,
        cam_pos, intrinsics, rot_mat,
    ) -> None:
        """Draw all targets: beacons (prominent), distractors (dim)."""
        for t in s.targets_all:
            if not t.visible:
                continue
            res = project_to_image((t.world_x, t.world_y, t.world_z), cam_pos, intrinsics, rot_mat)
            if res.depth <= 0:
                continue
            r = max(4.0, min(16.0, 500.0 / res.depth))

            is_selected = s.selected_object_id == t.target_id
            is_beacon = s.designated_beacon_id == t.target_id
            is_tracked = s.tracked_target_id == t.target_id

            if is_beacon:
                # Primary beacon: large, bright, with ring
                painter.setPen(QPen(QColor("#FFB300"), 3))
                painter.setBrush(QColor(Colors.TARGET))
                r = max(r, 10.0)
                painter.drawEllipse(QPointF(res.pixel_x, res.pixel_y), r, r)
                # Outer ring
                painter.setPen(QPen(QColor("#FFB300"), 1))
                painter.setBrush(QColor(0, 0, 0, 0))
                painter.drawEllipse(QPointF(res.pixel_x, res.pixel_y), r + 6, r + 6)
                # Label
                painter.setPen(QPen(QColor("#FFB300"), 1))
                font = painter.font()
                font.setPointSize(9)
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(QPointF(res.pixel_x + r + 5, res.pixel_y + 3), f"BEACON-{t.target_id}")
                # ★ marker
                painter.drawText(QPointF(res.pixel_x - 3, res.pixel_y - r - 5), "★")
            elif is_selected:
                painter.setPen(QPen(QColor("#FFFFFF"), 2))
                painter.setBrush(QColor("#4488FF"))
                painter.drawEllipse(QPointF(res.pixel_x, res.pixel_y), r, r)
                painter.setPen(QPen(QColor("#FFFFFF"), 1))
                font = painter.font()
                font.setPointSize(8)
                painter.setFont(font)
                painter.drawText(QPointF(res.pixel_x + r + 3, res.pixel_y + 3), f"OBJ-{t.target_id}")
            elif is_tracked:
                painter.setPen(QPen(QColor("#00FF88"), 2))
                painter.setBrush(QColor("#00FF8844"))
                painter.drawEllipse(QPointF(res.pixel_x, res.pixel_y), r, r)
                painter.setPen(QPen(QColor("#00FF88"), 1))
                font = painter.font()
                font.setPointSize(8)
                painter.setFont(font)
                painter.drawText(QPointF(res.pixel_x + r + 3, res.pixel_y + 3), f"OBJ-{t.target_id}")
            else:
                # Distractor / unselected: dim, small
                painter.setPen(QPen(QColor("#555555"), 1))
                painter.setBrush(QColor("#666666"))
                painter.drawEllipse(QPointF(res.pixel_x, res.pixel_y), max(r * 0.7, 2), max(r * 0.7, 2))

    def _draw_terminal_a(
        self, painter: QPainter, s: ApplicationViewState,
        cam_pos, intrinsics, rot_mat,
    ) -> None:
        """Draw Terminal A with orientation arrow."""
        ta = s.terminal_a
        res = project_to_image((ta.world_x, ta.world_y, ta.world_z), cam_pos, intrinsics, rot_mat)
        if res.depth <= 0:
            return
        r = max(8.0, min(24.0, 1200.0 / res.depth))

        # Diamond marker
        painter.setPen(QPen(QColor(Colors.ACCENT), 2))
        painter.setBrush(QColor(Colors.ACCENT))
        path = QPainterPath()
        path.moveTo(QPointF(res.pixel_x, res.pixel_y - r))
        path.lineTo(QPointF(res.pixel_x + r, res.pixel_y))
        path.lineTo(QPointF(res.pixel_x, res.pixel_y + r))
        path.lineTo(QPointF(res.pixel_x - r, res.pixel_y))
        path.closeSubpath()
        painter.drawPath(path)

        # Orientation arrow (pointing in yaw direction)
        yaw_rad = math.radians(ta.yaw_deg)
        arrow_len = r * 2.0
        ax = ta.world_x + math.sin(yaw_rad) * arrow_len
        az = ta.world_z + math.cos(yaw_rad) * arrow_len
        arrow_res = project_to_image((ax, ta.world_y, az), cam_pos, intrinsics, rot_mat)
        if arrow_res.depth > 0:
            painter.setPen(QPen(QColor(Colors.ACCENT), 2))
            painter.drawLine(QPointF(res.pixel_x, res.pixel_y), QPointF(arrow_res.pixel_x, arrow_res.pixel_y))
            # Arrowhead
            painter.setBrush(QColor(Colors.ACCENT))
            painter.drawEllipse(QPointF(arrow_res.pixel_x, arrow_res.pixel_y), 3, 3)

        # Label
        painter.setPen(QPen(QColor(Colors.ACCENT), 1))
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QPointF(res.pixel_x + r + 5, res.pixel_y - 4), "TERM-A")

    def _draw_terminal_b(
        self, painter: QPainter, s: ApplicationViewState,
        cam_pos, intrinsics, rot_mat,
    ) -> None:
        """Draw Terminal B (beacon) with square marker."""
        tb = s.terminal_b
        res = project_to_image((tb.world_x, tb.world_y, tb.world_z), cam_pos, intrinsics, rot_mat)
        if res.depth <= 0:
            return
        r = max(5.0, min(16.0, 800.0 / res.depth))

        is_primary = s.designated_beacon_id is not None
        color = "#FF6600" if is_primary else "#FF4444"
        painter.setPen(QPen(QColor(color), 2))
        painter.setBrush(QColor(color))
        painter.drawRect(res.pixel_x - r/2, res.pixel_y - r/2, r, r)

        painter.setPen(QPen(QColor(color), 1))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(QPointF(res.pixel_x + r/2 + 3, res.pixel_y + 3), f"TERM-B ({tb.terminal_id})")

    def _draw_connection(
        self, painter: QPainter, s: ApplicationViewState,
        cam_pos, intrinsics, rot_mat,
    ) -> None:
        """Draw the optical beam from Terminal A's optical axis direction.

        The beam does NOT point at the true beacon position. It points
        where Terminal A is actually aiming (camera pan/tilt). This
        visually demonstrates the consequence of misalignment.
        """
        ta = s.terminal_a
        tb = s.terminal_b

        # Terminal A position
        ta_pos = (ta.world_x, ta.world_y, ta.world_z)
        res_a = project_to_image(ta_pos, cam_pos, intrinsics, rot_mat)
        if res_a.depth <= 0:
            return

        # Beam direction = Terminal A's actual optical axis.
        # NEVER aim at the true beacon position: when the camera is
        # misaligned the beam must visibly miss.
        beam_dx, beam_dy, beam_dz = ta.optical_axis

        # Beam length: extend to beacon distance (visual only)
        beam_length = s.optical_link.range_m if s.optical_link.range_m > 0 else 500.0
        beam_end = (
            ta_pos[0] + beam_dx * beam_length,
            ta_pos[1] + beam_dy * beam_length,
            ta_pos[2] + beam_dz * beam_length,
        )
        res_end = project_to_image(beam_end, cam_pos, intrinsics, rot_mat)
        if res_end.depth <= 0:
            return

        # Beam color based on link status
        status = s.optical_link.status
        if status == "LOCKED":
            beam_color = QColor(0, 255, 255, 200)  # Bright cyan
        elif status == "DEGRADED":
            beam_color = QColor(255, 165, 0, 150)   # Orange
        elif status in ("LOST", "REACQUIRING"):
            beam_color = QColor(255, 50, 50, 100)   # Dim red
        else:
            beam_color = QColor(100, 100, 100, 60)  # Gray

        # Draw beam line
        painter.setPen(QPen(beam_color, 2))
        painter.drawLine(QPointF(res_a.pixel_x, res_a.pixel_y), QPointF(res_end.pixel_x, res_end.pixel_y))

        # Draw beam endpoint (where the beam actually lands)
        painter.setPen(QPen(beam_color, 1))
        painter.setBrush(QColor(0, 0, 0, 0))
        painter.drawEllipse(QPointF(res_end.pixel_x, res_end.pixel_y), 6, 6)

        # Draw beacon position with a different marker to show offset
        res_b = project_to_image((tb.world_x, tb.world_y, tb.world_z), cam_pos, intrinsics, rot_mat)
        if res_b.depth > 0:
            # Show angular error as distance between beam endpoint and beacon
            painter.setPen(QPen(QColor(255, 100, 100, 120), 1, Qt.DotLine))
            painter.drawLine(QPointF(res_end.pixel_x, res_end.pixel_y), QPointF(res_b.pixel_x, res_b.pixel_y))

            # Angular error label
            err = s.optical_link.angular_error_deg
            if err > 0.01:
                mid_x = (res_end.pixel_x + res_b.pixel_x) / 2
                mid_y = (res_end.pixel_y + res_b.pixel_y) / 2
                painter.setPen(QPen(QColor(255, 100, 100), 1))
                font = painter.font()
                font.setPointSize(7)
                painter.setFont(font)
                painter.drawText(QPointF(mid_x + 4, mid_y - 4), f"ERR: {err:.2f}°")

    def _draw_sphere_wave(
        self, painter: QPainter, s: ApplicationViewState,
        cam_pos, intrinsics, rot_mat,
    ) -> None:
        """Draw expanding sphere wave from Terminal A."""
        ta = s.terminal_a
        if not ta.active:
            return
        center = (ta.world_x, ta.world_y, ta.world_z)
        radius = s.scan_radius
        num_points = 64
        points = []
        for i in range(num_points):
            angle = 2.0 * math.pi * i / num_points
            sx = center[0] + radius * math.cos(angle)
            sy = center[1]
            sz = center[2] + radius * math.sin(angle)
            res = project_to_image((sx, sy, sz), cam_pos, intrinsics, rot_mat)
            if res.depth > 0:
                points.append(QPointF(res.pixel_x, res.pixel_y))

        if len(points) > 2:
            alpha = max(30, min(150, int(255 * (1.0 - radius / 2000.0))))
            painter.setPen(QPen(QColor(0, 255, 255, alpha), 2))
            painter.setBrush(QColor(0, 255, 255, alpha // 4))
            path = QPainterPath()
            path.moveTo(points[0])
            for p in points[1:]:
                path.lineTo(p)
            path.closeSubpath()
            painter.drawPath(path)

    def _draw_hud(self, painter: QPainter, s: ApplicationViewState | None) -> None:
        """Draw HUD overlay text."""
        if not s:
            return

        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)

        y = 18
        # View mode
        painter.setPen(QPen(QColor(Colors.ACCENT), 1))
        painter.drawText(10, y, f"VIEW: {s.camera_mode}")
        y += 16

        # Toggle states
        toggles = []
        if self._show_frustum:
            toggles.append("FRUSTUM")
        if self._show_trajectories:
            toggles.append("TRAJ")
        if self._show_grid:
            toggles.append("GRID")
        if self._follow_target_id is not None:
            toggles.append(f"FOLLOW-{self._follow_target_id}")
        if toggles:
            painter.setPen(QPen(QColor(Colors.MUTED), 1))
            painter.drawText(10, y, " | ".join(toggles))
            y += 16

        # Scan status
        if s.scan_active:
            painter.setPen(QPen(QColor("#00FF88"), 1))
            painter.drawText(10, y, f"SCANNING: {s.scan_radius:.0f}m  |  {s.scan_beacons_found} beacons found")
            y += 16

        # Atmospheric status
        if s.atmospheric_disturbance != "clear":
            painter.setPen(QPen(QColor("#FF8800"), 1))
            painter.drawText(10, y, f"ATMO: {s.atmospheric_disturbance.upper()}")
            y += 16

        # Object counts
        n_targets = sum(1 for t in s.targets_all if t.visible)
        n_beacons = sum(1 for t in s.targets_all if t.visible and t.target_id == s.designated_beacon_id)
        painter.setPen(QPen(QColor(Colors.MUTED), 1))
        font.setBold(False)
        painter.setFont(font)
        painter.drawText(10, y, f"OBJ: {n_targets}  |  BEACON: {'yes' if n_beacons else 'none'}")

        # Selected object info (bottom-right)
        if s.selected_object_id is not None:
            target = next((t for t in s.targets_all if t.target_id == s.selected_object_id), None)
            if target:
                painter.setPen(QPen(QColor("#FFFFFF"), 1))
                font.setPointSize(8)
                painter.setFont(font)
                info = f"OBJ-{target.target_id}: ({target.world_x:.0f}, {target.world_y:.0f}, {target.world_z:.0f})"
                painter.drawText(self.width() - 250, self.height() - 10, info)

        # Keyboard hints (bottom-left)
        painter.setPen(QPen(QColor(Colors.MUTED), 1))
        font.setPointSize(7)
        painter.setFont(font)
        painter.drawText(10, self.height() - 8, "Keys: C=center T=trajectories F=frustum G=grid Esc=unfollow")
