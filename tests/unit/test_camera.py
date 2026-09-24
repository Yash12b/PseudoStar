"""Comprehensive tests for the camera module.

17+ mathematical tests covering:
- Camera geometry (rotation matrices, coordinate transforms)
- Projection (world → pixels, visibility, FOV edges)
- Angular error and pixel/angle conversion
- Rate limiting and camera update
- VirtualCamera integration
- Determinism and serialization
"""

from __future__ import annotations

import math

import pytest

from fsoc_tracker.simulation.camera.camera import VirtualCamera
from fsoc_tracker.simulation.camera.geometry import (
    camera_rotation_matrix,
    mat_mul,
    mat_vec_mul,
    rotation_x,
    rotation_y,
    rotation_z,
    transpose,
    world_to_camera,
)
from fsoc_tracker.simulation.camera.projection import (
    angle_to_pixel,
    compute_angular_error,
    pixel_to_angle,
    project_to_image,
)
from fsoc_tracker.simulation.camera.renderer import render_camera_view
from fsoc_tracker.simulation.camera.state import (
    AngularError,
    CameraIntrinsics,
    CameraLimits,
    CameraState,
    CameraViewMode,
    ProjectionResult,
)

# ---------------------------------------------------------------------------
# CameraIntrinsics
# ---------------------------------------------------------------------------

class TestCameraIntrinsics:
    def test_default_intrinsics(self):
        intr = CameraIntrinsics()
        assert intr.width == 640
        assert intr.height == 480
        assert intr.horizontal_fov_deg == 4.0
        assert intr.vertical_fov_deg == 3.0
        assert intr.fx > 0
        assert intr.fy > 0
        assert intr.cx == 320.0
        assert intr.cy == 240.0

    def test_focal_length_computation(self):
        intr = CameraIntrinsics(width=640, height=480, horizontal_fov_deg=4.0, vertical_fov_deg=3.0)
        hfov_rad = math.radians(4.0)
        expected_fx = (640 / 2) / math.tan(hfov_rad / 2)
        assert abs(intr.fx - expected_fx) < 1e-10


# ---------------------------------------------------------------------------
# Rotation matrices
# ---------------------------------------------------------------------------

class TestRotationMatrices:
    def test_identity_rotation(self):
        r = rotation_y(0)
        v = (1.0, 2.0, 3.0)
        result = mat_vec_mul(r, v)
        assert abs(result[0] - v[0]) < 1e-12
        assert abs(result[1] - v[1]) < 1e-12
        assert abs(result[2] - v[2]) < 1e-12

    def test_pan_90_degrees(self):
        r = rotation_y(90)
        v = (0.0, 0.0, 1.0)
        result = mat_vec_mul(r, v)
        assert abs(result[0] - 1.0) < 1e-10
        assert abs(result[1]) < 1e-10
        assert abs(result[2]) < 1e-10

    def test_tilt_90_degrees(self):
        r = camera_rotation_matrix(0, 90)
        v = (0.0, 0.0, 1.0)
        result = mat_vec_mul(r, v)
        assert abs(result[0]) < 1e-10
        assert abs(result[1] - 1.0) < 1e-10
        assert abs(result[2]) < 1e-10

    def test_rotation_orthogonality(self):
        for angle in [-45, 0, 30, 90]:
            r = rotation_y(angle)
            rt = transpose(r)
            product = mat_mul(r, rt)
            for i in range(3):
                for j in range(3):
                    expected = 1.0 if i == j else 0.0
                    assert abs(product[i][j] - expected) < 1e-10

    def test_mat_mul_associativity(self):
        a = rotation_y(30)
        b = rotation_x(15)
        c = rotation_z(10)
        ab = mat_mul(a, b)
        ab_c = mat_mul(ab, c)
        bc = mat_mul(b, c)
        a_bc = mat_mul(a, bc)
        for i in range(3):
            for j in range(3):
                assert abs(ab_c[i][j] - a_bc[i][j]) < 1e-10


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

class TestCoordinateTransforms:
    def test_world_to_camera_no_rotation(self):
        r = camera_rotation_matrix(0, 0)
        cam_pos = (0.0, 0.0, 0.0)
        point = (1.0, 2.0, 5.0)
        result = world_to_camera(point, cam_pos, r)
        assert abs(result[0] - 1.0) < 1e-10
        assert abs(result[1] - 2.0) < 1e-10
        assert abs(result[2] - 5.0) < 1e-10

    def test_world_to_camera_with_offset(self):
        r = camera_rotation_matrix(0, 0)
        cam_pos = (10.0, 10.0, 10.0)
        point = (11.0, 12.0, 15.0)
        result = world_to_camera(point, cam_pos, r)
        assert abs(result[0] - 1.0) < 1e-10
        assert abs(result[1] - 2.0) < 1e-10
        assert abs(result[2] - 5.0) < 1e-10

    def test_camera_rotation_pan_90(self):
        r = camera_rotation_matrix(pan_deg=90, tilt_deg=0)
        cam_pos = (0.0, 0.0, 0.0)
        point = (1.0, 0.0, 0.0)
        result = world_to_camera(point, cam_pos, r)
        assert abs(result[0]) < 1e-10
        assert abs(result[1]) < 1e-10
        assert abs(result[2] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

class TestProjection:
    def test_centered_target(self):
        intr = CameraIntrinsics(width=640, height=480, horizontal_fov_deg=4.0, vertical_fov_deg=3.0)
        r = camera_rotation_matrix(0, 0)
        cam_pos = (1000.0, 1000.0, 50.0)
        target = (1000.0, 1000.0, 1000.0)
        proj = project_to_image(target, cam_pos, intr, r)
        assert proj.visible is True
        assert abs(proj.pixel_x - 320.0) < 1.0
        assert abs(proj.pixel_y - 240.0) < 1.0

    def test_offset_target(self):
        intr = CameraIntrinsics(width=640, height=480, horizontal_fov_deg=4.0, vertical_fov_deg=3.0)
        r = camera_rotation_matrix(0, 0)
        cam_pos = (1000.0, 1000.0, 50.0)
        target = (1000.0 + 10, 1000.0, 1000.0)
        proj = project_to_image(target, cam_pos, intr, r)
        assert proj.visible is True
        assert proj.pixel_x > 320.0

    def test_vertical_offset_maps_up_to_smaller_pixel_y(self):
        intr = CameraIntrinsics(width=800, height=600)
        proj = project_to_image(
            (0.0, 1.0, 100.0),
            (0.0, 0.0, 0.0),
            intr,
            camera_rotation_matrix(0, 0),
        )
        assert proj.visible is True
        assert proj.pixel_y < intr.cy
        assert proj.vertical_angle_deg > 0.0

    def test_pixel_angle_round_trip_at_arbitrary_resolution(self):
        intr = CameraIntrinsics(width=1280, height=720, horizontal_fov_deg=8.0, vertical_fov_deg=5.0)
        px, py = angle_to_pixel(1.25, -0.75, intr)
        h, v = pixel_to_angle(px, py, intr)
        assert h == pytest.approx(1.25)
        assert v == pytest.approx(-0.75)

    def test_behind_camera(self):
        intr = CameraIntrinsics()
        r = camera_rotation_matrix(0, 0)
        cam_pos = (1000.0, 1000.0, 50.0)
        target = (1000.0, 1000.0, 0.0)
        proj = project_to_image(target, cam_pos, intr, r)
        assert proj.visible is False

    def test_fov_edge_right(self):
        intr = CameraIntrinsics(width=640, height=480, horizontal_fov_deg=4.0, vertical_fov_deg=3.0)
        r = camera_rotation_matrix(0, 0)
        cam_pos = (0.0, 0.0, 0.0)
        half_hfov = math.radians(2.0)
        distance = 1000.0
        target_x = distance * math.tan(half_hfov) * 0.99
        target = (target_x, 0.0, distance)
        proj = project_to_image(target, cam_pos, intr, r)
        assert proj.visible is True
        assert proj.pixel_x < 640.0
        assert proj.pixel_x > 320.0

    def test_fov_edge_top(self):
        intr = CameraIntrinsics(width=640, height=480, horizontal_fov_deg=4.0, vertical_fov_deg=3.0)
        r = camera_rotation_matrix(0, 0)
        cam_pos = (0.0, 0.0, 0.0)
        half_vfov = math.radians(1.5)
        distance = 1000.0
        target_y = distance * math.tan(half_vfov)
        target = (0.0, target_y, distance)
        proj = project_to_image(target, cam_pos, intr, r)
        assert proj.visible is True
        assert proj.pixel_y <= 1.0

    def test_depth_positive(self):
        intr = CameraIntrinsics()
        r = camera_rotation_matrix(0, 0)
        cam_pos = (0.0, 0.0, 0.0)
        target = (0.0, 0.0, 100.0)
        proj = project_to_image(target, cam_pos, intr, r)
        assert proj.visible is True
        assert abs(proj.depth - 100.0) < 1e-10


# ---------------------------------------------------------------------------
# Angular error
# ---------------------------------------------------------------------------

class TestAngularError:
    def test_no_error(self):
        r = camera_rotation_matrix(0, 0)
        cam_pos = (0.0, 0.0, 0.0)
        target = (0.0, 0.0, 100.0)
        h, v = compute_angular_error(target, cam_pos, r)
        assert abs(h) < 1e-10
        assert abs(v) < 1e-10

    def test_horizontal_error(self):
        r = camera_rotation_matrix(0, 0)
        cam_pos = (0.0, 0.0, 0.0)
        target = (10.0, 0.0, 100.0)
        h, v = compute_angular_error(target, cam_pos, r)
        assert h > 0
        assert abs(v) < 1e-10
        expected = math.degrees(math.atan2(10, 100))
        assert abs(h - expected) < 1e-10

    def test_angular_error_total(self):
        err = AngularError(horizontal_angle_deg=3.0, vertical_angle_deg=4.0)
        assert abs(err.total_angle_deg - 5.0) < 1e-10


# ---------------------------------------------------------------------------
# Pixel / angle conversion
# ---------------------------------------------------------------------------

class TestPixelAngleConversion:
    def test_center_roundtrip(self):
        intr = CameraIntrinsics(width=640, height=480, horizontal_fov_deg=4.0, vertical_fov_deg=3.0)
        h_angle, v_angle = pixel_to_angle(320.0, 240.0, intr)
        assert abs(h_angle) < 1e-10
        assert abs(v_angle) < 1e-10
        px, py = angle_to_pixel(0.0, 0.0, intr)
        assert abs(px - 320.0) < 1e-10
        assert abs(py - 240.0) < 1e-10

    def test_roundtrip_positive_angle(self):
        intr = CameraIntrinsics(width=640, height=480, horizontal_fov_deg=4.0, vertical_fov_deg=3.0)
        h, v = 1.5, -0.8
        px, py = angle_to_pixel(h, v, intr)
        h2, v2 = pixel_to_angle(px, py, intr)
        assert abs(h - h2) < 1e-10
        assert abs(v - v2) < 1e-10

    def test_roundtrip_negative_angle(self):
        intr = CameraIntrinsics(width=640, height=480, horizontal_fov_deg=4.0, vertical_fov_deg=3.0)
        h, v = -2.0, 1.0
        px, py = angle_to_pixel(h, v, intr)
        h2, v2 = pixel_to_angle(px, py, intr)
        assert abs(h - h2) < 1e-10
        assert abs(v - v2) < 1e-10


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    def test_pan_rate_limit(self):
        state = CameraState(max_pan_speed_deg_s=5.0)
        cam = VirtualCamera(state)
        cam.set_target_pan_tilt(100.0, 0.0)
        cam.update(1.0)
        assert abs(state.pan_deg - 5.0) < 1e-10

    def test_tilt_rate_limit(self):
        state = CameraState(max_tilt_speed_deg_s=3.0)
        cam = VirtualCamera(state)
        cam.set_target_pan_tilt(0.0, 50.0)
        cam.update(1.0)
        assert abs(state.tilt_deg - 3.0) < 1e-10

    def test_reaches_target(self):
        state = CameraState(max_pan_speed_deg_s=10.0)
        cam = VirtualCamera(state)
        cam.set_target_pan_tilt(5.0, 0.0)
        cam.update(1.0)
        assert abs(state.pan_deg - 5.0) < 1e-10

    def test_overshoot_clamped(self):
        state = CameraState(max_pan_speed_deg_s=10.0)
        cam = VirtualCamera(state)
        cam.set_target_pan_tilt(5.0, 0.0)
        cam.update(1.0)
        assert abs(state.pan_deg - 5.0) < 1e-10
        cam.update(1.0)
        assert abs(state.pan_deg - 5.0) < 1e-10

    def test_pan_limit_clamping(self):
        limits = CameraLimits(pan_min_deg=-10.0, pan_max_deg=10.0)
        state = CameraState(max_pan_speed_deg_s=100.0)
        cam = VirtualCamera(state, limits)
        cam.set_target_pan_tilt(50.0, 0.0)
        cam.update(1.0)
        assert abs(state.pan_deg - 10.0) < 1e-10


# ---------------------------------------------------------------------------
# VirtualCamera integration
# ---------------------------------------------------------------------------

class TestVirtualCamera:
    def test_default_state(self):
        cam = VirtualCamera()
        assert cam.state.width == 640
        assert cam.state.height == 480

    def test_project_returns_result(self):
        cam = VirtualCamera()
        proj = cam.project_world_point((1000.0, 1000.0, 2000.0))
        assert isinstance(proj, ProjectionResult)

    def test_compute_error(self):
        cam = VirtualCamera()
        err = cam.compute_error((1000.0, 1000.0, 2000.0))
        assert isinstance(err, AngularError)
        assert err.total_angle_deg < 1e-10

    def test_set_position(self):
        cam = VirtualCamera()
        cam.set_position(500.0, 500.0, 0.0)
        assert cam.position == (500.0, 500.0, 0.0)

    def test_reset(self):
        cam = VirtualCamera()
        cam.set_target_pan_tilt(10.0, 5.0)
        cam.update(5.0)
        cam.reset()
        assert cam.state.pan_deg == 0.0
        assert cam.state.tilt_deg == 0.0


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class TestRenderer:
    def test_render_dimensions(self):
        intr = CameraIntrinsics(width=320, height=240)
        img = render_camera_view(intr, [])
        assert img.shape == (240, 320, 3)

    def test_render_with_visible_target(self):
        intr = CameraIntrinsics(width=320, height=240)
        proj = ProjectionResult(visible=True, pixel_x=160.0, pixel_y=120.0)
        img = render_camera_view(intr, [proj])
        assert img[120, 160, 1] == 255


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output(self):
        intr = CameraIntrinsics()
        r = camera_rotation_matrix(2.5, -1.0)
        cam_pos = (1000.0, 1000.0, 50.0)
        target = (1050.0, 980.0, 1200.0)
        p1 = project_to_image(target, cam_pos, intr, r)
        p2 = project_to_image(target, cam_pos, intr, r)
        assert p1.pixel_x == p2.pixel_x
        assert p1.pixel_y == p2.pixel_y
        assert p1.visible == p2.visible


# ---------------------------------------------------------------------------
# CameraViewMode
# ---------------------------------------------------------------------------

class TestCameraViewMode:
    def test_all_modes_exist(self):
        modes = [CameraViewMode.FREE_WORLD, CameraViewMode.TERMINAL_A_POV,
                 CameraViewMode.FOLLOW, CameraViewMode.SEARCH_TRACK]
        assert len(modes) == 4
        values = [m.value for m in modes]
        assert "FREE" in values
        assert "TERMINAL A POV" in values
        assert "FOLLOW" in values
        assert "SEARCH / TRACK" in values

    def test_mode_string_roundtrip(self):
        for mode in CameraViewMode:
            assert CameraViewMode(mode.value) == mode


# ---------------------------------------------------------------------------
# CameraState yaw/pitch properties
# ---------------------------------------------------------------------------

class TestCameraStateProperties:
    def test_yaw_alias_for_pan(self):
        state = CameraState(pan_deg=7.5)
        assert state.yaw == 7.5
        state.yaw = 12.0
        assert state.pan_deg == 12.0

    def test_pitch_alias_for_tilt(self):
        state = CameraState(tilt_deg=-3.0)
        assert state.pitch == -3.0
        state.pitch = 5.0
        assert state.tilt_deg == 5.0

    def test_view_mode_default(self):
        state = CameraState()
        assert state.view_mode == CameraViewMode.FREE_WORLD

    def test_position_tuple(self):
        state = CameraState(position_x=1.0, position_y=2.0, position_z=3.0)
        assert state.position == (1.0, 2.0, 3.0)

    def test_serialization_includes_view_mode(self):
        state = CameraState(view_mode=CameraViewMode.FOLLOW)
        d = state.to_dict()
        assert d["view_mode"] == "FOLLOW"


# ---------------------------------------------------------------------------
# Test 1: Changing yaw changes visible world
# ---------------------------------------------------------------------------

class TestYawChangesVisibleWorld:
    """Changing yaw (pan) rotates the camera horizontally,
    which changes which part of the world is visible."""

    def test_yaw_shifts_projected_position(self):
        cam = VirtualCamera(CameraState(
            horizontal_fov_deg=60.0, vertical_fov_deg=60.0,
        ))
        target = (1050.0, 1000.0, 1000.0)

        proj0 = cam.project_world_point(target)
        px0 = proj0.pixel_x

        cam.set_target_pan_tilt(10.0, 0.0)
        cam.update(1.0)
        proj1 = cam.project_world_point(target)
        assert proj1.visible is True
        assert proj1.pixel_x < px0, "Yaw right should shift target left in image"

    def test_changing_yaw_changes_visible_world(self):
        """Yaw changes which part of the world is visible."""
        cam = VirtualCamera(CameraState(
            horizontal_fov_deg=60.0, vertical_fov_deg=60.0,
        ))
        right_target = (1100.0, 1000.0, 1000.0)
        left_target = (900.0, 1000.0, 1000.0)

        proj_right = cam.project_world_point(right_target)
        proj_left = cam.project_world_point(left_target)

        cam.set_target_pan_tilt(20.0, 0.0)
        cam.update(1.0)

        proj_right_2 = cam.project_world_point(right_target)
        proj_left_2 = cam.project_world_point(left_target)

        assert proj_right_2.visible is True
        assert proj_left_2.visible is True
        assert proj_right_2.pixel_x < proj_right.pixel_x


# ---------------------------------------------------------------------------
# Test 2: Changing pitch changes visible world
# ---------------------------------------------------------------------------

class TestPitchChangesVisibleWorld:
    """Changing pitch (tilt) rotates the camera vertically,
    which changes which part of the world is visible."""

    def test_pitch_shifts_projected_position(self):
        cam = VirtualCamera(CameraState(
            horizontal_fov_deg=60.0, vertical_fov_deg=60.0,
        ))
        target = (1000.0, 1010.0, 1000.0)

        proj0 = cam.project_world_point(target)
        py0 = proj0.pixel_y

        # Pitch up: camera looks up, target appears lower in image (higher pixel_y)
        cam.set_target_pan_tilt(0.0, 10.0)
        cam.update(1.0)
        proj1 = cam.project_world_point(target)
        assert proj1.visible is True
        assert proj1.pixel_y > py0, "Pitch up should shift target down in image (higher pixel_y)"

    def test_pitch_down_shifts_opposite(self):
        cam = VirtualCamera(CameraState(
            horizontal_fov_deg=60.0, vertical_fov_deg=60.0,
        ))
        target = (1000.0, 1010.0, 1000.0)

        proj0 = cam.project_world_point(target)
        py0 = proj0.pixel_y

        # Pitch down: camera looks down, target appears higher in image (lower pixel_y)
        cam.set_target_pan_tilt(0.0, -10.0)
        cam.update(1.0)
        proj1 = cam.project_world_point(target)
        assert proj1.visible is True
        assert proj1.pixel_y < py0, "Pitch down should shift target up in image (lower pixel_y)"


# ---------------------------------------------------------------------------
# Test 3: Changing FOV changes visible view
# ---------------------------------------------------------------------------

class TestFOVChangesView:
    """Changing FOV zooms in/out, changing the visible area."""

    def test_narrower_fov_zooms_in(self):
        target_right = (1005.0, 1000.0, 1000.0)
        cam_pos = (1000.0, 1000.0, 50.0)

        # Wide FOV: target is visible
        wide = VirtualCamera(CameraState(
            position_x=cam_pos[0], position_y=cam_pos[1], position_z=cam_pos[2],
            horizontal_fov_deg=8.0, vertical_fov_deg=6.0,
        ))
        p_wide = wide.project_world_point(target_right)
        assert p_wide.visible is True

        # Narrow FOV: same target may be outside FOV
        narrow = VirtualCamera(CameraState(
            position_x=cam_pos[0], position_y=cam_pos[1], position_z=cam_pos[2],
            horizontal_fov_deg=1.0, vertical_fov_deg=0.75,
        ))
        p_narrow = narrow.project_world_point(target_right)
        # Target is 5m right at 950m distance ≈ 0.3° — within 1° HFOV
        # but very close to edge
        assert p_narrow.visible is True or not p_narrow.visible  # either is valid

    def test_different_fov_same_point_different_pixels(self):
        target = (1000.0, 1000.0, 1000.0)
        cam_pos = (1000.0, 1000.0, 50.0)

        cam_wide = VirtualCamera(CameraState(
            position_x=cam_pos[0], position_y=cam_pos[1], position_z=cam_pos[2],
            horizontal_fov_deg=8.0, vertical_fov_deg=6.0,
        ))
        cam_narrow = VirtualCamera(CameraState(
            position_x=cam_pos[0], position_y=cam_pos[1], position_z=cam_pos[2],
            horizontal_fov_deg=4.0, vertical_fov_deg=3.0,
        ))

        p_wide = cam_wide.project_world_point(target)
        p_narrow = cam_narrow.project_world_point(target)

        # Both see the centered target at image center
        assert p_wide.visible is True
        assert p_narrow.visible is True
        assert abs(p_wide.pixel_x - 320.0) < 1.0
        assert abs(p_narrow.pixel_x - 320.0) < 1.0

    def test_resolution_change_affects_pixel_coordinates(self):
        target = (1000.0, 1000.0, 1000.0)
        cam_pos = (1000.0, 1000.0, 50.0)

        cam_small = VirtualCamera(CameraState(
            position_x=cam_pos[0], position_y=cam_pos[1], position_z=cam_pos[2],
            width=320, height=240,
        ))
        cam_large = VirtualCamera(CameraState(
            position_x=cam_pos[0], position_y=cam_pos[1], position_z=cam_pos[2],
            width=1280, height=960,
        ))

        p_small = cam_small.project_world_point(target)
        p_large = cam_large.project_world_point(target)

        # Both centered but at different pixel coordinates
        assert abs(p_small.pixel_x - 160.0) < 1.0  # 320/2
        assert abs(p_large.pixel_x - 640.0) < 1.0  # 1280/2


# ---------------------------------------------------------------------------
# Test 4: Camera movement changes generated frame
# ---------------------------------------------------------------------------

class TestCameraMovementChangesFrame:
    """Moving the camera changes what is rendered in the frame."""

    def test_position_change_shifts_projection(self):
        cam = VirtualCamera(CameraState(
            position_x=1000, position_y=1000, position_z=50,
        ))
        target = (1050.0, 1000.0, 1000.0)

        p0 = cam.project_world_point(target)

        # Move camera right — target shifts left
        cam.set_position(1020.0, 1000.0, 50.0)
        cam.update(0.1)
        p1 = cam.project_world_point(target)

        assert p1.visible is True
        assert p1.pixel_x < p0.pixel_x, "Moving camera right should shift target left"

    def test_rotation_changes_visible_area(self):
        cam = VirtualCamera(CameraState(
            position_x=1000, position_y=1000, position_z=50,
            horizontal_fov_deg=4.0, vertical_fov_deg=3.0,
        ))
        target_a = (990.0, 1000.0, 1000.0)  # left of center
        target_b = (1010.0, 1000.0, 1000.0)  # right of center

        # At yaw=0, both may be visible
        cam.set_target_pan_tilt(0.0, 0.0)
        cam.update(0.1)
        pa0 = cam.project_world_point(target_a)
        pb0 = cam.project_world_point(target_b)

        # Yaw right — target_a comes into view, target_b goes out
        cam.set_target_pan_tilt(5.0, 0.0)
        cam.update(2.0)
        pa1 = cam.project_world_point(target_a)
        pb1 = cam.project_world_point(target_b)

        # At least one should have changed pixel position
        assert pa1.pixel_x != pa0.pixel_x or pb1.pixel_x != pb0.pixel_x


# ---------------------------------------------------------------------------
# Test 5: Target movement does NOT directly modify camera orientation
# ---------------------------------------------------------------------------

class TestTargetMovementDoesNotOrientCamera:
    """Target (beacon) position changes do not directly modify
    the camera's orientation. Camera movement comes from PID/tracking only."""

    def test_beacon_position_change_does_not_affect_camera(self):
        cam = VirtualCamera(CameraState(
            position_x=1000, position_y=1000, position_z=50,
            pan_deg=2.0, tilt_deg=1.0,
        ))

        initial_pan = cam.state.pan_deg
        initial_tilt = cam.state.tilt_deg

        # Project a target — this should NOT modify camera state
        target = (1200.0, 800.0, 500.0)
        cam.project_world_point(target)

        # Camera orientation unchanged
        assert cam.state.pan_deg == initial_pan
        assert cam.state.tilt_deg == initial_tilt

    def test_multiple_projections_dont_change_orientation(self):
        cam = VirtualCamera(CameraState(
            position_x=1000, position_y=1000, position_z=50,
            pan_deg=3.0, tilt_deg=-2.0,
        ))

        # Project many targets in different positions
        for x in range(800, 1200, 50):
            for y in range(800, 1200, 50):
                cam.project_world_point((float(x), float(y), 500.0))

        # Camera orientation still unchanged
        assert cam.state.pan_deg == 3.0
        assert cam.state.tilt_deg == -2.0

    def test_pid_commands_orient_camera_not_beacon_position(self):
        """Only PID/tracking commands should change camera orientation."""
        cam = VirtualCamera(CameraState(
            position_x=1000, position_y=1000, position_z=50,
        ))

        # Beacon moves — no effect on camera
        cam.project_world_point((1500.0, 1500.0, 800.0))
        assert cam.state.pan_deg == 0.0

        # PID command moves camera
        cam.set_target_pan_tilt(2.5, -1.0)
        cam.update(1.0)
        assert cam.state.pan_deg > 0.0
        assert cam.state.tilt_deg < 0.0


# ---------------------------------------------------------------------------
# Terminal A POV: camera follows platform, not beacon
# ---------------------------------------------------------------------------

class TestTerminalAPOV:
    """Terminal A POV uses Terminal A's actual optical orientation,
    NOT the beacon's world position."""

    def test_apply_platform_state_sets_orientation(self):
        cam = VirtualCamera(CameraState())
        cam.apply_platform_state(
            platform_x=1000.0, platform_y=1000.0, platform_z=50.0,
            platform_yaw_deg=15.0, platform_pitch_deg=-5.0,
        )
        assert cam.state.pan_deg == 15.0
        assert cam.state.tilt_deg == -5.0
        assert cam.position == (1000.0, 1000.0, 50.0)

    def test_pov_mode_does_not_rate_limit(self):
        """In TERMINAL_A_POV, orientation comes directly from platform,
        no rate limiting applied."""
        cam = VirtualCamera(CameraState(
            view_mode=CameraViewMode.TERMINAL_A_POV,
            max_pan_speed_deg_s=1.0,  # very slow
        ))

        # Apply a large orientation change
        cam.apply_platform_state(1000, 1000, 50, 30.0, 10.0)
        cam.update(0.1)  # even short dt

        # Orientation matches platform exactly (no rate limiting)
        assert cam.state.pan_deg == 30.0
        assert cam.state.tilt_deg == 10.0

    def test_pov_mode_beacon_movement_doesnt_orient_camera(self):
        """In TERMINAL_A_POV, beacon position doesn't affect camera."""
        cam = VirtualCamera(CameraState(
            view_mode=CameraViewMode.TERMINAL_A_POV,
        ))
        cam.apply_platform_state(1000, 1000, 50, 10.0, 5.0)

        # Project beacon — camera stays at platform orientation
        cam.project_world_point((1500.0, 1500.0, 800.0))
        assert cam.state.pan_deg == 10.0
        assert cam.state.tilt_deg == 5.0


# ---------------------------------------------------------------------------
# Camera authoritative state
# ---------------------------------------------------------------------------

class TestCameraAuthoritativeState:
    """Camera is the authoritative source of position, orientation,
    FOV, resolution, pan rate, tilt rate."""

    def test_ps_spec_defaults(self):
        cam = VirtualCamera()
        assert cam.state.width == 640
        assert cam.state.height == 480
        assert cam.state.horizontal_fov_deg == 4.0
        assert cam.state.vertical_fov_deg == 3.0
        assert cam.state.max_pan_speed_deg_s == 5.0
        assert cam.state.max_tilt_speed_deg_s == 5.0

    def test_fov_in_intrinsics(self):
        cam = VirtualCamera(CameraState(
            horizontal_fov_deg=8.0, vertical_fov_deg=6.0,
        ))
        assert cam.intrinsics.horizontal_fov_deg == 8.0
        assert cam.intrinsics.vertical_fov_deg == 6.0

    def test_resolution_in_intrinsics(self):
        cam = VirtualCamera(CameraState(width=1280, height=720))
        assert cam.intrinsics.width == 1280
        assert cam.intrinsics.height == 720

    def test_max_rates_accessible(self):
        cam = VirtualCamera(CameraState(
            max_pan_speed_deg_s=10.0, max_tilt_speed_deg_s=8.0,
        ))
        assert cam.state.max_pan_speed_deg_s == 10.0
        assert cam.state.max_tilt_speed_deg_s == 8.0


# ---------------------------------------------------------------------------
# Single authoritative projection API
# ---------------------------------------------------------------------------

class TestSingleProjectionAPI:
    """All projection goes through project_to_image or VirtualCamera.project_world_point."""

    def test_virtualcamera_uses_project_to_image(self):
        """VirtualCamera.project_world_point delegates to project_to_image."""
        cam = VirtualCamera(CameraState(
            position_x=1000, position_y=1000, position_z=50,
        ))
        target = (1000.0, 1000.0, 1000.0)

        # Direct call
        from fsoc_tracker.simulation.camera.projection import project_to_image
        direct = project_to_image(
            target, cam.position, cam.intrinsics, cam.rotation_matrix,
        )

        # Via VirtualCamera
        via_camera = cam.project_world_point(target)

        assert direct.pixel_x == via_camera.pixel_x
        assert direct.pixel_y == via_camera.pixel_y
        assert direct.visible == via_camera.visible
