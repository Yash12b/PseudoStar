"""GUI regression test — ensures projection/rendering paths don't crash."""
import pytest

from fsoc_tracker.gui.state import OpticalLinkView, TargetView, TerminalView
from fsoc_tracker.simulation.camera.geometry import camera_rotation_matrix
from fsoc_tracker.simulation.camera.projection import project_to_image
from fsoc_tracker.simulation.camera.state import CameraState


class TestProjectToImageContract:
    def test_tuple_input(self):
        cam = CameraState(
            horizontal_fov_deg=60.0, vertical_fov_deg=60.0,
            width=800, height=600,
        )
        pos = cam.position
        intr = cam.intrinsics
        rot = camera_rotation_matrix(cam.pan_deg, cam.tilt_deg, cam.roll_deg)
        result = project_to_image((100.0, 100.0, 500.0), pos, intr, rot)
        assert result.depth > 0

    def test_no_six_args(self):
        cam = CameraState()
        pos = cam.position
        intr = cam.intrinsics
        rot = camera_rotation_matrix(cam.pan_deg, cam.tilt_deg, cam.roll_deg)
        with pytest.raises(TypeError):
            project_to_image(-2000, 0, 0, pos, intr, rot)

    def test_behind_camera(self):
        cam = CameraState()
        pos = cam.position
        intr = cam.intrinsics
        rot = camera_rotation_matrix(cam.pan_deg, cam.tilt_deg, cam.roll_deg)
        result = project_to_image((0.0, 0.0, -100.0), pos, intr, rot)
        assert result.depth <= 0
        assert not result.visible

    def test_grid_projection_no_crash(self):
        cam = CameraState(
            position_x=0.0, position_y=500.0, position_z=-1000.0,
            pan_deg=0.0, tilt_deg=-30.0,
            horizontal_fov_deg=60.0, vertical_fov_deg=60.0,
            width=800, height=600,
        )
        pos = cam.position
        intr = cam.intrinsics
        rot = camera_rotation_matrix(cam.pan_deg, cam.tilt_deg, cam.roll_deg)
        for i in range(-2000, 2001, 400):
            r1 = project_to_image((float(i), 0.0, -2000.0), pos, intr, rot)
            r2 = project_to_image((float(i), 0.0, 2000.0), pos, intr, rot)
            r3 = project_to_image((-2000.0, 0.0, float(i)), pos, intr, rot)
            r4 = project_to_image((2000.0, 0.0, float(i)), pos, intr, rot)

    def test_target_projection(self):
        cam = CameraState(
            position_x=0.0, position_y=500.0, position_z=-1000.0,
            pan_deg=0.0, tilt_deg=-30.0,
        )
        pos = cam.position
        intr = cam.intrinsics
        rot = camera_rotation_matrix(cam.pan_deg, cam.tilt_deg, cam.roll_deg)
        result = project_to_image((1000.0, 1000.0, 100.0), pos, intr, rot)
        assert isinstance(result.depth, float)

    def test_terminal_projection(self):
        cam = CameraState(
            position_x=0.0, position_y=500.0, position_z=-1000.0,
            pan_deg=0.0, tilt_deg=-30.0,
        )
        pos = cam.position
        intr = cam.intrinsics
        rot = camera_rotation_matrix(cam.pan_deg, cam.tilt_deg, cam.roll_deg)
        ta = TerminalView(active=True, world_x=1000, world_y=1000, world_z=50)
        tb = TerminalView(active=True, world_x=700, world_y=800, world_z=200)
        r1 = project_to_image((ta.world_x, ta.world_y, ta.world_z), pos, intr, rot)
        r2 = project_to_image((tb.world_x, tb.world_y, tb.world_z), pos, intr, rot)
        assert r1.depth > 0
        assert r2.depth > 0


class TestTargetViewHasTargetId:
    def test_target_id_field(self):
        t = TargetView(target_id=42, visible=True, world_x=100, world_y=200, world_z=300)
        assert t.target_id == 42

    def test_target_id_default(self):
        t = TargetView()
        assert t.target_id == -1


class TestOpticalLinkView:
    def test_fields(self):
        ol = OpticalLinkView(
            status="LOCKED", angular_error_deg=0.5,
            beam_alignment_percent=95.0, range_m=1000.0,
            link_quality_percent=90.0,
        )
        assert ol.status == "LOCKED"
        assert ol.beam_alignment_percent == 95.0


class TestTerminalView:
    def test_beacon_association(self):
        tb = TerminalView(
            terminal_id="BEACON_3", active=True,
            world_x=700, world_y=800, world_z=200,
        )
        assert tb.terminal_id == "BEACON_3"
        assert tb.active
