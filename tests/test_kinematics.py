"""Regression tests for FK/IK and Jacobian utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from env.camera_utils import get_camera_parameters, project_point
from env.controllers import KinematicsHelper, infer_actuated_joints
from env.mujoco_env import FrankaPickPlaceEnv

ASSET_ROOT = Path(__file__).resolve().parents[1] / "env" / "mujoco_assets"
COLOR_VALUES = {
    "red": np.array([0.9, 0.2, 0.2]),
    "green": np.array([0.2, 0.9, 0.2]),
    "blue": np.array([0.2, 0.3, 0.9]),
    "yellow": np.array([0.95, 0.9, 0.2]),
    "purple": np.array([0.6, 0.3, 0.8]),
}


@pytest.fixture(scope="module")
def env() -> FrankaPickPlaceEnv:
    xml_path = ASSET_ROOT / "franka_scene.xml"
    if not xml_path.exists():
        pytest.skip("MuJoCo assets missing; see README for download instructions")
    # Require menagerie asset
    menagerie = ASSET_ROOT / "franka_emika_panda" / "panda.xml"
    if not menagerie.exists():
        pytest.skip("Franka menagerie asset missing; skip kinematics tests")
    environment = FrankaPickPlaceEnv(asset_root=ASSET_ROOT, gui=False)
    yield environment
    environment.close()


@pytest.fixture(scope="module")
def kinematics(env: FrankaPickPlaceEnv) -> KinematicsHelper:
    info = infer_actuated_joints(env.model)
    return KinematicsHelper(env.model, joint_info=info)


def _residuals(q: np.ndarray, kinematics: KinematicsHelper) -> tuple[float, float]:
    pos, quat = kinematics.forward_kinematics(q)
    q_star = kinematics.inverse_kinematics(pos, quat, initial_q=q)
    pos_star, quat_star = kinematics.forward_kinematics(q_star)
    pos_res = np.linalg.norm(pos - pos_star)
    quat_dot = abs(np.dot(quat, quat_star))
    quat_dot = np.clip(quat_dot, -1.0, 1.0)
    orient_res = 2.0 * np.arccos(quat_dot)
    return pos_res, orient_res


def test_fk_ik_round_trip(env: FrankaPickPlaceEnv, kinematics: KinematicsHelper) -> None:
    rng = np.random.default_rng(0)
    home = env._home_configuration  # type: ignore[attr-defined]
    limits = env._joint_limits  # type: ignore[attr-defined]
    samples = []
    for _ in range(50):
        sample = home + rng.uniform(-0.1, 0.1, size=home.shape)
        sample = np.clip(sample, limits[:, 0], limits[:, 1])
        samples.append(sample)

    pos_residuals = []
    ori_residuals = []
    for sample in samples:
        pos_res, ori_res = _residuals(sample, kinematics)
        pos_residuals.append(pos_res)
        ori_residuals.append(np.degrees(ori_res))

    assert float(np.mean(pos_residuals)) < 0.002
    assert float(np.percentile(pos_residuals, 95)) < 0.005
    assert float(np.mean(ori_residuals)) < 2.0
    assert float(np.percentile(ori_residuals, 95)) < 3.0


def test_jacobian_matches_finite_difference(env: FrankaPickPlaceEnv, kinematics: KinematicsHelper) -> None:
    home = env._home_configuration  # type: ignore[attr-defined]
    jac_analytic = kinematics.analytic_jacobian(home)
    jac_fd = kinematics.finite_difference_jacobian(home)
    diff = np.abs(jac_analytic - jac_fd)
    assert float(np.mean(diff)) < 1e-3
    assert float(np.max(diff)) < 5e-3


def test_manipulability_non_singular(env: FrankaPickPlaceEnv, kinematics: KinematicsHelper) -> None:
    home = env._home_configuration  # type: ignore[attr-defined]
    manip = kinematics.manipulability(home)
    assert manip > 1e-6


def _color_centroid(image: np.ndarray, color: np.ndarray) -> np.ndarray:
    diff = np.linalg.norm(image - color[None, None, :], axis=2)
    mask = diff < 0.25
    if mask.sum() == 0:
        raise AssertionError("Color mask empty; check rendering configuration")
    coords = np.argwhere(mask)
    centroid = coords.mean(axis=0)
    return np.array([centroid[1], centroid[0]], dtype=np.float64)


def test_camera_reprojection(env: FrankaPickPlaceEnv) -> None:
    env.reset(hindered=False)
    colors = list(COLOR_VALUES.keys())
    spacing = np.linspace(-0.1, 0.1, len(colors))
    for color, offset in zip(colors, spacing):
        addr = env._object_qpos_addrs[color]  # type: ignore[attr-defined]
        env._set_free_joint_pose(addr, np.array([0.5 + offset, offset * 0.5, 0.035]), np.array([1.0, 0.0, 0.0, 0.0]))  # type: ignore[attr-defined]
    env._active_objects = tuple(colors)  # type: ignore[attr-defined]
    mujoco.mj_forward(env.model, env.data)

    params = get_camera_parameters(env.model, "top")
    image = env.render(mode="rgb_array")

    for color in colors:
        site_id = env._object_site_ids[color]  # type: ignore[attr-defined]
        world = env.data.site_xpos[site_id]
        uv = project_point(world, params)
        centroid = _color_centroid(image, COLOR_VALUES[color])
        assert np.linalg.norm(uv - centroid) < 2.0
