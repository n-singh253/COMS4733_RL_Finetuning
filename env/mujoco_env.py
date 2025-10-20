"""MuJoCo Franka Panda pick-and-place environment with safety checks."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency for structured observations
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None  # type: ignore
    spaces = None  # type: ignore

try:  # pragma: no cover - MuJoCo is required at runtime
    import mujoco
except Exception as exc:  # pragma: no cover
    mujoco = None  # type: ignore

try:  # pragma: no cover - viewer optional
    import mujoco.viewer as mujoco_viewer
except Exception:  # pragma: no cover
    mujoco_viewer = None  # type: ignore

from .controllers import infer_actuated_joints

DEFAULT_XML = "franka_scene.xml"
_HIDDEN_POSE = np.array([2.0, 2.0, -1.0, 1.0, 0.0, 0.0, 0.0])
_OBJECT_COLORS = ("red", "green", "blue", "yellow", "purple")


@dataclass(slots=True)
class StepResult:
    """Container returned by :meth:`FrankaPickPlaceEnv.step`."""

    observation: Dict[str, np.ndarray]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, float]


class FrankaPickPlaceEnv:
    """Gym-like wrapper around a MuJoCo Franka Panda manipulation scene."""

    def __init__(
        self,
        asset_root: Path | str = Path("env/mujoco_assets"),
        *,
        gui: bool = False,
        width: int = 224,
        height: int = 224,
        seed: Optional[int] = 0,
    ) -> None:
        if mujoco is None:  # pragma: no cover - handled at runtime
            raise ImportError(
                "MuJoCo is required for FrankaPickPlaceEnv. Install 'mujoco' and place the"
                " Franka assets in env/mujoco_assets/."
            )

        self.asset_root = Path(asset_root)
        self.gui = gui
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)

        xml_path = self.asset_root / DEFAULT_XML
        if not xml_path.exists():
            raise FileNotFoundError(
                f"MuJoCo XML '{xml_path}' not found. Refer to the README for asset setup instructions."
            )

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self._default_qpos = self.model.qpos0.copy()
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

        self.step_dt = 0.04
        self.control_rate_hz = 1.0 / self.step_dt
        self.max_steps = 200
        self.success_height = 0.3
        self.workspace_extent = np.array([0.25, 0.25])
        self.bin_position = np.array([0.8, 0.25, 0.08])
        self.bin_radius = 0.08

        self.viewer: Optional[object] = None
        if self.gui:
            if mujoco_viewer is None:  # pragma: no cover - viewer optional
                raise RuntimeError("MuJoCo viewer is unavailable; install mujoco>=2.3.5 or disable GUI mode.")
            self.viewer = mujoco_viewer.launch_passive(self.model, self.data)

        self._gripper_site_id = self._get_site_id("panda_gripper_site")
        self._object_body_ids = {color: self._get_body_id(f"object_body_{color}") for color in _OBJECT_COLORS}
        self._object_site_ids = {color: self._get_site_id(f"object_site_{color}") for color in _OBJECT_COLORS}
        self._object_qpos_addrs = {color: self._free_joint_qpos_addr(body_id) for color, body_id in self._object_body_ids.items()}
        self._occluder_body = self._get_body_id("occluder_body")
        self._occluder_qpos_addr = self._free_joint_qpos_addr(self._occluder_body)
        self._light_id = self._get_light_id("top_light")

        joint_info = infer_actuated_joints(self.model)
        self._joint_ids = joint_info.joint_ids
        self._joint_qpos_indices = joint_info.qpos_indices
        self._joint_dof_indices = joint_info.dof_indices
        self._joint_limits = joint_info.limits
        self._home_configuration = self._default_qpos[self._joint_qpos_indices]

        self._active_objects: Tuple[str, ...] = tuple(_OBJECT_COLORS[:3])
        self._target_color = "red"
        self._instruction = ""
        self._hindered = False
        self._elapsed_steps = 0
        self._last_action = np.zeros(7, dtype=np.float64)
        self._max_abs_joint = 0.0
        self._max_abs_velocity = 0.0

        if spaces is not None:
            self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(7,), dtype=np.float32)
            self.observation_space = spaces.Dict(
                {
                    "rgb_static": spaces.Box(low=0.0, high=1.0, shape=(self.height, self.width, 3), dtype=np.float32),
                    "proprio": spaces.Box(low=-np.inf, high=np.inf, shape=(len(self._joint_qpos_indices),), dtype=np.float32),
                }
            )

    # ------------------------------------------------------------------
    # Reset and stepping
    # ------------------------------------------------------------------
    def reset(self, *, hindered: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._default_qpos
        self.data.qvel[:] = 0.0

        self._hindered = hindered
        self._elapsed_steps = 0
        self._max_abs_joint = 0.0
        self._max_abs_velocity = 0.0

        self._randomize_robot_pose()
        self._active_objects = self._randomize_objects()
        self._target_color = self.rng.choice(self._active_objects)
        self._instruction = f"Pick up the {self._target_color} sphere and place it in the goal bin."
        self._apply_hindered_modifications()

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {
            "instruction": self._instruction,
            "target_color": self._target_color,
            "hindered": hindered,
            "control_dt": self.step_dt,
        }
        return obs, info

    def step(self, action: np.ndarray) -> StepResult:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,):
            raise ValueError("Action must be a 7-D joint velocity vector.")
        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains non-finite values.")

        self._last_action = np.clip(action, -0.5, 0.5)
        if self.model.nu >= 7:
            self.data.ctrl[:7] = self._last_action
            self.data.ctrl[:] = np.clip(self.data.ctrl, -1.0, 1.0)
        else:  # pragma: no cover - fallback
            self.data.qvel[self._joint_dof_indices] = self._last_action

        substeps = max(1, int(round(self.step_dt / self.model.opt.timestep)))
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)

        self._enforce_limits()
        mujoco.mj_forward(self.model, self.data)
        self._update_safety_stats()

        obs = self._get_obs()
        reward = self._compute_reward()
        success = self._check_success()

        self._elapsed_steps += 1
        terminated = success
        truncated = self._elapsed_steps >= self.max_steps
        info = {
            "success": float(success),
            "target_color": self._target_color,
            "hindered": float(self._hindered),
            "distance": float(self._target_distance()),
            "joint_pos_max_abs": self._max_abs_joint,
            "joint_vel_max_abs": self._max_abs_velocity,
        }
        return StepResult(obs, reward, terminated, truncated, info)

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get current observation dictionary.
        
        Returns:
            Dictionary containing:
                - "rgb_static": (H, W, 3) float32 RGB image normalized to [0, 1]
                - "proprio": (7,) float32 joint positions in radians
        """
        image = self.render(mode="rgb_array")
        if not np.all(np.isfinite(image)):
            raise RuntimeError("Rendered image contains non-finite values.")
        proprio = self.data.qpos[self._joint_qpos_indices].astype(np.float32).copy()
        return {"rgb_static": image, "proprio": proprio}

    # ------------------------------------------------------------------
    # Reward and termination utilities
    # ------------------------------------------------------------------
    def _target_distance(self) -> float:
        gripper_pos = self.data.site_xpos[self._gripper_site_id]
        target_pos = self.data.site_xpos[self._object_site_ids[self._target_color]]
        return float(np.linalg.norm(gripper_pos - target_pos))

    def _compute_reward(self) -> float:
        return -self._target_distance()

    def _check_success(self) -> bool:
        """Check if the target object is successfully placed in the bin.
        
        Success requires:
        1. Object height above success_height threshold (lifted off table)
        2. Object within bin_radius of the bin's horizontal position
        """
        site_id = self._object_site_ids[self._target_color]
        obj_pos = self.data.site_xpos[site_id]
        
        # Check height criterion
        if obj_pos[2] < self.success_height:
            return False
        
        # Check horizontal proximity to bin
        horizontal_dist = np.linalg.norm(obj_pos[:2] - self.bin_position[:2])
        return bool(horizontal_dist < self.bin_radius)

    # ------------------------------------------------------------------
    # Randomisation helpers
    # ------------------------------------------------------------------
    def _randomize_robot_pose(self) -> None:
        home = self._home_configuration
        noise = self.rng.uniform(-0.05, 0.05, size=home.shape)
        self.data.qpos[self._joint_qpos_indices] = home + noise
        self.data.qvel[self._joint_dof_indices] = 0.0

    def _randomize_objects(self) -> Tuple[str, ...]:
        count = int(self.rng.integers(3, len(_OBJECT_COLORS) + 1))
        active = tuple(self.rng.choice(_OBJECT_COLORS, size=count, replace=False))
        for color in _OBJECT_COLORS:
            addr = self._object_qpos_addrs[color]
            if color in active:
                xy = self.rng.uniform(-self.workspace_extent, self.workspace_extent)
                pos = np.array([0.5 + xy[0], xy[1], 0.035], dtype=np.float64)
                quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                self._set_free_joint_pose(addr, pos, quat)
            else:
                self._set_free_joint_pose(addr, _HIDDEN_POSE[:3], _HIDDEN_POSE[3:])
        return active

    def _apply_hindered_modifications(self) -> None:
        # Reset lighting
        self.model.light_ambient[self._light_id] = np.array([0.4, 0.4, 0.4, 1.0])
        self.model.light_diffuse[self._light_id] = np.array([0.6, 0.6, 0.6])

        if not self._hindered:
            self._set_free_joint_pose(self._occluder_qpos_addr, _HIDDEN_POSE[:3], _HIDDEN_POSE[3:])
            return

        self.model.light_diffuse[self._light_id] = self.rng.uniform(0.2, 0.9, size=3)
        xy = self.rng.uniform(-self.workspace_extent * 0.5, self.workspace_extent * 0.5)
        pos = np.array([0.5 + xy[0], xy[1], 0.18], dtype=np.float64)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._set_free_joint_pose(self._occluder_qpos_addr, pos, quat)

    # ------------------------------------------------------------------
    # Safety helpers
    # ------------------------------------------------------------------
    def _enforce_limits(self) -> None:
        for idx, (lower, upper) in zip(self._joint_qpos_indices, self._joint_limits):
            if np.isfinite(lower) and self.data.qpos[idx] < lower:
                self.data.qpos[idx] = lower
            if np.isfinite(upper) and self.data.qpos[idx] > upper:
                self.data.qpos[idx] = upper
        for addr in self._free_joint_addresses():
            quat = self.data.qpos[addr + 3 : addr + 7]
            self.data.qpos[addr + 3 : addr + 7] = self._normalize_quaternion(quat)

    def _update_safety_stats(self) -> None:
        joints = self.data.qpos[self._joint_qpos_indices]
        vels = self.data.qvel[self._joint_dof_indices]
        if not np.all(np.isfinite(joints)) or not np.all(np.isfinite(vels)):
            raise RuntimeError("Non-finite values encountered in joint states.")
        self._max_abs_joint = max(self._max_abs_joint, float(np.max(np.abs(joints))))
        self._max_abs_velocity = max(self._max_abs_velocity, float(np.max(np.abs(vels))))

    # ------------------------------------------------------------------
    # Rendering and cleanup
    # ------------------------------------------------------------------
    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode == "rgb_array":
            self.renderer.update_scene(self.data)
            rgb = self.renderer.render()
            return np.clip(rgb, 0.0, 1.0).astype(np.float32)
        if mode == "human":  # pragma: no cover - viewer only
            if self.viewer is None:
                raise RuntimeError("Viewer not initialised; construct environment with gui=True.")
            self.viewer.sync()
            return np.zeros((self.height, self.width, 3), dtype=np.float32)
        raise ValueError("Unsupported render mode; expected 'rgb_array' or 'human'.")

    def close(self) -> None:
        if self.viewer is not None:  # pragma: no cover
            self.viewer.close()
            self.viewer = None
        if hasattr(self.renderer, "free"):
            self.renderer.free()  # type: ignore[attr-defined]
        self.renderer = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _get_body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Body '{name}' not found in MuJoCo model.")
        return body_id

    def _get_site_id(self, name: str) -> int:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"Site '{name}' not found in MuJoCo model.")
        return site_id

    def _get_light_id(self, name: str) -> int:
        light_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, name)
        if light_id < 0:
            raise ValueError(f"Light '{name}' not found in MuJoCo model.")
        return light_id

    def _free_joint_qpos_addr(self, body_id: int) -> int:
        joint_adr = self.model.body_jntadr[body_id]
        if joint_adr < 0:
            raise ValueError("Body does not have an associated joint for pose updates.")
        if self.model.jnt_type[joint_adr] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Expected a free joint for movable body.")
        return self.model.jnt_qposadr[joint_adr]

    def _set_free_joint_pose(self, addr: int, pos: Sequence[float], quat: Sequence[float]) -> None:
        self.data.qpos[addr : addr + 3] = np.asarray(pos, dtype=np.float64)
        self.data.qpos[addr + 3 : addr + 7] = self._normalize_quaternion(np.asarray(quat, dtype=np.float64))

    def _free_joint_addresses(self) -> Iterable[int]:
        for addr in self._object_qpos_addrs.values():
            yield addr
        yield self._occluder_qpos_addr

    @staticmethod
    def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return quat / norm

    @property
    def instruction(self) -> str:
        return self._instruction

    @property
    def target_color(self) -> str:
        return self._target_color

    def __del__(self) -> None:  # pragma: no cover - destructor best-effort
        try:
            self.close()
        except Exception:
            pass


def _smoke_test(asset_root: Path) -> None:  # pragma: no cover - CLI helper
    env = FrankaPickPlaceEnv(asset_root=asset_root, gui=False)
    obs, info = env.reset()
    print("Reset observation keys:", obs.keys())
    print("Instruction:", info["instruction"])
    for _ in range(5):
        action = np.zeros(7)
        result = env.step(action)
        print(
            f"step success={result.info['success']} reward={result.reward:.3f} "
            f"distance={result.info['distance']:.3f}"
        )
        if result.terminated or result.truncated:
            break
    env.close()


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test for FrankaPickPlaceEnv")
    parser.add_argument("--asset-root", type=Path, default=Path("env/mujoco_assets"))
    args = parser.parse_args()
    if mujoco is None:
        raise SystemExit("MuJoCo not installed")
    _smoke_test(args.asset_root)
