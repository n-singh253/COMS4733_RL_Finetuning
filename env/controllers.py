"""Kinematics and control utilities for the Franka MuJoCo environment."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency for import-time checks
    import mujoco
except Exception:  # pragma: no cover
    mujoco = None  # type: ignore


@dataclass(frozen=True)
class JointVelocityControllerConfig:
    kp: float = 150.0
    kd: float = 20.0
    max_velocity: float = 1.0


class JointVelocityController:
    """Simple PD controller for joint-velocity commands."""

    def __init__(self, config: Optional[JointVelocityControllerConfig] = None) -> None:
        self.config = config or JointVelocityControllerConfig()

    def __call__(self, target: np.ndarray, current: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        error = target - current
        damping = -self.config.kd * velocity
        command = self.config.kp * error + damping
        return np.clip(command, -self.config.max_velocity, self.config.max_velocity)


@dataclass(frozen=True)
class ActuatedJointInfo:
    """Indices describing the actuated arm joints."""

    joint_ids: np.ndarray
    qpos_indices: np.ndarray
    dof_indices: np.ndarray
    limits: np.ndarray


def infer_actuated_joints(model: "mujoco.MjModel", expected: int = 7) -> ActuatedJointInfo:
    """Infer indices for the actuated joints using the actuator mapping."""

    if mujoco is None:  # pragma: no cover - guard for documentation builds
        raise ImportError("MuJoCo is required to infer actuated joints")

    joint_ids: list[int] = []
    qpos_indices: list[int] = []
    dof_indices: list[int] = []

    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id][0])
        if joint_id < 0:
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if joint_id in joint_ids:
            continue
        joint_ids.append(joint_id)
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        dof_indices.append(int(model.jnt_dofadr[joint_id]))
        if len(joint_ids) == expected:
            break

    if len(joint_ids) < expected:
        raise RuntimeError(
            f"Expected at least {expected} actuated joints, found {len(joint_ids)}."
        )

    limits = np.zeros((len(joint_ids), 2), dtype=np.float64)
    for idx, joint_id in enumerate(joint_ids):
        limited = int(model.jnt_limited[joint_id]) if hasattr(model, "jnt_limited") else 0
        if limited:
            limits[idx] = model.jnt_range[joint_id]
        else:
            limits[idx] = np.array([-np.inf, np.inf])

    return ActuatedJointInfo(
        joint_ids=np.asarray(joint_ids, dtype=np.int32),
        qpos_indices=np.asarray(qpos_indices, dtype=np.int32),
        dof_indices=np.asarray(dof_indices, dtype=np.int32),
        limits=limits,
    )


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    result = quat.copy()
    result[1:] *= -1
    return result


def _quat_normalize(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quat / norm


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = _quat_normalize(quat)
    angle = 2.0 * math.acos(np.clip(quat[0], -1.0, 1.0))
    s = math.sqrt(max(1.0 - quat[0] * quat[0], 0.0))
    if s < 1e-8:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = quat[1:] / s
    return axis * angle


class KinematicsHelper:
    """Provides FK/IK and Jacobian utilities for the Franka arm."""

    def __init__(
        self,
        model: "mujoco.MjModel",
        *,
        site_name: str = "panda_gripper_site",
        joint_info: Optional[ActuatedJointInfo] = None,
    ) -> None:
        if mujoco is None:  # pragma: no cover
            raise ImportError("MuJoCo is required for KinematicsHelper")

        self.model = model
        self.data = mujoco.MjData(model)
        self.data.qpos[:] = model.qpos0
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in model")

        self.joint_info = joint_info or infer_actuated_joints(model)

    def forward_kinematics(self, q: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=np.float64)
        self._apply_configuration(q)
        mujoco.mj_forward(self.model, self.data)
        pos = self.data.site_xpos[self.site_id].copy()
        quat = self.data.site_xquat[self.site_id].copy()
        return pos, quat

    def analytic_jacobian(self, q: Sequence[float]) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        self._apply_configuration(q)
        mujoco.mj_forward(self.model, self.data)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        cols = self.joint_info.dof_indices
        return np.vstack([jacp[:, cols], jacr[:, cols]])

    def finite_difference_jacobian(self, q: Sequence[float], eps: float = 1e-6) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        base_pos, base_quat = self.forward_kinematics(q)
        num_joints = q.shape[0]
        jac = np.zeros((6, num_joints), dtype=np.float64)
        for j in range(num_joints):
            dq = np.zeros_like(q)
            dq[j] = eps
            pos, quat = self.forward_kinematics(q + dq)
            dpos = (pos - base_pos) / eps
            dquat = _quat_to_axis_angle(_quat_multiply(_quat_conjugate(base_quat), quat)) / eps
            jac[:3, j] = dpos
            jac[3:, j] = dquat
        return jac

    def manipulability(self, q: Sequence[float]) -> float:
        jac = self.analytic_jacobian(q)[:3, :]
        gram = jac @ jac.T
        value = float(np.linalg.det(gram))
        return max(value, 0.0)

    def inverse_kinematics(
        self,
        target_pos: Sequence[float],
        target_quat: Sequence[float],
        *,
        initial_q: Optional[Sequence[float]] = None,
        max_iters: int = 200,
        tol_pos: float = 1e-4,
        tol_ori: float = math.radians(2.0),
        damping: float = 1e-4,
        step_size: float = 1.0,
    ) -> np.ndarray:
        q = np.asarray(initial_q if initial_q is not None else self.model.qpos0[self.joint_info.qpos_indices], dtype=np.float64)
        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_quat = _quat_normalize(np.asarray(target_quat, dtype=np.float64))

        for _ in range(max_iters):
            pos, quat = self.forward_kinematics(q)
            pos_error = target_pos - pos
            orient_error = _quat_to_axis_angle(_quat_multiply(_quat_conjugate(quat), target_quat))
            if np.linalg.norm(pos_error) < tol_pos and np.linalg.norm(orient_error) < tol_ori:
                return q
            jac = self.analytic_jacobian(q)
            error = np.concatenate([pos_error, orient_error])
            jtj = jac.T @ jac + damping * np.eye(jac.shape[1])
            dq = step_size * np.linalg.solve(jtj, jac.T @ error)
            q = q + dq
            q = self._clamp_to_limits(q)

        raise RuntimeError("IK failed to converge within the allotted iterations")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_configuration(self, q: np.ndarray) -> None:
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.data.qpos[self.joint_info.qpos_indices] = q

    def _clamp_to_limits(self, q: np.ndarray) -> np.ndarray:
        q = q.copy()
        for i, (lower, upper) in enumerate(self.joint_info.limits):
            if np.isfinite(lower):
                q[i] = max(q[i], lower)
            if np.isfinite(upper):
                q[i] = min(q[i], upper)
        return q


__all__ = [
    "ActuatedJointInfo",
    "JointVelocityController",
    "JointVelocityControllerConfig",
    "KinematicsHelper",
    "infer_actuated_joints",
]
