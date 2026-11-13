"""Forward kinematics of the RoArm-M3 arm, taken from its URDF joint origins."""

import numpy as np


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def _translate(x, y, z):
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


def _joint(xyz, rpy, q):
    # URDF origin (fixed-axis roll-pitch-yaw), then rotation about the joint's z axis.
    return _translate(*xyz) @ _rz(rpy[2]) @ _ry(rpy[1]) @ _rx(rpy[0]) @ _rz(q)


def fk_link5(angles_deg):
    """4x4 pose of link5 (wrist roll) in the robot base frame, in meters.

    `angles_deg` holds base, shoulder, elbow, wrist pitch and wrist roll in degrees;
    extra entries (the gripper) are ignored.
    """
    q = np.radians(np.asarray(angles_deg[:5], dtype=float))
    half_pi = np.pi / 2
    t = _translate(0, 0, 0.0701)
    t = t @ _joint([0, 0, 0], [0, 0, 0], q[0])
    t = t @ _joint([0, 0, 0.051959], [-half_pi, -half_pi, 0], q[1])
    t = t @ _joint([0.236815, 0.030002, 0], [0, 0, half_pi], q[2])
    t = t @ _joint([0, -0.144586, 0], [0, 0, 0], q[3])
    t = t @ _joint([0.015147, -0.053653, 0], [half_pi, half_pi, 0], q[4])
    return t
