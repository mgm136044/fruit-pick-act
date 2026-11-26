"""SO-101 순기구학 / 수치 IK (numpy 만 사용).

관절 프레임은 로봇 USD 의 PhysicsRevoluteJoint (localPos0/localRot0/localPos1/localRot1/axis) 에서
explore_scene.py 가 뽑아 so101_chain.json 으로 저장한 것을 읽는다.
  T_world_child = T_world_parent · T(lp0, lr0) · R(axis, q) · T(lp1, lr1)^-1
좌표계: 로봇 베이스(root) 기준. 쿼터니언은 wxyz.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def quat_wxyz_to_mat(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0.0
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def mat_to_quat_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def tf(pos, quat_wxyz):
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_mat(quat_wxyz)
    T[:3, 3] = pos
    return T


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def rot_axis(axis, q):
    c, s = np.cos(q), np.sin(q)
    T = np.eye(4)
    if axis == "X":
        T[1:3, 1:3] = [[c, -s], [s, c]]
    elif axis == "Y":
        T[0, 0], T[0, 2], T[2, 0], T[2, 2] = c, s, -s, c
    else:
        T[0:2, 0:2] = [[c, -s], [s, c]]
    return T


def yaw_mat(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class SO101Kin:
    """base → gripper 체인 FK, 그리고 그리퍼 TCP(손가락 사이 점) 위치/방향 IK."""

    def __init__(self, chain_path=os.path.join(HERE, "so101_chain.json")):
        with open(chain_path) as f:
            c = json.load(f)
        self.joints = []
        for name in ARM:
            j = c["joints"][name]
            self.joints.append((tf(j["lp0"], j["lr0"]), j["axis"], inv(tf(j["lp1"], j["lr1"]))))
        # 베이스 body 프레임 → 첫 관절 부모 body 프레임 (보통 단위행렬)
        self.T_root_first = np.array(c.get("T_root_first", np.eye(4).tolist()))
        lim = c["limits_rad"]
        self.lo = np.array([lim[n][0] for n in ARM])
        self.hi = np.array([lim[n][1] for n in ARM])
        # TCP: gripper body 프레임에서의 위치, 접근 방향(손가락이 향하는 방향), 닫는 방향
        self.tcp = np.array(c["tcp"]["pos"])
        self.approach = np.array(c["tcp"]["approach"]) / np.linalg.norm(c["tcp"]["approach"])
        self.close_dir = np.array(c["tcp"]["close_dir"]) / np.linalg.norm(c["tcp"]["close_dir"])

    def fk_gripper(self, q):
        T = self.T_root_first.copy()
        for (A, axis, B), qi in zip(self.joints, q):
            T = T @ A @ rot_axis(axis, qi) @ B
        return T

    def tcp_frame(self, q):
        """(TCP 위치, 접근 방향, 닫는 방향) — 모두 로봇 베이스 기준."""
        T = self.fk_gripper(q)
        R = T[:3, :3]
        return R @ self.tcp + T[:3, 3], R @ self.approach, R @ self.close_dir

    def _residual(self, q, p_goal, a_goal, c_goal, w_a, w_c):
        p, a, c = self.tcp_frame(q)
        r = [p - p_goal]
        if a_goal is not None:
            r.append(w_a * (a - a_goal))
        if c_goal is not None:
            # 닫는 방향은 부호 무관 (±c 둘 다 같은 잡기) → 수평면 성분의 외적으로 오차
            cg = c_goal / np.linalg.norm(c_goal)
            r.append(w_c * np.array([np.cross(c, cg)[2], 0.0, 0.0]))
        return np.concatenate(r)

    def ik(self, p_goal, q0, a_goal=(0, 0, -1.0), c_goal=None, w_a=0.15, w_c=0.08, iters=80, tol=1e-4, reg=1e-3):
        """감쇠 최소제곱 IK. p_goal(m), a_goal = 접근 방향(기본: 아래), c_goal = 닫는 방향(수평, 부호 무관).
        위치를 가장 크게, 방향은 가중치로 (닿지 않으면 기울여서 위치를 맞춤). 돌려주는 값: (q, 위치 오차 m)."""
        q = np.clip(np.array(q0, dtype=np.float64), self.lo, self.hi)
        a_goal = None if a_goal is None else np.asarray(a_goal, dtype=np.float64)
        lam = 1e-2
        r = self._residual(q, p_goal, a_goal, c_goal, w_a, w_c)
        cost = r @ r
        for _ in range(iters):
            J = np.empty((len(r), 5))
            h = 1e-5
            for k in range(5):
                dq = np.zeros(5)
                dq[k] = h
                J[:, k] = (self._residual(q + dq, p_goal, a_goal, c_goal, w_a, w_c) - r) / h
            H = J.T @ J + (lam + reg) * np.eye(5)
            step = -np.linalg.solve(H, J.T @ r)
            qn = np.clip(q + step, self.lo, self.hi)
            rn = self._residual(qn, p_goal, a_goal, c_goal, w_a, w_c)
            cn = rn @ rn
            if cn < cost:
                q, r, cost = qn, rn, cn
                lam = max(lam * 0.3, 1e-6)
                if np.linalg.norm(step) < 1e-7:
                    break
            else:
                lam *= 10
                if lam > 1e6:
                    break
            if np.linalg.norm(r[:3]) < tol and cost < tol:
                break
        return q, float(np.linalg.norm(self.tcp_frame(q)[0] - p_goal))

    def ik_pos_first(self, p_goal, q0, a_goal=(0, 0, -1.0), c_goal=None, pos_tol=0.0015):
        """위치 우선 IK: 방향 가중치를 단계적으로 낮춰 위치 오차가 pos_tol 안에 들 때까지.
        (손목 관절 한계 때문에 수직 접근이 안 되는 곳에서는 기울여서 위치를 맞춘다)"""
        best = None
        q_init = np.array(q0, dtype=np.float64)
        for w_a in (0.15, 0.05, 0.015, 0.005, 0.0015, 0.0):
            q, e = self.ik(p_goal, q_init, a_goal=a_goal if w_a > 0 else None, c_goal=c_goal, w_a=w_a,
                           w_c=0.08)  # 닫는 방향(yaw)은 wrist_roll 로 위치와 독립적으로 맞출 수 있어 가중치 유지
            if best is None or e < best[1]:
                best = (q, e)
            if e < pos_tol:
                return q, e
            q_init = q
        return best

    def tilt_deg(self, q):
        a = self.tcp_frame(q)[1]
        return float(np.degrees(np.arccos(np.clip(-a[2], -1, 1))))
