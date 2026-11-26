"""스크립트 정책(상태머신)으로 과일 집기 시연을 사람 없이 생성 → record.py 와 같은 형식의 LeRobot 데이터셋.

  source ~/robot_course/activate.sh
  python gen_scripted.py --dataset ~/datasets/fruit_scripted_v1 --fruits all --per-fruit 25 --viz none

※ 사람 시연이 아님 (자동 생성). 과제 규칙상 허용되는지 확인 전 → 로컬에만 둘 것.

형식 (record.py 268~280, 487~493, 542 행과 같게)
  - LeRobotDataset.create(repo_id=local/<폴더명>, fps=30, robot_type=so101_follower, use_videos=True)
  - features: observation.state / action (float32, 6, names=FEATURE_NAMES), observation.images.front/wrist (video 480x640x3)
  - 프레임 t: state = 실제 관절(rad→LeRobot 단위), 이미지 = 그 시점 관측, action = 그 스텝에 보낸 관절 목표(LeRobot 단위)
  - task = task_text(fruit), save_episode(parallel_encoding=False)
장면 변형 (record.py 의 count / add / robot 과 같은 범위, 배치 코드는 record.py place_robot/place_fruits 를 옮김)
상태머신: 휴식 자세 → 과일 위 → 내려가기 → 쥐기 → 들기 → 접시 위 → 내려놓기 → 놓기 → 위로 → 휴식 자세 (성공만 저장)
"""
import argparse
import json
import math
import os
import sys
import time

KIT = os.environ.get("ROBOT_KIT", os.path.expanduser("~/student_kit"))
sys.path.insert(0, os.path.join(KIT, "isaac"))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

FRUIT_ALIASES = {"strawberry": 0, "orange": 1, "lemon": 2, "banana": 3}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--dataset", required=True, help="데이터셋 폴더")
ap.add_argument("--resume", action="store_true", help="기존 데이터셋에 이어서")
ap.add_argument("--fruits", default="all", help="all 또는 strawberry,orange,lemon,banana 중 쉼표 목록")
ap.add_argument("--per-fruit", type=int, default=5, help="과일마다 저장할 성공 에피소드 수")
ap.add_argument("--max-attempts", type=int, default=0, help="과일마다 최대 시도 (0 = per-fruit x 3)")
ap.add_argument("--vary-prob", type=float, default=0.5, help="장면 변형 에피소드 비율 (나머지는 기본 장면)")
ap.add_argument("--start-perturb-prob", type=float, default=0.25, help="시작 자세를 휴식 자세에서 조금 바꾸는 비율")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-sec", type=float, default=30.0, help="에피소드 최대 길이 (평가 기준 30초)")
ap.add_argument("--log", default=None, help="시도별 JSONL 로그 (기본 logs/gen_<데이터셋>.jsonl)")
ap.add_argument("--params", default=None, help="과일별 파라미터 덮어쓰기 JSON")
ap.add_argument("--no-save", action="store_true", help="데이터셋에 쓰지 않음 (튜닝용)")
ap.add_argument("--snap", type=int, default=0, help="처음 N 시도는 주요 순간 정면/손목 PNG 저장")
add_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True

FPS = 30
# ---- record.py 의 장면 변형 상수 (같은 값)
ROBOT_MAX_CM, ROBOT_MAX_DEG = 10.0, 30.0
AREA_X, AREA_Y = (1.95, 2.30), (-0.48, -0.26)
PLATE_CLEAR, ROBOT_CLEAR, GAP, GAP_BANANA = 0.14, 0.17, 0.075, 0.11

# ---- 과일별 파라미터 (explore_scene.py 로 잰 키트 메시 기준)
#   grasp_local: 과일 root(바닥 중심) 기준 잡는 점, width: 닫는 방향 폭(m), long_axis: 긴 축(로컬, 있으면 가로질러 쥠)
#   open/close: 그리퍼 명령 (LeRobot 단위 0 닫힘 ~ 100 열림), close = 접촉 각도보다 약간 더 닫음
FRUIT_P = {
    "Strawberry": {"grasp_local": [0, 0, 0.0228], "height": 0.0457, "width": 0.045, "long_axis": None, "open": 42.0, "close": 25.0},
    "Orange": {"grasp_local": [0, 0, 0.0268], "height": 0.0535, "width": 0.055, "long_axis": None, "open": 48.0, "close": 33.0},
    "Lemon": {"grasp_local": [0, 0, 0.0225], "height": 0.0451, "width": 0.051, "long_axis": None, "open": 46.0, "close": 29.0},
    "Banana": {"grasp_local": [-0.0285, 0, 0.0164], "height": 0.0312, "width": 0.034, "long_axis": [0, 1, 0], "open": 36.0, "close": 16.0},
}
COMMON = {
    "fixed_inner_x": -0.0105,  # gripper 프레임: 고정 손가락 안쪽 면 x (TCP 깊이에서)
    "fixed_clear": 0.005,      # 고정 손가락과 과일 사이 여유
    "tcp_depth": -0.085,       # gripper 프레임 z: 손가락 끝(-0.105)에서 2cm 안쪽
    "tip_below_tcp": 0.020,
    "tip_clear": 0.006,        # 쥘 때 손가락 끝을 조리대 위 이만큼
    "hover": 0.07,             # 과일 위 접근 높이 (잡는 점 기준)
    "lift": 0.10,
    "plate_rim": 0.037,        # 접시 높이 (root 는 바닥)
    "plate_inner": 0.008,      # 접시 안쪽 바닥 높이 (추정)
    "drop": 0.012,             # 내려놓을 때 과일 바닥과 접시 사이 여유
    "rim_clear": 0.03,
    "retreat": 0.06,
    "jit_hover": 0.008, "jit_grasp": 0.003, "jit_plate": 0.012,
    "ki": 0.08, "bias_max": 0.25,
    # 구간 길이 (프레임, 30fps) — 사람 시연(약 10초)과 비슷하게
    "T": {"to_hover": 60, "hover_wait": 5, "descend": 30, "close": 15, "settle": 6, "lift": 25, "to_plate": 50, "lower": 25,
          "open": 10, "retreat": 15, "to_rest": 50},
}
if args.params:
    with open(args.params) as f:
        ov = json.load(f)
    for k, v in ov.get("fruits", {}).items():
        FRUIT_P[k].update(v)
    for k, v in ov.get("common", {}).items():
        if k == "T":
            COMMON["T"].update(v)
        else:
            COMMON[k] = v


def min_jerk(n):
    import numpy as np
    t = np.linspace(0, 1, n + 1)[1:]
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def main():
    import numpy as np
    import torch
    from PIL import Image

    from course_env.mdp import active_fruit_on_plate, active_index, park_fruits
    from course_env.paths import FRUIT_KO, FRUITS, PLATE, copies
    from course_env.pick_fruit import SO101_CFG, PickFruitEnvCfg, task_text
    from course_env.so101 import FEATURE_NAMES, JOINTS, motor_to_rad, rad_to_motor, rest_pose_rad
    from kin import SO101Kin, quat_wxyz_to_mat

    kin = SO101Kin()
    q_rest6 = np.array([rest_pose_rad()[j] for j in JOINTS])
    rest_motor = rad_to_motor(q_rest6)
    rng = np.random.default_rng(args.seed)
    fruit_ids = list(range(4)) if args.fruits == "all" else [FRUIT_ALIASES[w.strip()] for w in args.fruits.split(",")]
    max_att = args.max_attempts or args.per_fruit * 3
    root = os.path.abspath(os.path.expanduser(args.dataset))
    log_path = args.log or os.path.join(HERE, "logs", f"gen_{os.path.basename(root.rstrip(os.sep))}.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    snap_dir = os.path.join(HERE, "logs", f"snap_{os.path.basename(root.rstrip(os.sep))}")

    # ---- 데이터셋 (record.py 268~280 행과 같게)
    ds = None
    if not args.no_save:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        repo_id = f"local/{os.path.basename(root.rstrip(os.sep))}"
        vec = {"dtype": "float32", "shape": (6,), "names": FEATURE_NAMES}
        img = {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
        features = {"observation.state": vec, "action": vec, "observation.images.front": img,
                    "observation.images.wrist": img}
        if args.resume:
            ds = LeRobotDataset.resume(repo_id, root=root, image_writer_threads=4)
        else:
            if os.path.exists(root):
                sys.exit(f"{root} 가 이미 있습니다 (--resume 또는 다른 이름)")
            ds = LeRobotDataset.create(repo_id, fps=FPS, features=features, root=root, robot_type="so101_follower",
                                       use_videos=True, image_writer_threads=4)
        print(f"데이터셋: {root} (지금까지 {ds.num_episodes} 에피소드)", flush=True)

    cfg = PickFruitEnvCfg()
    cfg.terminations.time_out = None
    cfg.terminations.success = None
    with launch_simulation(cfg, args):
        from isaaclab.envs import ManagerBasedRLEnv
        env = ManagerBasedRLEnv(cfg)
        dev = env.device
        ids = torch.tensor([0], device=dev)
        origin = env.scene.env_origins[0]
        robot = env.scene["robot"]
        base_pos = torch.tensor(SO101_CFG.init_state.pos, device=dev)
        base_yaw = math.pi
        zero6 = torch.zeros(1, 6, device=dev)
        ARM_K = float(SO101_CFG.actuators["arm"].stiffness)
        print("gravity comp shape", tuple(robot.data.gravity_compensation_forces.torch.shape), "arm K", ARM_K, flush=True)

        # ---------- record.py 의 배치 코드 (put / place_robot / place_fruits) ----------
        def put(name, x, y, z, yaw=0.0):
            pose = torch.zeros(1, 7, device=dev)
            pose[0, :3] = origin + torch.tensor([x, y, z], device=dev)
            pose[0, 3:] = torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)], device=dev)
            env.scene[name].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            env.scene[name].write_root_velocity_to_sim_index(root_velocity=zero6, env_ids=ids)

        def place_robot(lay):
            fwd, right, yaw = lay["robot"]
            x, y, z = (base_pos + torch.tensor([right / 100, fwd / 100, 0.0], device=dev)).tolist()
            pose = torch.zeros(1, 7, device=dev)
            th = base_yaw + math.radians(yaw)
            pose[0, :3] = origin + torch.tensor([x, y, z], device=dev)
            pose[0, 3:] = torch.tensor([0.0, 0.0, math.sin(th / 2), math.cos(th / 2)], device=dev)
            robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            robot.write_root_velocity_to_sim_index(root_velocity=zero6, env_ids=ids)

        def place_fruits(lay, fi):
            active_index(env)[0] = fi
            park_fruits(env, ids, list(FRUITS), torch.tensor([-1], device=dev))
            if lay["count"] == 1 and not lay["extra"]:  # 기본 장면: 기준 자리 ±3cm, yaw 0
                base = getattr(env.cfg.scene, FRUITS[fi]).init_state.pos
                dx, dy = rng.uniform(-0.03, 0.03, size=2)
                put(FRUITS[fi], base[0] + dx, base[1] + dy, base[2])
                return [FRUITS[fi]]
            names = copies(FRUITS[fi])[:lay["count"]]
            for oi, n in lay["extra"].items():
                if oi != fi:
                    names += copies(FRUITS[oi])[:n]
            plate = (env.scene[PLATE].data.root_pos_w.torch[0] - origin).tolist()
            z = getattr(env.cfg.scene, FRUITS[0]).init_state.pos[2]
            rb = (float(base_pos[0]) + lay["robot"][1] / 100, float(base_pos[1]) + lay["robot"][0] / 100)
            best = []
            for _ in range(200):
                spots = []
                for name in names:
                    banana = name.startswith("Banana")
                    for _ in range(100):
                        x, y = rng.uniform(*AREA_X), rng.uniform(*AREA_Y)
                        if (math.hypot(x - plate[0], y - plate[1]) >= PLATE_CLEAR
                                and math.hypot(x - rb[0], y - rb[1]) >= ROBOT_CLEAR and all(
                                math.hypot(x - sx, y - sy) >= (GAP_BANANA if banana or sb else GAP)
                                for _, sx, sy, sb in spots)):
                            spots.append((name, x, y, banana))
                            break
                if len(spots) > len(best):
                    best = spots
                if len(best) == len(names):
                    break
            for name, x, y, _ in best:
                put(name, x, y, z, rng.uniform(-math.pi, math.pi))
            return [b[0] for b in best if b[0].split("_")[0] == FRUITS[fi]]
        # ---------------------------------------------------------------------------

        def sample_layout(fi):
            if rng.random() >= args.vary_prob:
                return {"kind": "default", "count": 1, "extra": {}, "robot": (0.0, 0.0, 0.0)}
            use = {k: rng.random() < 0.6 for k in ("count", "extra", "robot")}
            if not any(use.values()):
                use[rng.choice(["count", "extra", "robot"])] = True
            lay = {"kind": "vary", "count": 1, "extra": {}, "robot": (0.0, 0.0, 0.0)}
            if use["count"]:
                lay["count"] = int(rng.choice([2, 3], p=[0.6, 0.4]))
            if use["extra"]:
                others = [i for i in range(4) if i != fi]
                for oi in rng.choice(others, size=int(rng.integers(1, 3)), replace=False):
                    lay["extra"][int(oi)] = int(rng.integers(1, 3))
            if use["robot"]:
                lay["robot"] = (float(rng.uniform(-ROBOT_MAX_CM, ROBOT_MAX_CM)), float(rng.uniform(-ROBOT_MAX_CM, ROBOT_MAX_CM)),
                                float(rng.uniform(-ROBOT_MAX_DEG, ROBOT_MAX_DEG)))
            return lay

        def pose_T(p7):
            T = np.eye(4)
            x, y, z, w = p7[3:7]
            T[:3, :3] = quat_wxyz_to_mat([w, x, y, z])
            T[:3, 3] = p7[:3]
            return T

        def base_T():
            return pose_T(robot.data.root_link_pose_w.torch[0].cpu().numpy().astype(np.float64))

        def obj_T(name):
            return pose_T(env.scene[name].data.root_link_pose_w.torch[0].cpu().numpy().astype(np.float64))

        def q_now():
            return robot.data.joint_pos.torch[0].cpu().numpy().astype(np.float64)

        def set_tcp(fruit):
            P = FRUIT_P[fruit]
            # 과일 중심을 고정 손가락 안쪽 면에서 (폭/2 + 여유) 떨어지게: 내려갈 때 고정 손가락이 과일에 걸리지 않게
            kin.tcp = np.array([COMMON["fixed_inner_x"] + COMMON["fixed_clear"] + P["width"] / 2, 0.0, COMMON["tcp_depth"]])

        def fruit_grasp_world(fruit, name):
            T = obj_T(name)
            P = FRUIT_P[fruit]
            c = T[:3, :3] @ np.array(P["grasp_local"]) + T[:3, 3]
            close_w = None
            if P["long_axis"] is not None:
                la = T[:3, :3] @ np.array(P["long_axis"], dtype=np.float64)
                la[2] = 0
                la /= max(np.linalg.norm(la), 1e-9)
                close_w = np.array([-la[1], la[0], 0.0])
            return c, close_w, T

        def to_base(Tb, p):
            return Tb[:3, :3].T @ (np.asarray(p) - Tb[:3, 3])

        def dir_to_base(Tb, d):
            return None if d is None else Tb[:3, :3].T @ np.asarray(d)

        def ik_ok(p_b, seed, c_b=None, tol=0.003, tilt_max=45.0):
            q, e = kin.ik_pos_first(p_b, seed, c_goal=c_b)
            return q, e, (e < tol and kin.tilt_deg(q) < tilt_max)

        def seed_for(p_b):
            pan = math.atan2(-p_b[0], -p_b[1])  # 정면 = 베이스 -y
            return np.array([np.clip(pan, kin.lo[0], kin.hi[0]), 0.3, -0.3, 1.4, 0.0])

        def cart_segment(p0, p1, q_seed, c_b, n):
            qs, q = [], q_seed
            for s in min_jerk(n):
                q, _ = kin.ik_pos_first(p0 + (p1 - p0) * s, q, c_goal=c_b)
                qs.append(q)
            return qs

        def joint_segment(q0, q1, n):
            return [q0 + (q1 - q0) * s for s in min_jerk(n)]

        def ramp(g0, g1, n):
            return [g0 + (g1 - g0) * s for s in min_jerk(n)]

        def log(rec):
            with open(log_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        totals = {}
        t_all = time.time()
        for fi in fruit_ids:
            fruit = FRUITS[fi]
            P = FRUIT_P[fruit]
            set_tcp(fruit)
            saved, attempts, relayouts = 0, 0, 0
            stats = {}
            while saved < args.per_fruit and attempts < max_att:
                t0 = time.time()
                # ---- 장면 만들기 (도달 가능한 배치가 나올 때까지)
                for _try in range(30):
                    lay = sample_layout(fi)
                    env.reset()
                    place_robot(lay)
                    targets = place_fruits(lay, fi)
                    hold = torch.tensor(q_rest6, dtype=torch.float32, device=dev).unsqueeze(0)
                    for _ in range(3):
                        obs, *_ = env.step(hold)
                    Tb = base_T()
                    plate_w = obj_T(PLATE)[:3, 3]
                    cand = []
                    for name in targets:
                        c_w, close_w, _ = fruit_grasp_world(fruit, name)
                        if c_w[2] < 0.5:
                            continue
                        cb = to_base(Tb, c_w)
                        cand.append((np.hypot(cb[0], cb[1]), name, c_w, close_w))
                    cand.sort(key=lambda x: x[0])
                    ok = False
                    for _, name, c_w, close_w in cand:
                        table_z = c_w[2] - P["grasp_local"][2]
                        g_w = c_w.copy()
                        g_w[2] = max(c_w[2], table_z + COMMON["tip_below_tcp"] + COMMON["tip_clear"])
                        g_b = to_base(Tb, g_w)
                        c_b = dir_to_base(Tb, close_w)
                        h_b = to_base(Tb, g_w + [0, 0, COMMON["hover"]])
                        pl_b = to_base(Tb, [plate_w[0], plate_w[1], plate_w[2] + COMMON["plate_rim"] + 0.08])
                        q_g, e_g, ok_g = ik_ok(g_b, seed_for(g_b), c_b)
                        q_h, e_h, ok_h = ik_ok(h_b, q_g, c_b, tilt_max=60)
                        q_p, e_p, ok_p = ik_ok(pl_b, seed_for(pl_b), None, tilt_max=75)
                        if ok_g and ok_h and ok_p:
                            ok = True
                            break
                    if ok:
                        break
                    relayouts += 1
                if not ok:
                    print("  도달 가능한 배치를 못 찾음 (30회)", flush=True)
                    attempts += 1
                    continue
                attempts += 1

                # ---- 시작 자세 (준비 단계: 기록 안 함). 가끔 휴식 자세에서 조금 바꿈
                q_start6 = q_rest6.copy()
                if rng.random() < args.start_perturb_prob:
                    q_start6[:5] += np.radians(rng.uniform([-12, -6, -10, -15, -15], [12, 10, 6, 10, 15]))
                    q_start6[5] = motor_to_rad(np.r_[np.zeros(5), rng.uniform(0, 12)])[5]
                    q_start6 = np.clip(q_start6, np.r_[kin.lo, -0.1745], np.r_[kin.hi, 1.7453])
                for k in range(20):
                    a = q_rest6 + (q_start6 - q_rest6) * min(1.0, (k + 1) / 12)
                    obs, *_ = env.step(torch.tensor(a, dtype=torch.float32, device=dev).unsqueeze(0))
                fruit_T0 = obj_T(name)

                # ---- 계획 A: 시작 → 과일 위 → 내려가기 → 쥐기 → 들기
                sp = float(rng.uniform(0.9, 1.1))
                T = {k: max(4, int(round(v * sp * rng.uniform(0.95, 1.05)))) for k, v in COMMON["T"].items()}
                c_w, close_w, _ = fruit_grasp_world(fruit, name)
                table_z = c_w[2] - P["grasp_local"][2]
                g_w = c_w.copy()
                g_w[2] = max(c_w[2], table_z + COMMON["tip_below_tcp"] + COMMON["tip_clear"])
                g_w[:2] += rng.uniform(-COMMON["jit_grasp"], COMMON["jit_grasp"], 2)
                h_w = g_w + np.r_[rng.uniform(-COMMON["jit_hover"], COMMON["jit_hover"], 2), COMMON["hover"]]
                l_w = g_w + [0, 0, COMMON["lift"]]
                c_b = dir_to_base(Tb, close_w)
                g_b, h_b, l_b = to_base(Tb, g_w), to_base(Tb, h_w), to_base(Tb, l_w)
                q_g, _, _ = ik_ok(g_b, seed_for(g_b), c_b)
                q_h, _, _ = ik_ok(h_b, q_g, c_b)
                q0 = q_now()
                plan = []  # (q5, gripper_motor, phase)
                g_start = float(rad_to_motor(q0)[5])
                n1 = T["to_hover"]
                grip1 = [g_start] * (n1 // 4) + ramp(g_start, P["open"], n1 - n1 // 4 - n1 // 6) + [P["open"]] * (n1 // 6)
                for q, gg in zip(joint_segment(q0[:5], q_h, n1), grip1):
                    plan.append((q, gg, "to_hover"))
                for _ in range(T["hover_wait"]):
                    plan.append((q_h, P["open"], "hover_wait"))
                for q in cart_segment(h_b, g_b, q_h, c_b, T["descend"]):
                    plan.append((q, P["open"], "descend"))
                for gg in ramp(P["open"], P["close"], T["close"]):
                    plan.append((plan[-1][0], gg, "close"))
                for _ in range(T["settle"]):
                    plan.append((plan[-1][0], P["close"], "settle"))
                for q in cart_segment(g_b, l_b, plan[-1][0], c_b, T["lift"]):
                    plan.append((q, P["close"], "lift"))

                # ---- 실행 + 기록
                n = 0
                bias = np.zeros(5)
                fail, info = None, {}
                key = {}
                max_frames = int(args.max_sec * FPS)

                def run(plan_items):
                    nonlocal n, bias, obs
                    for q_des, gmot, ph in plan_items:
                        if n >= max_frames:
                            return "시간 초과"
                        qa = q_now()
                        if ph in ("hover_wait", "descend", "close", "settle", "lift", "lower", "open", "retreat"):
                            bias[:4] += COMMON["ki"] * (q_des[:4] - qa[:4])
                            bias = np.clip(bias, -COMMON["bias_max"], COMMON["bias_max"])
                        ff = np.zeros(5)
                        if COMMON.get("gravity_ff", True):  # 중력 처짐 보정: 목표 += 중력 토크 / 관절 강성
                            tau_g = robot.data.gravity_compensation_forces.torch[0].cpu().numpy().astype(np.float64)
                            ff = tau_g[:5] / ARM_K
                        q_cmd = np.clip(q_des + ff + bias, kin.lo, kin.hi)
                        a_mot = rad_to_motor(np.r_[q_cmd, 0.0])
                        a_mot[5] = gmot
                        pol = obs["policy"]
                        if ds is not None:
                            ds.add_frame({
                                "observation.state": rad_to_motor(pol["joint_pos"][0].cpu().numpy()).astype(np.float32),
                                "action": a_mot.astype(np.float32),
                                "observation.images.front": pol["front"][0].cpu().numpy(),
                                "observation.images.wrist": pol["wrist"][0].cpu().numpy(),
                                "task": task_text(fruit),
                            })
                        if attempts <= args.snap and ph not in key:
                            os.makedirs(snap_dir, exist_ok=True)
                            for cam in ("front", "wrist"):
                                Image.fromarray(pol[cam][0].cpu().numpy()).save(
                                    os.path.join(snap_dir, f"{fruit}_{attempts:02d}_{n:03d}_{ph}_{cam}.png"))
                        key.setdefault(ph, n)
                        obs, *_ = env.step(torch.tensor(motor_to_rad(a_mot), dtype=torch.float32, device=dev).unsqueeze(0))
                        n += 1
                    return None

                # 구간마다 끝에서 TCP 오차(실제 관절 FK - 계획 관절 FK)와 관절 오차를 기록
                info["tcp_err_mm"], info["q_err_deg"] = {}, {}
                for ph_name in ("to_hover", "hover_wait", "descend", "close", "settle", "lift"):
                    seg = [p for p in plan if p[2] == ph_name]
                    if not seg:
                        continue
                    fail = run(seg)
                    qa = q_now()[:5]
                    info["tcp_err_mm"][ph_name] = [round(float(v) * 1000, 1) for v in
                                                   (kin.tcp_frame(qa)[0] - kin.tcp_frame(seg[-1][0])[0])]
                    info["q_err_deg"][ph_name] = [round(float(v), 1) for v in np.degrees(qa - seg[-1][0])]
                    if ph_name == "descend":
                        c_now, _, _ = fruit_grasp_world(fruit, name)
                        info["fruit_pushed_mm"] = round(float(np.linalg.norm((c_now - c_w)[:2])) * 1000, 1)
                        info["bias_deg"] = [round(float(v), 1) for v in np.degrees(bias)]
                    if fail:
                        break
                # 쥔 상태 확인
                fc_w, _, _ = fruit_grasp_world(fruit, name)
                tcp_w = Tb[:3, :3] @ kin.tcp_frame(q_now()[:5])[0] + Tb[:3, 3]
                info["lift_dz"] = float(fc_w[2] - c_w[2])
                info["fruit_tcp_dist"] = float(np.linalg.norm(fc_w - tcp_w))
                if fail is None and (info["lift_dz"] < 0.03 or info["fruit_tcp_dist"] > 0.05):
                    fail = "놓침"

                if fail is None:
                    # ---- 계획 B: 접시 위 → 내려놓기 → 놓기 → 위로 → 휴식 자세 (실제 쥔 위치를 보고)
                    q_l = q_now()[:5]
                    d_off = fc_w - tcp_w  # TCP → 과일 잡는 점
                    plate_w = obj_T(PLATE)[:3, 3]
                    pxy = plate_w[:2] + rng.uniform(-COMMON["jit_plate"], COMMON["jit_plate"], 2)
                    grasp_above_bottom = P["grasp_local"][2]
                    cz_place = plate_w[2] + COMMON["plate_inner"] + COMMON["drop"] + grasp_above_bottom
                    cz_pre = plate_w[2] + COMMON["plate_rim"] + COMMON["rim_clear"] + grasp_above_bottom
                    pp_w = np.r_[pxy, cz_pre] - d_off
                    pl_w = np.r_[pxy, cz_place] - d_off
                    pp_b, pl_b2 = to_base(Tb, pp_w), to_base(Tb, pl_w)
                    q_pp, e_pp, _ = ik_ok(pp_b, seed_for(pp_b), None, tilt_max=80)
                    plan2 = []
                    for q in joint_segment(plan[-1][0], q_pp, T["to_plate"]):
                        plan2.append((q, P["close"], "to_plate"))
                    for q in cart_segment(pp_b, pl_b2, q_pp, None, T["lower"]):
                        plan2.append((q, P["close"], "lower"))
                    g_rel = min(P["close"] + 14.0, P["open"])
                    for gg in ramp(P["close"], g_rel, T["open"]):
                        plan2.append((plan2[-1][0], gg, "open"))
                    r_b = pl_b2 + np.array([0, 0, COMMON["retreat"]])
                    for q in cart_segment(pl_b2, r_b, plan2[-1][0], None, T["retreat"]):
                        plan2.append((q, g_rel, "retreat"))
                    q_r0 = plan2[-1][0]
                    grip_r = ramp(g_rel, float(rest_motor[5]), T["to_rest"])
                    for q, gg in zip(joint_segment(q_r0, q_rest6[:5], T["to_rest"]), grip_r):
                        plan2.append((q, gg, "to_rest"))
                    info["place_ik_err_mm"] = round(e_pp * 1000, 2)
                    # 접시 위로 옮기는 동안 떨어졌는지 보기 위해 구간별로 실행
                    for ph_name in ("to_plate", "lower"):
                        seg = [p for p in plan2 if p[2] == ph_name]
                        fail = run(seg)
                        if fail:
                            break
                        fz = obj_T(name)[2, 3]
                        if ph_name == "to_plate" and fz < table_z + 0.03:
                            fail = "미끄러짐"
                            break
                    if fail is None:
                        fail = run([p for p in plan2 if p[2] in ("open", "retreat", "to_rest")])
                    if fail is None:
                        # 팔이 휴식 자세에 닿을 때까지 (최대 15프레임) 같은 목표 유지
                        extra = [(q_rest6[:5], float(rest_motor[5]), "to_rest")] * 15
                        for item in extra:
                            if np.max(np.abs(q_now()[:5] - q_rest6[:5])) < 0.08:
                                break
                            fail = run([item])
                            if fail:
                                break
                    if fail is None and not bool(active_fruit_on_plate(env, FRUITS)[0]):
                        fail = "접시 밖"
                success = fail is None
                fT = obj_T(name)
                pw = obj_T(PLATE)[:3, 3]
                info["final_dxy_mm"] = [round(float(v) * 1000, 1) for v in (fT[:2, 3] - pw[:2])]
                info["final_dz_mm"] = round(float(fT[2, 3] - pw[2]) * 1000, 1)
                ep_idx = None
                if ds is not None:
                    if success:
                        ds.save_episode(parallel_encoding=False)
                        ep_idx = ds.num_episodes - 1
                    else:
                        ds.clear_episode_buffer()
                if success:
                    saved += 1
                stats[fail or "성공"] = stats.get(fail or "성공", 0) + 1
                rec = {"fruit": fruit, "attempt": attempts, "success": success, "fail": fail, "frames": n,
                       "sec": round(n / FPS, 2), "wall_s": round(time.time() - t0, 1), "episode_index": ep_idx,
                       "layout": {"kind": lay["kind"], "count": lay["count"],
                                  "extra": {FRUITS[k]: v for k, v in lay["extra"].items()},
                                  "robot": [round(v, 2) for v in lay["robot"]]},
                       "target": name, "speed": round(sp, 3), "start_perturbed": bool(np.any(q_start6 != q_rest6)),
                       "key_frames": key, "fruit_xy_init": [round(float(v), 4) for v in fruit_T0[:2, 3]],
                       "relayouts_total": relayouts, "render": os.environ.get("COURSE_RENDER", "default"), **info}
                log(rec)
                print(f"[{FRUIT_KO[fruit]}] 시도 {attempts} {'성공' if success else '실패: ' + fail} · {n}프레임 "
                      f"({n / FPS:.1f}초) · {lay['kind']} count={lay['count']} extra={rec['layout']['extra']} "
                      f"robot={rec['layout']['robot']} · 들기 {info['lift_dz'] * 100:.1f}cm · {time.time() - t0:.0f}s"
                      f" · 저장 {saved}/{args.per_fruit}", flush=True)
            totals[fruit] = {"attempts": attempts, "saved": saved, "stats": stats, "relayouts": relayouts}
            print(f"== {FRUIT_KO[fruit]}: 시도 {attempts}, 성공 {saved}, {stats}", flush=True)
        if ds is not None:
            ds.finalize()
        print("SUMMARY", json.dumps(totals, ensure_ascii=False), f"총 {time.time() - t_all:.0f}s", flush=True)
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
