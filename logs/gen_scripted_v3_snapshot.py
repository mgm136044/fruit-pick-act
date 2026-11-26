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
ap.add_argument("--vary-use", default="count=0.6,extra=0.6,robot=0.6",
                help="변형 장면에서 각 변형을 쓸 확률 (v1·v2 기본값). v3 예: count=0.9,extra=0.4,robot=0.6")
ap.add_argument("--count3-prob", type=float, default=0.4, help="같은 과일 여러 개일 때 3개일 확률 (나머지 2개)")
ap.add_argument("--dart-prob", type=float, default=0.0,
                help="에피소드마다 접근·하강 중 실행 명령에 흔들림을 넣을 확률 (기록은 흔들림 없는 명령 = 바로잡는 동작, DART). 0=끔(v1·v2)")
ap.add_argument("--dart-deg", default="2.5,2.0,2.0,2.0", help="흔들림 크기(도): 어깨 회전, 어깨 들기, 팔꿈치, 손목 굽힘")
ap.add_argument("--single-random-prob", type=float, default=0.0,
                help="변형 장면 중 과일 1개(다른 과일 없음)일 때 조리대 영역 전체에 무작위로 놓을 확률 (record.py 기본은 기준 자리 ±3cm). 0=끔(v1·v2)")
ap.add_argument("--light-prob", type=float, default=0.0, help="에피소드마다 조명(DomeLight)을 바꿀 확률. 0=끔(v1·v2)")
ap.add_argument("--light-int", default="0.5,1.5", help="조명 세기 배율 범위")
ap.add_argument("--light-col", default="0.85,1.15", help="조명 색 채널별 배율 범위")
ap.add_argument("--start-perturb-prob", type=float, default=0.25, help="시작 자세를 휴식 자세에서 조금 바꾸는 비율")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-sec", type=float, default=30.0, help="에피소드 최대 길이 (평가 기준 30초)")
ap.add_argument("--log", default=None, help="시도별 JSONL 로그 (기본 logs/gen_<데이터셋>.jsonl)")
ap.add_argument("--params", default=None, help="과일별 파라미터 덮어쓰기 JSON")
ap.add_argument("--no-save", action="store_true", help="데이터셋에 쓰지 않음 (튜닝용)")
ap.add_argument("--snap", type=int, default=0, help="처음 N 시도는 주요 순간 정면/손목 PNG 저장")
add_launcher_args(ap)
args = ap.parse_args()
VARY_USE = {k: float(v) for k, v in (x.split("=") for x in args.vary_use.split(","))}
DART_SIG = __import__("numpy").radians([float(v) for v in args.dart_deg.split(",")])
LIGHT_INT = tuple(float(v) for v in args.light_int.split(","))
LIGHT_COL = tuple(float(v) for v in args.light_col.split(","))
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
    # v2: 딸기·레몬도 메시 주축으로 보면 길쭉함 (딸기 54.6x41.9, 레몬 58.5x45.6mm) → 짧은 축을 가로질러 쥠 (v1 은 long_axis None, 폭 45/51)
    "Strawberry": {"grasp_local": [0, 0, 0.0228], "height": 0.0457, "width": 0.042, "long_axis": [-0.73, -0.68, 0], "open": 42.0, "close": 23.0},
    "Orange": {"grasp_local": [0, 0, 0.0268], "height": 0.0535, "width": 0.055, "long_axis": None, "open": 48.0, "close": 33.0},
    "Lemon": {"grasp_local": [0, 0, 0.0225], "height": 0.0451, "width": 0.046, "long_axis": [0.77, 0.63, 0], "open": 46.0, "close": 25.5},
    # v2 바나나: 가운데(y=0) 중심선이 로컬 y 에서 12.3° 돌아가 있어 y=-20mm 지점(중심선이 y 와 거의 나란함, 폭 31.6mm)을 잡음
    #   (v1: grasp_local [-0.0285, 0, 0.0164], width 0.034, long_axis [0, 1, 0], 고정 손가락 여유 5mm)
    "Banana": {"grasp_local": [-0.0303, -0.020, 0.0160], "height": 0.0312, "width": 0.032, "long_axis": [0.06, 1, 0], "open": 36.0, "close": 16.0,
               "fixed_clear": 0.007},
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
    "release_extra": 6.0,
    "finger_check": True,      # v2: 벌린 손가락이 내려갈 자리에 다른 과일이 있으면 방향/대상/배치를 바꿈 (v1 은 없음)
    "finger_clear_min": 0.008,
    "balance_kinds": True,     # v2: 저장된 에피소드의 기본/변형 비율을 맞춤      # 놓을 때 접근 때보다 더 여는 양 (None 이면 v1 방식)
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
    drng = np.random.default_rng(args.seed + 7777)  # DART 전용 (기존 난수 순서를 바꾸지 않게)
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
            if lay["count"] == 1 and not lay["extra"] and not lay.get("single_random"):  # 기본 장면: 기준 자리 ±3cm, yaw 0
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

        if args.light_prob > 0:  # eval_proxy.py 와 같은 조명 (/World/Light)
            import omni.usd
            _light = omni.usd.get_context().get_stage().GetPrimAtPath("/World/Light")
            l_int, l_col = _light.GetAttribute("inputs:intensity"), _light.GetAttribute("inputs:color")
            base_int, base_col = l_int.Get(), l_col.Get()
        kind_saved = {"default": 0, "vary": 0}  # 과일마다 저장된 장면 종류 수 (비율 맞추기)

        def sample_layout(fi):
            # v2: 저장된 에피소드 기준으로 변형 비율이 vary_prob 에 가깝게 (v1 은 시도마다 독립 추첨 → 실패가 많은 변형 장면이 줄어듦)
            tot = kind_saved["default"] + kind_saved["vary"]
            if COMMON["balance_kinds"] and tot > 0:
                want_vary = kind_saved["vary"] < args.vary_prob * (tot + 1) - 0.5
            else:
                want_vary = rng.random() < args.vary_prob
            if not want_vary:
                return {"kind": "default", "count": 1, "extra": {}, "robot": (0.0, 0.0, 0.0)}
            use = {k: rng.random() < VARY_USE[k] for k in ("count", "extra", "robot")}
            if not any(use.values()):
                use[rng.choice(["count", "extra", "robot"])] = True
            lay = {"kind": "vary", "count": 1, "extra": {}, "robot": (0.0, 0.0, 0.0)}
            if use["count"]:
                lay["count"] = int(rng.choice([2, 3], p=[1 - args.count3_prob, args.count3_prob]))
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
            kin.tcp = np.array([COMMON["fixed_inner_x"] + P.get("fixed_clear", COMMON["fixed_clear"]) + P["width"] / 2, 0.0,
                                COMMON["tcp_depth"]])

        def fruit_grasp_world(fruit, name):
            T = obj_T(name)
            P = FRUIT_P[fruit]
            c = T[:3, :3] @ np.array(P["grasp_local"]) + T[:3, 3]
            close_w = None
            if P["long_axis"] is not None:
                la = T[:3, :3] @ np.array(P["long_axis"], dtype=np.float64)
                la[2] = 0
                if np.linalg.norm(la) > 0.3:  # 긴 축이 거의 수직이면 방향이 불안정 → 정렬 안 함
                    la /= np.linalg.norm(la)
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

        # 손가락 간섭 검사용: 과일 종류별 메시 점 (root 프레임, explore_scene.py 가 저장) 일부
        PTS = {}
        for f_ in FRUITS:
            pp = np.load(os.path.join(HERE, "logs", f"fruit_pts_{f_}.npy")).astype(np.float64)
            PTS[f_] = pp[np.random.default_rng(0).choice(len(pp), min(800, len(pp)), replace=False)]
        from orange_recover import gap_at
        finger_rejects = [0]

        def other_points(exclude):
            out = []
            for f_ in FRUITS:
                for o in copies(f_):
                    if o == exclude:
                        continue
                    T_ = obj_T(o)
                    if T_[2, 3] < 0.5:
                        continue
                    out.append(PTS[f_] @ T_[:3, :3].T + T_[:3, 3])
            return np.concatenate(out) if out else np.zeros((0, 3))

        def finger_clearance(Tb_, q5, P_, others):
            """벌린 두 손가락(끝 근처)과 다른 과일 메시 사이 최소 수평 거리 (m)."""
            if len(others) == 0:
                return 1.0
            gap = gap_at(P_["open"] * 1.1 - 10)
            xs = [COMMON["fixed_inner_x"] - 0.006, COMMON["fixed_inner_x"] + gap + 0.006]
            pts = np.array([[x, y, z] for x in xs for y in (-0.008, 0.0, 0.008) for z in (-0.100, -0.085, -0.070)])
            Tg = Tb_ @ kin.fk_gripper(q5)
            fw = pts @ Tg[:3, :3].T + Tg[:3, 3]
            d = np.hypot(fw[:, None, 0] - others[None, :, 0], fw[:, None, 1] - others[None, :, 1])
            return float(d.min())

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
            kind_saved.update({"default": 0, "vary": 0})
            stats = {}
            while saved < args.per_fruit and attempts < max_att:
                t0 = time.time()
                # ---- 장면 만들기 (도달 가능한 배치가 나올 때까지)
                for _try in range(30):
                    lay = sample_layout(fi)
                    if args.single_random_prob > 0 and lay["kind"] == "vary" and lay["count"] == 1 and not lay["extra"]:
                        lay["single_random"] = bool(drng.random() < args.single_random_prob)
                    if args.light_prob > 0:  # 조명: 실행마다 기준값에서 다시 정함 (기록 형식은 그대로)
                        if drng.random() < args.light_prob:
                            li = float(drng.uniform(*LIGHT_INT))
                            lc = [float(v) for v in drng.uniform(*LIGHT_COL, 3)]
                            l_int.Set(base_int * li)
                            l_col.Set(type(base_col)(*[base_col[i] * lc[i] for i in range(3)]))
                            lay["light"] = [round(li, 3)] + [round(v, 3) for v in lc]
                        else:
                            l_int.Set(base_int)
                            l_col.Set(base_col)
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
                    close_override = None
                    for _, name, c_w, close_w in cand:
                        table_z = c_w[2] - P["grasp_local"][2]
                        g_w = c_w.copy()
                        g_w[2] = max(c_w[2], table_z + COMMON["tip_below_tcp"] + COMMON["tip_clear"])
                        g_b = to_base(Tb, g_w)
                        pl_b = to_base(Tb, [plate_w[0], plate_w[1], plate_w[2] + COMMON["plate_rim"] + 0.08])
                        q_p, e_p, ok_p = ik_ok(pl_b, seed_for(pl_b), None, tilt_max=75)
                        if not ok_p:
                            continue
                        h_b = to_base(Tb, g_w + [0, 0, COMMON["hover"]])
                        # 닫는 방향 후보: 길쭉한 과일은 짧은 축 하나, 둥근 과일은 기본 방향 → ±45° → ±90°
                        opts = [close_w]
                        if close_w is None and COMMON["finger_check"]:
                            q0g, _, _ = ik_ok(g_b, seed_for(g_b), None)
                            d0 = Tb[:3, :3] @ kin.fk_gripper(q0g)[:3, 0]
                            d0[2] = 0
                            d0 /= max(np.linalg.norm(d0), 1e-9)
                            opts = [None] + [np.array([math.cos(t) * d0[0] - math.sin(t) * d0[1],
                                                       math.sin(t) * d0[0] + math.cos(t) * d0[1], 0.0])
                                             for t in np.radians([45, -45, 90])]
                        others = other_points(name) if COMMON["finger_check"] else None
                        for cw_opt in opts:
                            c_b = dir_to_base(Tb, cw_opt)
                            q_g, e_g, ok_g = ik_ok(g_b, seed_for(g_b), c_b)
                            q_h, e_h, ok_h = ik_ok(h_b, q_g, c_b, tilt_max=60)
                            if not (ok_g and ok_h):
                                continue
                            if others is not None and finger_clearance(Tb, q_g, P, others) < COMMON["finger_clear_min"]:
                                finger_rejects[0] += 1
                                continue
                            ok, close_override = True, cw_opt
                            break
                        if ok:
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
                dart_on = args.dart_prob > 0 and drng.random() < args.dart_prob
                if dart_on:  # 흔들림을 멈춤 동안 없애고 과일 바로 위로 돌아온 뒤 내려가도록 멈춤을 늘림
                    T["hover_wait"] += 10
                c_w, close_w, _ = fruit_grasp_world(fruit, name)
                if close_override is not None:
                    close_w = close_override
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
                # DART: 실행 명령에만 느린 무작위 흔들림(OU, 시간상수 약 1초). 기록하는 action 은 흔들림 없는 명령.
                # 과일 위로 가는 동안 최대, 멈춤 앞부분에서 0 으로 줄이고 멈춤을 10프레임 늘려 내려가기 전에 제자리로
                # (v3 시험: 내려가는 동안까지 남기면 팔이 늦게 돌아와 손가락이 과일 위에 얹힘 → 레몬 35mm 위에서 막힘)
                dart_x = np.zeros(4)
                n_hw = sum(1 for p in plan if p[2] == "hover_wait")
                hw_i = [0]
                fail, info = None, {}
                key = {}
                max_frames = int(args.max_sec * FPS)

                def run(plan_items):
                    nonlocal n, bias, obs, dart_x
                    for q_des, gmot, ph in plan_items:
                        if n >= max_frames:
                            return "시간 초과"
                        qa = q_now()
                        dn = np.zeros(5)  # DART 흔들림 (실행에만)
                        if dart_on:
                            dart_x = 0.967 * dart_x + DART_SIG * math.sqrt(1 - 0.967 ** 2) * drng.standard_normal(4)
                            if ph == "to_hover":
                                fac = 1.0
                            elif ph == "hover_wait":  # 멈춤 앞쪽 40% 안에 0 으로 → 나머지 동안 팔이 제자리로 돌아옴
                                fac = max(0.0, 1.0 - hw_i[0] / max(1, int(0.4 * n_hw)))
                                hw_i[0] += 1
                            else:
                                fac = 0.0
                            dn[:4] = fac * dart_x
                        if ph in ("hover_wait", "descend", "close", "settle", "lift", "lower", "open", "retreat"):
                            # 적분 보정은 실제로 보낸(흔들린) 목표 기준 → 흔들림을 상쇄하지 않고, 기록 action 도 오염 안 됨
                            bias[:4] += COMMON["ki"] * (q_des[:4] + dn[:4] - qa[:4])
                            bias = np.clip(bias, -COMMON["bias_max"], COMMON["bias_max"])
                        ff = np.zeros(5)
                        if COMMON.get("gravity_ff", True):  # 중력 처짐 보정: 목표 += 중력 토크 / 관절 강성
                            tau_g = robot.data.gravity_compensation_forces.torch[0].cpu().numpy().astype(np.float64)
                            ff = tau_g[:5] / ARM_K
                        q_cmd = np.clip(q_des + ff + bias, kin.lo, kin.hi)
                        a_mot = rad_to_motor(np.r_[q_cmd, 0.0])
                        a_mot[5] = gmot
                        a_exec = a_mot
                        if np.any(dn):
                            a_exec = rad_to_motor(np.r_[np.clip(q_cmd + dn, kin.lo, kin.hi), 0.0])
                            a_exec[5] = gmot
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
                        obs, *_ = env.step(torch.tensor(motor_to_rad(a_exec), dtype=torch.float32, device=dev).unsqueeze(0))
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
                    # 놓을 때: v1 은 min(close+14, open). 레몬이 손가락 사이에 끼어 딸려 간 실패(v1 레몬 7) 뒤로 open+release_extra 로 넓힘
                    g_rel = P["open"] + COMMON["release_extra"] if COMMON["release_extra"] is not None else min(P["close"] + 14.0, P["open"])
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
                    kind_saved[lay["kind"]] += 1
                stats[fail or "성공"] = stats.get(fail or "성공", 0) + 1
                rec = {"fruit": fruit, "attempt": attempts, "success": success, "fail": fail, "frames": n,
                       "sec": round(n / FPS, 2), "wall_s": round(time.time() - t0, 1), "episode_index": ep_idx,
                       "layout": {"kind": lay["kind"], "count": lay["count"],
                                  "extra": {FRUITS[k]: v for k, v in lay["extra"].items()},
                                  "robot": [round(v, 2) for v in lay["robot"]],
                                  "single_random": bool(lay.get("single_random", False)), "light": lay.get("light")},
                       "target": name, "speed": round(sp, 3), "start_perturbed": bool(np.any(q_start6 != q_rest6)), "dart": bool(dart_on),
                       "key_frames": key, "fruit_xy_init": [round(float(v), 4) for v in fruit_T0[:2, 3]],
                       "relayouts_total": relayouts, "finger_rejects_total": finger_rejects[0],
                       "close_override": None if close_override is None else [round(float(v), 3) for v in close_override], "render": os.environ.get("COURSE_RENDER", "default"), **info}
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
