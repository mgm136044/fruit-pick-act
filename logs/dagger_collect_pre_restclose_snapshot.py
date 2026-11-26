"""개입형 DAgger 수집기 + P0 평가 (91점 대응 지시 P1, Fable 검토 반영 11-26 10:00).
정책이 로봇을 몰고, 실패 조짐이 보이면 스크립트 전문가가 '지금 상태에서' 이어받아 끝까지 한다. 전문가가 성공한 것만 저장한다.

  source ~/robot_course/activate.sh
  COURSE_ASSETS=~/fruit_datagen/assets_c4 python dagger_collect.py --checkpoint te_models/act_mix2_100k_color_te \
      --dataset ~/datasets/fruit_dagger_r1a --attempts 40 --round 1 --viz none

정책 실행: 키트 lerobot_policy(load_policy + act), 체크포인트 config 그대로(제출 zip 폴더), 시작 3스텝 유지 — 서버와 같다.
장면 조건: eval_sweep.py 와 같은 형식(--conds).

롤아웃 두 종류
  평가형(1 − --early-frac): 넘겨받기 조건이 처음 걸린 것 = 그 조건의 실패 유형(P0 표). 그중 --control-frac 은 대조군:
                            넘겨받지 않고 --control-steps 까지 둔다(혼자 회복하는지 측정, 저장 안 함)
  이른 넘겨받기(--early-frac): 접근 중 무작위 시점(20~110스텝)에 넘겨받음 = 정책이 실제로 지나는 상태에 전문가 라벨
넘겨받기 조건 (Fable 설계 2-2, 시작값)
  T1_miss   쥐기 전, 손끝이 잡는 높이 +2.5cm 아래인데 가장 가까운 사본에서 수평 > --miss-mm 가 5스텝 연속 → 즉시
  T1_push   쥐기 전 목표 사본을 > --push-mm 밂 → 즉시
  T2_grasp  닫힌 뒤 손끝이 > --rise-cm 올라갔는데 들림 < 1.5cm → 0~--t2-delay 스텝 무작위 지연 뒤
  T3_drop   3cm 넘게 들렸던 사본이 손끝에서 6cm 넘게 멀어지고 접시 12cm 밖 → 과일이 멈춘 뒤 0~--t3-delay 스텝
  T4_stall  관절 변화 < 0.2°/스텝 이 --stall-steps 연속 / T4_noprog 이정표(접근 5cm·닫기·들림 3cm·접시 12cm)가 --milestone-steps 동안 그대로
  T5_late   --t5-steps 까지 3cm 넘게 든 적 없음
전문가 (gen_scripted.py 의 IK·구간·중력·적분 보정, 잡기 값 --params)
  쥐고 있음 → (낮으면 먼저 들기) 접시 위 → 내려놓기 → 놓기 → 위로(retreat)에서 끝. to_rest 는 넣지 않는다
  빈손     → 그리퍼 벌리며 위로 → 목표 위 → 내려가기 → 쥐기 → 들기 → 위와 같이. 이미 과일 위에 맞춰 있으면 바로 내려감
  전문가 잡기가 실패하면 기록을 지우고 다시 계획(최대 --expert-tries). 저장되는 것은 성공한 마지막 시도뿐
  목표 선택(관측만으로 정해지는 규칙): 쥔 사본 → 손끝이 어떤 사본 6cm 안·낮고 다른 사본보다 3cm 이상 가까우면 그 사본
            → 아니면 로봇에서 가장 가까운 사본(mix3 A 와 같은 규칙). 둘째와 2.5cm 미만이면 ambiguous 표시
  쓰러진 딸기·레몬(기울기 > --tilt-max)은 후보에서 뺀다. 과일이 구르는 중이면 멈출 때까지(최대 30스텝) 정책이 계속 몬다
저장: 넘겨받은 프레임부터(라벨을 거슬러 붙이지 않음). LeRobot 형식은 gen_scripted/record.py 와 같다.
기록(JSONL): 라운드, 운전 체크포인트, 조건·장면, 롤아웃 종류, 처음 걸린 조건·스텝, 넘겨받은 스텝·지연, 목표, ambiguous, 전문가 시도 수, 프레임 수, 결과.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import sys
import time

KIT = os.environ.get("ROBOT_KIT", os.path.expanduser("~/student_kit"))
sys.path.insert(0, os.path.join(KIT, "isaac"))
sys.path.insert(0, KIT)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

DEF_CONDS = ("xy6:xy=0.06;xy10:xy=0.10;yaw45:yaw=45;yaw180:yaw=180;rob5:rob=5/15;rob10:rob=10/30;plate6:plate=0.06;"
             "cnt23:count=2-3;cnt4:count=4;mix:count=2-4,rob=10/30,plate=0.06")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--no-save", action="store_true")
ap.add_argument("--conds", default=DEF_CONDS)
ap.add_argument("--attempts", type=int, default=24, help="조건마다 롤아웃 수 (과일 4종 돌아가며)")
ap.add_argument("--cond-weights", default=None, help="이름=배수,... (기본 모두 1). 예: cnt4=2,cnt23=1.5")
ap.add_argument("--max-saved", type=int, default=0)
ap.add_argument("--round", type=int, default=1)
ap.add_argument("--seed", type=int, default=100)
ap.add_argument("--params", default=os.path.join(HERE, "params_mix3.json"))
ap.add_argument("--max-sec", type=int, default=30, help="정책 + 전문가 합계 상한 (서버와 같음)")
ap.add_argument("--early-frac", type=float, default=0.30)
ap.add_argument("--control-frac", type=float, default=0.15)
ap.add_argument("--control-steps", type=int, default=450)
ap.add_argument("--t5-steps", type=int, default=330)
ap.add_argument("--t2-delay", type=int, default=90)
ap.add_argument("--t3-delay", type=int, default=45)
ap.add_argument("--miss-mm", type=float, default=12.0)
ap.add_argument("--push-mm", type=float, default=15.0)
ap.add_argument("--rise-cm", type=float, default=3.0)
ap.add_argument("--stall-steps", type=int, default=45)
ap.add_argument("--milestone-steps", type=int, default=150)
ap.add_argument("--tilt-max", type=float, default=25.0)
ap.add_argument("--expert-tries", type=int, default=2)
ap.add_argument("--log", default=None)
ap.add_argument("--gif-dir", default=None, help="저장한 에피소드 처음 --gif-n 개 + 전문가 실패 처음 --gif-n 개를 GIF(정면|손목, 전문가 구간 빨간 띠)")
ap.add_argument("--gif-n", type=int, default=12)
add_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True

FPS = 30
AREA_X, AREA_Y = (1.95, 2.30), (-0.48, -0.26)
PLATE_CLEAR, ROBOT_CLEAR, GAP, GAP_BANANA = 0.14, 0.17, 0.075, 0.11
CONTACT_CMD = {"Strawberry": 26.6, "Orange": 36.9, "Lemon": 29.6, "Banana": 18.2}
FRUIT_P = {  # gen_scripted.py 와 같은 v3 값 + --params
    "Strawberry": {"grasp_local": [0, 0, 0.0228], "height": 0.0457, "width": 0.042, "long_axis": [-0.73, -0.68, 0], "open": 42.0, "close": 23.0},
    "Orange": {"grasp_local": [0, 0, 0.0268], "height": 0.0535, "width": 0.055, "long_axis": None, "open": 48.0, "close": 33.0},
    "Lemon": {"grasp_local": [0, 0, 0.0225], "height": 0.0451, "width": 0.046, "long_axis": [0.77, 0.63, 0], "open": 46.0, "close": 25.5},
    "Banana": {"grasp_local": [-0.0303, -0.020, 0.0160], "height": 0.0312, "width": 0.032, "long_axis": [0.06, 1, 0], "open": 36.0, "close": 16.0,
               "fixed_clear": 0.007},
}
COMMON = {"fixed_inner_x": -0.0105, "fixed_clear": 0.005, "tcp_depth": -0.085, "tip_below_tcp": 0.020, "tip_clear": 0.006,
          "hover": 0.07, "lift": 0.10, "plate_rim": 0.037, "plate_inner": 0.008, "drop": 0.012, "rim_clear": 0.03, "retreat": 0.06,
          "release_extra": 6.0, "ki": 0.08, "bias_max": 0.25,
          "T": {"to_hover": 60, "hover_wait": 5, "descend": 30, "close": 15, "settle": 6, "lift": 25, "to_plate": 50, "lower": 25,
                "open": 10, "retreat": 15, "to_rest": 50}}
if args.params:
    ov = json.load(open(args.params))
    for k, v in ov.get("fruits", {}).items():
        FRUIT_P[k].update(v)
    for k, v in ov.get("common", {}).items():
        if k == "T":
            COMMON["T"].update(v)
        elif k in COMMON:
            COMMON[k] = v


def parse_conds(s):
    out = []
    for part in [p for p in s.split(";") if p.strip()]:
        name, _, kv = part.partition(":")
        c = {"name": name.strip()}
        for item in [x for x in kv.split(",") if x.strip()]:
            k, _, v = item.partition("=")
            c[k.strip()] = v.strip()
        out.append(c)
    return out


def min_jerk(n):
    import numpy as np
    t = np.linspace(0, 1, n + 1)[1:]
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def save_gif(frames, path):
    import numpy as np
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w, h = 192, 144
    out = []
    for fr, wr, expert in frames:
        im = Image.new("RGB", (w * 2 + 4, h), (250, 250, 248))
        im.paste(Image.fromarray(np.asarray(fr).astype(np.uint8)).resize((w, h)), (0, 0))
        im.paste(Image.fromarray(np.asarray(wr).astype(np.uint8)).resize((w, h)), (w + 4, 0))
        if expert:
            im.paste((220, 40, 40), (0, 0, w * 2 + 4, 5))
        out.append(im.quantize(colors=64, method=Image.Quantize.MEDIANCUT))
    out[0].save(path, save_all=True, append_images=out[1:], duration=133, loop=0, optimize=True)


def main():
    import numpy as np
    import torch

    from course_env.mdp import active_fruit_on_plate, active_index, park_fruits
    from course_env.paths import FRUIT_KO, FRUITS, PLATE, copies
    from course_env.pick_fruit import SO101_CFG, PickFruitEnvCfg, task_text
    from course_env.so101 import FEATURE_NAMES, JOINTS, motor_to_rad, rad_to_motor, rest_pose_rad
    from kin import SO101Kin, quat_wxyz_to_mat
    from lerobot_policy import load_policy

    conds = parse_conds(args.conds)
    if any("4" in c.get("count", "") for c in conds):
        assert len(copies(FRUITS[0])) >= 4, "count 4 조건은 COURSE_ASSETS=~/fruit_datagen/assets_c4 로 실행"
    weights = {k: float(v) for k, v in (x.split("=") for x in args.cond_weights.split(","))} if args.cond_weights else {}
    ckpt = os.path.abspath(os.path.expanduser(args.checkpoint))
    policy = load_policy(ckpt, "cuda", ["act"])
    pc = policy.policy.config
    driver = {"checkpoint": ckpt, "n_action_steps": pc.n_action_steps, "te": getattr(pc, "temporal_ensemble_coeff", None)}
    kin = SO101Kin()
    q_rest6 = np.array([rest_pose_rad()[j] for j in JOINTS])
    root = os.path.abspath(os.path.expanduser(args.dataset))
    log_path = args.log or os.path.join(HERE, "logs", f"dagger_{os.path.basename(root.rstrip(os.sep))}.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    ds = None
    if not args.no_save:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        repo_id = f"local/{os.path.basename(root.rstrip(os.sep))}"
        vec = {"dtype": "float32", "shape": (6,), "names": FEATURE_NAMES}
        img = {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
        features = {"observation.state": vec, "action": vec, "observation.images.front": img, "observation.images.wrist": img}
        if args.resume:
            ds = LeRobotDataset.resume(repo_id, root=root, image_writer_threads=4)
        else:
            if os.path.exists(root):
                sys.exit(f"{root} 가 이미 있습니다 (--resume 또는 다른 이름)")
            ds = LeRobotDataset.create(repo_id, fps=FPS, features=features, root=root, robot_type="so101_follower",
                                       use_videos=True, image_writer_threads=4)
        print(f"데이터셋: {root} (지금까지 {ds.num_episodes} 에피소드)", flush=True)

    cfg = PickFruitEnvCfg()
    cfg.seed = args.seed
    cfg.terminations.time_out = None
    cfg.terminations.success = None
    with launch_simulation(cfg, args):
        from isaaclab.envs import ManagerBasedRLEnv
        import omni.usd
        env = ManagerBasedRLEnv(cfg)
        dev = env.device
        ids = torch.tensor([0], device=dev)
        origin = env.scene.env_origins[0]
        org = origin.cpu().numpy().astype(np.float64)
        robot = env.scene["robot"]
        base_pos = torch.tensor(SO101_CFG.init_state.pos, device=dev)
        zero6 = torch.zeros(1, 6, device=dev)
        ARM_K = float(SO101_CFG.actuators["arm"].stiffness)
        light = omni.usd.get_context().get_stage().GetPrimAtPath("/World/Light")
        l_int, l_col = light.GetAttribute("inputs:intensity"), light.GetAttribute("inputs:color")
        base_int, base_col = l_int.Get(), l_col.Get()
        plate_base = list(getattr(env.cfg.scene, PLATE).init_state.pos)

        # ---------------- 장면 (eval_sweep.py 와 같은 규칙) ----------------
        def put(name, x, y, z, yaw=0.0):
            pose = torch.zeros(1, 7, device=dev)
            pose[0, :3] = origin + torch.tensor([x, y, z], device=dev)
            pose[0, 3:] = torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)], device=dev)
            env.scene[name].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            env.scene[name].write_root_velocity_to_sim_index(root_velocity=zero6, env_ids=ids)

        def place_robot(rob):
            fwd, right, yaw = rob
            x, y, z = (base_pos + torch.tensor([right / 100, fwd / 100, 0.0], device=dev)).tolist()
            th = math.pi + math.radians(yaw)
            pose = torch.zeros(1, 7, device=dev)
            pose[0, :3] = origin + torch.tensor([x, y, z], device=dev)
            pose[0, 3:] = torch.tensor([0.0, 0.0, math.sin(th / 2), math.cos(th / 2)], device=dev)
            robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            robot.write_root_velocity_to_sim_index(root_velocity=zero6, env_ids=ids)

        def setup(c, fi, rng):
            info = {}
            rob = (0.0, 0.0, 0.0)
            if "rob" in c:
                cm, deg = (float(v) for v in c["rob"].split("/"))
                rob = (float(rng.uniform(-cm, cm)), float(rng.uniform(-cm, cm)), float(rng.uniform(-deg, deg)))
            place_robot(rob)
            info["robot"] = [round(v, 2) for v in rob]
            rb = (float(base_pos[0]) + rob[1] / 100, float(base_pos[1]) + rob[0] / 100)
            plate = plate_base[:2]
            if "plate" in c:
                r = float(c["plate"])
                d = np.zeros(2)
                for _ in range(100):
                    dd = rng.uniform(-r, r, 2)
                    if math.hypot(plate_base[0] + dd[0] - rb[0], plate_base[1] + dd[1] - rb[1]) >= 0.22:
                        d = dd
                        break
                plate = [plate_base[0] + d[0], plate_base[1] + d[1]]
                put(PLATE, plate[0], plate[1], plate_base[2])
                info["plate_cm"] = [round(float(v) * 100, 1) for v in d]
            active_index(env)[0] = fi
            park_fruits(env, ids, list(FRUITS), torch.tensor([-1], device=dev))
            z = getattr(env.cfg.scene, FRUITS[0]).init_state.pos[2]
            if "count" in c:
                a_, _, b_ = c["count"].partition("-")
                n = int(rng.integers(int(a_), int(b_ or a_) + 1))
                names = copies(FRUITS[fi])[:n]
                best = []
                for _ in range(200):
                    spots = []
                    for name in names:
                        banana = name.startswith("Banana")
                        for _ in range(100):
                            x, y = rng.uniform(*AREA_X), rng.uniform(*AREA_Y)
                            if (math.hypot(x - plate[0], y - plate[1]) >= PLATE_CLEAR and math.hypot(x - rb[0], y - rb[1]) >= ROBOT_CLEAR
                                    and all(math.hypot(x - sx, y - sy) >= (GAP_BANANA if banana or sb else GAP) for _, sx, sy, sb in spots)):
                                spots.append((name, x, y, banana))
                                break
                    if len(spots) > len(best):
                        best = spots
                    if len(best) == len(names):
                        break
                for name, x, y, _ in best:
                    put(name, x, y, z, float(rng.uniform(-math.pi, math.pi)))
                info["count"] = len(best)
            else:
                base = getattr(env.cfg.scene, FRUITS[fi]).init_state.pos
                r = float(c.get("xy", 0.03))
                yaw = math.radians(float(c.get("yaw", 0)))
                x, y = base[0], base[1]
                for _ in range(100):
                    dx, dy = rng.uniform(-r, r, 2)
                    if (math.hypot(base[0] + dx - plate[0], base[1] + dy - plate[1]) >= PLATE_CLEAR
                            and math.hypot(base[0] + dx - rb[0], base[1] + dy - rb[1]) >= ROBOT_CLEAR):
                        x, y = base[0] + dx, base[1] + dy
                        break
                put(FRUITS[fi], x, y, base[2], float(rng.uniform(-yaw, yaw)) if yaw > 0 else 0.0)
                info["count"] = 1
            if c.get("light") == "1":
                li, lc = float(rng.uniform(0.5, 1.5)), rng.uniform(0.85, 1.15, 3).tolist()
                l_int.Set(base_int * li)
                l_col.Set(type(base_col)(*[base_col[i] * lc[i] for i in range(3)]))
                info["light"] = [round(li, 3)] + [round(v, 3) for v in lc]
            else:
                l_int.Set(base_int)
                l_col.Set(base_col)
            return info

        # ---------------- 전문가 (gen_scripted.py 와 같은 계산) ----------------
        def pose_T(p7):
            T_ = np.eye(4)
            x, y, z, w = p7[3:7]
            T_[:3, :3] = quat_wxyz_to_mat([w, x, y, z])
            T_[:3, 3] = p7[:3]
            return T_

        def base_T():
            return pose_T(robot.data.root_link_pose_w.torch[0].cpu().numpy().astype(np.float64))

        def obj_T(name):
            return pose_T(env.scene[name].data.root_link_pose_w.torch[0].cpu().numpy().astype(np.float64))

        def q_now():
            return robot.data.joint_pos.torch[0].cpu().numpy().astype(np.float64)

        def set_tcp(fruit):
            P_ = FRUIT_P[fruit]
            kin.tcp = np.array([COMMON["fixed_inner_x"] + P_.get("fixed_clear", COMMON["fixed_clear"]) + P_["width"] / 2, 0.0, COMMON["tcp_depth"]])

        def tcp_w():
            Tb = base_T()
            return Tb[:3, :3] @ kin.tcp_frame(q_now()[:5])[0] + Tb[:3, 3]

        def pos_w(n):
            return env.scene[n].data.root_pos_w.torch[0].cpu().numpy().astype(np.float64)

        def speed(n):
            return float(np.linalg.norm(env.scene[n].data.root_lin_vel_w.torch[0].cpu().numpy()))

        def tilt_deg(n):
            return float(np.degrees(np.arccos(np.clip(obj_T(n)[2, 2], -1.0, 1.0))))

        def fruit_grasp_world(fruit, name):
            T_ = obj_T(name)
            P_ = FRUIT_P[fruit]
            c_ = T_[:3, :3] @ np.array(P_["grasp_local"]) + T_[:3, 3]
            close_w = None
            if P_["long_axis"] is not None:
                la = T_[:3, :3] @ np.array(P_["long_axis"], dtype=np.float64)
                la[2] = 0
                if np.linalg.norm(la) > 0.3:
                    la /= np.linalg.norm(la)
                    close_w = np.array([-la[1], la[0], 0.0])
            return c_, close_w

        def to_base(Tb, p):
            return Tb[:3, :3].T @ (np.asarray(p) - Tb[:3, 3])

        def dir_to_base(Tb, d):
            return None if d is None else Tb[:3, :3].T @ np.asarray(d)

        def ik_ok(p_b, seed, c_b=None, tol=0.003, tilt_max=45.0):
            q, e = kin.ik_pos_first(p_b, seed, c_goal=c_b)
            return q, e, (e < tol and kin.tilt_deg(q) < tilt_max)

        def seed_for(p_b):
            pan = math.atan2(-p_b[0], -p_b[1])
            return np.array([np.clip(pan, kin.lo[0], kin.hi[0]), 0.3, -0.3, 1.4, 0.0])

        def cart_segment(p0, p1, q_seed, c_b, n):
            qs, q = [], q_seed
            for s_ in min_jerk(n):
                q, _ = kin.ik_pos_first(p0 + (p1 - p0) * s_, q, c_goal=c_b)
                qs.append(q)
            return qs

        def joint_segment(q0, q1, n):
            return [q0 + (q1 - q0) * s_ for s_ in min_jerk(n)]

        def ramp(g0, g1, n):
            return [g0 + (g1 - g0) * s_ for s_ in min_jerk(n)]

        T = COMMON["T"]

        def plan_grasp(fruit, name):
            """지금 자세에서 쥐고 들기까지. (plan, 정보) 또는 (None, 이유)."""
            P_ = FRUIT_P[fruit]
            Tb = base_T()
            q0 = q_now()
            g0 = float(rad_to_motor(q0)[5])
            c_w, close_w = fruit_grasp_world(fruit, name)
            table_z = c_w[2] - P_["grasp_local"][2]
            g_w = c_w.copy()
            g_w[2] = max(c_w[2], table_z + COMMON["tip_below_tcp"] + COMMON["tip_clear"])
            h_w, l_w = g_w + [0, 0, COMMON["hover"]], g_w + [0, 0, COMMON["lift"]]
            c_b = dir_to_base(Tb, close_w)
            g_b, h_b, l_b = to_base(Tb, g_w), to_base(Tb, h_w), to_base(Tb, l_w)
            q_g, _, okg = ik_ok(g_b, seed_for(g_b), c_b)
            q_h, _, okh = ik_ok(h_b, q_g, c_b)
            if not (okg and okh):
                return None, "IK"
            plan = []
            t_b = kin.tcp_frame(q0[:5])[0]
            aligned = (np.hypot(*(t_b[:2] - g_b[:2])) < 0.008 and g0 >= P_["open"] - 3 and g_b[2] - 0.005 <= t_b[2] <= h_b[2] + 0.01)
            if aligned:  # 이미 과일 위에 맞춰 있음 → 바로 내려감
                n_d = int(np.clip((t_b[2] - g_b[2]) / 0.0025, 8, T["descend"]))
                for q in cart_segment(t_b, g_b, q0[:5], c_b, n_d):
                    plan.append((q, P_["open"], "descend"))
            else:
                up_b = t_b.copy()
                up_b[2] = max(t_b[2], h_b[2])
                n_up = 18 if up_b[2] - t_b[2] > 0.01 else 8
                q_up, _ = kin.ik_pos_first(up_b, q0[:5], c_goal=None)
                for q, gg in zip(joint_segment(q0[:5], q_up, n_up), ramp(g0, P_["open"], n_up)):
                    plan.append((q, gg, "rise_open"))
                dq = float(np.abs(q_h - plan[-1][0]).max())
                n_mv = int(np.clip(dq / 0.03, 20, T["to_hover"]))
                for q in joint_segment(plan[-1][0], q_h, n_mv):
                    plan.append((q, P_["open"], "to_hover"))
                for _ in range(T["hover_wait"]):
                    plan.append((q_h, P_["open"], "hover_wait"))
                for q in cart_segment(h_b, g_b, q_h, c_b, T["descend"]):
                    plan.append((q, P_["open"], "descend"))
            for gg in ramp(P_["open"], P_["close"], T["close"]):
                plan.append((plan[-1][0], gg, "close"))
            for _ in range(T["settle"]):
                plan.append((plan[-1][0], P_["close"], "settle"))
            for q in cart_segment(g_b, l_b, plan[-1][0], c_b, T["lift"]):
                plan.append((q, P_["close"], "lift"))
            return plan, ("aligned" if aligned else "regrasp")

        def plan_place(fruit, name):
            """쥐고 있는 상태에서: (낮으면 들기) 접시 위 → 내려놓기 → 놓기 → 위로. 실제 쥔 위치를 보고."""
            P_ = FRUIT_P[fruit]
            Tb = base_T()
            q0 = q_now()
            fc_w, _ = fruit_grasp_world(fruit, name)
            d_off = fc_w - tcp_w()
            plate_w = obj_T(PLATE)[:3, 3]
            ga = P_["grasp_local"][2]
            cz_place = plate_w[2] + COMMON["plate_inner"] + COMMON["drop"] + ga
            cz_pre = plate_w[2] + COMMON["plate_rim"] + COMMON["rim_clear"] + ga
            pp_b, pl_b2 = to_base(Tb, np.r_[plate_w[:2], cz_pre] - d_off), to_base(Tb, np.r_[plate_w[:2], cz_place] - d_off)
            q_pp, _, okp = ik_ok(pp_b, seed_for(pp_b), None, tilt_max=80)
            if not okp:
                return None
            plan = []
            t_b = kin.tcp_frame(q0[:5])[0]
            if t_b[2] < pp_b[2] - 0.02:
                up = t_b.copy()
                up[2] = pp_b[2]
                q_up, _ = kin.ik_pos_first(up, q0[:5], c_goal=None)
                for q in joint_segment(q0[:5], q_up, 15):
                    plan.append((q, P_["close"], "lift"))
            start = plan[-1][0] if plan else q0[:5]
            for q in joint_segment(start, q_pp, T["to_plate"]):
                plan.append((q, P_["close"], "to_plate"))
            for q in cart_segment(pp_b, pl_b2, q_pp, None, T["lower"]):
                plan.append((q, P_["close"], "lower"))
            g_rel = P_["open"] + COMMON["release_extra"]
            for gg in ramp(P_["close"], g_rel, T["open"]):
                plan.append((plan[-1][0], gg, "open"))
            r_b = pl_b2 + np.array([0, 0, COMMON["retreat"]])
            for q in cart_segment(pl_b2, r_b, plan[-1][0], None, T["retreat"]):
                plan.append((q, g_rel, "retreat"))
            return plan

        # ---------------- 수집 ----------------
        tot = collections.Counter()
        saved, gif_ok, gif_bad = 0, 0, 0
        t_all = time.time()
        sched = []
        for c in conds:
            n_c = int(round(args.attempts * weights.get(c["name"], 1.0)))
            sched += [(c, k) for k in range(n_c)]
        # 조건을 섞어 돌림 (중간에 멈춰도 조건이 고르게)
        mrng = np.random.default_rng([args.seed, args.round, 99])
        sched = [sched[i] for i in mrng.permutation(len(sched))]
        for ri, (c, k) in enumerate(sched):
            if args.max_saved and saved >= args.max_saved:
                break
            rng = np.random.default_rng([args.seed, args.round, int(hashlib.md5(c["name"].encode()).hexdigest()[:8], 16), k])
            fi = k % 4
            fruit = FRUITS[fi]
            P = FRUIT_P[fruit]
            set_tcp(fruit)
            env.fixed_fruit = fi
            obs, _ = env.reset()
            info = setup(c, fi, rng)
            hold = obs["policy"]["joint_pos"].clone()
            for _ in range(3):
                obs, *_ = env.step(hold)
            policy.reset()
            t0 = time.time()
            mode = "early" if rng.random() < args.early_frac else "eval"
            control = mode == "eval" and rng.random() < args.control_frac
            early_t = int(rng.integers(20, 111))
            tgt = [n for n in copies(fruit) if pos_w(n)[2] - org[2] > 0.5]
            z0 = {n: pos_w(n)[2] for n in tgt}
            xy0 = {n: pos_w(n)[:2].copy() for n in tgt}
            Tb0 = base_T()
            gif_pol = collections.deque(maxlen=12)  # GIF 용 넘겨받기 전 정책 프레임 (4스텝마다)
            closes, prev_closed = [], False
            lifted_max = 0.0
            lift_copy = {n: 0.0 for n in tgt}  # 사본별 최고 들림 (떨어뜨린 사본 찾기)
            miss_run, still_run = 0, 0
            milestone, ms_t = 0, 0
            first = None  # (조건, 스텝)
            fire_at, fire_reason = None, None
            t2_seen, t3_seen = False, False
            q_prev = q_now()[:5]
            ok_policy = False
            limit = args.control_steps if control else args.max_sec * FPS
            t = 0
            for t in range(limit):
                pol = obs["policy"]
                st = rad_to_motor(pol["joint_pos"][0].cpu().numpy()).astype(np.float32)
                if args.gif_dir and t % 4 == 0:
                    gif_pol.append((pol["front"][0].cpu().numpy(), pol["wrist"][0].cpu().numpy(), False))
                a = policy.act({"front": pol["front"][0].cpu().numpy(), "wrist": pol["wrist"][0].cpu().numpy(), "state": st}, task_text(fruit))
                obs, *_ = env.step(torch.tensor(motor_to_rad(a), dtype=torch.float32, device=dev).unsqueeze(0))
                if bool(active_fruit_on_plate(env, FRUITS)[0]):
                    ok_policy = True
                    break
                tc = tcp_w()
                gp = {n: fruit_grasp_world(fruit, n)[0] for n in tgt}
                lifts = {n: pos_w(n)[2] - z0[n] for n in tgt}
                lift_now = max(lifts.values(), default=0.0)
                lifted_max = max(lifted_max, lift_now)
                for n in tgt:
                    lift_copy[n] = max(lift_copy[n], lifts[n])
                nn = min(tgt, key=lambda n: math.hypot(tc[0] - gp[n][0], tc[1] - gp[n][1]))
                dxy = math.hypot(tc[0] - gp[nn][0], tc[1] - gp[nn][1])
                closed = float(a[5]) < CONTACT_CMD[fruit] + 2.0
                if closed and not prev_closed:
                    closes.append({"t": t, "dxy_mm": round(dxy * 1000, 1), "dz_mm": round((tc[2] - gp[nn][2]) * 1000, 1), "z": float(tc[2])})
                prev_closed = closed
                q_cur = q_now()[:5]
                still_run = still_run + 1 if float(np.degrees(np.abs(q_cur - q_prev)).max()) < 0.2 else 0
                q_prev = q_cur
                pl = pos_w(PLATE)
                near_plate = min(math.hypot(pos_w(n)[0] - pl[0], pos_w(n)[1] - pl[1]) for n in tgt) < 0.12
                m = 1 if dxy < 0.05 else 0
                m = 2 if (closes and m) else m
                m = 3 if lifted_max > 0.03 else m
                m = 4 if (lifted_max > 0.03 and near_plate) else m
                if m > milestone:  # 가장 멀리 간 이정표가 바뀔 때만 진전으로 봄
                    milestone, ms_t = m, t
                reason = None
                if not closes:
                    if tc[2] < gp[nn][2] + 0.025 and dxy > args.miss_mm / 1000:
                        miss_run += 1
                    else:
                        miss_run = 0
                    if miss_run >= 5:
                        reason = "T1_miss"
                    elif max(float(np.linalg.norm(pos_w(n)[:2] - xy0[n])) for n in tgt) > args.push_mm / 1000:
                        reason = "T1_push"
                if reason is None and closes and closed and tc[2] - closes[-1]["z"] > args.rise_cm / 100 and lift_now < 0.015:
                    if not t2_seen:
                        t2_seen = True
                        if first is None:
                            first = ("T2_grasp", t)
                        if fire_at is None:
                            fire_at, fire_reason = t + int(rng.integers(0, args.t2_delay + 1)), "T2_grasp"
                if reason is None and lifted_max > 0.03 and not near_plate:
                    hn = max(tgt, key=lambda n: lift_copy[n])
                    if np.linalg.norm(fruit_grasp_world(fruit, hn)[0] - tc) > 0.06 and not t3_seen and speed(hn) < 0.02:
                        t3_seen = True
                        if first is None:
                            first = ("T3_drop", t)
                        if fire_at is None:
                            fire_at, fire_reason = t + int(rng.integers(0, args.t3_delay + 1)), "T3_drop"
                if reason is None and t > 30 and still_run >= args.stall_steps:
                    reason = "T4_stall"
                if reason is None and t - ms_t >= args.milestone_steps:
                    reason = "T4_noprog"
                if reason is None and t >= args.t5_steps and lifted_max < 0.03:
                    reason = "T5_late"
                if reason and first is None:
                    first = (reason, t)
                if mode == "early" and t >= early_t and not closes and fire_at is None:
                    fire_at, fire_reason = t, "early"
                if reason and fire_at is None:
                    fire_at, fire_reason = t, reason
                if not control and fire_at is not None and t >= fire_at:
                    if all(speed(n) < 0.02 for n in tgt) or t - fire_at > 30:
                        break
            rec = {"round": args.round, "driver": driver, "cond": c["name"], "k": k, "fruit": fruit, "mode": mode, "control": control,
                   "first_trigger": first, "closes": closes[:4], "lifted_max_cm": round(lifted_max * 100, 1), **info}
            if ok_policy or control or fire_at is None:
                res = "policy_ok" if ok_policy else ("control_fail" if control else "policy_timeout")
                tot[res] += 1
                rec.update({"result": res, "steps": t + 1, "wall_s": round(time.time() - t0, 1)})
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[{ri + 1}/{len(sched)} {c['name']}/{mode}{'/대조' if control else ''}] {FRUIT_KO[fruit]} {res} ({t + 1}스텝)"
                      f" 처음 조건 {first}", flush=True)
                continue
            t_take = t
            # ---- 넘겨받기
            frames_gif = list(gif_pol)
            n_exec, tries, expert_fail, name, ambiguous, kind = 0, 0, None, None, False, None
            bias = np.zeros(5)

            def run(items):
                nonlocal obs, bias, n_exec
                for q_des, gmot, ph in items:
                    if t_take + n_exec >= args.max_sec * FPS:
                        return "시간 초과(30초)"
                    qa = q_now()
                    if ph not in ("rise_open", "to_hover", "to_plate"):
                        bias[:4] += COMMON["ki"] * (q_des[:4] - qa[:4])
                        bias = np.clip(bias, -COMMON["bias_max"], COMMON["bias_max"])
                    tau_g = robot.data.gravity_compensation_forces.torch[0].cpu().numpy().astype(np.float64)
                    q_cmd = np.clip(q_des + tau_g[:5] / ARM_K + bias, kin.lo, kin.hi)
                    a_mot = rad_to_motor(np.r_[q_cmd, 0.0])
                    a_mot[5] = gmot
                    pol_ = obs["policy"]
                    if ds is not None:
                        ds.add_frame({"observation.state": rad_to_motor(pol_["joint_pos"][0].cpu().numpy()).astype(np.float32),
                                      "action": a_mot.astype(np.float32), "observation.images.front": pol_["front"][0].cpu().numpy(),
                                      "observation.images.wrist": pol_["wrist"][0].cpu().numpy(), "task": task_text(fruit)})
                    if args.gif_dir and n_exec % 4 == 0:
                        frames_gif.append((pol_["front"][0].cpu().numpy(), pol_["wrist"][0].cpu().numpy(), True))
                    obs, *_ = env.step(torch.tensor(motor_to_rad(a_mot), dtype=torch.float32, device=dev).unsqueeze(0))
                    n_exec += 1
                return None

            def held_copy():
                tc_ = tcp_w()
                hs = [n for n in tgt if pos_w(n)[2] - z0[n] > 0.015 and np.linalg.norm(fruit_grasp_world(fruit, n)[0] - tc_) < 0.05]
                return hs[0] if hs else None

            def choose_target():
                tc_ = tcp_w()
                plw = pos_w(PLATE)
                cand = [n for n in tgt if abs(pos_w(n)[2] - z0[n]) < 0.03 and speed(n) < 0.02
                        and math.hypot(pos_w(n)[0] - plw[0], pos_w(n)[1] - plw[1]) > 0.10
                        and not (fruit in ("Strawberry", "Lemon") and tilt_deg(n) > args.tilt_max)]
                if not cand:
                    return None, False
                d_t = {n: math.hypot(*(fruit_grasp_world(fruit, n)[0][:2] - tc_[:2])) for n in cand}
                srt = sorted(cand, key=lambda n: d_t[n])
                g0z = fruit_grasp_world(fruit, srt[0])[0][2]
                if d_t[srt[0]] < 0.06 and tc_[2] < g0z + COMMON["hover"] + 0.03 and (len(srt) == 1 or d_t[srt[1]] - d_t[srt[0]] >= 0.03):
                    return srt[0], False
                Tb = base_T()
                d_r = {n: float(np.hypot(*to_base(Tb, pos_w(n))[:2])) for n in cand}
                sr = sorted(cand, key=lambda n: d_r[n])
                return sr[0], (len(sr) > 1 and d_r[sr[1]] - d_r[sr[0]] < 0.025)

            hname = held_copy()
            while True:
                if hname is None:
                    name, ambiguous = choose_target()
                    if name is None:
                        expert_fail = "대상 없음(쓰러짐·접시·움직임)"
                        break
                    tries += 1
                    plan, kind = plan_grasp(fruit, name)
                    if plan is None:
                        expert_fail = f"계획 실패({kind})"
                        break
                    bias = np.zeros(5)
                    expert_fail = run(plan)
                    if expert_fail:
                        break
                    hname = held_copy()
                    if hname is None:
                        if tries < args.expert_tries:
                            if ds is not None:
                                ds.clear_episode_buffer()  # 실패한 시도는 저장하지 않음 → 다시 계획한 지점부터 기록
                            continue
                        expert_fail = "전문가 놓침"
                        break
                name = hname
                p2 = plan_place(fruit, name)
                if p2 is None:
                    expert_fail = "놓기 IK 실패"
                    break
                bias = np.zeros(5)
                expert_fail = run(p2)
                if not expert_fail and not bool(active_fruit_on_plate(env, FRUITS)[0]):
                    expert_fail = "접시 밖"
                break
            ok = expert_fail is None
            ep_idx, n_frames = None, 0
            if ds is not None:
                n_frames = ds.writer.episode_buffer["size"] if ds.writer.episode_buffer else 0
                if ok:
                    ds.save_episode(parallel_encoding=False)
                    ep_idx = ds.num_episodes - 1
                else:
                    ds.clear_episode_buffer()
            if ok:
                saved += 1
            if args.gif_dir and ((ok and gif_ok < args.gif_n) or (not ok and gif_bad < args.gif_n)):
                save_gif(frames_gif, os.path.join(args.gif_dir, f"r{args.round}_{ri:03d}_{c['name']}_{fruit.lower()}_{fire_reason}_{'ok' if ok else 'fail'}.gif"))
                gif_ok, gif_bad = gif_ok + ok, gif_bad + (not ok)
            res = "saved" if ok else "expert_fail"
            tot[f"{fire_reason}:{res}"] += 1
            rec.update({"result": res, "reason": fire_reason, "t_take": t_take, "delay": t_take - (first[1] if first else t_take),
                        "held_at_take": hname is not None and tries == 0, "target": name, "ambiguous": ambiguous, "plan_kind": kind,
                        "expert_tries": tries, "expert_fail": expert_fail, "frames": n_frames, "episode_index": ep_idx,
                        "wall_s": round(time.time() - t0, 1)})
            with open(log_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{ri + 1}/{len(sched)} {c['name']}/{mode}] {FRUIT_KO[fruit]} 처음 {first} → 넘겨받기 {fire_reason} @{t_take}"
                  f" → {'저장' if ok else '실패: ' + str(expert_fail)} · {n_frames}프레임 · 시도 {tries} · 누적 저장 {saved}", flush=True)
        if ds is not None:
            ds.finalize()  # 안 부르면 parquet footer 가 안 써져 데이터셋이 깨짐 (gen_scripted.py 와 같음)
        print("SUMMARY", json.dumps({"saved": saved, **tot}, ensure_ascii=False), f"총 {time.time() - t_all:.0f}s", flush=True)
        os._exit(0)


if __name__ == "__main__":
    main()
