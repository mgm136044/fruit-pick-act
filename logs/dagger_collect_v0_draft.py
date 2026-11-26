"""개입형 DAgger 수집기 (91점 대응 지시 P1): 정책이 로봇을 몰고, 실패 조짐이 보이면 스크립트 전문가가 이어받아 끝까지 한다.
넘겨받은 에피소드 중 전문가가 성공한 것만, 넘겨받기 직전 구간부터 저장한다(LeRobot 형식, gen_scripted/record.py 와 같은 features).

  source ~/robot_course/activate.sh
  COURSE_ASSETS=~/fruit_datagen/assets_c4 python dagger_collect.py --checkpoint te_models/act_mix2_100k_color_te \
      --dataset ~/datasets/dagger_r1_part1 --conds "..." --attempts 40 --round 1 --viz none

정책 실행은 키트 lerobot_policy(load_policy + act)와 같다(체크포인트 config 그대로, 제출 zip 폴더를 넣음).
장면 조건은 eval_sweep.py 와 같은 형식(--conds). 전문가는 gen_scripted.py 의 잡기 값(--params, 기본 params_mix3.json)과
IK·구간 계획·중력 보정·적분 보정을 그대로 쓰되, '지금 상태'에서 다시 계획한다:
  과일을 쥐고 있으면  → 접시로 옮겨 놓기 → 휴식 자세
  아니면              → 그리퍼 열며 위로 → 목표 위 → 내려가기 → 쥐기 → 들기 → (위와 같이 놓기)
목표: 같은 종류 사본 중 조리대 위에 있고 손끝에서 가장 가까운 것 (정책이 가던 것).
넘겨받기 조건 (정책 구간, 측정값):
  approach_miss  쥐기 시작 때 손끝이 가장 가까운 목표에서 수평 > --miss-cm, 또는 잡는 점보다 > 4cm 높음 → 5프레임 뒤
  grasp_fail     쥐기 시작 뒤 손끝이 쥔 높이보다 > --rise-cm 올라갔는데 목표가 < 1.5cm 들림 (빈손으로 들기)
  drop           목표를 3cm 넘게 들었다가 1cm 아래로 떨어짐(접시 12cm 밖) 또는 쥔 채 손끝에서 6cm 넘게 멀어짐
  stall          3초 동안 손끝 이동 합 < 1.5cm (시작 1초 뒤부터)
  slow           --soft-sec 초 안에 성공 못 함
저장 범위: 넘겨받기 --pre-frames 전(기본 15 = 0.5초)부터 휴식 자세까지. 앞 구간 action 은 전문가의 첫 명령(멈추고 회복 시작)으로 둔다.
정책이 혼자 성공한 에피소드는 저장하지 않는다(기록만).
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

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--no-save", action="store_true")
ap.add_argument("--conds", default="cnt23:count=2-3;cnt4:count=4;rob10:rob=10/30;xy10:xy=0.10,yaw=180;base:xy=0.03")
ap.add_argument("--attempts", type=int, default=40, help="조건마다 시도 수 (과일 4종 돌아가며)")
ap.add_argument("--max-saved", type=int, default=0, help="0 이면 제한 없음")
ap.add_argument("--round", type=int, default=1)
ap.add_argument("--seed", type=int, default=100)
ap.add_argument("--params", default=os.path.join(HERE, "params_mix3.json"))
ap.add_argument("--max-sec", type=int, default=30, help="정책 + 전문가 합계 상한")
ap.add_argument("--soft-sec", type=float, default=15.0)
ap.add_argument("--miss-cm", type=float, default=3.0)
ap.add_argument("--rise-cm", type=float, default=3.0)
ap.add_argument("--pre-frames", type=int, default=15)
ap.add_argument("--log", default=None)
ap.add_argument("--gif-dir", default=None, help="저장한 에피소드 처음 --gif-n 개를 GIF(정면|손목)로")
ap.add_argument("--gif-n", type=int, default=12)
add_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True

FPS = 30
AREA_X, AREA_Y = (1.95, 2.30), (-0.48, -0.26)
PLATE_CLEAR, ROBOT_CLEAR, GAP, GAP_BANANA = 0.14, 0.17, 0.075, 0.11
CONTACT_CMD = {"Strawberry": 26.6, "Orange": 36.9, "Lemon": 29.6, "Banana": 18.2}
# ---- gen_scripted.py 와 같은 과일·공통 값 (v3) + --params 덮어쓰기
FRUIT_P = {
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
    for fr, wr, tag in frames:
        im = Image.new("RGB", (w * 2 + 4, h), (250, 250, 248))
        im.paste(Image.fromarray(np.asarray(fr).astype(np.uint8)).resize((w, h)), (0, 0))
        im.paste(Image.fromarray(np.asarray(wr).astype(np.uint8)).resize((w, h)), (w + 4, 0))
        if tag:  # 전문가 구간 표시: 위쪽 빨간 띠
            im.paste((220, 40, 40), (0, 0, w * 2 + 4, 4))
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
        assert len(copies(FRUITS[0])) >= 4, "count=4 조건은 COURSE_ASSETS=~/fruit_datagen/assets_c4 로 실행"
    policy = load_policy(os.path.abspath(os.path.expanduser(args.checkpoint)), "cuda", ["act"])
    kin = SO101Kin()
    q_rest6 = np.array([rest_pose_rad()[j] for j in JOINTS])
    rest_motor = rad_to_motor(q_rest6)
    root = os.path.abspath(os.path.expanduser(args.dataset))
    log_path = args.log or os.path.join(HERE, "logs", f"dagger_{os.path.basename(root.rstrip(os.sep))}.jsonl")
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
            kin.tcp = np.array([COMMON["fixed_inner_x"] + P.get("fixed_clear", COMMON["fixed_clear"]) + P["width"] / 2, 0.0, COMMON["tcp_depth"]])

        def tcp_w():
            return base_T()[:3, :3] @ kin.tcp_frame(q_now()[:5])[0] + base_T()[:3, 3]

        def fruit_grasp_world(fruit, name):
            T = obj_T(name)
            P = FRUIT_P[fruit]
            c = T[:3, :3] @ np.array(P["grasp_local"]) + T[:3, 3]
            close_w = None
            if P["long_axis"] is not None:
                la = T[:3, :3] @ np.array(P["long_axis"], dtype=np.float64)
                la[2] = 0
                if np.linalg.norm(la) > 0.3:
                    la /= np.linalg.norm(la)
                    close_w = np.array([-la[1], la[0], 0.0])
            return c, close_w

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
            for s in min_jerk(n):
                q, _ = kin.ik_pos_first(p0 + (p1 - p0) * s, q, c_goal=c_b)
                qs.append(q)
            return qs

        def joint_segment(q0, q1, n):
            return [q0 + (q1 - q0) * s for s in min_jerk(n)]

        def ramp(g0, g1, n):
            return [g0 + (g1 - g0) * s for s in min_jerk(n)]

        T = COMMON["T"]

        def plan_grasp(fruit, name):
            """지금 자세에서: 그리퍼 열며 위로 → 목표 위 → 내려가기 → 쥐기 → 들기. 실패하면 None."""
            P = FRUIT_P[fruit]
            Tb = base_T()
            q0 = q_now()
            g0 = float(rad_to_motor(q0)[5])
            c_w, close_w = fruit_grasp_world(fruit, name)
            table_z = c_w[2] - P["grasp_local"][2]
            g_w = c_w.copy()
            g_w[2] = max(c_w[2], table_z + COMMON["tip_below_tcp"] + COMMON["tip_clear"])
            h_w, l_w = g_w + [0, 0, COMMON["hover"]], g_w + [0, 0, COMMON["lift"]]
            c_b = dir_to_base(Tb, close_w)
            g_b, h_b, l_b = to_base(Tb, g_w), to_base(Tb, h_w), to_base(Tb, l_w)
            q_g, _, okg = ik_ok(g_b, seed_for(g_b), c_b)
            q_h, _, okh = ik_ok(h_b, q_g, c_b)
            if not (okg and okh):
                return None
            plan = []
            t_b = kin.tcp_frame(q0[:5])[0]
            up_b = t_b.copy()
            up_b[2] = max(t_b[2], h_b[2])
            n_up = 15 if up_b[2] - t_b[2] > 0.01 else 6
            q_up, _ = kin.ik_pos_first(up_b, q0[:5], c_goal=None)  # 관절 공간으로 이어 첫 프레임 방향이 튀지 않게
            for q, gg in zip(joint_segment(q0[:5], q_up, n_up), ramp(g0, P["open"], n_up)):
                plan.append((q, gg, "rise_open"))
            dist = float(np.linalg.norm(h_b - up_b))
            n_mv = int(np.clip(dist / 0.004, 25, T["to_hover"]))
            for q in joint_segment(plan[-1][0], q_h, n_mv):
                plan.append((q, P["open"], "to_hover"))
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
            return plan

        def plan_place(fruit, name):
            """쥐고 있는 상태에서: 접시 위 → 내려놓기 → 놓기 → 위로 → 휴식 자세 (실제 쥔 위치 보고)."""
            P = FRUIT_P[fruit]
            Tb = base_T()
            q0 = q_now()
            fc_w, _ = fruit_grasp_world(fruit, name)
            d_off = fc_w - tcp_w()
            plate_w = obj_T(PLATE)[:3, 3]
            pxy = plate_w[:2]
            ga = P["grasp_local"][2]
            cz_place = plate_w[2] + COMMON["plate_inner"] + COMMON["drop"] + ga
            cz_pre = plate_w[2] + COMMON["plate_rim"] + COMMON["rim_clear"] + ga
            pp_b, pl_b2 = to_base(Tb, np.r_[pxy, cz_pre] - d_off), to_base(Tb, np.r_[pxy, cz_place] - d_off)
            q_pp, _, okp = ik_ok(pp_b, seed_for(pp_b), None, tilt_max=80)
            if not okp:
                return None
            plan = []
            t_b = kin.tcp_frame(q0[:5])[0]
            if t_b[2] < pp_b[2] - 0.02:  # 낮게 쥐고 있으면 먼저 들기 (관절 공간)
                up = t_b.copy()
                up[2] = pp_b[2]
                q_up, _ = kin.ik_pos_first(up, q0[:5], c_goal=None)
                for q in joint_segment(q0[:5], q_up, 15):
                    plan.append((q, P["close"], "lift"))
            start = plan[-1][0] if plan else q0[:5]
            for q in joint_segment(start, q_pp, T["to_plate"]):
                plan.append((q, P["close"], "to_plate"))
            for q in cart_segment(pp_b, pl_b2, q_pp, None, T["lower"]):
                plan.append((q, P["close"], "lower"))
            g_rel = P["open"] + COMMON["release_extra"]
            for gg in ramp(P["close"], g_rel, T["open"]):
                plan.append((plan[-1][0], gg, "open"))
            r_b = pl_b2 + np.array([0, 0, COMMON["retreat"]])
            for q in cart_segment(pl_b2, r_b, plan[-1][0], None, T["retreat"]):
                plan.append((q, g_rel, "retreat"))
            for q, gg in zip(joint_segment(plan[-1][0], q_rest6[:5], T["to_rest"]), ramp(g_rel, float(rest_motor[5]), T["to_rest"])):
                plan.append((q, gg, "to_rest"))
            return plan

        def first_cmd(plan):
            q_des, gmot, _ = plan[0]
            tau_g = robot.data.gravity_compensation_forces.torch[0].cpu().numpy().astype(np.float64)
            a_mot = rad_to_motor(np.r_[np.clip(q_des + tau_g[:5] / ARM_K, kin.lo, kin.hi), 0.0])
            a_mot[5] = gmot
            return a_mot

        # ---------------- 수집 ----------------
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        tot = collections.Counter()
        saved = 0
        gif_done = 0
        t_all = time.time()
        for c in conds:
            rng = np.random.default_rng([args.seed, args.round, int(hashlib.md5(c["name"].encode()).hexdigest()[:8], 16)])
            for ep in range(args.attempts):
                if args.max_saved and saved >= args.max_saved:
                    break
                fi = ep % 4
                fruit = FRUITS[fi]
                set_tcp(fruit)
                env.fixed_fruit = fi
                obs, _ = env.reset()
                info = setup(c, fi, rng)
                hold = obs["policy"]["joint_pos"].clone()
                for _ in range(3):
                    obs, *_ = env.step(hold)
                policy.reset()
                t0 = time.time()

                def pos(n):
                    return env.scene[n].data.root_pos_w.torch[0].cpu().numpy().astype(np.float64) - org
                tgt = [n for n in copies(fruit) if pos(n)[2] > 0.5]
                z0 = {n: pos(n)[2] for n in tgt}
                P = FRUIT_P[fruit]
                buf = collections.deque(maxlen=args.pre_frames)
                path = collections.deque(maxlen=90)
                closes = []
                prev_closed = False
                lifted_max = {n: 0.0 for n in tgt}
                reason, t_take, ok_policy = None, None, False
                max_steps = args.max_sec * FPS
                miss_at = None
                for t in range(int(args.soft_sec * FPS)):
                    pol = obs["policy"]
                    st = rad_to_motor(pol["joint_pos"][0].cpu().numpy()).astype(np.float32)
                    buf.append((pol["front"][0].cpu().numpy(), pol["wrist"][0].cpu().numpy(), st))
                    a = policy.act({"front": pol["front"][0].cpu().numpy(), "wrist": pol["wrist"][0].cpu().numpy(), "state": st}, task_text(fruit))
                    obs, *_ = env.step(torch.tensor(motor_to_rad(a), dtype=torch.float32, device=dev).unsqueeze(0))
                    if bool(active_fruit_on_plate(env, FRUITS)[0]):
                        ok_policy = True
                        break
                    tc = tcp_w() - org
                    path.append(tc)
                    lifts = {n: pos(n)[2] - z0[n] for n in tgt}
                    for n in tgt:
                        lifted_max[n] = max(lifted_max[n], lifts[n])
                    closed = float(a[5]) < CONTACT_CMD[fruit] + 2.0
                    if closed and not prev_closed:
                        gpts = {n: fruit_grasp_world(fruit, n)[0] - org for n in tgt}
                        nn = min(tgt, key=lambda n: math.hypot(tc[0] - gpts[n][0], tc[1] - gpts[n][1]))
                        dxy = math.hypot(tc[0] - gpts[nn][0], tc[1] - gpts[nn][1])
                        dz = tc[2] - gpts[nn][2]
                        closes.append({"t": t, "dxy_cm": round(dxy * 100, 1), "dz_cm": round(dz * 100, 1), "z": float(tc[2])})
                        if dxy > args.miss_cm / 100 or dz > 0.04:
                            miss_at = t + 5
                    prev_closed = closed
                    if miss_at is not None and t >= miss_at and max(lifts.values(), default=0) < 0.015:
                        reason = "approach_miss"
                    elif closes and closed and tc[2] - closes[-1]["z"] > args.rise_cm / 100 and max(lifts.values(), default=0) < 0.015:
                        reason = "grasp_fail"
                    elif max(lifted_max.values(), default=0) > 0.03 and max(lifts.values(), default=0) < 0.01:
                        pl = pos(PLATE)
                        if min(math.hypot(pos(n)[0] - pl[0], pos(n)[1] - pl[1]) for n in tgt) > 0.12:
                            reason = "drop"
                    elif closed and max(lifts.values(), default=0) > 0.03:
                        hn = max(tgt, key=lambda n: lifts[n])
                        if np.linalg.norm(fruit_grasp_world(fruit, hn)[0] - org - tc) > 0.06:
                            reason = "drop"
                    if reason is None and t > 30 and len(path) == path.maxlen:
                        pa = np.array(path)
                        if float(np.sum(np.linalg.norm(np.diff(pa, axis=0), axis=1))) < 0.015:
                            reason = "stall"
                    if reason:
                        t_take = t
                        break
                if ok_policy:
                    tot["policy_ok"] += 1
                    rec = {"cond": c["name"], "fruit": fruit, "ep": ep, "result": "policy_ok", "steps": t + 1, "closes": closes[:4], **info}
                    with open(log_path, "a") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(f"[{c['name']}] {FRUIT_KO[fruit]} 정책 혼자 성공 ({t + 1}스텝) — 저장 안 함", flush=True)
                    continue
                if reason is None:
                    reason, t_take = "slow", t
                # ---- 넘겨받기: 목표 = 조리대 위 사본 중 손끝에서 가장 가까운 것
                tc = tcp_w() - org
                lifts = {n: pos(n)[2] - z0[n] for n in tgt}
                held = [n for n in tgt if lifts[n] > 0.02 and np.linalg.norm(fruit_grasp_world(fruit, n)[0] - org - tc) < 0.05]
                plw = pos(PLATE)
                on_table = [n for n in tgt if abs(pos(n)[2] - z0[n]) < 0.04 and math.hypot(pos(n)[0] - plw[0], pos(n)[1] - plw[1]) > 0.10]
                expert_fail = None
                plans = []
                if held:
                    name = held[0]
                    p2 = plan_place(fruit, name)
                    plans = [p2] if p2 else []
                elif on_table:
                    name = min(on_table, key=lambda n: math.hypot(pos(n)[0] - tc[0], pos(n)[1] - tc[1]))
                    p1 = plan_grasp(fruit, name)
                    plans = [p1] if p1 else []
                else:
                    name = None
                if not plans:
                    expert_fail = "계획 실패(IK·대상 없음)"
                frames_gif = []
                n_rec = 0
                if not expert_fail:
                    a0 = first_cmd(plans[0]).astype(np.float32)
                    if ds is not None:
                        for fr, wr, st in buf:  # 넘겨받기 직전 구간: action = 전문가 첫 명령
                            ds.add_frame({"observation.state": st, "action": a0, "observation.images.front": fr,
                                          "observation.images.wrist": wr, "task": task_text(fruit)})
                            n_rec += 1
                    if args.gif_dir and gif_done < args.gif_n:
                        frames_gif += [(fr, wr, False) for fr, wr, _ in list(buf)[::4]]
                    bias = np.zeros(5)
                    n_exec = 0

                    def run(items):
                        nonlocal obs, bias, n_exec, n_rec
                        for q_des, gmot, ph in items:
                            if t_take + n_exec >= max_steps:
                                return "시간 초과"
                            qa = q_now()
                            if ph not in ("rise_open", "to_hover", "to_plate", "to_rest"):
                                bias[:4] += COMMON["ki"] * (q_des[:4] - qa[:4])
                                bias = np.clip(bias, -COMMON["bias_max"], COMMON["bias_max"])
                            tau_g = robot.data.gravity_compensation_forces.torch[0].cpu().numpy().astype(np.float64)
                            q_cmd = np.clip(q_des + tau_g[:5] / ARM_K + bias, kin.lo, kin.hi)
                            a_mot = rad_to_motor(np.r_[q_cmd, 0.0])
                            a_mot[5] = gmot
                            pol = obs["policy"]
                            if ds is not None:
                                ds.add_frame({"observation.state": rad_to_motor(pol["joint_pos"][0].cpu().numpy()).astype(np.float32),
                                              "action": a_mot.astype(np.float32), "observation.images.front": pol["front"][0].cpu().numpy(),
                                              "observation.images.wrist": pol["wrist"][0].cpu().numpy(), "task": task_text(fruit)})
                                n_rec += 1
                            if args.gif_dir and gif_done < args.gif_n and n_exec % 4 == 0:
                                frames_gif.append((pol["front"][0].cpu().numpy(), pol["wrist"][0].cpu().numpy(), True))
                            obs, *_ = env.step(torch.tensor(motor_to_rad(a_mot), dtype=torch.float32, device=dev).unsqueeze(0))
                            n_exec += 1
                        return None

                    expert_fail = run(plans[0])
                    if not held:
                        if not expert_fail:
                            fc, _ = fruit_grasp_world(fruit, name)
                            if pos(name)[2] - z0[name] < 0.03 or np.linalg.norm(fc - tcp_w()) > 0.05:
                                expert_fail = "전문가 놓침"
                            else:
                                p2 = plan_place(fruit, name)
                                expert_fail = run(p2) if p2 else "놓기 계획 실패"
                    if not expert_fail:
                        for _ in range(15):  # 휴식 자세에 닿을 때까지 같은 목표 유지
                            if np.max(np.abs(q_now()[:5] - q_rest6[:5])) < 0.08:
                                break
                            expert_fail = run([(q_rest6[:5], float(rest_motor[5]), "to_rest")])
                            if expert_fail:
                                break
                    if not expert_fail and not bool(active_fruit_on_plate(env, FRUITS)[0]):
                        expert_fail = "접시 밖"
                ok = expert_fail is None
                ep_idx = None
                if ds is not None:
                    if ok:
                        ds.save_episode(parallel_encoding=False)
                        ep_idx = ds.num_episodes - 1
                    else:
                        ds.clear_episode_buffer()
                if ok:
                    saved += 1
                    if frames_gif:
                        save_gif(frames_gif, os.path.join(args.gif_dir, f"r{args.round}_{c['name']}_{ep:02d}_{fruit.lower()}_{reason}.gif"))
                        gif_done += 1
                tot[f"take_{reason}_{'ok' if ok else 'fail'}"] += 1
                rec = {"cond": c["name"], "fruit": fruit, "ep": ep, "result": "saved" if ok else "expert_fail", "reason": reason,
                       "t_take": t_take, "held_at_take": bool(held), "target": name, "expert_fail": expert_fail,
                       "frames": n_rec, "episode_index": ep_idx, "round": args.round, "closes": closes[:4],
                       "lifted_max_cm": round(max(lifted_max.values(), default=0) * 100, 1), "wall_s": round(time.time() - t0, 1), **info}
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[{c['name']}] {FRUIT_KO[fruit]} 넘겨받기 {reason} @{t_take}스텝 → {'저장' if ok else '실패: ' + str(expert_fail)}"
                      f" · {n_rec}프레임 · 누적 저장 {saved}", flush=True)
        print("SUMMARY", json.dumps({"saved": saved, **tot}, ensure_ascii=False), f"총 {time.time() - t_all:.0f}s", flush=True)
        os._exit(0)


if __name__ == "__main__":
    main()
