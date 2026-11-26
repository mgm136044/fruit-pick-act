"""3단계-2/3: 오렌지 사람 시연(HF leisaac-pick-orange) 의 집기 구간을 수업 환경에서 재생해 다른 과일 시연을 만든다.

  python orange_replay.py --fruit orange --max-segments 40 --viz none                  # 오렌지로 위치 복원 검증
  python orange_replay.py --fruit lemon --max-segments 40 --dataset ~/datasets/fruit_from_orange_lemon --viz none

- 구간: orange_segments.json (orange_recover.py) 의 쥐기 하나 = (앞 구간이 끝난 뒤 ~ 놓고 조금 뒤)
  첫 구간은 원래 휴식 자세에서 시작, 2·3번째 구간은 휴식 자세 → 구간 시작 자세 보간(30프레임)을 앞에 붙임
- 장면: 로봇 베이스 기준 복원 위치에 과일, 놓은 위치에 접시 (로봇은 기본 위치)
- 다른 과일: 쥐는 구간 그리퍼 명령을 (오렌지 접촉 각도 - 과일 접촉 각도) 만큼 더 닫고,
  쥐는 높이는 과일 중심 높이 차이만큼 낮춤 (손가락 끝이 조리대에 닿지 않게 하한). 바나나는 긴 축을 손가락 닫는 방향에 수직으로 놓음
- 행동은 원래 LeRobot 단위 관절 목표를 그대로(높이 보정 구간만 IK 로 고쳐서) 보냄. 성공(접시 위)한 것만 저장
※ 사람 시연을 변형한 것이고 수업 환경 녹화가 아님. 로컬 실험용.
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

ap = argparse.ArgumentParser()
ap.add_argument("--fruit", required=True, choices=["strawberry", "orange", "lemon", "banana"])
ap.add_argument("--max-segments", type=int, default=40)
ap.add_argument("--stride", type=int, default=1, help="원본 행동을 몇 프레임마다 하나씩 쓸지 (1 = 그대로, 2 = 2배속)")
ap.add_argument("--dataset", default=None, help="성공한 것을 저장할 LeRobot 데이터셋 (없으면 저장 안 함)")
ap.add_argument("--no-height-shift", action="store_true")
ap.add_argument("--no-grip-shift", action="store_true")
ap.add_argument("--log", default=None)
ap.add_argument("--seed", type=int, default=0)
add_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True

FPS = 30
# gen_scripted.py 와 같은 과일 기하 (키트 메시 실측)
FRUIT_G = {
    "Strawberry": {"grasp_local": [0, 0, 0.0228], "width": 0.045, "long_axis": None},
    "Orange": {"grasp_local": [0, 0, 0.0268], "width": 0.055, "long_axis": None},
    "Lemon": {"grasp_local": [0, 0, 0.0225], "width": 0.051, "long_axis": None},
    "Banana": {"grasp_local": [-0.0303, -0.020, 0.0160], "width": 0.032, "long_axis": [0.06, 1, 0]},
}
NAME = {"strawberry": "Strawberry", "orange": "Orange", "lemon": "Lemon", "banana": "Banana"}
TABLE_Z_B = 0.028  # 조리대 높이 (로봇 베이스 기준, explore: 0.918 - 0.89)
TIP_MIN_Z_B = TABLE_Z_B + 0.026  # TCP 하한 (손가락 끝 6mm 여유)


def main():
    import numpy as np
    import pandas as pd
    import torch

    from course_env.mdp import active_fruit_on_plate, active_index, park_fruits
    from course_env.paths import FRUITS, PLATE
    from course_env.pick_fruit import SO101_CFG, PickFruitEnvCfg, task_text
    from course_env.so101 import FEATURE_NAMES, JOINTS, motor_to_rad, rad_to_motor, rest_pose_rad
    from kin import SO101Kin, quat_wxyz_to_mat
    from orange_recover import FIXED_INNER_X, TCP_DEPTH, gap_at

    fruit = NAME[args.fruit]
    G = FRUIT_G[fruit]
    kin = SO101Kin()
    rng = np.random.default_rng(args.seed)
    segs_all = json.load(open(os.path.join(HERE, "orange_segments.json")))
    q_rest6 = np.array([rest_pose_rad()[j] for j in JOINTS])
    rest_motor = rad_to_motor(q_rest6)
    log_path = args.log or os.path.join(HERE, "logs", f"replay_{args.fruit}.jsonl")

    # 그리퍼 각도(도) ↔ 벌림 폭: 접촉 각도 (폭 → 각도)
    qs = np.linspace(-10, 42, 300)
    gaps = np.array([gap_at(q) for q in qs])

    def contact_motor(w):
        qd = float(np.interp(w, gaps, qs))
        return (qd + 10) / 110 * 100

    # 재생할 구간 목록 (에피소드 순서대로, 구간마다 하나)
    items = []
    for r in segs_all:
        d = None
        for k, s in enumerate(r["segments"]):
            items.append((r["episode"], k, s, r["segments"][k + 1]["close_start"] if k + 1 < len(r["segments"]) else None))
    rng.shuffle(items)
    items = items[:args.max_segments]

    ds = None
    if args.dataset:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        root = os.path.abspath(os.path.expanduser(args.dataset))
        vec = {"dtype": "float32", "shape": (6,), "names": FEATURE_NAMES}
        img = {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
        feats = {"observation.state": vec, "action": vec, "observation.images.front": img, "observation.images.wrist": img}
        if os.path.exists(root):
            ds = LeRobotDataset.resume(f"local/{os.path.basename(root)}", root=root, image_writer_threads=4)
        else:
            ds = LeRobotDataset.create(f"local/{os.path.basename(root)}", fps=FPS, features=feats, root=root,
                                       robot_type="so101_follower", use_videos=True, image_writer_threads=4)

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
        zero6 = torch.zeros(1, 6, device=dev)
        fi = FRUITS.index(fruit)

        def put(name, x, y, z, yaw=0.0):
            pose = torch.zeros(1, 7, device=dev)
            pose[0, :3] = origin + torch.tensor([x, y, z], device=dev)
            pose[0, 3:] = torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)], device=dev)
            env.scene[name].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            env.scene[name].write_root_velocity_to_sim_index(root_velocity=zero6, env_ids=ids)

        def pose_T(p7):
            T = np.eye(4)
            x, y, z, w = p7[3:7]
            T[:3, :3] = quat_wxyz_to_mat([w, x, y, z])
            T[:3, 3] = p7[:3]
            return T

        def q_now():
            return robot.data.joint_pos.torch[0].cpu().numpy().astype(np.float64)

        totals = {"tried": 0, "success": 0, "fail": {}}
        t_all = time.time()
        for ep, k, s, next_close in items:
            t0 = time.time()
            d = pd.read_parquet(os.path.join(HERE, "hf_orange", "data", "chunk-000", f"episode_{ep:06d}.parquet"))
            ac = np.stack(d["action"].to_numpy()).astype(np.float64)
            st = np.stack(d["observation.state"].to_numpy()).astype(np.float64)
            prev_rel = 0
            r_ep = next(r for r in segs_all if r["episode"] == ep)
            if k > 0:
                prev_rel = r_ep["segments"][k - 1]["release"] + 15
            start = prev_rel
            end = (next_close - 10) if next_close is not None else len(ac)
            end = min(end, s["release"] + 60) if next_close is not None else end
            A = ac[start:end].copy()
            idx_close, idx_held, idx_rel = s["close_start"] - start, s["held"] - start, s["release"] - start

            # ---- 그리퍼 보정: 쥐는 구간 명령을 (오렌지 접촉 - 과일 접촉) 만큼 더 닫음
            w_or = s["width_est_mm"] / 1000
            dgrip = 0.0 if (args.no_grip_shift or fruit == "Orange") else contact_motor(w_or) - contact_motor(G["width"])
            A[idx_close:idx_rel, 5] = np.maximum(A[idx_close:idx_rel, 5] - dgrip, 0.0)

            # ---- 장면: 과일 중심 = 복원한 오렌지 중심 (수평), 접시 중심 = 놓은 TCP 위치
            Tb = pose_T(robot.data.root_link_pose_w.torch[0].cpu().numpy().astype(np.float64))
            env.reset()
            Tb = pose_T(robot.data.root_link_pose_w.torch[0].cpu().numpy().astype(np.float64))
            o_b = np.array(s["orange_b"])
            q_held = motor_to_rad(st[s["held"]])
            Tg = kin.fk_gripper(q_held[:5])
            close_b = Tg[:3, 0]  # 손가락 닫는 방향 (베이스 기준)
            yaw = 0.0
            if G["long_axis"] is not None:
                # 바나나: 긴 축(로컬 y) 을 닫는 방향에 수직 → 로컬 x 를 닫는 방향에 맞춤
                cw = Tb[:3, :3] @ close_b
                yaw = math.atan2(cw[1], cw[0])
            Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
            c_w = Tb[:3, :3] @ o_b + Tb[:3, 3]
            root_xy = c_w[:2] - (Rz @ np.array(G["grasp_local"]))[:2]
            z_spawn = getattr(env.cfg.scene, FRUITS[0]).init_state.pos[2]
            active_index(env)[0] = fi
            park_fruits(env, ids, list(FRUITS), torch.tensor([-1], device=dev))
            put(fruit, root_xy[0], root_xy[1], z_spawn, yaw)
            rel_w = Tb[:3, :3] @ np.array(s["release_b"]) + Tb[:3, 3]
            put(PLATE, rel_w[0], rel_w[1], 0.93)
            hold = torch.tensor(q_rest6, dtype=torch.float32, device=dev).unsqueeze(0)
            for _ in range(20):
                obs, *_ = env.step(hold)
            # 접시와 과일이 겹치면(놓은 위치가 쥔 위치와 가까우면) 시작부터 '접시 위'가 되므로 제외
            sep = float(np.hypot(*(rel_w[:2] - c_w[:2])))
            if sep < 0.12 or bool(active_fruit_on_plate(env, FRUITS)[0]):
                rec = {"fruit": fruit, "episode": ep, "segment": k, "success": None, "fail": "제외(접시와 겹침)",
                       "sep_m": round(sep, 3)}
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                totals["skipped"] = totals.get("skipped", 0) + 1
                print(f"[{fruit}] ep{ep}-{k} 제외: 접시와 과일 간격 {sep * 100:.0f}cm", flush=True)
                continue

            # ---- 높이 보정: 쥐기 전후 구간의 TCP 를 (과일 중심 높이 - 오렌지 중심 높이) 만큼 낮춤
            dz = 0.0
            if not args.no_height_shift:
                fc = pose_T(env.scene[fruit].data.root_link_pose_w.torch[0].cpu().numpy())
                c_now = fc[:3, :3] @ np.array(G["grasp_local"]) + fc[:3, 3]
                c_now_b = Tb[:3, :3].T @ (c_now - Tb[:3, 3])
                dz = c_now_b[2] - o_b[2]
            if abs(dz) > 0.002:
                kin.tcp = np.array([FIXED_INNER_X + w_or / 2, 0.0, TCP_DEPTH])
                n = len(A)
                wgt = np.zeros(n)
                a0, a1 = max(0, idx_close - 30), idx_close
                lift_end = idx_held
                tcp_held = kin.tcp_frame(motor_to_rad(A[idx_held])[:5])[0]
                while lift_end < idx_rel and kin.tcp_frame(motor_to_rad(A[lift_end])[:5])[0][2] < tcp_held[2] + 0.04:
                    lift_end += 1
                wgt[a0:a1] = np.linspace(0, 1, a1 - a0) if a1 > a0 else 1
                wgt[a1:lift_end] = 1
                b1 = min(n, lift_end + 20)
                wgt[lift_end:b1] = np.linspace(1, 0, b1 - lift_end)
                q_prev = None
                for t in np.flatnonzero(wgt > 0):
                    q = motor_to_rad(A[t])
                    p = kin.tcp_frame(q[:5])[0]
                    target = p + np.array([0, 0, dz * wgt[t]])
                    target[2] = max(target[2], min(p[2], TIP_MIN_Z_B))
                    qn, _ = kin.ik(target, q[:5] if q_prev is None else q_prev, a_goal=kin.tcp_frame(q[:5])[1], w_a=0.3)
                    q_prev = qn
                    A[t, :5] = rad_to_motor(np.r_[qn, q[5]])[:5]

            # ---- 2·3번째 구간: 휴식 자세 → 구간 시작 자세 보간 30프레임을 앞에
            if k > 0:
                m0 = rest_motor.copy()
                pre = np.array([m0 + (A[0] - m0) * (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5) for u in np.linspace(0, 1, 31)[1:]])
                A = np.vstack([pre, A])
                idx_close += 30
                idx_rel += 30
            A = A[::args.stride]

            # ---- 실행
            ok_at, n = None, 0
            for t in range(len(A)):
                pol = obs["policy"]
                if ds is not None:
                    ds.add_frame({"observation.state": rad_to_motor(pol["joint_pos"][0].cpu().numpy()).astype(np.float32),
                                  "action": A[t].astype(np.float32),
                                  "observation.images.front": pol["front"][0].cpu().numpy(),
                                  "observation.images.wrist": pol["wrist"][0].cpu().numpy(), "task": task_text(fruit)})
                obs, *_ = env.step(torch.tensor(motor_to_rad(A[t]), dtype=torch.float32, device=dev).unsqueeze(0))
                n += 1
                if n >= 30 * FPS:
                    break
            on = bool(active_fruit_on_plate(env, FRUITS)[0])
            fT = pose_T(env.scene[fruit].data.root_link_pose_w.torch[0].cpu().numpy())
            lifted = None
            fail = None
            if not on:
                fail = "시간 초과" if n >= 30 * FPS else ("놓침/미끄러짐" if fT[2, 3] < 0.93 else "접시 밖")
            if ds is not None:
                if on:
                    ds.save_episode(parallel_encoding=False)
                else:
                    ds.clear_episode_buffer()
            totals["tried"] += 1
            totals["success"] += int(on)
            if fail:
                totals["fail"][fail] = totals["fail"].get(fail, 0) + 1
            rec = {"fruit": fruit, "episode": ep, "segment": k, "success": on, "fail": fail, "frames": n,
                   "stride": args.stride, "dgrip": round(dgrip, 2), "dz_mm": round(dz * 1000, 1),
                   "yaw_deg": round(math.degrees(yaw), 1), "orange_w_mm": s["width_est_mm"],
                   "fruit_final": np.round(fT[:3, 3], 3).tolist(), "wall_s": round(time.time() - t0, 1)}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{fruit}] ep{ep}-{k} {'성공' if on else '실패: ' + fail} {n}프레임 dgrip={dgrip:.1f} dz={dz * 1000:.0f}mm"
                  f" · 누적 {totals['success']}/{totals['tried']}", flush=True)
        if ds is not None:
            ds.finalize()
        print("SUMMARY", json.dumps(totals, ensure_ascii=False), f"{time.time() - t_all:.0f}s", flush=True)
        os._exit(0)


if __name__ == "__main__":
    main()
