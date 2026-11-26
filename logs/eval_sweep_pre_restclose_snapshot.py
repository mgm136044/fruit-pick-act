"""P0 조건 스윕 하네스 (91점 대응 지시 P0): 체크포인트를 조건별로 돌려 조건 × 과일 × 실패 유형 표를 만든다.

  source ~/robot_course/activate.sh
  COURSE_ASSETS=~/fruit_datagen/assets_c4 python eval_sweep.py --checkpoint te_models/act_mix2_100k_color_te \
      --per-cond 24 --out logs/sweep_act_mix2_100k_color_te01.json --gif-dir report_html/assets/sweep_mix2_te01 --viz none

실행 방식은 키트 run_policy.py / 서버와 같다: lerobot_policy.load_policy + policy.act, 시작 자세 3스텝 유지,
과일 딸기→오렌지→레몬→바나나 순서, 같은 종류 아무 사본이나 접시 위면 즉시 성공(mdp.active_fruit_on_plate), 30초면 실패.
실행 설정은 체크포인트 config.json 그대로(제출 zip 과 같은 폴더를 넣는다).

조건 (--conds 로 고름, 이름:키=값,키=값 ; 기본 목록은 DEFAULT_CONDS):
  xy=r       과일 1개, 키트 기준 자리에서 ±r m (정사각형 균등, set_episode_events 와 같은 모양)
  yaw=d      과일 회전 ±d 도
  rob=c/d    로봇 위치 ±c cm, 회전 ±d 도
  plate=r    접시 ±r m (로봇 받침과 0.22m 이상)
  light=1    조명 세기 ×0.5~1.5, 색 ±15%
  count=a-b  같은 과일 a~b 개 (조리대 전체 무작위 자리·회전, eval_proxy 의 count 와 같은 규칙)
  distract=1 다른 종류 과일 1~2종 (참고용, 서버에 거의 없음)
조건마다 시드를 따로 둔다(조건 이름 기반) → 조건을 더하거나 빼도 다른 조건의 장면은 같다. 모델끼리 같은 장면 비교 가능.

실패 유형 (자동 분류, 에피소드 기록에서):
  엉뚱한 과일      다른 종류 과일을 3cm 넘게 들었다
  접시 밖에 놓음    목표를 2cm 넘게 들었고, 끝 위치가 접시 중심에서 15cm 안 (놓기·높이 문제)
  들고 가다 떨어뜨림 목표를 2cm 넘게 들었고, 끝 위치가 접시에서 15cm 밖
  접근 빗나감(닫기 전 밀어냄)  첫 쥐기 전에 목표를 15mm 넘게 밂
  접근 빗나감(30mm 초과)       쥐기 명령 때 손끝이 가장 가까운 목표에서 수평 30mm 넘게 떨어짐(모든 쥐기에서) → 위치 범위 데이터가 처방
  접근 빗나감(12~30mm)         같은 오차가 12~30mm → 하강 중 보정(DAgger T1)이 처방
  잡기 실패        12mm 안에서 쥐었는데 2cm 미만으로만 들림 (미끄러짐·덜 닫힘·짜여 빠짐)
  멈춤·망설임      쥐기 전 관절 변화 < 0.2°/스텝이 60스텝 넘게, 또는 마지막 20초 손끝 이동 합 < 5cm
  시간 초과        한 번도 쥐지 않고 계속 움직임
덧붙임 기록: 쥐기 횟수(다시 잡기), 잡기 실패 뒤 빈손으로 접시 10cm 안까지 갔는지(빈손 운반), 성공 스텝.
"""
import argparse
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

DEFAULT_CONDS = ("base:xy=0.03;xy6:xy=0.06;xy10:xy=0.10;yaw45:yaw=45;yaw180:yaw=180;"
                 "rob5:rob=5/15;rob10:rob=10/30;plate6:plate=0.06;light:light=1;cnt23:count=2-3;cnt4:count=4;distract:distract=1")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--conds", default=DEFAULT_CONDS)
ap.add_argument("--per-cond", type=int, default=24, help="조건마다 에피소드 수 (4의 배수: 과일 4종 같은 수)")
ap.add_argument("--max-sec", type=int, default=30)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", default=None)
ap.add_argument("--gif-dir", default=None, help="실패 에피소드는 모두, 성공은 조건마다 첫 1개를 GIF(정면|손목)로")
ap.add_argument("--gif-stride", type=int, default=4)
add_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True

AREA_X, AREA_Y = (1.95, 2.30), (-0.48, -0.26)  # eval_proxy / 생성기와 같은 조리대 범위
PLATE_CLEAR, ROBOT_CLEAR, GAP, GAP_BANANA = 0.14, 0.17, 0.075, 0.11
# 과일별 '닫힘' 판정: 접촉 명령값(간섭 표로 계산, 보고서 05) + 2. 정책 그리퍼 명령이 이 아래로 내려가면 쥐기
CONTACT_CMD = {"Strawberry": 26.6, "Orange": 36.9, "Lemon": 29.6, "Banana": 18.2}


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


def save_gif(frames, name):
    import numpy as np
    from PIL import Image
    os.makedirs(args.gif_dir, exist_ok=True)
    w = 192
    h = int(w * 3 / 4)
    out = []
    for fr, wr in frames:
        ims = [Image.fromarray(np.asarray(x).astype(np.uint8)).resize((w, h)) for x in (fr, wr)]
        im = Image.new("RGB", (w * 2 + 4, h), (250, 250, 248))
        im.paste(ims[0], (0, 0))
        im.paste(ims[1], (w + 4, 0))
        out.append(im.quantize(colors=64, method=Image.Quantize.MEDIANCUT))
    path = os.path.join(args.gif_dir, name)
    out[0].save(path, save_all=True, append_images=out[1:], duration=int(1000 * args.gif_stride / 30), loop=0, optimize=True)
    return path


def main():
    import numpy as np
    import torch

    from course_env.mdp import active_fruit_on_plate, active_index, park_fruits
    from course_env.paths import FRUIT_KO, FRUITS, PLATE, copies
    from course_env.pick_fruit import SO101_CFG, PickFruitEnvCfg, task_text
    from course_env.so101 import motor_to_rad, rad_to_motor
    from kin import SO101Kin, quat_wxyz_to_mat
    from lerobot_policy import load_policy

    conds = parse_conds(args.conds)
    if any(c.get("count") == "4" or c.get("count", "").endswith("-4") for c in conds):
        assert len(copies(FRUITS[0])) >= 4, "count=4 조건은 COURSE_ASSETS=~/fruit_datagen/assets_c4 로 실행"
    policy = load_policy(os.path.abspath(os.path.expanduser(args.checkpoint)), "cuda", ["act"])
    pc = policy.policy.config
    kin = SO101Kin()  # 기본 TCP(손가락 사이 점) — 분류용
    cfg = PickFruitEnvCfg()
    cfg.seed = args.seed
    cfg.terminations.time_out = None
    cfg.terminations.success = None
    results = []
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
        light = omni.usd.get_context().get_stage().GetPrimAtPath("/World/Light")
        l_int, l_col = light.GetAttribute("inputs:intensity"), light.GetAttribute("inputs:color")
        base_int, base_col = l_int.Get(), l_col.Get()
        plate_base = list(getattr(env.cfg.scene, PLATE).init_state.pos)

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

        def pos(name):  # 환경 원점 기준 위치
            return env.scene[name].data.root_pos_w.torch[0].cpu().numpy().astype(np.float64) - org

        def tcp_now():  # 환경 원점 기준 손끝 위치 (기구학 FK, 생성기와 같은 체인)
            p7 = robot.data.root_link_pose_w.torch[0].cpu().numpy().astype(np.float64)
            T = np.eye(4)
            x, y, z, w = p7[3:7]
            T[:3, :3] = quat_wxyz_to_mat([w, x, y, z])
            T[:3, 3] = p7[:3]
            q = robot.data.joint_pos.torch[0].cpu().numpy().astype(np.float64)[:5]
            return T[:3, :3] @ kin.tcp_frame(q)[0] + T[:3, 3] - org

        def setup(c, fi, rng):
            """조건 c 로 장면 배치. 돌려주는 값: 기록용 장면 정보."""
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
                for _ in range(100):
                    d = rng.uniform(-r, r, 2)
                    if math.hypot(plate_base[0] + d[0] - rb[0], plate_base[1] + d[1] - rb[1]) >= 0.22:
                        break
                else:
                    d = np.zeros(2)
                plate = [plate_base[0] + d[0], plate_base[1] + d[1]]
                put(PLATE, plate[0], plate[1], plate_base[2])
                info["plate_cm"] = [round(float(v) * 100, 1) for v in d]
            active_index(env)[0] = fi
            park_fruits(env, ids, list(FRUITS), torch.tensor([-1], device=dev))
            z = getattr(env.cfg.scene, FRUITS[0]).init_state.pos[2]
            if "count" in c or "distract" in c:
                if "count" in c:
                    a, _, b = c["count"].partition("-")
                    n = int(rng.integers(int(a), int(b or a) + 1))
                else:
                    n = 1
                names = copies(FRUITS[fi])[:n]
                extra = {}
                if "distract" in c:
                    for oi in rng.choice([i for i in range(4) if i != fi], size=int(rng.integers(1, 3)), replace=False):
                        extra[FRUITS[int(oi)]] = int(rng.integers(1, 3))
                        names += copies(FRUITS[int(oi)])[:extra[FRUITS[int(oi)]]]
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
                info["count"] = sum(1 for nm, *_ in best if nm in copies(FRUITS[fi]))
                info["extra"] = extra
            else:
                base = getattr(env.cfg.scene, FRUITS[fi]).init_state.pos
                r = float(c.get("xy", 0.03))
                yaw = math.radians(float(c.get("yaw", 0)))
                for _ in range(100):
                    dx, dy = rng.uniform(-r, r, 2)
                    x, y = base[0] + dx, base[1] + dy
                    if math.hypot(x - plate[0], y - plate[1]) >= PLATE_CLEAR and math.hypot(x - rb[0], y - rb[1]) >= ROBOT_CLEAR:
                        break
                else:
                    x, y, dx, dy = base[0], base[1], 0.0, 0.0
                th = float(rng.uniform(-yaw, yaw)) if yaw > 0 else 0.0
                put(FRUITS[fi], x, y, base[2], th)
                info["count"] = 1
                info["fruit_d_cm"] = [round(float(dx) * 100, 1), round(float(dy) * 100, 1)]
                info["yaw_deg"] = round(math.degrees(th), 1)
            lit = None
            if c.get("light") == "1":
                lit = (float(rng.uniform(0.5, 1.5)), rng.uniform(0.85, 1.15, 3).tolist())
                l_int.Set(base_int * lit[0])
                l_col.Set(type(base_col)(*[base_col[i] * lit[1][i] for i in range(3)]))
            else:
                l_int.Set(base_int)
                l_col.Set(base_col)
            info["light"] = None if lit is None else [round(lit[0], 3)] + [round(v, 3) for v in lit[1]]
            return info

        t_all = time.time()
        for c in conds:
            rng = np.random.default_rng([args.seed, int(hashlib.md5(c["name"].encode()).hexdigest()[:8], 16)])
            gif_ok_done = False
            for ep in range(args.per_cond):
                fi = ep % 4
                fruit = FRUITS[fi]
                env.fixed_fruit = fi
                obs, _ = env.reset()
                info = setup(c, fi, rng)
                hold = obs["policy"]["joint_pos"].clone()
                for _ in range(3):
                    obs, *_ = env.step(hold)
                policy.reset()
                tgt = [n for n in copies(fruit) if pos(n)[2] > 0.5]
                oth = [n for f in FRUITS if f != fruit for n in copies(f) if pos(n)[2] > 0.5]
                z0 = {n: pos(n)[2] for n in tgt + oth}
                table_z = min(z0[n] for n in tgt) if tgt else 0.0
                lift = {n: 0.0 for n in tgt + oth}
                thr = CONTACT_CMD[fruit] + 2.0
                closes, prev_closed = [], False
                tcp_path, empty_to_plate = [], False
                xy0 = {n: pos(n)[:2] for n in tgt}
                push_before_close = 0.0  # 첫 쥐기 전 목표 사본의 최대 수평 밀림
                still_run, still_max = 0, 0  # 쥐기 전 관절 변화 < 0.2°/스텝 연속 최대 길이
                q_prev = robot.data.joint_pos.torch[0, :5].cpu().numpy().astype(np.float64)
                frames = [] if args.gif_dir else None
                ok, t0 = False, time.time()
                for t in range(args.max_sec * 30):
                    pol = obs["policy"]
                    if frames is not None and t % args.gif_stride == 0:
                        frames.append((pol["front"][0].cpu().numpy(), pol["wrist"][0].cpu().numpy()))
                    o = {"front": pol["front"][0].cpu().numpy(), "wrist": pol["wrist"][0].cpu().numpy(),
                         "state": rad_to_motor(pol["joint_pos"][0].cpu().numpy()).astype(np.float32)}
                    a = policy.act(o, task_text(fruit))
                    obs, *_ = env.step(torch.tensor(motor_to_rad(a), dtype=torch.float32, device=dev).unsqueeze(0))
                    tc = tcp_now()
                    tcp_path.append(tc)
                    q_cur = robot.data.joint_pos.torch[0, :5].cpu().numpy().astype(np.float64)
                    if not closes:
                        push_before_close = max([push_before_close] + [float(np.linalg.norm(pos(n)[:2] - xy0[n])) for n in tgt])
                        if float(np.degrees(np.abs(q_cur - q_prev)).max()) < 0.2:
                            still_run += 1
                            still_max = max(still_max, still_run)
                        else:
                            still_run = 0
                    q_prev = q_cur
                    closed = float(a[5]) < thr
                    if closed and not prev_closed:  # 쥐기 시작
                        dxy = min(math.hypot(tc[0] - pos(n)[0], tc[1] - pos(n)[1]) for n in tgt) if tgt else 9.9
                        closes.append({"t": t, "dxy_cm": round(dxy * 100, 1), "tcp_h_cm": round((tc[2] - table_z) * 100, 1)})
                    prev_closed = closed
                    for n in tgt + oth:
                        lift[n] = max(lift[n], pos(n)[2] - z0[n])
                    if closes and max((lift[n] for n in tgt), default=0) < 0.02:  # 쥐었는데 못 듦 → 빈손으로 접시 근처?
                        pl = pos(PLATE)
                        if math.hypot(tc[0] - pl[0], tc[1] - pl[1]) < 0.10:
                            empty_to_plate = True
                    if bool(active_fruit_on_plate(env, FRUITS)[0]):
                        ok = True
                        break
                tipped = False
                for n_ in tgt:
                    qw = env.scene[n_].data.root_link_pose_w.torch[0].cpu().numpy().astype(np.float64)
                    Rz = quat_wxyz_to_mat([qw[6], qw[3], qw[4], qw[5]])[:, 2]
                    tipped = tipped or float(np.degrees(np.arccos(np.clip(Rz[2], -1, 1)))) > 25
                pl = pos(PLATE)
                dpl = min((math.hypot(pos(n)[0] - pl[0], pos(n)[1] - pl[1]) for n in tgt), default=9.9)
                max_lift = max((lift[n] for n in tgt), default=0.0)
                wrong = max((lift[n] for n in oth), default=0.0) > 0.03
                tp = np.array(tcp_path)
                last20 = tp[-600:] if len(tp) > 1 else tp
                move20 = float(np.sum(np.linalg.norm(np.diff(last20, axis=0), axis=1))) if len(last20) > 1 else 0.0
                if ok:
                    why = "성공"
                elif wrong:
                    why = "엉뚱한 과일"
                elif max_lift >= 0.02:
                    why = "접시 밖에 놓음" if dpl <= 0.15 else "들고 가다 떨어뜨림"
                elif push_before_close > 0.015:
                    why = "접근 빗나감(닫기 전 밀어냄)"
                elif closes:
                    dmin = min(cl["dxy_cm"] for cl in closes)
                    why = "접근 빗나감(30mm 초과)" if dmin > 3.0 else "접근 빗나감(12~30mm)" if dmin > 1.2 else "잡기 실패"
                else:
                    why = "멈춤·망설임" if (still_max > 60 or move20 < 0.05) else "시간 초과"
                rec = {"cond": c["name"], "fruit": fruit, "ep": ep, "success": ok, "steps": t + 1, "why": why,
                       "n_closes": len(closes), "closes": closes[:6], "max_lift_cm": round(max_lift * 100, 1),
                       "plate_dist_cm": round(dpl * 100, 1), "empty_to_plate": empty_to_plate,
                       "move_last20s_cm": round(move20 * 100, 1), "push_before_close_mm": round(push_before_close * 1000, 1),
                       "still_max_steps": still_max, "tipped": tipped, **info}
                if frames and (not ok or not gif_ok_done):
                    rec["gif"] = save_gif(frames, f"{c['name']}_{ep:02d}_{fruit.lower()}_{'ok' if ok else why.split('·')[0].replace(' ', '')}.gif")
                    gif_ok_done = gif_ok_done or ok
                results.append(rec)
                print(f"[{c['name']}] {FRUIT_KO[fruit]} {'성공' if ok else '실패'} ({t + 1}스텝, {time.time() - t0:.0f}s) · {why}"
                      f" · 쥐기 {len(closes)}회 · 최고 들림 {rec['max_lift_cm']}cm · 접시까지 {rec['plate_dist_cm']}cm", flush=True)
        # ---- 요약
        summ = {}
        for c in conds:
            rs = [r for r in results if r["cond"] == c["name"]]
            whys = {}
            for r in rs:
                whys[r["why"]] = whys.get(r["why"], 0) + 1
            summ[c["name"]] = {"n": len(rs), "success": round(float(np.mean([r["success"] for r in rs])), 3),
                               "per_fruit": {FRUIT_KO[f]: f"{sum(r['success'] for r in rs if r['fruit'] == f)}/{sum(1 for r in rs if r['fruit'] == f)}"
                                             for f in FRUITS},
                               "why": whys, "retry_success": sum(1 for r in rs if r["success"] and r["n_closes"] > 1),
                               "empty_to_plate": sum(1 for r in rs if r["empty_to_plate"] and not r["success"])}
        srv = [v["success"] for k, v in summ.items() if k != "distract"]
        rep = {"checkpoint": args.checkpoint, "n_action_steps": pc.n_action_steps,
               "temporal_ensemble": getattr(pc, "temporal_ensemble_coeff", None), "seed": args.seed, "conds": conds,
               "summary": summ, "mean_success_x100": round(float(np.mean(srv)) * 100, 1) if srv else None,
               "min_cond": min(((k, v["success"]) for k, v in summ.items() if k != "distract"), key=lambda kv: kv[1]) if srv else None,
               "episodes": results, "wall_s": round(time.time() - t_all)}
        out = args.out or os.path.join(HERE, "logs", f"sweep_{int(time.time())}.json")
        json.dump(rep, open(out, "w"), indent=1, ensure_ascii=False)
        print("SWEEP 평균(distract 제외) ×100:", rep["mean_success_x100"], "최저 조건:", rep["min_cond"],
              json.dumps({k: v["success"] for k, v in summ.items()}, ensure_ascii=False), flush=True)
        os._exit(0)


if __name__ == "__main__":
    main()
