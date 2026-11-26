"""로컬 '대리 비공개 환경' 채점: 체크포인트를 여러 변형 장면에서 돌려 장면별·과일별 성공률을 낸다.

  source ~/robot_course/activate.sh
  python eval_proxy.py --checkpoint ~/act_v1 --per-scene 8 --viz none
  python eval_proxy.py --checkpoint ~/act_v1 --scenes default,robot,light --per-scene 12 --n-action-steps 20 --viz none

채점 방식은 키트 run_policy.py / 서버와 같다: 과일을 딸기→오렌지→레몬→바나나 순서로 돌림, 시작 자세 3스텝 유지,
lerobot_policy.load_policy 로 체크포인트 실행, 접시 위에 놓이면 즉시 성공, 30초 넘으면 실패.
장면 (비공개 환경이 어떻게 다른지는 모름 → 키트 record.py 가 지원하는 변형 + 조명으로 추정):
  default  키트 기본 (과일 기준 자리 ±3cm)
  count    같은 과일 2~3개 (아무거나 접시에 놓으면 성공). 키트 장면은 종류마다 물체 3개라 최대 3
  count4   같은 과일 4개 (서버의 "바나나 4개 중 하나"). 종류당 4개 장면 사본 필요: COURSE_ASSETS=~/fruit_datagen/assets_c4 (build_scene_c4.py)
  distract 다른 과일 1~2종 함께 — 사용자 정보(2026-11-25): 서버 평가에는 이 장면이 없다 → 점수에서 빼고 일반화 참고로만 보고
  robot    로봇 위치 ±10cm, 회전 ±30°
  light    조명 세기 ×0.5~1.5, 색 약간 바꿈 (이미지 증강이 효과 있는지 보기)
  mixed    위를 무작위 조합
--n-action-steps / --temporal-ensemble: 체크포인트 config 를 바꾸지 않고 실행 설정만 바꿔 비교 (바꾼 값으로 config.json 을 고쳐 내면 서버도 같게 실행됨)
"""
import argparse
import json
import math
import os
import sys
import time

KIT = os.environ.get("ROBOT_KIT", os.path.expanduser("~/student_kit"))
sys.path.insert(0, os.path.join(KIT, "isaac"))
sys.path.insert(0, KIT)
HERE = os.path.dirname(os.path.abspath(__file__))
from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--scenes", default="default,count,robot,light,mixed,distract")
ap.add_argument("--per-scene", type=int, default=8, help="장면마다 에피소드 수 (4의 배수 권장)")
ap.add_argument("--max-sec", type=int, default=30)
ap.add_argument("--seed", type=int, default=123)
ap.add_argument("--n-action-steps", type=int, default=None)
ap.add_argument("--temporal-ensemble", type=float, default=None, help="예: 0.01 (n_action_steps=1 로 바뀜)")
ap.add_argument("--out", default=None)
ap.add_argument("--gif-dir", default=None, help="정책 롤아웃 GIF 저장 폴더 (정면+손목, 3스텝마다)")
ap.add_argument("--gif-per-scene", type=int, default=0, help="장면마다 처음 N 에피소드를 GIF 로")
ap.add_argument("--grip-remap", type=float, default=None,
                help="진단 전용(제출물에 넣지 않음): 그리퍼 명령을 40 + k·(a−40) 로 바꿔 보냄. 예: 1.5 = 더 깊게 쥐고 조금 더 벌림")
add_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True

AREA_X, AREA_Y = (1.95, 2.30), (-0.48, -0.26)
PLATE_CLEAR, ROBOT_CLEAR, GAP, GAP_BANANA = 0.14, 0.17, 0.075, 0.11


def save_gif(frames, name, w=192):
    """frames: [(front HxWx3, wrist HxWx3)] uint8 → 정면|손목 나란히 GIF (10fps = 3스텝마다, 실제 속도)."""
    import numpy as np
    from PIL import Image
    os.makedirs(args.gif_dir, exist_ok=True)
    h = int(w * 3 / 4)
    out = []
    for fr, wr in frames:
        ims = [Image.fromarray(np.asarray(x).astype(np.uint8)).resize((w, h)) for x in (fr, wr)]
        im = Image.new("RGB", (w * 2 + 4, h), (250, 250, 248))
        im.paste(ims[0], (0, 0))
        im.paste(ims[1], (w + 4, 0))
        out.append(im.quantize(colors=64, method=Image.Quantize.MEDIANCUT))
    path = os.path.join(args.gif_dir, name)
    out[0].save(path, save_all=True, append_images=out[1:], duration=100, loop=0, optimize=True)
    return path


def main():
    import numpy as np
    import torch

    from course_env.mdp import active_fruit_on_plate, active_index, park_fruits
    from course_env.paths import FRUIT_KO, FRUITS, PLATE, copies
    if "count4" in args.scenes.split(","):
        assert len(copies(FRUITS[0])) >= 4, "count4 장면은 COURSE_ASSETS=~/fruit_datagen/assets_c4 로 실행"
    from course_env.pick_fruit import SO101_CFG, PickFruitEnvCfg, task_text
    from course_env.so101 import motor_to_rad, rad_to_motor
    from lerobot_policy import load_policy

    policy = load_policy(os.path.abspath(os.path.expanduser(args.checkpoint)), "cuda", ["act"])
    pc = policy.policy.config
    if args.temporal_ensemble is not None:
        pc.temporal_ensemble_coeff = args.temporal_ensemble
        pc.n_action_steps = 1
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        policy.policy.temporal_ensembler = ACTTemporalEnsembler(args.temporal_ensemble, pc.chunk_size)
    elif args.n_action_steps is not None:
        pc.n_action_steps = args.n_action_steps
    policy.policy.reset()
    rng = np.random.default_rng(args.seed)
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
        robot = env.scene["robot"]
        base_pos = torch.tensor(SO101_CFG.init_state.pos, device=dev)
        zero6 = torch.zeros(1, 6, device=dev)
        light = omni.usd.get_context().get_stage().GetPrimAtPath("/World/Light")
        l_int, l_col = light.GetAttribute("inputs:intensity"), light.GetAttribute("inputs:color")
        base_int, base_col = l_int.Get(), l_col.Get()

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

        def place_fruits(fi, count, extra, rob):  # record.py place_fruits 와 같은 규칙
            active_index(env)[0] = fi
            park_fruits(env, ids, list(FRUITS), torch.tensor([-1], device=dev))
            if count == 1 and not extra:
                base = getattr(env.cfg.scene, FRUITS[fi]).init_state.pos
                dx, dy = rng.uniform(-0.03, 0.03, size=2)
                put(FRUITS[fi], base[0] + dx, base[1] + dy, base[2])
                return
            names = copies(FRUITS[fi])[:count]
            for oi, n in extra.items():
                if oi != fi:
                    names += copies(FRUITS[oi])[:n]
            plate = (env.scene[PLATE].data.root_pos_w.torch[0] - origin).tolist()
            z = getattr(env.cfg.scene, FRUITS[0]).init_state.pos[2]
            rb = (float(base_pos[0]) + rob[1] / 100, float(base_pos[1]) + rob[0] / 100)
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

        def scene_params(kind, fi):
            count, extra, rob, lit = 1, {}, (0.0, 0.0, 0.0), None
            if kind == "mixed":
                kind = rng.choice(["count", "robot", "light"], size=int(rng.integers(1, 4)), replace=False).tolist()  # 서버에 없는 distract 제외
            kinds = kind if isinstance(kind, list) else [kind]
            if "count" in kinds:
                count = int(rng.integers(2, 4))
            if "count4" in kinds:
                count = 4
            if "distract" in kinds:
                for oi in rng.choice([i for i in range(4) if i != fi], size=int(rng.integers(1, 3)), replace=False):
                    extra[int(oi)] = int(rng.integers(1, 3))
            if "robot" in kinds:
                rob = (float(rng.uniform(-10, 10)), float(rng.uniform(-10, 10)), float(rng.uniform(-30, 30)))
            if "light" in kinds:
                lit = (float(rng.uniform(0.5, 1.5)), rng.uniform(0.85, 1.15, 3).tolist())
            return count, extra, rob, lit, kinds

        t_all = time.time()
        for kind in args.scenes.split(","):
            for ep in range(args.per_scene):
                fi = ep % 4
                fruit = FRUITS[fi]
                count, extra, rob, lit, kinds = scene_params(kind, fi)
                env.fixed_fruit = fi
                obs, _ = env.reset()
                place_robot(rob)
                place_fruits(fi, count, extra, rob)
                if lit:
                    l_int.Set(base_int * lit[0])
                    l_col.Set(type(base_col)(*[base_col[i] * lit[1][i] for i in range(3)]))
                else:
                    l_int.Set(base_int)
                    l_col.Set(base_col)
                hold = obs["policy"]["joint_pos"].clone()
                for _ in range(3):
                    obs, *_ = env.step(hold)
                policy.reset()
                ok, t0 = False, time.time()
                frames = [] if (args.gif_dir and ep < args.gif_per_scene) else None
                # 진단: 대상 종류 사본들의 처음 높이 → 최고 들림, 끝 위치 (못 잡음 / 들었다 떨어뜨림 / 접시 근처 등 분류)
                tgt = [n for n in copies(fruit) if float((env.scene[n].data.root_pos_w.torch[0] - origin)[2]) > 0.5]
                z0 = {n: float(env.scene[n].data.root_pos_w.torch[0, 2]) for n in tgt}
                lift = {n: 0.0 for n in tgt}
                for t in range(args.max_sec * 30):
                    pol = obs["policy"]
                    if frames is not None and t % 3 == 0:
                        frames.append((pol["front"][0].cpu().numpy(), pol["wrist"][0].cpu().numpy()))
                    o = {"front": pol["front"][0].cpu().numpy(), "wrist": pol["wrist"][0].cpu().numpy(),
                         "state": rad_to_motor(pol["joint_pos"][0].cpu().numpy()).astype(np.float32)}
                    a = policy.act(o, task_text(fruit))
                    if args.grip_remap is not None:
                        a = a.copy()
                        a[5] = 40.0 + args.grip_remap * (a[5] - 40.0)
                    obs, *_ = env.step(torch.tensor(motor_to_rad(a), dtype=torch.float32, device=dev).unsqueeze(0))
                    for n in tgt:
                        lift[n] = max(lift[n], float(env.scene[n].data.root_pos_w.torch[0, 2]) - z0[n])
                    if bool(active_fruit_on_plate(env, FRUITS)[0]):
                        ok = True
                        break
                pl = env.scene[PLATE].data.root_pos_w.torch[0]
                dpl = {n: round(float(torch.linalg.norm(env.scene[n].data.root_pos_w.torch[0, :2] - pl[:2])) * 100, 1) for n in tgt}
                max_lift = round(max(lift.values()) * 100, 1) if lift else None
                min_dpl = min(dpl.values()) if dpl else None
                why = ("성공" if ok else "못 들어올림(잡기 실패)" if (max_lift or 0) < 2.0
                       else "들었지만 접시 밖" if (min_dpl or 0) > 10 else "접시 근처(놓기·높이 문제)")
                results.append({"scene": kind, "kinds": kinds, "fruit": fruit, "success": ok, "steps": t + 1,
                                "count": count, "extra": {FRUITS[k]: v for k, v in extra.items()}, "robot": rob,
                                "light": lit, "max_lift_cm": max_lift, "plate_dist_cm": min_dpl, "why": why})
                if frames:
                    results[-1]["gif"] = save_gif(frames, f"{kind}_{ep:02d}_{fruit.lower()}_{'ok' if ok else 'fail'}.gif")
                print(f"[{kind}] {FRUIT_KO[fruit]} {'성공' if ok else '실패'} ({t + 1}스텝, {time.time() - t0:.0f}s) · {why} · 최고 들림 {max_lift}cm · 접시까지 {min_dpl}cm", flush=True)
        summ = {}
        why_all = {}
        for r in results:
            why_all[r["why"]] = why_all.get(r["why"], 0) + 1
        for kind in args.scenes.split(","):
            rs = [r for r in results if r["scene"] == kind]
            summ[kind] = {"success": round(float(np.mean([r["success"] for r in rs])), 3),
                          "per_fruit": {FRUIT_KO[f]: round(float(np.mean([r["success"] for r in rs if r["fruit"] == f])), 2)
                                        for f in FRUITS if any(r["fruit"] == f for r in rs)}}
        server_like = {k: v for k, v in summ.items() if k != "distract"}
        overall = round(float(np.mean([v["success"] for v in server_like.values()])), 3) if server_like else 0.0
        rep = {"checkpoint": args.checkpoint, "n_action_steps": pc.n_action_steps,
               "temporal_ensemble": getattr(pc, "temporal_ensemble_coeff", None), "grip_remap": args.grip_remap, "scenes": summ,
               "score_mean_of_scenes_x100": round(overall * 100, 1), "score_note": "distract 제외 (서버에 없음)", "why": why_all, "episodes": results, "wall_s": round(time.time() - t_all)}
        out = args.out or os.path.join(HERE, "logs", f"eval_proxy_{int(time.time())}.json")
        json.dump(rep, open(out, "w"), indent=1, ensure_ascii=False)
        print("SCORE(서버형 장면 성공률 평균 ×100, distract 제외):", rep["score_mean_of_scenes_x100"], json.dumps(summ, ensure_ascii=False), flush=True)
        os._exit(0)


if __name__ == "__main__":
    main()
