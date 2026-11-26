"""장면 탐색 (1회): 생성기에 필요한 기하 정보를 뽑아 so101_chain.json / scene_geom.json 으로 저장.

  python explore_scene.py --viz none

- 로봇 USD 의 회전 관절 프레임 (FK 용) → so101_chain.json
- FK 검증: 무작위 관절값을 시뮬레이터에 쓰고 gripper 링크 자세와 비교
- 그리퍼/조(jaw) 메시 점을 gripper 링크 프레임으로 → 손가락 끝, 벌림 폭
- 과일·접시 메시의 root 기준 AABB, 리셋 후 안착 자세 → 조리대 높이
키트 원본은 고치지 않고 import 만 한다.
"""
import argparse
import json
import os
import sys

KIT = os.environ.get("ROBOT_KIT", os.path.expanduser("~/student_kit"))
sys.path.insert(0, os.path.join(KIT, "isaac"))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

ap = argparse.ArgumentParser()
add_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True


def main():
    import numpy as np
    import torch
    from PIL import Image

    from course_env.paths import FRUITS, PLATE
    from course_env.pick_fruit import PickFruitEnvCfg
    from course_env.so101 import JOINTS

    cfg = PickFruitEnvCfg()
    cfg.terminations.time_out = None
    cfg.terminations.success = None
    out = {}
    with launch_simulation(cfg, args):
        from isaaclab.envs import ManagerBasedRLEnv
        from pxr import Usd, UsdGeom, UsdPhysics

        env = ManagerBasedRLEnv(cfg)
        dev = env.device
        robot = env.scene["robot"]
        ids = torch.tensor([0], device=dev)
        obs, _ = env.reset()
        print("joint_names", robot.data.joint_names)
        print("body_names", robot.data.body_names)
        lim = robot.data.joint_pos_limits.torch[0].cpu().numpy()
        print("joint limits (rad)", np.round(lim, 4).tolist())

        import omni.usd
        stage = omni.usd.get_context().get_stage()
        rroot = "/World/envs/env_0/Robot"

        # ---- 관절 프레임
        joints = {}
        for prim in Usd.PrimRange(stage.GetPrimAtPath(rroot)):
            if prim.IsA(UsdPhysics.RevoluteJoint):
                j = UsdPhysics.RevoluteJoint(prim)
                q0, q1 = j.GetLocalRot0Attr().Get(), j.GetLocalRot1Attr().Get()
                joints[prim.GetName()] = {
                    "path": str(prim.GetPath()),
                    "body0": [str(t) for t in j.GetBody0Rel().GetTargets()],
                    "body1": [str(t) for t in j.GetBody1Rel().GetTargets()],
                    "lp0": list(j.GetLocalPos0Attr().Get()), "lr0": [q0.GetReal(), *q0.GetImaginary()],
                    "lp1": list(j.GetLocalPos1Attr().Get()), "lr1": [q1.GetReal(), *q1.GetImaginary()],
                    "axis": j.GetAxisAttr().Get(),
                    "lower_deg": j.GetLowerLimitAttr().Get(), "upper_deg": j.GetUpperLimitAttr().Get(),
                }
        for k, v in joints.items():
            print("JOINT", k, v["body0"], "->", v["body1"], "axis", v["axis"], "lp0", np.round(v["lp0"], 4),
                  "lp1", np.round(v["lp1"], 4))
        # 링크 xform 스케일 확인
        xc = UsdGeom.XformCache()
        for b in robot.data.body_names:
            p = stage.GetPrimAtPath(f"{rroot}/{b}")
            if p.IsValid():
                m = xc.GetLocalToWorldTransform(p)
                print("BODY", b, "scale", np.round([m.GetRow3(i).GetLength() for i in range(3)], 4))

        lim_d = {n: [float(lim[i, 0]), float(lim[i, 1])] for i, n in enumerate(robot.data.joint_names)}
        chain = {"joints": {}, "limits_rad": lim_d}
        name_map = {}
        for jn in JOINTS:
            cand = [k for k in joints if k == jn] or [k for k in joints if jn in k]
            name_map[jn] = cand[0] if cand else None
        print("name_map", name_map)
        for jn, k in name_map.items():
            if k:
                chain["joints"][jn] = joints[k]

        # ---- gripper 링크 기준 메시 점 (gripper, jaw)
        bnames = robot.data.body_names
        gi = bnames.index("gripper")
        pose_w = robot.data.body_link_pose_w.torch[0].cpu().numpy()  # (B, 7) pos + xyzw
        root_w = robot.data.root_link_pose_w.torch[0].cpu().numpy()

        def T_of(p7):
            from kin import quat_wxyz_to_mat
            T = np.eye(4)
            x, y, z, w = p7[3:7]
            T[:3, :3] = quat_wxyz_to_mat([w, x, y, z])
            T[:3, 3] = p7[:3]
            return T


        def usd_T(path):
            return np.array(xc.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T  # Gf 는 행벡터 규약 → 전치

        def mesh_points(path, ref_path):
            """path 아래 메시 점을 ref_path prim 의 (USD) 프레임으로. USD 변환끼리만 써서 Fabric 동기화와 무관."""
            T_ref_inv = np.linalg.inv(usd_T(ref_path))
            pts = []
            for p in Usd.PrimRange(stage.GetPrimAtPath(path), Usd.TraverseInstanceProxies()):
                if p.IsA(UsdGeom.Mesh):
                    v = UsdGeom.Mesh(p).GetPointsAttr().Get()
                    if v is None:
                        continue
                    M = np.array(xc.GetLocalToWorldTransform(p)).T
                    P = np.c_[np.array(v, dtype=np.float64), np.ones(len(v))] @ M.T
                    pts.append((P @ T_ref_inv.T)[:, :3])
            return np.concatenate(pts) if pts else np.zeros((0, 3))

        def dump_tree(path, depth=4):
            base = path.count("/")
            for p in Usd.PrimRange(stage.GetPrimAtPath(path), Usd.TraverseInstanceProxies()):
                if p.GetPath().pathString.count("/") - base <= depth:
                    print("  prim", p.GetPath(), p.GetTypeName())

        gp = mesh_points(f"{rroot}/gripper", f"{rroot}/gripper")   # gripper 링크 프레임
        jp = mesh_points(f"{rroot}/jaw", f"{rroot}/jaw")           # jaw 링크 프레임
        # jaw 링크 USD 자세를 gripper 링크 프레임으로 (USD 에 저장된 관절값에서) + 그때의 관절값
        T_g_j_usd = np.linalg.inv(usd_T(f"{rroot}/gripper")) @ usd_T(f"{rroot}/jaw")
        print("gripper mesh pts", gp.shape, "min", np.round(gp.min(0), 4) if len(gp) else None,
              "max", np.round(gp.max(0), 4) if len(gp) else None)
        print("jaw mesh pts (jaw frame)", jp.shape, "min", np.round(jp.min(0), 4) if len(jp) else None,
              "max", np.round(jp.max(0), 4) if len(jp) else None)
        print("T_g_j_usd", np.round(T_g_j_usd, 4).tolist())
        np.savez(os.path.join(HERE, "logs", "gripper_pts.npz"), gripper=gp, jaw=jp, T_g_j_usd=T_g_j_usd)

        # ---- FK 검증용: 무작위 관절값 → 링크 자세
        rng = np.random.default_rng(0)
        samples = []
        jn_sim = robot.data.joint_names
        for _ in range(12):
            q = np.array([rng.uniform(lim[i, 0] * 0.8, lim[i, 1] * 0.8) for i in range(len(jn_sim))], dtype=np.float32)
            robot.write_joint_state_to_sim_index(position=torch.tensor(q, device=dev).unsqueeze(0),
                                                 velocity=torch.zeros(1, len(q), device=dev), env_ids=ids)
            pw = robot.data.body_link_pose_w.torch[0].cpu().numpy()
            rw = robot.data.root_link_pose_w.torch[0].cpu().numpy()
            samples.append({"q": q.tolist(), "gripper": pw[gi].tolist(), "root": rw.tolist(),
                            "jaw": pw[bnames.index("jaw")].tolist() if "jaw" in bnames else None})
        chain["fk_samples"] = samples
        chain["root_pose_w"] = root_w.tolist()
        chain["body_names"] = bnames
        chain["sim_joint_names"] = jn_sim
        with open(os.path.join(HERE, "so101_chain_raw.json"), "w") as f:
            json.dump(chain, f, indent=1)

        # ---- 과일·접시 기하 (root 기준 AABB) 와 안착 자세
        geom = {}
        for fi, n in enumerate(FRUITS):
            env.fixed_fruit = fi
            obs, _ = env.reset()
            hold = obs["policy"]["joint_pos"].clone()
            z0 = float(env.scene[n].data.root_pos_w.torch[0, 2])
            for _ in range(45):
                obs, *_ = env.step(hold)
            ob = env.scene[n]
            p7 = ob.data.root_link_pose_w.torch[0].cpu().numpy()
            T_wo = T_of(p7)
            prim_path = ob.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_[^/]+", "env_0")
            parent = prim_path.rsplit("/", 1)[0]
            pts = mesh_points(parent, prim_path)   # root(강체) 프레임
            if len(pts) == 0:
                dump_tree(parent)
            world_pts = pts @ T_wo[:3, :3].T + T_wo[:3, 3]  # 시뮬레이터 안착 자세로 옮긴 점
            np.save(os.path.join(HERE, "logs", f"fruit_pts_{n}.npy"), pts.astype(np.float32))
            mass = float(ob.data.default_mass.torch[0].sum()) if hasattr(ob.data, "default_mass") else None
            geom[n] = {"root_pose_settled": p7.tolist(), "z_spawn": z0,
                       "aabb_local": [pts.min(0).tolist(), pts.max(0).tolist()],
                       "world_zmin": float(world_pts[:, 2].min()), "world_zmax": float(world_pts[:, 2].max()),
                       "mass": mass, "n_pts": int(len(pts))}
            print("FRUIT", n, "settled", np.round(p7, 4), "local aabb", np.round(pts.min(0), 4), np.round(pts.max(0), 4),
                  "world z", round(float(world_pts[:, 2].min()), 4), round(float(world_pts[:, 2].max()), 4), "mass", mass)
            Image.fromarray(obs["policy"]["front"][0].cpu().numpy().astype(np.uint8)).save(
                os.path.join(HERE, "logs", f"explore_front_{n}.png"))
            Image.fromarray(obs["policy"]["wrist"][0].cpu().numpy().astype(np.uint8)).save(
                os.path.join(HERE, "logs", f"explore_wrist_{n}.png"))
        pl = env.scene[PLATE]
        p7 = pl.data.root_link_pose_w.torch[0].cpu().numpy()
        ppath = pl.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_[^/]+", "env_0")
        pts = mesh_points(ppath.rsplit("/", 1)[0], ppath)
        np.save(os.path.join(HERE, "logs", "fruit_pts_Plate.npy"), pts.astype(np.float32))
        geom[PLATE] = {"root_pose_settled": p7.tolist(), "aabb_local": [pts.min(0).tolist(), pts.max(0).tolist()]}
        print("PLATE", np.round(p7, 4), "aabb", np.round(pts.min(0), 4), np.round(pts.max(0), 4))
        geom["env_origin"] = env.scene.env_origins[0].cpu().numpy().tolist()
        geom["robot_root"] = robot.data.root_link_pose_w.torch[0].cpu().numpy().tolist()
        with open(os.path.join(HERE, "scene_geom.json"), "w") as f:
            json.dump(geom, f, indent=1)
        print("EXPLORE_DONE")
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
