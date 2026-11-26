"""종류마다 과일 4개인 장면 사본 만들기 (키트 원본은 그대로, 별도 폴더에 덧씌우는 층만 만듦).

서버 평가에 '바나나 4개 중 하나 옮기기'가 있는데 키트 장면은 종류당 3개라 로컬에서 재현이 안 됨 → 사본으로 재현.
  <out>/robots                         → 키트 robots 폴더 링크
  <out>/scenes/kitchen_fruit/scene.usd → 원본 scene.usd 를 sublayer 로 깔고 <과일>_4 만 추가 (원본 <과일>_3 정의를 복사,
                                         객체 참조는 절대 경로로 바꿈)
  <out>/scenes/kitchen_fruit/objects.json → 원본 + <과일>_4 (보관 위치는 키트 paths.PARK 규칙을 16개 기준으로 다시 계산)
사용:
  source ~/robot_course/activate.sh
  python build_scene_c4.py --out ~/fruit_datagen/assets_c4 [--copies 4]
  COURSE_ASSETS=~/fruit_datagen/assets_c4 python gen_scripted.py ...   # 환경이 objects.json 에서 4개를 읽음
"""
import argparse
import json
import os

from pxr import Sdf, Usd, UsdPhysics

ap = argparse.ArgumentParser()
ap.add_argument("--kit-assets", default=os.path.expanduser("~/student_kit/isaac/assets"))
ap.add_argument("--out", default=os.path.expanduser("~/fruit_datagen/assets_c4"))
ap.add_argument("--copies", type=int, default=4)
a = ap.parse_args()

FRUITS = ("Strawberry", "Orange", "Lemon", "Banana")  # 키트 paths.FRUITS 와 같은 순서
src_dir = os.path.join(a.kit_assets, "scenes", "kitchen_fruit")
src_usd = os.path.join(src_dir, "scene.usd")
out_dir = os.path.join(a.out, "scenes", "kitchen_fruit")
os.makedirs(out_dir, exist_ok=True)
rob = os.path.join(a.out, "robots")
if not os.path.exists(rob):
    os.symlink(os.path.join(a.kit_assets, "robots"), rob)

src = Sdf.Layer.FindOrOpen(src_usd)
objs = json.load(open(os.path.join(src_dir, "objects.json")))
base_n = sum(1 for k in objs if k == FRUITS[0] or k.startswith(FRUITS[0] + "_"))
assert base_n >= 2, base_n

# 키트 paths.PARK 와 같은 규칙: FRUIT_OBJECTS 순서 i 에 대해 (0.3 + 0.2*(i%4), -2.2 - 0.2*(i//4), 0.01)
names = [o for f in FRUITS for o in [f] + [f"{f}_{i}" for i in range(2, a.copies + 1)]]
park = {o: (round(0.3 + 0.2 * (i % 4), 4), round(-2.2 - 0.2 * (i // 4), 4), 0.01) for i, o in enumerate(names)}

dst_path = os.path.join(out_dir, "scene.usd")
if os.path.exists(dst_path):
    os.remove(dst_path)  # 이 스크립트가 만든 사본만 다시 만듦 (키트 파일 아님)
dst = Sdf.Layer.CreateNew(dst_path, args={"format": "usda"})
dst.subLayerPaths.append(os.path.abspath(src_usd))
dst.defaultPrim = src.defaultPrim
for k in ("upAxis", "metersPerUnit", "kilogramsPerUnit"):
    if src.pseudoRoot.HasInfo(k):
        dst.pseudoRoot.SetInfo(k, src.pseudoRoot.GetInfo(k))
root = "/" + src.defaultPrim
Sdf.CreatePrimInLayer(dst, root).specifier = Sdf.SpecifierOver

new_objs = dict(objs)
for f in FRUITS:
    for o in [f"{f}_{i}" for i in range(2, base_n + 1)]:
        new_objs[o] = list(park[o])  # 기존 사본 보관 위치도 16개 기준 PARK 로 (환경은 reset 때 PARK 로 옮김)
    tmpl = f"{f}_{base_n}"
    for i in range(base_n + 1, a.copies + 1):
        n = f"{f}_{i}"
        assert Sdf.CopySpec(src, f"{root}/{tmpl}", dst, f"{root}/{n}")
        child = dst.GetPrimAtPath(f"{root}/{n}/{tmpl}")
        child.name = n
        child = dst.GetPrimAtPath(f"{root}/{n}/{n}")
        refs = child.referenceList.prependedItems
        new_refs = [Sdf.Reference(os.path.normpath(os.path.join(src_dir, r.assetPath)), r.primPath, r.layerOffset) for r in refs]
        child.referenceList.prependedItems.clear()
        for r in new_refs:
            child.referenceList.prependedItems.append(r)
        child.attributes["xformOp:translate"].default = park[n]
        new_objs[n] = list(park[n])
dst.Save()
json.dump(new_objs, open(os.path.join(out_dir, "objects.json"), "w"), indent=1)

# 확인: 조립된 장면에서 종류별 강체 수·참조
st = Usd.Stage.Open(dst_path, load=Usd.Stage.LoadNone)
st.Load()
for f in FRUITS:
    got = []
    for i in range(1, a.copies + 1):
        n = f if i == 1 else f"{f}_{i}"
        p = st.GetPrimAtPath(f"{root}/{n}/{n}")
        ok = p.IsValid() and p.HasAPI(UsdPhysics.RigidBodyAPI) and len(list(Usd.PrimRange(p))) > 2
        got.append((n, ok))
    print(f, got)
print("defaultPrim", st.GetDefaultPrim().GetPath(), "→", dst_path)
