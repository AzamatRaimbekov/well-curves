"""Пересчёт точности автообводки на всех обучающих задачах, где есть .improj исполнителя.
Для каждого грида: задание из геометрии исполнителя → автообводка → сверка. Итог: output/report/eval_v2.json|csv.
python tools/batch_eval.py [API ...]   (последовательно, один процесс — сканы большие)"""
import csv
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from wellcurves import improj, pipeline, verify  # noqa: E402
import job_from_improj  # noqa: E402

OUT = ROOT / "output" / "report"
OUT.mkdir(parents=True, exist_ok=True)
rows = []
only = set(sys.argv[1:])
for d in sorted((ROOT / "data" / "training").iterdir()):
    if only and d.name not in only:
        continue
    projs, tifs = list(d.glob("*.improj")), list(d.glob("*.tif"))
    if not projs or not tifs:
        continue
    proj, tif = projs[0], tifs[0]
    try:
        n = len(improj.load(proj)["Grids"])
    except Exception as e:  # noqa: BLE001
        rows.append({"api": d.name, "status": f"improj: {e}"}); continue
    for gi in range(n):
        t0 = time.time()
        try:
            job = job_from_improj.make(proj, tif.resolve(), gi)
            if not job["tracks"][0]["curves"]:
                continue
            jp = ROOT / "data" / "jobs" / f"ref_{d.name}_g{gi}.json"
            jp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
            pipeline.run(jp)
            res = verify.compare(ROOT / "output" / jp.stem / "auto.improj", proj, 0.5, 1.0)
            want = {c["name"] for c in job["tracks"][0]["curves"]}
            for r in res:
                if r.get("curve") in want and r.get("grid") == job["tracks"][0]["name"]:
                    rows.append({"api": d.name, "g": gi, "sec": round(time.time() - t0), **{k: v for k, v in r.items() if k != "worst"}})
        except Exception as e:  # noqa: BLE001
            rows.append({"api": d.name, "g": gi, "status": f"ошибка: {e}", "tb": traceback.format_exc()[-400:]})
        print(d.name, gi, round(time.time() - t0), "s", flush=True)
        (OUT / "eval_v2.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
keys = []
for r in rows:
    for k in r:
        if k not in keys and k != "tb":
            keys.append(k)
with open(OUT / "eval_v2.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
print("done", len(rows))
