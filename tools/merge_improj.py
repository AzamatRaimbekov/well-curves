"""Склеить несколько .improj одного скана в один проект (все секции в одном файле для редактора).
python tools/merge_improj.py <выход.improj> <a.improj> <b.improj> ...  [--suffix "имя1" "имя2" ...]
Гриды (дорожки) копируются как есть; к именам добавляется папка-источник, если имена совпадают."""
import json, sys
from pathlib import Path


def main():
    out, srcs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    base, seen = None, set()
    for p in srcs:
        d = json.loads(p.read_text(encoding="utf-8-sig"))
        if base is None:
            base = d; grids = d["Grids"]; d["Grids"] = []
        else:
            grids = d["Grids"]
        for g in grids:
            if g["Name"] in seen:
                g["Name"] = f'{g["Name"]} [{p.parent.name}]'
            seen.add(g["Name"]); base["Grids"].append(g)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8-sig")
    print(out, "гридов:", len(base["Grids"]), "кривых:", sum(len(c["Items"]) for g in base["Grids"] for c in g["Curves"]))


if __name__ == "__main__":
    main()
