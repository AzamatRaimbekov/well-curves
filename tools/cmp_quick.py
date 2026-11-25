"""Сравнение двух прогонов quick_eval по одинаковым кривым: python tools/cmp_quick.py A.json B.json [--list]"""
import json, sys
def load(p):
    return {(r["api"], r["g"], r["curve"]): r["good_%"] for r in json.load(open(p, encoding="utf-8")) if r.get("status") == "ok"}
a, b = load(sys.argv[1]), load(sys.argv[2])
keys = sorted(set(a) & set(b))
if not keys:
    sys.exit("нет общих кривых")
ma = sum(a[k] for k in keys) / len(keys); mb = sum(b[k] for k in keys) / len(keys)
print(f"общих {len(keys)}: {ma:.1f} -> {mb:.1f}  лучше на 20+: {sum(b[k]-a[k]>=20 for k in keys)}  хуже на 20+: {sum(a[k]-b[k]>=20 for k in keys)}")
if "--list" in sys.argv:
    for k in sorted(keys, key=lambda k: b[k] - a[k]):
        if abs(b[k] - a[k]) >= 20:
            print(f"  {k[0]} g{k[1]} {k[2][:30]:30} {a[k]:6.1f} -> {b[k]:6.1f}")
