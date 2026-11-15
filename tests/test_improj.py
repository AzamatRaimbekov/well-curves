"""Проверки на эталонах РЕ-БУЛАК: чтение, пересборка без потерь, автопроверка ловит известные ошибки."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wellcurves import check, improj, improj_tools  # noqa: E402

REF = ROOT / "data" / "reference" / "tz"


def test_roundtrip_all():
    for f in sorted(REF.glob("*.improj")):
        assert improj_tools.roundtrip_check(f), f.name


def test_gamma_range_2done():
    p = improj.load(REF / "2done.improj")
    g = p["Grids"][0]
    v = improj.curve_values(g, g["Curves"][0]["Items"][0])
    assert 6499 < v["depth"].min() < 6502 and 8503 < v["depth"].max() < 8507
    assert 0 < v["value"].min() and v["value"].max() < 300  # с уровнем перехлёста 150..300


def test_check_catches_zero_scale():
    for name, bad in [("4done.improj", "Surface Weight On Bit"), ("555.improj", "Gamma Ray (HSGR)")]:
        res = check.check_project(improj.load(REF / name), dpi=200)
        assert any(i["level"] == "error" and bad in i["where"] for i in res["issues"]), name


def test_scale_detection():
    p = improj.load(REF / "2done.improj")
    assert check.detect_scale(p["Grids"][0], 200)["rate_som"] == 45


if __name__ == "__main__":
    for t in [test_roundtrip_all, test_gamma_range_2done, test_check_catches_zero_scale, test_scale_detection]:
        t()
        print("ok", t.__name__)
