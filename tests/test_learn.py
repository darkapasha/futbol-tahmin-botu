"""Ogrenme dongusu testi (gecici veritabaniyla; gercek tablolara ve params.json'a dokunmaz).

  .venv/bin/python -m tests.test_learn
"""
import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from sim import learn, run
from sim import common


def check(name, cond, info=""):
    print(("OK  " if cond else "HATA") + f" {name} {info}", flush=True)
    return cond


def synthetic(n, bias, rng, league="ENG"):
    """n mac; gercek goller beklenenin (1+bias) kati Poisson."""
    eh, ea = rng.uniform(0.8, 2.2, n), rng.uniform(0.6, 1.8, n)
    hg, ag = rng.poisson(eh * (1 + bias)), rng.poisson(ea * (1 + bias))
    days = pd.date_range("2026-07-01", periods=n, freq="12h")
    return pd.DataFrame({"match_id": range(n), "ymd": days.strftime("%Y%m%d"), "league": league,
                         "home_key": [f"t{i % 20}" for i in range(n)], "away_key": [f"t{(i + 7) % 20}" for i in range(n)],
                         "hg": hg, "ag": ag, "exp_h": eh, "exp_a": ea, "p_home": 0.45, "p_draw": 0.27, "p_away": 0.28})


def main():
    ok = True
    rng = np.random.default_rng(3)
    today = pd.Timestamp("2026-10-01")
    ch = learn.detect(synthetic(200, 0.0, rng), {}, today)
    ok &= check("dogru tahminlerde ayar yok", not [c for c in ch if c[0] != "takim"], str([c[:2] for c in ch]))
    ch = learn.detect(synthetic(200, 0.25, rng), {}, today)
    lg = [c for c in ch if c[0] == "lig_gol"]
    ok &= check("israrli fazla golde lig gol carpani artar", lg and lg[0][4] > 1.0, str([(c[0], c[4]) for c in lg]))
    ok &= check("ayar sinirli (<= %6)", all(c[4] <= 1.06 for c in lg))
    ch = learn.detect(synthetic(30, 0.25, rng), {}, today)
    ok &= check("az macta (30) ayar yok", not [c for c in ch if c[0] == "lig_gol"])

    # evaluate + apply: gecici veritabani ve parametre dosyasiyla
    tmp = Path(tempfile.mkdtemp())
    orig_con, orig_params, orig_hist = learn.sim_con, learn.PARAMS, learn.HIST_DIR
    db = tmp / "sim.sqlite"
    learn.sim_con = lambda: sqlite3.connect(db)
    learn.PARAMS, learn.HIST_DIR = tmp / "params.json", tmp / "hist"
    learn.PARAMS.write_text(json.dumps({"version": "test"}))
    try:
        con = sqlite3.connect(db)
        con.executescript(run.SCHEMA)
        res = pd.read_sql("SELECT match_id, utc_time, league, home, away FROM fm_matches WHERE home_goals IS NOT NULL "
                          "AND utc_time >= '2026-10-01'", common.src_con())
        for r in res.itertuples():
            d = {"p_home": 0.4, "p_draw": 0.3, "p_away": 0.3}
            con.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (r.match_id, "2026-09-30T08:00:00+00:00", r.utc_time[:10].replace("-", ""), r.league, r.utc_time, r.home, r.away,
                         "h", "a", "predicted", "test", 0.4, 0.3, 0.3, 1.4, 1.1, 0.5, 0.5, "1-1", json.dumps(d)))
        con.commit()
        ev = learn.evaluate()
        ok &= check("evaluate: biten maclar eslesti", len(ev) == len(res), f"{len(ev)}/{len(res)}")
        ok &= check("evaluate: ikinci calistirmada tekrar yok", len(learn.evaluate()) == 0)
        p = {"version": "test"}
        learn.apply([("lig_gol", "ENG", ["goal_scale", "ENG"], 1.0, 1.03, {"z": 3.4})], p)
        saved = json.loads(learn.PARAMS.read_text())
        ok &= check("apply: ayar kaydedildi", saved["adjust"]["goal_scale"]["ENG"] == 1.03)
        ok &= check("apply: eski parametre yedeklendi", len(list(learn.HIST_DIR.glob("params_*.json"))) == 1)
        n_log = sqlite3.connect(db).execute("SELECT COUNT(*) FROM learn_log").fetchone()[0]
        ok &= check("apply: learn_log kaydi", n_log == 1)
        learn.rollback()
        ok &= check("rollback: onceki hale dondu", "adjust" not in json.loads(learn.PARAMS.read_text()))
    finally:
        learn.sim_con, learn.PARAMS, learn.HIST_DIR = orig_con, orig_params, orig_hist
    print("\nSONUC:", "hepsi gecti" if ok else "hata var")


if __name__ == "__main__":
    main()
