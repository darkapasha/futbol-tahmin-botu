"""Gunun tahminleri: FotMob'dan gunun maclarini ve mac oncesi kadrolari ceker, her maci simule eder,
sonucu data/sim.sqlite 'predictions' tablosuna yazar.

  .venv/bin/python -m sim.run [YYYYMMDD] [--n 10000] [--no-fetch]
  .venv/bin/python -m sim.run show <match_id>
"""
import argparse
import json
from datetime import date, datetime, timezone

import pandas as pd

from . import fixtures
from .common import load_params, sim_con

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
  match_id INTEGER, created_at TEXT, ymd TEXT, league TEXT, utc_time TEXT, home TEXT, away TEXT,
  home_key TEXT, away_key TEXT, lineup_type TEXT, version TEXT,
  p_home REAL, p_draw REAL, p_away REAL, exp_h REAL, exp_a REAL, over25 REAL, btts REAL,
  top_score TEXT, data TEXT, PRIMARY KEY (match_id, created_at));
"""


def version():
    p = load_params()
    return p.get("version", "v1")


def predict_day(ymd, n=10000, fetch=True, P=None, quiet=False, match_ids=None):
    from .match import Predictor
    if fetch:
        fixtures.update(ymd)
    pm, pl = fixtures.load(match_ids=match_ids, ymd=None if match_ids else ymd)
    pm = pm[pm.finished == 0]
    if pm.empty:
        print(f"{ymd}: tahmin edilecek mac yok")
        return []
    P = P or Predictor(pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)))
    con = sim_con()
    con.executescript(SCHEMA)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = []
    for m in pm.itertuples():
        try:
            o = _one(P, m, pl[pl.match_id == m.match_id], n)
        except Exception as e:  # tek mac hatasi gunu durdurmasin
            print(f"  ! {m.home} - {m.away}: {e}")
            continue
        con.execute("INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (int(m.match_id), created, ymd, m.league, m.utc_time, m.home, m.away, o["home_key"], o["away_key"],
                     m.lineup_type, version(), o["p_home"], o["p_draw"], o["p_away"], o["exp_goals"][0],
                     o["exp_goals"][1], o["over"]["2.5"], o["btts"], o["scores"][0][0],
                     json.dumps(o, ensure_ascii=False, default=float)))
        con.commit()
        out.append(o)
        if not quiet:
            print(line(o))
    return out


def _one(P, m, rows, n):
    hk = P.resolver.fotmob(m.home_id, m.home, m.league)
    ak = P.resolver.fotmob(m.away_id, m.away, m.league)
    o = P.predict(m.league, hk, ak, n=n, seed=int(m.match_id) % 2**31, fm_home=m.home_id, fm_away=m.away_id,
                  match_id=m.match_id, home_coach=m.home_coach, away_coach=m.away_coach,
                  pre_rows=rows if len(rows) else None,
                  confirmed=(m.lineup_type not in (None, "predicted", "lastStarting11")),
                  home_name=m.home, away_name=m.away, referee=getattr(m, "referee", None),
                  date=m.utc_time[:10] if m.utc_time else None)
    o.update(match_id=int(m.match_id), home_name=m.home, away_name=m.away, utc_time=m.utc_time,
             lineup_type=m.lineup_type)
    return o


def line(o):
    sc = ", ".join(f"{s['name']} %{100 * s['p']:.0f}" for s in (o["scorers"][0][:2] + o["scorers"][1][:2]))
    return (f"{o['league']} {o['home_name']} - {o['away_name']}: 1 %{100 * o['p_home']:.0f}  X %{100 * o['p_draw']:.0f}  "
            f"2 %{100 * o['p_away']:.0f} | xG {o['exp_goals'][0]:.2f}-{o['exp_goals'][1]:.2f} | 2.5 ust %{100 * o['over']['2.5']:.0f}"
            f" | KG %{100 * o['btts']:.0f} | skor {o['scores'][0][0]} | {sc}")


def latest(match_id):
    con = sim_con()
    r = con.execute("SELECT data FROM predictions WHERE match_id = ? ORDER BY created_at DESC LIMIT 1", (match_id,)).fetchone()
    return json.loads(r[0]) if r else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default=date.today().strftime("%Y%m%d"))
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    if a.cmd == "show":
        print(json.dumps(latest(int(a.arg)), ensure_ascii=False, indent=1))
    else:
        predict_day(a.cmd, a.n, not a.no_fetch)
