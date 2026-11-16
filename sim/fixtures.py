"""Gunun maclari ve mac oncesi bilgiler (FotMob): muhtemel / aciklanan 11, eksikler, hoca.

Ana veritabanina yazmaz (kaziyicilarin 'bitmis mac' mantigini bozmasin diye); her sey
data/sim.sqlite'taki pre_match ve pre_lineup tablolarina gider.

  .venv/bin/python -m sim.fixtures 20261024
"""
import sys
from datetime import date, datetime, timezone

import pandas as pd

from config import LEAGUES
from src.net import get_json

from .common import sim_con

BASE = "https://www.fotmob.com/api/data"
FM_TO_LEAGUE = {v["fotmob"]: k for k, v in LEAGUES.items()}
FM_TO_LEAGUE[73] = "UEL"  # UEFA Avrupa Ligi: yalnizca tahmin (gecmis verisi yok, takimlar lig/Elo puaniyla)

SCHEMA = """
CREATE TABLE IF NOT EXISTS pre_match (
  match_id INTEGER PRIMARY KEY, league TEXT, utc_time TEXT, home_id INTEGER, home TEXT,
  away_id INTEGER, away TEXT, home_coach TEXT, away_coach TEXT, lineup_type TEXT,
  started INTEGER, finished INTEGER, fetched_at TEXT);
CREATE TABLE IF NOT EXISTS cup_results (
  match_id INTEGER PRIMARY KEY, league TEXT, utc_time TEXT, home_goals INTEGER, away_goals INTEGER);
CREATE TABLE IF NOT EXISTS pre_lineup (
  match_id INTEGER, team_id INTEGER, player_id INTEGER, name TEXT, role TEXT,
  position_id INTEGER, market_value INTEGER, note TEXT,
  PRIMARY KEY (match_id, team_id, player_id, role));
"""


def day_matches(ymd):
    d = get_json(f"{BASE}/matches?date={ymd}", "fotmob", use_cache=False, store=False) or {}
    out = []
    for lg in d.get("leagues", []):
        league = FM_TO_LEAGUE.get(lg.get("primaryId"))
        if not league:
            continue
        for m in lg["matches"]:
            out.append(dict(match_id=int(m["id"]), league=league, utc_time=(m.get("status") or {}).get("utcTime"),
                            home_id=int(m["home"]["id"]), home=m["home"]["name"],
                            away_id=int(m["away"]["id"]), away=m["away"]["name"]))
    return out


def record_results(ymd):
    """Ana veritabaninin kazimadigi kupa maclarinin (Avrupa Ligi) sonuclari: ogrenme dongusu icin."""
    con = sim_con()
    con.executescript(SCHEMA)
    d = get_json(f"{BASE}/matches?date={ymd}", "fotmob", use_cache=False, store=False) or {}
    n = 0
    for lg in d.get("leagues", []):
        league = FM_TO_LEAGUE.get(lg.get("primaryId"))
        if league not in ("UEL",):
            continue
        for m in lg["matches"]:
            st = m.get("status") or {}
            if st.get("finished") and not st.get("cancelled"):
                con.execute("INSERT OR REPLACE INTO cup_results VALUES (?,?,?,?,?)",
                            (int(m["id"]), league, st.get("utcTime"), m["home"].get("score"), m["away"].get("score")))
                n += 1
    con.commit()
    return n


def _ensure_cols(con):
    cols = {r[1] for r in con.execute("PRAGMA table_info(pre_match)")}
    if "referee" not in cols:
        con.execute("ALTER TABLE pre_match ADD COLUMN referee TEXT")
    if "weather" not in cols:  # hava durumu: gecmisi yok, ileride test edilebilsin diye kaydediliyor
        con.execute("ALTER TABLE pre_match ADD COLUMN weather TEXT")


def fetch_prematch(con, fx):
    md = get_json(f"{BASE}/matchDetails?matchId={fx['match_id']}", "fotmob", use_cache=False, store=False)
    if not md or not md.get("content"):
        return False
    lu = md["content"].get("lineup") or {}
    st = md["header"].get("status") or {}
    coaches = []
    con.execute("DELETE FROM pre_lineup WHERE match_id = ?", (fx["match_id"],))
    for k in ("homeTeam", "awayTeam"):
        t = lu.get(k) or {}
        coaches.append((t.get("coach") or {}).get("name"))
        for role in ("starters", "subs", "unavailable"):
            for p in t.get(role) or []:
                u = p.get("unavailability") or {}
                note = f"{u.get('type')}: {u.get('expectedReturn')}" if u else None
                con.execute("INSERT OR REPLACE INTO pre_lineup VALUES (?,?,?,?,?,?,?,?)",
                            (fx["match_id"], t.get("id"), p.get("id"), p.get("name"), role.rstrip("s"),
                             p.get("positionId"), p.get("marketValue"), note))
    con.execute("INSERT OR REPLACE INTO pre_match (match_id, league, utc_time, home_id, home, away_id, away, "
                "home_coach, away_coach, lineup_type, started, finished, fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (fx["match_id"], fx["league"], fx["utc_time"], fx["home_id"], fx["home"], fx["away_id"], fx["away"],
                 coaches[0], coaches[1], lu.get("lineupType"), int(bool(st.get("started"))),
                 int(bool(st.get("finished"))), datetime.now(timezone.utc).isoformat(timespec="seconds")))
    info = (md["content"].get("matchFacts") or {}).get("infoBox") or {}
    ref = (info.get("Referee") or {}).get("text")
    _ensure_cols(con)
    import json
    w = md["content"].get("weather")
    con.execute("UPDATE pre_match SET referee = ?, weather = ? WHERE match_id = ?",
                (ref, json.dumps(w, ensure_ascii=False) if w else None, fx["match_id"]))
    con.commit()
    return True


def update(ymd):
    con = sim_con()
    con.executescript(SCHEMA)
    fxs = day_matches(ymd)
    for fx in fxs:
        ok = fetch_prematch(con, fx)
        print(f"  {fx['league']} {fx['home']} - {fx['away']}: {'tamam' if ok else 'detay yok'}", flush=True)
    return fxs


def load(match_ids=None, ymd=None):
    con = sim_con()
    con.executescript(SCHEMA)
    _ensure_cols(con)
    pm = pd.read_sql("SELECT * FROM pre_match", con)
    if ymd:
        pm = pm[pm.utc_time.str[:10].str.replace("-", "") == ymd]
    if match_ids is not None:
        pm = pm[pm.match_id.isin(match_ids)]
    pl = pd.read_sql("SELECT * FROM pre_lineup", con)
    return pm, pl[pl.match_id.isin(pm.match_id)]


if __name__ == "__main__":
    update(sys.argv[1] if len(sys.argv) > 1 else date.today().strftime("%Y%m%d"))
