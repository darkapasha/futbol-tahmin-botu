"""FotMob: 7 hedef lig icin xG, istatistik, kadro, hoca, hakem, sut haritasi.

Kullanim:
  .venv\\Scripts\\python -m src.scrape_fotmob history [TUR UCL ...]   # gecmis sezonlar (yavas, devam ettirilebilir); --current: yalnizca bu sezon
  .venv\\Scripts\\python -m src.scrape_fotmob day [YYYYMMDD]          # o gunun fiksturu (sabah isi)

Her mac kendi islemiyle yazilir; yarida kesilirse kaldigi yerden devam eder.
"""
import sqlite3
import sys
from datetime import date

from config import DB_PATH, LEAGUES, SEASONS
from src.net import get_json

BASE = "https://www.fotmob.com/api/data"
FM_TO_LEAGUE = {v["fotmob"]: k for k, v in LEAGUES.items()}

SCHEMA = """
CREATE TABLE IF NOT EXISTS fm_fixtures (
  match_id INTEGER PRIMARY KEY, league TEXT, season TEXT, round TEXT, utc_time TEXT,
  home_id INTEGER, home TEXT, away_id INTEGER, away TEXT,
  finished INTEGER, cancelled INTEGER, score TEXT);
CREATE TABLE IF NOT EXISTS fm_matches (
  match_id INTEGER PRIMARY KEY, league TEXT, season TEXT, utc_time TEXT,
  home_id INTEGER, home TEXT, away_id INTEGER, away TEXT, home_goals INTEGER, away_goals INTEGER,
  home_xg REAL, away_xg REAL, referee TEXT, referee_id INTEGER, stadium TEXT, attendance INTEGER,
  home_formation TEXT, away_formation TEXT, home_coach TEXT, home_coach_id INTEGER,
  away_coach TEXT, away_coach_id INTEGER);
CREATE TABLE IF NOT EXISTS fm_stats (
  match_id INTEGER, period TEXT, key TEXT, title TEXT, home TEXT, away TEXT,
  PRIMARY KEY (match_id, period, key));
CREATE TABLE IF NOT EXISTS fm_lineups (
  match_id INTEGER, team_id INTEGER, player_id INTEGER, name TEXT, role TEXT,
  position_id INTEGER, shirt TEXT, x REAL, y REAL, rating REAL, market_value INTEGER,
  PRIMARY KEY (match_id, player_id));
CREATE TABLE IF NOT EXISTS fm_unavailable (
  match_id INTEGER, team_id INTEGER, player_id INTEGER, name TEXT, reason TEXT, expected_return TEXT);
CREATE TABLE IF NOT EXISTS fm_shots (
  shot_id INTEGER PRIMARY KEY, match_id INTEGER, team_id INTEGER, player_id INTEGER, player TEXT,
  minute INTEGER, x REAL, y REAL, xg REAL, xgot REAL, event_type TEXT, shot_type TEXT,
  situation TEXT, period TEXT, on_target INTEGER, blocked INTEGER, own_goal INTEGER);
CREATE TABLE IF NOT EXISTS fm_failed (match_id INTEGER PRIMARY KEY, tries INTEGER);
CREATE INDEX IF NOT EXISTS ix_fm_matches_date ON fm_matches(utc_time);
CREATE INDEX IF NOT EXISTS ix_fm_fixtures_date ON fm_fixtures(utc_time);
CREATE INDEX IF NOT EXISTS ix_fm_shots_match ON fm_shots(match_id);
"""


def connect():
    con = sqlite3.connect(DB_PATH, timeout=60)
    con.executescript(SCHEMA)
    return con


def season_str(year):
    return f"{year}/{year + 1}"


def save_fixture(con, league, season, m):
    st = m["status"]
    con.execute(
        "INSERT OR REPLACE INTO fm_fixtures VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (int(m["id"]), league, season, str(m.get("round", "")), st.get("utcTime"),
         int(m["home"]["id"]), m["home"]["name"], int(m["away"]["id"]), m["away"]["name"],
         int(bool(st.get("finished"))), int(bool(st.get("cancelled"))), st.get("scoreStr")),
    )


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def save_details(con, league, season, md):
    mid = int(md["general"]["matchId"])
    c = md["content"]
    teams = md["header"]["teams"]

    # Istatistikler (uzun bicim: her satir bir metrik)
    xg = [None, None]
    periods = ((c.get("stats") or {}).get("Periods")) or {}
    for period, block in periods.items():
        for group in (block or {}).get("stats") or []:
            for s in group.get("stats", []):
                key, vals = s.get("key"), s.get("stats") or [None, None]
                if not key or len(vals) != 2:
                    continue
                con.execute("INSERT OR REPLACE INTO fm_stats VALUES (?,?,?,?,?,?)",
                            (mid, period, key, s.get("title"), str(vals[0]), str(vals[1])))
                if period == "All" and key == "expected_goals" and vals[0] is not None:
                    xg = [_num(vals[0]), _num(vals[1])]

    # Kadro, dizilis, hoca, eksikler
    lu = c.get("lineup") or {}
    side = {}
    for k in ("homeTeam", "awayTeam"):
        t = lu.get(k) or {}
        side[k] = t
        for role in ("starters", "subs"):
            for p in t.get(role) or []:
                lay = p.get("horizontalLayout") or {}
                con.execute(
                    "INSERT OR REPLACE INTO fm_lineups VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (mid, t.get("id"), p.get("id"), p.get("name"), role[:-1], p.get("positionId"),
                     p.get("shirtNumber"), lay.get("x"), lay.get("y"),
                     (p.get("performance") or {}).get("rating"), p.get("marketValue")))
        for p in t.get("unavailable") or []:
            u = p.get("unavailability") or {}
            con.execute("INSERT INTO fm_unavailable VALUES (?,?,?,?,?,?)",
                        (mid, t.get("id"), p.get("id"), p.get("name"), u.get("type"), u.get("expectedReturn")))

    # Sut haritasi
    for s in (c.get("shotmap") or {}).get("shots") or []:
        con.execute(
            "INSERT OR REPLACE INTO fm_shots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (s["id"], mid, s.get("teamId"), s.get("playerId"), s.get("playerName"), s.get("min"),
             s.get("x"), s.get("y"), s.get("expectedGoals"), s.get("expectedGoalsOnTarget"),
             s.get("eventType"), s.get("shotType"), s.get("situation"), s.get("period"),
             int(bool(s.get("isOnTarget"))), int(bool(s.get("isBlocked"))), int(bool(s.get("isOwnGoal")))))

    info = (c.get("matchFacts") or {}).get("infoBox") or {}
    ref = info.get("Referee") or {}
    stadium = info.get("Stadium") or {}
    home, away = side["homeTeam"], side["awayTeam"]
    hc, ac = home.get("coach") or {}, away.get("coach") or {}
    con.execute(
        "INSERT OR REPLACE INTO fm_matches VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (mid, league, season, md["general"].get("matchTimeUTCDate"),
         teams[0]["id"], teams[0]["name"], teams[1]["id"], teams[1]["name"],
         teams[0].get("score"), teams[1].get("score"), xg[0], xg[1],
         ref.get("text"), ref.get("id"), stadium.get("name"), info.get("Attendance"),
         home.get("formation"), away.get("formation"),
         hc.get("name"), hc.get("id"), ac.get("name"), ac.get("id")))


def fetch_details(con, league, season, match_id):
    md = get_json(f"{BASE}/matchDetails?matchId={match_id}", "fotmob", use_cache=False, store=False)
    if not md or not md.get("content"):
        print(f"  ! {match_id}: detay alinamadi")
        con.execute("INSERT INTO fm_failed VALUES (?, 1) "
                    "ON CONFLICT(match_id) DO UPDATE SET tries = tries + 1", (match_id,))
        con.commit()
        return False
    try:
        save_details(con, league, season, md)
    except Exception as e:  # tek bozuk mac butun cekimi durdurmasin
        con.rollback()
        print(f"  ! {match_id}: ayrıştırılamadı ({type(e).__name__}: {e})")
        return False
    con.commit()
    return True


def history(leagues, seasons=SEASONS):
    con = connect()
    for league in leagues:
        fm_id = LEAGUES[league]["fotmob"]
        for year in seasons:
            season = season_str(year)
            current = year == SEASONS[-1]
            data = get_json(f"{BASE}/leagues?id={fm_id}&season={season.replace('/', '%2F')}",
                            "fotmob", use_cache=not current)
            if not data:
                print(f"  {league} {season}: fikstur yok")
                continue
            ms = data["fixtures"]["allMatches"]
            for m in ms:
                save_fixture(con, league, season, m)
            con.commit()
            # Her seferinde veritabanina bakilir: ayni anda calisan baska bir
            # cekim (orn. guncel sezon doldurma) ayni maci tekrar cekmesin.
            done = {r[0] for r in con.execute("SELECT match_id FROM fm_matches WHERE league=?", (league,))}
            # FotMob'da detayi olmayan maclar (orn. erken eleme turlari) 3 denemeden sonra birakilir
            done |= {r[0] for r in con.execute("SELECT match_id FROM fm_failed WHERE tries >= 3")}
            todo = [m for m in ms if m["status"].get("finished") and int(m["id"]) not in done]
            print(f"  {league} {season}: {len(ms)} mac, {len(todo)} detay cekilecek")
            for i, m in enumerate(todo, 1):
                if not con.execute("SELECT 1 FROM fm_matches WHERE match_id=?", (int(m["id"]),)).fetchone():
                    fetch_details(con, league, season, int(m["id"]))
                if i % 25 == 0:
                    print(f"    {i}/{len(todo)}")
    con.close()


def day(ymd):
    """O gunun maclarini fikstura yazar; bitmis olanlarin detayini da ceker."""
    con = connect()
    y, mo = int(ymd[:4]), int(ymd[4:6])
    season = season_str(y if mo >= 7 else y - 1)
    data = get_json(f"{BASE}/matches?date={ymd}", "fotmob", use_cache=False, store=False)
    found = 0
    for lg in (data or {}).get("leagues", []):
        league = FM_TO_LEAGUE.get(lg.get("primaryId"))
        if not league:
            continue
        for m in lg["matches"]:
            found += 1
            st = m.get("status") or {}
            m2 = {"id": m["id"], "round": "", "home": m["home"], "away": m["away"], "status": st}
            save_fixture(con, league, season, m2)
            done = con.execute("SELECT 1 FROM fm_matches WHERE match_id=?", (m["id"],)).fetchone()
            if st.get("finished") and not done:
                fetch_details(con, league, season, m["id"])
    con.commit()
    con.close()
    print(f"{ymd}: hedef liglerde {found} mac")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "day"
    if mode == "history":
        args = sys.argv[2:]
        seasons = SEASONS
        if "--current" in args:  # yalnizca icinde bulunulan sezon
            args.remove("--current")
            seasons = SEASONS[-1:]
        history(args or ["TUR", "UCL", "ENG", "ESP", "GER", "ITA", "FRA"], seasons)
    else:
        day(sys.argv[2] if len(sys.argv) > 2 else date.today().strftime("%Y%m%d"))
