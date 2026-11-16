"""Indirilmis hazir veri setlerini SQLite'a yukler.

Kullanim:  .venv\\Scripts\\python -m src.load_static
Tablolar her calismada bastan yazilir (idempotent).
"""
import sqlite3

import pandas as pd

from config import DB_PATH, LEAGUES, RAW, START_DATE

XG_DIVS = {v["xgabora"]: k for k, v in LEAGUES.items() if v["xgabora"]}
TM_COMPS = {v["tm"]: k for k, v in LEAGUES.items() if v["tm"]}


def write(con, df, table, indexes=()):
    df.to_sql(table, con, if_exists="replace", index=False, chunksize=20000)
    for col in indexes:
        con.execute(f"CREATE INDEX IF NOT EXISTS ix_{table}_{col} ON {table}({col})")
    print(f"  {table}: {len(df):,} satir")


def load_xgabora(con):
    m = pd.read_csv(RAW / "xgabora" / "Matches.csv", low_memory=False)
    m = m[(m.MatchDate >= START_DATE) & m.Division.isin(XG_DIVS)].copy()
    m.insert(0, "league", m.Division.map(XG_DIVS))
    m.columns = [c.lower() for c in m.columns]
    write(con, m, "xg_matches", ["league", "matchdate", "hometeam", "awayteam"])

    elo = pd.read_csv(RAW / "xgabora" / "EloRatings.csv")
    elo = elo[elo.date >= "2020-07-01"]
    write(con, elo, "elo", ["club", "date"])


def load_transfermarkt(con):
    d = RAW / "transfermarkt"

    games = pd.read_csv(d / "games.csv", low_memory=False)
    games = games[games.competition_id.isin(TM_COMPS) & (games.date >= START_DATE)].copy()
    games.insert(0, "league", games.competition_id.map(TM_COMPS))
    write(con, games, "tm_games", ["league", "date", "home_club_id", "away_club_id"])
    ids = set(games.game_id)

    def filtered(name, key="game_id", usecols=None):
        parts = []
        for chunk in pd.read_csv(d / f"{name}.csv", chunksize=500_000, low_memory=False, usecols=usecols):
            parts.append(chunk[chunk[key].isin(ids)])
        return pd.concat(parts, ignore_index=True)

    write(con, filtered("club_games"), "tm_club_games", ["game_id", "club_id"])
    write(con, filtered("game_lineups"), "tm_lineups", ["game_id", "player_id", "club_id"])
    write(con, filtered("game_events"), "tm_events", ["game_id", "player_id"])
    write(con, filtered("appearances"), "tm_appearances", ["game_id", "player_id"])

    write(con, pd.read_csv(d / "clubs.csv"), "tm_clubs", ["club_id"])
    write(con, pd.read_csv(d / "competitions.csv"), "tm_competitions")
    write(con, pd.read_csv(d / "players.csv", low_memory=False), "tm_players", ["player_id", "current_club_id"])

    vals = pd.read_csv(d / "player_valuations.csv")
    write(con, vals[vals.date >= "2020-07-01"], "tm_valuations", ["player_id", "date"])
    tr = pd.read_csv(d / "transfers.csv")
    write(con, tr[tr.transfer_date >= "2020-07-01"], "tm_transfers", ["player_id"])


def load_injuries(con):
    inj = pd.read_csv(RAW / "salimt" / "player_injuries.csv")
    inj = inj[inj.from_date >= "2020-07-01"]
    write(con, inj, "injuries", ["player_id", "from_date"])


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=60)
    print("xgabora (sonuc, sut, oran, Elo)")
    load_xgabora(con)
    print("transfermarkt (kadro, hoca, hakem, dizilis)")
    load_transfermarkt(con)
    print("sakatliklar")
    load_injuries(con)
    con.commit()
    con.close()


if __name__ == "__main__":
    main()
