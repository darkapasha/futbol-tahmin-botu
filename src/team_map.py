"""Takim adlarini kaynaklar arasinda eslestirir (ortak anahtar: Transfermarkt club_id).

Isimleri metinden tahmin etmek yerine maclari eslestirir: ayni lig, ayni gun
(+-1), ayni skor tek bir Transfermarkt maciyla eslesiyorsa iki takim da oy alir.
Her kaynak ismi en cok oy aldigi club_id'ye baglanir.

Kullanim:  .venv\\Scripts\\python -m src.team_map
Sonuc: team_map tablosu (source, name, team_key, tm_club_id, tm_name, votes, share)
team_key = "tm:<club_id>", baglanamayan FotMob takimlari icin "fm:<id>". Transfermarkt verisi bittikten sonra (Temmuz 2026)
lige gelen takimlar xgabora'ya ayni yontemle baglanir ve "x:<ad>" anahtari alir.
"""
import sqlite3

import pandas as pd

from config import DB_PATH


def load(con):
    tm = pd.read_sql("SELECT league, date, home_club_id h_id, away_club_id a_id, "
                     "home_club_goals hg, away_club_goals ag FROM tm_games", con)
    src = {
        "xgabora": pd.read_sql("SELECT league, matchdate date, hometeam home, awayteam away, "
                               "fthome hg, ftaway ag FROM xg_matches", con),
        "understat": pd.read_sql("SELECT league, substr(datetime,1,10) date, home, away, "
                                 "home_goals hg, away_goals ag FROM us_matches WHERE is_result", con),
        "fotmob": pd.read_sql("SELECT league, substr(utc_time,1,10) date, CAST(home_id AS TEXT) home, "
                              "CAST(away_id AS TEXT) away, home_goals hg, away_goals ag FROM fm_matches", con),
    }
    return tm, src


def votes_for(tm, df):
    df = df.dropna(subset=["hg", "ag"]).copy()
    df["hg"], df["ag"] = df.hg.astype(int), df.ag.astype(int)
    df["d"] = pd.to_datetime(df.date)
    t = tm.copy()
    t["d"] = pd.to_datetime(t.date)
    rows = []
    for shift in (-1, 0, 1):
        x = df.assign(d=df.d + pd.Timedelta(days=shift))
        rows.append(x.merge(t, on=["league", "d", "hg", "ag"], suffixes=("", "_tm")))
    m = pd.concat(rows)
    # Ayni kaynak macina birden fazla aday dusuyorsa belirsizdir, at
    key = ["league", "date", "home", "away"]
    m = m[m.groupby(key).h_id.transform("nunique") == 1]
    v = pd.concat([m[["home", "h_id"]].set_axis(["name", "tm_club_id"], axis=1),
                   m[["away", "a_id"]].set_axis(["name", "tm_club_id"], axis=1)])
    c = v.value_counts().rename("votes").reset_index()
    c["share"] = c.votes / c.groupby("name").votes.transform("sum")
    return c.sort_values("votes", ascending=False).drop_duplicates("name")


def main():
    con = sqlite3.connect(DB_PATH, timeout=60)
    tm, src = load(con)
    clubs = pd.read_sql("SELECT club_id tm_club_id, name tm_name FROM tm_clubs", con)
    tm = tm.assign(h_id="tm:" + tm.h_id.astype(str), a_id="tm:" + tm.a_id.astype(str))

    # 1. tur: her kaynak Transfermarkt'a
    maps = {}
    for name, df in src.items():
        c = votes_for(tm, df).rename(columns={"tm_club_id": "team_key"})
        maps[name] = c

    # 2. tur: yeni takimlar. xgabora'da eslesmeyenler kendi anahtarini alir,
    # diger kaynaklarin eslesmeyenleri xgabora uzerinden baglanir.
    xg = src["xgabora"]
    xkey = dict(zip(maps["xgabora"].name, maps["xgabora"].team_key))
    for n in (set(xg.home) | set(xg.away)) - set(xkey):
        xkey[n] = f"x:{n}"
        maps["xgabora"] = pd.concat([maps["xgabora"], pd.DataFrame(
            [{"name": n, "team_key": xkey[n], "votes": 0, "share": 1.0}])])
    anchor = xg.assign(h_id=xg.home.map(xkey), a_id=xg.away.map(xkey))[["league", "date", "hg", "ag", "h_id", "a_id"]]
    for name in ("understat", "fotmob"):
        df = src[name]
        left = df[~df.home.isin(maps[name].name) | ~df.away.isin(maps[name].name)]
        if len(left):
            c = votes_for(anchor.dropna(), left).rename(columns={"tm_club_id": "team_key"})
            c = c[~c.name.isin(maps[name].name)]
            maps[name] = pd.concat([maps[name], c])

    # 3. tur: hicbir yere baglanamayan FotMob takimlari (orn. ilk kez SL'ye
    # gelen yabanci kulupler) kendi kimligiyle "fm:<id>" anahtarini alir.
    fm = src["fotmob"]
    for n in (set(fm.home) | set(fm.away)) - set(maps["fotmob"].name):
        maps["fotmob"] = pd.concat([maps["fotmob"], pd.DataFrame(
            [{"name": n, "team_key": f"fm:{n}", "votes": 0, "share": 1.0}])])

    out = []
    for name, c in maps.items():
        c = c.copy()
        c["tm_club_id"] = c.team_key.where(c.team_key.str.startswith("tm:")).str[3:].astype("Int64")
        c = c.merge(clubs, on="tm_club_id", how="left")
        c.insert(0, "source", name)
        all_names = set(src[name].home) | set(src[name].away)
        weak = c[(c.votes < 3) | (c.share < 0.8)]
        print(f"{name}: {len(c)}/{len(all_names)} takim eslesti, {len(weak)} az kanitli")
        out.append(c)
    res = pd.concat(out)[["source", "name", "team_key", "tm_club_id", "tm_name", "votes", "share"]]
    res.to_sql("team_map", con, if_exists="replace", index=False)
    con.commit()
    con.close()


if __name__ == "__main__":
    main()
