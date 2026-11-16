"""football-data.co.uk: guncel sezon sonuclari, acilis/kapanis oranlari, hakem; ve yaklasan fikstur oranlari.

Kullanim:  .venv/bin/python -m src.scrape_footballdata
Turkiye'den erisim engelli, sunucudan calisir. Takim adlari xgabora ile ayni
(ayni kaynak), bu yuzden team_map'teki xgabora eslesmeleri gecerlidir.
Tablolar: fd_matches (guncel sezon, her calismada bastan), fd_fixtures (yaklasan maclar).
"""
import io
import sqlite3

import pandas as pd
import requests

from config import DB_PATH, LEAGUES, SEASONS, USER_AGENT

BASE = "https://www.football-data.co.uk"
DIVS = {v["xgabora"]: k for k, v in LEAGUES.items() if v["xgabora"]}


def fetch_csv(url):
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.content.decode("utf-8-sig", errors="replace")))


def tidy(df):
    df = df[df.Div.isin(DIVS)].copy()
    df.insert(0, "league", df.Div.map(DIVS))
    df.insert(1, "date", pd.to_datetime(df.Date, dayfirst=True).dt.strftime("%Y-%m-%d"))
    df.columns = [c.lower() for c in df.columns]
    return df.loc[:, ~df.columns.duplicated()]


def main():
    y = SEASONS[-1]
    code = f"{y % 100:02d}{(y + 1) % 100:02d}"
    parts = []
    for div in DIVS:
        try:
            parts.append(fetch_csv(f"{BASE}/mmz4281/{code}/{div}.csv"))
        except Exception as e:  # tek lig dusse de digerleri yazilsin
            print(f"  ! {div}: {e}")
    con = sqlite3.connect(DB_PATH, timeout=60)
    if parts:
        m = tidy(pd.concat(parts, ignore_index=True))
        m.to_sql("fd_matches", con, if_exists="replace", index=False)
        print(f"fd_matches: {len(m)} mac, son tarih {m.date.max()}")
    try:
        f = tidy(fetch_csv(f"{BASE}/fixtures.csv"))
        f.to_sql("fd_fixtures", con, if_exists="replace", index=False)
        print(f"fd_fixtures: {len(f)} yaklasan mac")
    except Exception as e:
        print(f"  ! fixtures: {e}")
    con.commit()
    con.close()


if __name__ == "__main__":
    main()
