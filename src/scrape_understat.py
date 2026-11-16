"""Understat: ilk 5 lig icin mac bazli xG ve takim bazli mac istatistikleri.

Kullanim:  .venv\\Scripts\\python -m src.scrape_understat [--fresh]
Lig-sezon basina tek istek (toplam ~30). Bitmis sezonlar onbellekten gelir;
--fresh yalnizca icinde bulunulan sezonu yeniden ceker.
"""
import json
import sqlite3
import sys

import pandas as pd

from config import DB_PATH, LEAGUES, SEASONS
from src.net import _cache_path, get_json


def fetch(code, season, fresh):
    url = f"https://understat.com/getLeagueData/{code}/{season}"
    headers = {
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"https://understat.com/league/{code}/{season}",
    }
    data = get_json(url, "understat", headers=headers, use_cache=not fresh)
    if data is None and fresh:
        # Site kapali/bakimdaysa son basarili cekimi kullan; yoksa tablo yeniden
        # yazilirken guncel sezonun xG'si tamamen silinir.
        path = _cache_path("understat", url)
        if path.exists():
            print(f"  {code} {season}: site yanit vermedi, son kayitli veri kullaniliyor")
            data = json.loads(path.read_text(encoding="utf-8"))
    return data


def main(fresh=None):
    if fresh is None:
        fresh = "--fresh" in sys.argv
    matches, team_games = [], []
    for league, cfg in LEAGUES.items():
        code = cfg["understat"]
        if not code:
            continue
        for season in SEASONS:
            data = fetch(code, season, fresh and season == SEASONS[-1])
            if not data:
                print(f"  {league} {season}: veri yok")
                continue
            for m in data["dates"]:
                matches.append({
                    "league": league, "season": season, "us_id": int(m["id"]),
                    "datetime": m["datetime"], "is_result": m["isResult"],
                    "home": m["h"]["title"], "away": m["a"]["title"],
                    "home_goals": m["goals"]["h"], "away_goals": m["goals"]["a"],
                    "home_xg": m["xG"]["h"], "away_xg": m["xG"]["a"],
                    "fc_home": (m.get("forecast") or {}).get("w"),
                    "fc_draw": (m.get("forecast") or {}).get("d"),
                    "fc_away": (m.get("forecast") or {}).get("l"),
                })
            for t in data["teams"].values():
                for h in t["history"]:
                    team_games.append({
                        "league": league, "season": season, "team": t["title"],
                        "date": h["date"], "h_a": h["h_a"],
                        "xg": h["xG"], "xga": h["xGA"], "npxg": h["npxG"], "npxga": h["npxGA"],
                        "ppda_att": h["ppda"]["att"], "ppda_def": h["ppda"]["def"],
                        "oppda_att": h["ppda_allowed"]["att"], "oppda_def": h["ppda_allowed"]["def"],
                        "deep": h["deep"], "deep_allowed": h["deep_allowed"],
                        "scored": h["scored"], "missed": h["missed"],
                        "xpts": h["xpts"], "pts": h["pts"],
                    })
            print(f"  {league} {season}: {len(data['dates'])} mac")

    con = sqlite3.connect(DB_PATH, timeout=60)
    pd.DataFrame(matches).to_sql("us_matches", con, if_exists="replace", index=False)
    pd.DataFrame(team_games).to_sql("us_team_games", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS ix_us_matches_date ON us_matches(datetime)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_us_team_games ON us_team_games(team, date)")
    con.commit()
    con.close()
    print(f"us_matches: {len(matches):,}  us_team_games: {len(team_games):,}")


if __name__ == "__main__":
    main()
