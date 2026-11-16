"""StatsBomb acik veri: taktik kalibrasyonu icin olay verisi (ihtiyac olunca).

Tum depoyu (~5 GB) indirmek yerine istenen yarisma-sezonun maclarini ve
olaylarini GitHub'dan tek tek ceker, data/cache/statsbomb altinda saklar.

Kullanim:
  .venv\\Scripts\\python -m src.statsbomb list                 # mevcut yarisma-sezonlar
  .venv\\Scripts\\python -m src.statsbomb fetch <comp> <season> # o sezonun tum olaylari
"""
import sys

from src.net import get_json

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def competitions():
    return get_json(f"{BASE}/competitions.json", "statsbomb")


def matches(comp_id, season_id):
    return get_json(f"{BASE}/matches/{comp_id}/{season_id}.json", "statsbomb")


def events(match_id):
    return get_json(f"{BASE}/events/{match_id}.json", "statsbomb")


if __name__ == "__main__":
    if sys.argv[1:2] == ["list"]:
        for c in competitions():
            print(c["competition_id"], c["season_id"], c["competition_name"], c["season_name"])
    elif sys.argv[1:2] == ["fetch"]:
        ms = matches(int(sys.argv[2]), int(sys.argv[3]))
        for i, m in enumerate(ms, 1):
            events(m["match_id"])
            print(f"{i}/{len(ms)} {m['home_team']['home_team_name']} - {m['away_team']['away_team_name']}")
