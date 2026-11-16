"""Simulasyon motoru ortak yardimcilar: baglantilar, parametre dosyasi, takim cozumleme.

Ana veritabani ve profiller yalnizca okunur. Motorun kendi urettikleri (kalibrasyon
tablolari, tahminler) data/sim.sqlite'a, global parametreler sim/params.json'a yazilir.
"""
import difflib
import json
import re
import sqlite3
import unicodedata
from pathlib import Path

import pandas as pd

from config import DATA, DB_PATH

SIM_DB = DATA / "sim.sqlite"
CUPS = ("UCL", "UEL")  # Avrupa kupalari: takimin ligi sayilmaz
PROFILE_DB = DATA / "profiles.sqlite"
PARAMS = Path(__file__).resolve().parent / "params.json"


def ro(path):
    return sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True, timeout=60)


def src_con():
    return ro(DB_PATH)


def prof_con():
    con = ro(PROFILE_DB)
    con.execute(f"ATTACH 'file:{DB_PATH.as_posix()}?mode=ro' AS src")
    return con


def sim_con():
    return sqlite3.connect(SIM_DB, timeout=60)


def load_params():
    return json.loads(PARAMS.read_text()) if PARAMS.exists() else {}


def save_params(p):
    PARAMS.write_text(json.dumps(p, indent=1, ensure_ascii=False, default=float))


# ---------------------------------------------------------------- takim adi cozumleme
_DROP = {"fc", "afc", "cf", "sc", "sk", "jk", "ac", "as", "ss", "ssc", "us", "cd", "ud", "rc", "rcd",
         "club", "de", "futbol", "football", "calcio", "spor", "kulubu", "sad", "the", "1", "fk",
         "vfl", "vfb", "tsg", "sv", "fsv", "bc", "1899", "1846", "1900", "1904", "04", "05", "09", "s.p.a."}


def norm(name):
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = [t for t in s.split() if t not in _DROP]
    return " ".join(toks) or s.strip()


def _score(a, b):
    if a == b:
        return 1.0
    r = difflib.SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    if ta and tb and (ta <= tb or tb <= ta):
        r = max(r, 0.85)
    return r


class TeamResolver:
    """FotMob takim kimligi / herhangi bir kaynak adi -> team_key.

    Once team_map'e (mac eslestirmesiyle kurulmus) bakar; FotMob'un yeni sezon maclari henuz
    eslesmediyse ayni ligde son 2 sezonda oynamis takimlarin adlarina benzerlikle baglar."""

    def __init__(self, con=None):
        con = con or src_con()
        self.map = pd.read_sql("SELECT source, name, team_key FROM team_map", con)
        self.by_src = {(r.source, r.name): r.team_key for r in self.map.itertuples()}
        pc = ro(PROFILE_DB)
        tm = pd.read_sql("SELECT DISTINCT league, team_key FROM team_match WHERE date >= date('now','-730 days') "
                         "AND team_key NOT LIKE 'fm:%'", pc)
        names = self.map[~self.map.team_key.str.startswith("fm:")][["name", "team_key"]]
        clubs = pd.read_sql("SELECT 'tm:' || club_id team_key, name FROM tm_clubs", con)
        allnames = pd.concat([names, clubs]).drop_duplicates()
        allnames = allnames[~allnames.name.str.fullmatch(r"\d+")]
        allnames["n"] = allnames.name.map(norm)
        self.cands = allnames.merge(tm, on="team_key")
        self.cands_all = allnames

    def fotmob(self, team_id, name, league=None):
        k = self.by_src.get(("fotmob", str(team_id)))
        if k and not k.startswith("fm:"):
            return k
        return self.by_name(name, league) or f"fm:{team_id}"

    def by_name(self, name, league=None):
        c = self.cands
        if league is not None and league not in CUPS:
            c = c[c.league == league]
        n = norm(name)
        best = c.assign(s=c.n.map(lambda x: _score(n, x))).sort_values("s", ascending=False)
        if best.empty or best.s.iat[0] < 0.78:
            return None
        top = best[best.s >= best.s.iat[0] - 0.02].team_key.unique()
        return top[0] if len(top) == 1 else None
