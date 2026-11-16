"""Profil modulu ortak yardimcilar: veritabani baglantilari, dizilis ve pozisyon cozumleme.

Ana veritabanina (data/futbol.sqlite) yalnizca okuma modunda baglanir; 1. adimin
kaziyicilari ayni anda yazarken kilitlenme olmasin diye profiller ayri bir dosyaya
(data/profiles.sqlite) yazilir. Iki dosya ATTACH ile birlikte sorgulanabilir.
"""
import re
import sqlite3

from config import DATA, DB_PATH

PROFILE_DB = DATA / "profiles.sqlite"


def src_con():
    return sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=60)


def out_con():
    return sqlite3.connect(PROFILE_DB, timeout=60)


def parse_formation(f):
    """'4-3-3 Attacking' -> (defans, orta saha, forvet, '4-3-3'). Bilinmiyorsa Nones."""
    if not isinstance(f, str):
        return None, None, None, None
    nums = [int(x) for x in re.findall(r"\d", f.split()[0])]
    if len(nums) < 3 or sum(nums) != 10:
        return None, None, None, None
    return nums[0], sum(nums[1:-1]), nums[-1], "-".join(map(str, nums))


# Transfermarkt pozisyonlari: kanat / merkez ve hat (1 defans, 2 orta, 3 hucum)
WIDE_ATT = {"Left Winger", "Right Winger", "Left Midfield", "Right Midfield"}
FULLBACK = {"Left-Back", "Right-Back"}
CENTRAL = {"Centre-Back", "Defensive Midfield", "Central Midfield", "Attacking Midfield",
           "Centre-Forward", "Second Striker", "Sweeper"}
LINE = {"Defender": 1, "Midfield": 2, "Attack": 3, "Goalkeeper": 0}

# Gol aciklamasi ("..., Header, 3. Goal of the Season Assist: , Corner, ...") -> tur
SET_PIECE_ASSIST = {"Corner", "Free kick", "Throw-in", "Goal-kick"}
SET_PIECE_SHOT = {"Direct free kick", "Direct corner"}


def goal_kind(desc):
    d = desc or ""
    shot, _, assist = d.partition("Assist:")
    s = shot.split(",")[1].strip() if "," in shot else ""
    a = assist.split(",")[1].strip() if assist.count(",") else assist.strip()
    pen = s in ("Penalty", "Penalty rebound")
    return {
        "own": s == "Own-goal",
        "pen": pen,
        "header": s == "Header",
        "counter": s == "Counter attack goal",
        "setpiece": (s in SET_PIECE_SHOT or a in SET_PIECE_ASSIST) and not pen,
        "cross": a == "Cross",
        "longshot": s == "Long distance kick",
    }
