"""Veri saglik kontrolu: kaynaklardan biri kesilirse tahminler sessizce bozulmasin, abonelere uyari gitsin.

  .venv/bin/python -m bot.health
"""
import subprocess
from datetime import datetime, timedelta, timezone

import pandas as pd

from config import ROOT
from sim.common import sim_con, src_con

TR = timezone(timedelta(hours=3))


def _unit_result(unit):
    try:
        out = subprocess.run(["systemctl", "show", unit, "-p", "Result", "-p", "ExecMainExitTimestamp"],
                             capture_output=True, text=True, timeout=10).stdout
        d = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
        return d.get("Result"), d.get("ExecMainExitTimestamp", "")
    except Exception:
        return None, ""


def _league_games_soon(now):
    try:
        pm = pd.read_sql("SELECT league, utc_time FROM pre_match", sim_con())
    except Exception:
        return True
    t = pd.to_datetime(pm.utc_time, utc=True)
    lo = pd.Timestamp(now).tz_convert("UTC")
    soon = pm[(t >= lo) & (t < lo + pd.Timedelta(days=1))]
    return bool((~soon.league.isin(["UCL", "UEL"])).any())


def _games_expected(last_fm, now):
    """Son sonuctan bu yana bitmis olmasi gereken bir mac var mi? Milli ara / kis arasi icin."""
    try:
        pm = pd.read_sql("SELECT utc_time FROM pre_match", sim_con())
    except Exception:
        return True
    t = pd.to_datetime(pm.utc_time, utc=True)
    lo = pd.Timestamp(last_fm, tz="UTC") if pd.Timestamp(last_fm).tzinfo is None else pd.Timestamp(last_fm)
    hi = pd.Timestamp(now).tz_convert("UTC") - pd.Timedelta(hours=3)
    return bool(((t > lo + pd.Timedelta(hours=3)) & (t < hi)).any())


def check():
    """Sorun listesi (bos = her sey yolunda)."""
    problems = []
    now = datetime.now(TR)
    for unit, name in (("futbol-daily.service", "sabah veri guncellemesi"), ("futbol-predict.service", "sabah tahmin isi")):
        res, ts = _unit_result(unit)
        if res and res != "success":
            problems.append(f"{name} hata ile bitti ({res})")
    con = src_con()
    # FotMob: sezon icinde son 4 gunde hic bitmis mac gelmediyse
    last_fm = con.execute("SELECT MAX(utc_time) FROM fm_matches WHERE home_goals IS NOT NULL").fetchone()[0]
    if last_fm and pd.Timestamp(last_fm[:10]) < pd.Timestamp(now.date()) - pd.Timedelta(days=4) and now.month not in (6, 7)             and _games_expected(last_fm, now):
        problems.append(f"FotMob'dan 4 gündür sonuç gelmiyor (son: {last_fm[:10]})")
    # football-data oranlari
    try:
        n, last = con.execute("SELECT COUNT(*), MAX(date) FROM fd_fixtures").fetchone()
        # Hafta ici yalniz UCL/UEL oynandiginda liste dogal olarak bos; sadece
        # onumuzdeki 24 saatte lig maci varsa uyar.
        if not n and _league_games_soon(now):
            problems.append("football-data oran tablosu boş")
    except Exception:
        problems.append("football-data oran tablosu okunamadı")
    # Understat (ilk 5 lig xG): son guncelleme
    try:
        last_us = con.execute("SELECT MAX(date) FROM us_team_games").fetchone()[0]
        if last_us and pd.Timestamp(last_us[:10]) < pd.Timestamp(now.date()) - pd.Timedelta(days=10) and now.month not in (6, 7):
            problems.append(f"Understat xG 10 gündür güncellenmedi (son: {last_us[:10]})")
    except Exception:
        pass
    # bugun icin tahmin uretildi mi (mac varsa)
    sc = sim_con()
    try:
        pm = pd.read_sql("SELECT match_id, utc_time FROM pre_match", sc)
        pm["day"] = pd.to_datetime(pm.utc_time, utc=True).dt.tz_convert(TR).dt.strftime("%Y%m%d")
        today = pm[pm.day == now.strftime("%Y%m%d")]
        if len(today):
            pr = pd.read_sql("SELECT DISTINCT match_id FROM predictions", sc)
            missing = set(today.match_id) - set(pr.match_id)
            if missing:
                problems.append(f"bugünkü {len(missing)} maç için tahmin üretilemedi")
    except Exception as e:
        problems.append(f"tahmin tablosu okunamadı: {e}")
    log = ROOT / "logs" / "predict.log"
    if log.exists():
        tail = log.read_text(errors="ignore").splitlines()[-200:]
        errs = [l for l in tail if l.strip().startswith("!") or "Traceback" in l]
        if errs:
            problems.append(f"sabah işinde {len(errs)} hata satırı (örn: {errs[-1].strip()[:80]})")
    return problems


if __name__ == "__main__":
    p = check()
    print("sorun yok" if not p else "\n".join(p))
