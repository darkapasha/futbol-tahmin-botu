"""Sabah tahmin isi (veri guncellemesinden sonra, systemd: futbol-predict.timer).

  .venv/bin/python -m bot.pipeline              # ogren + bugunun tahminleri + yorumlar
  .venv/bin/python -m bot.pipeline 20261024     # belirli gun
  .venv/bin/python -m bot.pipeline lineups      # mac saatine yakin aciklanan 11'lerle tahmini tazele
  .venv/bin/python -m bot.pipeline yorum 20261024   # o gunun yorumlarini yeniden uret

1. Dunun tahminlerini sonuclarla karsilastir; kalici hata / yeni oruntu varsa ayarla (sim.learn)
2. Gunun maclarini ve muhtemel 11'leri cek, her maci 10 bin kez simule et (sim.run)
3. Her mac icin yerel LLM ile taktik yorum (bot.commentary)
"""
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

from sim import fixtures, learn, run
from sim.common import sim_con

from . import commentary

TR = timezone(timedelta(hours=3))


def today_tr():
    return datetime.now(TR).strftime("%Y%m%d")


def morning(ymd=None):
    ymd = ymd or today_tr()
    print(f"== {datetime.now(TR):%Y-%m-%d %H:%M} sabah isi, gun {ymd}", flush=True)
    try:
        fixtures.record_results((datetime.strptime(ymd, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d"))
    except Exception as e:
        print("kupa sonuclari alinamadi:", e)
    try:
        learn.daily()
    except Exception as e:  # ogrenme hatasi tahmini durdurmasin
        print("ogrenme hatasi:", e)
    out = run.predict_day(ymd)
    # sonraki iki gunun maclari da onceden tahmin edilsin (/yarin, hafta sonu programi);
    # yorumlar yalnizca bugun icin uretilir, digerleri /mac istenince
    base = datetime.strptime(ymd, "%Y%m%d")
    for k in (1, 2):
        try:
            run.predict_day((base + timedelta(days=k)).strftime("%Y%m%d"))
        except Exception as e:
            print("ileri gun tahmin hatasi:", e)
    for o in out:
        commentary.get(o, datetime.now(timezone.utc).isoformat(timespec="seconds"))
        print("  yorum hazir:", o["home_name"], "-", o["away_name"], flush=True)
    return out


def lineups(window=(40, 90)):
    """Baslamasina 40-90 dk kalan maclar: aciklanan 11 ve maca yakin oranlarla tahmini bir kez yenile
    (kapanisa yakin oran acilistan daha isabetli). Degisen maclari dondurur."""
    con = sim_con()
    now = datetime.now(timezone.utc)
    pm = pd.read_sql("SELECT * FROM pre_match WHERE finished = 0", con)
    if pm.empty:
        return []
    pm["t"] = pd.to_datetime(pm.utc_time, utc=True)
    mins = (pm.t - now).dt.total_seconds() / 60
    due = pm[(mins >= window[0]) & (mins <= window[1])]
    changed = []
    from sim.match import Predictor
    P = None
    refreshed = False
    for m in due.itertuples():
        old = run.latest(m.match_id)
        last = con.execute("SELECT MAX(created_at) FROM predictions WHERE match_id = ?", (m.match_id,)).fetchone()[0]
        if last and pd.Timestamp(last) >= m.t - pd.Timedelta(minutes=window[1] + 5):
            continue  # bu pencerede zaten yenilendi
        fx = dict(match_id=m.match_id, league=m.league, utc_time=m.utc_time, home_id=m.home_id, home=m.home,
                  away_id=m.away_id, away=m.away)
        fixtures.fetch_prematch(con, fx)
        if not refreshed:
            try:
                from sim import market
                market.refresh_live()
            except Exception as e:
                print("canli oran alinamadi:", e)
            refreshed = True
        P = P or Predictor(pd.Timestamp(now.replace(tzinfo=None)))
        new = run.predict_day(m.utc_time[:10].replace("-", ""), fetch=False, P=P, quiet=True, match_ids=[m.match_id])
        if new and old:
            d = max(abs(new[0][k] - old[k]) for k in ("p_home", "p_draw", "p_away"))
            changed.append((new[0], old, d))
    return changed


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "yorum":
        from . import views
        for o in views.day_predictions(sys.argv[2] if len(sys.argv) > 2 else today_tr()):
            commentary.get(o, datetime.now(timezone.utc).isoformat(timespec="seconds"), force=True)
            print("  yorum:", o["home_name"], "-", o["away_name"], flush=True)
    elif arg == "lineups":
        for new, old, d in lineups():
            print(new["home_name"], new["away_name"], f"degisim {d:.3f}")
    else:
        morning(arg)
