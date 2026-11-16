"""Gunluk ogrenme dongusu: dunun tahminlerini gercek sonuclarla karsilastirir, yalnizca
SUREKLI ve ANLAMLI bir hata ya da yeni bir oruntu varsa ayar yapar.

  .venv/bin/python -m sim.learn            # degerlendir + gerekiyorsa ayarla (sabah isi)
  .venv/bin/python -m sim.learn report     # son 30 / 90 gunun basarisi
  .venv/bin/python -m sim.learn replay     # kurallari gecmis backtest uzerinde dene (ise yariyor mu?)
  .venv/bin/python -m sim.learn rollback   # son ayari geri al

Kurallar (her biri ayri, kucuk adimli, sinirli):
  - Lig gol seviyesi: son 120 gunde ligde >= 60 mac, gercek gol - beklenen gol Poisson z > 3
        -> lig gol carpani sapmanin %30'u kadar (toplam en fazla +-%6)
  - Lig ev sahibi avantaji: gol farki artigi z > 3 -> %30 adim (en fazla +-0.05 log)
  - Beraberlik: gercek beraberlik sayisi - tahmin z > 3 -> esitlik durumundaki tempo carpani
        +-%3 (en fazla +-%10)
  - Takim: son 8 macinda takimin attigi/yedigi, puan modelinin beklentisinden z > 3.5 sapiyorsa
        (puanlar zaten her gun yeniden kuruldugu halde) -> 30 gunluk gecici duzeltme, %30 adim
  Esikler, 2025/26 sezonunda geriye donuk denemede (replay) gevsek esiklerin isabeti
  bozdugu goruldugu icin bilerek siki tutuldu: model ancak gercekten israrli bir hatada degisir.
  - Yeni hoca: takimin hocasi son 60 gunde degistiyse oruntu olarak kaydedilir. Puanlamada ek agirlik
        (ratings new_coach_w) gecmis denemede isabeti dusurdugu icin varsayilan olarak kapali.
Her degisiklik learn_log tablosuna kanitiyla yazilir; params.json'un onceki hali
sim/params_history/ altina kopyalanir (silinmez), 'rollback' ile geri alinabilir.
"""
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .common import PARAMS, norm, sim_con, src_con

HIST_DIR = PARAMS.parent / "params_history"
Z = 3.0
STEP = 0.3  # bulunan sapmanin ne kadari duzeltilir
SCHEMA = """
CREATE TABLE IF NOT EXISTS pred_eval (
  match_id INTEGER PRIMARY KEY, ymd TEXT, league TEXT, home_key TEXT, away_key TEXT, version TEXT,
  hg INTEGER, ag INTEGER, p_home REAL, p_draw REAL, p_away REAL, exp_h REAL, exp_a REAL,
  over25 REAL, btts REAL, ll REAL, brier REAL, rps REAL, evaluated_at TEXT);
CREATE TABLE IF NOT EXISTS learn_log (
  ts TEXT, rule TEXT, target TEXT, old TEXT, new TEXT, evidence TEXT);
"""


def load_params():
    return json.loads(PARAMS.read_text()) if PARAMS.exists() else {}


def save_params(p):
    """Bu modulun PARAMS yoluna yazar (testler gecici bir yol verebilsin diye)."""
    PARAMS.write_text(json.dumps(p, indent=1, ensure_ascii=False, default=float))


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------ degerlendirme
def evaluate():
    """Bitmis maclarin son tahminini (mac basladiktan once yapilan en son) sonucla eslestirir."""
    con = sim_con()
    con.executescript(SCHEMA)
    pr = pd.read_sql("SELECT * FROM predictions", con)
    if pr.empty:
        return pd.DataFrame()
    pr = pr[pr.created_at.str[:16] <= pr.utc_time.str[:16]].sort_values("created_at").groupby("match_id").last().reset_index()
    res = pd.read_sql("SELECT match_id, home_goals hg, away_goals ag FROM fm_matches WHERE home_goals IS NOT NULL", src_con())
    try:  # kupa maclari (Avrupa Ligi) ana veritabaninda yok, sim.sqlite'tan
        cup = pd.read_sql("SELECT match_id, home_goals hg, away_goals ag FROM cup_results WHERE home_goals IS NOT NULL", con)
        res = pd.concat([res, cup[~cup.match_id.isin(res.match_id)]])
    except Exception:
        pass
    m = pr.merge(res, on="match_id")
    done = set(pd.read_sql("SELECT match_id FROM pred_eval", con).match_id)
    m = m[~m.match_id.isin(done)]
    if m.empty:
        return m
    y = np.where(m.hg > m.ag, 0, np.where(m.hg == m.ag, 1, 2))
    P = m[["p_home", "p_draw", "p_away"]].values
    o = np.eye(3)[y]
    m["ll"] = -np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1))
    m["brier"] = ((P - o) ** 2).sum(1)
    m["rps"] = ((np.cumsum(P, 1) - np.cumsum(o, 1))[:, :2] ** 2).sum(1) / 2
    m["evaluated_at"] = now()
    cols = ["match_id", "ymd", "league", "home_key", "away_key", "version", "hg", "ag", "p_home", "p_draw", "p_away",
            "exp_h", "exp_a", "over25", "btts", "ll", "brier", "rps", "evaluated_at"]
    m[cols].to_sql("pred_eval", con, if_exists="append", index=False)
    con.commit()
    return m[cols]


# ------------------------------------------------------------ kurallar
def detect(ev, params, today, prev_coach=None):
    """ev: degerlendirilmis tahminler (pred_eval bicimi). Onerilen ayarlari dondurur."""
    adj = params.setdefault("adjust", {})
    out = []
    d = ev.copy()
    d["date"] = pd.to_datetime(d.ymd)
    w = d[d.date >= today - pd.Timedelta(days=120)]

    for lg, g in w.groupby("league"):
        if len(g) < 60:
            continue
        act, exp = (g.hg + g.ag).sum(), (g.exp_h + g.exp_a).sum()
        z = (act - exp) / np.sqrt(exp)
        if abs(z) > Z:
            old = adj.get("goal_scale", {}).get(lg, 1.0)
            new = float(np.clip(old * (1 + STEP * (act - exp) / exp), 0.94, 1.06))
            if abs(new - old) > 1e-3:
                out.append(("lig_gol", lg, ["goal_scale", lg], old, new,
                            dict(mac=len(g), gercek=int(act), beklenen=round(float(exp), 1), z=round(float(z), 2))))
        res = ((g.hg - g.ag) - (g.exp_h - g.exp_a))
        var = (g.exp_h + g.exp_a).sum()
        z = res.sum() / np.sqrt(var)
        if abs(z) > Z:
            old = adj.get("hfa", {}).get(lg, 0.0)
            delta = STEP * res.mean() / (g.exp_h + g.exp_a).mean()
            new = float(np.clip(old + delta, -0.05, 0.05))
            if abs(new - old) > 1e-3:
                out.append(("ev_avantaji", lg, ["hfa", lg], old, new,
                            dict(mac=len(g), artik_gol_farki=round(float(res.mean()), 3), z=round(float(z), 2))))

    if len(w) >= 150:
        dr = (w.hg == w.ag).astype(float)
        z = (dr.sum() - w.p_draw.sum()) / np.sqrt((w.p_draw * (1 - w.p_draw)).sum())
        if abs(z) > Z:
            old = adj.get("level_factor", 1.0)
            new = float(np.clip(old * (0.97 if z > 0 else 1.03), 0.9, 1.1))  # beraberlik az tahmin -> esitlikte tempo dusur
            if abs(new - old) > 1e-3:
                out.append(("beraberlik", "tum", ["level_factor"], old, new,
                            dict(mac=len(w), gercek=int(dr.sum()), tahmin=round(float(w.p_draw.sum()), 1), z=round(float(z), 2))))

    # takim bazli kalici sapma (son 8 mac)
    long = pd.concat([
        pd.DataFrame({"team": d.home_key, "date": d.date, "gf": d.hg, "ga": d.ag, "xf": d.exp_h, "xa": d.exp_a}),
        pd.DataFrame({"team": d.away_key, "date": d.date, "gf": d.ag, "ga": d.hg, "xf": d.exp_a, "xa": d.exp_h})])
    tadj = adj.get("team", {})
    for team, g in long.sort_values("date").groupby("team"):
        g = g.tail(8)
        if len(g) < 6 or g.date.max() < today - pd.Timedelta(days=21):
            continue
        for kind, a, e in (("att", g.gf.sum(), g.xf.sum()), ("def", g.ga.sum(), g.xa.sum())):
            z = (a - e) / np.sqrt(max(e, 1))
            if abs(z) > 3.5:
                old = tadj.get(team, {}).get(kind, 1.0)
                new = float(np.clip(1 + STEP * (a - e) / max(e, 1), 0.9, 1.12))
                out.append(("takim", team, ["team", team, kind], old, new,
                            dict(son_mac=len(g), gercek=int(a), beklenen=round(float(e), 1), z=round(float(z), 2),
                                 bitis=str((today + pd.Timedelta(days=30)).date()))))

    # yeni hoca
    for team, (coach, since) in (prev_coach or {}).items():
        cur = adj.get("new_coach", {}).get(team)
        if cur is None or cur[0] != since:
            out.append(("yeni_hoca", team, ["new_coach", team], cur, [since, coach],
                        dict(hoca=coach, baslangic=since)))
    return out


def coach_changes(today, days=60):
    """FotMob maclarindan: son `days` gun icinde hocasi degisen takimlar -> {team_key: (hoca, ilk mac tarihi)}."""
    from .common import TeamResolver
    con = src_con()
    m = pd.read_sql("SELECT league, substr(utc_time,1,10) d, home_id, home, away_id, away, home_coach, away_coach "
                    "FROM fm_matches WHERE utc_time >= ? ORDER BY utc_time",
                    con, params=((today - pd.Timedelta(days=400)).strftime("%Y-%m-%d"),))
    long = pd.concat([m[["league", "d", "home_id", "home", "home_coach"]].set_axis(["league", "d", "tid", "name", "coach"], axis=1),
                      m[["league", "d", "away_id", "away", "away_coach"]].set_axis(["league", "d", "tid", "name", "coach"], axis=1)])
    long = long.dropna(subset=["coach"]).sort_values("d")
    r = TeamResolver(con)
    out = {}
    for tid, g in long.groupby("tid"):
        c = g.coach.map(norm).values
        if len(c) < 2 or c[-1] == c[-2] and (g.coach.map(norm) == c[-1]).all():
            continue
        start = g.d[g.coach.map(norm) == c[-1]]
        # son hocanin kesintisiz ilk maci
        idx = len(c) - 1
        while idx > 0 and c[idx - 1] == c[-1]:
            idx -= 1
        if idx == 0:
            continue
        since = g.d.iloc[idx]
        if pd.Timestamp(since) >= today - pd.Timedelta(days=days):
            out[r.fotmob(tid, g.name.iloc[-1], g.league.iloc[-1])] = (g.coach.iloc[-1], since)
    return out


def apply(changes, params):
    if not changes:
        return params
    HIST_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if PARAMS.exists():
        shutil.copy(PARAMS, HIST_DIR / f"params_{stamp}.json")
    con = sim_con()
    con.executescript(SCHEMA)
    adj = params.setdefault("adjust", {})
    for rule, target, path, old, new, ev in changes:
        node = adj
        for k in path[:-1]:
            node = node.setdefault(k, {})
        if rule == "takim":
            node[path[-1]] = new
            node["until"] = ev["bitis"]
        else:
            node[path[-1]] = new
        con.execute("INSERT INTO learn_log VALUES (?,?,?,?,?,?)",
                    (now(), rule, target, json.dumps(old), json.dumps(new), json.dumps(ev, ensure_ascii=False)))
    params["version"] = f"v1-{stamp}"
    save_params(params)
    con.commit()
    return params


def expire(params, today):
    """Suresi dolan takim duzeltmelerini kaldir (kayit learn_log'da kalir)."""
    t = params.get("adjust", {}).get("team", {})
    for k in [k for k, v in t.items() if pd.Timestamp(v.get("until", "2100-01-01")) < today]:
        t.pop(k)
    nc = params.get("adjust", {}).get("new_coach", {})
    for k in [k for k, v in nc.items() if pd.Timestamp(v[0]) < today - pd.Timedelta(days=120)]:
        nc.pop(k)
    return params


def daily():
    today = pd.Timestamp.now().normalize()
    new = evaluate()
    print(f"degerlendirilen yeni mac: {len(new)}" + (f", ort. log-loss {new.ll.mean():.3f}" if len(new) else ""))
    ev = pd.read_sql("SELECT * FROM pred_eval", sim_con())
    params = expire(load_params(), today)
    ch = detect(ev, params, today, coach_changes(today))
    for c in ch:
        print(f"  ayar: {c[0]} {c[1]}: {c[3]} -> {c[4]}  kanit {c[5]}")
    if ch:
        apply(ch, params)
    else:
        save_params(params)
        print("  kalici bir hata ya da yeni oruntu yok; model degismedi")
    return new, ch


def report(days=(30, 90)):
    ev = pd.read_sql("SELECT * FROM pred_eval", sim_con())
    if ev.empty:
        print("henuz degerlendirilmis tahmin yok")
        return
    ev["date"] = pd.to_datetime(ev.ymd)
    for d in days:
        w = ev[ev.date >= pd.Timestamp.now() - pd.Timedelta(days=d)]
        if len(w):
            hit = (np.argmax(w[["p_home", "p_draw", "p_away"]].values, 1) ==
                   np.where(w.hg > w.ag, 0, np.where(w.hg == w.ag, 1, 2))).mean()
            print(f"son {d} gun: {len(w)} mac, isabet %{100 * hit:.1f}, log-loss {w.ll.mean():.3f}, RPS {w.rps.mean():.3f}")


def rollback():
    files = sorted(HIST_DIR.glob("params_*.json"))
    if not files:
        print("geri alinacak kayit yok")
        return
    cur = HIST_DIR / f"params_rolledback_{datetime.now():%Y%m%d_%H%M%S}.json"
    shutil.copy(PARAMS, cur)
    shutil.copy(files[-1], PARAMS)
    con = sim_con()
    con.execute("INSERT INTO learn_log VALUES (?,?,?,?,?,?)", (now(), "geri_al", files[-1].name, "", "", "{}"))
    con.commit()
    print("geri alindi:", files[-1].name, "(onceki hali", cur.name, ")")


def replay(table="backtest", start="2025-08-01"):
    """Kurallari gecmiste haftalik uygula; Poisson yaklasimiyla 1X2 log-loss'a etkisini olc."""
    from . import hist
    from .rolling import outcome_probs, score_matrix
    b = pd.read_sql(f"SELECT * FROM {table}", sim_con())
    g = hist.games()
    m = b.merge(g[["game_id", "date", "league", "home", "away", "hg", "ag"]], on="game_id")
    m = m.rename(columns={"home": "home_key", "away": "away_key", "eg_h": "exp_h", "eg_a": "exp_a",
                          "p_h": "p_home", "p_d": "p_draw", "p_a": "p_away"})
    m["ymd"] = m.date.dt.strftime("%Y-%m-%d")
    params = {"adjust": {}}
    base_ll, new_ll, n = 0.0, 0.0, 0
    weeks = pd.date_range(pd.Timestamp(start) + pd.Timedelta(days=120), m.date.max(), freq="7D")
    log = []
    for d0 in weeks:
        past = m[m.date < d0]
        ch = detect(past, params, d0)
        ch = [c for c in ch if c[0] != "yeni_hoca"]
        for rule, target, path, old, new, ev in ch:
            node = params["adjust"]
            for k in path[:-1]:
                node = node.setdefault(k, {})
            node[path[-1]] = new
            if rule == "takim":
                node["until"] = ev["bitis"]
            log.append((d0.date(), rule, target, round(new, 3)))
        expire(params, d0)
        wk = m[(m.date >= d0) & (m.date < d0 + pd.Timedelta(days=7))]
        if wk.empty:
            continue
        a = params["adjust"]
        gs = wk.league.map(a.get("goal_scale", {})).fillna(1.0).values
        hf = wk.league.map(a.get("hfa", {})).fillna(0.0).values
        ta = lambda k, kind: np.array([a.get("team", {}).get(t, {}).get(kind, 1.0) for t in k])
        mh = wk.mu_h.values * gs * np.exp(hf) * ta(wk.home_key, "att") * ta(wk.away_key, "def")
        ma = wk.mu_a.values * gs * np.exp(-hf) * ta(wk.away_key, "att") * ta(wk.home_key, "def")
        y = np.where(wk.hg > wk.ag, 0, np.where(wk.hg == wk.ag, 1, 2))
        p0 = outcome_probs(score_matrix(wk.mu_h.values, wk.mu_a.values))
        p1 = outcome_probs(score_matrix(mh, ma))
        base_ll += -np.log(p0[np.arange(len(y)), y]).sum()
        new_ll += -np.log(p1[np.arange(len(y)), y]).sum()
        n += len(y)
    print(f"{n} mac: ayarsiz log-loss {base_ll / n:.4f}, ayarli {new_ll / n:.4f}, ayar sayisi {len(log)}")
    for r in log[:40]:
        print("  ", r)
    return base_ll / n, new_ll / n


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "daily"
    {"daily": daily, "report": report, "rollback": rollback}.get(cmd, lambda: replay(*sys.argv[2:]))()
