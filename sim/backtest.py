"""Backtest: motoru gecmis bir donemde, yalnizca o gune kadarki veriyle oynatir ve gercek sonuc
ve bahis piyasasiyla karsilastirir.

  .venv/bin/python -m sim.backtest 2025-07-01 2026-06-01 [--n 2000] [--workers 3] [--no-players]

Sizinti olmasin diye hoca tablolari (skor tepkisi, rakibe gore plan) baslangic tarihinden onceki
maclarla yeniden kestirilir. Sonuclar data/sim.sqlite 'backtest' tablosuna yazilir.
Olcutler: 1X2 log-loss / Brier / RPS, 2.5 ust log-loss, KG log-loss, golcu kalibrasyonu,
gol / beraberlik / kirmizi kart ortalamalari, olasilik dilimlerine gore kalibrasyon.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import calibrate, dynamics, hist
from .common import load_params, sim_con, src_con

_P = None


def _init(start, with_players, overrides=None):
    global _P
    from .match import Predictor
    params = {**load_params(), **(overrides or {})}
    g = hist.games()
    ev = hist.events()
    pred = pd.read_sql("SELECT * FROM pred_hist", sim_con())
    pre = pred[pred.game_id.isin(g[g.date < start].game_id)]
    cs = dynamics.coach_state(g[g.date < start], ev, pre, params["dynamics"], k=params.get("coach_state_k", 6.0))
    t = calibrate.team_rows(g[g.date < start], pre, params.get("ratings", {}).get("xg_w", 0.6))
    cp = calibrate.coach_plan(t)
    _P = Predictor(start, games=g, with_players=with_players, params=params, coach_state=cs, coach_plan=cp)


def _run_week(args):
    d0, rows, n = args
    P = _P
    P.set_date(d0)
    out = []
    for r in rows:
        try:
            o = P.predict(r["league"], r["home"], r["away"], n=n, seed=hash(r["game_id"]) % 2**31,
                          home_coach=r["home_coach"], away_coach=r["away_coach"])
        except Exception as e:  # noqa
            print("hata", r["game_id"], e, flush=True)
            continue
        sc = o["scorers"]
        out.append(dict(
            game_id=r["game_id"], p_h=o["p_home"], p_d=o["p_draw"], p_a=o["p_away"],
            eg_h=o["exp_goals"][0], eg_a=o["exp_goals"][1], over25=o["over"]["2.5"], over15=o["over"]["1.5"],
            over35=o["over"]["3.5"], btts=o["btts"], p_red=o["p_red"], mu_h=o["info"]["mu"][0], mu_a=o["info"]["mu"][1],
            ht_h=o["p_ht"][0], ht_d=o["p_ht"][1], ht_a=o["p_ht"][2],
            scorers=json.dumps([[s["name"], round(s["p"], 4)] for side in sc for s in side], ensure_ascii=False)))
    return out


def rps(p, y):
    """Ranked probability score (1X2 sirasi onemli: ev, beraberlik, deplasman)."""
    o = np.zeros_like(p)
    o[np.arange(len(y)), y] = 1
    return float(np.mean(((np.cumsum(p, 1) - np.cumsum(o, 1))[:, :2] ** 2).sum(1) / 2))


def ll_bin(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def report(start, end, table="backtest"):
    con = sim_con()
    b = pd.read_sql(f"SELECT * FROM {table}", con)
    g = hist.games().rename(columns={"p_h": "mk_h", "p_d": "mk_d", "p_a": "mk_a"})
    m = b.merge(g, on="game_id")
    m = m[(m.date >= start) & (m.date < end)]
    y = np.where(m.hg > m.ag, 0, np.where(m.hg == m.ag, 1, 2))
    P = m[["p_h", "p_d", "p_a"]].values
    P = P / P.sum(1, keepdims=True)
    res = {"maclar": len(m)}
    res["1x2_logloss"] = float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1))))
    res["1x2_brier"] = float(np.mean(((P - np.eye(3)[y]) ** 2).sum(1)))
    res["1x2_rps"] = rps(P, y)
    # Poisson puan modeli (motor olmadan) ayni maclarda
    from .rolling import outcome_probs, score_matrix
    pp = outcome_probs(score_matrix(m.mu_h, m.mu_a))
    res["poisson_logloss"] = float(-np.mean(np.log(np.clip(pp[np.arange(len(y)), y], 1e-9, 1))))
    tot = (m.hg + m.ag).values
    res["ust25_logloss"] = ll_bin(m.over25.values, (tot > 2.5).astype(float))
    res["kg_logloss"] = ll_bin(m.btts.values, ((m.hg > 0) & (m.ag > 0)).values.astype(float))
    res["gol_ort_gercek"], res["gol_ort_motor"] = float(tot.mean()), float((m.eg_h + m.eg_a).mean())
    res["beraberlik_gercek"], res["beraberlik_motor"] = float((y == 1).mean()), float(m.p_d.mean())
    res["ust25_gercek"], res["ust25_motor"] = float((tot > 2.5).mean()), float(m.over25.mean())
    res["kg_gercek"], res["kg_motor"] = float(((m.hg > 0) & (m.ag > 0)).mean()), float(m.btts.mean())
    return res, m


def market_compare(m):
    """Bahis oranlari olan maclarda motor vs piyasa (1X2 ve 2.5 ust)."""
    src = src_con()
    x = pd.read_sql("SELECT matchdate date, hometeam, awayteam, oddhome, odddraw, oddaway, over25 o25, under25 u25 FROM xg_matches", src)
    km = pd.read_sql("SELECT name, team_key FROM team_map WHERE source='xgabora'", src)
    km = dict(zip(km.name, km.team_key))
    x["home"], x["away"] = x.hometeam.map(km), x.awayteam.map(km)
    x["date"] = pd.to_datetime(x.date)
    j = m.merge(x, on=["date", "home", "away"], how="inner").dropna(subset=["oddhome", "odddraw", "oddaway"])
    y = np.where(j.hg > j.ag, 0, np.where(j.hg == j.ag, 1, 2))
    inv = 1 / j[["oddhome", "odddraw", "oddaway"]].values
    mk = inv / inv.sum(1, keepdims=True)
    P = j[["p_h", "p_d", "p_a"]].values
    out = {"piyasali_mac": len(j)}
    ll = lambda Q: float(-np.mean(np.log(np.clip(Q[np.arange(len(y)), y], 1e-9, 1))))
    out["motor_1x2_ll"], out["piyasa_1x2_ll"] = ll(P), ll(mk)
    out["motor_rps"], out["piyasa_rps"] = rps(P, y), rps(mk, y)
    # karisim: piyasa + motor (motorun piyasaya kattigi bilgi)
    for w in (0.2, 0.35, 0.5):
        out[f"karisim_{w}_ll"] = ll((1 - w) * mk + w * P)
    o = j.dropna(subset=["o25", "u25"])
    if len(o):
        io = 1 / o[["o25", "u25"]].values
        po = io[:, 0] / io.sum(1)
        yo = ((o.hg + o.ag) > 2.5).values.astype(float)
        out["ust_mac"] = len(o)
        out["motor_ust25_ll"], out["piyasa_ust25_ll"] = ll_bin(o.over25.values, yo), ll_bin(po, yo)
    # basit deger bahsi: motor olasiligi * oran > 1.05 olan sonuca 1 birim
    odds = j[["oddhome", "odddraw", "oddaway"]].values
    ev = P * odds
    pick = ev.argmax(1)
    bet = ev.max(1) > 1.05
    win = pick == y
    profit = np.where(bet, np.where(win, odds[np.arange(len(y)), pick] - 1, -1), 0)
    out["deger_bahsi_sayi"], out["deger_bahsi_roi"] = int(bet.sum()), float(profit.sum() / max(bet.sum(), 1))
    return out


def calib_table(m):
    rows = []
    for col, yv in (("p_h", m.hg > m.ag), ("p_d", m.hg == m.ag), ("p_a", m.hg < m.ag), ("over25", (m.hg + m.ag) > 2.5)):
        bins = pd.cut(m[col], [0, .1, .2, .3, .4, .5, .6, .7, .8, 1])
        t = pd.DataFrame({"bin": bins, "p": m[col], "y": yv.astype(float)}).groupby("bin", observed=True).agg(
            n=("y", "size"), tahmin=("p", "mean"), gercek=("y", "mean"))
        t["olcu"] = col
        rows.append(t.reset_index())
    return pd.concat(rows)


def scorer_calib(m):
    """Golcu olasiliklari vs Transfermarkt gol olaylari (yalnizca TM maclari)."""
    src = src_con()
    gids = [gid for gid in m.game_id if not gid.startswith(("fm:", "xg:"))]
    if not gids:
        return None
    e = pd.read_sql(f"SELECT game_id, player_id, description FROM tm_events WHERE type='Goals' AND game_id IN ({','.join(gids)})", src)
    names = pd.read_sql("SELECT player_id, name FROM tm_players", src).set_index("player_id").name
    e["name"] = e.player_id.map(names)
    scored = e.groupby("game_id").name.apply(set).to_dict()
    rows = []
    for gid, sc in zip(m.game_id, m.scorers):
        s = scored.get(int(gid), set()) if gid.isdigit() else set()
        for name, p in json.loads(sc):
            rows.append((p, name in s))
    d = pd.DataFrame(rows, columns=["p", "y"])
    d["bin"] = pd.cut(d.p, [0, .1, .2, .3, .4, .5, .7, 1])
    return d.groupby("bin", observed=True).agg(n=("y", "size"), tahmin=("p", "mean"), gercek=("y", "mean"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start")
    ap.add_argument("end")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--no-players", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="parametre=deger (json), orn. use_coach=false")
    ap.add_argument("--table", default="backtest")
    a = ap.parse_args()
    if not a.report_only:
        g = hist.games()
        sel = g[(g.date >= a.start) & (g.date < a.end)]
        weeks = []
        wkno = (sel.date - pd.Timestamp(a.start)).dt.days // 7
        for w, wk in sel.groupby(wkno):
            d0 = pd.Timestamp(a.start) + pd.Timedelta(days=7 * int(w))
            if len(wk):
                weeks.append((d0, wk[["game_id", "league", "home", "away", "home_coach", "away_coach"]]
                              .astype(object).where(wk.notna(), None).to_dict("records"), a.n))
        t0 = time.time()
        out = []
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.start, not a.no_players, {k: json.loads(v) for k, v in (x.split("=", 1) for x in a.set)})) as ex:
            for i, r in enumerate(ex.map(_run_week, weeks), 1):
                out.extend(r)
                if i % 5 == 0:
                    print(f"{i}/{len(weeks)} hafta, {len(out)} mac, {time.time() - t0:.0f} sn", flush=True)
        con = sim_con()
        pd.DataFrame(out).to_sql(a.table, con, if_exists="replace", index=False)
        con.commit()
    res, m = report(a.start, a.end, a.table)
    res.update(market_compare(m))
    for k, v in res.items():
        print(f"  {k:24s} {v:.4f}" if isinstance(v, float) else f"  {k:24s} {v}")
    print(calib_table(m).round(3).to_string())
    sc = scorer_calib(m)
    if sc is not None:
        print("golcu kalibrasyonu\n", sc.round(3).to_string())
    by_lg = m.assign(ll=-np.log(np.clip(np.where(m.hg > m.ag, m.p_h, np.where(m.hg == m.ag, m.p_d, m.p_a)), 1e-9, 1)),
                     gol=m.hg + m.ag, egol=m.eg_h + m.eg_a, red=m.p_red)
    print(by_lg.groupby("league")[["ll", "gol", "egol", "red"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
