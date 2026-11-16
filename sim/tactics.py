"""Taktik eslesme testi: dizilis ve oyun tarzi carpismalari, puan modelinin bilmedigi bir sey katiyor mu?

  .venv/bin/python -m sim.tactics            # egit (2021/22-2023/24) + 2024/25 ve 2025/26'da sina

Her takim-mac icin atilan gol (gol/xG karisimi), mac oncesi puan modelinin beklentisine (mu) gore
Poisson modeliyle aciklanir: log E[gol] = log mu + b . x
x (hepsi mac oncesi bilinen):
  - dizilis eslesmesi: kendi dizilisi x rakip dizilisi (en sik 8 dizilis, 3'lu/4'lu/5'li savunma)
  - oyun tarzi carpismalari (bir onceki sezonun stil endeksleriyle, sizinti yok):
      pres x rakibin topa sahip olma oyunu, direkt oyun x rakibin presi, kanat oyunu x rakibin 3'lu savunmasi,
      duran top tehdidi x rakibin duran top zaafi, kontra x rakibin topa sahip olmasi
  - eksiklerle dizilis degisimi: takim bu macta alisik oldugu dizilisten farkli mi (ilk 11 belli olunca)
Kat sayilar L2 cezasiyla (asiri ogrenmeye karsi) egitilir; yalnizca sonraki sezonlarda sinanir.
"""
import json

import numpy as np
import pandas as pd

from . import calibrate, hist
from .common import load_params, prof_con, sim_con
from .rolling import outcome_probs, score_matrix

TOP_FORM = ["4-2-3-1", "4-3-3", "3-4-2-1", "4-4-2", "3-5-2", "4-1-4-1", "3-4-3", "5-4-1"]
STYLE = ["press_idx", "control_idx", "directness_idx", "width_style_idx", "setpiece_threat_idx", "setpiece_weak_idx"]


def dataset():
    g = hist.games()
    pred = pd.read_sql("SELECT * FROM pred_hist", sim_con())
    xw = load_params().get("ratings", {}).get("xg_w", 0.75)
    t = calibrate.team_rows(g, pred, xw)
    pc = prof_con()
    tm = pd.read_sql("SELECT game_id, team_key, season, formation, opp_formation, back3, n_def, opp_n_def FROM team_match", pc)
    t = t.merge(tm.rename(columns={"team_key": "team"}), on=["game_id", "team"], how="inner")
    # alisik dizilis: takimin onceki 10 macindaki en sik dizilis (sizinti yok)
    t = t.sort_values("date")
    def usual(s):
        from collections import Counter
        out, hist_ = [], []
        for f in s:
            out.append(Counter(hist_[-10:]).most_common(1)[0][0] if hist_ else None)
            if isinstance(f, str):
                hist_.append(f)
        return pd.Series(out, index=s.index, dtype=object)
    t["usual_form"] = t.groupby("team").formation.transform(usual)
    # onceki sezonun stil endeksleri
    ts = pd.read_sql(f"SELECT team_key team, season, {', '.join(STYLE)} FROM team_season", pc)
    ts["season"] = ts.season + 1
    t = t.merge(ts, on=["team", "season"], how="left")
    opp = ts.rename(columns={"team": "opp", **{c: f"opp_{c}" for c in STYLE}})
    keys = g[["game_id", "home", "away"]]
    t = t.merge(keys, on="game_id")
    t["opp"] = np.where(t.is_home == 1, t.away, t.home)
    t = t.merge(opp, on=["opp", "season"], how="left")
    return t


def features(t):
    X = {}
    fa = t.formation.where(t.formation.isin(TOP_FORM), "diger")
    fo = t.opp_formation.where(t.opp_formation.isin(TOP_FORM), "diger")
    for a in TOP_FORM:
        for b in TOP_FORM:
            X[f"f:{a}|{b}"] = ((fa == a) & (fo == b)).astype(float)
    nd, od = t.n_def.fillna(4), t.opp_n_def.fillna(4)
    for a in (3, 4, 5):
        for b in (3, 4, 5):
            X[f"d:{a}|{b}"] = ((nd == a) & (od == b)).astype(float)
    z = lambda c: t[c].fillna(0).clip(-3, 3)
    X["pres_x_rakip_topa_sahip"] = z("press_idx") * z("opp_control_idx")
    X["direkt_x_rakip_pres"] = z("directness_idx") * z("opp_press_idx")
    X["kanat_x_rakip_3lu"] = z("width_style_idx") * (od == 3).astype(float)
    X["duran_top_x_rakip_zaaf"] = z("setpiece_threat_idx") * z("opp_setpiece_weak_idx")
    X["direkt_x_rakip_topa_sahip"] = z("directness_idx") * z("opp_control_idx")
    X["topa_sahip_x_rakip_pres"] = z("control_idx") * z("opp_press_idx")
    X["dizilis_degisti"] = ((t.usual_form.notna()) & (t.formation != t.usual_form)).astype(float)
    X["rakip_dizilis_degisti"] = 0.0  # asagida rakip satirindan doldurulur
    return pd.DataFrame(X, index=t.index)


def fit_poisson(X, y, off, w=None, lam=30.0, iters=30):
    """L2 cezali Poisson regresyonu (offset = log mu), Newton adimlari."""
    n, k = X.shape
    b = np.zeros(k)
    w = np.ones(n) if w is None else w
    for _ in range(iters):
        eta = off + X @ b
        mu = np.exp(eta)
        grad = X.T @ (w * (y - mu)) - lam * b
        H = (X * (w * mu)[:, None]).T @ X + lam * np.eye(k)
        step = np.linalg.solve(H, grad)
        b += step
        if np.abs(step).max() < 1e-6:
            break
    return b


def evaluate(t, X, b, label):
    """Mac duzeyinde: ev ve deplasman beklentisi duzeltilip 1X2 log-loss ve isabet."""
    adj = np.exp(X.values @ b) if b is not None else np.ones(len(t))
    d = t.assign(adj=adj)
    h = d[d.is_home == 1].set_index("game_id")
    a = d[d.is_home == 0].set_index("game_id")
    ids = h.index.intersection(a.index)
    h, a = h.loc[ids], a.loc[ids]
    mh, ma = h.mu_for * h.adj, a.mu_for * a.adj
    g = hist.games().set_index("game_id").loc[ids]
    P = outcome_probs(score_matrix(mh.values, ma.values))
    y = np.where(g.hg > g.ag, 0, np.where(g.hg == g.ag, 1, 2))
    ll = float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1))))
    acc = float((P.argmax(1) == y).mean())
    tot = (g.hg + g.ag).values > 2.5
    Pm = score_matrix(mh.values, ma.values)
    i, j = np.indices(Pm.shape[1:])
    po = (Pm * ((i + j) > 2)).sum((1, 2))
    ou = float(-np.mean(tot * np.log(po) + (1 - tot) * np.log(1 - po)))
    return dict(model=label, mac=len(ids), logloss=round(ll, 4), isabet=round(acc, 4), ust25_ll=round(ou, 4))


def main():
    t = dataset()
    X = features(t)
    # rakibin dizilis degisikligi: ayni macin diger satiri
    other = t[["game_id", "is_home"]].assign(v=X["dizilis_degisti"].values)
    other["is_home"] = 1 - other.is_home
    X["rakip_dizilis_degisti"] = t[["game_id", "is_home"]].merge(other, on=["game_id", "is_home"], how="left").v.fillna(0).values
    y = t.y_for.values
    off = np.log(t.mu_for.values)
    out, coefs = [], {}
    for test_start, test_end in (("2024-08-01", "2025-07-01"), ("2025-08-01", "2026-07-01")):
        tr = (t.date < pd.Timestamp(test_start) - pd.Timedelta(days=20)).values
        te = ((t.date >= test_start) & (t.date < test_end)).values
        for lam in (10.0, 30.0, 100.0):
            b = fit_poisson(X.values[tr], y[tr], off[tr], lam=lam)
            r = evaluate(t[te], X[te], b, f"taktik (lam={lam:g})")
            r["test"] = test_start[:4]
            out.append(r)
            coefs[(test_start[:4], lam)] = dict(zip(X.columns, b))
        r = evaluate(t[te], X[te], None, "mevcut model")
        r["test"] = test_start[:4]
        out.append(r)
    res = pd.DataFrame(out)
    print(res.to_string(index=False))
    b = pd.Series(coefs[("2025", 30.0)])
    sty = b[~b.index.str.startswith(("f:", "d:"))]
    print("\nstil / degisim kat sayilari (exp = gol carpani):")
    print(np.exp(sty).round(3).to_string())
    fb = b[b.index.str.startswith("f:")].sort_values()
    print("\nen guclu dizilis eslesmeleri (gol carpani):")
    print(np.exp(pd.concat([fb.head(5), fb.tail(5)])).round(3).to_string())
    con = sim_con()
    res.to_sql("tactics_test", con, if_exists="replace", index=False)
    con.commit()
    return res, b


if __name__ == "__main__":
    main()
