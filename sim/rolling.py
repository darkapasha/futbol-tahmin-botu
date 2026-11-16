"""Gecmise donuk tahmin: her hafta yalnizca o tarihe kadarki veriyle puanlari yeniden kurar,
o haftanin maclari icin mac oncesi beklenen gol (mu) uretir. Kalibrasyonun ve backtest'in temeli.

  .venv/bin/python -m sim.rolling tune    # puan parametrelerini log-loss ile ayarla
"""
import sys

import numpy as np
import pandas as pd

from . import hist, ratings
from .common import load_params, save_params


_VB = None
_VC = {}  # tarih -> kadro degerleri (tune boyunca tekrar hesaplanmasin)


def value_book():
    global _VB
    if _VB is None:
        from .value import ValueBook
        _VB = ValueBook()
    return _VB


def rolling(g, params=None, start="2022-08-01", step_days=7):
    out = []
    p = {**ratings.DEFAULTS, **(params or {})}
    use_val = p.get("val_att", 0) != 0 or p.get("val_def", 0) != 0
    vcache = _VC
    dates = pd.date_range(start, g.date.max() + pd.Timedelta(days=1), freq=f"{step_days}D")
    for d0, d1 in zip(dates[:-1], dates[1:]):
        wk = g[(g.date >= d0) & (g.date < d1)]
        if wk.empty:
            continue
        vals = None
        if use_val:
            if d0 not in vcache:
                vcache[d0] = value_book().at(d0)
            vals = vcache[d0]
        r = ratings.fit(g, d0, params, values=vals)
        mh, ma = zip(*(r.rate(h, a, lg) for h, a, lg in zip(wk.home, wk.away, wk.league)))
        out.append(pd.DataFrame({"game_id": wk.game_id.values, "mu_h": mh, "mu_a": ma,
                                 "net_h": [r.net(t, lg) for t, lg in zip(wk.home, wk.league)],
                                 "net_a": [r.net(t, lg) for t, lg in zip(wk.away, wk.league)]}))
    return pd.concat(out, ignore_index=True)


def _pmf(mu, maxg=10):
    k = np.arange(maxg + 1)
    lf = np.cumsum(np.r_[0, np.log(np.arange(1, maxg + 1))])
    return np.exp(-mu[:, None] + k[None, :] * np.log(mu[:, None]) - lf[None, :])


def score_matrix(mh, ma, rho=-0.05, maxg=10):
    mh, ma = np.asarray(mh, float), np.asarray(ma, float)
    P = _pmf(mh, maxg)[:, :, None] * _pmf(ma, maxg)[:, None, :]
    # Dixon-Coles
    P[:, 0, 0] *= 1 - mh * ma * rho
    P[:, 0, 1] *= 1 + mh * rho
    P[:, 1, 0] *= 1 + ma * rho
    P[:, 1, 1] *= 1 - rho
    return P / P.sum(axis=(1, 2), keepdims=True)


def outcome_probs(P):
    n = P.shape[1]
    i, j = np.indices((n, n))
    return np.stack([(P * (i > j)).sum((1, 2)), (P * (i == j)).sum((1, 2)), (P * (i < j)).sum((1, 2))], 1)


def logloss(probs, hg, ag):
    y = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-9, 1))))


def evaluate(g, pred, label=""):
    m = g.merge(pred, on="game_id")
    pr = outcome_probs(score_matrix(m.mu_h, m.mu_a))
    ll = logloss(pr, m.hg.values, m.ag.values)
    mk = m.dropna(subset=["p_h"])
    mpr = mk[["p_h", "p_d", "p_a"]].values
    mpr = mpr / mpr.sum(1, keepdims=True)
    sub = outcome_probs(score_matrix(mk.mu_h, mk.mu_a))
    res = dict(n=len(m), ll=ll, n_mkt=len(mk), ll_model_on_mkt=logloss(sub, mk.hg.values, mk.ag.values),
               ll_mkt=logloss(mpr, mk.hg.values, mk.ag.values))
    print(label, {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}, flush=True)
    return res


def tune(g):
    p = {**ratings.DEFAULTS, **load_params().get("ratings", {})}
    grid = {"val_att": [0.05, 0.1, 0.15, 0.2], "val_def": [0.05, 0.1, 0.15, 0.2], "prior_k": [2, 4, 8, 12],
            "half_life": [150, 180, 240], "xg_w": [0.55, 0.65, 0.75],
            "new_off_att": [-0.3, -0.18, -0.08, 0.0], "new_off_def": [-0.3, -0.15, -0.05, 0.0]}
    base = evaluate(g, rolling(g, p), "baslangic")["ll"]
    for rnd in range(2):
        for k, vals in grid.items():
            for v in vals:
                if v == p[k]:
                    continue
                q = {**p, k: v}
                ll = evaluate(g, rolling(g, q), f"{k}={v}")["ll"]
                if ll < base - 1e-4:
                    base, p = ll, q
                    print("  -> secildi", k, v, flush=True)
    # dogrudan devreye almaz: aday dosyaya yazar, backtest'ten sonra elle / calibrate ile alinir
    import json
    from .common import PARAMS
    PARAMS.with_name("ratings_tuned.json").write_text(json.dumps(p, indent=1))
    print("son", p, base)


if __name__ == "__main__":
    g = hist.games()
    if sys.argv[1:2] == ["tune"]:
        tune(g)
    else:
        evaluate(g, rolling(g, load_params().get("ratings")), "mevcut")
