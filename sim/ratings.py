"""Takim gucu: zamanla sonumlenen hucum / savunma puanlari (Poisson, gol + xG karisimi).

log mu_ev  = c + ev_avantaji[lig] + hucum[ev]  - savunma[dep]
log mu_dep = c +                    hucum[dep] - savunma[ev]

- Hedef: gol ile xG'nin karisimi (xG varsa); xG sansi ayiklar, gol bitiriciligi ve
  xG modelinin kacirdiklarini tasir.
- Yakin mac agir (yari omur `half_life` gun).
- Her takim `prior_k` sahte macla kendi liginin ortalamasina, ligde ilk sezonunu
  geciren (yukselen) takimlar ortalamanin `new_off` kadar altina buzulur.
- Ligler arasi seviye Sampiyonlar Ligi maclariyla baglanir (ortak tek olcek).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DEFAULTS = dict(half_life=180.0, xg_w=0.6, prior_k=4.0, new_off_att=-0.18, new_off_def=-0.15,
                hfa_prior=0.22, window_days=1100, val_att=0.0, val_def=0.0)


@dataclass
class Ratings:
    asof: pd.Timestamp
    c: float
    hfa: dict
    att: dict
    dfn: dict
    league_of: dict
    weight: dict
    params: dict = field(default_factory=dict)

    def _unknown(self, team, league):
        """Hic maci olmayan takim: lig ortalamasi + yukselen takim farki (+ kadro degeri varsa)."""
        lm = self.league_mean(league)
        z = 0.0
        vals = getattr(self, "values", {})
        if team in vals:
            lv = [vals[k] for k, l in self.league_of.items() if l == league and k in vals]
            if len(lv) >= 3:
                z = float(np.clip(vals[team] - np.mean(lv), -2.5, 2.5))
        return (lm[0] + self.params["new_off_att"] + self.params["val_att"] * z,
                lm[1] + self.params["new_off_def"] + self.params["val_def"] * z)

    def rate(self, home, away, league, neutral=False):
        uh = (self.att[home], self.dfn[home]) if home in self.att else self._unknown(home, league)
        ua = (self.att[away], self.dfn[away]) if away in self.att else self._unknown(away, league)
        a_h, d_h = uh
        a_a, d_a = ua
        h = 0.0 if neutral else self.hfa.get(league, self.params["hfa_prior"])
        return float(np.exp(self.c + h + a_h - d_a)), float(np.exp(self.c + a_a - d_h))

    def net(self, team, league):
        """Tek sayilik guc (hucum + savunma); rakip sinifi (guclu/benzer/zayif) icin."""
        if team in self.att:
            return float(self.att[team] + self.dfn[team])
        lm = self.league_mean(league)
        return lm[0] + lm[1] + self.params["new_off_att"] + self.params["new_off_def"]

    def league_mean(self, league):
        ks = [k for k, l in self.league_of.items() if l == league and self.weight.get(k, 0) > 5]
        if not ks:
            return 0.0, 0.0
        return float(np.mean([self.att[k] for k in ks])), float(np.mean([self.dfn[k] for k in ks]))


def _season_start(d):
    return pd.Timestamp(year=d.year if d.month >= 7 else d.year - 1, month=7, day=1)


def fit(g, asof, params=None, iters=60, values=None, boost=None):
    """g: hist.games() ciktisi. Yalnizca asof'tan onceki maclar kullanilir.
    values: team_key -> log(kadro degeri) (value.ValueBook.at); verilirse on bilgi
    lig ortalamasi + val_att/val_def * (takim degeri - lig ortalama degeri) olur."""
    p = {**DEFAULTS, **(params or {})}
    asof = pd.Timestamp(asof)
    d = g[(g.date < asof) & (g.date >= asof - pd.Timedelta(days=p["window_days"]))]
    teams = pd.Index(pd.unique(np.concatenate([d.home.values, d.away.values])))
    hi, ai = teams.get_indexer(d.home), teams.get_indexer(d.away)
    leagues = pd.Index(sorted(d.league.unique()))
    li = leagues.get_indexer(d.league)
    age = (asof - d.date).dt.days.values
    w = 0.5 ** (age / p["half_life"])
    # yeni hoca: o takimin yeni hocayla oynadigi maclara ek agirlik (new_coach_w). 2022-26 denemesinde
    # 2 kat agirlik isabeti dusurdu (0.9755 -> 0.9773); varsayilan 1 = kapali, hoca degisimi yalnizca kaydedilir
    if p.get("new_coach_w", 1.0) != 1.0:
        for team, since in (boost or {}).items():
            sel = ((d.home.values == team) | (d.away.values == team)) & (d.date.values >= np.datetime64(since))
            w = np.where(sel, w * p["new_coach_w"], w)

    def target(goals, xg):
        x = xg.values.astype(float)
        y = goals.values.astype(float)
        return np.where(np.isnan(x), y, (1 - p["xg_w"]) * y + p["xg_w"] * x)

    yh, ya = target(d.hg, d.hxg), target(d.ag, d.axg)

    # takimin guncel ligi ve yukselen takim isareti
    dom = d[~d.league.isin(["UCL", "UEL"])]
    long = pd.concat([dom[["date", "league", "home"]].rename(columns={"home": "t"}),
                      dom[["date", "league", "away"]].rename(columns={"away": "t"})]).sort_values("date")
    league_of = long.groupby("t").league.last().to_dict()
    ss = _season_start(asof - pd.Timedelta(days=1))
    prev = long[(long.date < ss) & (long.date >= ss - pd.Timedelta(days=330))]
    prev_set = set(zip(prev.t, prev.league))
    newcomer = np.array([(t in league_of) and ((t, league_of[t]) not in prev_set) and len(prev) > 0
                         for t in teams])
    grp = pd.Index(sorted(set(league_of.values()) | {"OTHER"}))
    tg = grp.get_indexer([league_of.get(t, "OTHER") for t in teams])

    n = len(teams)
    # kadro degeri: lig icinde ortalamadan sapma (log)
    z = np.zeros(n)
    if values:
        lv = np.array([values.get(t, np.nan) for t in teams])
        for j in range(len(grp)):
            sel = (tg == j) & ~np.isnan(lv)
            if sel.sum() >= 3:
                z[sel] = lv[sel] - lv[sel].mean()
        z = np.clip(z, -2.5, 2.5)
    A, D = np.zeros(n), np.zeros(n)
    H = np.full(len(leagues), p["hfa_prior"])
    c = np.log(max(np.average(np.r_[yh, ya], weights=np.r_[w, w]), 0.1))
    k = p["prior_k"]
    wt = np.bincount(hi, w, n) + np.bincount(ai, w, n)
    for _ in range(iters):
        # lig ortalamasi (yeterli verisi olan takimlar) -> prior
        est = wt > 5
        ma = np.array([A[(tg == j) & est].mean() if ((tg == j) & est).any() else 0.0 for j in range(len(grp))])
        md = np.array([D[(tg == j) & est].mean() if ((tg == j) & est).any() else 0.0 for j in range(len(grp))])
        PA = ma[tg] + np.where(newcomer, p["new_off_att"], 0.0) + p["val_att"] * z
        PD = md[tg] + np.where(newcomer, p["new_off_def"], 0.0) + p["val_def"] * z
        eh = np.exp(c + H[li] + A[hi] - D[ai])
        ea = np.exp(c + A[ai] - D[hi])
        obs = np.bincount(hi, w * yh, n) + np.bincount(ai, w * ya, n)
        exp_ = np.bincount(hi, w * eh, n) + np.bincount(ai, w * ea, n)
        A_new = A + np.log((obs + k * np.exp(c + PA)) / (exp_ + k * np.exp(c + A)))
        obs_c = np.bincount(ai, w * yh, n) + np.bincount(hi, w * ya, n)
        exp_c = np.bincount(ai, w * eh, n) + np.bincount(hi, w * ea, n)
        D_new = D - np.log((obs_c + k * np.exp(c - PD)) / (exp_c + k * np.exp(c - D)))
        A, D = 0.5 * A + 0.5 * A_new, 0.5 * D + 0.5 * D_new
        eh = np.exp(c + H[li] + A[hi] - D[ai])
        ea = np.exp(c + A[ai] - D[hi])
        num = np.bincount(li, w * yh, len(leagues)) + 30 * np.exp(p["hfa_prior"])
        den = np.bincount(li, w * eh, len(leagues)) + 30 * np.exp(H)
        H = H + 0.5 * np.log(num / den)
        c += np.log((w * (yh + ya)).sum() / (w * (eh + ea)).sum())
    r = Ratings(asof, float(c), dict(zip(leagues, H)), dict(zip(teams, A)), dict(zip(teams, D)),
                league_of, dict(zip(teams, wt)), p)
    r.values = values or {}
    return r
