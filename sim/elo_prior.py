"""Kapsadigimiz 7 lig disindaki takimlar (Avrupa kupalari: Ararat, Ferencvaros...) icin guc on bilgisi.

ClubElo puani ile bizim hucum/savunma puanimiz arasindaki iliski, iki kaynakta da olan takimlardan
dogrusal olarak kestirilir; Elo'su bulunan ama macimiz olmayan takim bu iliskiyle puanlanir.
ClubElo'da bulunamayan takim icin ilk 11'inin piyasa degeri kullanilir (FotMob mac oncesi kadrosu):
log(ilk 11 degeri) -> hucum/savunma iliskisi, FotMob'da son ilk 11'i olan bilinen takimlardan kestirilir.
Ikisi de yoksa kupalarin alt sirasi icin tipik bir Elo (DEFAULT_ELO).
"""
import difflib

import numpy as np
import pandas as pd

from .common import norm, src_con

DEFAULT_ELO = 1430.0


class EloPrior:
    def __init__(self, ratings, asof, con=None):
        con = con or src_con()
        e = pd.read_sql("SELECT date, club, country, elo FROM elo WHERE date <= ?", con,
                        params=(pd.Timestamp(asof).strftime("%Y-%m-%d"),))
        last = e.sort_values("date").groupby("club").last()
        last = last[last.date >= (pd.Timestamp(asof) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")]
        self.elo = last.elo
        self.names = {norm(c): c for c in last.index}
        # Elo -> (hucum, savunma): xgabora maclarindaki mac oncesi Elo ile team_key eslesmesi uzerinden
        x = pd.read_sql("SELECT matchdate, hometeam t, homeelo elo FROM xg_matches WHERE homeelo IS NOT NULL "
                        "UNION ALL SELECT matchdate, awayteam, awayelo FROM xg_matches WHERE awayelo IS NOT NULL", con)
        km = pd.read_sql("SELECT name, team_key FROM team_map WHERE source='xgabora'", con)
        x["key"] = x.t.map(dict(zip(km.name, km.team_key)))
        x = x[x.matchdate < pd.Timestamp(asof).strftime("%Y-%m-%d")].sort_values("matchdate").groupby("key").last()
        known = [(k, r.elo) for k, r in x.iterrows() if k in ratings.att and ratings.weight.get(k, 0) > 8]
        if len(known) >= 20:
            el = np.array([v for _, v in known])
            A = np.array([ratings.att[k] for k, _ in known])
            D = np.array([ratings.dfn[k] for k, _ in known])
            X = np.c_[np.ones(len(el)), (el - 1500) / 100]
            self.ca = np.linalg.lstsq(X, A, rcond=None)[0]
            self.cd = np.linalg.lstsq(X, D, rcond=None)[0]
        else:
            self.ca = self.cd = None

    def fit_value(self, ratings, resolver, con):
        """Bilinen takimlarin son FotMob ilk 11 degeri -> (hucum, savunma) dogrusal iliski."""
        q = ("SELECT l.team_id, m.home_id, m.home, m.away_id, m.away, m.league, SUM(l.market_value) v, m.utc_time "
             "FROM fm_lineups l JOIN fm_matches m USING (match_id) WHERE l.role='starter' "
             "AND m.utc_time >= '2026-07-01' GROUP BY l.match_id, l.team_id")
        x = pd.read_sql(q, con)
        x = x[x.v > 0].sort_values("utc_time").groupby("team_id").last()
        rows = []
        for tid, r in x.iterrows():
            name = r.home if r.home_id == tid else r.away
            k = resolver.fotmob(tid, name, r.league)
            if k in ratings.att and ratings.weight.get(k, 0) > 8:
                rows.append((np.log(r.v), ratings.att[k], ratings.dfn[k]))
        if len(rows) < 20:
            self.va = self.vd = None
            return len(rows)
        a = np.array(rows)
        X = np.c_[np.ones(len(a)), a[:, 0] - 18]
        self.va = np.linalg.lstsq(X, a[:, 1], rcond=None)[0]
        self.vd = np.linalg.lstsq(X, a[:, 2], rcond=None)[0]
        return len(rows)

    def find(self, name):
        n = norm(name)
        if n in self.names:
            return self.elo[self.names[n]]
        m = difflib.get_close_matches(n, list(self.names), n=1, cutoff=0.82)
        if m:
            return self.elo[self.names[m[0]]]
        # "Viktoria Plzen" -> "Plzen", "Ferencvaros" -> "Ferencvaros": son kelime / ilk kelime dene
        for part in (n.split()[-1], n.split()[0]) if n else ():
            if len(part) >= 4 and part in self.names:
                return self.elo[self.names[part]]
        return None

    def rating(self, name, xi_value=None):
        """(hucum, savunma, kaynak). Elo ve ilk 11 degeri ikisi de varsa ortalamasi."""
        est = []
        elo = self.find(name)
        if elo is not None and self.ca is not None:
            x = np.array([1.0, (float(elo) - 1500) / 100])
            est.append((float(x @ self.ca), float(x @ self.cd), f"Elo {float(elo):.0f}"))
        if xi_value and xi_value > 0 and getattr(self, "va", None) is not None:
            x = np.array([1.0, np.log(xi_value) - 18])
            est.append((float(x @ self.va), float(x @ self.vd), f"ilk 11 degeri {xi_value / 1e6:.0f} M EUR"))
        if not est and self.ca is not None:
            x = np.array([1.0, (DEFAULT_ELO - 1500) / 100])
            est.append((float(x @ self.ca), float(x @ self.cd), "varsayilan"))
        if not est:
            return None
        return (float(np.mean([e[0] for e in est])), float(np.mean([e[1] for e in est])),
                " + ".join(e[2] for e in est))
