"""Isabet gelistirme testleri 2 (22.10.2026): acilis / kapanis orani, oran hareketi, kalibrasyon, beraberlik.

  .venv/bin/python -m sim.odds_tests

Oranlar: football-data.co.uk sezon CSV'leri (data/cache/fd_hist, 6 ulke ligi): Avg* = acilis tarafi
(maclardan ~2-3 gun once toplanan ortalama), AvgC* = kapanis (mac basladiginda). Motor olasiliklari
backtest tablolarindan (bt3_2425, bt3_2526). Her ayar bir sezonda secilip OBUR sezonda sinanir.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import hist
from .common import sim_con, src_con
from config import DATA

FD = DATA / "cache" / "fd_hist"
DIV = {"E0": "ENG", "SP1": "ESP", "D1": "GER", "I1": "ITA", "F1": "FRA", "T1": "TUR"}


def load_fd():
    rows = []
    for f in sorted(FD.glob("*.csv")):
        d = pd.read_csv(f, encoding="utf-8-sig", on_bad_lines="skip")
        d["league"] = DIV[f.stem.split("_")[1]]
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d["date"] = pd.to_datetime(d.Date, dayfirst=True, errors="coerce")
    km = pd.read_sql("SELECT name, team_key FROM team_map WHERE source='xgabora'", src_con())
    km = dict(zip(km.name, km.team_key))
    d["home"], d["away"] = d.HomeTeam.map(km), d.AwayTeam.map(km)
    return d.dropna(subset=["home", "away", "date"])


def probs(d, h, x, a):
    o = d[[h, x, a]].astype(float).values
    inv = 1 / o
    return inv / inv.sum(1, keepdims=True)


def ll(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1))))


def temp(P, T):
    Q = P ** (1 / T)
    return Q / Q.sum(1, keepdims=True)


def draw_adj(P, k):
    Q = P.copy()
    Q[:, 1] *= k
    return Q / Q.sum(1, keepdims=True)


def season_data(table, fd):
    b = pd.read_sql(f"SELECT game_id, p_h, p_d, p_a, over25 FROM {table}", sim_con())
    g = hist.games()[["game_id", "date", "home", "away", "hg", "ag"]]
    m = b.merge(g, on="game_id").merge(fd, on=["date", "home", "away"], suffixes=("", "_fd"))
    cols = ["AvgH", "AvgD", "AvgA", "AvgCH", "AvgCD", "AvgCA"]
    m = m.dropna(subset=cols)
    y = np.where(m.hg > m.ag, 0, np.where(m.hg == m.ag, 1, 2))
    E = m[["p_h", "p_d", "p_a"]].values
    E = E / E.sum(1, keepdims=True)
    O, C = probs(m, "AvgH", "AvgD", "AvgA"), probs(m, "AvgCH", "AvgCD", "AvgCA")
    ou = {}
    for k, (a, b_) in {"open": ("Avg>2.5", "Avg<2.5"), "close": ("AvgC>2.5", "AvgC<2.5")}.items():
        if a in m and b_ in m:
            io = 1 / m[[a, b_]].astype(float).values
            ou[k] = io[:, 0] / io.sum(1)
    return dict(m=m, y=y, E=E, O=O, C=C, ou=ou, tot=(m.hg + m.ag).values > 2.5)


def main():
    fd = load_fd()
    S = {"2024/25": season_data("bt3_2425", fd), "2025/26": season_data("bt3_2526", fd)}
    out = []
    for name, s in S.items():
        y, E, O, C = s["y"], s["E"], s["O"], s["C"]
        r = dict(sezon=name, mac=len(y), motor=ll(E, y), acilis=ll(O, y), kapanis=ll(C, y),
                 harman_acilis=ll(0.8 * O + 0.2 * E, y), harman_kapanis=ll(0.8 * C + 0.2 * E, y))
        for w in (0.0, 0.1, 0.2, 0.3):
            r[f"kapanis_motor_{int(w * 100)}"] = ll((1 - w) * C + w * E, y)
        # oran hareketi: kapanis + k*(kapanis-acilis) (hareketin devam ettigi varsayimi)
        for k in (0.25, 0.5):
            M = np.clip(C + k * (C - O), 0.01, 0.98)
            r[f"hareket_k{k}"] = ll(0.8 * (M / M.sum(1, keepdims=True)) + 0.2 * E, y)
        out.append(r)
    res = pd.DataFrame(out).set_index("sezon").T.round(4)
    print("1X2 LOG-LOSS (dusuk iyi)\n", res.to_string())

    # kalibrasyon ve beraberlik: bir sezonda sec, obur sezonda sina (harman_acilis uzerine)
    cal = []
    names = list(S)
    for tr_n, te_n in ((names[0], names[1]), (names[1], names[0])):
        tr, te = S[tr_n], S[te_n]
        Ptr, Pte = 0.8 * tr["O"] + 0.2 * tr["E"], 0.8 * te["O"] + 0.2 * te["E"]
        Ts = np.arange(0.85, 1.16, 0.01)
        T = Ts[np.argmin([ll(temp(Ptr, t), tr["y"]) for t in Ts])]
        ks = np.arange(0.85, 1.16, 0.01)
        k = ks[np.argmin([ll(draw_adj(Ptr, kk), tr["y"]) for kk in ks])]
        cal.append(dict(egitim=tr_n, test=te_n, sicaklik=round(float(T), 2), beraberlik_k=round(float(k), 2),
                        once=round(ll(Pte, te["y"]), 4), sicaklik_sonra=round(ll(temp(Pte, T), te["y"]), 4),
                        beraberlik_sonra=round(ll(draw_adj(Pte, k), te["y"]), 4)))
    print("\nKALIBRASYON / BERABERLIK (bir sezonda ayarla, obur sezonda sina)\n", pd.DataFrame(cal).to_string(index=False))

    # guvenilirlik tablosu (harman, acilis): tahmin edilen vs gercek
    rel = []
    for name, s in S.items():
        P = 0.8 * s["O"] + 0.2 * s["E"]
        for j, lab in enumerate(("1", "X", "2")):
            bins = pd.cut(P[:, j], [0, .2, .3, .4, .5, .6, .7, 1])
            t = pd.DataFrame({"b": bins, "p": P[:, j], "y": (s["y"] == j)}).groupby("b", observed=True).agg(
                n=("y", "size"), tahmin=("p", "mean"), gercek=("y", "mean"))
            t["sonuc"], t["sezon"] = lab, name
            rel.append(t.reset_index())
    rel = pd.concat(rel)
    print("\nGUVENILIRLIK (harman)\n", rel[rel.n >= 30].round(3).to_string(index=False))

    # 2.5 ust: motor / acilis / kapanis / harman
    ou_rows = []
    for name, s in S.items():
        yo = s["tot"]
        f = lambda p: float(-np.mean(yo * np.log(np.clip(p, 1e-6, 1)) + (1 - yo) * np.log(np.clip(1 - p, 1e-6, 1))))
        r = dict(sezon=name, motor=f(s["m"].over25.values))
        for k, p in s["ou"].items():
            r[k] = f(p)
            r[f"harman_{k}"] = f(0.8 * p + 0.2 * s["m"].over25.values)
        ou_rows.append(r)
    print("\n2.5 UST LOG-LOSS\n", pd.DataFrame(ou_rows).round(4).to_string(index=False))
    con = sim_con()
    res.reset_index().to_sql("odds_test", con, if_exists="replace", index=False)
    pd.DataFrame(cal).to_sql("calib_test", con, if_exists="replace", index=False)
    con.commit()


if __name__ == "__main__":
    main()
