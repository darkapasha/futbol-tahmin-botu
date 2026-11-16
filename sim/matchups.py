"""Oyuncu eslesme testi: iyi bir kanat oyuncusu zayif bir beke karsi oynarsa takim (ve oyuncu) daha cok gol bulur mu?

  .venv/bin/python -m sim.matchups

Her Transfermarkt macinda iki takimin ilk 11'i mevkiye gore karsi karsiya getirilir; oyuncu kalitesi
icin o tarihteki piyasa degeri (log) kullanilir:
  sol kanat (Left Winger / Left Midfield) x rakip sag bek (Right-Back; yoksa sag kanat bek / Right Midfield)
  sag kanat x rakip sol bek, forvetler x rakip stoperler, orta saha x rakip orta saha
Ozellik = kendi oyuncumuzun degeri - karsisindakinin degeri (log). Ilk 11'in genel deger farki da ayrica
konur; boylece "takim zaten daha iyi" etkisi eslesme etkisinden ayrilir.

1) Takim duzeyi: atilan gol ~ Poisson(mu x exp(b . x)), mu = mac oncesi puan modeli (+ kadro orani).
   2021/22-2023/24'te egitilir, 2024/25 ve 2025/26'da 1X2 log-loss / isabet ile mevcut modelle karsilastirilir.
2) Oyuncu duzeyi: kanat oyuncusunun gol atma olasiligi, karsisindaki bekin zayifligiyla artiyor mu?
"""
import numpy as np
import pandas as pd

from . import calibrate, hist
from .common import load_params, sim_con, src_con
from .tactics import evaluate, fit_poisson

LEFT_ATT = {"Left Winger", "Left Midfield"}
RIGHT_ATT = {"Right Winger", "Right Midfield"}
FWD = {"Centre-Forward", "Second Striker"}
CB = {"Centre-Back", "Sweeper"}
MID = {"Central Midfield", "Defensive Midfield", "Attacking Midfield"}


def lineups():
    con = src_con()
    lu = pd.read_sql("SELECT game_id, club_id, player_id, date, position FROM tm_lineups "
                     "WHERE type='starting_lineup' AND date >= '2021-07-01'", con)
    val = pd.read_sql("SELECT player_id, date, market_value_in_eur v FROM tm_valuations", con)
    lu["date"], val["date"] = pd.to_datetime(lu.date), pd.to_datetime(val.date)
    lu = pd.merge_asof(lu.sort_values("date"), val.sort_values("date"), on="date", by="player_id",
                       allow_exact_matches=False)
    lu["lv"] = np.log(lu.v.clip(lower=1e5))
    return lu


def unit_values(lu):
    """mac x takim: her bolgenin (log) degeri."""
    def agg(g):
        p = g.position
        def pick(sets, how="max"):
            s = g.lv[p.isin(sets)]
            return (s.max() if how == "max" else s.mean()) if len(s) else np.nan
        return pd.Series({
            "left_att": pick(LEFT_ATT), "right_att": pick(RIGHT_ATT), "fwd": pick(FWD), "mid": pick(MID, "mean"),
            "rb": pick({"Right-Back"}), "lb": pick({"Left-Back"}), "rm": pick({"Right Midfield"}), "lm": pick({"Left Midfield"}),
            "cb": pick(CB, "mean"), "xi": g.lv.mean(), "n": len(g)})
    u = lu.groupby(["game_id", "club_id"]).apply(agg).reset_index()
    # bek yoksa (3'lu savunma) kanat bek, o da yoksa stoper ortalamasi
    u["rdef"] = u.rb.fillna(u.rm).fillna(u.cb)
    u["ldef"] = u.lb.fillna(u.lm).fillna(u.cb)
    return u


def build():
    g = hist.games()
    pred = pd.read_sql("SELECT * FROM pred_hist", sim_con())
    xw = load_params().get("ratings", {}).get("xg_w", 0.75)
    t = calibrate.team_rows(g[g.source == "tm"], pred, xw)
    lu = lineups()
    u = unit_values(lu)
    u["team"] = "tm:" + u.club_id.astype(int).astype(str)
    u["game_id"] = u.game_id.astype(str)
    t = t.merge(u.drop(columns="club_id"), on=["game_id", "team"])
    opp = u.drop(columns="club_id").rename(columns=lambda c: c if c == "game_id" else f"o_{c}")
    t = t.merge(g[["game_id", "home", "away"]], on="game_id")
    t["o_team"] = np.where(t.is_home == 1, t.away, t.home)
    t = t.merge(opp, on=["game_id", "o_team"])
    t = t[(t.n >= 10) & (t.o_n >= 10)]
    return t, lu


def features(t):
    X = pd.DataFrame(index=t.index)
    X["xi_farki"] = (t.xi - t.o_xi).fillna(0)
    X["sol_kanat_x_sag_bek"] = (t.left_att - t.o_rdef).fillna(0)
    X["sag_kanat_x_sol_bek"] = (t.right_att - t.o_ldef).fillna(0)
    X["forvet_x_stoper"] = (t.fwd - t.o_cb).fillna(0)
    X["orta_saha_x_orta_saha"] = (t.mid - t.o_mid).fillna(0)
    # en buyuk kanat ustunlugu (Trossard ornegi: bir kanatta belirgin fark)
    X["en_iyi_kanat_ustunlugu"] = np.fmax(X.sol_kanat_x_sag_bek, X.sag_kanat_x_sol_bek).clip(lower=0)
    return X.clip(-3, 3)


def player_test(t, lu):
    """Kanat oyuncusu gol atti mi ~ kendi degeri + karsisindaki bekin degeri (lojistik, egitim/test)."""
    con = src_con()
    goals = pd.read_sql("SELECT game_id, player_id, COUNT(*) g FROM tm_events WHERE type='Goals' GROUP BY 1, 2", con)
    goals["game_id"] = goals.game_id.astype(str)
    w = lu[lu.position.isin(LEFT_ATT | RIGHT_ATT)].copy()
    w["game_id"] = w.game_id.astype(str)
    w["team"] = "tm:" + w.club_id.astype(str)
    side = t[["game_id", "team", "o_rdef", "o_ldef", "date"]]
    w = w.drop(columns="date").merge(side, on=["game_id", "team"])
    w["opp_def"] = np.where(w.position.isin(LEFT_ATT), w.o_rdef, w.o_ldef)
    w = w.merge(goals, on=["game_id", "player_id"], how="left")
    w["scored"] = (w.g.fillna(0) > 0).astype(float)
    w = w.dropna(subset=["lv", "opp_def"])
    out = []
    for ts in ("2024-08-01", "2025-08-01"):
        tr, te = w[w.date < ts], w[(w.date >= ts) & (w.date < str(int(ts[:4]) + 1) + "-07-01")]
        for cols, name in ((["lv"], "yalniz oyuncu degeri"), (["lv", "opp_def"], "oyuncu + karsisindaki bek")):
            Xtr = np.c_[np.ones(len(tr)), tr[cols].values]
            Xte = np.c_[np.ones(len(te)), te[cols].values]
            b = np.zeros(Xtr.shape[1])
            for _ in range(25):  # lojistik Newton
                p = 1 / (1 + np.exp(-Xtr @ b))
                H = (Xtr * (p * (1 - p))[:, None]).T @ Xtr + 1e-3 * np.eye(len(b))
                b += np.linalg.solve(H, Xtr.T @ (tr.scored.values - p))
            p = np.clip(1 / (1 + np.exp(-Xte @ b)), 1e-6, 1 - 1e-6)
            ll = -np.mean(te.scored * np.log(p) + (1 - te.scored) * np.log(1 - p))
            out.append(dict(test=ts[:4], model=name, oyuncu_mac=len(te), logloss=round(float(ll), 5),
                            bek_katsayisi=round(float(b[-1]), 3) if len(cols) == 2 else None))
    return pd.DataFrame(out)


def main():
    t, lu = build()
    X = features(t)
    y, off = t.y_for.values, np.log(t.mu_for.values)
    rows, coefs = [], None
    for ts, te_ in (("2024-08-01", "2025-07-01"), ("2025-08-01", "2026-07-01")):
        tr = (t.date < pd.Timestamp(ts) - pd.Timedelta(days=20)).values
        te = ((t.date >= ts) & (t.date < te_)).values
        r = evaluate(t[te], X[te], None, "mevcut model")
        r["test"] = ts[:4]
        rows.append(r)
        for cols, name in ((["xi_farki"], "yalniz ilk 11 deger farki"), (list(X.columns), "oyuncu eslesmeleri")):
            b = fit_poisson(X[cols].values[tr], y[tr], off[tr], lam=30)
            r = evaluate(t[te], X[cols][te], b, name)
            r["test"] = ts[:4]
            rows.append(r)
            if name == "oyuncu eslesmeleri":
                coefs = dict(zip(cols, b))
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    print("\neslesme kat sayilari (1 log-birim deger farki basina gol carpani):")
    for k, v in coefs.items():
        print(f"  {k:26s} {np.exp(v):.3f}")
    pt = player_test(t, lu)
    print("\noyuncu duzeyi (kanatci gol atti mi):")
    print(pt.to_string(index=False))
    con = sim_con()
    res.to_sql("matchup_test", con, if_exists="replace", index=False)
    pt.to_sql("matchup_player_test", con, if_exists="replace", index=False)
    con.commit()


if __name__ == "__main__":
    main()
