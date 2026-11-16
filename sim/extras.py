"""Alti ek parametrenin tek tek testi (22.10.2026, Salim'in istegi).

  .venv/bin/python -m sim.extras

Her parametre ayri eklenir; mac oncesi puan modeli (pred_hist) uzerine L2 cezali Poisson duzeltmesi
2021/22-2023/24 maclariyla egitilir, 2024/25 ve 2025/26'da ayri ayri sinanir. Yalnizca IKI SEZONDA DA
tutarli iyilestiren eklenmeye aday olur.
 1) Bahis oranlari: backtest tablolarindaki motor olasiliklari ile piyasanin harmani
 2) xG agirligi: guc modelinde gol/xG karisimi (0.5 / 0.75 / 1.0)
 3) Yorgunluk: dinlenme gunu farki, hafta ici Avrupa maci, Avrupa deplasmanindan donus
 4) Motivasyon: sezon sonu (son %25) kume / sampiyonluk yarisi, "hicbir seye oynamiyor", derbi
 5) Kesin ilk 11: macin gercek ilk 11 degeri / son 10 mac ortalamasi vs bir onceki macin 11'i
 6) Hakem: hakemin gecmis kirmizi kart ve gol ortalamasi -> kirmizi kart ve 2.5 ust tahmini
    Hava durumu: gecmis veri yok (FotMob sayfasinda var ama arsivlenmemis), test edilemedi.
"""
import numpy as np
import pandas as pd

from . import calibrate, hist, rolling
from .common import load_params, norm, sim_con, src_con
from .rolling import outcome_probs, score_matrix
from .tactics import fit_poisson

TESTS = (("2024-08-01", "2025-07-01"), ("2025-08-01", "2026-07-01"))
DERBY_NAMES = [("Galatasaray", "Fenerbahce"), ("Galatasaray", "Besiktas"), ("Fenerbahce", "Besiktas"),
               ("Manchester City", "Manchester United"), ("Liverpool", "Everton"), ("Arsenal", "Tottenham"),
               ("Arsenal", "Chelsea"), ("Chelsea", "Tottenham"), ("Real Madrid", "Atletico"), ("Real Madrid", "Barcelona"),
               ("Barcelona", "Espanyol"), ("Sevilla", "Betis"), ("Inter", "Milan"), ("Roma", "Lazio"), ("Juventus", "Torino"),
               ("Juventus", "Inter"), ("Dortmund", "Schalke"), ("Bayern", "Dortmund"), ("Paris Saint", "Marseille"),
               ("Lyon", "Saint-Etienne"), ("Genoa", "Sampdoria"), ("Napoli", "Roma")]


def base():
    g = hist.games()
    pred = pd.read_sql("SELECT * FROM pred_hist", sim_con())
    xw = load_params().get("ratings", {}).get("xg_w", 0.75)
    t = calibrate.team_rows(g, pred, xw)
    t = t.merge(g[["game_id", "home", "away", "hg", "ag", "season"]], on="game_id")
    t["opp"] = np.where(t.is_home == 1, t.away, t.home)
    t["goals"] = np.where(t.is_home == 1, t.hg, t.ag)
    return g, t


def eval_games(t, adj, label, test):
    d = t.assign(adj=adj)
    h, a = d[d.is_home == 1].set_index("game_id"), d[d.is_home == 0].set_index("game_id")
    ids = h.index.intersection(a.index)
    h, a = h.loc[ids], a.loc[ids]
    mh, ma = (h.mu_for * h.adj).values, (a.mu_for * a.adj).values
    P3 = score_matrix(mh, ma)
    P = outcome_probs(P3)
    y = np.where(h.hg > h.ag, 0, np.where(h.hg == h.ag, 1, 2))
    i, j = np.indices(P3.shape[1:])
    po = np.clip((P3 * ((i + j) > 2)).sum((1, 2)), 1e-6, 1 - 1e-6)
    yo = (h.hg + h.ag).values > 2.5
    return dict(test=test, model=label, mac=len(ids),
                logloss=round(float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)))), 4),
                isabet=round(float((P.argmax(1) == y).mean()), 4),
                ust25_ll=round(float(-np.mean(yo * np.log(po) + (1 - yo) * np.log(1 - po))), 4))


def run_feature(t, X, label, lam=30.0):
    """X: takim satiri basina ozellikler. Her test sezonu icin egit + sina, mevcut ile karsilastir."""
    rows = []
    X = X.fillna(0).clip(-5, 5)
    for ts, te_ in TESTS:
        tr = (t.date < pd.Timestamp(ts) - pd.Timedelta(days=20)).values
        te = ((t.date >= ts) & (t.date < te_)).values
        b = fit_poisson(X.values[tr], t.goals.values[tr].astype(float), np.log(t.mu_for.values[tr]), lam=lam)
        rows.append(eval_games(t[te], np.ones(te.sum()), "mevcut", ts[:4]))
        r = eval_games(t[te], np.exp(X.values[te] @ b), label, ts[:4])
        r["carpanlar"] = {c: round(float(np.exp(v)), 3) for c, v in zip(X.columns, b)}
        rows.append(r)
    return rows


# ---------------------------------------------------------------- 1) bahis oranlari
def test_odds():
    from .backtest import market_compare
    out = []
    for table, (ts, te_) in (("bt3_2425", TESTS[0]), ("bt3_2526", TESTS[1])):
        b = pd.read_sql(f"SELECT * FROM {table}", sim_con())
        g = hist.games().rename(columns={"p_h": "mk_h", "p_d": "mk_d", "p_a": "mk_a"})
        m = b.merge(g, on="game_id")
        r = market_compare(m)
        out.append(dict(test=ts[:4], mac=r["piyasali_mac"], motor=round(r["motor_1x2_ll"], 4),
                        piyasa=round(r["piyasa_1x2_ll"], 4), karisim_20=round(r["karisim_0.2_ll"], 4),
                        karisim_35=round(r["karisim_0.35_ll"], 4), karisim_50=round(r["karisim_0.5_ll"], 4),
                        motor_ust=round(r.get("motor_ust25_ll", np.nan), 4), piyasa_ust=round(r.get("piyasa_ust25_ll", np.nan), 4)))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 2) xG agirligi
def test_xg():
    g = hist.games()
    out = []
    base_p = load_params().get("ratings", {})
    for w in (0.5, 0.75, 1.0):
        pr = rolling.rolling(g, {**base_p, "xg_w": w}, start="2024-08-01")
        m = g.merge(pr, on="game_id")
        for ts, te_ in TESTS:
            s = m[(m.date >= ts) & (m.date < te_)]
            P = outcome_probs(score_matrix(s.mu_h, s.mu_a))
            y = np.where(s.hg > s.ag, 0, np.where(s.hg == s.ag, 1, 2))
            out.append(dict(test=ts[:4], xg_agirligi=w, mac=len(s),
                            logloss=round(float(-np.mean(np.log(P[np.arange(len(y)), y]))), 4),
                            isabet=round(float((P.argmax(1) == y).mean()), 4)))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 3) yorgunluk
def fatigue(g, t):
    long = pd.concat([g[["date", "league", "home", "game_id"]].rename(columns={"home": "team"}).assign(away=0),
                      g[["date", "league", "away", "game_id"]].rename(columns={"away": "team"}).assign(away=1)])
    long = long.sort_values("date")
    long["prev_date"] = long.groupby("team").date.shift()
    long["prev_league"] = long.groupby("team").league.shift()
    long["prev_away"] = long.groupby("team").away.shift()
    long["rest"] = (long.date - long.prev_date).dt.days.clip(upper=14)
    long["euro_mid"] = ((long.prev_league == "UCL") & (long.rest <= 4) & (long.league != "UCL")).astype(float)
    long["euro_away"] = (long.euro_mid * long.prev_away).astype(float)
    f = long[["game_id", "team", "rest", "euro_mid", "euro_away"]]
    x = t[["game_id", "team", "opp"]].merge(f, on=["game_id", "team"], how="left")
    x = x.merge(f.rename(columns={"team": "opp", "rest": "o_rest", "euro_mid": "o_euro_mid", "euro_away": "o_euro_away"}),
                on=["game_id", "opp"], how="left")
    X = pd.DataFrame({"dinlenme_farki_gun": (x.rest - x.o_rest).clip(-7, 7) / 3,
                      "kisa_dinlenme_3gun_alti": (x.rest <= 3).astype(float),
                      "rakip_kisa_dinlenme": (x.o_rest <= 3).astype(float),
                      "hafta_ici_avrupa": x.euro_mid, "rakip_hafta_ici_avrupa": x.o_euro_mid,
                      "avrupa_deplasmanindan_donus": x.euro_away}, index=t.index)
    return X


# ---------------------------------------------------------------- 4) motivasyon
def standings_features(g, t):
    dom = g[g.league != "UCL"].sort_values("date")
    rows = []
    for (lg, s), d in dom.groupby(["league", "season"]):
        teams = pd.unique(np.r_[d.home.values, d.away.values])
        n = len(teams)
        total_rounds = 2 * (n - 1)
        pts = dict.fromkeys(teams, 0)
        gp = dict.fromkeys(teams, 0)
        for r in d.itertuples():
            table = sorted(pts.values(), reverse=True)
            top, safe = table[0], table[-4] if n > 4 else 0  # 17. (ya da sondan 4.) siradaki puan ~ guvenli bolge
            for team, opp in ((r.home, r.away), (r.away, r.home)):
                late = gp[team] >= 0.75 * total_rounds
                title = late and top - pts[team] <= 6
                releg = late and pts[team] - safe <= 3
                rows.append(dict(game_id=r.game_id, team=team, late=float(late), title=float(title), releg=float(releg),
                                 nothing=float(late and not title and not releg)))
            hp = 3 if r.hg > r.ag else (1 if r.hg == r.ag else 0)
            ap = 3 if r.ag > r.hg else (1 if r.hg == r.ag else 0)
            pts[r.home] += hp
            pts[r.away] += ap
            gp[r.home] += 1
            gp[r.away] += 1
    f = pd.DataFrame(rows)
    x = t[["game_id", "team", "opp"]].merge(f, on=["game_id", "team"], how="left")
    x = x.merge(f.rename(columns={"team": "opp", "title": "o_title", "releg": "o_releg", "nothing": "o_nothing"})
                [["game_id", "opp", "o_title", "o_releg", "o_nothing"]], on=["game_id", "opp"], how="left")
    # derbi
    clubs = pd.read_sql("SELECT club_id, name FROM tm_clubs", src_con())
    clubs["n"] = clubs.name.map(norm)
    def ids(name):
        k = norm(name)
        return {f"tm:{c}" for c in clubs[clubs.n.str.contains(k, regex=False)].club_id}
    pairs = set()
    for a, b in DERBY_NAMES:
        for i in ids(a):
            for j in ids(b):
                pairs |= {(i, j), (j, i)}
    derby = np.array([(tt, oo) in pairs for tt, oo in zip(x.team, x.opp)], float)
    X = pd.DataFrame({"sampiyonluk_yarisi": x.title, "kume_yarisi": x.releg, "hicbir_seye_oynamiyor": x.nothing,
                      "rakip_sampiyonluk_yarisi": x.o_title, "rakip_kume_yarisi": x.o_releg,
                      "rakip_hicbir_seye_oynamiyor": x.o_nothing, "derbi": derby}, index=t.index)
    return X


# ---------------------------------------------------------------- 5) kesin ilk 11
def lineup_features(t):
    con = src_con()
    lu = pd.read_sql("SELECT game_id, club_id, player_id, date FROM tm_lineups WHERE type='starting_lineup' "
                     "AND date >= '2021-06-01'", con)
    val = pd.read_sql("SELECT player_id, date, market_value_in_eur v FROM tm_valuations", con)
    lu["date"], val["date"] = pd.to_datetime(lu.date), pd.to_datetime(val.date)
    lu = pd.merge_asof(lu.sort_values("date"), val.sort_values("date"), on="date", by="player_id",
                       allow_exact_matches=False)
    xi = lu.groupby(["game_id", "club_id", "date"]).v.sum(min_count=8).reset_index().sort_values("date")
    xi["usual"] = xi.groupby("club_id").v.transform(lambda s: s.shift(1).rolling(10, min_periods=4).mean())
    xi["prev"] = xi.groupby("club_id").v.shift(1)
    xi["team"] = "tm:" + xi.club_id.astype(str)
    xi["game_id"] = xi.game_id.astype(str)
    xi["r_kesin"] = np.log((xi.v / xi.usual).clip(0.5, 1.6))
    xi["r_onceki"] = np.log((xi.prev / xi.usual).clip(0.5, 1.6))
    f = xi[["game_id", "team", "r_kesin", "r_onceki"]]
    x = t[["game_id", "team", "opp"]].merge(f, on=["game_id", "team"], how="left")
    x = x.merge(f.rename(columns={"team": "opp", "r_kesin": "o_kesin", "r_onceki": "o_onceki"}), on=["game_id", "opp"], how="left")
    return (pd.DataFrame({"kesin_11_orani_farki": x.r_kesin - x.o_kesin}, index=t.index),
            pd.DataFrame({"onceki_mac_11_orani_farki": x.r_onceki - x.o_onceki}, index=t.index))


# ---------------------------------------------------------------- 6) hakem
def test_referee(g):
    con = src_con()
    tg = pd.read_sql("SELECT game_id, date, referee, home_club_goals hg, away_club_goals ag FROM tm_games", con)
    reds = pd.read_sql("SELECT game_id, COUNT(*) r FROM tm_events WHERE type='Cards' AND "
                       "(description LIKE '%Red card%' OR description LIKE '%Second yellow%') GROUP BY 1", con)
    tg = tg.merge(reds, on="game_id", how="left").fillna({"r": 0}).sort_values("date")
    tg["date"] = pd.to_datetime(tg.date)
    tg["red"] = (tg.r > 0).astype(float)
    tg["tot"] = tg.hg + tg.ag
    grp = tg.groupby("referee")
    k = 10.0
    tg["ref_red"] = ((grp.red.cumsum() - tg.red) + k * tg.red.mean()) / (grp.cumcount() + k)
    tg["ref_goals"] = ((grp.tot.cumsum() - tg.tot) + k * tg.tot.mean()) / (grp.cumcount() + k)
    tg["game_id"] = tg.game_id.astype(str)
    out = []
    for table, (ts, te_) in (("bt3_2425", TESTS[0]), ("bt3_2526", TESTS[1])):
        b = pd.read_sql(f"SELECT game_id, p_red, over25 FROM {table}", sim_con()).merge(tg, on="game_id")
        tr = tg[(tg.date < pd.Timestamp(ts) - pd.Timedelta(days=20))]
        # kirmizi kart: logit(motor p_red) + c * (hakem orani - ortalama); c egitimden (basit tarama)
        def ll(p, y):
            p = np.clip(p, 1e-4, 1 - 1e-4)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        mean_r, mean_g = tr.red.mean(), tr.tot.mean()
        best_c, best_l = 0.0, None
        trb = tr.dropna(subset=["ref_red"])
        base_p = np.full(len(trb), trb.red.mean())
        for c in np.arange(0, 12.1, 0.5):
            p = 1 / (1 + np.exp(-(np.log(base_p / (1 - base_p)) + c * (trb.ref_red - mean_r))))
            l_ = ll(p, trb.red.values)
            if best_l is None or l_ < best_l:
                best_c, best_l = c, l_
        lp = np.log(b.p_red / (1 - b.p_red))
        p_new = 1 / (1 + np.exp(-(lp + best_c * (b.ref_red - mean_r))))
        # 2.5 ust: logit(motor over25) + d * (hakem gol ort - ortalama); egitim donemi icin motor backtest'i
        # olmadigindan birkac d degeri iki sezonda da gosterilir (ayni d iki sezonda da iyiyse anlamli)
        o_rows = []
        for d in (0.0, 0.1, 0.2, 0.3):
            po = 1 / (1 + np.exp(-(np.log(b.over25 / (1 - b.over25)) + d * (b.ref_goals - mean_g))))
            o_rows.append((d, ll(po, (b.tot > 2.5).astype(float).values)))
        out.append(dict(test=ts[:4], mac=len(b), kirmizi_mevcut=round(ll(b.p_red, b.red.values), 4),
                        kirmizi_hakemli=round(ll(p_new, b.red.values), 4), hakem_katsayisi=best_c,
                        ust25_mevcut=round(o_rows[0][1], 4),
                        ust25_hakemli={d: round(v, 4) for d, v in o_rows[1:]}))
    return pd.DataFrame(out)


def main():
    res = {}
    print("1) BAHIS ORANLARI (motor / piyasa / harman, 1X2 log-loss)")
    res["oran"] = test_odds()
    print(res["oran"].to_string(index=False))
    print("\n2) xG AGIRLIGI (guc modeli, Poisson 1X2)")
    res["xg"] = test_xg()
    print(res["xg"].to_string(index=False))
    g, t = base()
    print("\n3) YORGUNLUK")
    rows = run_feature(t, fatigue(g, t), "yorgunluk")
    res["yorgunluk"] = pd.DataFrame(rows)
    print(res["yorgunluk"].drop(columns="carpanlar").to_string(index=False)); print("  carpanlar:", rows[-1]["carpanlar"])
    print("\n4) MOTIVASYON + DERBI")
    rows = run_feature(t, standings_features(g, t), "motivasyon")
    res["motivasyon"] = pd.DataFrame(rows)
    print(res["motivasyon"].drop(columns="carpanlar").to_string(index=False)); print("  carpanlar:", rows[-1]["carpanlar"])
    print("\n5) KESIN ILK 11 vs ONCEKI MACIN 11'I")
    kesin, onceki = lineup_features(t)
    r1 = run_feature(t, onceki, "onceki macin 11'i")
    r2 = run_feature(t, kesin, "kesin ilk 11")
    res["ilk11"] = pd.DataFrame(r1 + [r for r in r2 if r["model"] != "mevcut"])
    print(res["ilk11"].drop(columns="carpanlar").to_string(index=False))
    print("  carpanlar (kesin):", r2[-1]["carpanlar"], " (onceki):", r1[-1]["carpanlar"])
    print("\n6) HAKEM (kirmizi kart ve 2.5 ust log-loss)")
    res["hakem"] = test_referee(g)
    print(res["hakem"].to_string(index=False))
    con = sim_con()
    for k, v in res.items():
        v.astype(str).to_sql(f"extras_{k}", con, if_exists="replace", index=False)
    con.commit()


if __name__ == "__main__":
    main()
