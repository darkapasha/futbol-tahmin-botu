"""Surpriz maclar analizi: favorinin kazanamadigi maclarda tekrar eden desen var mi?

  .venv/bin/python -m sim.surpriz

Yalnizca okur (futbol.sqlite, profiles.sqlite, sim.sqlite pred_hist); sonuclari data/sim.sqlite
'surpriz_*' tablolarina yazar. Uretim modeline / params.json'a dokunmaz.
Egitim < 2024-08, sinama 2024/25 ve 2025/26. Bir desen ancak egitimde anlamli (|z|>2) ve
iki sinama sezonunda da ayni yonde ise 'tutuyor' sayilir.
"""
import numpy as np
import pandas as pd

from . import hist
from .common import norm, sim_con, src_con
from .extras import fatigue, standings_features
from .rolling import outcome_probs, score_matrix

PERIODS = {"egitim": ("2021-10-01", "2024-08-01"), "2024/25": ("2024-08-01", "2025-07-01"),
           "2025/26": ("2025-08-01", "2026-07-01")}


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def load():
    g = hist.games()
    pred = pd.read_sql("SELECT game_id, mu_h, mu_a FROM pred_hist", sim_con())
    g = g.merge(pred, on="game_id", how="left")
    d = g[g.p_h.notna() & g.mu_h.notna() & (g.league != "UCL")].copy()
    s = d[["p_h", "p_d", "p_a"]].sum(1)
    for c in ("p_h", "p_d", "p_a"):
        d[c] = d[c] / s
    M = outcome_probs(score_matrix(d.mu_h.values, d.mu_a.values))
    d["m_h"], d["m_d"], d["m_a"] = M[:, 0], M[:, 1], M[:, 2]
    fh = d.p_h >= d.p_a
    d["fav_home"] = fh.astype(float)
    d["fav"], d["dog"] = np.where(fh, d.home, d.away), np.where(fh, d.away, d.home)
    d["p_fav"], d["p_dog"] = np.where(fh, d.p_h, d.p_a), np.where(fh, d.p_a, d.p_h)
    d["m_fav"], d["m_dog"] = np.where(fh, d.m_h, d.m_a), np.where(fh, d.m_a, d.m_h)
    gf, ga = np.where(fh, d.hg, d.ag), np.where(fh, d.ag, d.hg)
    d["fav_win"], d["draw"], d["dog_win"] = (gf > ga).astype(float), (gf == ga).astype(float), (gf < ga).astype(float)
    d["upset"] = 1 - d.fav_win           # favori kazanamadi
    d["p_upset"] = 1 - d.p_fav
    d["donem"] = None
    for k, (a, b) in PERIODS.items():
        d.loc[(d.date >= a) & (d.date < b), "donem"] = k
    return g, d[d.donem.notna()].copy()


def team_long(g):
    """Takim bakisindan uzun tablo: puan, piyasa beklenen puan, gol-xG, hoca."""
    m = g[g.p_h.notna() & (g.league != "UCL")].copy()
    s = m[["p_h", "p_d", "p_a"]].sum(1)
    h = pd.DataFrame({"game_id": m.game_id, "date": m.date, "team": m.home, "coach": m.home_coach,
                      "pts": np.where(m.hg > m.ag, 3, np.where(m.hg == m.ag, 1, 0)),
                      "xpts": (3 * m.p_h + m.p_d) / s, "gmx": m.hg - m.hxg})
    a = pd.DataFrame({"game_id": m.game_id, "date": m.date, "team": m.away, "coach": m.away_coach,
                      "pts": np.where(m.ag > m.hg, 3, np.where(m.hg == m.ag, 1, 0)),
                      "xpts": (3 * m.p_a + m.p_d) / s, "gmx": m.ag - m.axg})
    L = pd.concat([h, a]).sort_values(["team", "date"]).reset_index(drop=True)
    gb = L.groupby("team")
    L["form5"] = gb.pts.transform(lambda x: x.shift().rolling(5, min_periods=3).sum())
    L["over5"] = (L.pts - L.xpts).groupby(L.team).transform(lambda x: x.shift().rolling(5, min_periods=3).sum())
    L["luck5"] = gb.gmx.transform(lambda x: x.shift().rolling(5, min_periods=3).sum())
    L["win_streak"] = gb.pts.transform(lambda x: (x.shift() == 3).astype(int).groupby((x.shift() != 3).cumsum()).cumsum())
    L["lose_streak"] = gb.pts.transform(lambda x: (x.shift() == 0).astype(int).groupby((x.shift() != 0).cumsum()).cumsum())
    # yeni hoca: ayni hocayla kacinci mac (takimin verideki ilk hocasi sayilmaz)
    ch = (L.coach != gb.coach.shift()) & L.coach.notna()
    L["run"] = ch.astype(int).groupby(L.team).cumsum()
    L["coach_n"] = L.groupby(["team", "run"]).cumcount() + 1
    L["new_coach"] = ((L.run > 1) & (L.coach_n <= 5)).astype(float)
    return L[["game_id", "team", "form5", "over5", "luck5", "win_streak", "lose_streak", "new_coach"]]


def referees():
    t = pd.read_sql("SELECT game_id, referee FROM tm_games", src_con())
    t["game_id"] = t.game_id.astype(str)
    return t


def odds(d):
    """Gercek oranlar (ortalama + en yuksek) xg_matches'ten; ROI ve buro farki icin."""
    x = pd.read_sql("SELECT matchdate date, hometeam, awayteam, oddhome, odddraw, oddaway, maxhome, maxdraw, maxaway "
                    "FROM xg_matches", src_con())
    km = pd.read_sql("SELECT name, team_key FROM team_map WHERE source='xgabora'", src_con())
    km = dict(zip(km.name, km.team_key))
    x["home"], x["away"] = x.hometeam.map(km), x.awayteam.map(km)
    x["date"] = pd.to_datetime(x.date)
    x = x.drop_duplicates(["date", "home", "away"])
    j = d[["game_id", "date", "home", "away"]].merge(x, on=["date", "home", "away"], how="left").set_index("game_id")
    return j


def features(g, d):
    t = pd.DataFrame({"game_id": d.game_id.values, "team": d.fav.values, "opp": d.dog.values}, index=d.index)
    F = fatigue(g, t)
    S = standings_features(g, t)
    L = team_long(g)
    f = t.merge(L.add_prefix("f_").rename(columns={"f_game_id": "game_id", "f_team": "team"}), on=["game_id", "team"], how="left")
    f = f.merge(L.add_prefix("d_").rename(columns={"d_game_id": "game_id", "d_team": "opp"}), on=["game_id", "opp"], how="left")
    f.index = t.index
    o = odds(d).reindex(d.game_id)
    fh = d.fav_home.values == 1
    dog_avg = np.where(fh, o.oddaway, o.oddhome)
    dog_max = np.where(fh, o.maxaway, o.maxhome)
    d["odd_dog"], d["odd_draw"] = dog_avg, o.odddraw.values
    d["odd_fav"] = np.where(fh, o.oddhome, o.oddaway)
    wd = d.date.dt.dayofweek
    X = pd.DataFrame(index=d.index)
    X["favori_deplasmanda"] = 1 - d.fav_home
    X["model_favoriye_piyasadan_5puan_az_veriyor"] = ((d.m_fav - d.p_fav) < -0.05).astype(float)
    X["model_favoriye_piyasadan_5puan_fazla_veriyor"] = ((d.m_fav - d.p_fav) > 0.05).astype(float)
    X["favori_kisa_dinlenme_3gun"] = F.kisa_dinlenme_3gun_alti.values
    X["surprizci_kisa_dinlenme_3gun"] = F.rakip_kisa_dinlenme.values
    X["favori_hafta_ici_avrupada_oynadi"] = F.hafta_ici_avrupa.values
    X["favori_avrupa_deplasmanindan_dondu"] = F.avrupa_deplasmanindan_donus.values
    X["surprizci_hafta_ici_avrupada_oynadi"] = F.rakip_hafta_ici_avrupa.values
    X["milli_ara_sonrasi"] = ((F.dinlenme_farki_gun.values == 0) & (wd.values >= 4)).astype(float) * 0  # asagida
    X["derbi"] = S.derbi.values
    X["favori_sampiyonluk_yarisinda"] = S.sampiyonluk_yarisi.fillna(0).values
    X["surprizci_kume_yarisinda"] = S.rakip_kume_yarisi.fillna(0).values
    X["favori_hicbir_seye_oynamiyor"] = S.hicbir_seye_oynamiyor.fillna(0).values
    X["favori_son5_beklentinin_ustunde"] = (f.f_over5 > 2).astype(float).values
    X["favori_son5_beklentinin_altinda"] = (f.f_over5 < -2).astype(float).values
    X["surprizci_son5_beklentinin_ustunde"] = (f.d_over5 > 2).astype(float).values
    X["surprizci_son5_beklentinin_altinda"] = (f.d_over5 < -2).astype(float).values
    X["favori_3mac_ust_uste_kazandi"] = (f.f_win_streak >= 3).astype(float).values
    X["favori_2mac_ust_uste_kaybetti"] = (f.f_lose_streak >= 2).astype(float).values
    X["surprizci_3mac_ust_uste_kaybetti"] = (f.d_lose_streak >= 3).astype(float).values
    X["favori_son5_xgden_sansli_3gol"] = (f.f_luck5 > 3).astype(float).values
    X["surprizci_son5_xgden_sanssiz_3gol"] = (f.d_luck5 < -3).astype(float).values
    X["favoride_yeni_hoca_ilk5"] = f.f_new_coach.fillna(0).values
    X["surprizcide_yeni_hoca_ilk5"] = f.d_new_coach.fillna(0).values
    X["hafta_ici_lig_maci"] = wd.isin([1, 2, 3]).astype(float).values
    X["sezonun_ilk_ayi_agustos"] = (d.date.dt.month == 8).astype(float).values
    X["sezon_sonu_mayis"] = (d.date.dt.month == 5).astype(float).values
    spread = dog_max / dog_avg - 1
    X["burolar_surprizciye_cok_farkli_oran"] = (spread > np.nanquantile(spread, 0.8)).astype(float)
    X = X.drop(columns=["milli_ara_sonrasi"]).fillna(0.0)
    for lg in sorted(d.league.unique()):
        X[f"lig_{lg}"] = (d.league == lg).astype(float).values
    for a, b in ((0.0, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 1.0)):
        X[f"favori_olasiligi_{int(a*100)}_{int(b*100)}"] = ((d.p_fav >= a) & (d.p_fav < b)).astype(float).values
    return X


def recalibrate(d, pcol, target):
    """Piyasa olasiligini egitim donemine gore duzelt (a + b*logit p). Oranlardaki 'favori-surprizci
    yanliligi' yuzunden ham piyasa surprizlere fazla olasilik veriyor; desenler bu duzeltmenin uzerine bakilir."""
    tr = (d.donem == "egitim").values
    x = logit(d[pcol].values)
    w = fit_offset_logit(x[tr][:, None], d[target].values[tr], np.zeros(tr.sum()), lam=0.0)
    return 1 / (1 + np.exp(-(w[0] + w[1] * x))), w


def univariate(d, X, target="upset", pcol="p_upset"):
    rows = []
    for c in X.columns:
        r = {"desen": c}
        for k in PERIODS:
            m = (d.donem == k) & (X[c] == 1)
            y, p = d.loc[m, target], d.loc[m, pcol]
            n = int(m.sum())
            diff = float((y - p).mean()) if n else np.nan
            se = float(np.sqrt((p * (1 - p)).sum()) / n) if n else np.nan
            r[f"{k}_n"], r[f"{k}_gercek"], r[f"{k}_beklenen"] = n, round(float(y.mean()), 3) if n else np.nan, round(float(p.mean()), 3) if n else np.nan
            r[f"{k}_fark"], r[f"{k}_z"] = round(diff, 3), round(diff / se, 2) if n and se > 0 else np.nan
        sgn = np.sign(r["egitim_fark"])
        r["tutuyor"] = bool(abs(r["egitim_z"]) > 2 and np.sign(r["2024/25_fark"]) == sgn and np.sign(r["2025/26_fark"]) == sgn)
        r["testte_de_anlamli"] = bool(r["tutuyor"] and abs(r["2024/25_z"]) > 1.5 and abs(r["2025/26_z"]) > 1.5)
        rows.append(r)
    return pd.DataFrame(rows)


def fit_offset_logit(X, y, off, lam=20.0, iters=50):
    """Lojistik regresyon, piyasa olasiligi ofset (sabit katsayi 1), L2 cezali Newton."""
    Xa = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xa.shape[1])
    R = np.eye(len(w)) * lam
    R[0, 0] = 0
    for _ in range(iters):
        z = off + Xa @ w
        p = 1 / (1 + np.exp(-z))
        gr = Xa.T @ (p - y) + R @ w
        H = Xa.T @ (Xa * (p * (1 - p))[:, None]) + R
        step = np.linalg.solve(H, gr)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def multivariate(d, X, target, pcol, label):
    cols = [c for c in X.columns if not c.startswith("favori_olasiligi_")]
    tr = (d.donem == "egitim").values
    out = []
    w = fit_offset_logit(X.values[tr][:, [X.columns.get_loc(c) for c in cols]], d[target].values[tr], logit(d[pcol].values[tr]))
    for k in PERIODS:
        m = (d.donem == k).values
        Xs = X.values[m][:, [X.columns.get_loc(c) for c in cols]]
        p0 = d[pcol].values[m]
        p1 = 1 / (1 + np.exp(-(logit(p0) + np.c_[np.ones(m.sum()), Xs] @ w)))
        y = d[target].values[m]
        out.append(dict(hedef=label, donem=k, mac=int(m.sum()), piyasa_ll=round(ll(p0, y), 5), piyasa_desenler_ll=round(ll(p1, y), 5),
                        motor_ll=round(ll(d[pcol.replace("p_", "m_")].values[m] if pcol != "p_upset" else 1 - d.m_fav.values[m], y), 5)))
    coefs = pd.DataFrame({"desen": ["sabit"] + cols, "katsayi": np.round(w, 3)})
    return pd.DataFrame(out), coefs


def roi(d, X, desenler):
    """Desen olan maclarda surprizciye (ve 'favori kazanamaz' icin surprizci + beraberlik) 1'er birim."""
    rows = []
    for c in ["hepsi"] + desenler:
        for k in PERIODS:
            m = (d.donem == k) & d.odd_dog.notna()
            if c != "hepsi":
                m &= X[c] == 1
            s = d[m]
            prof = np.where(s.dog_win == 1, s.odd_dog - 1, -1)
            rows.append(dict(desen=c, donem=k, bahis=len(s), surprizci_kazanma=round(float(s.dog_win.mean()), 3) if len(s) else np.nan,
                             roi_ortalama_oran=round(float(prof.mean()), 3) if len(s) else np.nan,
                             favoriye_roi=round(float(np.where(s.fav_win == 1, s.odd_fav - 1, -1).mean()), 3) if len(s) else np.nan))
    return pd.DataFrame(rows)


def referee_persistence(d):
    r = referees()
    x = d.merge(r, on="game_id", how="left").dropna(subset=["referee"])
    x["res"] = x.upset - x.p_upset
    a = x[x.donem == "egitim"].groupby("referee").agg(n_e=("res", "size"), r_e=("res", "mean"))
    b = x[x.donem != "egitim"].groupby("referee").agg(n_t=("res", "size"), r_t=("res", "mean"))
    j = a.join(b, how="inner")
    j = j[(j.n_e >= 30) & (j.n_t >= 20)]
    top = j[j.r_e > 0.06]
    return dict(hakem=len(j), korelasyon=round(float(j.r_e.corr(j.r_t)), 3),
                egitimde_surprizci_hakem=len(top), onlarin_egitim_farki=round(float(top.r_e.mean()), 3) if len(top) else None,
                onlarin_test_farki=round(float((top.r_t * top.n_t).sum() / top.n_t.sum()), 3) if len(top) else None,
                test_mac=int(top.n_t.sum()) if len(top) else 0)


def odds_movement():
    """Acilis -> kapanis oran hareketi; yalnizca 2026/27 basinda (fd_matches) var."""
    f = pd.read_sql("SELECT * FROM fd_matches", src_con())
    f = f.dropna(subset=["avgh", "avgd", "avga", "avgch", "avgcd", "avgca"]) if "avgch" in f else f.iloc[0:0]
    if not len(f):
        return {}
    o = 1 / f[["avgh", "avgd", "avga"]].values
    o /= o.sum(1, keepdims=True)
    c = 1 / f[["avgch", "avgcd", "avgca"]].values
    c /= c.sum(1, keepdims=True)
    fh = o[:, 0] >= o[:, 2]
    po, pc = np.where(fh, o[:, 0], o[:, 2]), np.where(fh, c[:, 0], c[:, 2])
    gf, ga = np.where(fh, f.fthg, f.ftag), np.where(fh, f.ftag, f.fthg)
    up = (gf <= ga).astype(float)
    drift = pc - po  # favorinin olasiligi kapanisa dogru dustu mu
    res = {"mac": len(f)}
    for name, m in (("favori_orani_yukseldi_2puan", drift < -0.02), ("favori_orani_dustu_2puan", drift > 0.02)):
        res[name] = dict(n=int(m.sum()), surpriz=round(float(up[m].mean()), 3), acilis_beklenen=round(float(1 - po[m].mean()), 3),
                         kapanis_beklenen=round(float(1 - pc[m].mean()), 3))
    res["acilis_ll"], res["kapanis_ll"] = round(ll(1 - po, up), 4), round(ll(1 - pc, up), 4)
    return res


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    g, d = load()
    X = features(g, d)
    print("mac:", d.groupby("donem").size().to_dict())
    print("genel: favori kazanamadi", d.groupby("donem").agg(gercek=("upset", "mean"), piyasa=("p_upset", "mean"),
                                                             motor=("m_fav", lambda s: 1 - s.mean())).round(3).to_string())
    con = sim_con()
    d["p_upset_ham"], d["p_dog_ham"] = d.p_upset, d.p_dog
    d["p_upset"], w1 = recalibrate(d, "p_upset_ham", "upset")
    d["p_dogwin"], w2 = recalibrate(d, "p_dog_ham", "dog_win")
    print("piyasa duzeltmesi (sabit, egim):", np.round(w1, 3), np.round(w2, 3))
    print(d.groupby("donem").agg(gercek=("upset", "mean"), duzeltilmis=("p_upset", "mean"), sk=("dog_win", "mean"), sk_d=("p_dogwin", "mean")).round(3).to_string())
    U = univariate(d, X)
    U.to_sql("surpriz_desen", con, if_exists="replace", index=False)
    cols = ["desen", "egitim_n", "egitim_fark", "egitim_z", "2024/25_n", "2024/25_fark", "2024/25_z", "2025/26_n", "2025/26_fark", "2025/26_z", "tutuyor", "testte_de_anlamli"]
    print("\n== favori kazanamadi (gercek - piyasa) ==")
    print(U[cols].to_string(index=False))
    U2 = univariate(d, X, "dog_win", "p_dogwin")
    U2.to_sql("surpriz_desen_galibiyet", con, if_exists="replace", index=False)
    print("\n== surprizci kazandi (gercek - piyasa) ==")
    print(U2[cols].to_string(index=False))
    # buyuk surpriz: piyasa %25'in altinda verirken kazanan
    big = d.p_dog_ham < 0.25
    print("\nbuyuk surpriz (surprizciye <%25):", d[big].groupby("donem").agg(n=("dog_win", "size"), gercek=("dog_win", "mean"),
                                                                           piyasa=("p_dog_ham", "mean"), duzeltilmis=("p_dogwin", "mean"), motor=("m_dog", "mean")).round(3).to_string())
    U3 = univariate(d[big], X[big], "dog_win", "p_dogwin")
    U3.to_sql("surpriz_desen_buyuk", con, if_exists="replace", index=False)
    print(U3[U3.tutuyor][cols].to_string(index=False))
    M1, C1 = multivariate(d, X, "upset", "p_upset", "favori_kazanamadi")
    print("\n== hepsi birlikte (egitimde ogrenilen, testte denendi) ==")
    print(M1.to_string(index=False))
    print(C1[C1.katsayi.abs() > 0.03].to_string(index=False))
    M1.to_sql("surpriz_model", con, if_exists="replace", index=False)
    C1.to_sql("surpriz_katsayi", con, if_exists="replace", index=False)
    # motor surprizleri piyasadan iyi goruyor mu?
    print("\n== motor vs piyasa: surprizciye verilen olasilik ==")
    for k in PERIODS:
        s = d[d.donem == k]
        y = s.dog_win.values
        print(k, "piyasa", round(ll(s.p_dog.values, y), 4), "motor", round(ll(s.m_dog.values, y), 4),
              "| motor surprizciye daha cok verdiginde (>5 puan):", int(((s.m_dog - s.p_dog) > 0.05).sum()), "mac, gercek",
              round(float(s[(s.m_dog - s.p_dog) > 0.05].dog_win.mean()), 3), "piyasa", round(float(s[(s.m_dog - s.p_dog) > 0.05].p_dog.mean()), 3))
    R = referee_persistence(d)
    print("\n== hakem ==", R)
    mv = odds_movement()
    print("\n== oran hareketi (2026/27 basi) ==", mv)
    keep = U[U.tutuyor].desen.tolist()
    RO = roi(d, X, keep)
    RO.to_sql("surpriz_roi", con, if_exists="replace", index=False)
    print("\n== ROI (surprizciye ortalama oranla 1 birim) ==")
    print(RO.to_string(index=False))


if __name__ == "__main__":
    main()
