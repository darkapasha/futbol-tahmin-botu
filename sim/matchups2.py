"""Oyuncu eslesme testi 2: piyasa degeri yerine / yaninda FORM PUANI, oyuncu tipi ve sakatliktan donus.

  .venv/bin/python -m sim.matchups2

Veri: FotMob maclari (oyuncu mac puani, sahadaki yeri x/y, sut haritasi). Suan TUR, ENG, ESP, GER (kismen)
ve SL icin birkac sezon dolu.
- Form puani: oyuncunun AYNI SEZON, AYNI TURNUVADAKI (lig maci -> lig, SL -> SL) onceki maclarindaki FotMob
  puaninin ortalamasi, 3 maclik 6.8 on degerle buzulmus. Yalnizca mactan onceki maclar (sizinti yok).
- Eslesme: kendi sol kanatcimiz (on uc, y < 0.3) x rakibin sag beki (y > 0.7 en geri oyuncu); sag kanat x sol bek.
- Kanatci tipi: o sezon ayni turnuvada mac basina xG'si (golcu kanatci) yuksek mi dusuk mu (oyun kurucu).
- Sakatliktan / uzun aradan donus: oyuncunun takiminin onceki 4+ macini kacirip ilk kez ilk 11'de olmasi.

Testler (egitim < 2024-08, sinama 2024/25 ve 2025/26):
 A) Takim: 1X2 log-loss / isabet; mevcut model vs + ilk 11 form farki vs + kanat/bek form eslesmesi
    vs + donen oyuncu sayisi
 B) Oyuncu: kanatci gol atti mi; deger, form, karsi bekin formu, tip x bek etkilesimi, donus
"""
import numpy as np
import pandas as pd

from . import calibrate, hist
from .common import TeamResolver, load_params, sim_con, src_con
from .tactics import evaluate, fit_poisson

PRIOR_R, PRIOR_N = 6.8, 3.0


def load():
    con = src_con()
    lu = pd.read_sql("SELECT l.match_id, l.team_id, l.player_id, l.name, l.role, l.position_id, l.x, l.y, l.rating, "
                     "l.market_value, m.league, m.season, m.utc_time, m.home_id, m.home, m.away_id, m.away, "
                     "m.home_goals, m.away_goals FROM fm_lineups l JOIN fm_matches m USING (match_id) "
                     "WHERE m.home_goals IS NOT NULL", con)
    lu["date"] = pd.to_datetime(lu.utc_time.str[:10])
    shots = pd.read_sql("SELECT match_id, player_id, SUM(xg) xg, SUM(event_type='Goal' AND own_goal=0) goals "
                        "FROM fm_shots GROUP BY 1, 2", con)
    lu = lu.merge(shots, on=["match_id", "player_id"], how="left")
    lu["xg"], lu["goals"] = lu.xg.fillna(0), lu.goals.fillna(0)
    return lu.sort_values(["date", "match_id"]).reset_index(drop=True)


def add_form(lu):
    """Mac oncesi form: ayni oyuncu, ayni sezon, ayni turnuva, onceki maclar (yalnizca oynadigi)."""
    played = lu.rating.notna()
    key = [lu.player_id, lu.season, lu.league]
    r = lu.rating.where(played)
    cs = r.fillna(0).groupby(key).cumsum() - r.fillna(0)
    cn = played.astype(float).groupby(key).cumsum() - played.astype(float)
    lu["form"] = (cs + PRIOR_R * PRIOR_N) / (cn + PRIOR_N)
    lu["form_n"] = cn
    st = lu.role == "starter"
    xs = lu.xg.where(st)
    lu["xg_pm"] = (xs.fillna(0).groupby(key).cumsum() - xs.fillna(0) + 0.1 * 3) / \
                  (st.astype(float).groupby(key).cumsum() - st.astype(float) + 3)
    # takimin kac macini ust uste kacirdi (ilk 11'e donus)
    lu["team_game_no"] = lu.groupby("team_id").match_id.rank(method="dense")
    s = lu[st].sort_values("date")
    prev = s.groupby("player_id").team_game_no.shift()
    lu.loc[s.index, "gap"] = s.team_game_no - prev - 1
    lu["returning"] = ((lu.gap >= 4) & (lu.role == "starter")).astype(float)
    return lu


def team_table(lu):
    st = lu[lu.role == "starter"].copy()
    gk = st.position_id == 11
    st["is_lw"] = (st.x >= 0.6) & (st.y < 0.3)
    st["is_rw"] = (st.x >= 0.6) & (st.y > 0.7)
    def flank_def(g, side):
        d = g[~(g.position_id == 11) & ((g.y > 0.7) if side == "r" else (g.y < 0.3))]
        return d.sort_values("x").iloc[0] if len(d) else None
    rows = []
    for (mid, tid), g in st.groupby(["match_id", "team_id"]):
        rb, lb = flank_def(g, "r"), flank_def(g, "l")
        lw = g[g.is_lw].sort_values("x", ascending=False)
        rw = g[g.is_rw].sort_values("x", ascending=False)
        rows.append(dict(match_id=mid, team_id=tid, xi_form=g.form[g.position_id != 11].mean(),
                         xi_val=np.log(g.market_value.clip(lower=1e5)).mean(),
                         lw_form=lw.form.iat[0] if len(lw) else np.nan, rw_form=rw.form.iat[0] if len(rw) else np.nan,
                         lw_pid=lw.player_id.iat[0] if len(lw) else np.nan, rw_pid=rw.player_id.iat[0] if len(rw) else np.nan,
                         rb_form=rb.form if rb is not None else np.nan, lb_form=lb.form if lb is not None else np.nan,
                         rb_val=np.log(max(rb.market_value or 1e5, 1e5)) if rb is not None else np.nan,
                         lb_val=np.log(max(lb.market_value or 1e5, 1e5)) if lb is not None else np.nan,
                         returning=g.returning.sum()))
    return pd.DataFrame(rows)


def attach_mu(lu, tt):
    """FotMob macini puan modelinin gecmis tahminine (pred_hist) bagla: tarih (+-1) ve takim anahtari."""
    g = hist.games()
    pred = pd.read_sql("SELECT * FROM pred_hist", sim_con())
    g = g.merge(pred, on="game_id")
    r = TeamResolver()
    m = lu.drop_duplicates("match_id")[["match_id", "date", "league", "home_id", "home", "away_id", "away",
                                        "home_goals", "away_goals"]].copy()
    m["hk"] = [r.fotmob(i, n, lg) for i, n, lg in zip(m.home_id, m.home, m.league)]
    m["ak"] = [r.fotmob(i, n, lg) for i, n, lg in zip(m.away_id, m.away, m.league)]
    out = []
    for shift in (0, -1, 1):
        x = m.assign(d=m.date + pd.Timedelta(days=shift)).merge(
            g[["date", "home", "away", "game_id", "mu_h", "mu_a"]], left_on=["d", "hk", "ak"],
            right_on=["date", "home", "away"], suffixes=("", "_g"))
        out.append(x)
    x = pd.concat(out).drop_duplicates("match_id")
    return x[["match_id", "date", "league", "home_id", "away_id", "home_goals", "away_goals", "mu_h", "mu_a"]]


def team_rows(mm, tt):
    h = mm.merge(tt.add_prefix("h_").rename(columns={"h_match_id": "match_id", "h_team_id": "home_id"}), on=["match_id", "home_id"])
    h = h.merge(tt.add_prefix("a_").rename(columns={"a_match_id": "match_id", "a_team_id": "away_id"}), on=["match_id", "away_id"])
    rows = []
    for side, o in (("h", "a"), ("a", "h")):
        d = pd.DataFrame({
            "game_id": h.match_id.astype(str), "date": h.date, "is_home": int(side == "h"),
            "y_for": h.home_goals if side == "h" else h.away_goals,
            "mu_for": h.mu_h if side == "h" else h.mu_a,
            "xi_form_farki": h[f"{side}_xi_form"] - h[f"{o}_xi_form"],
            "xi_deger_farki": h[f"{side}_xi_val"] - h[f"{o}_xi_val"],
            "sol_kanat_x_sag_bek_form": h[f"{side}_lw_form"] - h[f"{o}_rb_form"],
            "sag_kanat_x_sol_bek_form": h[f"{side}_rw_form"] - h[f"{o}_lb_form"],
            "donen_oyuncu": h[f"{side}_returning"], "rakip_donen_oyuncu": h[f"{o}_returning"]})
        rows.append(d)
    t = pd.concat(rows, ignore_index=True)
    # evaluate() hist.games'teki gol sutunlarini game_id ile okur; burada fm macinin kendisi kullanilir
    return t


def evaluate_fm(t, X, b, label):
    adj = np.exp(X.values @ b) if b is not None else np.ones(len(t))
    d = t.assign(adj=adj)
    h, a = d[d.is_home == 1].set_index("game_id"), d[d.is_home == 0].set_index("game_id")
    ids = h.index.intersection(a.index)
    h, a = h.loc[ids], a.loc[ids]
    from .rolling import outcome_probs, score_matrix
    P = outcome_probs(score_matrix((h.mu_for * h.adj).values, (a.mu_for * a.adj).values))
    y = np.where(h.y_for > a.y_for, 0, np.where(h.y_for == a.y_for, 1, 2))
    ll = float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1))))
    return dict(model=label, mac=len(ids), logloss=round(ll, 4), isabet=round(float((P.argmax(1) == y).mean()), 4))


def player_test(lu, tt):
    """Kanatci gol atti mi (lojistik). Karsi bekin formu/degeri, kanatci tipi x bek etkilesimi, donus."""
    st = lu[(lu.role == "starter") & (lu.x >= 0.6) & ((lu.y < 0.3) | (lu.y > 0.7))].copy()
    st["left"] = st.y < 0.3
    opp = tt[["match_id", "team_id", "rb_form", "lb_form", "rb_val", "lb_val"]]
    st["opp_id"] = np.where(st.team_id == st.home_id, st.away_id, st.home_id)
    st = st.merge(opp.rename(columns={"team_id": "opp_id"}), on=["match_id", "opp_id"])
    st["fb_form"] = np.where(st.left, st.rb_form, st.lb_form)
    st["fb_val"] = np.where(st.left, st.rb_val, st.lb_val)
    st["lv"] = np.log(st.market_value.clip(lower=1e5))
    st["scorer_type"] = (st.xg_pm - st.xg_pm.median()) / st.xg_pm.std()
    st["scored"] = (st.goals > 0).astype(float)
    st = st.dropna(subset=["fb_form", "fb_val", "lv", "form"])
    st["fb_form_c"] = st.fb_form - PRIOR_R
    st["form_c"] = st.form - PRIOR_R
    st["tip_x_bek"] = st.scorer_type * st.fb_form_c
    models = {
        "deger + golcu tipi (taban)": ["lv", "scorer_type"],
        "+ karsi bekin DEGERI": ["lv", "scorer_type", "fb_val"],
        "+ kendi formu + karsi bekin FORMU": ["lv", "scorer_type", "form_c", "fb_form_c"],
        "+ tip x bek etkilesimi": ["lv", "scorer_type", "form_c", "fb_form_c", "tip_x_bek"],
        "+ sakatliktan donus": ["lv", "scorer_type", "form_c", "fb_form_c", "tip_x_bek", "returning"],
    }
    out = []
    for ts in ("2024-08-01", "2025-08-01"):
        tr = st[st.date < ts]
        te = st[(st.date >= ts) & (st.date < str(int(ts[:4]) + 1) + "-07-01")]
        for name, cols in models.items():
            Xtr, Xte = np.c_[np.ones(len(tr)), tr[cols].values], np.c_[np.ones(len(te)), te[cols].values]
            b = np.zeros(Xtr.shape[1])
            for _ in range(30):
                p = 1 / (1 + np.exp(-Xtr @ b))
                H = (Xtr * (p * (1 - p))[:, None]).T @ Xtr + 1.0 * np.eye(len(b))
                b += np.linalg.solve(H, Xtr.T @ (tr.scored.values - p) - 1.0 * b)
            p = np.clip(1 / (1 + np.exp(-Xte @ b)), 1e-6, 1 - 1e-6)
            ll = -np.mean(te.scored * np.log(p) + (1 - te.scored) * np.log(1 - p))
            out.append(dict(test=ts[:4], model=name, kanatci_mac=len(te), logloss=round(float(ll), 5),
                            katsayilar={c: round(float(v), 3) for c, v in zip(cols, b[1:])}))
    return pd.DataFrame(out)


def main():
    lu = add_form(load())
    tt = team_table(lu)
    mm = attach_mu(lu, tt)
    t = team_rows(mm, tt).dropna(subset=["mu_for"]).reset_index(drop=True)
    feats = ["xi_deger_farki", "xi_form_farki", "sol_kanat_x_sag_bek_form", "sag_kanat_x_sol_bek_form",
             "donen_oyuncu", "rakip_donen_oyuncu"]
    X = t[feats].fillna(0).clip(-3, 3)
    variants = {"+ ilk 11 deger farki": ["xi_deger_farki"],
                "+ ilk 11 FORM farki": ["xi_deger_farki", "xi_form_farki"],
                "+ kanat/bek form eslesmesi": ["xi_deger_farki", "xi_form_farki", "sol_kanat_x_sag_bek_form", "sag_kanat_x_sol_bek_form"],
                "+ sakatliktan donenler": feats}
    rows = []
    y, off = t.y_for.values.astype(float), np.log(t.mu_for.values)
    for ts, te_ in (("2024-08-01", "2025-07-01"), ("2025-08-01", "2026-07-01")):
        tr = (t.date < pd.Timestamp(ts) - pd.Timedelta(days=20)).values
        te = ((t.date >= ts) & (t.date < te_)).values
        r = evaluate_fm(t[te], X[te], None, "mevcut model")
        r["test"] = ts[:4]
        rows.append(r)
        for name, cols in variants.items():
            b = fit_poisson(X[cols].values[tr], y[tr], off[tr], lam=30)
            r = evaluate_fm(t[te], X[cols][te], b, name)
            r["test"], r["katsayi"] = ts[:4], {c: round(float(np.exp(v)), 3) for c, v in zip(cols, b)}
            rows.append(r)
    res = pd.DataFrame(rows)
    print("TAKIM DUZEYI (FotMob'da puani olan maclar):")
    print(res.drop(columns="katsayi").to_string(index=False))
    print("son varyantin gol carpanlari (2025 testi):", rows[-1]["katsayi"])
    pt = player_test(lu, tt)
    print("\nOYUNCU DUZEYI (kanatci gol atti mi):")
    print(pt.drop(columns="katsayilar").to_string(index=False))
    print("katsayilar (2025 egitimi, son model):", pt.katsayilar.iloc[-1])
    con = sim_con()
    res.astype(str).to_sql("matchup2_test", con, if_exists="replace", index=False)
    pt.astype(str).to_sql("matchup2_player_test", con, if_exists="replace", index=False)
    con.commit()


if __name__ == "__main__":
    main()
