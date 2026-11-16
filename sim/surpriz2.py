"""Surpriz maclar, genis arama (surpriz.py'nin devami).

  .venv/bin/python -m sim.surpriz2

~50 surekli isaret (kadro degeri, sakatlar, puan durumu, takvim, H2H, takim/hoca/hakem gecmisi, gol
beklentisi, saat...) x 4 hedef (favori kazanamadi, surprizci kazandi, beraberlik, 2.5 ust).
A) dilim taramasi, B) ikili kombinasyon taramasi (coklu test kontrollu), C) numpy gradient boosting ve
L2 lojistik (piyasa ofsetli), D) takim / hoca bazli kalicilik. Egitim < 2024-08, sinama 2024/25, 2025/26.
Yalnizca okur; sonuclar data/sim.sqlite 'surpriz2_*' tablolarina.
"""
import itertools

import numpy as np
import pandas as pd

from . import surpriz as S
from .common import sim_con, src_con

TESTS = ("2024/25", "2025/26")


# ------------------------------------------------------------------ veri
def schedule():
    """Tum maclar (lig + CL) takim bakisindan: onceki / sonraki mac tarihi ve turu."""
    t = pd.read_sql("SELECT game_id, date, competition_type ct, home_club_id h, away_club_id a FROM tm_games", src_con())
    t["date"] = pd.to_datetime(t.date)
    L = pd.concat([t.assign(team=t.h, away=0), t.assign(team=t.a, away=1)])
    L["team"] = "tm:" + L.team.astype(str)
    L["game_id"] = L.game_id.astype(str)
    L = L.sort_values(["team", "date"])
    gb = L.groupby("team")
    L["rest"] = (L.date - gb.date.shift()).dt.days
    L["next_days"] = (gb.date.shift(-1) - L.date).dt.days
    L["next_cl"] = (gb.ct.shift(-1) == "international_cup").astype(float)
    L["prev_cl_away"] = ((gb.ct.shift() == "international_cup") & (gb.away.shift() == 1)).astype(float)
    return L[["game_id", "team", "rest", "next_days", "next_cl", "prev_cl_away"]]


def xi_values():
    con = src_con()
    lu = pd.read_sql("SELECT game_id, club_id, player_id, date FROM tm_lineups WHERE type='starting_lineup' AND date >= '2021-06-01'", con)
    val = pd.read_sql("SELECT player_id, date, market_value_in_eur v FROM tm_valuations", con)
    lu["date"], val["date"] = pd.to_datetime(lu.date), pd.to_datetime(val.date)
    lu = pd.merge_asof(lu.sort_values("date"), val.sort_values("date"), on="date", by="player_id", allow_exact_matches=False)
    xi = lu.groupby(["game_id", "club_id", "date"]).v.sum(min_count=8).reset_index().sort_values("date")
    xi["usual"] = xi.groupby("club_id").v.transform(lambda s: s.shift(1).rolling(10, min_periods=4).mean())
    xi["team"] = "tm:" + xi.club_id.astype(str)
    xi["game_id"] = xi.game_id.astype(str)
    xi["xi_val"] = np.log1p(xi.v / 1e6)
    xi["rotation"] = np.log((xi.v / xi.usual).clip(0.4, 1.8))
    return xi[["game_id", "team", "xi_val", "rotation", "usual"]], val


def injured(d_keys, val):
    """Mac gunu sakat oyuncularin toplam piyasa degeri (oyuncunun o tarihteki kulubu: son maci)."""
    con = src_con()
    inj = pd.read_sql("SELECT player_id, from_date, end_date, days_missed FROM injuries", con)
    inj["from_date"] = pd.to_datetime(inj.from_date, errors="coerce")
    inj["end_date"] = pd.to_datetime(inj.end_date, errors="coerce")
    inj["end_date"] = inj.end_date.fillna(inj.from_date + pd.to_timedelta(inj.days_missed.fillna(14), unit="D"))
    inj = inj.dropna(subset=["from_date"])
    ap = pd.read_sql("SELECT player_id, player_club_id club, date FROM tm_appearances", con)
    ap["date"] = pd.to_datetime(ap.date)
    inj = pd.merge_asof(inj.sort_values("from_date"), ap.sort_values("date").rename(columns={"date": "from_date"}),
                        on="from_date", by="player_id")
    inj = pd.merge_asof(inj.sort_values("from_date"), val.sort_values("date").rename(columns={"date": "from_date"}),
                        on="from_date", by="player_id")
    inj = inj.dropna(subset=["club"])
    inj["team"] = "tm:" + inj.club.astype(int).astype(str)
    inj["v"] = inj.v.fillna(0) / 1e6
    out = np.zeros(len(d_keys))
    cnt = np.zeros(len(d_keys))
    by = dict(tuple(inj.groupby("team")))
    for team, idx in d_keys.groupby("team").groups.items():
        if team not in by:
            continue
        I = by[team]
        dates = d_keys.loc[idx, "date"].values[:, None]
        act = (I.from_date.values[None, :] <= dates) & (I.end_date.values[None, :] >= dates)
        pos = d_keys.index.get_indexer(idx)
        out[pos] = act @ I.v.values
        cnt[pos] = act.sum(1)
    return out, cnt


def table_and_history(d):
    """Sirayla gezerek: puan durumu, H2H, takim/hoca/hakem gecmis sapmalari (yalniz onceki maclar)."""
    ref = S.referees().set_index("game_id").referee
    pts, gp = {}, {}
    team_res, coach_res, ref_res, ref_home = {}, {}, {}, {}
    h2h = {}
    rows = []
    d = d.sort_values(["date", "game_id"])
    for r in d.itertuples():
        lg = (r.league, r.season)
        P, G = pts.setdefault(lg, {}), gp.setdefault(lg, {})
        teams_tbl = sorted(P.items(), key=lambda kv: -kv[1] / max(G[kv[0]], 1))
        rank = {k: i + 1 for i, (k, _) in enumerate(teams_tbl)}
        nT = max(len(rank), 1)
        f, o = r.fav, r.dog
        fc = r.home_coach if r.fav_home else r.away_coach
        dc = r.away_coach if r.fav_home else r.home_coach
        rf = ref.get(r.game_id)

        def shr(lst, k=20):
            return (sum(lst[-38:]) / (len(lst[-38:]) + k)) if lst else 0.0

        def shr_all(lst, k=30):
            return (sum(lst) / (len(lst) + k)) if lst else 0.0
        key = tuple(sorted((f, o)))
        last = h2h.get(key)
        rows.append(dict(
            game_id=r.game_id,
            f_ppg=P.get(f, 0) / max(G.get(f, 0), 1), d_ppg=P.get(o, 0) / max(G.get(o, 0), 1),
            f_rank=rank.get(f, nT) / nT, d_rank=rank.get(o, nT) / nT, sezon_ilerleme=G.get(f, 0) / 38,
            f_takim_sapma38=shr(team_res.get(f, [])), d_takim_sapma38=shr(team_res.get(o, [])),
            f_hoca_sapma=shr_all(coach_res.get(fc, [])), d_hoca_sapma=shr_all(coach_res.get(dc, [])),
            hakem_surpriz_gecmis=shr_all(ref_res.get(rf, []), 40), hakem_ev_gecmis=shr_all(ref_home.get(rf, []), 40),
            h2h_surprizci_puan=(np.nan if last is None else (last[1] if last[0] == o else last[2])),
        ))
        # guncelle
        hp = 3 if r.hg > r.ag else (1 if r.hg == r.ag else 0)
        ap = 3 if r.ag > r.hg else (1 if r.hg == r.ag else 0)
        P[r.home] = P.get(r.home, 0) + hp
        P[r.away] = P.get(r.away, 0) + ap
        G[r.home] = G.get(r.home, 0) + 1
        G[r.away] = G.get(r.away, 0) + 1
        xh = 3 * r.p_h + r.p_d
        xa = 3 * r.p_a + r.p_d
        team_res.setdefault(r.home, []).append(hp - xh)
        team_res.setdefault(r.away, []).append(ap - xa)
        if isinstance(r.home_coach, str):
            coach_res.setdefault(r.home_coach, []).append(hp - xh)
        if isinstance(r.away_coach, str):
            coach_res.setdefault(r.away_coach, []).append(ap - xa)
        if isinstance(rf, str):
            ref_res.setdefault(rf, []).append(r.upset - r.p_upset)
            ref_home.setdefault(rf, []).append(float(r.hg > r.ag) - r.p_h)
        h2h[key] = (r.away, ap, hp) if True else None
        # (son macta deplasman takimi ve iki tarafin puani)
        h2h[key] = (r.away, ap, hp)
    # h2h tuple: (away_team, away_pts, home_pts) -> surprizcinin puani
    return pd.DataFrame(rows).set_index("game_id")


def build():
    g, d = S.load()
    Xb = S.features(g, d)
    d["p_upset_ham"], d["p_dog_ham"] = d.p_upset, d.p_dog
    d["p_upset"], _ = S.recalibrate(d, "p_upset_ham", "upset")
    d["p_dogwin"], _ = S.recalibrate(d, "p_dog_ham", "dog_win")
    d["p_draw"], _ = S.recalibrate(d, "p_d", "draw")
    # gol beklentisi (2.5 ust orani) ve baslama saati
    x = pd.read_sql("SELECT matchdate date, matchtime, hometeam, awayteam, over25, under25, homeelo, awayelo FROM xg_matches", src_con())
    km = pd.read_sql("SELECT name, team_key FROM team_map WHERE source='xgabora'", src_con())
    km = dict(zip(km.name, km.team_key))
    x["home"], x["away"] = x.hometeam.map(km), x.awayteam.map(km)
    x["date"] = pd.to_datetime(x.date)
    x = x.drop_duplicates(["date", "home", "away"])
    j = d[["game_id", "date", "home", "away"]].merge(x, on=["date", "home", "away"], how="left")
    j.index = d.index
    io = 1 / j[["over25", "under25"]].values
    d["p_over_ham"] = io[:, 0] / io.sum(1)
    d["over"] = ((d.hg + d.ag) > 2.5).astype(float)
    ok = d.p_over_ham.notna()
    d["p_over"] = np.nan
    d.loc[ok, "p_over"], _ = S.recalibrate(d[ok], "p_over_ham", "over")

    fh = d.fav_home.values == 1
    Z = pd.DataFrame(index=d.index)
    Z["piyasa_favori"] = d.p_fav
    Z["piyasa_beraberlik"] = d.p_d
    Z["piyasa_2.5ust"] = d.p_over_ham
    Z["motor_favori_farki"] = d.m_fav - d.p_fav
    Z["motor_beraberlik_farki"] = d.m_d - d.p_d
    Z["motor_gol_toplami"] = d.mu_h + d.mu_a
    he, ae = j.homeelo.values, j.awayelo.values
    diff = np.where(fh, he - ae, ae - he) + np.where(fh, 60, -60)
    Z["elo_favori_farki_vs_piyasa"] = 1 / (1 + 10 ** (-diff / 400)) - (d.p_fav + d.p_d / 2)
    Z["saat"] = pd.to_numeric(j.matchtime.str[:2], errors="coerce").values
    Z["hafta_gunu"] = d.date.dt.dayofweek
    Z["ay"] = d.date.dt.month
    # takvim
    sc = schedule()
    keys_f = pd.DataFrame({"game_id": d.game_id.values, "team": d.fav.values})
    keys_d = pd.DataFrame({"game_id": d.game_id.values, "team": d.dog.values})
    for p, k in (("favori", keys_f), ("surprizci", keys_d)):
        m = k.merge(sc, on=["game_id", "team"], how="left")
        Z[f"{p}_dinlenme"] = m.rest.clip(upper=21).values
        Z[f"{p}_sonraki_maca_gun"] = m.next_days.clip(upper=21).values
        Z[f"{p}_sonraki_mac_avrupa_4gun"] = ((m.next_cl == 1) & (m.next_days <= 4)).astype(float).values
        Z[f"{p}_avrupa_deplasmanindan"] = m.prev_cl_away.values
    Z["dinlenme_farki"] = Z.favori_dinlenme - Z.surprizci_dinlenme
    # kadro degeri, rotasyon, sakatlar
    xi, val = xi_values()
    for p, k in (("favori", keys_f), ("surprizci", keys_d)):
        m = k.merge(xi, on=["game_id", "team"], how="left")
        Z[f"{p}_11_deger"] = m.xi_val.values
        Z[f"{p}_rotasyon"] = m.rotation.values
        kk = pd.DataFrame({"team": k.team.values, "date": d.date.values})
        v, c = injured(kk, val)
        Z[f"{p}_sakat_deger"] = np.log1p(v)
        Z[f"{p}_sakat_sayi"] = c
        Z[f"{p}_sakat_orani"] = v / (m.usual.values / 1e6)
    Z["11_deger_farki"] = Z.favori_11_deger - Z.surprizci_11_deger
    Z["sakat_farki"] = Z.favori_sakat_deger - Z.surprizci_sakat_deger
    # form
    L = S.team_long(g)
    L2 = L.copy()
    for p, k in (("favori", keys_f), ("surprizci", keys_d)):
        m = k.merge(L2, on=["game_id", "team"], how="left")
        Z[f"{p}_form5"] = m.form5.values
        Z[f"{p}_beklenti_ustu5"] = m.over5.values
        Z[f"{p}_xg_sans5"] = m.luck5.values
        Z[f"{p}_galibiyet_serisi"] = m.win_streak.values
        Z[f"{p}_maglubiyet_serisi"] = m.lose_streak.values
        Z[f"{p}_yeni_hoca"] = m.new_coach.values
    H = table_and_history(d).reindex(d.game_id)
    for c in H.columns:
        Z[c] = H[c].values
    Z["puan_farki"] = Z.f_ppg - Z.d_ppg
    Z["sira_farki"] = Z.d_rank - Z.f_rank
    Z["burolar_oran_farki"] = (np.where(fh, S.odds(d).reindex(d.game_id).maxaway, S.odds(d).reindex(d.game_id).maxhome)
                               / d.odd_dog.values - 1) if "odd_dog" in d else np.nan
    Z["favori_deplasmanda"] = 1 - d.fav_home
    Z["derbi"] = Xb.derbi
    for lg in sorted(d.league.unique()):
        Z[f"lig_{lg}"] = (d.league == lg).astype(float)
    return g, d, Z


# ------------------------------------------------------------------ analizler
TARGETS = {"favori_kazanamadi": ("upset", "p_upset"), "surprizci_kazandi": ("dog_win", "p_dogwin"),
           "beraberlik": ("draw", "p_draw"), "ust_2.5": ("over", "p_over")}


def zstat(y, p):
    n = len(y)
    if n == 0:
        return np.nan, np.nan
    se = np.sqrt((p * (1 - p)).sum()) / n
    diff = float((y - p).mean())
    return diff, diff / se if se > 0 else np.nan


def flags(d, Z):
    """Her surekli isaret icin egitim donemine gore alt / ust %20 bayragi; ikili olanlar aynen."""
    tr = d.donem == "egitim"
    F = {}
    for c in Z.columns:
        v = Z[c]
        u = v.dropna().unique()
        if len(u) <= 2:
            F[c] = (v == 1).astype(float)
            continue
        lo, hi = np.nanquantile(v[tr], 0.2), np.nanquantile(v[tr], 0.8)
        if lo < hi:
            F[f"{c}<=alt20"] = (v <= lo).astype(float)
            F[f"{c}>=ust20"] = (v >= hi).astype(float)
    return pd.DataFrame(F, index=Z.index)


def scan(d, F, tname, cols=None, min_n=150):
    yc, pc = TARGETS[tname]
    ok = d[pc].notna()
    per = {k: (d.donem == k) & ok for k in S.PERIODS}
    tst = per["2024/25"] | per["2025/26"]
    rows = []
    for c in (cols or F.columns):
        m = F[c] == 1 if isinstance(c, str) else (F[c[0]] == 1) & (F[c[1]] == 1)
        r = {"hedef": tname, "desen": c if isinstance(c, str) else f"{c[0]} + {c[1]}"}
        mt = m & per["egitim"]
        if mt.sum() < min_n:
            continue
        for k in S.PERIODS:
            mm = m & per[k]
            r[f"{k}_n"] = int(mm.sum())
            r[f"{k}_fark"], r[f"{k}_z"] = zstat(d.loc[mm, yc].values, d.loc[mm, pc].values)
        mm = m & tst
        r["test_fark"], r["test_z"] = zstat(d.loc[mm, yc].values, d.loc[mm, pc].values)
        rows.append(r)
    return pd.DataFrame(rows)


def verdict(R, ztr=2.0):
    s = np.sign(R.egitim_fark)
    R["tutuyor"] = (R.egitim_z.abs() > ztr) & (np.sign(R["2024/25_fark"]) == s) & (np.sign(R["2025/26_fark"]) == s) & (R.test_z.abs() > 2) & (np.sign(R.test_fark) == s)
    return R


# --------------- numpy gradient boosting (derinlik 2, piyasa ofsetli)
def binize(Ztr, Zall, nb=16):
    B = np.zeros(Zall.shape, np.int16)
    for j in range(Zall.shape[1]):
        v = Ztr[:, j]
        q = np.unique(np.nanquantile(v[~np.isnan(v)], np.linspace(0, 1, nb + 1)[1:-1])) if (~np.isnan(v)).any() else np.array([])
        b = np.searchsorted(q, Zall[:, j], side="right") + 1
        b[np.isnan(Zall[:, j])] = 0
        B[:, j] = b
    return B


def best_split(B, g, h, idx, lam, nb):
    best = (0.0, None, None)
    G, Hs = g[idx].sum(), h[idx].sum()
    base = G * G / (Hs + lam)
    for j in range(B.shape[1]):
        gb = np.bincount(B[idx, j], weights=g[idx], minlength=nb + 2)
        hb = np.bincount(B[idx, j], weights=h[idx], minlength=nb + 2)
        gl, hl = np.cumsum(gb)[:-1], np.cumsum(hb)[:-1]
        gr, hr = G - gl, Hs - hl
        ok = (hl > 15) & (hr > 15)
        gain = np.where(ok, gl ** 2 / (hl + lam) + gr ** 2 / (hr + lam) - base, 0)
        k = int(gain.argmax())
        if gain[k] > best[0]:
            best = (float(gain[k]), j, k)
    return best


def gbm_fit(B, y, off, rounds, lr=0.03, lam=50.0, nb=16):
    F = off.copy()
    trees = []
    for _ in range(rounds):
        p = 1 / (1 + np.exp(-F))
        g, h = p - y, p * (1 - p)
        idx = np.arange(len(y))
        _, j, k = best_split(B, g, h, idx, lam, nb)
        if j is None:
            break
        L, R = idx[B[idx, j] <= k], idx[B[idx, j] > k]
        node = [j, k]
        for side in (L, R):
            _, j2, k2 = best_split(B, g, h, side, lam, nb)
            if j2 is None:
                v = -g[side].sum() / (h[side].sum() + lam)
                node.append((None, None, v, v))
            else:
                a, b = side[B[side, j2] <= k2], side[B[side, j2] > k2]
                node.append((j2, k2, -g[a].sum() / (h[a].sum() + lam), -g[b].sum() / (h[b].sum() + lam)))
        trees.append(node)
        F += lr * gbm_tree_pred(node, B)
    return trees


def gbm_tree_pred(node, B):
    j, k, Lc, Rc = node
    out = np.empty(len(B))
    left = B[:, j] <= k
    for mask, (j2, k2, a, b) in ((left, Lc), (~left, Rc)):
        if j2 is None:
            out[mask] = a
        else:
            out[mask] = np.where(B[mask, j2] <= k2, a, b)
    return out


def gbm_predict(trees, B, off, lr=0.03):
    F = off.copy()
    for t in trees:
        F += lr * gbm_tree_pred(t, B)
    return 1 / (1 + np.exp(-F))


def ml_tests(d, Z, tname):
    yc, pc = TARGETS[tname]
    ok = d[pc].notna().values
    D, ZZ = d[ok], Z[ok]
    y = D[yc].values
    off = S.logit(D[pc].values)
    per = D.donem.values
    tr = per == "egitim"
    fit_part = tr & (D.date < "2023-08-01").values
    val_part = tr & ~fit_part
    Zv = ZZ.values.astype(float)
    out = []
    # L2 lojistik (standartlastirilmis, eksikler 0)
    mu, sd = np.nanmean(Zv[tr], 0), np.nanstd(Zv[tr], 0) + 1e-9
    Xs = np.nan_to_num(np.clip((Zv - mu) / sd, -3, 3))
    best = None
    for lam in (30, 100, 300, 1000, 3000):
        w = S.fit_offset_logit(Xs[fit_part], y[fit_part], off[fit_part], lam=lam)
        p = 1 / (1 + np.exp(-(off[val_part] + np.c_[np.ones(val_part.sum()), Xs[val_part]] @ w)))
        v = S.ll(p, y[val_part])
        if best is None or v < best[0]:
            best = (v, lam)
    w = S.fit_offset_logit(Xs[tr], y[tr], off[tr], lam=best[1])
    p_lr = 1 / (1 + np.exp(-(off + np.c_[np.ones(len(y)), Xs] @ w)))
    coefs = pd.Series(w[1:], index=ZZ.columns).sort_values(key=np.abs, ascending=False)
    # GBM: tur sayisi dogrulama ile
    B = binize(Zv[fit_part], Zv)
    trees = gbm_fit(B[fit_part], y[fit_part], off[fit_part], 400)
    best_r, best_v = 0, S.ll(1 / (1 + np.exp(-off[val_part])), y[val_part])
    F = off[val_part].copy()
    for i, t in enumerate(trees):
        F += 0.03 * gbm_tree_pred(t, B[val_part])
        v = S.ll(1 / (1 + np.exp(-F)), y[val_part])
        if v < best_v - 1e-6:
            best_r, best_v = i + 1, v
    B2 = binize(Zv[tr], Zv)
    trees2 = gbm_fit(B2[tr], y[tr], off[tr], max(best_r, 1)) if best_r else []
    p_gb = gbm_predict(trees2, B2, off)
    used = pd.Series([ZZ.columns[t[0]] for t in trees2]).value_counts().head(8).to_dict() if trees2 else {}
    for k in S.PERIODS:
        m = per == k
        p0 = 1 / (1 + np.exp(-off[m]))
        out.append(dict(hedef=tname, donem=k, mac=int(m.sum()), piyasa=round(S.ll(p0, y[m]), 5),
                        lojistik=round(S.ll(p_lr[m], y[m]), 5), boosting=round(S.ll(p_gb[m], y[m]), 5)))
    # test sezonlarinda mac bazinda fark anlamli mi? (eslestirilmis z)
    te = per != "egitim"
    p0 = np.clip(1 / (1 + np.exp(-off[te])), 1e-6, 1 - 1e-6)
    l0 = -(y[te] * np.log(p0) + (1 - y[te]) * np.log(1 - p0))
    for name, pp in (("lojistik", p_lr), ("boosting", p_gb)):
        q = np.clip(pp[te], 1e-6, 1 - 1e-6)
        dl = l0 - (-(y[te] * np.log(q) + (1 - y[te]) * np.log(1 - q)))
        out.append(dict(hedef=tname, donem=f"test_{name}_kazanc_z", mac=int(te.sum()), piyasa=np.nan,
                        lojistik=round(float(dl.mean() / (dl.std() / np.sqrt(len(dl)) + 1e-12)), 2) if name == "lojistik" else np.nan,
                        boosting=round(float(dl.mean() / (dl.std() / np.sqrt(len(dl)) + 1e-12)), 2) if name == "boosting" else np.nan))
    return pd.DataFrame(out), dict(lojistik_lambda=best[1], lojistik_en_buyuk=coefs.head(6).round(3).to_dict(),
                                   boosting_tur=best_r, boosting_isaretler=used)


def persistence(d):
    """Piyasa bazi takimlari / hocalari surekli yanlis mi fiyatliyor? Egitim sapmasi vs test sapmasi."""
    rows = []
    long = []
    for side, team, coach, pts, xp in (("h", "home", "home_coach", "hp", "xh"), ("a", "away", "away_coach", "ap", "xa")):
        pass
    x = d.copy()
    x["hp"] = np.where(x.hg > x.ag, 3, np.where(x.hg == x.ag, 1, 0))
    x["ap"] = np.where(x.ag > x.hg, 3, np.where(x.hg == x.ag, 1, 0))
    for team, coach, pts, xp in (("home", "home_coach", "hp", 3 * x.p_h + x.p_d), ("away", "away_coach", "ap", 3 * x.p_a + x.p_d)):
        long.append(pd.DataFrame({"team": x[team], "coach": x[coach], "res": x[pts] - xp, "donem": x.donem}))
    L = pd.concat(long)
    for key, min_tr, min_te in (("team", 60, 30), ("coach", 40, 25)):
        a = L[L.donem == "egitim"].groupby(key).res.agg(["mean", "size"])
        b = L[L.donem != "egitim"].groupby(key).res.agg(["mean", "size"])
        jn = a.join(b, lsuffix="_e", rsuffix="_t", how="inner")
        jn = jn[(jn.size_e >= min_tr) & (jn.size_t >= min_te)]
        top = jn[jn.mean_e > 0.2]
        bot = jn[jn.mean_e < -0.2]
        rows.append(dict(tur=key, adet=len(jn), korelasyon=round(float(jn.mean_e.corr(jn.mean_t)), 3),
                         egitimde_piyasayi_yenen=len(top), onlarin_egitim=round(float(top.mean_e.mean()), 3),
                         onlarin_test=round(float(top.mean_t.mean()), 3),
                         egitimde_piyasanin_altinda=len(bot), onlarin_egitim2=round(float(bot.mean_e.mean()), 3),
                         onlarin_test2=round(float(bot.mean_t.mean()), 3)))
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 70)
    g, d, Z = build()
    con = sim_con()
    print("mac", d.groupby("donem").size().to_dict(), "isaret", Z.shape[1])
    print("eksik oranlari:", Z.isna().mean().round(2)[Z.isna().mean() > 0.05].to_dict())
    F = flags(d, Z)
    print("bayrak", F.shape[1])
    allA, allB = [], []
    for t in ([] if __import__("os").environ.get("SADECE_ML") else TARGETS):
        A = verdict(scan(d, F, t))
        allA.append(A)
        # ikili kombinasyonlar
        pairs = [p for p in itertools.combinations(F.columns, 2) if p[0].split("<=")[0].split(">=")[0] != p[1].split("<=")[0].split(">=")[0]]
        Bt = verdict(scan(d, F, t, cols=pairs, min_n=150), ztr=3.0)
        allB.append(Bt)
        nz = (Bt.egitim_z.abs() > 3).sum()
        print(f"\n### {t}: tekli {len(A)} desen, egitimde |z|>2: {(A.egitim_z.abs() > 2).sum()}, tutan: {A.tutuyor.sum()}"
              f" | ikili {len(Bt)} kombinasyon, egitimde |z|>3: {nz}, testte de tutan: {Bt.tutuyor.sum()}"
              f" (tesadufen beklenen ~{len(Bt) * 0.0027 * 0.25 * 0.05:.1f}+)")
        show = ["desen", "egitim_n", "egitim_fark", "egitim_z", "2024/25_n", "2024/25_fark", "2025/26_fark", "test_fark", "test_z"]
        if A.tutuyor.any():
            print("tekli tutanlar:\n", A[A.tutuyor][show].round(3).to_string(index=False))
        top = A.reindex(A.egitim_z.abs().sort_values(ascending=False).index).head(6)
        print("egitimde en guclu tekliler:\n", top[show].round(3).to_string(index=False))
        if Bt.tutuyor.any():
            print("ikili tutanlar:\n", Bt[Bt.tutuyor][show].round(3).to_string(index=False))
        topb = Bt.reindex(Bt.egitim_z.abs().sort_values(ascending=False).index).head(8)
        print("egitimde en guclu ikililer:\n", topb[show].round(3).to_string(index=False))
    if allA:
        A = pd.concat(allA)
        B = pd.concat(allB)
        A.to_sql("surpriz2_tekli", con, if_exists="replace", index=False)
        B[B.egitim_z.abs() > 2.5].to_sql("surpriz2_ikili", con, if_exists="replace", index=False)
    print("\n== makine ogrenmesi (egitimde ogrenildi; test sezonlarinda log-loss, dusuk iyi) ==")
    M = []
    for t in TARGETS:
        r, info = ml_tests(d, Z, t)
        M.append(r)
        print(r.to_string(index=False))
        print("  ", info)
    pd.concat(M).to_sql("surpriz2_ml", con, if_exists="replace", index=False)
    if __import__("os").environ.get("SADECE_ML"):
        return
    print("\n== takim / hoca kaliciligi (puan - piyasanin beklenen puani, mac basi) ==")
    P = persistence(d)
    print(P.to_string(index=False))
    P.to_sql("surpriz2_kalicilik", con, if_exists="replace", index=False)


if __name__ == "__main__":
    main()
