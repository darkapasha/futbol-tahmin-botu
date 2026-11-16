"""Motor kalibrasyonu: gecmis maclardan tum parametreleri kestirir ve kaydeder.

  .venv/bin/python -m sim.calibrate

1. Haftalik yeniden kurulan puanlarla her gecmis mac icin mac oncesi beklenen gol (pred_hist)
2. Mac ici dinamik carpanlari (dakika, skor durumu, kirmizi kart)  -> params.json
3. Hocalarin skor durumuna tepkisi (gerideyken / ondeyken)          -> sim.sqlite coach_state
4. Hocalarin rakibe gore plani (guclu / benzer / zayif rakip)       -> sim.sqlite coach_plan
5. Kadro degeri carpaninin ussu (beta)                              -> params.json
"""
import numpy as np
import pandas as pd

from . import dynamics, hist, rolling
from .common import load_params, save_params, sim_con, src_con, prof_con

CLASS_T = 0.2  # guc farki esigi (hucum+savunma, log olcek) -> guclu / benzer / zayif


def blend(goals, xg, w):
    x = np.asarray(xg, float)
    y = np.asarray(goals, float)
    return np.where(np.isnan(x), y, (1 - w) * y + w * x)


def team_rows(g, pred, xg_w):
    m = g.merge(pred, on="game_id")
    h = pd.DataFrame({"game_id": m.game_id, "date": m.date, "league": m.league, "coach": m.home_coach,
                      "team": m.home, "is_home": 1, "y_for": blend(m.hg, m.hxg, xg_w), "y_ag": blend(m.ag, m.axg, xg_w),
                      "mu_for": m.mu_h, "mu_ag": m.mu_a, "diff": m.net_a - m.net_h})
    a = pd.DataFrame({"game_id": m.game_id, "date": m.date, "league": m.league, "coach": m.away_coach,
                      "team": m.away, "is_home": 0, "y_for": blend(m.ag, m.axg, xg_w), "y_ag": blend(m.hg, m.hxg, xg_w),
                      "mu_for": m.mu_a, "mu_ag": m.mu_h, "diff": m.net_h - m.net_a})
    t = pd.concat([h, a], ignore_index=True)
    t["opp_class"] = np.where(t["diff"] > CLASS_T, "stronger", np.where(t["diff"] < -CLASS_T, "weaker", "similar"))
    return t


def coach_plan(t, k=20.0):
    """Hocanin rakip sinifina gore, puan modelinin beklentisinden sistematik sapmasi.
    Hocanin genel sapmasi cikarilir (o zaten takim puaninda); kalan = rakibe ozel plan."""
    t = t.dropna(subset=["coach"]).copy()
    t["r_for"] = np.log((t.y_for + 0.3) / (t.mu_for + 0.3))
    t["r_ag"] = np.log((t.y_ag + 0.3) / (t.mu_ag + 0.3))
    base = t.groupby("coach")[["r_for", "r_ag"]].mean()
    s = t.groupby(["coach", "opp_class"]).agg(n=("r_for", "size"), r_for=("r_for", "mean"), r_ag=("r_ag", "mean")).reset_index()
    s = s.join(base, on="coach", rsuffix="_all")
    shrink = s.n / (s.n + k)
    s["att"] = np.exp(shrink * (s.r_for - s.r_for_all))
    s["dfn"] = np.exp(shrink * (s.r_ag - s.r_ag_all))  # >1: daha cok yiyor (acik oyun)
    # plana gecme olasiligi: o sinifta ana dizilisten sapma orani (coach_split)
    sp = pd.read_sql("SELECT coach, split, main_formation_used FROM coach_split WHERE split LIKE 'opp_class=%'", prof_con())
    sp["opp_class"] = sp.split.str.split("=").str[1]
    s = s.merge(sp[["coach", "opp_class", "main_formation_used"]], on=["coach", "opp_class"], how="left")
    s["p_plan"] = (1 - s.main_formation_used.fillna(0.6)).clip(0.25, 1.0)
    return s[["coach", "opp_class", "n", "att", "dfn", "p_plan"]]


def lineup_beta(g, pred, xg_w):
    """Gecmis maclarda ilk 11 degeri / son 10 mac ortalamasi oraninin gol beklentisine etkisi."""
    con = src_con()
    lu = pd.read_sql("SELECT game_id, club_id, player_id, date FROM tm_lineups WHERE type='starting_lineup' "
                     "AND date >= '2021-06-01'", con)
    val = pd.read_sql("SELECT player_id, date, market_value_in_eur v FROM tm_valuations", con)
    lu["date"], val["date"] = pd.to_datetime(lu.date), pd.to_datetime(val.date)
    lu = pd.merge_asof(lu.sort_values("date"), val.sort_values("date"), on="date", by="player_id",
                       allow_exact_matches=False)
    xi = lu.groupby(["game_id", "club_id", "date"]).v.sum(min_count=8).reset_index().sort_values("date")
    xi["usual"] = xi.groupby("club_id").v.transform(lambda s: s.shift(1).rolling(10, min_periods=4).mean())
    xi["ratio"] = (xi.v / xi.usual).clip(0.5, 1.6)
    xi["team"] = "tm:" + xi.club_id.astype(str)
    xi["game_id"] = xi.game_id.astype(str)
    rmap = xi.set_index(["game_id", "team"]).ratio
    m = g.merge(pred, on="game_id")
    m = m[m.source == "tm"]
    rh = pd.Series(list(zip(m.game_id, m.home))).map(rmap.to_dict()).values.astype(float)
    ra = pd.Series(list(zip(m.game_id, m.away))).map(rmap.to_dict()).values.astype(float)
    ok = ~np.isnan(rh) & ~np.isnan(ra)
    m, rh, ra = m[ok], rh[ok], ra[ok]
    yh, ya = m.hg.values, m.ag.values
    best = None
    for b in np.arange(0, 0.81, 0.05):
        lh = m.mu_h.values * (rh / ra) ** b
        la = m.mu_a.values * (ra / rh) ** b
        # gol ortalamasini koru (puan modeli zaten ortalamayi tutturuyor)
        ll = (yh * np.log(lh) - lh + ya * np.log(la) - la).sum()
        print(f"  beta={b:.2f} loglik={ll:.1f}")
        if best is None or ll > best[1]:
            best = (b, ll)
    print("kadro orani dagilimi:", np.percentile(rh, [5, 25, 50, 75, 95]).round(2), "n =", ok.sum())
    return float(best[0])


def main():
    params = load_params()
    rp = params.get("ratings")
    xg_w = (rp or {}).get("xg_w", 0.6)
    g = hist.games()
    ev = hist.events()
    print("1) gecmis tahminler")
    pred = rolling.rolling(g, rp, start="2021-09-01")
    con = sim_con()
    pred.to_sql("pred_hist", con, if_exists="replace", index=False)
    rolling.evaluate(g, pred[pred.game_id.isin(g[g.date >= "2022-08-01"].game_id)], "puan modeli (Poisson)")

    print("2) mac ici dinamikler")
    d = dynamics.fit(g, ev, pred)
    dynamics.describe(d)
    params["dynamics"] = d

    print("3) hoca skor durumu tepkisi")
    cs = dynamics.coach_state(g, ev, pred, d, k=params.get("coach_state_k", 6.0))
    cs.to_sql("coach_state", con, if_exists="replace", index=False)
    print(cs.sort_values("att_trail").tail(5).round(2).to_string())

    print("4) hoca rakibe gore plan")
    t = team_rows(g, pred, xg_w)
    print("rakip sinifi paylari:", t.opp_class.value_counts(normalize=True).round(2).to_dict())
    cp = coach_plan(t)
    cp.to_sql("coach_plan", con, if_exists="replace", index=False)
    print(cp[cp.n >= 15].sort_values("att").iloc[[0, 1, 2, -3, -2, -1]].round(2).to_string())

    print("5) kadro degeri carpani")
    params["lineup_beta"] = lineup_beta(g, pred, xg_w)
    print("beta =", params["lineup_beta"])
    params["class_t"] = CLASS_T
    save_params(params)
    con.commit()


if __name__ == "__main__":
    main()
